"""The ITC retrieval gate, shared by scripts/check_itc_gate.py and the runner.

One all-to-all bidirectional InfoNCE over a fixed, ordered subset of valid
(image, report) pairs, in eval mode, with no queue and no gradient. Reported:

* ``delta_nats = ln(N) - L_itc`` -- nats of separation over chance; scales with
  1/temperature, so only comparable at the same temperature;
* mean rank of the true pair -- scale-free;
* R@1 / R@5 in both directions against a one-sided binomial test of chance.

``meets_threshold`` requires BOTH ``delta_nats >= min_delta`` and R@5 above
chance in both directions (decided 2026-09-24): delta alone moves with the
temperature and R@5 alone ignores calibration.

Only the subset is fixed here; the caller supplies the loader, so the gate can
run on a feature-cache loader inside training or on decoded images offline.
"""

from __future__ import annotations

import math

DEFAULT_PAIRS = 256
MIN_DELTA_NATS = 0.10
RECALL_ALPHA = 0.01


def recall_at_k(sim, k):
    """Fraction of rows whose true pair (the diagonal) ranks in the top k."""
    k = min(int(k), sim.shape[1])
    true = sim.diagonal()[:, None]
    rank = (sim > true).sum(dim=1)
    return float((rank < k).float().mean())


def binomial_upper_quantile(n, p, alpha):
    """Smallest c with P(X >= c) <= alpha for X ~ Binomial(n, p); exact."""
    tail = 1.0
    for c in range(n + 1):
        if tail <= alpha + 1e-15:
            return c
        tail -= math.comb(n, c) * p**c * (1 - p) ** (n - c)
    return n + 1


def recall_gate(sim_i2t, sim_t2i, n, alpha=RECALL_ALPHA):
    """R@1 / R@5 both ways, and whether R@5 beats chance both ways.

    Under chance a true pair lands in the top k with probability k/n, so the
    hit count over n queries is Binomial(n, k/n); "above chance" means reaching
    its one-sided 1 - alpha quantile. R@1 at n=256 has an expected count of 1,
    too small to decide on its own, so it is reported but not required.
    """
    out = {}
    for k in (1, 5):
        threshold = binomial_upper_quantile(n, min(k / n, 1.0), alpha)
        for name, sim in (("i2t", sim_i2t), ("t2i", sim_t2i)):
            r = recall_at_k(sim, k)
            out[f"R@{k}_{name}"] = round(r, 4)
            out[f"R@{k}_{name}_above_chance"] = bool(round(r * n) >= threshold)
        out[f"R@{k}_chance"] = round(k / n, 4)
        out[f"R@{k}_significant_hits"] = int(threshold)
    out["recall_above_chance"] = bool(
        out["R@5_i2t_above_chance"] and out["R@5_t2i_above_chance"]
    )
    return out


def itc_features(model, samples, device):
    """(image [B, Q, D], text [B, D]) features ITC compares, as forward() does."""
    import torch
    import torch.nn.functional as F

    moved = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in samples.items()
    }
    shared_visual = model.encode_samples(moved)
    image_embeds = shared_visual.tokens
    image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=device)
    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_output = model.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        use_cache=True,
        return_dict=True,
    )
    image_features = F.normalize(model.vision_proj(query_output.last_hidden_state), dim=-1)
    text_tokens = model.tokenizer(
        samples["text_output"],
        padding="max_length",
        truncation=True,
        max_length=model.max_txt_len,
        return_tensors="pt",
    ).to(device)
    text_output = model.Qformer.bert(
        text_tokens.input_ids, attention_mask=text_tokens.attention_mask, return_dict=True
    )
    text_features = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0]), dim=-1)
    return image_features, text_features


def collect_pairs(model, loader, pairs, device):
    """First ``pairs`` studies with usable FINDINGS, in loader order."""
    import torch

    image_chunks, text_chunks = [], []
    scanned = kept = 0
    with torch.no_grad():
        for samples in loader:
            img, txt = itc_features(model, samples, device)
            scanned += img.shape[0]
            keep = samples.get("generation_mask")
            if keep is None:
                raise KeyError(
                    "the dataset emitted no generation_mask; refusing to score "
                    "pairs whose validity is unknown"
                )
            keep = keep.to(torch.bool).cpu()
            img, txt = img.float().cpu()[keep], txt.float().cpu()[keep]
            if kept + img.shape[0] > pairs:
                room = pairs - kept
                img, txt = img[:room], txt[:room]
            image_chunks.append(img)
            text_chunks.append(txt)
            kept += img.shape[0]
            if kept >= pairs:
                break
    if not image_chunks:
        raise ValueError("the loader yielded no batch")
    return torch.cat(image_chunks), torch.cat(text_chunks), scanned


def model_temperature(model):
    if bool(getattr(model, "itc_temp_learnable", True)):
        return float(model.temp.detach().clamp(min=1e-3, max=0.5).cpu()), True
    return float(model.itc_temp_fixed.detach().cpu()), False


def score_itc(image_features, text_features, temperature, min_delta=MIN_DELTA_NATS):
    """The gate numbers for already-collected features."""
    import torch
    import torch.nn.functional as F

    n = image_features.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 pairs to contrast, got {n}")
    sim_i2t = torch.einsum("bqd,nd->bnq", image_features, text_features).amax(-1)
    sim_t2i = torch.einsum("bd,nqd->bnq", text_features, image_features).amax(-1)
    targets = torch.arange(n)
    loss_itc = 0.5 * (
        F.cross_entropy(sim_i2t / temperature, targets)
        + F.cross_entropy(sim_t2i / temperature, targets)
    )
    chance = math.log(n)
    delta = chance - float(loss_itc)
    rank_i2t = (sim_i2t > sim_i2t.diagonal()[:, None]).sum(dim=1).float().mean()
    rank_t2i = (sim_t2i > sim_t2i.diagonal()[:, None]).sum(dim=1).float().mean()
    recall = recall_gate(sim_i2t, sim_t2i, n)
    return {
        "pairs": n,
        "temperature": round(temperature, 6),
        "loss_itc": round(float(loss_itc), 4),
        "chance_ln_n": round(chance, 4),
        "delta_nats": round(delta, 4),
        "min_delta": min_delta,
        "mean_rank_of_true_pair_i2t": round(float(rank_i2t), 2),
        "mean_rank_of_true_pair_t2i": round(float(rank_t2i), 2),
        "chance_rank": round((n - 1) / 2, 2),
        **recall,
        "delta_meets_threshold": bool(delta >= min_delta),
        "meets_threshold": bool(delta >= min_delta and recall["recall_above_chance"]),
    }


def run_gate(model, loader, device, pairs=DEFAULT_PAIRS, min_delta=MIN_DELTA_NATS):
    """Collect ``pairs`` valid pairs from ``loader`` and score them."""
    image_features, text_features, scanned = collect_pairs(model, loader, pairs, device)
    n = image_features.shape[0]
    if n < pairs:
        raise ValueError(f"only {n} of {scanned} scanned studies have usable FINDINGS; needed {pairs}")
    temperature, learnable = model_temperature(model)
    report = score_itc(image_features, text_features, temperature, min_delta)
    report.update(
        {
            "temp_learnable": learnable,
            "studies_scanned": scanned,
            "valid_fraction": round(n / scanned, 4) if scanned else None,
        }
    )
    return report
