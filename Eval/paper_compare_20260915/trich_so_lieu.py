"""Collect the verified numbers both documents cite into one aggregate JSON (no identifiers)."""
import json, sys
from pathlib import Path
SRC = Path(sys.argv[1]); OUT = Path(sys.argv[2])
J = lambda *p: json.load(open(SRC.joinpath(*p)))
def pair(r):
    return {"delta": r["delta"], "lo": r["ci_lower"], "hi": r["ci_upper"],
            "sig": bool(r["ci_excludes_zero"] and not r["at_zero_boundary"])}
d = {}
agg = J("stage1", "ft_main", "metrics.json")["classification"]["aggregates"]
d["s1"] = {k: agg[k] for k in ("macro_auroc", "micro_auroc", "macro_auprc", "positive_macro_f1",
                              "positive_macro_precision", "positive_macro_recall", "macro_specificity")}
d["s1"]["n"] = J("stage1", "ft_main", "metrics.json")["metadata"]["num_samples"]
bl = {b["baseline"]: b for b in J("stage1", "ft_main", "metrics.json")["classification"]["baselines"]}
mq = J("stage1", "ft_masked_q", "metrics.json")["classification"]
mbl = {b["baseline"]: b for b in mq["baselines"]}
d["framing"] = {"presence_model_f1": agg["positive_macro_f1"], "presence_all_positive_f1": bl["all_positive"]["positive_macro_f1"],
                "presence_majority_f1": bl["majority_class"]["positive_macro_f1"],
                "masked_model_f1": mq["aggregates"]["positive_macro_f1"], "masked_all_positive_f1": mbl["all_positive"]["positive_macro_f1"],
                "masked_majority_f1": mbl["majority_class"]["positive_macro_f1"]}
d["s1pair"] = {}
for name in ("unfreeze", "gate", "minpos", "plateau", "depth", "hier_shallow", "hier_deep"):
    p = J("stage1", f"paired_{name}.json")
    d["s1pair"][name] = {r["metric"]: pair(r) for r in p["results"]}
    d["s1pair"][name]["n"] = p["num_studies"]
d["gen"] = {}
for f in ("test_main", "stop_probe_25", "cue_rules_100", "pilot_300"):
    s = J("standard", f"{f}.json")
    d["gen"][f] = {"n": s["n_studies"], "res": s["results"],
                   "pair": {p["comparison"].replace(" - ", "_minus_"): {m: pair(v) for m, v in p.items() if isinstance(v, dict)}
                            for p in s["pairs"]}}
d["gen"]["protocol"] = J("standard", "test_main.json")["protocol"]
ds = J("data_stats.json")
d["data"] = {"blank_fraction_raw": ds["raw"]["chexpert_blank_fraction"], "train_studies": ds["splits"]["train"]["studies"],
             "val_studies": ds["splits"]["val"]["studies"], "test_studies": ds["splits"]["test"]["studies"],
             "train_aux_view": ds["splits"]["train"]["studies_with_aux_view"],
             "train_images": ds["splits"]["train"]["images"]}
# numbers typed from the IEEE Access PDF (docs/paper/...), Tables 1-3, 5-9 and Sections III-IV, read 2026-09-15
d["paper"] = {"bleu_1": 0.390, "bleu_2": 0.255, "bleu_3": 0.175, "bleu_4": 0.102, "meteor": 0.173, "rouge_l": 0.280, "cider": 0.291,
              "bertscore": 0.426, "radgraph_f1": 0.224, "radcliq": 1.211, "ce_precision": 0.411, "ce_recall": 0.465, "ce_f1": 0.428,
              "cls_precision": 0.87, "cls_recall": 0.78, "cls_f1": 0.73, "chexpert_f1": 0.699,
              "loss_cls": 0.551, "loss_cls_ol": 0.649, "loss_cls_ol_sl": 0.667, "loss_all": 0.701,
              "enc_rn50": 0.602, "enc_vit": 0.473, "enc_swin": 0.467, "enc_all": 0.701,
              "steps_repr": 20000, "steps_gen": 5000, "lora_r": 8, "lora_alpha": 16, "n_test": 3269}
json.dump(d, open(OUT, "w"), indent=1)
print("written", OUT)
