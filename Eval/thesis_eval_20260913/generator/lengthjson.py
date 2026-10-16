"""Aggregate length / template diagnostics for the thesis. Numbers only, no text."""
import json, re
from collections import Counter
import numpy as np
def words(t): return re.findall(r"\w+", str(t).lower())
T = "/home/phuong/thesis_eval_20260913"
def load(path): return {json.loads(l)["sample_key"]: json.loads(l) for l in open(path)}
out = {}
test = {n: load(f"{T}/gen_{n}/generated_test.jsonl") for n in ("zeroshot", "armA", "armC")}
keys = sorted(test["armC"])
ref = np.array([len(words(test["armC"][k]["reference"])) for k in keys])
for n, d in test.items():
    L = np.array([len(words(d[k]["generated"])) for k in keys])
    outs = Counter((d[k]["generated"] or "").strip() for k in keys)
    top_text, top_count = outs.most_common(1)[0]
    short = L <= 10
    out[n] = {"split": "test", "n": len(keys), "median_words": float(np.median(L)),
              "share_le10": float(short.mean()), "share_le5": float((L <= 5).mean()),
              "distinct_outputs": len(outs), "top1_share": top_count / len(keys),
              "top1_words": len(words(top_text)), "top5_share": sum(c for _, c in outs.most_common(5)) / len(keys),
              "ref_median_when_short": float(np.median(ref[short])) if short.any() else None,
              "ref_median_when_long": float(np.median(ref[~short])) if (~short).any() else None}
lc = np.array([len(words(test["armC"][k]["generated"])) for k in keys])
la = np.array([len(words(test["armA"][k]["generated"])) for k in keys])
out["overlap"] = {"short_both": int(((lc <= 10) & (la <= 10)).sum()), "short_armA": int((la <= 10).sum()),
                  "short_armC": int((lc <= 10).sum()), "ref_share_le15": float((ref <= 15).mean()),
                  "ref_median": float(np.median(ref))}
v = load("/home/phuong/cue4_fixedstop_20260909/conditional/generated_val.jsonl")
Lv = np.array([len(words(r["generated"])) for r in v.values()])
out["armC_val_conditional"] = {"split": "val", "n": len(v), "median_words": float(np.median(Lv)),
                               "share_le10": float((Lv <= 10).mean())}
json.dump(out, open(f"{T}/length_diagnostics.json", "w"), indent=2)
print(json.dumps(out, indent=1))
