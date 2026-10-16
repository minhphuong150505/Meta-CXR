"""Gom số liệu tổng hợp (không dữ liệu bệnh nhân) thành so_lieu.json."""
import csv, json
R = "raw/"
def J(p): return json.load(open(R + p))
def rows(p): return list(csv.DictReader(open(R + p)))
def jl(p): return [json.loads(l) for l in open(R + p) if l.strip()]

new = J("eval_20261005_paper/test/metrics.json")["classification"]
old = J("eval_20260930_3class/test/metrics.json")["classification"]
eq = J("eval_20261005_paper/test_eq22/metrics.json")["classification"]
val = J("eval_20261005_paper/val/metrics.json")["classification"]
paired = J("eval_20261005_paper/paired_vs_3class_test.json")
cut = J("eval_20261005_paper/cutpoints/cutpoints_report.json")
D = {
 "paper": {"precision": 0.87, "recall": 0.78, "f1": 0.73, "f1_5": 0.701},
 "n_test": paired["n"], "n_val": None,
 "new": {k: v for k, v in new["aggregates"].items()},
 "ci": {k: {"lo": v["lower"], "hi": v["upper"]} for k, v in new["intervals"].items()},
 "val": val["aggregates"],
 "eq22": eq["aggregates"],
 "paired": paired["metrics"], "paired_resamples": paired["resamples"],
 "cut": cut,
 "p1b": jl("run_20261005_paper/phase1b/mimic_cxr_full_blip2/phase_metrics.jsonl"),
 "p1c": jl("run_20261005_paper/phase1c/mimic_cxr_full_blip2/phase_metrics.jsonl"),
}
pn = {r["pathology"]: r for r in rows("eval_20261005_paper/test/per_pathology_metrics.csv")}
po = {r["pathology"]: r for r in rows("eval_20260930_3class/test/per_pathology_metrics.csv")}
D["per"] = []
for name, r in pn.items():
    o = po[name]
    g = lambda d, k: (float(d[k]) if d.get(k) not in (None, "", "nan") else None)
    D["per"].append({"name": name, "nN": int(r["n_negative"]), "nP": int(r["n_positive"]), "nU": int(r["n_uncertain"]),
        "wP": g(r, "weighted_precision"), "wR": g(r, "weighted_recall"), "wF1": g(r, "weighted_f1"),
        "aN": g(r, "auroc_negative"), "aP": g(r, "auroc_positive"), "aU": g(r, "auroc_uncertain"),
        "rN": g(r, "recall_negative"), "rP": g(r, "recall_positive"), "rU": g(r, "recall_uncertain"),
        "pP": g(r, "precision_positive"),
        "old_wF1": g(o, "weighted_f1"), "old_aP": g(o, "auroc_positive"), "old_aU": g(o, "auroc_uncertain")})
json.dump(D, open("so_lieu.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: D[k] for k in ("eq22",)}, indent=0)[:900])
print(cut.keys())
print(json.dumps(cut, indent=0)[:1500])

# số liệu dẫn xuất
v = json.load(open(R + "eval_20261005_paper/val/metrics.json"))
conf = v["classification"]["confusion"]
first = next(iter(conf.values()))
D["n_val"] = int(v["metadata"]["num_samples"])
D["best_1b_epoch"] = max(D["p1b"], key=lambda r: r["macro_recall"])["epoch"]
D["win_wF1"] = sum(1 for r in D["per"] if r["wF1"] > r["old_wF1"])
D["win_aP"] = sum(1 for r in D["per"] if r["aP"] > r["old_aP"])
D["win_aU"] = sum(1 for r in D["per"] if r["aU"] is not None and r["old_aU"] is not None and r["aU"] > r["old_aU"])
D["n_aU"] = sum(1 for r in D["per"] if r["aU"] is not None and r["old_aU"] is not None)
D["n_wF1_above_paper"] = sum(1 for r in D["per"] if r["wF1"] > 0.73)
c = D["cut"]["rules"]
D["cutA"] = c["argmax (paper)"]; D["cut1"] = c["one cutpoint"]; D["cut2"] = c["two cutpoints"]
D.pop("cut")
json.dump(D, open("so_lieu.json", "w"), indent=1, ensure_ascii=False)
print(D["n_val"], D["best_1b_epoch"], D["win_wF1"], D["win_aP"], D["win_aU"], D["n_aU"], D["n_wF1_above_paper"])
print({k: (D["cut2"].get(k)) for k in ("uncertain_predicted","uncertain_recall","uncertain_precision")}, list(D["cut1"].keys()))
for r in D["per"]: print(r["name"], round(r["wF1"],4), round(r["old_wF1"],4), r["aU"], r["old_aU"])
D["base"] = {b["baseline"]: b for b in new["baselines"]}
json.dump(D, open("so_lieu.json", "w"), indent=1, ensure_ascii=False)

# 2026-10-08: vì sao F1 weighted cao hơn bài báo trong khi P, R weighted không cao hơn
CLS = ("negative", "positive", "uncertain")
fz = {"zd1": [], "zd0": []}
never = []
below = []
for r in rows("eval_20261005_paper/test/per_pathology_metrics.csv"):
    nc = {c: int(r["n_" + c]) for c in CLS}
    N = sum(nc.values())
    p1 = p0 = 0.0
    for c in CLS:
        if nc[c] == 0:
            continue
        p, rc = float(r["precision_" + c]), float(r["recall_" + c])
        p1 += nc[c] / N * p
        if p == 1.0 and rc == 0.0:          # tp = 0 mà P = 1: lớp không bao giờ được dự đoán
            never.append({"name": r["pathology"], "cls": c, "n": nc[c]})
        else:
            p0 += nc[c] / N * p
    fz["zd1"].append(p1); fz["zd0"].append(p0)
    P, Rw, F = (float(r[k]) for k in ("weighted_precision", "weighted_recall", "weighted_f1"))
    if F < min(P, Rw):
        below.append(r["pathology"])
    if r["pathology"] == "Cardiomegaly":
        D["cardio"] = {c: {"n": nc[c], "p": float(r["precision_" + c]), "r": float(r["recall_" + c]),
                           "f1": float(r["f1_" + c])} for c in CLS}
mean = lambda x: sum(x) / len(x)
D["zd"] = {"p_zd1": mean(fz["zd1"]), "p_zd0": mean(fz["zd0"]),
           "never": never, "n_never": len(never),
           "n_never_unc": sum(1 for x in never if x["cls"] == "uncertain"),
           "n_never_unc_findings": len({x["name"] for x in never if x["cls"] == "uncertain"}),
           "below": below, "n_below": len(below)}
p = D["paper"]
D["paper"]["f1_from_pr"] = 2 * p["precision"] * p["recall"] / (p["precision"] + p["recall"])

# Ví dụ minh họa tự dựng (không phải dữ liệu): 1 bệnh, 100 ca lớp A và 100 ca lớp B.
def toy(cm):  # cm[true][pred], hai lớp
    out = {}
    for k in (0, 1):
        pred = cm[0][k] + cm[1][k]; true = sum(cm[k]); tp = cm[k][k]
        P = tp / pred; R = tp / true
        out[k] = {"p": P, "r": R, "f1": 2 * P * R / (P + R), "w": true / 200}
    agg = {m: sum(out[k]["w"] * out[k][m] for k in (0, 1)) for m in ("p", "r", "f1")}
    return {"A": out[0], "B": out[1], **agg}
D["toy"] = {"X": toy([[50, 50], [0, 100]]), "Y": toy([[75, 25], [25, 75]])}
json.dump(D, open("so_lieu.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: D[k] for k in ("zd", "toy", "cardio")}, indent=1, ensure_ascii=False)[:2500], D["paper"])
