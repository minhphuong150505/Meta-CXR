# PLAN 2026-10-25 — Multi-view step 0: does arm A actually use its auxiliary view?

Planner/executor: Claude (planning session over SSH). User go-ahead 2026-10-25.
Follows `PLAN-2026-10-24-multi-view-ablation.md`, whose F1 gain was the same on
single-view and two-view studies (so not attributable to the second view).
**Inference only, no training.** One evaluate-only GPU pass (~10 min) + CPU.

## Question

Is `ViewFusion` in `run_20261005_paper` (arm A) doing anything? Its residual
branches (`w_o`, last FFN Linear) start at zero and get 5 epochs of 1b + 1 of 1c.
If they stayed near zero, the multi-view design is untested rather than
refuted, and step 1 (a design change) is justified.

## Measurements

1. **Counterfactual:** score A's own `phase1c/checkpoint_best.pth` on test with
   `model.data.max_aux_views=0` (multi-view still on, so the same StreamAdapters
   run on the anchor; fusion is gated to exactly zero for every study). The
   only difference from `A_rescore.npz` (plan of 2026-10-24, G1) is the
   auxiliary view. `max_aux_views` lives in `model.data`, which no phase block
   sets, so `--options` reaches it.
2. **Weights:** RMS of each fusion block's `w_o` and `ffn.3` (zero-init) against
   `w_q` / `w_v` (random-init) in the same checkpoint.
3. **Per finding** on views_2 (views_1 as control): weighted F1 and one-vs-rest
   AUROC (Pos, Unc), A − B (multi_view false run) and A − A_noaux, paired
   bootstrap 1,000 × seed 16. Clinically, the lateral view should matter most
   for Pleural Effusion (posterior costophrenic angle), retrocardiac
   Consolidation / Atelectasis, and Cardiomegaly.

## Commands (host)

```bash
cd ~/Meta-CXR && git pull --ff-only
PY=~/.venvs/meta-cxr-stage1-311/bin/python
A=~/run_20261005_paper; AD=$A/phase1c/mimic_cxr_full_blip2; OUT=~/nomv_probe_20261024
# guards: ntfs3, GPU idle, no training process, $OUT/A_noaux.npz absent
touch $OUT/.s0_marker
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled $PY -m pretraining.train \
  --cfg-path pretraining/configs/mimic_cxr_full.yaml \
  --options run.phase=phase1c run.phase_root=$A run.evaluate=True run.save_predictions=True \
    run.output_dir=$OUT/A_noaux run.resume_ckpt_path=$AD/checkpoint_best.pth \
    run.distributed=false run.world_size=1 model.data.max_aux_views=0 > $OUT/A_noaux.log 2>&1
grep -c "Loading evaluation weights from" $OUT/A_noaux.log        # must be 1
grep "\[test\].*complementary" $OUT/A_noaux.log                    # 0 with a complementary view (multi_view=True, max_aux_views=0)
f=$(find $A $OUT -name test_predictions_epoch_provided.npz -newer $OUT/.s0_marker | head -1); mv "$f" $OUT/A_noaux.npz
$PY scripts/compare_stage1_predictions.py --subgroups-from baseline \
  --baseline $OUT/A_rescore.npz --candidate $OUT/A_noaux.npz --output $OUT/noaux_vs_A.json > $OUT/noaux_vs_A.log
$PY ~/aux_probe.py $AD/checkpoint_best.pth $OUT/A_rescore.npz $OUT/A_noaux.npz \
  ~/run_20261024_nomv/phase1c/mimic_cxr_full_blip2/result/test_predictions_epoch_best.npz $OUT/aux_probe.json
```

`~/aux_probe.py` is an ad-hoc analysis script (numpy, sklearn, torch.load; reads
the three `.npz` and the checkpoint, prints aggregates only). Not in the repo.

## Reading rule — fixed before any result

Sanity (else STOP): on `views_1` every A − A_noaux delta is 0 and
`changed_argmax` is 0 (identical input).

On `views_2` (1,247 studies), from `noaux_vs_A.log` (it prints A_noaux − A;
flip the sign):

- **Fusion inert:** `changed_argmax` < 0.01, A − A_noaux on `auroc_mean` and
  `mean_weighted_f1_5` both with CIs crossing zero and |Δ| < 0.005, and the
  zero-init weights' RMS well below the random-init ones. → the multi-view
  design was never really trained; step 1 (design change, pilot on val) is
  justified.
- **Fusion used and helps:** A − A_noaux on `auroc_mean` or
  `mean_weighted_f1_5` > 0 with a CI excluding zero. → the second view
  contributes inside A; the novelty claim rests on this counterfactual plus the
  per-finding table (strongest if it is the lateral-sensitive findings).
- **Fusion used and hurts:** the same, significantly < 0.
- Anything else: "used, effect not measurable".

Per-finding results are descriptive (14 findings × 2 metrics, no multiplicity
correction): report them, do not select on them.

## Data handling

`.npz`, JSON and logs stay on the host. The report below carries aggregates only.

## Execution report

(appended after the run)
