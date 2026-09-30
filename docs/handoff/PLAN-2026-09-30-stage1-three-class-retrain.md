# PLAN 2026-09-30 — Stage-1 retrain under D-023 (three classes only)

Planner: Claude (dev box). Requested by the user 2026-09-30: every earlier
Stage-1 number was produced or scored under a binary framing, so Stage 1 is
retrained on the D-023 code and scored with the paper's protocol.

## Decision: reuse phase 1a, retrain 1b + 1c

Phase 1a trains only the META-Former (Q-Former + shared projection) with
ITC/ITM/LM; `lambda_cls` is 0 and MHCAC is not run, so no CheXpert label
touches it. D-023 changed labels, losses and metrics only. The 1a checkpoint of
`run_20260925b_3phase` (D-022 code, gate passed at epoch 2: `delta_nats` +2.93,
R@5 0.664 / 0.641) is therefore copied into the new root and only 1b + 1c run.
Saves ~15.5 h plus the feature-cache build. `Blip2Qformer.load_state_dict`
passes through `drop_retired_state`, so any retired head in the old file is
dropped on load.

Checkpoint selection in 1b and 1c is `selection_metric: loss` on val, so no
binary metric picks a checkpoint.

## Commands (host)

```bash
cd ~/Meta-CXR && git fetch && git checkout feat/stage2-finding-tokens && git pull
findmnt -no FSTYPE /mnt/drive1tb                 # must be ntfs3
CUDA_VISIBLE_DEVICES="" ~/.venvs/meta-cxr-stage1-311/bin/python -m pytest tests/ -q
# expected: only the 4 test_native_independence failures (missing private config)

ROOT=$HOME/run_20260930_3class
mkdir -p $ROOT && cp ~/run_20260925b_3phase/checkpoint_phase1a.pth $ROOT/
ROOT=$ROOT PHASES="phase1b phase1c" setsid nohup bash scripts/run_stage1_phases.sh \
    > $ROOT.log 2>&1 < /dev/null &
```

Expected: ~7h53m (1b) + ~4h45m (1c), as measured on the D-022 run.

## Evaluation

```bash
python scripts/evaluate_stage1.py --predictions <phase1c test .npz> --output-dir ~/eval_20260930_3class
```

Paper references: weighted P/R/F1 over 14 findings 0.87 / 0.78 / 0.73;
`mean_weighted_f1_5` 0.701; one-vs-rest AUROC per class (Fig. 5).

## Abort conditions

- tests fail beyond the known 4;
- `/mnt/drive1tb` is not `ntfs3`;
- another GPU process is running;
- NaN/inf in any loss, OOM, or `checkpoint_phase1b.pth` missing after 1b.

## Execution report

_(executor appends here)_
