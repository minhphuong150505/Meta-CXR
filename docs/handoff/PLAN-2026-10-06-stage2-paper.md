# PLAN 2026-10-06 — Stage 2 as the paper (D-027), chained after run_20261005_paper

## What runs

MedGemma 1.5 4B, QLoRA NF4, **no native image**: the 32 Q-Former soft tokens
are the whole visual channel, MHCAC's Positive / Negative / Uncertain lists go in
as text, the instruction is the reference `inference.py` one byte for byte.

| | value |
|---|---|
| pipeline mode | `meta_cxr_qformer_with_mhcac_prompt` |
| prompt | `--prompt-style paper` (code `daaa8e8`) |
| cues | `--cue-rule paper_thresholds`, Eq. 22 thresholds fitted on val by `eval_after_paper.sh` |
| LoRA | r=8, alpha=16 (paper; the default under `paper`) |
| Stage 1 | `~/run_20261005_paper/phase1c/mimic_cxr_full_blip2/checkpoint_best.pth` (D-026) |
| section | `findings_only` |
| batch | 2 x accum 8, 1 epoch, recovery adapter every 32 updates |

## Launch (done 2026-10-05 22:54, pid 27300)

`~/stage2_paper_after_stage1.sh` (copy of the script in this session's
scratchpad; the host file is the deployed one):

1. waits until `~/eval_20261005_paper/STATUS` is `DONE` (aborts on `FAILED`);
2. requires the checkpoint and `thresholds_eq22_val.json`, an empty GPU and
   `ntfs3` on `/mnt/drive1tb`;
3. `git pull --ff-only` and requires `daaa8e8` in `HEAD`;
4. **smoke** into `~/stage2_paper_20261006/smoke` (200 train / 20 val / 10
   test) and checks the adapter: `status complete`, `prompt_style paper`,
   `r 8`, `lora_alpha 16`, `prompt_version paper_build_instruction`;
5. only then the **full run** into `~/stage2_paper_20261006/full` (whole train
   split, val generation 300, whole test split), same check at the end.

Every step refuses an existing output directory. Status lines go to
`~/stage2_paper_20261006.status`; logs to `~/stage2_paper_20261006/{smoke,full}.log`.

## Expected

Stage 1 phase 1c was at iteration 6,250 / 27,844 at 22:51 (ETA ~3h36m), then
Stage-1 scoring. Smoke ~15–30 min including the Stage-1 record pass. The full
run has no measured throughput for this mode: arm C (native image + soft
tokens) ran 3.37 s/it, 82h28m per epoch; without the 256 image tokens the
sequence is shorter, so expect less, not measured. Plus ~73 min for the
Stage-1 record pass and several hours of test generation.

## Abort conditions

Any non-zero exit, a failed adapter check, OOM. **Do not relaunch into
`full/` after a mid-epoch death**: Stage-2 resume is epoch-level and would write
the partial adapter out as complete (CLAUDE.md, "Do not resume a Stage-2 run
that died mid-epoch").

## Not verified

`--prompt-style paper`, `--cue-rule paper_thresholds` through the training
runner, and a 3-phase (D-020/D-026) Stage-1 checkpoint feeding
`build_stage1_records` have never run on GPU. The smoke is what checks them.

## Execution report

_(to be appended)_
