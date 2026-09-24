# Blank CheXpert cells: IGNORE (-100) -> NEGATIVE (0), as in the original META-CXR

## Plan (the user's prompt, verbatim)

# NHIỆM VỤ: Chuyển ô CheXpert trống (blank) từ IGNORE (-100) sang NEGATIVE (0), giống bài META-CXR gốc

Repo: ~/Meta-CXR (github.com/minhphuong150505/Meta-CXR). Đọc CLAUDE.md và docs/handoff/README.md trước khi làm.
Tạo file bàn giao: docs/handoff/PLAN-2026-09-24-blank-as-negative.md (chép prompt này vào phần Plan). Khi xong, nối `## Execution report` vào cùng file.

## Bối cảnh
- Bài gốc (Edirisinghe et al., IEEE Access 2025) ghi: "missing (NaN) values were treated as the negative class". Repo gốc dùng `fillna(0.0)`.
- Repo này hiện map blank → IGNORE_LABEL = -100 và loại từng ô khỏi loss. Ô trống chiếm ~79.5% ma trận nhãn.
- Quyết định mới: blank → 0 (negative), tính loss bình thường. Lý do: khớp với bài gốc và khớp với framing đánh giá `study_presence`, vốn đã coi blank là vắng mặt.

## Phạm vi — CHỈ làm những việc sau
1. **Thêm một khóa config** `model.mhcac.blank_label_policy` với hai giá trị `negative | ignore`.
   - Mặc định trong `pretraining/configs/mimic_cxr_full.yaml` đặt là `negative`.
   - Nếu code không tìm thấy khóa thì mặc định `ignore`, để các config cũ và ablation vẫn tái lập được.
   - Nếu gặp giá trị lạ thì raise ValueError, không fallback im lặng.
2. **`model/lavis/data/ReportDataset.py`** (khối `.replace(-1, 2).fillna(IGNORE_LABEL)`, khoảng dòng 357–375):
   - Khi policy = `negative`: ô blank của một study CÓ bản ghi CheXpert → 0.
   - GIỮ NGUYÊN các chỗ sau:
     - Study KHÔNG khớp bản ghi CheXpert nào (khối `fillna(IGNORE_LABEL)` khoảng dòng 541, `_chexpert_merge != "both"`) vẫn là -100 cho mọi nhãn và vẫn bị `classification_valid` loại. Tuyệt đối không biến chúng thành 14 số 0.
     - Mention targets (`_mention_*`) vẫn được tính TRƯỚC bước fill, như hiện tại.
     - `excluded_labels` vẫn được áp SAU bước fill, như hiện tại.
3. **`preporcessing/preprocess_mimic_cxr.py`** (khoảng dòng 102–122): code tự ghi là phải giữ đồng bộ với ReportDataset. Thêm cùng policy đó, qua CLI flag hoặc tham số, mặc định khớp với YAML. Cập nhật comment giải thích.
4. **Tính lại `class_weights`** trong `mimic_cxr_full.yaml`:
   - Dùng đúng công thức hiện có: `[1.0, n_neg/n_pos, n_neg/n_unc]`, kappa = 1, cap 10.
   - Đếm trên split TRAIN theo mức study, với blank đã tính là negative.
   - Viết một script nhỏ, chạy trên CPU, trong `scripts/`. Script in bảng pos/neg/unc/blank-cũ theo từng nhãn và in weight mới.
   - Dán bảng vào comment YAML theo đúng format đang có. Giữ bảng cũ trong comment, ghi rõ là "masked-policy, superseded".
   - Kiểm tra `default_class_weights` trong `model/lavis/models/blip2_models/blip2_qformer.py` (khoảng dòng 526). Nếu nó được dùng khi YAML không set thì báo lại, KHÔNG tự sửa.
5. **Tests**:
   - `tests/test_blank_label_masking.py` đang khẳng định "blank must never train as a negative". Parametrize theo policy: `ignore` giữ nguyên toàn bộ assert cũ; `negative` assert blank → 0.
   - Thêm test: study không có bản ghi CheXpert vẫn là -100 dưới cả hai policy.
   - Thêm test: mention targets giống hệt nhau dưới cả hai policy.
   - Thêm test: policy lạ thì raise.
6. **Kiểm tra evaluator**: `training/evaluation/label_framing.py`, `scripts/evaluate_stage1.py`, `scripts/calibrate_thresholds.py`, file `.npz` do eval hook ghi ra.
   - Xác định: khi nhãn trong `.npz` đã là 0 thay vì -100, framing `study_presence` có cho ra ĐÚNG ma trận ground truth như trước không? Framing `masked_polarity` có còn tính được không?
   - Nếu `masked_polarity` mất khả năng phân biệt blank thì BÁO LẠI kèm đề xuất (ví dụ lưu thêm mask blank vào `.npz`). KHÔNG tự sửa evaluator.
7. **Tài liệu, cập nhật trong cùng commit**: README.md (mục ngữ nghĩa nhãn), CLAUDE.md, docs/so_sanh_voi_repo_goc.md mục 1, và một mục quyết định mới trong struct/project/_meta/DECISIONS.md. Ghi rõ đây là đảo ngược quyết định cũ, kèm lý do ở phần Bối cảnh.

## GIỮ NGUYÊN — các quyết định người dùng CHƯA chốt (không được tự đổi)
- `uncertain_policy: ignore_uncertain`
- `excluded_labels: ["No Finding"]` → nhưng HÃY BÁO số pos/neg của No Finding dưới policy mới
- `lambda_gate: 0.5` và toàn bộ mention gate / `gate_class_weights` / `mention_conditioned_pos_weights`
- `label_smoothing: 0.05`, mọi lambda khác, encoder_finetune, batch 16 × accum 4
- Contrastive loss (AbnormalitySpecificLoss), teacher_cls và distill sẽ tự nhận blank = 0 qua dataset. KHÔNG thêm logic lọc cặp blank.
- Stage 2, cue rules, prompt — không đụng.

## Kiểm chứng (bắt buộc trước khi báo xong)
1. CPU test suite trên training host (`CUDA_VISIBLE_DEVICES=""`). Baseline hiện tại theo README: 1,012 passed, 2 skipped. Báo số mới; mọi test fail mới phải được giải thích.
2. Thống kê nhãn trên train/val/test dưới cả hai policy:
   - Số ô 0/1/2/-100 theo từng nhãn.
   - Số study có `classification_valid = True`.
   - Xác nhận tổng số ô -100 dưới policy `negative` chỉ đến từ study không có bản ghi CheXpert cộng các cột `excluded_labels`.
3. Smoke Stage 1 ngắn trên GPU: `run.truncate_train` nhỏ, 1 epoch, output_dir MỚI trên /home.
   - Trước khi chạy phải kiểm tra `pgrep` và `nvidia-smi` rảnh, và output_dir chưa tồn tại. Chỉ bấm launch MỘT lần.
   - Báo: loss từng term ở vài iter đầu và cuối, `s/it`, `max mem`, không NaN/inf.
4. KHÔNG chạy full training 10 epoch. Chờ người dùng xác nhận các mục ở phần "GIỮ NGUYÊN".

## Execution report cần có
- Lệnh đã chạy và exit status.
- Diff tóm tắt theo file.
- Bảng class weight cũ/mới.
- Bảng thống kê nhãn.
- Kết quả pytest.
- Kết quả smoke.
- Phát hiện về evaluator ở mục 6.
- Số pos/neg của No Finding.
- Mọi chỗ bạn thấy mâu thuẫn với giả định trong prompt này: DỪNG và ghi lại, không tự ứng biến.

Tóm tắt log, không dán nguyên file. Tuyệt đối không đưa dữ liệu bệnh nhân (dicom_id, study_id, text báo cáo) vào commit, handoff hay tóm tắt.

---

## Goal

`blank_label_policy: negative` ships in `mimic_cxr_full.yaml` with class weights
recomputed for it, label statistics verified on all three splits, the CPU suite
green on the host, and one 1-epoch GPU smoke clean. No full run.

## Preconditions

- The planning checkout's working tree (branch `feat/stage2-finding-tokens`),
  **not yet committed** — see "Why it is not committed" in the report below.
  Once committed and pushed, the host needs `git pull` first.
- `model/lavis/data/chexpert_labels.py` is under a git-ignored directory: it must
  be committed with `git add -f`, or the host pulls a ReportDataset that imports a
  module it does not have.
- Host: `findmnt -no FSTYPE /mnt/drive1tb` prints `ntfs3`; venv
  `~/.venvs/meta-cxr-stage1-311`.

## Commands (executor, on the host)

```bash
cd ~/Meta-CXR && git pull && git log --oneline -1
PY=~/.venvs/meta-cxr-stage1-311/bin/python

# 1. Label counts + class weights (CPU, a few minutes). Paths come from the
#    host's own configs/env_config.yaml: chexpert_csv, and processed_dir
#    (.../processed/full_allviews_v2).
$PY scripts/count_chexpert_blank_policy.py \
    --chexpert-csv <chexpert_csv> \
    --split-dir <processed_dir> \
    --cfg-path pretraining/configs/mimic_cxr_full.yaml \
    --json-out $HOME/blank_policy_counts_20260924.json
echo rc=$?

# 2. CPU suite on the host
CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/ training/test_stage2_utils.py -q --tb=line 2>&1 | tail -30

# --- planner pastes the new class_weights into the YAML, commits, pushes; then:
git pull

# 3. Smoke -- guard first, launch ONCE. Run the guard IN a shell on the host, not
#    inside `ssh ... "pgrep -f ..."`: that matches the ssh command line itself.
pgrep -af "pretraining.train|run_medgemma_qlora|generate_stage2" ; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
test ! -e $HOME/smoke_blankneg_20260924 || { echo "output dir exists, abort"; exit 1; }
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled $PY -m pretraining.train \
  --cfg-path pretraining/configs/mimic_cxr_full.yaml \
  --options run.output_dir=$HOME/smoke_blankneg_20260924 run.truncate_train=2000 \
    run.truncate_val=200 run.max_epoch=1 run.eval_start_epoch=0 \
  > $HOME/smoke_blankneg_20260924.log 2>&1 < /dev/null
echo rc=$?
grep -m1 "CheXpert blank_label_policy" $HOME/smoke_blankneg_20260924.log   # must say: negative
```

## Expected

- Step 1 exits 0 (the `-100` provenance check passes on every split under
  `negative`). Under `negative`, `-100` cells per label = studies with no record
  + studies whose record is all blank (+ every study for an `excluded_labels`
  column — none today, see Mismatch 1).
- `classification_valid` under `negative` ≥ under `ignore` (it can only grow:
  studies whose only label was excluded regain cells; with `excluded_labels: []`
  it should be equal).
- Step 2: the host baseline failures only (the four `test_native_independence`
  env-config ones, as recorded in CLAUDE.md) plus nothing new.
- Step 3: the log line says `negative`; `loss_cls` is finite and **higher** than
  a masked-policy smoke would be (about 5x more cells carry a target: ~2.86 of 14 per study before, ~13-14 now);
  `loss_gate` unaffected in scale; no NaN/inf; `s/it` ~0.25–0.45; `max mem`
  within the 9.8 GiB of the deep-unfreeze recipe.

## Abort if

- Step 1 exits non-zero (provenance check) — a study with no CheXpert
  information is being filled with zeros somewhere.
- Any new pytest failure outside the recorded host baseline.
- The smoke log does not print `CheXpert blank_label_policy: negative`, or any
  loss term is NaN/inf, or OOM.
- The GPU or the output path is already in use. Do not retry; report.

---

## Execution report — 2026-09-24, planning checkout only (no host)

- **Host:** unreachable. `tailscale status` lists `minhphuong 100.116.167.90
  ... offline, last seen 10d ago`; `ssh phuong@100.116.167.90` → `connect ...
  port 22: Connection timed out`. No dataset exists on the planning machine
  (searched for the CheXpert export and `full_allviews_v2`: none).
- **So verification steps 1–3 were NOT run, and the class weights were NOT
  recomputed.** Everything that needs no data is done and CPU-tested.

### What changed, by file

| file | change |
|---|---|
| `model/lavis/data/chexpert_labels.py` (**new**, git-ignored dir → `git add -f`) | numpy/pandas only. `IGNORE_LABEL`, `BLANK_LABEL_POLICIES`, `DEFAULT_BLANK_LABEL_POLICY="ignore"`, `resolve_blank_label_policy` (unknown → `ValueError`), `map_chexpert_labels`, `prepare_chexpert_labels` (mention targets before the fill, `excluded_labels` after), `attach_chexpert_labels` (join, `classification_valid`, `mention_valid`, `-100` for unmatched studies) |
| `model/lavis/data/ReportDataset.py` | the label block and the merge block now call the two helpers above; reads `model.mhcac.blank_label_policy`; logs `CheXpert blank_label_policy: <p>`. Behaviour under `ignore` is unchanged by construction (same expressions, moved) |
| `preporcessing/preprocess_mimic_cxr.py` | `--blank-label-policy {negative,ignore}`, default `negative`; `clean_chexpert(df, policy)`. Label columns are still not written, so **no manifest rebuild is needed** |
| `pretraining/configs/mimic_cxr_full.yaml` | `model.mhcac.blank_label_policy: negative` with the reversal rationale; a **PENDING** warning above `class_weights` (table unchanged) |
| `scripts/count_chexpert_blank_policy.py` (**new**) | CPU counts per split × policy, `-100` provenance check (non-zero exit on failure), class weights for `negative`. Tested on synthetic data only |
| `tests/test_blank_label_masking.py` | rewritten around the real helpers: parametrized by policy; unmatched study `-100` under both; mention targets identical under both; unknown policy raises; all-blank record stays `-100`; `excluded_labels` after the fill; preprocessing ≡ dataset mapping; shipped YAML selects `negative` |
| README, CLAUDE.md, `docs/so_sanh_voi_repo_goc.md` §1, `struct/…/DECISIONS.md` D-018, `struct/` pages for ReportDataset, `chexpert_labels.py` (new), preprocess, scripts index, `count_chexpert_blank_policy.py` (new) | reversal recorded with the prompt's rationale; old text kept as history |

### Tests (CPU planning box, `~/venv`, no torchvision/transformers)

`CUDA_VISIBLE_DEVICES="" python -m pytest tests/ --ignore=tests/test_blip2_negative_sampling.py --ignore=tests/test_encoder_ablation.py`
→ **15 failed, 1019 passed, 22 skipped**. The failure list is **byte-identical**
to the same command on the pre-change tree (stashed): all 15 are missing
torchvision/transformers on this box (`test_cue_contract` ×6,
`test_generation_stop_tokens` ×4, `test_native_independence` ×4,
`test_stage1_eval_hook` ×1). No new failure. `ruff check` on the four edited
Python files: 2 findings, both pre-existing in `preprocess_mimic_cxr.py`.
The host suite has **not** run.

### Findings — mismatches with the prompt's assumptions (stopped, not improvised)

1. **`excluded_labels` is `[]` in `mimic_cxr_full.yaml`, not `["No Finding"]`.**
   It was put back on 2026-08-15 (see the YAML comment and CLAUDE.md). I left it
   at `[]`. Consequence: under `negative`, `No Finding` is **in** the
   classification head, and its blanks become real negatives — which for the
   first time gives it a P/N split. From the recorded train counts it should be
   **74,305 positive / ~148,453 negative** (the not-mentioned count; the exact
   figure, minus no-information studies, is what step 1 prints). Only the code
   *default* (key absent) is `["No Finding"]`. Decide whether that is wanted.
2. **All-blank CheXpert records.** The prompt defines the `-100` exception as
   "studies with no CheXpert record". A study can also have a record whose 14
   cells are all blank (`preprocess_mimic_cxr.py` already counts and flags these
   `has_chexpert_label=False`). I kept those at `-100` too, under both policies —
   filling them would invent a fully-normal study, and `classification_valid`
   already excludes them via the processed flag. Upstream's `fillna(0.0)` would
   zero them. Step 1's `record_all_blank` field gives the count.
3. **`default_class_weights` in `blip2_qformer.py:526` IS used when the YAML does
   not set `class_weights`** (`if class_weights is None: class_weights =
   default_class_weights`, `:543`). It holds the masked-policy sqrt weights. Not
   touched, per the prompt. Any config under `negative` without its own
   `class_weights` would silently train with them.
4. **The "1,012 passed, 2 skipped" baseline** in the prompt is not what this box
   measures (see above); the host number is the one to compare against.

### Evaluator (item 6) — read from code, not measured

- The eval hook (`model/lavis/tasks/image_text_pretrain.py:134-141`) writes
  `labels` to the `.npz` with every cell it did not score set to **-1** (not
  -100): `valid = sample_mask & (labels >= 0) & (labels < C)`. Under `ignore` a
  blank was -1; under `negative` it is **0**.
- **`study_presence`: identical ground truth.** `frame_labels` maps
  `labels == POSITIVE → 1`, everything else → 0, so -1 and 0 both become 0.
  Studies with no CheXpert information are -1 under both policies. ✔
- **`masked_polarity`: still computes, but answers a different question.** It
  keeps labels as-is and masks only -1, so under `negative` every blank is
  scored as an explicit negative — this is no longer "polarity given mention".
  Nothing warns. **Do not quote `masked_polarity` from a `negative` run.**
- The in-training validation metrics (`positive_macro_f1` etc. in the eval hook's
  confusion matrix) change meaning the same way, and `val_loss` (the
  `selection_metric`) now sums over roughly 5x more cells, so its scale is not
  comparable to any earlier run's.
- **Proposal, not implemented:** have the eval hook also save
  `mention_targets` (already in the batch) into the `.npz`; `frame_labels`
  under `masked_polarity` could then set `labels[mention == 0] = -1` and recover
  the historical framing exactly, whatever the training policy.

### Why it is not committed

The YAML now selects `negative` but still carries the masked-policy
`class_weights`, which are wrong for it (e.g. Atelectasis `w_pos` 0.034 was
derived from 44,718 pos / 1,502 neg; under `negative` its negatives grow by the
blank count). Pushing that to the public remote would let the next pull on
the host launch a mis-weighted run. Commit once step 1 has produced the weights
and they are pasted into the YAML (old table kept, labelled "masked-policy,
superseded").

### Not done

- Steps 1–3 of "Kiểm chứng" (host offline).
- Class-weight table in the YAML (needs step 1).
- No full run (as instructed).

---

## Execution report — 2026-09-24 19:30–19:40, host `minhphuong` (100.116.167.90)

The host came back up (uptime 11 min at 19:32, `/mnt/drive1tb` = `ntfs3`, GPU
0 % / 18 MiB, no training process). Because the change was still uncommitted,
the working tree was copied as a snapshot to `~/blankneg_20260924_src` (tracked
files + the new files, host's own `configs/env_config.yaml`); `~/Meta-CXR`
(on `main`, `79139fc`) was not touched. The snapshot has no `.git`.

### Step 1 — counts and class weights → exit 0

`scripts/count_chexpert_blank_policy.py --chexpert-csv
/mnt/drive1tb/mimic-cxr-jpg-full/mimic-cxr-2.0.0-chexpert.csv.gz --split-dir
.../processed/full_allviews_v2`. Raw output: `~/blank_policy_counts_20260924.{txt,json}`.

| split | studies | no record | record all blank | `classification_valid` (both policies) | `-100` provenance under `negative` |
|---|---:|---:|---:|---:|---|
| train | 222,758 | 8 | 2,371 | 220,379 | ✔ 33,306 = 2,379 × 14, 0 stray |
| val | 1,808 | 0 | 22 | 1,786 | ✔ 308 = 22 × 14 |
| test | 3,269 | 0 | 21 | 3,248 | ✔ 294 = 21 × 14 |

`classification_valid` is equal under both policies because
`excluded_labels: []`. Train, study level:

| label | pos | neg `ignore` | neg `negative` | unc | old `w_pos` (`ignore`) | new `w_pos` (`negative`) |
|---|---:|---:|---:|---:|---:|---:|
| No Finding | 74,305 | 0 | 146,074 | 0 | 1.000 | 1.966 |
| Enlarged Cardiomediastinum | 6,968 | 5,168 | 204,328 | 9,083 | 0.742 | 10.000 (raw 29.32) |
| Cardiomegaly | 43,602 | 15,613 | 170,908 | 5,869 | 0.358 | 3.920 |
| Lung Opacity | 50,099 | 2,993 | 166,554 | 3,726 | 0.060 | 3.324 |
| Lung Lesion | 6,091 | 841 | 213,174 | 1,114 | 0.138 | 10.000 (raw 35.00) |
| Edema | 26,093 | 25,133 | 181,509 | 12,777 | 0.963 | 6.956 |
| Consolidation | 10,476 | 7,817 | 205,717 | 4,186 | 0.746 | 10.000 (raw 19.64) |
| Pneumonia | 16,093 | 23,904 | 186,489 | 17,797 | 1.485 | 10.000 (raw 11.59) |
| Atelectasis | 44,718 | 1,502 | 165,620 | 10,041 | 0.034 | 3.704 |
| Pneumothorax | 10,171 | 41,271 | 209,102 | 1,106 | 4.060 | 10.000 (raw 20.56) |
| Pleural Effusion | 52,759 | 26,615 | 161,981 | 5,639 | 0.504 | 3.070 |
| Pleural Other | 1,933 | 126 | 217,703 | 743 | 0.065 | 10.000 (raw 112.62) |
| Fracture | 4,283 | 877 | 215,548 | 548 | 0.205 | 10.000 (raw 50.33) |
| Support Devices | 64,868 | 3,422 | 155,281 | 230 | 0.053 | 2.394 |

`w_uncertain` is 10.000 (capped) everywhere except No Finding (1.000, no
uncertain cells); it is inert under `ignore_uncertain`, which drops class-2
cells (`mhcac/loss.py:108`). Val and test per-label counts are in the raw file.

**No Finding under `negative`: 74,305 positive / 146,074 negative** (train);
val 582 / 1,204; test 568 / 2,680. (The planning-side estimate of ~148,453 was
the not-mentioned count including the 2,379 no-information studies.)

⚠ **7 of 14 `w_pos` sit at the cap of 10**, with raw ratios up to 112.62
(Pleural Other). That is the prompt's formula applied as specified; whether the
cap should stay at 10 is not decided here.

### Step 2 — CPU suite on the host → 3 failed, 1087 passed, 2 skipped

`CUDA_VISIBLE_DEVICES="" python -m pytest tests/` in the snapshot. Raw:
`~/blankneg_pytest_20260924.log`.

- First pass found one real failure, now fixed:
  `test_encoder_finetune.py::TestShippedConfig::test_every_kappa_is_one` pinned
  Pneumothorax `w_pos` at 4.060, its masked-policy ratio. It now asserts `w_pos
  == min(n_neg/n_pos, 10)` for six labels against the `negative` counts, and that
  the shipped policy is `negative`.
- The remaining 3 (`test_explain_runner::test_an_ignored_repo_path_is_accepted`,
  `test_notebook_privacy::test_the_private_notebook_is_still_ignored_and_untracked`,
  `test_trainer_resume::test_git_sha_is_recorded_in_this_repo`) need a git
  checkout; the snapshot has none. All three **pass** in `~/Meta-CXR`.
- `training/test_stage2_utils.py` named in the prompt does not exist in this
  checkout.

### Step 3 — GPU smoke → clean

Guards passed (no training process, GPU 0 % / 18 MiB, output dir and log absent);
launched once from the snapshot:
`pretraining.train --options run.output_dir=$HOME/smoke_blankneg_20260924
run.truncate_train=2000 run.truncate_val=200 run.truncate_test=200
run.max_epoch=1 run.eval_start_epoch=0`. Log: `~/smoke_blankneg_20260924.log`.

- `CheXpert blank_label_policy: negative` printed.
- 125 iterations in 45 s, **0.3624 s/it**; `max mem` 6,892 → **9,839 MiB**.
- Losses, iteration 0 → 124 (epoch average in brackets):
  `loss_cls` 1.3481 → 1.0943 (1.1910), `loss_teacher_cls` 1.3367 → 1.0915
  (1.1966), `loss_gate` 0.9819 → 0.9366 (0.9693), `loss_mpc` 4.3651 → 3.6364
  (4.0291), `loss_contrastive` 0.2672 → 0.2765 (0.2750), `loss_distill` 0.0049 →
  0.0054, `loss_view_consistency` 0.0011 → 0.0007; `loss_itc/itm/lm` 0 as
  configured. Zero `nan`/`inf` in the log.
- Validation, checkpoint_best and test prediction files written. In the `.npz`,
  `-1` appears only in rows that are entirely `-1` (val 2 rows = 28 cells, test
  1 row = 14): the no-information studies. Blanks arrive as `0`, as predicted in
  the evaluator section above. Probabilities all finite.
- The 2,000-study smoke says the pipeline runs; it says nothing about quality.

### Not done

- No full 10-epoch run (as instructed; waiting on the "GIỮ NGUYÊN" items).
- The `.npz` `mention_targets` proposal for `masked_polarity` is not
  implemented.
- `default_class_weights` in `blip2_qformer.py` is unchanged (masked-policy).

---

## Decision update (user, 2026-09-24) — overrides "GIỮ NGUYÊN"

1. `uncertain_policy`: keep `ignore_uncertain`.
2. No Finding back in the head, all 14 labels: `excluded_labels: []`; its class
   weight from the same formula, `w_uncertain = 1.0` when `n_unc = 0`; evaluator
   must report macro 13 (without No Finding) and macro 14; a test that No Finding
   carries valid 0/1. Stage 2 / cue rules: report only.
3. Mention gate OFF (option A): `lambda_gate: 0`, code kept; nothing gate-related
   may reach the P/N/U loss or class weights; eval/calibrate default to `q_pos`
   and must raise if a marginal score is requested with the gate off; Stage 2
   not touched, only listed; docs updated (D-019).

## Execution report — 2026-09-24 19:45–19:55, planning checkout + host `minhphuong`

### Findings the prompt asked to report

- **`excluded_labels` was already `[]`** (since 2026-08-15); unchanged. Only the
  code default in `ReportDataset.py` and `count_chexpert_blank_policy.py`
  (`["No Finding"]` when the key is absent) still excludes it — left as the
  back-compat default. Other No Finding special cases found by grep:
  `training/evaluation/schemas.py::META_LABELS` (No Finding + Support Devices out
  of the primary macro; kept, 13/14 views added instead),
  `scripts/probe_soft_tokens.py` (same meta-label set, diagnostic),
  `mhcac/utils.py`, `mhcac/mhcac_7.py` (legacy label lists);
  Stage 2, not touched: `stage2/prompts/ontology.py`,
  `training/train_eval_figure9_llm_variants_200.py:215,433,517`,
  `scripts/calibrate_cue_precision.py:35`, `training/medgemma/finding_tokens.py:78`
  (No Finding excluded from the 13 finding tokens), `safety/claims.py:48`.
- **The gate was configured over all 14 labels** (`gate_class_weights` has 14
  rows incl. No Finding); no No-Finding-specific gate mask exists. Nothing
  changed there beyond `lambda_gate: 0`.
- **No Finding class weight:** `[1.0, 1.966, 1.000]` — 74,305 pos / 146,074 neg /
  0 unc on train; `w_uncertain = 1.0` with the reason in the YAML comment.

### Gate/mention/marginal classification (grep over `mhcac/ model/ training/ scripts/`)

| where | kind | with `lambda_gate = 0` |
|---|---|---|
| `mhcac/mhcac_12.py` `mention_heads` | parallel head on the pooled representation; does not feed `student_logits` | exists, no gradient |
| `mhcac/loss.py::MentionGateLoss`, `blip2_qformer` `loss_gate` | (a) separate BCE term | added only under `if self.lambda_gate > 0`; logs `0.0000` |
| `mhcac/loss.py::MentionConditionedClassificationLoss` + `mention_conditioned_pos_weights` | (a) separate hierarchical objective; replaces `lambda_cls` only when on | not constructed at `lambda_mc = 0` |
| `gate_class_weights` | (a) only into `MentionGateLoss` | inert |
| P/N/U `class_weights` override | (b) — **none found**: no gate table ever reached `ClassificationLoss` | pinned by test |
| `image_text_pretrain.py` mention export → `.npz` | (c) | **not exported**, `metadata.mention_gate_trained=False` (new) |
| `label_framing.py` `marginal_presence`; `evaluate_stage1.py` / `calibrate_thresholds.py` `--score` | (c) default already `conditional_positive` | `marginal_presence` now **raises** on a gate-off file (new) |
| `threshold_calibration.py:190`, `runner_base.py:765`, `evaluate_explanation.py:339` | comments / tuple unpacking only | — |
| `train_eval_figure9_llm_variants_200.py` (`build_stage1_records` `return_mention=True`, `record["mention_logits"]`, cue rules `marginal_positive` / `mention_gated`), `run_medgemma_qlora.py` (`marginal_positive` default for MHCAC-prompt modes), `generate_stage2_reports.py` (`--cue-rule`, copies `mention_logits`), `finding_tokens.py` (`full` uses `m`), `calibrate_cue_precision.py` (`mention_probabilities × q_pos`) | **(d) Stage 2 — NOT changed** | would read a **random** head from a gate-off Stage-1 checkpoint |

### Code changes

- `mhcac/loss.py`: `mention_gate_is_trained()`, `build_classification_losses()`
  (moved the three constructions and the lambda-conflict checks out of
  `Blip2Qformer.__init__`, behaviour unchanged).
- `blip2_qformer.py`: uses the builder; sets `self.mention_gate_trained`.
- `image_text_pretrain.py`: no mention export when the gate is untrained;
  writes `mention_gate_trained` into prediction metadata.
- `label_framing.py`: `MENTION_GATE_TRAINED_KEY`; `marginal_presence` raises on
  gate-off files (older files without the key behave as before).
- `classification_metrics.py`: `<metric>_13labels` / `<metric>_14labels` for
  macro AUROC/AUPRC, positive-macro F1/P/R, macro specificity. Primary macro
  still 12 labels.
- `mimic_cxr_full.yaml`: `lambda_gate: 0.0` with rationale; No Finding weight
  comment.
- `tests/test_gate_off.py` (17 tests).

### Verification

- CPU box: same 15 baseline failures as before (torchvision/transformers
  missing), no new ones.
- Host snapshot `~/blankneg_20260924_src`: **3 failed, 1104 passed, 2 skipped**
  (= previous 1087 + 17 new); the 3 are the git-only tests, which pass in
  `~/Meta-CXR`. Log `~/gateoff_pytest_20260924.log`.
- Smoke `~/smoke_gateoff_20260924` (same guards, launched once, rc 0; log
  `~/smoke_gateoff_20260924.log`): 125 iters, **0.3528 s/it**, `max mem`
  **9,839 MiB**, no NaN/inf, no traceback; log line "mention gate is untrained
  ... not exporting mention_probabilities".

  | iter | total loss | `loss_cls` | `loss_teacher_cls` | `loss_gate` |
  |---|---:|---:|---:|---:|
  | 0 (gate on, earlier smoke) | 2.6545 | 1.3481 | 1.3367 | 0.9819 |
  | 0 (gate off) | **2.1636** | 1.3481 | 1.3367 | **0.0000** |
  | 50 (gate off) | 1.9585 | 1.2298 | 1.2412 | 0.0000 |
  | 124 (gate off) | 1.7435 | 1.0863 | 1.0834 | 0.0000 |

  2.6545 − 0.5 × 0.9819 = 2.1636: the only difference from the gate-on smoke at
  iteration 0 is the removed gate term.
- Prediction `.npz` carry no `mention_probabilities`,
  `metadata = {"mention_gate_trained": false}`.
  `calibrate_thresholds.py --score marginal_presence` → exit 2 with the new
  message; the default score calibrates and `evaluate_stage1.py` reports
  `macro_auroc` / `_13labels` / `_14labels` (smoke values, meaningless as
  quality).

### Not done

- No full run. Stage 2 not touched (see table). `default_class_weights` in
  `blip2_qformer.py` unchanged.
