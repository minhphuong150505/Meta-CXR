# PLAN 2026-10-28 — Mô hình Stage 2 cuối có đọc cue và soft token không (hai phép thử chỉ suy luận)

Trạng thái: **USER DUYỆT 2026-10-28** (sửa code, commit + push, chạy GPU). Code `--eval-cues` đã
viết, test CPU qua; chưa chạy GPU.

## Vì sao

Mô hình Stage 2 đang báo cáo là P2 + LR 3e-4, 2 epoch, nằm ở `~/stage2_p2_20261011/full`. Nó đọc ảnh
qua hai kênh: 32 soft token (đã sụp gần như về một vector) và danh sách cue P/N/U của MHCAC (theo
cutpoint). Hai điều đã biết:

- v3 **bỏ qua cue**: thay cue bằng cue lấy từ nhãn thật không đổi chỉ số nào (1.000 study test,
  greedy; `docs/TONG_HOP_2026-10-11.md` mục 4.2).
- Một adapter train cùng recipe trên 10.000 study **có đọc soft token**: tráo soft token giữa các
  study làm BERTScore giảm 0,040 và CheXbert micro F1-14 giảm 0,083 (greedy, 500 val;
  `PLAN-2026-10-15-qformer-refit.md`, mục Pilot A/B).

Chưa đo trên chính mô hình cuối:

- **(O)** Nó có đọc cue không, và cue tốt hơn có làm báo cáo tốt hơn không?
- **(P)** Nó dựa vào kênh soft token (kênh ảnh) đến mức nào?

Hai phép thử này trả lời câu hỏi "soft token sụp và cue bị bỏ qua có làm mô hình tệ không". Không
train gì, không đổi mô hình báo cáo.

## Thay đổi code (opt-in; không truyền cờ thì bit-identical)

`scripts/generate_stage2_reports.py` thêm cờ `--eval-cues {predicted,oracle}`, mặc định `predicted`.

- **`predicted`** (mặc định): đúng như hiện nay. Cue lấy từ MHCAC theo `--cue-rule`.
- **`oracle`**: sau khi dựng record Stage 1, thay `pred_groups` của mọi record bằng lớp thật CheXpert.
  - Dùng lại đúng hàm đã dùng lúc train: `training/dataio/oracle_cues.true_label_codes` và
    `mix_oracle_cues(model_share=0.0)`, với cùng ánh xạ nhãn D-018.
  - Bỏ qua `No Finding`, như lúc train. Ô không có thông tin CheXpert giữ lớp của MHCAC.
  - Thay thế diễn ra sau cache Stage 1, nên không đổi danh tính cache.
  - Chỉ in số đếm tổng hợp (`cells`, `changed`, `unknown_truth`), không in gì theo study.
- **Từ chối** các trường hợp sau:
  - `oracle` với chế độ không có cue MHCAC hoặc với `--cue-rule none`;
  - `oracle` dùng cùng `--soft-token-ablation`, để mỗi arm chỉ có một can thiệp.
- `generation_summary.json` ghi thêm `eval_cues` và số đếm.
- **Test CPU** (`tests/test_generate_stage2_reports.py`):
  - mặc định không đổi;
  - ba trường hợp từ chối ở trên;
  - `oracle` thay đúng lớp trên bản ghi giả, bỏ qua `No Finding`, và giữ lớp MHCAC khi ô không có
    thông tin.
- **Tài liệu:** `CLAUDE.md`, `README.md`, trang `struct/` của script này, sửa cùng commit.

`--soft-token-ablation permute_across` đã có sẵn từ 2026-10-15, nên phép thử (P) không cần sửa code.

## Các arm

Tất cả dùng cùng adapter (mô hình cuối), cùng 1.360 study val, beam 4, length penalty 2,0, tối đa
256 token mới (đúng cấu hình đã báo cáo), seed 16. Mỗi arm chỉ khác nhau một can thiệp:

| arm | cue | soft token |
|---|---|---|
| C (đối chứng) | MHCAC, cutpoint (như báo cáo) | của chính study |
| O | **nhãn thật** (`--eval-cues oracle`) | của chính study |
| P | MHCAC, cutpoint | **của study khác** (`--soft-token-ablation permute_across`, hoán vị không điểm bất động) |

Indication và FINDINGS lần chụp trước giữ nguyên ở cả ba arm. Test chỉ dùng cho bước kiểm tra tương
đương (G0), không chấm chỉ số nào trên test.

## Các bước (executor, trên máy train)

Checkout `~/Meta-CXR-v3`: đây là checkout của run train, và plan MBR 2026-10-14 đã tái lập 20/20 beam
từ checkout này.

```bash
cd ~/Meta-CXR-v3 && git pull     # lấy commit có --eval-cues
findmnt -no FSTYPE /mnt/drive1tb  # phải là ntfs3
nvidia-smi                        # GPU rảnh; không job nào khác đang chạy
OUT=~/s2probe_20261028; test ! -e $OUT || exit 1
```

Cờ Stage 1, prompt và adapter phải **chép nguyên từ `~/stage2_genfull.sh`**, script đã sinh
`gen_test_beam_full`. Gồm: `--pipeline-mode meta_cxr_qformer_with_mhcac_prompt`,
`--prompt-style paper_context`, `--cue-rule cutpoints`,
`--threshold-path configs/stage1_cutpoints/run_20261005_paper.json`, `--adapter`, `--checkpoint-root`,
`--stage1-run`, `--num-beams 4`, `--length-penalty 2.0`, `--max-new-tokens`. Mỗi arm chỉ đổi
`--split`, `--limit`, `--output-dir`, `--stage1-cache-dir $OUT/cache` và đúng một cờ can thiệp.
Không ghi gì vào thư mục của run đã báo cáo.

0. **G0 — kiểm tra tương đương (khoảng 2 phút).** Sinh lại 20 study test bằng arm C
   (`--split test --limit 20`, mẫu cố định theo seed 16). Ghép theo `sample_key` với
   `~/stage2_p2_20261011/gen_test_beam_full`; cả 20 phải trùng từng chữ. Không trùng thì **dừng**:
   code mới không tái lập được mô hình đã báo cáo.
1. **Arm C** trên val: `--split val --limit 0`, khoảng 85 phút.
2. **Arm O**: như C, thêm `--eval-cues oracle`, khoảng 85 phút. Ghi lại số đếm `changed` và
   `unknown_truth`.
3. **Arm P**: như C, thêm `--soft-token-ablation permute_across`, khoảng 85 phút.
4. **Chấm điểm.** Dùng `score_stage2_paper_protocol.py --pred C=... --pred O=... --pred P=...
   --baseline C` (GPU, vài phút). Dùng `score_ce_chexbert.py` với cùng cách truyền `--pred` và
   `--baseline C`, trong `~/.venvs/chexbert`, đặt `HF_HUB_OFFLINE=1` (CPU, khoảng 20 phút). Bootstrap
   ghép cặp 2.000 lần, seed 16.

**Dừng ngay nếu:**
- G0 không đạt 20/20;
- có bất kỳ lỗi sinh nào;
- số record val khác 1.360;
- ba file không cùng danh sách `sample_key` và cùng tham chiếu theo từng dòng;
- arm O có `changed` = 0;
- arm P có study nào giữ soft token của chính nó.

## Quy tắc đọc (chốt trước khi chạy)

Đây là cùng một mô hình, giải mã tất định, và chỉ đầu vào thay đổi. Vì vậy nhiễu seed huấn luyện
không áp dụng, và bootstrap ghép cặp theo study là khoảng tin cậy đúng. NB của Stage 2 dùng để so
hai mô hình khác nhau, không dùng ở đây.

**Chỉ số:**
- Chỉ số quyết định là CheXbert micro F1-14 (nội dung lâm sàng).
- Chỉ số phụ là CheXbert micro F1-5 và BERTScore.
- Báo cáo kèm, không dùng để quyết định: ROUGE-L, BLEU-4, CIDEr, CE lexicon, tỷ lệ báo cáo trùng y
  hệt với arm C, và độ dài trung vị.

**Phép thử O (O − C):**
- **"Mô hình đọc cue và cue tốt hơn thì báo cáo tốt hơn":** CheXbert micro F1-14 có cận dưới > 0.
- **"Không xác lập được việc đọc cue":** CI của CheXbert micro F1-14 và micro F1-5 đều chứa 0.
  Viết là "không xác lập", không viết là "chứng minh bỏ qua".
- ⚠ Cue thật lấy từ nhãn của chính báo cáo tham chiếu, nên O là **mức trần có rò nhãn**. Không bao
  giờ được báo cáo O như một kết quả đạt được. Nó chỉ trả lời câu hỏi cơ chế, và cho biết nếu Stage 1
  phân loại hoàn hảo thì Stage 2 tối đa được thêm bao nhiêu.
- Khác với lần thử oracle trên v3: mô hình này được train với 80% cue lấy từ nhãn thật, nên cue thật
  là đầu vào **trong phân phối** của nó. Cue MHCAC mới là phần lệch phân phối.

**Phép thử P (P − C):**
- **"Mô hình đọc kênh soft token":** CheXbert micro F1-14 hoặc BERTScore có cận trên < 0.
- Độ lớn của mức tụt chỉ mang tính mô tả. Đặt cạnh số của adapter 10.000 study (BERTScore −0,040,
  CheXbert −0,083) nhưng không so trực tiếp, vì khác cách giải mã, khác tập study và khác budget.
- ⚠ Đầu vào lệch phân phối: phép thử cho biết kênh có được đọc, nhưng không đo đúng phần đóng góp
  của kênh đó.

**Đọc chung (mô tả):** so độ lớn của |O − C| và |P − C| trên CheXbert micro F1-14 để thấy kênh nào
mang nhiều nội dung lâm sàng hơn với mô hình cuối. Không có quy tắc thắng/thua cho phần này.

## Những gì plan này không làm

- Không train và không đổi mô hình báo cáo.
- Không chấm test; test chỉ dùng 20 study cho G0.
- Không thêm biến thể multi-view.
- Kết quả O hay P đều không tự kéo theo một run mới; mọi bước tiếp theo là quyết định riêng của user.

## Chi phí

- Code và test CPU: khoảng 1 giờ phía planner.
- GPU: khoảng 4,5 giờ (3 arm × ~85 phút, cộng G0 và lần dựng record val khoảng 1 phút).
- CPU: khoảng 20 phút cho CheXbert.
- Đĩa: dưới 200 MB (cache record val khoảng 70 MB và ba file JSONL).

## Xử lý dữ liệu

File JSONL sinh ra, cache record và log ở lại trên máy train. Chỉ số tổng hợp và CI mới được mang về
plan này. Không đưa văn bản báo cáo, `sample_key` hay ID nào vào git.

## Execution report

**G0, 2026-10-28 16:13 (`~/s2probe.sh`, checkout `~/Meta-CXR-v3` at `4c06f55`): FAILED, 8/20 identical.**
The script stopped there as planned. Diagnosis (CPU and log only, no report text):
- **Stage-1 records are identical.** The rebuilt test cache matches the cache of the reported run on
  every field except the cache-internal `sample_key`, which depends on the YAML mtime. Soft tokens
  are bit-identical, and `pred_groups`, `class_logits`, `image_path` and `ref` match on 2,800/2,800.
  The reported run also logged `missing=758`.
- **Same decoding settings:** stop ids `[1, 106]`, beam 4, length penalty 2, 256 new tokens,
  `paper_context` + `cutpoints`.
- **No default-path generation change** between `6dd5688` (the MBR gate, 20/20 on 2026-10-14) and
  `4c06f55`.
- **Unchanged environment:** the MedGemma snapshot dates from 2026-08-30, the adapter from 2026-10-13,
  the venv from August, and no driver or CUDA package changed after 2026-10-14.
- **Generation is deterministic.** A second G0 run (`g0b`) is identical to the first, 20/20. Both match
  the reported output on 8/20.

The cause of the 8/20 mismatch is **unexplained**. The two candidates, both untested, are a code
difference that was missed and a post-reboot numeric change such as a different cuBLAS kernel choice.

**User decision, 2026-10-28:** run arms C, O and P as planned without a C2 control, because
generation is deterministic. G0 is replaced by "deterministic between two runs". The absolute numbers
of C may differ slightly from the reported model; the paired O − C and P − C comparisons stay valid
because all arms run under the same conditions. Part 2 (`~/s2probe2.sh` → `~/s2probe2.log`) was
launched at 16:51:45.

**Part 2, 2026-10-28 16:51:45 → 20:49:51 (`~/s2probe2.sh`): DONE, no abort.** Generation took
C 76 min, O 77 min, P 80 min, and scoring 4 min. Outputs are in `~/s2probe_20261028/{C,O,P}`; only
the aggregates below left the host.

- **Cohort check passed:** 1,360/1,360 studies in each arm, with the same `sample_key` order and the
  same references, and `n_failed` 0.
- **Oracle cue counts:** 17,680 cells (1,360 × 13). 17,407 came from the true label; 273 kept MHCAC's
  cue because 21 studies have no CheXpert information. **1,966 cells changed (11.1%)**, so the
  cutpoint cues already agree with the true class on about 89% of cells.
- **Reports identical to C:** O 459/1,360 (33.8%), P 150/1,360 (11.0%).

Absolute values on val, n=1,360, beam 4, lp 2 (paper protocol; CheXbert in rrg mode):

| arm | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore | lexicon CE | CheXbert micro F1-14 | micro F1-5 | macro F1-14 | median words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C (predicted cues) | 0.2199 | 0.3928 | 0.2014 | 0.7066 | 0.5070 | 0.3403 | 0.5447 | 0.5699 | 0.4109 | 36 |
| O (true-label cues) | 0.2293 | 0.4034 | 0.2089 | 0.7616 | 0.5192 | 0.4359 | 0.6356 | 0.6789 | 0.5447 | 37 |
| P (soft tokens permuted) | 0.1796 | 0.3388 | 0.1787 | 0.4063 | 0.4478 | 0.2896 | 0.4097 | 0.4353 | 0.3218 | 42 |

Paired per-study bootstrap (2,000 resamples, seed 16), with the deciding metric in bold:

| | **CheXbert micro F1-14** | micro F1-5 | BERTScore | macro F1-14 | macro F1-5 | ROUGE-L | BLEU-4 | CIDEr | lexicon CE |
|---|---|---|---|---|---|---|---|---|---|
| O − C | **+0.0908 [+0.0752, +0.1063]** | +0.1089 [+0.0853, +0.1330] | +0.0122 [+0.0064, +0.0180] | +0.1338 [+0.1094, +0.1573] | +0.1268 [+0.0974, +0.1561] | +0.0106 [+0.0057, +0.0157] | +0.0087 [+0.0034, +0.0141] | +0.0551 [+0.0106, +0.1027] | +0.0956 [+0.0733, +0.1178] |
| P − C | **−0.1350 [−0.1550, −0.1142]** | −0.1346 [−0.1629, −0.1058] | −0.0592 [−0.0672, −0.0513] | −0.0892 [−0.1119, −0.0643] | −0.1259 [−0.1571, −0.0943] | −0.0540 [−0.0624, −0.0466] | −0.0541 [−0.0637, −0.0457] | −0.3002 [−0.3793, −0.2296] | −0.0507 [−0.0739, −0.0265] |

**Reading, by the pre-declared rule:**
- **O: "the model reads its cues, and better cues give better reports".** The lower bound of
  CheXbert micro F1-14 is +0.075 > 0, and every metric moves up. Changing 11% of the cue cells
  raises CheXbert micro F1-14 by 0.091. This reverses the v3 result (v3 ignored oracle cues). The
  likely cause is that this adapter was trained with `oracle_mix`, but that is not separated here.
  ⚠ O is a **label-leaking upper bound**: its cues come from the labels of the reference report that
  CheXbert also reads. It is never a reportable result. It says how much Stage 2 could gain from
  Stage-1 cues if Stage 1 classified perfectly.
- **P: "the model reads the soft-token channel".** Both CheXbert micro F1-14 (upper bound −0.114) and
  BERTScore (upper bound −0.051) are below 0. The model reads the collapsed soft tokens, and reports
  get longer without them (median 36 → 42 words). The drop is larger than for the 10,000-study adapter
  (BERTScore −0.040, CheXbert −0.083; greedy, 500 val, so not directly comparable).
  ⚠ The input is out of distribution: this shows the channel is read, not how much it contributes.
- **Together (descriptive):** |P − C| = 0.135 > |O − C| = 0.091 on CheXbert micro F1-14. Both
  channels carry clinical content for the final model. The two numbers measure different
  interventions (one destroys a channel, the other improves one), so they do not rank the channels.

Not established by this probe: what share of the full-run gain over v3 comes from the image channel
versus the indication and prior-findings context. One model, one decoding, val only. The 8/20 G0
mismatch stays unexplained. C's absolute values may differ slightly from the reported model's
generation of the same studies; the paired comparisons are unaffected.
