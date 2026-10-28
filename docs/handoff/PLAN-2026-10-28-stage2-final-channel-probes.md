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
