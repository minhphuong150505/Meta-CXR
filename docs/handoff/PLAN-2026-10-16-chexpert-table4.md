# Bảng 4 của bài báo — phân loại chéo miền trên CheXpert validation

## Mục tiêu

Chấm Stage 1 của mô hình đã báo cáo (`run_20261005_paper/phase1c`, `checkpoint_best.pth`),
chỉ train trên MIMIC-CXR, trên tập **CheXpert validation** (200 study / 234 ảnh, nhãn 0/1 do hội
đồng bác sĩ chẩn đoán hình ảnh đọc), đúng giao thức Bảng 4 của bài báo: xác suất 3 lớp →
`p = p1 / (p0 + p1)` (Eq. 21), AUC và F1 trên 5 bệnh. Bài báo: AUC trung bình **0,824**, F1
**0,699**.

Đây là phép đo, không phải lựa chọn mô hình: không có quy tắc nhận/không nhận, kết quả không
đổi mô hình báo cáo.

## Code (commit này)

- `model/lavis/data/external_eval.py` (chỉ pandas): đổi `valid.csv` của CheXpert thành hai bảng
  mà `MIMIC_CXR_Dataset` đã đọc (dòng ảnh + nhãn theo study). Từ chối: đường dẫn không nằm trong
  `valid/` (chặn nhầm `train.csv`, nhãn do labeler, có -1), nhãn khác 0/1, các view của cùng study
  mang nhãn khác nhau, thiếu ảnh dưới `vis_root`.
- `MIMIC_CXR_Dataset` dùng nó khi config có `datasets.mimic_cxr.external_eval`; không có khoá =
  MIMIC-CXR như cũ. Tiền xử lý ảnh, chọn view (frontal làm anchor, lateral làm aux) và collate là
  đúng đường mọi số MIMIC đã đi qua. CheXpert không có báo cáo → `target_valid=False` mọi dòng;
  ITC/ITM/LM trả 0 khi không có cặp hợp lệ (đã đọc code), chỉ bộ phân loại được chấm.
- Test CPU: `tests/test_external_eval_chexpert.py` (15 test).
- **Chưa chạy trên GPU, và chưa thử trên file `valid.csv` thật** — chỉ trên bảng giả có cùng cột.

## Điều kiện trước (user)

1. Đăng ký và tải **CheXpert-v1.0-small** từ Stanford AIMI (Research Use Agreement). Bài báo dùng
   bản small (code gốc đọc `/workspace/CheXpert-v1.0-small/valid.csv`).
2. Chỉ cần phần validation, đặt trên `/home` (ext4) — **không** ghi vào `/mnt/drive1tb` (volume
   NTFS đang dirty):

   ```bash
   mkdir -p ~/datasets/chexpert && cd ~/datasets/chexpert
   unzip <zip> 'CheXpert-v1.0-small/valid.csv' 'CheXpert-v1.0-small/valid/*'
   ls CheXpert-v1.0-small/valid | wc -l      # 200 thư mục patient
   ```

   CheXpert cũng là dữ liệu có thoả thuận sử dụng: không commit ảnh, `valid.csv` hay `.npz`.

## Lệnh (máy train)

```bash
cd ~/Meta-CXR && git fetch && git checkout feat/stage2-finding-tokens && git pull
PY=~/.venvs/meta-cxr-stage1-311/bin/python
CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/test_external_eval_chexpert.py -q   # 15 passed

# guards: card free, nothing else running, no stale *_provided.npz to overwrite
ps -eo pid,args --no-headers | grep -E "pretraining.train|run_medgemma_qlora|generate_stage2" | grep -v grep && exit 1
nvidia-smi --query-gpu=memory.used --format=csv,noheader          # < 1000 MiB
ROOT=~/run_20261005_paper
CK=$ROOT/phase1c/mimic_cxr_full_blip2/checkpoint_best.pth
find $ROOT -name "*_predictions_epoch_provided.npz" | grep . && exit 1   # evaluate-only writes IN PLACE
OUT=~/chexpert_table4_20261016; test ! -e $OUT || exit 1; mkdir -p $OUT
CX=~/datasets/chexpert

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled $PY -m pretraining.train \
  --cfg-path pretraining/configs/mimic_cxr_full.yaml \
  --options run.phase=phase1c run.phase_root=$ROOT \
    run.evaluate=True run.test_splits=[test] run.save_predictions=True \
    run.resume_ckpt_path=$CK run.output_dir=$OUT/eval \
    run.distributed=false run.world_size=1 \
    datasets.mimic_cxr.external_eval.kind=chexpert_val \
    datasets.mimic_cxr.external_eval.valid_csv=$CX/CheXpert-v1.0-small/valid.csv \
    datasets.mimic_cxr.external_eval.vis_root=$CX \
    datasets.mimic_cxr.external_eval.splits=[test] \
  > $OUT/eval.log 2>&1
echo "exit $?"
grep -c "Loading evaluation weights from" $OUT/eval.log      # phải = 1, nếu 0 thì kết quả vô nghĩa
grep "EXTERNAL eval set" $OUT/eval.log                       # 234 images
grep "\[test\].*studies from" $OUT/eval.log                  # 200 studies, ~ số có view bổ sung
find $ROOT $OUT -name "test_predictions_epoch_provided.npz" -mmin -60 -exec mv {} $OUT/ \;
ls $OUT/test_predictions_epoch_provided.npz

$PY scripts/evaluate_chexpert_crossdomain.py \
  --predictions $OUT/test_predictions_epoch_provided.npz --output $OUT/table4.json
```

## Kỳ vọng

- Suy luận vài phút. `missing=`/`unexpected=` khi nạp như các lần evaluate trước.
- 234 ảnh, 200 study; script in bảng 5 bệnh, cột AUC/F1 cạnh số bài báo, `n` mỗi bệnh = 200.
- Đọc lại: AUC và F1 từng bệnh, trung bình, `n`; số study có view bổ sung.

## Dừng và báo lại nếu

- `Loading evaluation weights from` không xuất hiện đúng 1 lần.
- `load_chexpert_val` từ chối file (cột, nhãn, đường dẫn, ảnh thiếu) — gửi nguyên thông báo lỗi,
  không sửa CSV bằng tay.
- `n` khác 200, hoặc nhãn trong `.npz` có giá trị ngoài 0/1.
- Bất kỳ lỗi nào trong forward: không vá config, báo lại dòng lỗi đầu và frame cuối.

## Giới hạn phải ghi cạnh kết quả

- **F1 dùng ngưỡng 0,5** trên `p1/(p0+p1)`; bài báo không nói ngưỡng. AUC không phụ thuộc ngưỡng.
- **MedCLIP-Swin được pretrain trên MIMIC-CXR + CheXpert** với split không kiểm chứng được — có thể
  đã thấy ảnh cùng nguồn; nêu là giới hạn.
- Ảnh bản small (~320 px cạnh ngắn) được phóng lên 448/224 — khác độ phân giải MIMIC-JPG; bài báo
  cũng dùng bản small.
- Nhãn CheXpert val là 0/1, không có Uncertain: không so trực tiếp với số 3 lớp trên MIMIC.
- Không phải giao thức headline (cutpoints): Bảng 4 là phép tính nhị phân của chính bài báo (D-023
  cho phép đúng hai phép này).
