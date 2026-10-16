---
title: Báo cáo kết quả thực nghiệm
subtitle: Tài liệu mẫu minh họa cú pháp, số liệu trong file này là số minh họa
chapter_new_page: true
data: so_lieu_mau.json
---
<!-- Chú thích HTML như dòng này bị bỏ qua khi dựng tài liệu. -->

::heading Tóm tắt
Đoạn tóm tắt một đến hai câu. Các dòng viết liền nhau
được nối thành một đoạn văn; một dòng trống sẽ ngắt sang đoạn mới.

# Thiết lập thực nghiệm

## Dữ liệu
Tập kiểm tra gồm {{n_test|n}} nghiên cứu. Số liệu được chèn từ file dữ liệu
bằng cú pháp hai dấu ngoặc nhọn, nên khi số liệu thay đổi chỉ cần dựng lại.

### Tiêu chí đánh giá
Mô tả độ đo, cách tính, và vì sao chọn độ đo đó.

# Kết quả

## Kết quả chính
::table So sánh mô hình gốc và mô hình đề xuất trên tập kiểm tra
::widths 6, 3.5, 3.5
::font 10.5
| Mô hình | AUROC macro | F1 macro |
|---|---|---|
| Mô hình gốc | {{baseline.macro_auroc|f4}} | {{baseline.f1|f4}} |
| Mô hình đề xuất | **{{proposed.macro_auroc|f4}}** | **{{proposed.f1|f4}}** |
::note Giá trị in đậm là kết quả tốt hơn có ý nghĩa thống kê.

Mô hình đề xuất tăng AUROC macro {{delta.macro_auroc|sf4}}, khoảng tin cậy 95%
[{{delta.lo|sf4}}; {{delta.hi|sf4}}], tương đương {{delta.macro_auroc / baseline.macro_auroc|pct1}} so với mô hình gốc.
::check D["delta"]["lo"] > 0 | Văn bản nói mức tăng có ý nghĩa nhưng khoảng tin cậy chạm 0
::check D["proposed"]["macro_auroc"] > D["baseline"]["macro_auroc"] | Mô hình đề xuất không tốt hơn

::refs
A. Johnson et al., "MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs," arXiv:1901.07042, 2019.
J. Irvin et al., "CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison," in Proc. AAAI, 2019.
