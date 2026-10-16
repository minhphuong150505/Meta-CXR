---
name: thesis-docs
description: Tạo tài liệu tiếng Việt trình bày kiểu luận văn (.docx và .pdf) từ một file nội dung thuần văn bản, có chèn số liệu từ JSON và kiểm tra kết luận khớp số liệu. Load khi người dùng cần báo cáo, tài liệu đánh giá thực nghiệm, chương luận văn, tài liệu trình thầy hướng dẫn, hoặc muốn dựng lại tài liệu sau khi đổi nội dung hay số liệu. Không ký tự đặc biệt, không icon, không bookmark.
---

# Tài liệu kiểu luận văn từ một file nội dung

Mọi tài liệu dùng chung một trình dựng; chỉ file nội dung `.md` (và file số liệu `.json`
nếu có) thay đổi giữa các tài liệu. Prompt gốc của người dùng và các prompt mẫu để
tái sử dụng nằm ở `PROMPT.md`.

| File | Vai trò |
|---|---|
| `scripts/render_doc.py` | Dựng `.docx` (và `.pdf` với `--pdf`) từ file nội dung. Cú pháp đầy đủ ở docstring đầu file |
| `scripts/fmt.py` | Định dạng trang, tiêu đề, bảng, tài liệu tham khảo, số kiểu Việt Nam |
| `templates/mau_tai_lieu.md` | File nội dung mẫu dùng mọi cú pháp; sao chép để bắt đầu tài liệu mới |
| `templates/so_lieu_mau.json` | Số liệu minh họa cho file mẫu, không phải kết quả thật |
| `PROMPT.md` | Prompt gốc nguyên văn, prompt tổng quát cho tài liệu đầy đủ, prompt ngắn |

## Môi trường

```bash
# một lần; python-docx không kéo theo torch, không cài vào ~/venv
python3 -m venv ~/.venvs/thesis-docs && ~/.venvs/thesis-docs/bin/pip install python-docx
```

PDF dùng LibreOffice (`libreoffice`, bản snap) và `pdftoppm` để soát trang. Bản snap
không đọc được `/tmp`; trình dựng tự chuyển đổi trong `$HOME` nên đầu ra có thể nằm ở đâu cũng được.

## Quy trình

1. **Thu số liệu trước, viết sau.** Tính lại từ artifact thật và ghi vào một file JSON
   tổng hợp. Không viết số từ trí nhớ hay từ `CLAUDE.md` mà chưa đối chiếu: bản thống kê
   dữ liệu ngày 2026-09-14 tìm ra ba con số trong tài liệu đã lấy từ một manifest cũ.
   Số của bài báo khác phải kiểm chứng từ nguồn; không kiểm chứng được thì ghi rõ trong văn bản.
2. **Sao chép `templates/mau_tai_lieu.md`** thành file nội dung mới, đặt cạnh file số liệu.
3. **Viết nội dung**, chèn số bằng `{{duong.dan|f4}}` thay vì gõ tay, để khi số liệu đổi
   chỉ cần dựng lại.
4. **Mỗi câu kết luận có một `::check`.** Ví dụ viết "tăng có ý nghĩa thống kê" thì kiểm
   tra cận dưới khoảng tin cậy lớn hơn 0. Kiểm tra sai thì không file nào được ghi ra.
5. **Dựng:**
   `~/.venvs/thesis-docs/bin/python .claude/skills/thesis-docs/scripts/render_doc.py noi_dung.md dau_ra.docx --pdf`
6. **Soát PDF bằng mắt** (`pdftoppm -r 70 -png -f 1 -l 5 dau_ra.pdf trang`, rồi đọc ảnh):
   tiêu đề cột bị ngắt giữa từ, bảng tràn lề, tiêu đề mồ côi cuối trang. Sửa rồi dựng lại.
7. **Giao file** vào thư mục bị git bỏ qua (`Eval/<ten>_<ngay>/`), kèm file nội dung và
   file số liệu để dựng lại được.

## Quy tắc trình bày

Trình dựng áp dụng sẵn: A4, Times New Roman 13, giãn dòng 1,5, lề trên dưới phải 2 cm và
trái 2,5 cm, số trang góc phải dưới, tiêu đề đánh số `1.`, `1.1.`, `1.1.1.`, chú thích bảng
đặt trên bảng, tài liệu tham khảo kiểu IEEE thụt treo, không chèn mục lục tự động (trường
TOC sinh bookmark ẩn). Người viết nội dung chịu trách nhiệm phần còn lại:

- Văn phong học thuật, đoạn văn liền mạch. Không gạch đầu dòng, không icon, không mũi tên,
  không gạch ngang dài; trình dựng từ chối mọi ký tự từ U+2000 trở lên và `*`, `#`, `` ` ``
  còn sót. Liệt kê bằng "Thứ nhất, ... Thứ hai, ..." hoặc "Bước 1, ...".
- Dấu phẩy thập phân, dấu chấm hàng nghìn: dùng định dạng `f4`, `pct1`, `n` thay vì tự gõ.
  Khoảng tin cậy viết `[+0,0136; +0,0263]`.
- Bảng rộng tối đa 16,5 cm tổng `::widths`. Từ 6 cột trở lên tiêu đề tự thu về 8,5 pt;
  tiêu đề vẫn bị ngắt giữa từ thì chuyển bảng sang dạng dài (thêm cột "Phần" hoặc "Tập",
  mỗi tổ hợp một dòng) thay vì thêm cột.
- Mọi số kèm n. Mọi so sánh có khoảng tin cậy thì ghi rõ phương pháp (bootstrap ghép cặp,
  số lần lặp, hạt giống). Khoảng tin cậy có cận cách 0 dưới 0,0005 là "chưa xác lập",
  không phải có ý nghĩa.
- Chỉ đặt cạnh nhau các kết quả cùng bộ dữ liệu, cùng phân chia, cùng cách tính. Khác giao
  thức thì đưa vào bảng riêng và nói rõ không so trực tiếp được.
- Không có độ đo thì ghi "không có", không ghi 0.

## Bảo mật dữ liệu

MIMIC-CXR thuộc DUA của PhysioNet. File nội dung, file số liệu và tài liệu chỉ được chứa
số tổng hợp: không `subject_id`, `study_id`, `dicom_id`, đường dẫn ảnh thật, văn bản báo cáo
hay đầu ra sinh của mô hình. Trước khi giao, quét tài liệu:

```bash
~/.venvs/thesis-docs/bin/python -c "import docx,re,sys; d=docx.Document(sys.argv[1]); \
t=' '.join(p.text for p in d.paragraphs)+' '.join(c.text for tb in d.tables for r in tb.rows for c in r.cells); \
print(re.findall(r'\b[ps]\d{7,8}\b', t)[:5])" dau_ra.docx      # phải in []
```

Đầu ra để trong `Eval/` (git bỏ qua); không commit tài liệu chứa số liệu dẫn xuất.

## Tham chiếu

Tài liệu đánh giá đầy đủ ngày 2026-09-13/14 (33 trang, hai giai đoạn, cắt bỏ, so sánh
công trình liên quan) được dựng bằng một trình dựng riêng viết thẳng Python, trước khi có
skill này: `Eval/thesis_eval_20260913/generator/` (`build.py`, `ch1_data.py`, `fmt.py`,
`datastats.py`, `verified_refs.md`) cùng số liệu ở `Eval/thesis_eval_20260913/data/`. Chỉ có
trên máy phát triển, không có trong git. Dùng nó làm mẫu khi tài liệu cần logic phức tạp hơn
cú pháp của file nội dung, ví dụ bảng sinh theo vòng lặp từ nhiều file kết quả.
