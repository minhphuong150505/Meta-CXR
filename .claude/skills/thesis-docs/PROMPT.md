# Prompt tạo tài liệu kiểu luận văn

## 1. Prompt gốc của người dùng (2026-09-13, giữ nguyên văn)

> hiện tại tôi cần một file docs có toàn bộ tính toán evaluation của pipeline của tôi trên 14 nhãn trên nhánh chính không phải train mới gần đây vì nó không cải thiện gì bạn hãy cho tôi câu prompt để yêu cầu claude làm cho tôi một file đó dựa vào docs có trên repository trên github hiện tại, tôi cần nó trình theo dạng luận văn không kí tự đặc biệt không icon, không bookmark. chia ra 2 stage với các eval classification và các bài toàn nlp thì họ dùng gì. Ngoài ra tôi cần nó so với các bài toán hiện dại để cho có thể so sánh với pipeline của tôi so với họ và dùng đúng với các họ tính giống tôi, nhớ ghi tài liệu tham khảo vào, eval này chỉ tính trên bộ dữ liệu mimic-cxr-jpg không phải mimic-cxr không phải cùng bộ dữ liệu mới có thể so sánh được, tôi cần bạn tính lại vài thứ trước khi đưa prompt cho ai, nếu bạn có thể tạo được thì bạn hãy làm cho tôi luôn, ví dụ tôi cần một số table để so sánh những cải tiến của tôi so với bài báo gốc thì nó có cải thiện gì, và cả bật tắt cái loss để chứng minh là khi có những cái loss đó thì cái eval nó tăng như nào. Tôi cần một kết quả thực nghiệm đầy đủ nhất, để có thể trình bày với thầy hướng dẫn nên bạn hãy làm giống với bài báo thực tế trình bày không kí tự tham khảo các thụt đầu dòng và tham khảo thêm các skill làm docs tiếng việt về luận văn trên mạng

Yêu cầu bổ sung sau đó (2026-09-14): thêm phần thống kê tiền xử lý dữ liệu, cách chia
train, val, test, số P, N, U của từng nhãn, và thống kê dữ liệu đầu vào.

## 2. Prompt tổng quát, dùng cho tài liệu đầy đủ

Thay các chỗ trong ngoặc vuông rồi gửi nguyên văn.

```text
Dùng skill thesis-docs. Tạo tài liệu tiếng Việt trình bày theo kiểu luận văn về [CHỦ ĐỀ, ví dụ: toàn bộ
đánh giá thực nghiệm của pipeline trên nhánh main].

Phạm vi:
- Chỉ dùng [NHÁNH hoặc COMMIT] và [CHECKPOINT/MÔ HÌNH CHÍNH]; các thử nghiệm [LIỆT KÊ] chỉ đưa vào như
  nghiên cứu cắt bỏ, không thay mô hình chính.
- Chỉ dùng bộ dữ liệu [TÊN BỘ DỮ LIỆU và PHIÊN BẢN]. Chỉ so sánh với công trình dùng cùng bộ dữ liệu,
  cùng phân chia và cùng cách tính độ đo; công trình khác giao thức phải ghi rõ là không so trực tiếp được.

Cấu trúc: [DANH SÁCH CHƯƠNG, ví dụ: 1 Dữ liệu và tiền xử lý, 2 Giai đoạn 1 phân loại,
3 Giai đoạn 2 sinh báo cáo, 4 So sánh với bài báo gốc, 5 Thảo luận và hạn chế], cuối cùng là tài liệu
tham khảo theo chuẩn IEEE.

Nội dung bắt buộc:
- Thống kê dữ liệu đầu vào và quy trình tiền xử lý: số mẫu mỗi tập, cách chia, phân bố lớp của từng nhãn.
- Với mỗi bài toán, nêu độ đo mà các công trình trong lĩnh vực dùng và cách tính, rồi báo cáo đúng các độ đo đó.
- Bảng so sánh các cải tiến so với [BÀI BÁO GỐC].
- Bảng bật tắt từng thành phần [DANH SÁCH LOSS hoặc MÔ-ĐUN] để chứng minh đóng góp của từng thành phần,
  kèm khoảng tin cậy bootstrap ghép cặp và kết luận có ý nghĩa thống kê hay không.
- Bảng đặt kết quả cạnh các công trình hiện đại [DANH SÁCH], ghi rõ giao thức của từng công trình.

Quy tắc:
- Tính lại mọi số liệu từ artifact thật trước khi viết; không lấy số từ trí nhớ. Số liệu trích từ bài báo
  phải kiểm chứng từ nguồn và ghi tài liệu tham khảo.
- Mỗi câu kết luận phải có một dòng ::check tương ứng trong file nội dung.
- Không ký tự đặc biệt, không icon, không bookmark, không gạch đầu dòng trong văn bản; dùng dấu phẩy thập phân.
- Không đưa dữ liệu bệnh nhân, định danh hay văn bản báo cáo vào tài liệu.
- Xuất .docx và .pdf vào [THƯ MỤC ĐẦU RA, mặc định Eval/<tên>_<ngày>/], soát PDF trước khi báo xong.
```

## 3. Prompt ngắn, dùng cho tài liệu ngắn

```text
Dùng skill thesis-docs. Tạo tài liệu ngắn khoảng [SỐ TRANG] trang về [CHỦ ĐỀ], dành cho [NGƯỜI ĐỌC].
Số liệu lấy từ [ĐƯỜNG DẪN FILE KẾT QUẢ hoặc "tính lại từ ..."]. Cần các mục: [DANH SÁCH MỤC].
Cần các bảng: [DANH SÁCH BẢNG]. Tài liệu tham khảo: [CÓ hoặc KHÔNG].
Xuất vào [THƯ MỤC ĐẦU RA].
```

## 4. Khi đã có file nội dung

Chỉ sửa file nội dung (và file số liệu nếu có), rồi dựng lại:

```bash
~/.venvs/thesis-docs/bin/python .claude/skills/thesis-docs/scripts/render_doc.py \
    <noi_dung>.md <dau_ra>.docx [--data <so_lieu>.json] --pdf
```
