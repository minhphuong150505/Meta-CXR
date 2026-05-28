# Hướng dẫn quản lý crontab (trên VM `meta-cxr-l4`)

Cron hiện tại đặt **trên VM**, không phải máy local. Lý do: VM luôn bật để train, nên cron chạy 24/7 kể cả khi máy bạn tắt.

---

## 1. SSH vào VM

```bash
gcloud compute ssh meta-cxr-l4 --zone=asia-southeast1-c
```

Tất cả lệnh bên dưới chạy **sau khi đã SSH vào VM**.

## 2. Xem crontab trên VM

```bash
crontab -l
```

Output hiện tại:
```
17 * * * * /home/phuong/check_and_reorder.sh >> /home/phuong/logs/check_and_reorder.log 2>&1
```

Đọc 5-field: `phút giờ ngày-tháng tháng ngày-tuần`
- `17 * * * *` = mỗi giờ vào phút :17
- `0 9 * * *` = mỗi ngày 9:00 sáng
- `*/5 * * * *` = mỗi 5 phút

## 3. Kiểm tra cron daemon đang chạy

```bash
systemctl is-active cron
```

Output `active` = OK.

## 4. Xem log của lần chạy

```bash
tail -50 ~/logs/check_and_reorder.log
```

Mỗi lượt :17 sẽ append vào file này. Tìm dòng `REORDER COMPLETE` → đã trigger relaunch xong.

## 5. Tắt cron

### Cách A — xoá nhanh (1 dòng, an toàn)
```bash
crontab -l | grep -v 'check_and_reorder.sh' | crontab -
crontab -l   # verify
```

### Cách B — sửa bằng editor
```bash
crontab -e
```
Vim: `dd` xoá dòng, `:wq` lưu.

### Cách C — xoá toàn bộ crontab
```bash
crontab -r
```
> Xoá hết, không hỏi lại.

## 6. Tạm tắt mà không xoá

```bash
crontab -e
```
Thêm `#` đầu dòng:
```
#17 * * * * /home/phuong/check_and_reorder.sh >> /home/phuong/logs/check_and_reorder.log 2>&1
```

## 7. Chạy thử script ngay (không đợi :17)

```bash
~/check_and_reorder.sh
```

## 8. Khi nào nên tắt cron?

- Sau khi thấy `REORDER COMPLETE` trong `~/logs/check_and_reorder.log` (đã trigger relaunch).
- Hoặc khi marker `~/logs/.reordered_done` đã tồn tại — cron vẫn fire nhưng không làm gì, có thể tắt cho gọn.
- Khi bạn huỷ kế hoạch reorder.

**Lệnh tắt nhanh (chạy từ máy local, không cần SSH thủ công):**
```bash
gcloud compute ssh meta-cxr-l4 --zone=asia-southeast1-c --command='crontab -l | grep -v check_and_reorder.sh | crontab -; crontab -l'
```

## 9. File liên quan trên VM

| Đường dẫn | Mục đích |
|---|---|
| `~/check_and_reorder.sh` | Script được crontab gọi |
| `~/logs/check_and_reorder.log` | Log mỗi lượt fire |
| `~/logs/master_run.log` | Log của 7 runs gốc (script đọc để detect trigger) |
| `~/logs/.reordered_done` | Marker — tồn tại = đã trigger 1 lần, sẽ skip lần sau |
| `~/run_reversed.sh` | Script reversed (07→06→05→04→03→02), được tạo lúc trigger |
| `~/logs/master_run_reversed.log` | Log của reversed runs (sau trigger) |

## 10. File local liên quan

| Đường dẫn | Mục đích |
|---|---|
| `META-CXR/cloud/check_and_reorder_on_vm.sh` | Source của script đã upload lên VM (giữ làm backup / chỉnh sửa) |
| `META-CXR/cloud/CRONTAB_GUIDE.md` | File này |

## 11. Lưu ý

- Cron trên VM dùng `PATH` hạn chế — script đã export PATH đầy đủ ở đầu file.
- VM phải bật (`RUNNING`) cron mới chạy. Nếu `gcloud compute instances stop` → cron tê liệt cho đến khi start lại.
- Sửa lại script: edit `META-CXR/cloud/check_and_reorder_on_vm.sh` local rồi:
  ```bash
  gcloud compute scp META-CXR/cloud/check_and_reorder_on_vm.sh meta-cxr-l4:~/check_and_reorder.sh --zone=asia-southeast1-c
  ```
