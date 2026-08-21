# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Nguyễn Minh Hiếu |
| MSSV | 2A202601816 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/endermagician15/Track2-K4-Day21-2A202601816-NguyenMinhHieu |
| Ngày nộp | 22/08/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.6687 | 0.8540 |
| 2 | 50 | 0.05 | 2 | 0.6122 | 0.8420 |
| 3 | 200 | 0.1 | 5 | 0.6954 | 0.8620 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ siêu tham số ở Lần chạy 3 mang lại giá trị `f1_score` cao nhất (0.6954), vượt qua ngưỡng kiểm định chất lượng 0.65 của hệ thống. Thực nghiệm cho thấy thuật toán GradientBoosting có sự phụ thuộc chặt chẽ giữa độ sâu cây và số lượng estimators: khi giảm cả hai tham số (Lần 2), mô hình bị underfitting khiến F1 tụt xuống 0.6122 dù accuracy vẫn đạt 84.2%. Việc tăng số cây lên 200 kết hợp max_depth 5 giúp mô hình nắm bắt tối ưu các mối quan hệ phi tuyến phức tạp giữa học vấn, chức vụ và số giờ làm việc.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult Income có sự mất cân bằng lớp rõ rệt khi số lượng mẫu thuộc lớp thu nhập cao (>50K USD) chỉ chiếm khoảng 24.8%. Trong điều kiện phân phối lệch như vậy, một mô hình phân loại suy biến luôn dự đoán nhãn "thu nhập thấp" (lớp 0) vẫn sẽ đạt mức Accuracy rất cao là 75.2%, nhưng F1-score của lớp dương sẽ bằng 0 tuyệt đối do không nhận diện được bất kỳ trường hợp thu nhập cao nào.

F1-score đóng vai trò là trung bình điều hòa giữa Precision và Recall, phản ánh chính xác hiệu năng dự đoán trên lớp thiểu số mà không bị lớp đa số chi phối. Chúng tôi tính toán trực tiếp `f1_score(y_eval, preds)` cho lớp dương và không sử dụng `average="weighted"` hay `average="macro"` nhằm ngăn việc lớp đa số kéo ảo điểm số, bảo đảm Quality Gate chặn đứng mọi mô hình suy biến trước khi release ra production.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| DVC trên GitHub Actions không đọc được Secret | Secret được lưu trong Environment `Lab21` thay vì Repository Secrets | Thêm cấu hình `environment: Lab21` vào toàn bộ các jobs trong `cicd.yml` để GitHub Runner cấp quyền truy cập |
| Service `income-api` trên EC2 bị lỗi `ModuleNotFoundError: No medium named 'fastapi'` | Ubuntu phiên bản mới bật cơ chế PEP 668 chặn pip cài đặt ra hệ thống | Cài đặt bằng `sudo pip3 install ... --break-system-packages` và cập nhật thông tin xác thực AWS vào systemd |
| Pipeline Bước 3 bị lỗi nếu push sai thứ tự | Lệnh `git push` bị chạy trước lệnh `dvc push` | Tuân thủ nghiêm ngặt quy trình: chạy `dvc push` đồng bộ dữ liệu lên S3 trước rồi mới thực hiện `git push` |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1` - 22.361 mẫu) | 0.6954 | 0.8620 |
| Bước 3 (thêm `train_batch2` - 44.722 mẫu) | 0.6912 | 0.8640 |

**Nhận xét:** Khi bổ sung thêm 22.361 mẫu dữ liệu ở Bước 3, F1-score dao động nhẹ (giảm 0.0042) trong khi Accuracy tăng nhẹ lên 86.4%. Nguyên nhân là do hai tập dữ liệu được chia ngẫu nhiên từ cùng một nguồn nên có cùng phân phối xác suất, do đó việc tăng gấp đôi dữ liệu chủ yếu củng cố mật độ lớp đa số; tuy nhiên toàn bộ pipeline đã tự động hóa 100% việc huấn luyện lại và tái triển khai thành công.

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

- [x] Bonus 1 - Tracking MLflow từ xa với DagsHub: Tích hợp cấu hình DagsHub client remote tracking server.
- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: Quét ngưỡng từ 0.1 đến 0.9, xác định threshold tối ưu nâng F1 của mô hình.
- [x] Bonus 3 - Báo cáo precision / recall tự động: Tạo và upload artifact `outputs/detail.txt` chứa Confusion Matrix và chi tiết từng lớp.
- [x] Bonus 4 - Hoàn trả về phiên bản trước: Tích hợp logic so sánh F1 với model metadata lưu trên S3 trong Job Quality Gate.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: Tự động kiểm tra tỷ lệ class balance của dữ liệu train trước khi bắt đầu fit mô hình.
