# PCB Defect Detection - OpenCV, không dùng AI

Ứng dụng desktop phát hiện vùng khác biệt giữa ảnh PCB tham chiếu và ảnh PCB kiểm tra. Phương pháp gồm căn chỉnh ECC, Gaussian Blur, Absolute Difference, Thresholding, Canny, Morphological Processing và Contour.

## 1. Cài đặt

Python 3.10 trở lên được khuyến nghị.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Chạy giao diện

```bash
python main.py
```

Chọn ảnh mẫu (`*_temp.jpg`) và ảnh kiểm tra (`*_test.jpg`), điều chỉnh ngưỡng và diện tích tối thiểu, sau đó nhấn Phân tích ảnh.

## 3. Đánh giá DeepPCB

Tải dữ liệu từ https://github.com/tangsanli5201/DeepPCB rồi chạy:

```bash
python -m evaluation.batch_evaluate --dataset "DUONG_DAN_DATASET"
python -m evaluation.compare_methods --dataset "DUONG_DAN_DATASET"
python -m evaluation.generate_report --csv outputs/reports/batch_results.csv
```

## 4. Kiểm thử

```bash
python -m unittest discover -s tests -v
```

## 5. Giới hạn phương pháp

Đây là baseline xử lý ảnh cổ điển, không có mô hình AI. Chất lượng phát hiện phụ thuộc mạnh vào độ căn chỉnh, ánh sáng, ngưỡng và chất lượng ảnh. Bộ phân loại tên lỗi trong giao diện chỉ là heuristic theo hình học/độ tương phản; không phải phân loại đã được huấn luyện hoặc xác nhận. Các nhãn DeepPCB gồm open, short, mousebite, spur, copper và pin-hole; chúng không tương ứng một-một với các mô tả nứt/xước/lỗ/sai màu trong mọi ảnh PCB thực tế.

Diện tích hiển thị là pixel². Chỉ quy đổi sang mm² khi có hệ số pixel/mm đã hiệu chuẩn cho thiết bị chụp.
