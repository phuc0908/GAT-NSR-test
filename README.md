# GAT-NSR: Movie Recommendation System

Hệ thống gợi ý phim sử dụng mô hình Graph Attention Network (GAT-NSR), kết hợp giữa hành vi người dùng và mạng lưới xã hội.

## Cấu trúc dự án
*   `train.py`: Huấn luyện mô hình từ dữ liệu FilmTrust.
*   `app.py`: Web demo sử dụng Flask.
*   `model.py` & `layers.py`: Kiến trúc GAT và các tầng xử lý.
*   `dataset.py`: Tiền xử lý dữ liệu.

## Hướng dẫn sử dụng

### 1. Cài đặt
Cài đặt PyTorch và Flask:
```bash
pip install torch numpy flask
```

### 2. Huấn luyện
Chạy lệnh sau để tạo file trọng số `gat_nsr_model.pth`:
```bash
python train.py
```

### 3. Khởi chạy Web Demo
Khởi động server:
```bash
python app.py
```
Truy cập: http://127.0.0.1:5000

---
Dự án được xây dựng nhằm mục đích tìm hiểu về Graph Neural Networks trong hệ thống gợi ý.




