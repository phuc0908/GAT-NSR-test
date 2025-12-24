# Hệ Thống Gợi Ý Phim GAT-NSR (Graph Attention Network)

Đây là hệ thống gợi ý phim thông minh sử dụng Deep Learning (GAT-NSR) để phân tích sở thích người dùng dựa trên:
1.  **Lịch sử xem phim**: Người dùng đã xem và chấm điểm phim nào.
2.  **Mạng xã hội**: Người dùng tin tưởng ai (bạn bè), có gu giống ai.

## Cấu Trúc Thư Mục

- **`train.py`**: File dùng để dạy (huấn luyện) AI. Chạy file này đầu tiên để tạo ra "bộ não" (`gat_nsr_model.pth`).
- **`app.py`**: Server chạy trang web. Nó dùng "bộ não" đã học để gợi ý phim cho người dùng.
- **`model.py`**: Chứa kiến trúc mạng neuron GAT-NSR (Code xử lý chính).
- **`layers.py`**: Các công thức toán học (Graph Attention) để tính toán sự tương đồng giữa người dùng.
- **`dataset.py`**: Đọc dữ liệu từ thư mục `filmtrust/` và chuyển thành dạng số.
- **`templates/index.html`**: Giao diện trang web.

---

## Hướng Dẫn Cài Đặt & Chạy (Workflow)

Để hệ thống hoạt động, bạn cần làm theo đúng 3 bước sau:

### Bước 1: Cài đặt thư viện
Mở Terminal (Ctrl + `) và chạy lệnh sau để cài các công cụ cần thiết:

```bash
pip install torch numpy flask
```

### Bước 2: Huấn Luyện AI ("Dạy học")
Trước khi gợi ý, AI cần học từ dữ liệu cũ. Chạy lệnh:

```bash
python train.py
```

*   **Kết quả**: Bạn sẽ thấy `Loss` giảm dần (từ 1.2 xuống 0.5...). Khi chạy xong, nó sẽ tạo ra file `gat_nsr_model.pth`.
*   **Lưu ý**: Nếu bạn sửa code model, bạn PHẢI chạy lại bước này.

### Bước 3: Chạy Web App (Demo)
Sau khi đã có file model, bạn chạy lệnh sau để mở trang web:

```bash
python app.py
```

*   Mở trình duyệt và truy cập: `http://127.0.0.1:5000`
*   Nhập ID người dùng (ví dụ: `1`, `50`, `100`...) và xem kết quả gợi ý.

---

## Nguyên Lý Hoạt Động (Giải thích đơn giản)

1.  **Thu Thập (Dataset)**: Hệ thống đọc danh sách "Ai xem phim gì" và "Ai chơi với ai".
2.  **Quan Sát (GAT Layer)**:
    *   Khi AI nhìn vào bạn (User A), nó sẽ nhìn sang bạn bè của bạn (User B, User C).
    *   Nếu bạn tin tưởng User B nhiều, AI sẽ hiểu "Gu của A chắc giống B".
    *   Đồng thời, AI nhìn vào các phim bạn đã xem. Nếu bạn xem nhiều phim hành động, nó sẽ hiểu bạn thích hành động.
3.  **Hợp Nhất (Fusion)**:
    *   Vector Xã hội (Từ bạn bè) + Vector Sở thích (Từ phim) = **Latent Vector User A**.
4.  **Dự Đoán (Prediction)**:
    *   AI lấy **Latent Vector User A** so sánh với **Latent Vector Item X** (phim chưa xem).
    *   Nếu thấy khớp, nó chấm điểm cao => Gợi ý cho bạn.


