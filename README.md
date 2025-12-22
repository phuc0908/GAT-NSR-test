# Hệ Thống Gợi Ý Phim GAT-NSR (Graph Attention Network)

Đây là hệ thống gợi ý phim thông minh sử dụng Deep Learning (GAT-NSR) để phân tích sở thích người dùng dựa trên:
1.  **Lịch sử xem phim**: Người dùng đã xem và chấm điểm phim nào.
2.  **Mạng xã hội**: Người dùng tin tưởng ai (bạn bè), có gu giống ai.

## 📂 Cấu Trúc Thư Mục

- **`train.py`**: File dùng để dạy (huấn luyện) AI. Chạy file này đầu tiên để tạo ra "bộ não" (`gat_nsr_model.pth`).
- **`app.py`**: Server chạy trang web. Nó dùng "bộ não" đã học để gợi ý phim cho người dùng.
- **`model.py`**: Chứa kiến trúc mạng neuron GAT-NSR (Code xử lý chính).
- **`layers.py`**: Các công thức toán học (Graph Attention) để tính toán sự tương đồng giữa người dùng.
- **`dataset.py`**: Đọc dữ liệu từ thư mục `filmtrust/` và chuyển thành dạng số.
- **`templates/index.html`**: Giao diện trang web.

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy (Workflow)

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

## 🧠 Nguyên Lý Hoạt Động (Giải thích đơn giản)

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

### ❓ "Latent Vector" là gì?
Trong máy tính, "Latent Vector" là một **dãy số** (ví dụ: `[0.9, 0.1, ... 0.5]`).
*   Hãy tưởng tượng mỗi con số đại diện cho một tính cách ngầm:
    *   Số đầu tiên: Độ thích phim Hành động (0.9 = Rất thích).
    *   Số thứ hai: Độ thích phim Tình cảm (0.1 = Không thích).
*   **Vector Xã hội**: Là dãy số đúc kết từ gu của bạn bè bạn.
*   **Vector Sở thích**: Là dãy số đúc kết từ các phim bạn đã xem.
=> Gộp lại ta được "Latent Vector" toàn diện của bạn dưới dạng số học.

### 🧮 Latent Vector được tính toán như thế nào?

Quá trình tính ra vector này gồm 3 bước (như trong file `model.py`):

1.  **Bước 1: Khởi tạo (Embedding)**
    *   Mỗi User và Item ban đầu được gán một vector ngẫu nhiên.
2.  **Bước 2: Lắng nghe (Attention)**
    *   **User Vector** = (0.7 x Vector Bạn thân) + (0.3 x Vector Bạn xã giao) ...
    *   Đồng thời cộng thêm thông tin các phim đã xem + điểm số đã chấm.
3.  **Bước 3: Tổng hợp (Fusion)**
    *   `Vector Cuối Cùng = Kết hợp [Vector Xã Hội + Vector Sở Thích]`
    *   Máy tính dùng hàm toán học (Linear + ReLU) để nén thông tin này lại thành một vector gọn gàng nhất.


---

## 🛠 Xử Lý Lỗi Thường Gặp

**1. Lỗi "No module named 'torch'"**
> Bạn chưa cài thư viện. Hãy chạy lại Bước 1.

**2. Lỗi "size mismatch" hoặc "Error loading state_dict"**
> Code mô hình đã thay đổi nhưng bạn đang dùng file save cũ.
> **Khắc phục**: Xóa file `gat_nsr_model.pth` đi và chạy lại Bước 2 (`python train.py`).

**3. Web không hiện gợi ý nào?**
> Có thể User ID bạn nhập không tồn tại trong tập dữ liệu. Hãy thử số nhỏ (1, 2, 3).
