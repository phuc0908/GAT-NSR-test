import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import FilmTrustDataset
from model import GATNSR
import time

def main():
    # 1. Chọn thiết bị (GPU nếu có, không thì CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang chạy trên: {device}")
    
    # 2. Chuẩn bị Dữ liệu
    print("Đang đọc dữ liệu...")
    data_dir = 'd:/HOCKY_6/ChuyenDe3_HeKhuyenNghi/code/filmtrust'
    dataset = FilmTrustDataset(data_dir)
    
    # Tạo loader để lấy từng lô (batch) dữ liệu khi train
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    # Chuyển dữ liệu đồ thị sang thiết bị (GPU/CPU)
    social_adj = dataset.social_edge_index.to(device)
    interact_adj = dataset.interaction_edge_index.to(device)
    interact_ratings = dataset.interaction_ratings.to(device)
    
    # 3. Khởi tạo Mô hình
    model = GATNSR(
        num_users=dataset.num_users, 
        num_items=dataset.num_items,
        feature_dim=32
    ).to(device)
    
    # Công cụ tối ưu hóa (Người sửa lỗi)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    # Hàm tính lỗi (MSE - Sai số bình phương trung bình)
    criterion = nn.MSELoss()
    
    print("Mô hình đã sẵn sàng!")
    
    # 4. Bắt đầu Huấn luyện
    epochs = 10 # Số lần học lặp lại
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()
        
        for u, i, r in train_loader:
            # Chuyển batch hiện tại sang thiết bị
            u, i, r = u.to(device), i.to(device), r.to(device)
            
            # Xóa sạch gradient cũ
            optimizer.zero_grad()
            
            # Mô hình dự đoán
            prediction = model(u, i, social_adj, interact_adj, interact_ratings)
            
            # Tính lỗi
            loss = criterion(prediction, r)
            
            # Tính toán sửa lỗi (Backpropagation)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # In kết quả sau mỗi vòng lặp
        avg_loss = total_loss / len(train_loader)
        print(f"Vòng {epoch+1}: Lỗi trung bình = {avg_loss:.4f} (Hết {time.time()-start_time:.2f} giây)")
        
    print("Huấn luyện xong!")
    torch.save(model.state_dict(), 'gat_nsr_model.pth')
    print("Đã lưu model vào gat_nsr_model.pth")

if __name__ == '__main__':
    main()
