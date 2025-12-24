import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SocialRecDataset
from model import GATNSR
import time

def main():
    # 1. Chọn thiết bị (GPU nếu có, không thì CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang chạy trên: {device}")
    
    # 2. Chuẩn bị Dữ liệu
    print("Đang đọc dữ liệu...")
    dataset_name = 'filmtrust'  # filmtrust OR epinions
    data_dir = f'd:/HOCKY_6/ChuyenDe3_HeKhuyenNghi/code/{dataset_name}'
    dataset = SocialRecDataset(data_dir)
    
    print(f"Tổng số mẫu: {len(dataset)}")
    
    # Tạo loader - Paper: batch_size = 128
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Chuyển dữ liệu đồ thị sang thiết bị (GPU/CPU)  
    social_adj = dataset.social_edge_index.to(device)
    interact_adj = dataset.interaction_edge_index.to(device)
    interact_ratings = dataset.interaction_ratings.to(device)
    
    # 3. Khởi tạo Mô hình
    model = GATNSR(
        num_users=dataset.num_users, 
        num_items=dataset.num_items,
        feature_dim=64
    ).to(device)
    
    # Công cụ tối ưu hóa
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Hàm tính lỗi (MSE)
    criterion = nn.MSELoss()
    
    print("Mô hình đã sẵn sàng!")
    
    # 4. Bắt đầu Huấn luyện
    epochs = 5 # Số lần học lặp lại
    model.train()
    
    for epoch in range(epochs):
        total_mse = 0
        total_mae = 0
        start_time = time.time()
        
        for u, i, r in train_loader:
            # Chuyển batch hiện tại sang thiết bị
            u, i, r = u.to(device), i.to(device), r.to(device)
            
            # Xóa sạch gradient cũ
            optimizer.zero_grad()
            
            # Mô hình dự đoán
            prediction = model(u, i, social_adj, interact_adj, interact_ratings)
            
            # Tính lỗi (MSE)
            mse_loss = criterion(prediction, r)
            
            # MAE
            with torch.no_grad():
                mae_loss = torch.mean(torch.abs(prediction - r))
            
            # Tính toán sửa lỗi (Backpropagation)
            mse_loss.backward()
            optimizer.step()
            
            total_mse += mse_loss.item()
            total_mae += mae_loss.item()
        
        # Tính toán các chỉ số trung bình
        avg_mse = total_mse / len(train_loader)
        avg_mae = total_mae / len(train_loader)
        avg_rmse = torch.sqrt(torch.tensor(avg_mse)).item()
        
        # In kết quả
        print(f"Vòng {epoch+1}: MSE = {avg_mse:.4f} | MAE = {avg_mae:.4f} | RMSE = {avg_rmse:.4f} ({time.time()-start_time:.2f}s)")
        
    print("-" * 30)
    print(f"Huấn luyện hoàn tất!")
    print(f"Chỉ số cuối cùng:")
    print(f" - MSE:  {avg_mse:.4f}")
    print(f" - MAE:  {avg_mae:.4f}")
    print(f" - RMSE: {avg_rmse:.4f}")
    torch.save(model.state_dict(), 'gat_nsr_model.pth')
    # torch.save(model.state_dict(), 'gat_nsr_model_epinions.pth')
    print("Đã lưu model vào gat_nsr_model.pth")

if __name__ == '__main__':
    main()

# adj = viết tắt của Adjacency (Kề nhau, Liền kề)
