import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SocialRecDataset
from model import GATNSR
import time

def main():
    # 1. Thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang chạy trên: {device}")
    
    # 2. Load và Split Dữ liệu (80-10-10)
    print("Đang đọc dữ liệu...")
    dataset_name = 'filmtrust'
    data_dir = f'd:/HOCKY_6/ChuyenDe3_HeKhuyenNghi/code/{dataset_name}'
    dataset = SocialRecDataset(data_dir)
    
    # SPLIT: 80% train, 10% val, 10% test
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    
    train_data, val_data, test_data = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Tổng: {total} | Train: {train_size} | Val: {val_size} | Test: {test_size}")
    
    # Loaders
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    
    # Graph data
    social_adj = dataset.social_edge_index.to(device)
    interact_adj = dataset.interaction_edge_index.to(device)
    interact_ratings = dataset.interaction_ratings.to(device)
    
    # 3. Model
    model = GATNSR(dataset.num_users, dataset.num_items, feature_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Mô hình sẵn sàng!")
    
    # Hàm đánh giá
    def evaluate(loader):
        model.eval()
        total_mse, total_mae = 0, 0
        with torch.no_grad():
            for u, i, r in loader:
                u, i, r = u.to(device), i.to(device), r.to(device)
                pred = model(u, i, social_adj, interact_adj, interact_ratings)
                total_mse += criterion(pred, r).item()
                total_mae += torch.abs(pred - r).mean().item()
        
        mse = total_mse / len(loader)
        mae = total_mae / len(loader)
        return mse, mae, mse**0.5
    
    # 4. Training
    best_val = float('inf')
    for epoch in range(10):
        model.train()
        train_mse, train_mae = 0, 0
        start = time.time()
        
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            optimizer.zero_grad()
            pred = model(u, i, social_adj, interact_adj, interact_ratings)
            loss = criterion(pred, r)
            loss.backward()
            optimizer.step()
            
            train_mse += loss.item()
            with torch.no_grad():
                train_mae += torch.abs(pred - r).mean().item()
        
        t_mse = train_mse / len(train_loader)
        t_mae = train_mae / len(train_loader)
        v_mse, v_mae, v_rmse = evaluate(val_loader)
        
        print(f"E{epoch+1}: Train MSE={t_mse:.4f} MAE={t_mae:.4f} | Val MSE={v_mse:.4f} MAE={v_mae:.4f} ({time.time()-start:.1f}s)")
        
        if v_mse < best_val:
            best_val = v_mse
            torch.save(model.state_dict(), 'gat_nsr_best.pth')
            print("  ✓ Best")
    
    # 5. Test
    print("\n=== TEST SET ===")
    model.load_state_dict(torch.load('gat_nsr_best.pth'))
    test_mse, test_mae, test_rmse = evaluate(test_loader)
    print(f"MSE={test_mse:.4f} MAE={test_mae:.4f} RMSE={test_rmse:.4f}")
    
    torch.save(model.state_dict(), 'gat_nsr_model2.pth')

if __name__ == '__main__':
    main()
