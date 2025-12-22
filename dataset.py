import torch
from torch.utils.data import Dataset
import os

class FilmTrustDataset(Dataset):
    """
    Dataset đơn giản để load FilmTrust.
    Chỉ làm nhiệm vụ: Đọc file -> Tạo ID -> Chuyển thành Tensor.
    """
    def __init__(self, data_dir):
        self.ratings_file = os.path.join(data_dir, 'ratings.txt')
        self.trust_file = os.path.join(data_dir, 'trust.txt')
        
        # Dictionary để ánh xạ ID gốc (ví dụ "10A") sang ID số (1, 2, 3...)
        self.user_map = {}
        self.item_map = {}
        
        # 1. Đọc file Ratings (Tương tác)
        # Lưu lại danh sách (u, i, r) để huấn luyện
        self.interactions = []
        user_list, item_list, rating_list = [], [], []
        
        if os.path.exists(self.ratings_file):
            with open(self.ratings_file, 'r') as f:
                for line in f:
                    # File format: user_id item_id rating
                    parts = line.split() 
                    if len(parts) < 3: continue
                    
                    u_id = self.get_id(parts[0], self.user_map)
                    i_id = self.get_id(parts[1], self.item_map)
                    r = float(parts[2])
                    
                    self.interactions.append((u_id, i_id, r))
                    user_list.append(u_id)
                    item_list.append(i_id)
                    rating_list.append(r)
        
        # Tạo Tensor "Danh sách cạnh tương tác" [2, Số_cạnh] cho GAT
        self.interaction_edge_index = torch.tensor([user_list, item_list], dtype=torch.long)
        self.interaction_ratings = torch.tensor(rating_list, dtype=torch.float)

        # 2. Đọc file Trust (Xã hội)
        trustor_list, trustee_list = [], []
        if os.path.exists(self.trust_file):
            with open(self.trust_file, 'r') as f:
                for line in f:
                    # File format: user1 user2 trus
                    parts = line.split()
                    if len(parts) < 2: continue
                    
                    # Chỉ lấy user đã có trong user_map (để tránh lỗi thừa node)
                    if parts[0] in self.user_map and parts[1] in self.user_map:
                        u1 = self.user_map[parts[0]]
                        u2 = self.user_map[parts[1]]
                        trustor_list.append(u1)
                        trustee_list.append(u2)
        
        # Tạo Tensor "Danh sách cạnh xã hội" [2, Số_cạnh] cho GAT
        self.social_edge_index = torch.tensor([trustor_list, trustee_list], dtype=torch.long)

        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map)
        
        print(f"Dataset đã tải: {self.num_users} Users, {self.num_items} Items.")

    def get_id(self, original_id, mapper):
        if original_id not in mapper:
            mapper[original_id] = len(mapper)
        return mapper[original_id]

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        # Trả về 1 mẫu dữ liệu: (user, item, rating)
        u, i, r = self.interactions[idx]
        return torch.tensor(u), torch.tensor(i), torch.tensor(r, dtype=torch.float)
