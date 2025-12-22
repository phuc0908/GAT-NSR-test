import torch
import torch.nn as nn
from layers import GraphAttentionLayer

class GATNSR(nn.Module):
    """
    Mô hình GAT-NSR phiên bản Đơn Giản & Dễ Hiểu.
    """
    def __init__(self, num_users, num_items, feature_dim=32):
        super(GATNSR, self).__init__()
        
        # 1. Tạo các vector đại diện (Embedding)
        self.user_embedding = nn.Embedding(num_users, feature_dim)
        self.item_embedding = nn.Embedding(num_items, feature_dim)
        self.rating_embedding = nn.Embedding(10, feature_dim) # Điểm số 0..9 (tượng trưng 0.5..5.0)

        # 2. Các lớp Attention (GAT)
        # Lớp nhìn vào bạn bè (Social)
        self.social_gat = GraphAttentionLayer(feature_dim)
        
        # Lớp nhìn vào items đã xem (User Interaction)
        self.user_interact_gat = GraphAttentionLayer(feature_dim)
        
        # Lớp nhìn vào người đã xem item (Item Interaction)
        self.item_interact_gat = GraphAttentionLayer(feature_dim)
        
        # 3. Các lớp Hợp nhất (Fusion) - Dùng Linear đơn giản
        self.user_fusion = nn.Linear(2 * feature_dim, feature_dim)
        self.item_fusion = nn.Linear(2 * feature_dim, feature_dim)
        
        # 4. Bộ dự đoán (Prediction MLP)
        self.predict_layer = nn.Sequential(
            nn.Linear(2 * feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_ids, item_ids, social_adj, interact_adj, interact_ratings):
        """
        Luồng tính toán (Forward Pass)
        """
        # Bước 1: Lấy toàn bộ Vector
        device = user_ids.device
        all_u_ids = torch.arange(self.user_embedding.num_embeddings).to(device)
        all_i_ids = torch.arange(self.item_embedding.num_embeddings).to(device)
        
        u_vectors = self.user_embedding(all_u_ids)
        i_vectors = self.item_embedding(all_i_ids)
        
        # Rating
        r_indices = (interact_ratings * 2 - 1).long().clamp(0, 9)
        r_vectors = self.rating_embedding(r_indices)
        
        # Bước 2: Học Vector User (User Modeling)
        
        # A. Nhìn từ góc độ Xã hội
        # Input: Nguồn=User, Target=User (mặc định nếu để None)
        user_social_vector = self.social_gat(u_vectors, social_adj, target_vecs=u_vectors, rating_vecs=None)
        
        # B. Nhìn từ góc độ Sở thích
        # Cạnh: Item -> User. Đảo chiều
        item_to_user_adj = torch.stack([interact_adj[1], interact_adj[0]])
        # Input: Nguồn=Item, Target=User.
        # QUAN TRỌNG: Phải truyền u_vectors vào target_vecs để GAT biết kích thước đầu ra là User
        user_history_vector = self.user_interact_gat(
            source_vecs=i_vectors, 
            edge_index=item_to_user_adj, 
            target_vecs=u_vectors, 
            rating_vecs=r_vectors
        )
        
        # C. Tổng hợp
        # Giờ cả 2 vector đều có kích thước [Num_Users, Dim]. Ghép vô tư.
        u_cat = torch.cat([user_social_vector, user_history_vector], dim=1)
        final_user_vector = torch.relu(self.user_fusion(u_cat))
        
        
        # Bước 3: Học Vector Item (Item Modeling)
        
        # A. Nhìn xem ai đã xem Item này
        # Input: Nguồn=User, Target=Item.
        # QUAN TRỌNG: Truyền i_vectors vào target.
        item_history_vector = self.item_interact_gat(
            source_vecs=u_vectors, 
            edge_index=interact_adj, 
            target_vecs=i_vectors, 
            rating_vecs=r_vectors
        )
        
        # B. Tổng hợp
        i_cat = torch.cat([i_vectors, item_history_vector], dim=1)
        final_item_vector = torch.relu(self.item_fusion(i_cat))
        
        
        # Bước 4: Dự đoán
        batch_u = final_user_vector[user_ids]
        batch_i = final_item_vector[item_ids]
        
        cat_prediction = torch.cat([batch_u, batch_i], dim=1)
        score = self.predict_layer(cat_prediction)
        
        return score.view(-1)
