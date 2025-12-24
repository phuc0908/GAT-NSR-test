import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Lớp GAT (Attention): Tính trọng số cho hàng xóm.
    Hỗ trợ cả đồ thị 2 phía (Bipartite: Item -> User) và 1 phía (Social: User -> User).
    """
    def __init__(self, feature_dim):
        super(GraphAttentionLayer, self).__init__()
        # W: Biến đổi đặc trưng Nguồn (Source)
        self.W_src = nn.Linear(feature_dim, feature_dim, bias=False)
        
        # W: Biến đổi đặc trưng Đích (Target)
        self.W_dst = nn.Linear(feature_dim, feature_dim, bias=False)
        
        # W_rating: Biến đổi đặc trưng Rating
        self.W_rating = nn.Linear(feature_dim, feature_dim, bias=False)
        
        # a: Vector tính điểm Attention
        # [Nguồn || Đích || Rating]
        self.a = nn.Linear(3 * feature_dim, 1, bias=False)
        
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, source_vecs, edge_index, target_vecs=None, rating_vecs=None):
        """
        source_vecs: Vector của bên gửi (ví dụ: Item)
        edge_index: [Nguồn, Đích]
        target_vecs: Vector của bên nhận (ví dụ: User). Nếu None thì coi như Nguồn=Đích (Social).
        rating_vecs: Vector điểm số (nếu có)
        """
        if target_vecs is None:
            target_vecs = source_vecs
            
        N_target = target_vecs.size(0)
        
        # 1. Chiếu vector (Projection)
        h_src_all = self.W_src(source_vecs)
        h_dst_all = self.W_dst(target_vecs)
        
        # 2. Lấy vector tại các đầu cạnh
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        
        h_src = h_src_all[src_idx]
        h_dst = h_dst_all[dst_idx]
        
        # 3. Tính Attention Score
        if rating_vecs is not None:
            h_rating = self.W_rating(rating_vecs)
            
            # Ghép: [Nguồn || Đích || Rating]
            a_input = torch.cat([h_src, h_dst, h_rating], dim=1)
            # Thông tin truyền đi (Message)
            message = h_src + h_rating
        else:
            zeros = torch.zeros_like(h_src)
            a_input = torch.cat([h_src, h_dst, zeros], dim=1)
            message = h_src

        # Hàm f(.) Tính điểm Attention (2)(4)(7)
        scores = self.leakyrelu(self.a(a_input))
        attention = torch.exp(scores)
        
        # 4. Tổng hợp (Aggregation) (5)(9)
        weighted_message = message * attention
        
        # Tạo vector kết quả với kích thước bằng số nút ĐÍCH (User)
        h_new = torch.zeros(N_target, h_src.size(1)).to(source_vecs.device)
        
        # Cộng dồn vào đích
        h_new.scatter_add_(0, dst_idx.unsqueeze(1).expand(-1, h_new.size(1)), weighted_message)
        
        # Chuẩn hóa (Softmax partition) (3)(8)
        sum_weight = torch.zeros(N_target, 1).to(source_vecs.device)
        sum_weight.scatter_add_(0, dst_idx.unsqueeze(1), attention)
        
        h_new = h_new / (sum_weight + 1e-8)
        
        return F.elu(h_new)
