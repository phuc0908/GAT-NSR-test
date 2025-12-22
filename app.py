from flask import Flask, render_template, request, jsonify
import torch
from dataset import FilmTrustDataset
from model import GATNSR
import os

app = Flask(__name__)

# --- Cấu hình ---
DATA_DIR = 'd:/HOCKY_6/ChuyenDe3_HeKhuyenNghi/code/filmtrust'
MODEL_PATH = 'gat_nsr_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Biến toàn cục ---
dataset = None
model = None

def load_system():
    global dataset, model
    
    # 1. Load Dataset (để lấy map ID và dữ liệu đồ thị)
    print("Đang tải dữ liệu...")
    dataset = FilmTrustDataset(DATA_DIR)
    
    # 2. Khởi tạo Model
    print("Đang khởi tạo Model...")
    model = GATNSR(dataset.num_users, dataset.num_items, feature_dim=32).to(DEVICE)
    
    # 3. Load Trọng số đã train
    if os.path.exists(MODEL_PATH):
        print("Đang load file trọng số đã train...")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            print("Load model thành công!")
        except Exception as e:
            print(f"\n[Lỗi] Không thể load model cũ do code đã thay đổi: {e}")
            print(">>> VUI LÒNG CHẠY LẠI 'python train.py' ĐỂ HUẤN LUYỆN LẠI MODEL MỚI! <<<\n")
            exit(1)
    else:
        print("CHÚ Ý: Chưa tìm thấy file 'gat_nsr_model.pth'. Hãy chạy train.py trước!")

# start app
load_system()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    user_id_input = request.json.get('user_id')
    
    # Chuyển đổi ID từ input (String/Int) sang ID nội bộ
    # Vì dataset dùng map, ta cần check xem user có tồn tại không
    # Trong dataset.py đơn giản lúc nãy ta dùng user_map nội bộ trong init.
    # Để map chuẩn, ta nên lưu user_map ra file hoặc load lại y hệt.
    # Ở đây load lại dataset sẽ tái tạo lại map y hệt (nếu thứ tự file không đổi).
    
    # Tìm ID nội bộ. Vì user_map string -> int.
    # Giả sử input la ID gốc trong file ratings.txt
    user_original_id = str(user_id_input)
    
    if user_original_id not in dataset.user_map:
        return jsonify({'error': 'User ID không tồn tại trong hệ thống!'}), 404
        
    internal_user_id = dataset.user_map[user_original_id]
    
    # --- Logic Dự Đoán ---
    # 1. Lấy tất cả Items
    all_items = torch.arange(dataset.num_items).to(DEVICE)
    # Lặp lại User ID cho khớp số lượng Items
    user_tensor = torch.tensor([internal_user_id]).repeat(dataset.num_items).to(DEVICE)
    
    # 2. Chuẩn bị dữ liệu đồ thị (Graph Data)
    social_adj = dataset.social_edge_index.to(DEVICE)
    interact_adj = dataset.interaction_edge_index.to(device=DEVICE)
    interact_ratings = dataset.interaction_ratings.to(DEVICE)
    
    # 3. Dự đoán
    with torch.no_grad():
        # Mô hình trả về điểm số cho cặp (User này, Tất cả Item)
        predictions = model(user_tensor, all_items, social_adj, interact_adj, interact_ratings)
    
    # 4. Lấy Top 10
    scores, top_indices = torch.topk(predictions, 10)
    
    # 5. Format kết quả trả về
    # Cần map ngược từ internal_id -> original_id cho item
    # Tạo map ngược
    inv_item_map = {v: k for k, v in dataset.item_map.items()}
    
    results = []
    for score, idx in zip(scores, top_indices):
        item_internal = idx.item()
        item_original = inv_item_map.get(item_internal, "Unknown")
        results.append({
            'item_id': item_original,
            'score': round(score.item(), 2)
        })
        
    return jsonify({
        'user_id': user_original_id,
        'recommendations': results
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
