import os

def count_subfolders(path):
    if not os.path.exists(path):
        print(f"❌ Đường dẫn không tồn tại: {path}")
        return

    # Cách 1: Chỉ đếm các folder con nằm ngay bên trong (Cấp 1)
    # os.listdir liệt kê tất cả, os.path.isdir để lọc ra folder
    immediate_subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    # Xuất kết quả
    print(f"📁 Thư mục kiểm tra: {path}")
    print("-" * 30)
    print(f"🔹 Số folder con trực tiếp (Cấp 1): {len(immediate_subfolders)}")

# --- Thay đường dẫn của bạn vào đây ---
target_path = 'C:\\Users\\dhint\\CHEX-DATA\\CHEXPLUS\\processed\\images'

count_subfolders(target_path)