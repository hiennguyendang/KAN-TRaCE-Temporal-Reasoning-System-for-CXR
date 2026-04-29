def count_lines(file_path):
    """Đếm số dòng trong một file txt."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {file_path}")
        return 0

def report_split_ratio(train_path, test_path, val_path):
    # Đếm số dòng
    train_count = count_lines(train_path)
    test_count = count_lines(test_path)
    val_count = count_lines(val_path)

    total = train_count + test_count + val_count
    
    if total == 0:
        print("Không có dữ liệu để tính toán.")
        return

    # Tính tỷ lệ phần trăm
    train_pct = (train_count / total) * 100
    test_pct = (test_count / total) * 100
    val_pct = (val_count / total) * 100

    # Xuất kết quả
    print(f"{'File':<15} | {'Số dòng':<10} | {'Tỷ lệ (%)'}")
    print("-" * 40)
    print(f"{'Train':<15} | {train_count:<10} | {train_pct:.2f}%")
    print(f"{'Test':<15} | {test_count:<10} | {test_pct:.2f}%")
    print(f"{'Validation':<15} | {val_count:<10} | {val_pct:.2f}%")
    print("-" * 40)
    print(f"{'Tổng cộng':<15} | {total:<10} | 100.00%")

# --- Cấu hình đường dẫn của bạn ---
path_train = 'C:\\Users\\dhint\\CHEX-DATA\\MIMIC-CXR\\ImaGenome\\silver_dataset\\splits\\train.csv'
path_test = 'C:\\Users\\dhint\\CHEX-DATA\\MIMIC-CXR\\ImaGenome\\silver_dataset\\splits\\test.csv'
path_val = 'C:\\Users\\dhint\\CHEX-DATA\\MIMIC-CXR\\ImaGenome\\silver_dataset\\splits\\valid.csv'

report_split_ratio(path_train, path_test, path_val)