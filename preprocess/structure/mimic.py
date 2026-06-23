import pandas as pd

def analyze_csv_splits(file_path, column_name=None):
    """
    Đếm số lượng và tính tỷ lệ các nhãn train, validation, test trong file CSV.
    
    :param file_path: Đường dẫn tới file .csv
    :param column_name: Tên cột chứa nhãn (nếu để None, code sẽ tìm trong toàn bộ file)
    """
    try:
        # Đọc file CSV
        df = pd.read_csv(file_path)
        
        # Nếu không chỉ định cột, ta sẽ làm phẳng toàn bộ dữ liệu thành một chuỗi các giá trị
        if column_name:
            data_series = df[column_name].astype(str).str.lower()
        else:
            # Tìm trong tất cả các cột
            data_series = df.astype(str).values.flatten()
            data_series = pd.Series(data_series).str.lower()

        # Đếm số lượng
        counts = data_series.value_counts()
        
        target_labels = ['train', 'validate', 'test']
        results = {}
        total_target_count = 0

        # Lấy số lượng cụ thể cho từng nhãn
        for label in target_labels:
            count = counts.get(label, 0)
            results[label] = count
            total_target_count += count

        # Xuất kết quả
        print(f"{'Nhãn':<12} | {'Số lượng':<10} | {'Tỷ lệ (%)'}")
        print("-" * 35)
        
        if total_target_count > 0:
            for label in target_labels:
                count = results[label]
                percentage = (count / total_target_count) * 100
                print(f"{label:<12} | {count:<10} | {percentage:.2f}%")
        else:
            print("Không tìm thấy các từ khóa 'train', 'validate' hoặc 'test' trong file.")

    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file tại đường dẫn đã cung cấp.")
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")

# --- Thay đổi đường dẫn file và tên cột của bạn tại đây ---
duong_dan_file = 'C:\\Users\\dhint\\CHEX-DATA\\MIMIC-CXR\\mimic-cxr-2.0.0-split.csv' 
ten_cot = 'split'  # Thay bằng tên cột chứa nhãn, hoặc để None nếu muốn quét toàn bộ file

analyze_csv_splits(duong_dan_file, ten_cot)