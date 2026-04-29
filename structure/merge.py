import json

def merge_jsonl_files(input_paths, output_path):
    """
    Gộp nhiều file .jsonl thành một file duy nhất.
    
    :param input_paths: Danh sách các đường dẫn file nguồn (List)
    :param output_path: Đường dẫn file đích (String)
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for file_path in input_paths:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        # Kiểm tra dòng trống trước khi ghi
                        if line.strip():
                            outfile.write(line.strip() + '\n')
        print(f"✅ Đã gộp thành công vào: {output_path}")
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")

# --- Cấu hình đường dẫn của bạn ---
path_A = 'C:\\Users\\dhint\\CHEX-DATA\\MIMIC-CXR\\metadata\\mimic_metadata_unified_14dim.jsonl'
path_B = 'C:\\Users\\dhint\\CHEX-DATA\\CHEXPLUS\\metadata\\chexplus_metadata_unified_14dim.jsonl'
path_D = './data/train/train_metadata_unified_14dim.jsonl'  # Đường dẫn file đích sau khi gộp

# Thực thi
merge_jsonl_files([path_A, path_B], path_D)