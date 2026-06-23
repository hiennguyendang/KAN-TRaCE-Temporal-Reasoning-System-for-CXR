import json

def count_empty_vs_non_empty(file_path):
    """
    Đếm số lượng chuỗi rỗng và khác rỗng trong trường 'findings' của file .jsonl
    """
    empty_count = 0
    non_empty_count = 0
    missing_field_count = 0
    total_lines = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    total_lines += 1
                    
                    # Lấy giá trị trường 'findings'
                    # Nếu không có trường này, mặc định trả về None
                    value = data.get('findings')

                    if value is None:
                        missing_field_count += 1
                    elif isinstance(value, str):
                        if value.strip() == "":
                            empty_count += 1
                        else:
                            non_empty_count += 1
                    else:
                        # Trường hợp findings không phải là string (vd: số, null, dict...)
                        # Tùy yêu cầu bạn có thể đếm vào nhóm khác rỗng hoặc bỏ qua
                        non_empty_count += 1 

                except json.JSONDecodeError:
                    continue

        # In kết quả
        print(f"📊 Kết quả kiểm tra trường 'findings':")
        print("-" * 45)
        print(f"⚪ Chuỗi rỗng (Empty):      {empty_count:>8}")
        print(f"⚫ Chuỗi có dữ liệu:       {non_empty_count:>8}")
        print(f"❓ Không có trường này:    {missing_field_count:>8}")
        print("-" * 45)
        print(f"📝 Tổng số dòng hợp lệ:    {total_lines:>8}")

        if total_lines > 0:
            print(f"\n📈 Tỉ lệ lấp đầy dữ liệu: {(non_empty_count/total_lines)*100:.2f}%")

    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file.")

# --- Thay đường dẫn file của bạn ở đây ---
path_jsonl = 'C:\\Users\\dhint\\CHEX-DATA\\MIMIC-CXR\\metadata\\mimic_metadata_unified_14dim.jsonl'
count_empty_vs_non_empty(path_jsonl)