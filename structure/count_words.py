import json
import re
import os

def extract_values(obj):
    """Hàm đệ quy để lấy tất cả các chuỗi văn bản từ file JSON bất kể cấu trúc lồng nhau"""
    values = []
    if isinstance(obj, dict):
        for v in obj.values():
            values.extend(extract_values(v))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(extract_values(item))
    elif isinstance(obj, str):
        values.append(obj)
    return values

def count_compare_word_forms(file_path):
    # Danh sách các hình thái từ của "compare"
    word_forms = [
        "compare", "compares", "compared", "comparing", 
        "comparison", "comparisons", "comparability",
        "comparable", "comparative", "incomparable",
        "comparatively", "incomparably"
    ]
    
    if not os.path.exists(file_path):
        return f"Lỗi: Không tìm thấy file tại đường dẫn: {file_path}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Lấy toàn bộ văn bản từ các giá trị trong JSON và chuyển về chữ thường
        all_text = " ".join(extract_values(data)).lower()
        
        results = {}
        for word in word_forms:
            # Sử dụng \b để đảm bảo tìm đúng từ, không bị dính vào từ khác
            pattern = rf'\b{re.escape(word)}\b'
            count = len(re.findall(pattern, all_text))
            results[word] = count
            
        return results

    except json.JSONDecodeError:
        return "Lỗi: File không phải định dạng JSON hợp lệ hoặc file bị trống."
    except Exception as e:
        return f"Lỗi không xác định: {e}"

# --- THIẾT LẬP ĐƯỜNG DẪN ---
file_name = r'C:\Users\dhint\CHEX-DATA\CHEXPLUS\radgraph-XL-annotations\section_findings.json'

# --- THỰC THI ---
print(f"Đang phân tích file: {file_name}...")
stats = count_compare_word_forms(file_name)

if isinstance(stats, dict):
    print("\nKết quả thống kê:")
    found_any = False
    for word, count in stats.items():
        if count > 0:
            print(f" - {word}: {count}")
            found_any = True
    
    if not found_any:
        print("=> Không tìm thấy bất kỳ hình thái nào của từ 'compare' trong file này.")
else:
    # In ra thông báo lỗi nếu có
    print(stats)