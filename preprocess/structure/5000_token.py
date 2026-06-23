def extract_5000_words_raw(input_file, output_file, limit=5000):
    try:
        # Đọc file dưới dạng văn bản thuần túy (không parse JSON)
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Tách thành các từ dựa trên khoảng trắng
        words = content.split()
        
        # Lấy 5000 từ đầu tiên
        result_words = words[:limit]
        result_text = " ".join(result_words)
        
        # Lưu vào file .txt
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result_text)
            
        print(f"Hoàn thành! Đã trích xuất {len(result_words)} từ vào '{output_file}'.")
        
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file đầu vào.")
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")

# Chạy thử
extract_5000_words_raw('C:\\Users\\dhint\\CHEX-DATA\\CHEXPLUS\\radgraph-XL-annotations\\section_impression.json', 'output_2.txt')