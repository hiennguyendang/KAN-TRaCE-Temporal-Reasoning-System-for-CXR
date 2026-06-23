def extract_first_50_lines(input_path, output_path, num_lines=50):
    """
    Đọc n dòng đầu tiên của file nguồn và ghi vào file txt.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                for i, line in enumerate(infile):
                    if i >= num_lines:
                        break
                    outfile.write(line)
        
        print(f"✅ Đã trích xuất {num_lines} dòng đầu tiên vào: {output_path}")
        
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file nguồn.")
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")

# --- Cấu hình đường dẫn của bạn ---
file_json_nguon = 'C:\\Users\\dhint\\CHEX-DATA\\MIMIC-CXR\\ImaGenome\\silver_dataset\\study_level_attribute_rdfgraphs.json'
file_txt_dich = 'ket_qua_50_dong.txt'

# Thực thi
extract_first_50_lines(file_json_nguon, file_txt_dich, 300)