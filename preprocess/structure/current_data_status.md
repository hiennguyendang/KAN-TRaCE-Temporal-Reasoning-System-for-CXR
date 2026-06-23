## I. CHIẾN LỰC HỢP NHẤT DATASET
* Tạo không gian nhãn thành mảng 21 chiều (21-class Unified Label Space) để đồng nhất giữa chuẩn của Chexpert và NIH.
* Các nhãn Null, Uncertain (`-1`) và Các nhãn ngoại lại (nhãn khác tập NIH <> Chexpert) được gán thành `-100.0` thể hiện tín hiệu **Ignore**.

## II. HIỆN TRẠNG CÁC TẬP DỮ LIỆU

### 1. Tập MIMIC-CXR
* **Số lượng tổng:** `227,833` ảnh.
* **Phân loại theo Chất lượng Báo cáo:**
    * **Có FINDINGS:** `66,548` ảnh. 
    * **Không có FINDINGS:** `161,285` ảnh.
* **Phân loại theo Trạng thái Split:**
    * Train: `368,960` ảnh (Đang cần cân nhắc cắt bớt bù sang Val).
    * Valid: `2,991` ảnh. (Tỷ lệ quá thấp ~0.8%, không có ý nghĩa thống kê)
    * Test: `5,159` ảnh.


### 2. Tập ChestXplus
* **Số lượng tổng:** `112,120` ảnh.
* **Phân loại theo Chất lượng Báo cáo:**
    * **Có FINDINGS:** `49,861` ảnh. 
    * **Không có FINDINGS:** `141,101` ảnh.
* **Phân loại theo Trạng thái Split:** 
    * Train: `64,548` patient.
    * Valid: `149` patient (Ít đến vô lý, chỉ ~0.2%).

### 3. Tập NIH
* **Số lượng tổng:** `190,962` ảnh.
* **Phân loại theo Chất lượng Báo cáo:**
    * **Có FINDINGS:** `0` ảnh. 
    * **Không có FINDINGS:** `190,962` ảnh.
    * *Lưu ý: Để phục vụ cho ý định train bằng Contrastive Learning phần "Impression" được sinh tự động bằng rule-based "The chest X-ray shows evidence of [disease_1, disease_2, ...].".*
* **Phân loại theo Trạng thái Split:**
    * Train: `86,524` ảnh.
    * Test: `25,596` ảnh.

## III. VẤN ĐỀ & ĐỀ XUẤT

1. **Khó khăn trong việc Tách tập ChestXplus (Splitting):**
    * *Vấn đề:* Hiện tại không có tập Test cũng như số lượng của tập Valid quá lố bịch. Ngoài ra, để đánh giá đúng năng lực Text-guided, tập Val và Test BẮT BUỘC phải được bốc 100% từ nhóm có FINDINGS.
    * *Đề xuất:* Tự chia lại ChestXplus thành Train/Val/Test (tỷ lệ dự kiến 80/10/10), miễn là đảm bảo Patient-disjoint.
2. **Quá tải dữ liệu Yếu (Weak Data Overload):**
    * *Vấn đề:* 141,101 ảnh không FINDINGS cũng không có BBox của ChestXplus khi qua các mô hình sẽ sinh ra lượng BBox rác khổng lồ.
    * *Đề xuất:* Cân nhắc lấy mẫu ngẫu nhiên khoảng 50k ảnh từ cục 150k này để giảm tải cho GPU lúc Train KAN, rút ngắn thời gian hội tụ.
3. **Ý tưởng làm giàu dữ liệu (Data Augmentation):**
    * *Vấn đề:* Nhánh Text-Guide hiện tại bị lệch cực nặng vì chỉ có 49,861 ảnh có FINDINGS của CheXplus. Trong khi nhánh Strong có 227,833 ảnh từ MIMIC và nhánh Weak có tận 141,101 + 190,962 từ CheXplus và NIH.
    * *Đề xuất:* Trong số 227,833 ảnh của MIMIC, có tới **161,285 ảnh chứa phần "Findings"** chi tiết. Có thể cân nhắc overlap phần này sang Text-guided.

## IV. ĐỀ XUẤT CHIẾN LƯỢC: LOẠI BỎ NIH KHỎI TẬP HUẤN LUYỆN
**Lập luận bảo vệ đề xuất:**
1. **Đạt được sự đồng bộ về nhãn:** Tập NIH có 7 nhãn không hề tồn tại trong tập 14 nhãn của Chexpert, việc cố gắng nhồi nhét NIH vào pipeline khiến trường label phình to bất thường và phải xử lý một cách rất miễn cưỡng. Loại bỏ tập NIH có thể trả label về đúng bản chất của nó và tránh gây nhiễu không cần thiết cho mô hình.
2. **Lệch pha về Ngôn ngữ (Artificial vs. Natural):** Tập NIH hoàn toàn không có báo cáo text.  Việc dùng code rule-based để sinh một câu văn thô cứng sẽ làm vỡ không gian nhúng tự nhiên của Text Encoder (CXR-BERT). Mô hình đang được học văn phong bác sĩ thực tế cực xịn từ MIMIC/ChestXplus, nhét text giả vào sẽ làm giảm chất lượng biểu diễn ngôn ngữ.
3. **Dư thừa chức năng:** NIH thuộc nhóm "Giám sát yếu" (Chỉ có nhãn, không BBox). Vai trò này có thể thay thế bằng 150,000 ảnh ChestXplus không có phần FINDINGS. Việc nhét thêm NIH sẽ làm tỷ lệ tập Yếu phình to, gây quá tải bộ lọc nhiễu của KAN.
4. **Đạt được tỉ lệ vàng trong kiến trúc:** Loại bỏ NIH sẽ gần như tạo ra "Tỷ lệ Vàng" `1:1:1` cho kiến trúc đa nhánh: 
* **Nhánh Mạnh (Strong - Cung cấp BBox chuẩn):** Toàn bộ ~227,800 ảnh MIMIC.
* **Nhánh Text-Guided (Hướng dẫn không gian bằng Văn bản):** ~210,800 ảnh (Gồm 161k MIMIC + 49,8k ChestXplus có FINDINGS).
* **Nhánh Yếu (Bãi tập lọc nhiễu cho KAN):** ~207,600 ảnh (Gồm các ảnh khuyết Findings của MIMIC và ~150k ảnh Bạc của ChestXplus).