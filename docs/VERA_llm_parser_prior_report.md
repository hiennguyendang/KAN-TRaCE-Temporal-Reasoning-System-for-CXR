# VERA — LLM Parser & hướng dùng Prior-Report để nâng accuracy M4 (nhánh B11-B)

> File riêng cho phần LLM parser. Tập trung vào **hướng đã chọn: dùng report của ảnh prior (khi có lúc
> launch) như một tín hiệu sạch ở đầu-prior của M4 để NÂNG ACCURACY** — không phải dùng làm cờ hedge.
> Lưu ý quan trọng: chọn hướng accuracy ⇒ đây **không phải mẹo inference**, mà là **một nhánh có phần
> train riêng** (B11-B trong `VERA_methodology_concerns.md`). v1 lõi vẫn thuần ảnh; nhánh này là mở rộng.

---

## 1. Ba vai của LLM parser — và vai nào khiến nó sống

Parser (report → scene-graph: bệnh nào ở vùng nào) có thể đảm nhận tới ba việc, **rủi ro rất khác nhau**:

| Vai | Khi nào | Tính chất | Trạng thái |
|-----|---------|-----------|-----------|
| **(1) Sinh weak GT cho CheXplus** | Lúc train | Nhãn yếu, rủi ro nhiễm (B9) | Tuỳ ablation CheXplus; **nhiều khả năng bỏ** |
| **(2) Sinh nhãn cho MIMIC** | Lúc train | — | **KHÔNG cần** (ImaGenome 240k đã phủ trọn 210k MIMIC train) |
| **(3) Parse report của *prior* lúc launch** | Lúc inference | **GT thật, verify được** (report bệnh nhân) | **Đây là vai khiến parser sống độc lập với CheXplus** |

Kết luận đổi so với trước: parser **không chết theo CheXplus**. Kể cả khi ablation loại CheXplus, parser
vẫn có lý do tồn tại nhờ **vai (3)** — một việc làm ở *inference*, không phụ thuộc dữ liệu train ngoài.

---

## 2. Ý tưởng cốt lõi

Lúc launch, tình huống thật là **bất đối xứng**: ảnh current chỉ có ảnh; ảnh prior thường **kèm report**
(đã được bác sĩ đọc ở lần khám cũ). Report prior là **ground-truth mạnh, verify được** về trạng thái
*prior* → parse ra scene-graph-prior → nạp vào **đầu-prior của M4** để M4 so sánh với current chính xác hơn.

**Hai ràng buộc bất biến, không được vi phạm:**

- **Chỉ chạm đầu-prior.** Report prior nói về *prior*, KHÔNG nói gì về *current*. Trong contract
  `412 = 128×3 + 14×2`, nó chỉ được phép thay/chỉnh phần **`disease_logit_prior[14]`**. Đầu current
  (`disease_logit_curr`) **luôn từ M3** (launch không bao giờ có report current). Phép so sánh tiến triển
  vẫn do M4 làm.
- **Chỉ chạm kênh 14-logit, KHÔNG chạm kênh 128-feature.** Scene-graph cho "bệnh ở vùng" → ánh xạ sang
  *disease logit*, không sang feature 128 chiều. Giữ nguyên `feat_prior[128]` từ ảnh prior (BioViL-T);
  chỉ nâng cấp kênh symbolic 14-logit. Như vậy đường Siamese/feature không bị động vào.

---

## 3. Vì sao đây KHÔNG phải mẹo inference — rào cản thật nằm ở train

Rào cản mà bạn cảm thấy ("cái logit") nghe như vấn đề định dạng, nhưng định dạng là phần **rẻ** (xem §4).
Rào cản **thật** sâu hơn: **M4 chưa từng học cách dùng một đầu-prior sạch.**

Một M4 train hoàn toàn với prior-từ-M3 (nhiễu) có thể đã ngầm học **bù trừ** cho cái nhiễu đó (vd "prior
M3 thường under-confident nên tự cộng thêm"). Nạp một prior **sạch** vào model quen-bù-nhiễu → nó **bù
nhầm** → có thể *tệ hơn*, dù input "tốt hơn". Đây là lý do không thể "thay lúc launch" một cách ngây thơ.

➡️ Hệ quả: muốn prior-report sạch *thật sự* kéo accuracy, **M4 phải từng thấy đầu-prior-sạch lúc train**.
Đó là việc train, không phải inference. Chọn hướng accuracy = cam kết phần train này.

---

## 4. Cơ chế TRAIN (phần làm cho nó nâng-accuracy)

**Modality dropout ở đầu-prior.** Lúc train, với một tỉ lệ mẫu, **thay** `disease_logit_prior` (từ M3
trên ảnh prior) bằng **tín hiệu prior sạch** lấy từ scene-graph (lúc train có thể dùng nhãn ImaGenome /
scene-graph làm proxy cho "prior sạch"). M4 học được chế độ: *"khi đầu-prior sạch thì tin nó nhiều hơn."*

- **Cờ `prior_clean_available`** nối vào input M4 (1 bit) để model biết mình đang ở chế độ nào — sạch hơn
  là để model tự đoán. (Có thể train không cờ kiểu dropout thuần, nhưng cờ tường minh dễ kiểm soát + trung thực hơn.)
- **Tỉ lệ dropout** phản ánh thực tế launch (bao nhiêu % ca có prior-report) — đừng đặt 100% (sẽ quên chế
  độ thuần-ảnh) hay 0% (vô hiệu).

**Định dạng tín hiệu prior sạch (calibrated soft pseudo-logit).** KHÔNG nạp nhãn cứng 0/1. Chuyển
present/absent → **logit mềm giả trong cùng thang calibrate của M3**, map vào vùng **cao-nhưng-không-bão-hoà
(≈0.85–0.9)**, KHÔNG 0.95–0.99. Lý do: tiến triển nằm ở **biên độ** thay đổi; ghim prior ở trần làm M4 khó
đọc "worsened" (tăng thêm từ một prior đã kịch trần). Chừa biên độ là bắt buộc.

---

## 5. Cơ chế INFERENCE (lúc launch)

- **Đầu current:** luôn từ M3. Không ngoại lệ.
- **Đầu prior:**
  - *Có report prior* → parser → scene-graph-prior → soft pseudo-logit (§4) → nạp vào đầu-prior; bật cờ.
    Tuỳ chọn **fusion thay vì thay thẳng**: `prior_final = w·prior_scenegraph + (1−w)·prior_M3`, với **w
    theo từng finding** (cao ở finding parser parse tốt, thấp ở finding parser hay sót) — để khi parser
    bỏ sót một finding mà M3 bắt được thì không mất tín hiệu.
  - *Không report prior* → đầu-prior từ M3; tắt cờ. Đây là chế độ thuần-ảnh, trùng v1.

---

## 6. Bẫy & điều kiện (đọc kỹ trước khi cam kết)

1. **Train/inference phải khớp chế độ.** Nếu train không có chế độ prior-sạch (§4) thì §5 sẽ làm M4 hành
   xử lạ. Đây là điều kiện *tiên quyết*, không phải tuỳ chọn.
2. **Độ tin parser theo từng finding.** Parser sai khác nhau theo loại finding → đo accuracy per-finding
   của parser; dùng nó đặt `w` (§5) và quyết finding nào được phép tin từ scene-graph.
3. **Finding không nhìn được trên ảnh.** Report có thể ghi điều suy từ lâm sàng/bệnh sử, không nhìn được
   trên ảnh đơn. Với **đầu-prior** thì vẫn tạm chấp nhận được (report *là* ground-truth trạng thái prior),
   nhưng phải ý thức: nếu một tiến triển dựa trên prior-chỉ-biết-từ-report, câu đó kém "verify-từ-ảnh" hơn
   — cân nhắc dán nhãn/độ tin khác ở M5.
4. **Eval cả hai chế độ + ablation bắt buộc.** Claim "nâng accuracy" phải đo so với baseline **không**
   prior-report. Báo cáo: accuracy M4 (có vs không prior-report), và độ phủ thực tế (% ca có prior-report).
5. **Đây là nhánh B11-B, KHÔNG phải v1 lõi.** v1 vẫn thuần ảnh, một câu chuyện không dấu hoa thị. Nhánh
   này là **section/đóng góp riêng** với cờ + ablation. Đừng trộn nó vào đường chính khiến v1 mất tính
   đối xứng "mọi claim từ ảnh".
6. **Bất đối xứng là tính năng, không phải lỗi.** Đừng cố làm current cũng "có report" cho cân — current
   không bao giờ có report lúc launch; chính sự bất đối xứng này là cái bài mô tả.

---

## 7. Vì sao hướng này hợp luận điểm VERA (chứ không phản bội nó)

Khác bẫy privileged-information (B12) và OsteoGA (counterfactual GAN): ở đây tín hiệu thêm vào là **report
thật của bệnh nhân** — verify được, không phải model bịa. Nó **không** dạy M4 "biết nhiều hơn cái nhìn
thấy" theo kiểu hallucinate; nó thay một **ước lượng nhiễu (M3 đọc ảnh prior)** bằng một **quan sát thật
đã được bác sĩ ghi (report prior)**. Đó là *nâng cấp nguồn*, không phải *bịa nguồn*. Miễn giữ §6.3 (cẩn
thận với finding không-nhìn-được) thì nhánh này nhất quán với "mọi claim verify được".

---

## 8. Việc cần làm / đo
- [ ] **Đo accuracy parser per-finding** (report→scene-graph) trên subset có nhãn → đặt `w`, chọn finding tin được.
- [ ] **Thêm chế độ modality-dropout đầu-prior** vào train M4 (+ cờ `prior_clean_available`).
- [ ] **Map present/absent → soft pseudo-logit** ở thang M3, vùng 0.85–0.9 (chừa biên độ).
- [ ] **Ablation**: M4 accuracy {không prior-report} vs {có, thay thẳng} vs {có, fusion theo finding}; báo % độ phủ.
- [ ] **Quyết w**: hằng số vs theo-finding (theo §6.2).
- [ ] Giữ nhánh này **tách khỏi v1** trong cấu trúc bài (section riêng / future-work-đã-làm).

## 9. Liên hệ các file khác
- `VERA_methodology_concerns.md`: nhánh này = **B11-B**; ràng buộc thuần-ảnh của v1 = **B11-A**; bẫy
  privileged-info = **B12**; nguyên tắc "train trên thứ có lúc inference" = **A1**.
- Số phận **CheXplus** (vai parser số 1) vẫn do ablation B10 quyết — **độc lập** với vai số 3 ở file này.
- Input 14-logit của M4 (đầu *current*) vẫn từ **M3 logit mềm**, không từ scene-graph (xem thảo luận M4).
