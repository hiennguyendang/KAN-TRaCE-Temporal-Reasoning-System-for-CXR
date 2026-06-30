# VERA — Kiến trúc & Lộ trình Hiện thực

**VERA: Verifiable, Evidence-grounded Regional Assembly for Faithful Temporal Chest X-ray Reporting**
*(Lắp ráp theo vùng — có kiểm chứng, dựa trên bằng chứng — để sinh báo cáo X-quang ngực theo thời gian một cách trung thực.)*

---

## 0. Tóm tắt cho người đọc nhanh

VERA là một hệ thống AI đa phương thức cho ảnh X-quang lồng ngực (CXR) với một mục tiêu khác biệt: **không
chỉ chẩn đoán, mà giải thích được và kiểm chứng được từng câu trong báo cáo**.

Khác biệt cốt lõi nằm ở **cách sinh báo cáo**. Hầu hết hệ thống hiện nay để một mô hình ngôn ngữ (LLM) tự
do "viết" báo cáo từ ảnh — cách này trôi chảy nhưng dễ **bịa** (hallucinate): nói ra bệnh, vị trí, hay diễn
biến mà ảnh không hề chứng minh. VERA lật ngược: báo cáo cuối cùng chỉ là **bản đọc lại (readout) của một
bảng dự đoán có cấu trúc và verify được**. Không có khâu "sinh chẩn đoán tự do". Hệ quả: **ảo giác bị triệt
tiêu ngay từ thiết kế (by construction)**, chứ không phải "giảm bớt" hay "bắt lỗi sau".

Bốn chữ trong tên gánh bốn ý:
- **V**erifiable — mọi mệnh đề trong báo cáo truy ngược được về một ô dữ liệu cụ thể.
- **E**vidence-grounded — nội dung neo vào bằng chứng có cấu trúc, không tự bịa.
- **R**egional — suy luận trên **29 vùng giải phẫu** của lồng ngực.
- **A**ssembly — báo cáo được **lắp ráp** từ output các mô-đun, không sinh ra một cách tự do.

Phụ đề: *Faithful* (trung thực — phản ánh đúng lý do thật của mô hình) và *Temporal* (so sánh ảnh cũ ↔ ảnh
mới để mô tả diễn biến bệnh).

---

## 1. Vấn đề & Động cơ

Trong thực hành lâm sàng, một báo cáo X-quang sai cách nguy hiểm nhất không phải là báo cáo *thiếu trôi
chảy*, mà là báo cáo **nghe rất hợp lý nhưng sai sự thật**. Một bác sĩ ít kinh nghiệm rất dễ bị một câu văn
tự tin dẫn đi sai hướng. Vì vậy VERA ưu tiên **tính trung thực và khả năng kiểm chứng** hơn là độ trôi chảy
ngôn ngữ — và chấp nhận đây là một **điểm vận hành (operating point) có chủ đích vì an toàn người bệnh**.

Hai loại ảo giác mà VERA nhắm tới:
1. **Ảo giác nội dung** — nói ra một bệnh/vị trí/thiết bị mà ảnh không có.
2. **Ảo giác thời gian** — mô tả "bệnh nặng lên / cải thiện" trong khi **không hề có ảnh cũ** để so sánh.

VERA giải quyết cả hai bằng *thiết kế*, không bằng *hậu kiểm*.

---

## 2. Đóng góp chính

1. **Báo cáo = readout của bảng verify được.** Triệt tiêu ảo giác *by construction*: nếu một mệnh đề không
   truy được về một ô trong bảng dự đoán (M3/M4), nó **không thể** xuất hiện trong báo cáo.
2. **Suy luận theo vùng (29 vùng giải phẫu).** Mọi khẳng định bệnh lý gắn với câu trả lời "ở đâu", giúp bác
   sĩ định vị và đối chiếu nhanh.
3. **So sánh thời gian trung thực.** Diễn biến (cải thiện / ổn định / nặng lên) được suy ra từ **ảnh prior
   thật của chính bệnh nhân**, và **tự động tắt sạch** mọi ngôn ngữ thời gian khi không có ảnh cũ.
4. **Khung giải thích phân tầng.** Mỗi câu kèm: vùng dẫn dắt, độ tin cậy đã hiệu chỉnh (calibrated), và (tuỳ
   điều kiện) các *concept* y khoa hỗ trợ — kèm cơ chế *từ chối trả lời (abstain)* khi không chắc.

---

## 3. Kiến trúc tổng thể

VERA gồm **một bước tiền xử lý (M0)** và **năm mô-đun (M1–M5)** chảy nối tiếp nhau:

```
            ẢNH X-QUANG (current) + (tuỳ chọn) ẢNH PRIOR
                              │
        ┌─────────────────────▼──────────────────────┐
        │ M0  TIỀN XỬ LÝ                             │
        │  resize 512 → center-crop 448 · 14 nhãn    │
        │  CheXpert {1/0/-100} · bbox 29 vùng · cặp  │
        │  prior↔current · nhãn tiến triển           │
        └─────────────────────┬──────────────────────┘
                              │
        ┌─────────────────────▼──────────────────────┐
        │ M1  ENCODER  (BioViL-T, đóng băng)         │
        │   ảnh → lưới đặc trưng 196×512 + 1 vector  │
        │   toàn ảnh  →  [197 × 512]                 │
        └───────────┬───────────────────┬────────────┘
                    │                   │
   ┌────────────────▼───────┐           │
   │ M2  SCENE GRAPH        │           │
   │  (a) Detector 29 vùng  │  bbox     │  lưới đặc trưng
   │  (b) LLM đọc report →  │  29 vùng  │
   │      finding theo vùng │           │
   └────────────────┬───────┘           │
                    │  29 bbox          │
        ┌───────────▼───────────────────▼─────────────────┐
        │ M3  PHÂN LOẠI THEO VÙNG  (C-head)               │
        │  attention-pool 196→29 (mask theo bbox)         │
        │  → 69 concept → 14 bệnh CheXpert / vùng         │
        │  + nhánh toàn-ảnh cho finding quan hệ           │
        │  Ra:  region_logit[29,14] · region_feat[29,512] │
        └───────────┬───────────────────┬─────────────────┘
        (current)   │                   │  (chạy lại cho ảnh prior)
                    │                   ▼
                    │       ┌───────────────────────────┐
                    │       │ M4  TIẾN TRIỂN THỜI GIAN  │
                    │       │  Siamese current ↔ prior  │
                    │       │  → 29×14×3                │
                    │       │  {cải thiện/ổn định/nặng} │
                    │       └───────────┬───────────────┘
                    │                   │
        ┌───────────▼───────────────────▼─────────────┐
        │ M5  LẮP RÁP BÁO CÁO FAITHFUL (6 tầng)       │
        │  bảng M3/M4 → assert/hedge/abstain/omit     │
        │  → realize (template / bảng) → VERIFY       │
        │  (xác định, KHÔNG dùng LLM để verify)       │
        └─────────────────────┬───────────────────────┘
                              ▼
              BÁO CÁO + provenance từng câu
              + bản đồ phủ 29 vùng + change-ledger
```

> **Nguyên tắc bất biến xuyên suốt:** mọi thứ "chỉ có lúc huấn luyện" (report của bác sĩ, scene graph nhãn-
> từ-report) chỉ được dùng làm **nhãn giám sát khi train**. Lúc vận hành thật (inference), đầu vào **chỉ là
> ảnh** → detector → M3 → M4 → M5. Không thành phần nào được phụ thuộc một input "đôi khi vắng mặt".

---

## 4. M0 — Tiền xử lý dữ liệu

M0 chuẩn hoá ba thứ về một dạng đồng nhất: **ảnh**, **hộp giới hạn vùng (bbox)**, và **nhãn**.

### 4.1 Chuẩn hoá ảnh (geometry duy nhất, dùng chung mọi nơi)
- **Bước 1 — Resize:** thu/phóng ảnh sao cho **cạnh ngắn = 512 px** (giữ nguyên tỷ lệ, nội suy BILINEAR).
- **Bước 2 — Center-crop:** cắt **448 × 448** ở chính giữa. Nếu một cạnh ngắn hơn 448 (hiếm) thì đệm đen.
- **Vì sao crop vuông 448:** đồng nhất kích thước đầu vào, khớp với encoder M1, và loại bớt viền nhiễu.
- Geometry này là **một chuẩn duy nhất** áp cho *mọi* ảnh và *mọi* dataset — ảnh current và ảnh prior phải
  qua **đúng cùng một phép biến đổi**, nếu không phép so sánh thời gian ở M4 sẽ vô nghĩa.

### 4.2 Toạ độ hộp giới hạn (bbox) của 29 vùng
- Bbox giải phẫu từ Chest ImaGenome được **rescale qua đúng geometry trên** → toạ độ trong không gian crop
  448 × 448.
- Hộp nào bị đẩy **hoàn toàn ra ngoài** khung crop → quy về **sentinel `(0,0,0,0)`**; toàn bộ pipeline phía
  sau **lọc bỏ** sentinel này (vùng đó coi như "không đánh giá được" trên ảnh đã crop).

### 4.3 Nhãn bệnh — quy ước 3 trạng thái (RẤT QUAN TRỌNG)
14 lớp bệnh theo chuẩn **CheXpert** (thứ tự cố định, *No Finding* ở vị trí 8). Mỗi nhãn nhận **một trong ba**
giá trị:

| Giá trị | Ý nghĩa | Cách dùng khi train |
|--------:|---------|---------------------|
| `1` | **Dương tính** (report khẳng định có) | tính loss bình thường |
| `0` | **Âm tính** (report khẳng định không) | tính loss bình thường |
| `-100` | **Không rõ / không nhắc tới** | **bị mask — KHÔNG đưa vào loss** |

- **Chính sách "uncertain → unknown" (chủ đích):** nhãn **uncertain** (`-1`) của CheXpert được **quy về
  `-100`**, tức gộp chung với "không nhắc tới". Đây là lựa chọn **U-Ignore** có chủ ý: ta **không** ép mô
  hình học một câu trả lời cứng cho những trường hợp bản thân report cũng mơ hồ.
- **Tuyệt đối KHÔNG gộp `-100 → 0`.** Coi "không nhắc tới" = "âm tính" sẽ ép mô hình học âm tính giả cho rất
  nhiều nhãn (đặc biệt với các report không đề cập đủ 14 lớp) → thiên lệch âm tính + nhiễu lớp hiếm.
- Hệ quả kỹ thuật: hàm mất mát là **masked BCE** — chỉ tính trên các ô `1`/`0`, bỏ qua mọi ô `-100`.

### 4.4 Ghép cặp thời gian & nhãn tiến triển
- **Ghép cặp prior ↔ current:** theo cùng bệnh nhân, sắp theo thời điểm chụp; ảnh cũ là *prior*, ảnh mới là
  *current*. Ảnh đầu tiên của một bệnh nhân không có prior → **không tạo cặp** (sẽ chảy xuống tầng "tắt ngôn
  ngữ thời gian" ở M5, đây là hành vi đúng, không phải lỗi dữ liệu).
- **Nhãn tiến triển (cho M4):** lấy từ **`comparison_cues`** của Chest ImaGenome — tín hiệu so sánh mà NLP đã
  bóc tách sẵn từ report. Chỉ tồn tại đúng **ba giá trị**, ánh xạ 1–1 sang ba lớp tiến triển:
  `no change → ổn định` · `improved → cải thiện` · `worsened → nặng lên`.

### 4.5 Thống kê dữ liệu (tham khảo)
| Nguồn | Ảnh | Bệnh nhân | Có report? | Có scene graph (bbox + finding)? |
|-------|----:|----------:|:----------:|:--------------------------------:|
| **MIMIC-CXR + ImaGenome** | 222,168 | 63,334 | có | **có** (silver toàn bộ + gold người-gán) |
| **CheXplus** | 191,046 | 64,686 | có | sinh từ report (nhãn yếu) |

- **Cặp thời gian (MIMIC):** 253,306 cặp prior↔current.
- **Nhãn tiến triển (region-instances):** ổn định 373,696 · nặng lên 190,708 · cải thiện 130,281.

---

## 5. M1 — Encoder (BioViL-T, đóng băng)

- **Vai trò:** chuyển mỗi ảnh CXR thành biểu diễn vector không gian để các mô-đun sau khai thác.
- **Backbone:** **BioViL-T** — một encoder thị giác-ngôn ngữ chuyên cho ảnh y khoa ngực, **được đóng băng**
  (frozen): VERA không huấn luyện lại nó, chỉ dùng nó như một bộ trích đặc trưng ổn định.
- **Đầu ra cho mỗi ảnh:**
  - một **lưới đặc trưng không gian 196 × 512** (tương ứng lưới 14 × 14 ô, mỗi ô là vector 512 chiều) —
    "ảnh nhìn thấy gì, ở đâu";
  - một **vector toàn ảnh 512 chiều** — tóm tắt ngữ cảnh toàn cục.
  - Gộp lại: tensor `[197 × 512]` cho mỗi ảnh.
- Vì M1 đóng băng nên đặc trưng của mỗi ảnh là **xác định (deterministic)** → có thể tính trước & lưu cache
  một lần, dùng lại cho cả M3 và M4.

---

## 6. M2 — Scene Graph (bản đồ giải phẫu — ngữ nghĩa)

M2 dựng một "bản đồ cảnh" của lồng ngực, gồm **hai nhánh độc lập**:

### 6.1 Nhánh không gian — Detector 29 vùng
- Một detector (họ YOLO) được tinh chỉnh để **khoanh 29 vùng giải phẫu** trên ảnh CXR (phổi, thuỳ phổi,
  trung thất, tim, góc sườn hoành, khí quản, carina, cơ hoành, xương đòn, cột sống, cung động mạch chủ…).
- 29 hộp này chính là **mặt nạ định vị** để M3 gom đặc trưng theo vùng. **Đây là điểm khớp giữa M2 và M3.**
- **Quy ước quan trọng:** dùng **hộp do detector dự đoán** cho cả lúc train lẫn lúc infer M3 (để phân phối
  dữ liệu lúc train khớp lúc vận hành). Hộp "gold" người-gán chỉ dùng để huấn luyện detector.

### 6.2 Nhánh ngữ nghĩa — LLM đọc report → finding theo vùng
- Một mô hình ngôn ngữ **Qwen 2.5 (3B)** được tinh chỉnh (QLoRA) để học ánh xạ
  `(report + danh sách vùng đã phát hiện) → các finding theo từng vùng` ở dạng JSON gọn, tương thích định
  dạng ImaGenome.
- Cùng với finding, LLM bóc ra các **cue so sánh** (`comparison_cues`) — nguồn nhãn tiến triển cho M4.
- **Vai trò khi vận hành:** nhánh này **chỉ chạy lúc huấn luyện** (để sinh nhãn cho dữ liệu chưa có scene
  graph, ví dụ CheXplus). Lúc vận hành thật, M3/M4 **thuần ảnh** — không cần report current.

---

## 7. M3 — Phân loại bệnh theo vùng

**Đầu vào:** lưới đặc trưng `196×512` (từ M1) + 29 bbox (từ M2). **Đầu ra:** bảng bệnh-theo-vùng + các tín
hiệu định vị chảy xuống M4/M5.

### 7.1 Attention-pool: gom 196 ô đặc trưng về 29 vùng
- Mỗi vùng có một **truy vấn học được (learned query)**; truy vấn này "chú ý" (attend) lên 196 ô đặc trưng,
  nhưng **bị mask theo bbox của vùng đó** (ô ngoài hộp bị triệt). Kết quả: một vector cho mỗi vùng → `29×512`.
- **Vì sao attention-pool thay vì lấy trung bình hộp:** (a) cứu được các **tổn thương nhỏ, khu trú** (lấy
  trung bình sẽ làm loãng tín hiệu nhỏ trong một hộp lớn); (b) trọng số chú ý `α` **chính là** tín hiệu định
  vị *trung thực* "mô hình lấy tín hiệu từ chỗ nào trong vùng" — không phải bản đồ nhiệt hậu kiểm.
- **Giữ nguyên cấu trúc 29 vùng** xuyên suốt — tuyệt đối không "ép phẳng" về mức toàn ảnh (sẽ phá M4 và M5).

### 7.2 Tầng concept → tầng bệnh
- Từ đặc trưng vùng, M3 dự đoán **69 concept y khoa** (43 finding giải phẫu + 10 bệnh + 12 ống/đường truyền +
  4 thiết bị), rồi từ đó ra **14 nhãn bệnh CheXpert cho mỗi vùng**.
- 69 concept đóng vai một **"nút thắt giải thích"** tiềm năng: nếu đủ tin cậy, ta nói được "bệnh d *vì*
  concept c". (Điều kiện để được phép tuyên bố điều này — xem 7.3.)

### 7.3 Ba hướng head & tiêu chí chọn theo *faithfulness*
M3 hỗ trợ **ba hướng**, khác nhau **không chỉ ở độ chính xác mà ở việc "giải thích bằng concept có trung
thực không"**:

| Hướng | Đường dự đoán bệnh | Tính trung thực | Vai trò |
|------|--------------------|-----------------|---------|
| **A — Direct** | đặc trưng vùng → bệnh | **where-faithful** vô điều kiện (giải thích = "ở đâu") | **mặc định an toàn** |
| **B — Concept Bottleneck** | bệnh **chỉ** qua 69 concept | con đường **duy nhất** cho giải thích "vì sao" trung thực | bật nếu qua kiểm định |
| **C — Hybrid** | concept ⊕ đặc trưng ảnh | rủi ro **rò rỉ** (concept có thể chỉ trang trí) | accuracy cao nhưng phải qua test |

- **Quy tắc quyết định (theo faithfulness, KHÔNG theo accuracy):**
  1. Đo **F1 "concept đoán từ ảnh"** trên dữ liệu người-gán. Nếu kém ⇒ "giải thích bằng concept" bị loại, hệ
     ship bằng **hướng A** (where-faithful), concept hạ xuống phần phụ lục.
  2. Hướng **B** chỉ được tuyên bố "vì sao" trung thực nếu qua **concept-intervention test** (can thiệp bật/
     tắt concept → dự đoán bệnh đổi đúng hướng).
  3. Hướng **C** phải qua **leakage test** (xoá/ngẫu nhiên hoá kênh concept; nếu accuracy gần như không tụt
     ⇒ concept chỉ trang trí ⇒ **không** được trình là "vì sao").
- **Hướng A luôn là lưới an toàn.** Đừng để accuracy cao của C kéo hệ rời trục "readout verify được".

### 7.4 Nhánh toàn-ảnh cho finding *quan hệ*
- Vài finding mang tính **quan hệ/toàn cục**, không nằm gọn trong một hộp: tim to (cardiomegaly), phù lan
  toả, thể tích phổi thấp. Chúng đi qua một **head toàn-ảnh** riêng.
- **Cách trình bày (đã chốt):** **không** cố nhồi các finding này vào một vùng cụ thể. Trong báo cáo, chúng
  được dán nhãn rõ là **"đánh giá trên toàn ảnh"** — tức ta trung thực rằng *không chắc nó nằm ở vùng nào*,
  thay vì gán bừa một toạ độ giả.

### 7.5 Xử lý mất cân bằng & metric
- Dữ liệu lệch nặng (lớp hiếm rất ít dương tính) → dùng **pos_weight log-scale** (kiểu RADAR) cho mỗi số hạng
  BCE.
- **Metric chuẩn = macro-F1 + per-class.** **Không** dùng accuracy (bị lớp đa số kéo lệch).

### 7.6 Lưu ý kiến trúc
- **Neck tắt:** giữ nguyên đặc trưng vùng **512 chiều** (tín hiệu giàu hơn) thay vì nén xuống. Đây là contract
  chia sẻ với M4.
- **Head có thể hoán đổi MLP ↔ KAN:** mặc định là **MLP**; biến thể **KAN** (Kolmogorov-Arnold Network) đặt ở
  *tầng suy luận có cấu trúc* (không phải ở backbone) như một đóng góp/ablation — đổi một cờ là so sánh được.

---

## 8. M4 — Tiến triển bệnh theo thời gian

**Mục tiêu:** với mỗi `(vùng, bệnh)`, xác định diễn biến giữa ảnh prior và ảnh current: **cải thiện / ổn
định / nặng lên** → tensor `29 × 14 × 3`.

### 8.1 Cấu trúc Siamese (chia sẻ trọng số)
- "Siamese" = **một nhánh dùng chung trọng số**, chạy **hai lần** — một cho ảnh current, một cho ảnh prior —
  để hai ảnh nằm **cùng một không gian biểu diễn**, so sánh mới có nghĩa.
- Nhánh chia sẻ đó **chính là** đường M1 → attention-pool → M3 (đã đóng băng sau khi train M3). Vì M3 đóng
  băng nên đặc trưng vùng là xác định → tính trước & cache một lần; M4 chỉ **tra cache**. ⇒ Siamese gần như
  *miễn phí* về kiến trúc.

### 8.2 Contract đầu vào (mỗi vùng)
Với mỗi vùng, head tiến triển nhận: đặc trưng hai ảnh **và cả hiệu của chúng**, kèm logit bệnh hai thời điểm:

```
[ feat_curr(512) ; feat_prior(512) ; (feat_curr − feat_prior)(512) ]  →  1536
[ logit_curr(14) ; logit_prior(14) ]                                  →    28
                                                       tổng mỗi vùng  =  1564
```

- Giữ **cả hai vế lẫn hiệu** (không chỉ lấy phép trừ) để head **tự học** cách so sánh, không ép sẵn dấu trừ.
- Tín hiệu cốt lõi là **hiệu đặc trưng `curr − prior`** — chênh lệch tường minh giữa hai thời điểm.

### 8.3 Tăng cường dữ liệu bằng đảo thời gian (time-flip)
- Một phép augment hiệu quả: **hoán đổi thứ tự hai ảnh** (prior ↔ current) **và đảo nhãn tương ứng**
  (*cải thiện ↔ nặng lên*; *ổn định* giữ nguyên). Cách này nhân đôi tín hiệu hướng và buộc mô hình học diễn
  biến đối xứng.
- *Lưu ý:* loại trừ lớp **Support Devices** khỏi phép đảo (việc gắn/rút thiết bị không đối xứng theo thời
  gian như tiến triển bệnh).

### 8.4 Đầu ra, mất cân bằng & metric
- Ra `29×14×3`. Chỉ giám sát ô `(vùng, bệnh)` có cue và vùng hiện diện ở **cả** hai ảnh; ô khác để mask.
- Lớp **"ổn định" áp đảo** → dùng class-weight. **Metric = macro-F1 + per-class + change-only F1.** Nếu
  accuracy ≈ tỉ lệ lớp "ổn định" thì đó là **cờ đỏ** (mô hình chỉ đoán bừa lớp đa số).
- **Head MLP mặc định, KAN là ablation** (giữ cùng interface với M3).

### 8.5 Dự phòng cho vận hành: cờ "prior có report" & khả năng hai bộ trọng số
Lúc vận hành thật, tình huống ở đầu-prior **bất đối xứng**: ảnh prior **có khi đã kèm report** (lần khám cũ
đã được bác sĩ đọc), có khi **chỉ có ảnh**. Hai chế độ này cho M4 chất lượng đầu-prior khác hẳn nhau, nên M4
khả năng cần:
- **Một cờ `prior_report_available`** nối vào đầu vào, để mô hình **biết mình đang ở chế độ nào** (đầu-prior
  "sạch" từ report, hay "thuần ảnh" ước lượng từ M3) thay vì phải tự đoán.
- **Khả năng hai bộ trọng số** (hoặc một bộ huấn luyện theo kiểu *modality-dropout* ở đầu-prior để chịu được
  cả hai chế độ). Lý do: một M4 chỉ quen đầu-prior-thuần-ảnh có thể đã ngầm học **bù trừ** cho cái nhiễu đó;
  nạp đột ngột một đầu-prior *sạch* vào sẽ khiến nó **bù nhầm** → có thể *tệ hơn* dù input "tốt hơn". Vì vậy
  muốn dùng report-prior để nâng chất lượng thì **M4 phải được train cho biết chế độ đó** — không thể chỉ
  "thay lúc launch" một cách ngây thơ.

> Đây là phần **dự phòng/mở rộng**, gắn với nhánh prior-report ở **mục 12.3** — giữ tách khỏi đường chính
> thuần-ảnh của v1; chỉ chạm **đầu-prior** (kênh `logit_prior`), không đụng đầu-current (current khi launch
> không bao giờ có report).

---

## 9. M5 — Lắp ráp báo cáo trung thực

M5 biến bảng dự đoán M3 (bệnh theo vùng) + M4 (tiến triển) thành một báo cáo **là bản đọc lại của bảng — KHÔNG
sinh chẩn đoán nào mới**. Phần lớn xác định (deterministic), chạy trên CPU.

Hai con số mục tiêu: **out-of-table ≈ 0** (không có finding nào lọt ra ngoài bảng) và **temporal-halluc = 0**
(không có ngôn ngữ thời gian khi thiếu prior).

### 9.1 Sáu tầng theo độ tin cậy
| Tầng | Nội dung |
|------|----------|
| **1 — Lõi cấu trúc** | Từ `region_logit[29,14]` (M3) + tiến triển `29×14×3` (M4), áp ngưỡng → mỗi finding nhận **assert / hedge / abstain / omit**. Mọi câu về sau phải truy về một mục ở đây. |
| **2 — Định vị "ở đâu"** | Vùng dẫn dắt cho mỗi bệnh (chính xác, faithful) + trọng số `α` (tín hiệu nội vùng, dán nhãn "lấy tín hiệu từ đâu") + **bản đồ phủ 29 vùng**. |
| **3 — Hiệu chỉnh độ tin (calibration) + abstain** | Temperature scaling theo từng lớp; đặt ngưỡng `τ`. Dưới ngưỡng → **hedge** (ngôn ngữ dè dặt) hoặc **abstain** ("nhường radiologist"). Không chắc thì nói không chắc — đó là một phần của trung thực. |
| **4 — Cổng thời gian (rủi ro #1)** | **Không có prior ⇒ TẮT SẠCH mọi ngôn ngữ thời gian** (không tồn tại đường code nào sinh ra từ thời gian). **Có prior ⇒ cụm tiến triển = đọc thẳng argmax của M4 (1:1)**, không để mô hình tự "đoán diễn biến". |
| **5 — Hiện thực hoá văn bản** | **Template** (mặc định, trung thực tuyệt đối, đọc khô) hoặc **paraphraser có ràng buộc** (LLM viết mượt nhưng *prose-from-table*: chỉ được diễn đạt lại các finding đã liệt kê, **cấm thêm/bớt**, và phải qua tầng 6). |
| **6 — Kiểm chứng (verify)** | **Round-trip:** trích lại nhãn từ báo cáo đã sinh → **đối chiếu với bảng M3/M4** → loại mọi finding lọt ra ngoài (out-of-table). **Luật phủ (coverage):** mọi ô assert-dương phải được xử lý; thiếu ô nào → bật cờ (bắt lỗi bỏ-sót). |

### 9.2 Hai ràng buộc bất biến của M5
- **Verifier luôn xác định, KHÔNG bao giờ là LLM.** (Một LLM verifier sẽ tự bịa, phá vỡ chính mục đích.) Đây
  là chỗ sau này cắm CheXbert/RadGraph vào.
- **Temporal-halluc = 0 by construction:** một cụm tiến triển chỉ được phát ra khi tồn tại ô M4 cho ảnh đó.
  Không prior → không có ô M4 → **không có đường code** nào sinh ra từ ngữ thời gian.

### 9.3 Định dạng đầu ra & trực quan hoá
- Mỗi báo cáo kèm **provenance từng câu** — mỗi mệnh đề trỏ về ô nguồn `(vùng, bệnh, độ tin, tiến triển M4)`;
  câu nào không truy được → tô đỏ (vừa là viz, vừa là cơ chế bắt ảo giác).
- **Bản đồ phủ 29 vùng** (bình thường / bất thường / không chắc / không-đánh-giá-được) — biến "sự im lặng"
  thành một khẳng định verify được.
- **Change-ledger** prior→current (finding | prior | current | hướng) — đọc thẳng argmax M4.
- **Tuỳ chọn xuất báo cáo dạng BẢNG:** ngoài văn xuôi, M5 chừa ngõ xuất báo cáo ở **dạng bảng** (mỗi dòng một
  finding kèm vùng/độ tin/diễn biến) cho ngữ cảnh cần đối chiếu nhanh và máy đọc.

---

## 10. Dữ liệu & phân vai (data provenance)

Nguyên tắc bất di bất dịch: **nhãn yếu (do mô hình khác sinh) chỉ được dùng để TRAIN/PRETRAIN, TUYỆT ĐỐI
không dùng để đánh giá.** Mọi con số trong bảng kết quả đến từ **nhãn người-gán**.

| Dataset | Vai trò huấn luyện | Vai trò đánh giá |
|---------|--------------------|------------------|
| **MIMIC + ImaGenome** | trục chính: train M3 + M4 (nhãn tiến triển từ `comparison_cues`) | **eval chính** (trên phần người-gán / gold) |
| **CheXplus** | pretrain/augment (scene graph yếu sinh từ report) — giá trị là **đa dạng phân phối**, quyết giữ/bỏ bằng ablation | **eval-never** |

- **Nhãn CheXplus** lấy sẵn từ bộ dữ liệu gốc, chỉ **sắp lại đúng thứ tự 14 lớp như MIMIC**.
- **Vòng lặp nhãn yếu cần cảnh giác:** M2 (LLM) sinh nhãn yếu cho CheXplus rồi train M3/M4 — đây là một dạng
  *chưng cất ngầm* (M3/M4 học bắt chước M2). Rủi ro: trần hiệu năng bị khoá bởi M2, và lỗi của M2 *chính là
  ảo giác* nên có thể dạy M3/M4 ảo giác theo. Cách kiểm soát: lằn ranh "weak → train, người-gán → eval";
  cân nhắc **pretrain trên CheXplus-yếu rồi finetune kết thúc trên MIMIC-sạch** để dữ liệu sạch có "tiếng nói
  cuối".

---

## 11. Đánh giá & Kiểm chứng tính trung thực

VERA tách bạch **ba nhóm metric** và chủ động **sở hữu trade-off** (hi sinh fluency lấy faithfulness):

**A. Hiệu quả lâm sàng (đúng/sai):**
- **macro-F1 + per-class** cho M3 (14 bệnh) và M4 (3 lớp tiến triển, kèm **change-only F1**). Không dùng
  accuracy.

**B. Tính trung thực (faithfulness) — trục bán hàng của VERA:**
- **out-of-table rate** & **temporal-halluc rate** — đo bằng round-trip verify (tầng 6). Mục tiêu ≈ 0.
- **Kiểm chứng concept (cho hướng B/C của M3):** F1 "concept-từ-ảnh", **intervention test** (B),
  **leakage test** (C) — quyết concept có được trình là "vì sao" hay không.
- **Calibration thực chứng:** **reliability diagram + ECE**, tách per-class, đặc biệt soi **lớp hiếm**; lớp
  nào calibrate kém thì mặc định **hedge**.
- **Đường cong deletion/insertion mức vùng** (bằng chứng *phụ*): xoá/chèn dần đặc trưng vùng theo độ quan
  trọng để đo độ faithful của tín hiệu định vị. Lưu ý kết quả nhạy với "giá trị thay thế" (zero / mean /
  noise) → phải nêu rõ baseline và kiểm độ bền với ≥1 baseline khác. Đây là *cần-nhưng-không-đủ*; bằng chứng
  faithfulness **chính** vẫn là cấp-construction (out-of-table, temporal-guard).

**C. Độ trôi chảy (fluency):**
- CheXbert-F1 / BLEU / v.v. — báo cáo *cùng* hai nhóm trên, kèm lập luận VERA là **điểm vận hành cố ý** vì an
  toàn người bệnh.

**Các ablation/đối chứng đáng giá:**
- **KAN vs MLP** (parity ở head M3/M4) · **có/không CheXplus** (đo trên test MIMIC sạch) ·
  **template vs constrained-paraphrase** (đo cả fluency lẫn out-of-table) ·
  **so sánh faithfulness xuyên paradigm** (chạy round-trip trên *output của baseline sinh tự do* để ước
  lượng out-of-table/temporal-halluc của chúng — để so sánh công bằng).
- *(Tuỳ chọn)* **reader study nhỏ** với bác sĩ ít kinh nghiệm (faithful vs free-prose, đo tỉ lệ bị dẫn sai) —
  biến động cơ an toàn thành bằng chứng; nếu không làm thì hạ xuống *motivation*, không over-claim.
- *(Tuỳ chọn)* kiểm định **độ ổn định thống kê** (bootstrap khoảng tin cậy / cross-validation) cho các số
  chính — chưa chốt, cân nhắc khi viết bài.

> **Tiền lệ cùng lab — OsteoGA:** dùng đúng cấu trúc difference-feature/Siamese, nhưng ảnh thứ hai của họ là
> **counterfactual do GAN bịa** (không verify được). Ảnh thứ hai của VERA-M4 là **prior thật của bệnh nhân**
> (verify được) ⇒ VERA là *phiên bản faithful* của ý tưởng difference-feature đó. Trích dẫn như precedent,
> nêu rõ khác biệt.

---

## 12. Hạn chế đã biết & Hướng phát triển

VERA v1 cố ý giữ **phạm vi hẹp mà sạch** ("một câu chuyện không dấu hoa thị"). Các hướng dưới đây *có giá
trị* nhưng được **tách riêng** để không làm loãng luận điểm cốt lõi:

1. **Nhãn đánh giá thời gian người-gán (ưu tiên cao nhất).** Nhãn tiến triển dùng để *train* M4 đến từ
   `comparison_cues` (nhãn yếu, do NLP bóc). Để *đánh giá* M4 một cách thuyết phục, cần một **tập test
   temporal do người gán** (ví dụ MS-CXR-T), tách bạch khỏi nhãn train. Cả claim "temporal faithful" đứng
   hay sụp ở đây → khởi động tìm nguồn **sớm, song song** với train.
2. **Concept bottleneck đầy đủ làm kênh "vì sao".** Chỉ bật khi qua được các kiểm định ở mục 11.B; nếu không,
   concept ở lại phần phụ lục với báo cáo trung thực vì sao chưa dùng làm "vì sao".
3. **Tận dụng report của ảnh prior (bất đối xứng thông tin).** Lúc vận hành, ảnh prior **thường đã có report**
   (lần khám cũ đã được đọc). Report prior là **bằng chứng thật, verify được** → có thể parse ra trạng thái
   prior sạch để nâng độ chính xác của M4 ở phía prior. Đây là một **nhánh mở rộng riêng** (có cờ "prior-
   report-available" + chế độ huấn luyện hai luồng + ablation bắt buộc), **không** trộn vào v1 thuần-ảnh.
4. **Grounding cho finding toàn cục.** Định nghĩa rõ "grounding mức toàn-ảnh" cho lớp finding quan hệ (mục
   7.4) và dán nhãn khác hẳn với grounding theo vùng — trung thực rằng đây là lớp "toàn cục", không phải "ô
   vùng".
5. **Chưng cất từ thông tin đặc quyền (privileged information).** Hướng "train với thông tin đầy đủ, vận hành
   khi thiếu vẫn giữ tri thức" hấp dẫn nhưng có **bẫy riêng**: nếu teacher kết luận nhờ *đọc report* mà ảnh
   không đủ tín hiệu, thì đang dạy student "nói X mà không nhìn thấy X" — ảo giác bị chưng cất vào trọng số.
   ⇒ Tách thành **study riêng**, với cổng gác "chỉ chưng cất finding chứng minh được là suy ra từ ảnh".

---

## 13. Một câu tóm lại

> VERA không cố làm cho một mô hình sinh báo cáo **bớt** bịa. Nó **thay khâu sinh tự do bằng khâu lắp ráp từ
> một bảng verify được** — nên báo cáo *không thể* nói điều gì mà bảng không chứng minh, và *không thể* nói
> về thời gian khi không có quá khứ để so. Trung thực ở đây là **thuộc tính của kiến trúc**, không phải một
> chỉ số cần tối ưu.
</content>
</invoke>
