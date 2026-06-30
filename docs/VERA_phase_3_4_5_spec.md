# VERA — Đặc tả chi tiết Phase 3 / 4 / 5

**Tên bài (chốt):**
**VERA: Verifiable, Evidence-grounded Regional Assembly for Faithful Temporal Chest X-ray Reporting**

*Nghĩa:* "Lắp ráp theo vùng, có kiểm chứng và dựa trên bằng chứng, cho việc sinh báo cáo X-quang ngực
theo thời gian một cách trung thực." Bốn chữ acronym gánh bốn ý cốt lõi: **V**erifiable (mọi claim
truy/kiểm lại được) · **E**vidence-grounded (nội dung neo vào bằng chứng có cấu trúc, không tự bịa) ·
**R**egional (29 vùng giải phẫu) · **A**ssembly (lắp ráp từ output, KHÔNG sinh tự do). Phụ đề: *Faithful*
(theo nghĩa xAI: phản ánh đúng lý do thật của model, đối lập plausibility) · *Temporal* (so sánh prior↔current).

**Trục novelty mà mọi đặc tả dưới đây phục vụ:** report là **readout của một bảng dự đoán verify được**,
không có lõi sinh chẩn đoán → hallucination bị triệt *by construction*, không phải *giảm* hay *bắt sau*.
Bất cứ chi tiết thiết kế nào kéo VERA rời trục này đều phải bị đánh dấu và cân nhắc lại.

---

## PHASE 3 — M3: phân loại bệnh theo vùng (29 × 14) + định vị

### 3.0 Đầu vào / đầu ra tổng quát
- **Vào:** feature grid `196×512` (14×14×512) từ M1/BioViL-T (ảnh current); `29` bbox giải phẫu từ M2.
- **Ra (các tín hiệu chảy xuống M4/M5):**
  - `region_logit[29,14]` — logit bệnh theo vùng (→ `softmax_r` cho grounding "ở đâu").
  - `α[29,196]` — trọng số attention-pool (→ grounding nội vùng, **phụ**, dán nhãn "tín hiệu lấy từ đâu").
  - `region_feat[29,128]` — đặc trưng vùng sau neck, **chia sẻ với M4**.
  - confidence đã calibrate (đẩy sang Phase 5 tầng 3).
  - (tùy hướng) `concept_act[29,69]` — kích hoạt concept (chỉ có ở hướng B/C bên dưới).

### 3.1 Attention-pool: gom `196×512` → `29×512`
Mỗi vùng `r` có một **query học được**; query attend trên 196 ô feature, **mask theo bbox** của vùng đó
(ô ngoài box bị triệt hoặc giảm trọng số). Kết quả: một vector `512` cho mỗi vùng → `29×512`.

- Lý do dùng attention-pool thay mean-ROI: (a) cứu finding **focal nhỏ** (mean làm loãng tín hiệu nhỏ trong
  box lớn); (b) `α` là **trọng số pool thật** → dùng làm tín hiệu grounding *faithful* (không phải saliency post-hoc).
- **Giữ cấu trúc 29 vùng** — tuyệt đối không flatten `196×512` về image-level (sẽ phá M4 và M5).

### 3.2 Neck: `512 → 128`
`Linear 512→128 + LayerNorm + GELU` áp lên `29×512` → **`region_feat[29,128]`**. Lý do giữ neck: feature
gọn cho T-MLP (M4) + chuẩn hoá. `region_feat[128]` **dùng chung** cho disease head (Phase 3) *lẫn* T-MLP (Phase 4) —
đây là chỗ Siamese của M4 "miễn phí" về kiến trúc (xem Phase 4).

### 3.3 Ba hướng head đang thử — và faithfulness của từng hướng
> Quan trọng: ba hướng này khác nhau **không chỉ về accuracy mà về việc "lời giải thích bằng concept có
> faithful không"**. Với VERA, đây là khác biệt sống còn, không phải chi tiết kỹ thuật.

**Hướng A — Direct:** `region_feat → MLP → 14 label`.
- Đường mặc định, **faithful không điều kiện**. "Giải thích" của hướng này = **grounding theo vùng**
  (`softmax_r`, `α`), tức "ở đâu" — đúng với phạm vi đã descope của VERA (where-faithful, không why-sign).
- Không có concept, không có rủi ro concept-nhiễu. Đây là **fallback an toàn** nếu B/C trượt kiểm faithfulness.

**Hướng B — Concept Bottleneck (CBM):** `region_feat → MLP → 69 concept → MLP → 14 label`.
- Bệnh được dự đoán **CHỈ qua tầng concept** (bottleneck cứng). Nếu bottleneck thật, ta nói được "bệnh d
  vì concept c" một cách *ante-hoc* → đây là con đường **duy nhất** cho "why" faithful.
- **Hai điều kiện sống còn (đều đo được):**
  1. **Go/no-go concept-từ-ảnh.** Lúc launch không có report → 69 concept phải đoán **từ ảnh**. Phải đo
     **F1 concept-từ-ảnh trên silver MIMIC** trước. Concept tự đoán tệ ⇒ "why" thành plausible-mà-sai ⇒ bỏ hướng B.
  2. **Nhãn concept nhiễu.** 69 concept đến từ silver/pseudo-label (M2 attribute-từ-report + step-7 LLM).
     Cắm nhãn nhiễu vào *đường nhân quả* = đúng cái hallucination cần tránh. Phải kiểm độ tin nhãn concept.
- **Kiểm bottleneck có thật không (concept-intervention test):** can thiệp tay lên concept (bật/tắt c) →
  dự đoán bệnh phải đổi đúng hướng. Không đổi ⇒ bottleneck giả ⇒ concept không faithful.

**Hướng C — Hybrid (concept + ảnh joint):** `region_feat → MLP → 69 concept`; rồi `MLP(concept ⊕ ảnh) → 14 label`.
- **CẢNH BÁO — đây là hướng nguy hiểm nhất cho VERA.** Cho disease head một **kênh phụ thẳng từ ảnh** bên
  cạnh concept ⇒ **phá bottleneck**: bệnh có thể được quyết *vòng qua* concept. Hệ quả là **CBM leakage** —
  concept activations *trông* như giải thích nhưng có thể **không drive quyết định** (model quyết từ kênh ảnh,
  concept chỉ trang trí). Accuracy hướng C thường **cao nhất** (kênh ảnh bù lỗi concept) → đây là cám dỗ.
- Nếu giữ hướng C: **CẤM trình concept của nó như "why" faithful.** Cùng lắm concept là *dự đoán phụ*. Muốn
  dùng làm "why" thì phải qua **leakage test** (xem 3.4).

### 3.4 Tiêu chí chọn hướng — theo faithfulness, KHÔNG theo accuracy
Quy tắc quyết định cho bài VERA:
1. Chạy **go/no-go concept-từ-ảnh** (F1 trên silver). Trượt ⇒ "why-bằng-concept" off the table cho cả B và C;
   VERA ship với **hướng A** (where-faithful), concept hạ xuống ablation/phụ lục.
2. Với **B**: pass concept-intervention test (can thiệp concept → bệnh đổi đúng) thì mới được tuyên bố "why" faithful.
3. Với **C**: chạy **leakage test** — zero/randomize kênh concept nhưng giữ kênh ảnh; nếu accuracy bệnh **gần
   như không tụt** ⇒ concept trang trí ⇒ **không** được trình là "why".
4. **Headline "why" của paper chỉ được bật nếu một hướng pass CẢ go/no-go LẪN faithfulness test.** Ngược lại,
   VERA = where-faithful (hướng A) + concept để ở phần ablation với báo cáo trung thực vì sao không dùng làm "why".

> Cách trình bày này biến "ba hướng đang thử" từ rủi ro lật-luận-điểm thành **một ablation có nguyên tắc**:
> ta để *con số faithfulness* quyết hướng nào vào bài, không để accuracy quyết.

### 3.5 Nhánh global (finding quan hệ, không nằm trong 1 box)
Cardiomegaly, phù lan toả, low lung volumes là finding *quan hệ* → GAP toàn ảnh → **GlobalHead**. Gộp với
đường-vùng ở **image-logit** qua **cổng học được** (mượn gate global-local CGPR/PCF của EViKO:
`g = σ(gate(CLS)); fused = g⊙global + (1−g)⊙local`). Chỉ **mượn cơ chế gate**, KHÔNG bê EViKO nguyên bản.

### 3.6 Mất cân bằng (ưu tiên hàng đầu) & metric
- pos_weight log-scale kiểu RADAR `α_i = log(1 + |D|/w_i)` / focal loss.
- **Metric: macro-F1 + per-class.** KHÔNG dùng accuracy (bị lớp đa số kéo).
- **Region-level supervision** cho M3 là **blocker chất lượng** — confidence mức-vùng còn yếu đến khi có. Giải sớm.

---

## PHASE 4 — M4: tiến triển thời gian theo vùng (29 × 14 × 3)

### 4.1 Siamese curr−prior
"Siamese" = hai nhánh **dùng chung trọng số**, mỗi nhánh xử lý một ảnh, cho feature trong **cùng không gian**
để so sánh có nghĩa. Ở đây "nhánh chia sẻ trọng số" **chính là** đường M1→attention-pool→neck của Phase 3,
chạy **hai lần** (ảnh current và ảnh prior) → mỗi ảnh ra `region_feat[29,128]`. Không phải tháp mới ⇒ gần như
miễn phí về kiến trúc.

### 4.2 Contract đầu vào head (giữ nguyên, không đổi so với T-KAN)
`region_in_dim = 412 = 128×3 + 14×2`, gồm cho mỗi vùng:
- `feat_curr[128]` + `feat_prior[128]` + `diff = feat_curr − feat_prior [128]` → `384`
- `disease_logit_curr[14]` + `disease_logit_prior[14]` → `28`

Lưu ý: giữ **cả hai vế lẫn hiệu** (không chỉ lấy diff) để head tự học, không ép sẵn dấu trừ.
**14 disease logit** lấy từ output Phase 3 — **bất kể Phase 3 đi hướng A/B/C, contract M4 không đổi**
(M4 tiêu thụ `region_feat[128]` + 14 logit; cách 14 logit được tạo ra không ảnh hưởng shape).

### 4.3 Head & đầu ra
- Head: **MLP default, KAN ablation** (giữ infra parity `KANHead↔MLPHead`).
- Ra: `29×14×3` — {improved, stable, worsened} cho mỗi (vùng, bệnh).

### 4.4 Mất cân bằng, metric, ràng buộc launch
- Imbalance: class-weight ("stable" áp đảo).
- **Metric: macro-F1 + per-class.** accuracy ≈ "stable" là **cờ đỏ** (model chỉ đoán bừa lớp đa số).
- **Launch:** không có prior → M4 tắt; M5 tầng-4 (temporal guard) chịu trách nhiệm tắt sạch ngôn ngữ thời gian.

### 4.5 Precedent OsteoGA (cùng lab) — và điểm khác biệt faithful
OsteoGA (SoICT'23) dùng đúng cấu trúc difference-feature/Siamese (backbone chia sẻ chạy trên ảnh gốc + ảnh
khôi phục). **Khác biệt sống còn:** ảnh thứ hai của họ là **counterfactual do GAN bịa** (không verify được);
ảnh thứ hai của M4 là **prior thật** của bệnh nhân (verify được). ⇒ **M4 = phiên bản faithful của ý tưởng
difference-feature đó.** Cite OsteoGA như precedent cùng lab, nêu rõ mình khác ở đâu.

---

## PHASE 5 — M5: faithful assembler (6 tầng theo độ tin cậy)

**Nguyên tắc bất biến:** M5 **KHÔNG sinh chẩn đoán**, KHÔNG reasoner/CoT tự do. Mọi mệnh đề là readout của
ô trong bảng M3/M4. Hai con số phải đạt: **out-of-table ≈ 0**, **temporal-halluc = 0** (báo thêm *residual
out-of-table sau verify* — đừng tuyên bố 0 tuyệt đối vì verifier có recall hữu hạn).

### Tầng 1 — Structured core
Lấy `region_logit[29,14]` (M3) + tiến triển `29×14×3` (M4). Calibrate (tầng 3) rồi áp ngưỡng → mỗi finding
nhận một trong: **assert / hedge / omit**. Đây là "lõi" của report, mọi câu về sau phải truy về một mục ở đây.

### Tầng 2 — Grounding "ở đâu"
- `softmax_r(region_logit[r,d])` → vùng nào dẫn dắt cho bệnh d (**exact**, faithful).
- `α` (attention-pool) → tín hiệu nội vùng, **phụ**, dán nhãn "model lấy tín hiệu từ đâu", KHÔNG "vì sao".
- (Điều kiện) Nếu Phase 3 hướng B/C **pass** faithfulness test ở 3.4 → concept trở thành kênh "why" faithful
  mà tầng này được phép surface ("d vì concept c1, c2"). Nếu **không** pass → tầng này chỉ nói "ở đâu".

### Tầng 3 — Calibration + abstention
Temperature scaling; đặt ngưỡng `τ_assert / τ_uncertain / τ_prog`. Dưới ngưỡng → **hedge** (ngôn ngữ dè dặt)
hoặc **abstain** ("nhường radiologist"). Không chắc thì nói không chắc — đây là một phần của faithfulness.

### Tầng 4 — Temporal guard (rủi ro #1)
- **Không prior → tắt SẠCH mọi ngôn ngữ thời gian (by construction).** Đây là chỗ VERA đạt temporal-halluc = 0
  *bất khả* chứ không phải "số nhỏ" — khác biệt cốt lõi với TRACE (sinh) và đám steering (giảm FilBERT).
- **Có prior → cụm tiến triển = readout argmax M4 (1:1)**, KHÔNG để LLM tự sinh từ "interval change".

### Tầng 5 — Realize (hiện thực hoá văn bản)
- **Template mặc định** — faithful tuyệt đối, đọc khô.
- **Constrained paraphraser (tùy chọn)** — LLM viết mượt nhưng **prose-from-table, KHÔNG prose-from-image**:
  nội dung lấy từ bảng, **cấm thêm/bớt finding**. (Đây là lời giải cho tranh luận prose-vs-template với đồng đội:
  cho phép prose, nhưng prose lấy nội dung *từ bảng verify được*, rồi qua tầng 6.)
- Ablation đẹp cho paper: **template vs constrained-paraphrase** — đo cả fluency lẫn out-of-table rate trên cùng bảng.

### Tầng 6 — Verify (round-trip + coverage)
- **Round-trip CheXbert/RadGraph:** report đã sinh → trích lại nhãn → **diff với bảng M3/M4** → loại mọi
  finding/nhãn trôi ra ngoài (out-of-table). Bắt các lỗi *thêm-thừa* (finding bịa, chẩn đoán bịa, thiết bị bịa,
  câu trái chiều M4).
- **Luật coverage:** report phải xử lý **mọi ô assert-positive** (assert hoặc hedge); thiếu ô nào → cờ bật.
  Bắt lỗi *bỏ-sót* (under-reporting) mà round-trip không bắt được.

### 5.x Visualization gắn với M5 (faithful-first, là readout trực tiếp)
- **Provenance per-câu** — mỗi mệnh đề trỏ về ô (vùng, bệnh, conf, M4); câu không truy được → đỏ. *Vừa là viz
  vừa là cơ chế bắt hallucination.*
- **Coverage map 29 vùng** — bình thường/bất thường/không chắc/không-đánh-giá-được → biến "im lặng" thành
  khẳng định verify được.
- **Confidence-as-typography** — assert đậm, hedge nhạt/dè dặt, omit xám mờ.
- **Change-ledger** prior→current (finding | prior | current | hướng) — readout argmax M4.
- Đường cong **deletion/insertion** mức vùng — visualize chính độ faithful (đặt làm phụ; đảm bảo cấp construction
  ở tầng 4 & 6 mới là claim faithfulness chính).

---

## Phụ chú — liên kết ngược các quyết định đã chốt
- Concept-bottleneck (Phase 3 hướng B/C) là thứ handoff **hoãn** và gắn go/no-go. Việc đang thử B/C **không sai**,
  nhưng phải đi kèm 3.4 (tiêu chí faithfulness) để không vô tình ship "why" plausible-mà-không-faithful — đúng cái
  bẫy VERA tồn tại để tránh.
- Hướng A luôn là **fallback faithful an toàn**. Đừng để accuracy cao của hướng C kéo bài rời trục readout-verify-được.
