# Phase 5 — Module 5: Faithful Report Assembler (kế hoạch cuối)

> Tài liệu bàn giao để hiện thực M5 trong một phiên chat khác. Viết theo style
> `phase_3_4.md`. **Đọc kỹ §0 trước khi code** — nó ghi lại các quyết định *và lý do*,
> để không vô tình thêm lại những thứ đã được cân nhắc rồi loại bỏ (LLM reasoner tự do,
> saliency heatmap, concept bottleneck).
>
> M5 = **Module 5** của pipeline KAN-TRaCE. Nó **không sinh ra chẩn đoán mới**. Nó là
> bộ **lắp ráp report trung thực** (faithful assembler) biến đầu ra có cấu trúc của
> M2/M3/M4 thành một report hướng bác sĩ, có grounding + calibration + abstention +
> chống temporal-hallucination + vòng verify.

---

## 0. Bối cảnh & triết lý thiết kế (ĐỌC TRƯỚC KHI CODE)

### 0.1 — Vì sao M5 KHÔNG phải "CoT LLM suy luận"

Ý tưởng ban đầu là dùng một LLM làm chain-of-thought để "suy luận" ra report. **Sai về bản
chất** trong pipeline này: M2/M3/M4 đã làm xong toàn bộ phần suy luận chẩn đoán —
*bệnh gì* (M3 `region_logits 29×14`), *ở đâu* (bbox + ROI-pool), *tiến triển ra sao*
(M4 `prog_logits 29×14×3`). Nếu M5 cố suy luận lại thì:
- Thừa (kết luận đã có sẵn ở thượng nguồn).
- **Hallucination cao**: cho LLM ăn feature + label rồi bảo "giải thích" → nó bịa ra
  bằng chứng nghe hợp lý nhưng không phản ánh lý do thật của model.

→ M5 chỉ *bề mặt hóa + tổng hợp + grounding + kiểm chứng* những gì đã quyết, **không sinh
finding mới**, **không tự diễn giải chẩn đoán**.

### 0.2 — Ràng buộc sống còn: LÚC LAUNCH KHÔNG CÓ REPORT

Đây là ràng buộc định hình toàn bộ M5. Khi triển khai thực, bác sĩ **rất có thể chỉ có
2 ảnh (prior + current), KHÔNG có report**. Hệ quả:

| Thứ | Lúc finetune/train | Lúc LAUNCH thật |
|-----|--------------------|-----------------|
| Ảnh prior + current | ✅ | ✅ |
| Feature BioViL-T (M1) | ✅ | ✅ |
| BBox 29 vùng (M2 detector) | ✅ | ✅ |
| M3 `region_logits[29,14]`, `image_logits[14]`, `region_mask[29]` | ✅ | ✅ |
| M4 `prog_logits[29,14,3]`, `pair_mask` | ✅ (nếu có prior) | ✅ (nếu có prior) |
| **Report** | ✅ | ❌ |
| **Scene graph attributes/relations (nhánh M2 LLM)** | ✅ (silver/pseudo) | ❌ |

**Nhánh attribute/relationship của M2 phụ thuộc report → KHÔNG tồn tại lúc launch.**
Vì vậy M5 **không được** dựa vào scene-graph attribute lúc inference. Mọi thứ M5 dùng phải
nằm trong: bbox + M3 logits + M4 logits + (tùy chọn) feature ảnh.

### 0.3 — Thứ bậc tin cậy (trust hierarchy) — nguyên tắc xếp hạng output

M5 trình bày thông tin theo thứ tự tin cậy giảm dần. Phần đáng tin = phần có grounding +
verify được; phần sinh tự do = phần rủi ro, bị chặn tối đa.

1. **Cấu trúc (cao nhất):** finding + vùng + bbox + confidence (đã calibrate) + tiến triển
   (readout 1:1 từ M4). Verify được từng ô.
2. **Grounding "ở đâu":** vùng nào dẫn dắt quyết định (đọc thẳng từ LSE softmax — exact).
3. **Calibration + abstention:** phơi bày độ bất định; không chắc → hedge hoặc nhường lại
   cho bác sĩ.
4. **(Tùy chọn) Văn xuôi:** chỉ là lớp realize, mặc định bằng template; nếu dùng LLM thì
   chỉ paraphrase có ràng buộc, không thêm/bớt finding.
5. **Verify (bao trùm):** round-trip CheXbert chặn mọi finding trôi ra ngoài bảng M3/M4.

### 0.4 — Bảng quyết định đã chốt (và lý do)

| Hạng mục | Quyết định | Lý do |
|----------|------------|-------|
| Vai trò M5 | Faithful assembler, **không** phải reasoner/generator tự do | M2/M3/M4 đã suy luận xong; sinh tự do = hallucination |
| Output mặc định | **Cấu trúc** (findings list), văn xuôi là lớp phủ tùy chọn | Bác sĩ verify từng claim dễ hơn trên cấu trúc; văn mượt che chỗ không chắc |
| Nguồn "ở đâu" | `softmax_r(region_logit[r,d])` từ LSE — **exact, faithful** | Model đã self-explaining mức vùng; không cần LIME/SHAP |
| Saliency heatmap | **LOẠI** | Saliency dễ unfaithful; ROI-pool không sinh saliency nội-box đáng tin; "vùng nào" đã đủ + free |
| LIME / SHAP | **LOẠI** | Thừa với region attribution sẵn có; trên feature mờ thì không đọc được "vì sao" |
| Concept bottleneck (ante-hoc) | **HOÃN/LOẠI cho v1** | Là đường duy nhất cho "why" faithful, NHƯNG cắm nhãn concept nhiễu (silver + pseudo step-7 do LLM sinh) vào *đường nhân quả* → đúng cái hallucination ta tránh, quay lại dạng nhãn train |
| "Vì sao = dấu hiệu" | **Descope có ý thức** → "ở đâu + bác sĩ tự đọc dấu hiệu trên box" | Không post-hoc nào trên feature mờ cho "why-sign" faithful; phân vai: model chỉ chỗ, bác sĩ đọc |
| LLM trong M5 | Chỉ làm **constrained paraphraser** (nếu cần văn xuôi), không reasoner | Văn mượt là cosmetic; không để nó reintroduce hallucination đã loại |
| Visual prompting (mark-rồi-VLM) | **Không dùng cho v1** | Là công cụ sinh/grounding, không phải faithfulness; lại trao quyền cho VLM tự do + nghiêng về *thay thế* M3/M4 |
| Temporal (so sánh prior) | Gate cứng; readout 1:1 từ M4; không prior → tắt sạch ngôn ngữ thời gian | Temporal hallucination là rủi ro #1 (M4 temporal); model hay bịa "stable/unchanged" |
| Confidence | σ(M3)/softmax(M4) **sau temperature scaling** | Logit thô overconfident; ngưỡng abstain phải đặt trên prob đã calibrate |

---

## 1. Hợp đồng I/O (I/O contract)

### 1.1 — Đầu vào M5 (tất cả có lúc launch)

```python
inputs = {
  # Từ M3 (C-KAN) — bắt buộc
  "region_logits":  Tensor[29, 14],   # logit bệnh theo vùng (pre-sigmoid)
  "image_logits":   Tensor[14],       # = LSE_r(region_logits) có mask
  "region_mask":    Tensor[29],       # 1 = vùng tồn tại (bbox hợp lệ), 0 = thiếu/sentinel
  "boxes":          Tensor[29, 4],    # bbox theo không gian resized 512-short-side
  "boxes_original": Tensor[29, 4],    # bbox pixel gốc (để overlay cho bác sĩ)

  # Từ M4 (T-KAN) — chỉ khi có prior
  "has_prior":      bool,
  "prog_logits":    Optional[Tensor[29, 14, 3]],  # 0=improved,1=stable,2=worsened
  "pair_mask":      Optional[Tensor[29, 14]],     # 1 = ô (vùng,bệnh) được giám sát/hợp lệ

  # Tham số calibrate (fit sẵn trên val, xem §6)
  "temp_disease":   float,            # nhiệt độ cho M3 (hoặc per-disease vector[14])
  "temp_prog":      float,            # nhiệt độ cho M4
  "thresholds":     dict,             # ngưỡng assert/hedge/omit (per-disease nếu cần)

  # Metadata hằng số
  "region_names":   List[str](29),    # khớp REGION_NAMES / dataset.yaml
  "label_names":    List[str](14),    # khớp LABEL_NAMES / label.json
}
```

**Không có lúc launch (KHÔNG được phụ thuộc):** report, scene-graph attributes/relations.

### 1.2 — Đầu ra M5

Một **report object có cấu trúc** (xem schema §5) gồm: danh sách finding (mỗi cái có
vùng, bbox, confidence calibrate, nhãn hedge/assert, tiến triển nếu có), danh sách vùng
được flag để bác sĩ xem, và (tùy chọn) một chuỗi văn xuôi đã qua verify. Văn xuôi **không
bao giờ** chứa finding ngoài danh sách cấu trúc.

---

## 2. Kiến trúc M5 — sơ đồ tầng

```
                M3 (29×14, image_logits, mask, boxes)   M4 (29×14×3, pair_mask, has_prior)
                                 │                                  │
                                 ▼                                  │
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ TẦNG A — Structured findings core                                          │
  │   calibrate → chọn finding (assert/hedge/omit) theo ngưỡng                 │
  │   mỗi finding: {disease, region, bbox, p_cal, status}                      │
  └──────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ TẦNG B — Grounding "ở đâu" (faithful, free)                                │
  │   region attribution = softmax_r(region_logit[r,d])                        │
  │   (tùy chọn) region counterfactual: ablate vùng → call lật?                │
  └──────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ TẦNG C — Calibration + abstention                                          │
  │   p_cal = σ(logit / T); band assert/hedge/omit; cờ "nhường radiologist"    │
  └──────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ TẦNG D — Temporal guard (rủi ro #1)                                        │
  │   has_prior=False → tắt MỌI ngôn ngữ thời gian                             │
  │   has_prior=True  → progression = argmax M4 (readout 1:1), chỉ ô pair_mask │
  └──────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ TẦNG E — Realize (tùy chọn)                                                │
  │   mặc định: TEMPLATE (faithful tuyệt đối)                                  │
  │   tùy chọn: LLM constrained paraphraser (không thêm/bớt finding)           │
  └──────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ TẦNG F — Verify (round-trip)                                               │
  │   CheXbert(report) → 14-vector → so với finding đã assert → cờ/loại lệch   │
  └──────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                        report object (§5) → UI bác sĩ
```

---

## 3. Chi tiết từng tầng

### TẦNG A — Structured findings core

Với mỗi (vùng r, bệnh d) sao cho `region_mask[r] == 1`:

```
p_region[r,d]  = sigmoid(region_logits[r,d] / temp_disease)
p_image[d]     = sigmoid(image_logits[d]   / temp_disease)
```

- Quyết định **present/absent** ở **mức ảnh** dùng `p_image[d]` so với ngưỡng (§3-C).
- Với mỗi bệnh được giữ, gán **vùng đại diện** = `argmax_r region_logits[r,d]` (chỉ trong
  vùng `region_mask=1`) → đó là vùng để overlay bbox cho bác sĩ.
- Bỏ `region_mask=0` ra khỏi mọi tính toán (vùng thiếu/sentinel).

Mỗi finding tạo ra một record sơ bộ: `{disease, region, bbox, bbox_original, p_image, p_region, status(chờ §3-C)}`.

### TẦNG B — Grounding "ở đâu" (faithful, free)

**Region attribution (bắt buộc, exact).** Vì `image_logit[d] = LSE_{r:mask}(region_logit[r,d])`,
phần đóng góp của vùng r vào quyết định mức-ảnh **chính xác** bằng softmax có mask:

```
attr[r,d] = exp(region_logit[r,d]) / Σ_{r':mask[r']=1} exp(region_logit[r',d])
```

→ "Bệnh d được dẫn dắt bởi vùng nào, bao nhiêu %." Đây là attribution thật của phép LSE,
**không phải xấp xỉ** — không cần LIME/SHAP.

**Region counterfactual (tùy chọn, rẻ, faithful).** Để khẳng định "call bản lề ở vùng nào":

```
for r in regions_with_mask:
    masked = region_mask.clone(); masked[r] = 0
    image_logit_ablated[d] = LSE_{r':masked}(region_logit[r',d])
    if present(image_logit[d]) and not present(image_logit_ablated[d]):
        → vùng r là "bản lề" cho bệnh d
```

Trung thực vì là perturbation thật trên chính phép tính của model. Dùng để đánh dấu vùng
ưu tiên cho bác sĩ xem.

> **KHÔNG làm:** saliency mức pixel, gradient attribution nội-box, LIME, SHAP. (Xem §0.4.)

### TẦNG C — Calibration + abstention

**Calibration:** mọi prob phải qua temperature scaling (fit theo §6). Đặt ngưỡng trên prob
**đã calibrate**, không trên logit thô.

**Band 3 mức (per-disease nếu lệch nhãn nặng):**

```
p_cal >= τ_assert[d]              → status = "assert"   (khẳng định)
τ_uncertain[d] <= p_cal < τ_assert[d] → status = "hedge"  (possible / cannot exclude)
p_cal <  τ_uncertain[d]           → status = "omit"     (không nhắc / coi như âm)
```

**Cờ nhường radiologist:** đặt `flag_review = True` khi finding rơi vào "hedge" **và** là
bệnh nguy hiểm/khó (danh sách cấu hình được), hoặc khi confidence mức-vùng không đáng tin
(xem rủi ro §10). Khi đó UI ghi rõ "model không chắc — cần bác sĩ xem vùng [bbox]".

Ánh xạ ngôn ngữ hedge (dùng ở tầng E): `assert → "<finding> tại <region>"`;
`hedge → "possible <finding> tại <region>, cannot be excluded"`.

### TẦNG D — Temporal guard (rủi ro #1)

Temporal hallucination (bịa "stable/unchanged/new so với prior") là kiểu lỗi phổ biến nhất
trong RRG và là rủi ro lớn nhất của bạn vì M4 temporal. Quy tắc cứng:

```
if not has_prior:
    → KHÔNG phát BẤT KỲ ngôn ngữ so sánh thời gian nào.
       (không "stable", "unchanged", "new", "improved", "worsened", "compared to prior")
       Report chỉ mô tả trạng thái hiện tại.
else:
    for mỗi finding đã assert/hedge ở (r,d):
        if pair_mask[r,d] == 1:
            p3 = softmax(prog_logits[r,d,:] / temp_prog)
            cls = argmax(p3); conf = max(p3)
            if conf >= τ_prog:
                progression = {0:"improved", 1:"stable", 2:"worsened"}[cls]   # readout 1:1
            else:
                progression = None   # không đủ chắc → không nói tiến triển
        else:
            progression = None       # ô không được giám sát → không nói tiến triển
```

**Từ tiến triển phải là readout argmax của M4, KHÔNG để LLM tự sinh.** LLM (nếu có ở tầng E)
chỉ được diễn đạt lại đúng nhãn này.

### TẦNG E — Realize (tùy chọn)

**Mặc định: TEMPLATE (faithful tuyệt đối).** Sinh report cấu trúc/bán-cấu-trúc thẳng từ
records. Ví dụ một dòng finding:

```
[<Disease>] <region>, <status-phrasing>, confidence <p_cal>[, <progression> so với prior].
```

Bác sĩ theo dõi từng dòng, mỗi dòng gắn bbox overlay. Đây là sản phẩm chính, đủ dùng,
zero hallucination.

**Tùy chọn: LLM constrained paraphraser** (chỉ khi cần report văn xuôi theo quy ước lâm sàng).
Hợp đồng cứng cho LLM:
- **Input:** danh sách finding đã chốt (text template) + clinical context nếu có.
- **KHÔNG** đưa ảnh raw cho LLM (tránh nó "thấy" và bịa thêm).
- **System prompt (ý chính):** "Bạn diễn đạt lại danh sách finding dưới đây thành văn xuôi
  lâm sàng. TUYỆT ĐỐI không thêm finding mới, không bỏ finding nào, không đổi vùng, không
  đổi từ tiến triển. Chỉ thay đổi cách hành văn."
- **Decoding:** có thể ép coverage (mỗi finding present xuất hiện ≥1 lần).
- **Bắt buộc qua tầng F sau khi sinh.**

> LLM ở đây là *renderer*, không phải *reasoner*. Nếu phiên hiện thực thấy mình đang cho LLM
> "suy nghĩ thêm" → dừng lại, đó là vi phạm §0.1.

### TẦNG F — Verify (round-trip, bao trùm)

```
labels_from_report = CheXbert(report_text)        # 14-vector (pos/neg/uncertain/blank)
asserted = {d : status[d] in ("assert","hedge")}
for d in 14:
    if labels_from_report[d] == positive and d not in asserted:
        → HALLUCINATION (finding trôi ra ngoài bảng) → loại câu / regenerate
    if d in asserted and labels_from_report[d] != positive:
        → DROP/OMISSION (finding bị đánh rơi khi realize) → cờ
```

Với template output, vòng này gần như luôn pass (report tất định từ nhãn). Với LLM-paraphrase,
nó bắt drift. **Không pass verify → không xuất report**, rơi về template.

---

## 4. Những gì M5 KHÔNG làm (non-goals — đừng vô tình thêm lại)

- ❌ **Không** chẩn đoán lại / không sinh finding mới. M5 chỉ lắp ráp cái M3/M4 đã quyết.
- ❌ **Không** dùng LLM làm reasoner/CoT tự do. LLM chỉ paraphrase có ràng buộc.
- ❌ **Không** saliency heatmap, **không** LIME/SHAP, **không** gradient attribution nội-box.
- ❌ **Không** concept bottleneck ở v1 (lý do: nhãn concept nhiễu cắm vào đường nhân quả).
- ❌ **Không** phụ thuộc scene-graph attribute lúc launch (không có report).
- ❌ **Không** visual prompting / mark-rồi-VLM ở v1.
- ❌ **Không** khẳng định "vì sao = dấu hiệu X" như sự thật của model. "Vì sao" = "ở đâu
  (faithful) + bác sĩ tự đọc dấu hiệu trên box". Phân vai: model chỉ chỗ, bác sĩ đọc dấu hiệu.

---

## 5. Schema output (để hiện thực không mơ hồ)

```python
@dataclass
class Finding:
    disease: str                 # ∈ LABEL_NAMES (14)
    region: str                  # ∈ REGION_NAMES (29) — vùng đại diện
    bbox: list[float]            # 4 — không gian resized 512
    bbox_original: list[float]   # 4 — pixel gốc (để overlay)
    p_cal: float                 # confidence đã calibrate (mức ảnh)
    status: str                  # "assert" | "hedge"
    region_attr: float           # softmax_r(region_logit[r,d]) cho vùng đại diện
    is_hinge: bool               # từ region counterfactual (tùy chọn)
    progression: str | None      # "improved"|"stable"|"worsened"|None
    prog_conf: float | None
    flag_review: bool            # nhường radiologist

@dataclass
class M5Report:
    findings: list[Finding]
    review_regions: list[str]    # vùng cần bác sĩ xem kỹ
    has_prior: bool
    prose: str | None            # văn xuôi đã qua verify (nếu bật tầng E LLM)
    verify_passed: bool
```

---

## 6. Calibration & ngưỡng (cách đặt)

- **Temperature scaling:** trên tập **val** của M3/M4, tìm `T` tối thiểu hóa NLL của
  `σ(logit/T)` (M3) / `softmax(logit/T)` (M4) so với nhãn. Một scalar mỗi head; cân nhắc
  per-disease `T[14]` cho M3 vì lệch nhãn. Đây là 1 bước nhẹ, **không train lại model**.
- **Ngưỡng `τ_assert / τ_uncertain / τ_prog`:** chọn trên val theo mục tiêu lâm sàng. Gợi ý:
  ưu tiên **precision cao cho assert** (giảm dương tính giả — bịa bệnh là tệ nhất với bác sĩ
  ít kinh nghiệm), recall đẩy sang band "hedge". Đặt per-disease vì thưa/lệch nhãn.
- **(Tùy chọn nâng cao) Conformal prediction:** thay confidence trần bằng tập dự đoán có
  coverage đảm bảo (vd 90%). Hợp bối cảnh lâm sàng. Để sau v1.

---

## 7. Đánh giá / metrics

| Nhóm | Metric | Đo gì |
|------|--------|-------|
| Lâm sàng | CheXbert 14/5 Macro-F1, Micro-F1 (vs reference khi có) | Độ đúng nội dung |
| Faithfulness/consistency | **Round-trip agreement rate** (tầng F) | % finding trong report khớp bảng M3/M4 |
| Hallucination | **Out-of-table rate** | % câu nói finding KHÔNG có trong M3/M4 (phải ≈ 0) |
| Temporal | **Temporal-hallucination rate** | % câu so sánh thời gian khi `has_prior=False` (phải = 0) |
| Calibration | **ECE** (Expected Calibration Error) trước/sau temperature scaling | Prob có khớp accuracy |
| Abstention | **Coverage / risk** | Trade-off giữa % ca dám assert và lỗi trên phần assert |
| Grounding | (nếu có nhãn box) IoU vùng đại diện vs GT | "Ở đâu" có đúng không |

Mục tiêu cốt lõi: **out-of-table rate ≈ 0** và **temporal-hallucination rate = 0**. Đây là
hai con số phải khoe trong báo cáo — chúng định nghĩa "đáng tin" của M5.

---

## 8. Phụ thuộc & file dự kiến (mirror style phase_3/4)

```
src/phase_5/
  __init__.py
  constants.py        # REGION_NAMES(29), LABEL_NAMES(14), PROG_CLASS_NAMES, ngưỡng mặc định
  calibrate.py        # fit temperature scaling trên val; lưu temp_disease/temp_prog
  attribution.py      # softmax_r(region_logit) + region counterfactual (ablate vùng)
  assemble.py         # tầng A→D: chọn finding, grounding, calibrate/abstain, temporal guard
  realize.py          # tầng E: template (mặc định) + LLM constrained paraphraser (tùy chọn)
  verify.py           # tầng F: CheXbert round-trip, out-of-table / omission check
  report.py           # dataclass Finding / M5Report, serialize JSON
  run_m5.py           # pipeline: nạp M3/M4 output → M5Report; CLI
  verify_m5.py        # smoke test trên vài ca (giống verify_shapes/verify_tkan)
```

Phụ thuộc: checkpoint M3 (`runs/phase_3/ckan_best.pt`), M4 (`runs/phase_4/`), **CheXbert
labeler** (đã có trong pipeline cho M2/`FINDING_TO_LABEL`), tập val để calibrate. Env `chex`,
`KMP_DUPLICATE_LIB_OK=TRUE PYTHONNOUSERSITE=1`.

---

## 9. Thứ tự hiện thực (cho phiên sau)

1. `constants.py` + `report.py` (schema) — chốt I/O trước.
2. `calibrate.py` — fit temperature scaling trên val; verify ECE giảm.
3. `attribution.py` — region attribution (LSE softmax) + counterfactual; test exact trên ca dummy.
4. `assemble.py` — tầng A→D (chọn finding, grounding, calibrate/abstain, **temporal guard**).
   Test riêng `has_prior=False` ⇒ 0 câu thời gian.
5. `realize.py` — **template trước** (faithful), chạy trọn vòng ra report cấu trúc.
6. `verify.py` — round-trip CheXbert; đo out-of-table rate (template phải ≈ 0).
7. (Tùy chọn) LLM paraphraser trong `realize.py` + bắt buộc qua `verify.py`; so out-of-table
   rate template vs LLM.
8. `run_m5.py` + `verify_m5.py` — wire end-to-end; metrics §7.

> Nguyên tắc: **ship bản template + verify trước**, đo 2 con số cốt lõi (out-of-table = 0,
> temporal-halluc = 0). Chỉ thêm LLM paraphraser khi bản cấu trúc đã chạy sạch.

---

## 10. Rủi ro đã biết & quyết định liên quan

1. **Temporal hallucination = rủi ro #1.** M4 temporal đẩy thẳng vào đó. Tầng D là bắt buộc,
   không phải tùy chọn. `has_prior=False` ⇒ tắt sạch ngôn ngữ thời gian.
2. **Confidence mức-vùng còn yếu** đến khi M3 có **giám sát region-level** (hiện là TODO của
   phase 3 — supervision mới ở image-level qua LSE). Trước khi đó: dùng `p_image` để quyết
   present/absent (đáng tin hơn), `p_region`/`attr` chỉ để *chỉ vùng* cho bác sĩ, không để
   khẳng định mạnh. Khi thêm region supervision thì nâng cấp.
3. **Giới hạn "why" có ý thức.** M5 v1 không trả lời được "vì sao = dấu hiệu" một cách
   faithful — đây là quyết định, không phải thiếu sót. Lý do: post-hoc trên feature mờ không
   cho why-sign faithful; concept bottleneck cho được nhưng cắm nhãn nhiễu vào nhân quả.
4. **Cổng go/no-go nếu muốn xét lại concept bottleneck (tương lai, ngoài v1):** trước khi
   cam kết, đo F1 của một concept head đoán-concept-từ-ảnh trên **silver MIMIC** (tập sạch
   nhất). Concept tự nó đoán tệ ⇒ bottleneck sẽ khuếch đại cái tệ lên tận quyết định bệnh ⇒
   không làm. Concept đoán tốt ⇒ mới cân nhắc nâng M3 thành dạng `feature → concept (supervise)
   → [bottleneck, không skip] → bệnh`, KAN đặt ở bước concept→bệnh.

---

## 11. TL;DR cho phiên hiện thực

M5 **không sinh chẩn đoán, không suy luận tự do**. Nó **lắp ráp** report từ M3/M4:
chọn finding (calibrate + abstain) → grounding "ở đâu" bằng LSE softmax (exact) →
temporal guard cứng (M4 readout 1:1, không prior thì câm thời gian) → realize bằng
**template** (LLM chỉ paraphrase có ràng buộc nếu cần) → **round-trip CheXbert verify**.
Hai con số phải đạt: **out-of-table rate ≈ 0**, **temporal-hallucination rate = 0**.
Phần "vì sao" được descope thành "ở đâu (faithful) + bác sĩ đọc dấu hiệu trên box".
Không saliency, không LIME/SHAP, không concept bottleneck, không visual prompting ở v1.
