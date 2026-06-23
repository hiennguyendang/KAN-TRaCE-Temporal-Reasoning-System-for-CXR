# Phase 3 & 4 — Bản cập nhật kiến trúc MLP (C-MLP / T-MLP)

> Phần này **bổ sung/ghi đè** thiết kế gốc ở `phase_3_4.md` (vốn dùng `ROI-pool mean + KAN`).
> Mục tiêu: (1) sửa hai điểm yếu của ROI-pool mean — *mất chi tiết focal* và *mất thông tin
> global*; (2) chuyển head mặc định sang **MLP** (giữ KAN làm ablation) vì bằng chứng cho thấy
> trên task vision/classification MLP nhìn chung ngang-hoặc-hơn KAN và ổn định hơn.
> **Quyết định cuối:** đường vùng dùng **attention-pool**, thêm **nhánh global**, **giữ neck**,
> head **MLP** (KAN = ablation). Để dữ liệu quyết KAN-vs-MLP qua infra parity sẵn có.

---

## 0. Vì sao đổi (tóm tắt lý do)

| Vấn đề ở thiết kế gốc | Bằng chứng / lý do | Cách sửa ở bản này |
|----------------------|--------------------|--------------------|
| Mean-ROI-pool **làm loãng finding focal nhỏ** (nốt trong box to bị chia ~1/N) | Mean dồn đều theo diện tích | **Attention-pool** (trọng số học được, dồn vào ô "nóng") |
| ROI theo 29 vùng **mất finding quan hệ/lan toả** (cardiomegaly = tỉ lệ tim/lồng ngực) | Finding quan hệ không nằm trong 1 box | **Nhánh global** song song, gộp ở image-logit |
| KAN ở reasoning layer **chưa chắc lợi** | "KAN or MLP: A Fairer Comparison": ngoài task ký hiệu, MLP nhìn chung ≥ KAN; KAN mong manh (phân kỳ theo seed, bất ổn khi stack). Interpretability của KAN chỉ hiện khi input là **concept có tên** — mà v1 không có concept layer (feature mờ) | **MLP làm default**; **KAN giữ làm ablation** (đã có infra `KANHead↔MLPHead`); KAN chỉ "lên ngôi" nếu thắng trên dữ liệu hoặc khi làm concept bottleneck sau này |
| FastKAN còn **to hơn** MLP trong cấu hình hiện tại | Verify cũ: KAN 148k vs MLP 75k (M3); KAN 672k vs MLP 206k (M4) | MLP rẻ + nhanh hơn |

> Lưu ý: KAN **đã** được dùng cho phân loại ảnh y tế (Taylor-KAN, MedKAN, DEQ-KAN, KANBalance trên CXR),
> nên KAN không phải primitive "chưa kiểm chứng". Cái **mới** của ta là *tổ hợp*
> (attention-pool 29 giải phẫu + head reasoning + temporal head feeding faithful assembler M5),
> nên hãy de-risk bằng cách **ablate** từng lựa chọn thay vì cược mù.

---

## 1. Đường vùng — Attention-pool (thay mean-ROI-pool)

Thay trung bình-có-trọng-số-diện-tích bằng **trung bình có trọng số *học được*** (cross-attention
kiểu DETR: query = vùng, key/value = ô lưới, mask theo box):

```
Input: feats [B,196,512], boxes [B,29,4]
- Query:  29 region query vectors  Q [29, d]   (nn.Parameter học, hoặc embed theo region-id/box)
- Key/Value: K = W_k · feats,  V = W_v · feats   (projection của 196 ô)
- Mask M[r,c] = 1 nếu ô c (tâm ô) nằm trong / overlap box r, else 0   (dựng từ boxes, như roi_pool cũ)
- Attention:
    score[r,c] = (Q_r · K_c) / sqrt(d)
    score[r,c] = -inf nếu M[r,c] = 0
    α[r,c]     = softmax_c(score[r,c])           # CHỈ trên ô trong box r
    pooled_r   = Σ_c α[r,c] · V_c
- region_mask[r] = 1 nếu box r hợp lệ (≠ sentinel (0,0,0,0) và w,h>0), else 0
                   (vùng mask=0 → pooled_r = 0, loại khỏi head & aggregation)
Output: region_pooled [B,29,512], region_mask [B,29], α [B,29,196]
```

**Lợi ích kép:**
- Cứu tín hiệu focal: 1 ô nóng có thể nhận α≈1 thay vì bị chia đều.
- **`α` là attribution nội-vùng *faithful*** — vì nó *chính là* trọng số pool thật, không phải xấp xỉ
  như gradient-saliency (đã loại ở kế hoạch M5). → cung cấp thêm grounding cho M5: "trong vùng phổi
  phải, quyết định bị dẫn dắt bởi vùng con này". (Nói có chừng: đây là attention *pooling* nên trọng số
  thực sự quyết định đóng góp — vững hơn "attention = explanation" trong transformer.)

**Cài đặt:** single-head để bắt đầu (vài head sau nếu cần); LN + residual chuẩn. Đây là thay thế cho
phần pooling trong `roi_pool.py` (giữ logic dựng mask overlap từ box).

### 1b. (Tùy chọn / ablation) Predict-then-pool

Đảo thứ tự pool & phân loại — head nhỏ chạy **per-ô** rồi **max-pool theo box**:

```
cell_logits [196,14] = SharedCellHead(feats)            # MLP nhỏ trên từng ô
region_logits[r,d]   = max_{c ∈ box r} cell_logits[c,d] # max → 1 ô nóng kéo cả vùng
```

- **Lợi:** giữ focal mạnh nhất + tặng bản đồ mức-ô mịn (localization mịn hơn 29 vùng).
- **Hại (lý do KHÔNG làm đường chính):** cho ra region **logit**, KHÔNG cho region **feature** →
  **T-MLP hụt input** (Siamese cần feature 29×128). Mỗi ô lại thiếu ngữ cảnh lân cận.
- → Để dành làm **head phụ / ablation** nếu muốn cell-map, không thay attention-pool.

---

## 2. Nhánh global (bắt finding quan hệ/lan toả)

Đường vùng lo finding khu trú; nhánh global lo cardiomegaly / edema lan toả / low lung volumes —
những thứ **không nằm trong một box**.

```
global_feat [B,512] = pool TOÀN BỘ 196 ô
    - mặc định đơn giản: Global Average Pool
    - tùy chọn: attention với 1 global query (không mask) → có α_global
global_logits [B,14] = GlobalHead-MLP(global_feat)        # [512 → h → 14]
```

**Gộp với đường vùng ở mức image-logit** (để model tự học, không cần route tay):

```
region_image_logits[d] = LSE_{r:mask}(region_logits[r,d])
image_logits[d]        = region_image_logits[d] + g_d · global_logits[d]
                         # g_d = cổng học được per-disease (sigmoid), hoặc concat[28]→linear[14]
```

- Không cần bbox (pool toàn ảnh); **không nuôi T-MLP**.
- Thành thật: bệnh quyết nhờ nhánh global không "chỉ vào 1 box" được — đúng bản chất finding quan hệ,
  không phải lỗi. (Để hiển thị vẫn có thể overlay box giải phẫu liên quan, nhưng *quyết định* dùng
  ngữ cảnh toàn ảnh.)

---

## 3. Neck — GIỮ LẠI (lý do đổi)

| | Thiết kế gốc (KAN) | Bản MLP này |
|--|--------------------|-------------|
| Lý do tồn tại của neck | Giảm 512→128 cho KAN khỏi nuốt 512 | MLP ăn 512 trực tiếp được → lý do (a) mất |
| Lý do còn lại | Cấp feature gọn cho T-KAN | **Vẫn còn + quan trọng hơn:** giữ input T-MLP gọn + LayerNorm ổn định |

**Quyết định: giữ `Neck = Linear(512→128) + LN + GELU`.** `region_feat[128]` là biểu diễn **chia sẻ**
cho cả disease head lẫn T-MLP. Nếu bỏ neck (feed 512 thẳng vào head) thì T-MLP concat 3×512+28 = 1564
chiều thay vì 412 → nặng hơn nhiều và mất chuẩn hoá. → neck giờ được biện minh bằng "feature vùng gọn +
chuẩn hoá", không phải vì KAN.

---

## 4. C-MLP (Module 3) — sơ đồ shape đầy đủ

```
feats [B,196,512] (BioViL-T)  +  boxes [B,29,4]
  │
  ├─ ĐƯỜNG VÙNG ───────────────────────────────────────────────────────────────
  │   attention-pool (29 query × 196 ô, mask theo box)
  │     → region_pooled [B,29,512] + region_mask [B,29] + α [B,29,196]
  │   Neck = Linear(512→128) + LN + GELU
  │     → region_feat [B,29,128]              ◄── nuôi disease head LẪN T-MLP
  │   Disease head = MLP dùng chung 29 vùng [128 → 64 → 14]
  │     → region_logits [B,29,14]
  │   LSE aggregate (mask)
  │     → region_image_logits [B,14]
  │
  └─ ĐƯỜNG GLOBAL ─────────────────────────────────────────────────────────────
      GAP toàn 196 ô → global_feat [B,512]
      GlobalHead-MLP [512 → 128 → 14] → global_logits [B,14]

  COMBINE:  image_logits [B,14] = region_image_logits + gate ⊙ global_logits
            → masked BCE (giữ -100) với nhãn ảnh, có pos_weight/focal (xem §7)
```

Đầu ra trả về: `image_logits[14]`, `region_logits[29,14]`, `region_feat[29,128]`, `region_mask[29]`,
`α[29,196]`, `global_logits[14]`. (region_feat + α phục vụ M4 và M5.)

---

## 5. T-MLP (Module 4) — nhận vào cái gì

**Y hệt hợp đồng của T-KAN cũ**, chỉ đổi head KAN→MLP. Input = **`region_feat[29,128]` từ C-MLP**
(không phải global_feat), cho cả study prior và current:

```
Siamese (C-MLP dùng chung trọng số):
   prior_feat [29,128] ; curr_feat [29,128] ; merged = curr − prior [29,128]
per-region concat:
   [prior ; curr ; merged ; prior_labels ; curr_labels]
   region_in_dim = 412 = 128×3 + 14×2
Head = MLP [412 → 128 → 42]  → prog_logits [B,29,14,3]
   → masked CE (ignore_index=-100, bỏ ô pair_mask=0)
```

- Nhãn: GT lúc train / C-MLP pred lúc infer (teacher forcing — như cũ).
- 3 lớp: `0=improved, 1=stable, 2=worsened`.
- **Nhánh global KHÔNG đụng T-MLP** — progression là per-region; cardiomegaly nằm ở vùng "cardiac
  silhouette" (1 trong 29) nên region_feat của nó đã mang.
- Cờ tùy chọn vẫn giữ: `use_hadamard` (+128), `no_labels` (−28), `label_input gt|pred`.

---

## 6. Head MLP — không cần loại đặc biệt

- Kiến trúc: `Linear → LN → GELU → Linear`, **dùng chung 29 vùng** (áp trên trục vùng, batch tự nhiên).
  `MLPHead` này **đã tồn tại** trong `heads.py` (qua `build_head`) → đổi default là việc cấu hình.
- **KHÔNG cần** MLP đặc biệt. Cái quyết định chất lượng là **ba đòn bẩy ở §7** (imbalance ≫ attention-pool
  > norm/reg), không phải kiểu MLP. Đừng cầu kỳ hóa MLP.

---

## 7. Ba đòn bẩy chất lượng (cái quyết định, KHÔNG phải kiểu MLP)

> Với pipeline này, ba thứ dưới **không ngang nhau**. Ưu tiên: **§7.1 imbalance ≫ §7.2 attention-pool
> > §7.3 norm/reg**. Nếu chỉ làm tốt một thứ, làm §7.1 — vì nó quyết recall lớp hiếm (M3) và các lớp
> minority improved/worsened (M4), đúng phần lâm sàng quan trọng. Và **đổi metric sang macro-F1 +
> per-class** (xem §7.4) để biết các đòn bẩy có ăn thua không.

### 7.1 — Mất cân bằng (đòn bẩy #1, sống còn)

**Vì sao giết chất lượng ở đúng chỗ quan trọng:**
- M3 đa nhãn cực lệch: "No Finding"/Support Devices/Lung Opacity/Cardiomegaly đông; Pneumonia/
  Consolidation/Pleural Other/Lung Lesion/Fracture hiếm. Hệ quả thấy rõ ở RADAR Table 7: Support
  Devices F1 0.844 nhưng Consolidation 0.237, Pleural Other 0.228, Lung Lesion 0.291. Model **học bỏ
  qua finding hiếm** → recall thấp đúng bệnh ít gặp nhưng quan trọng.
- Cấu trúc 29×14 làm tệ hơn: hầu hết ô là âm (finding chỉ ở vài vùng). Nếu thêm giám sát region-level
  (TODO) thì tỉ lệ dương/âm mỗi ô cực thấp.
- M4 lệch nặng về "stable": verify cho acc≈0.51 ≈ luôn đoán stable → **vô dụng** dù accuracy nghe ổn.
- Tương tác `-100`: masked BCE không coi "không nhắc" = âm (tốt), nhưng tỉ lệ dương/âm *trong ô được
  giám sát* vẫn lệch.

**Cách giải quyết (xếp theo hiệu quả/đơn giản):**
1. **`pos_weight` log-scale (M3) — làm trước.** Công thức RADAR `α_i = log(1 + |D_train|/w_i)`,
   `w_i` = tần suất bệnh i. Log-scale **chặn trần** nên lớp siêu hiếm không thổi loss bay. Truyền
   `pos_weight[14]` vào `BCEWithLogitsLoss`. **Lưu ý mask:** đếm `w_i` chỉ trên ô *không bị mask*
   (bỏ -100), và áp weight cũng chỉ trên ô không mask.
2. **Focal loss** nếu recall lớp hiếm vẫn kém: hạ trọng số ví dụ dễ (γ=2 chuẩn), bản α-balanced ghép
   class weight; áp sigmoid-focal per class. Có thể kết hợp pos_weight + focal.
3. **Tune ngưỡng per-disease (rẻ, hiệu quả to, hay bị quên).** Ngưỡng 0.5 mặc định *sai* cho lớp lệch.
   Tinh chỉnh `τ[d]` trên val (tối đa F1 / cố định precision). Thường lợi hơn cả trò loss xịn. Đây
   chính là `τ_assert/τ_uncertain` của kế hoạch M5 → nối thẳng vào calibration/abstention.
4. **M4:** class-weight nghịch tần suất (hoặc "effective number" Cui 2019) cho CE 3 lớp.
5. **Cảnh báo:** đừng over-weight dương tính → đẻ false positive. Với hệ hướng bác sĩ, FP là tệ nhất
   (gây tin nhầm). Tune để ca biên rơi vào band "hedge" của M5, đừng thành dương tính tự tin.

### 7.2 — Attention-pool (đòn bẩy #2, nuôi mọi thứ phía sau)

**Vì sao ảnh hưởng chất lượng:** module học-được, có-thể-bất-ổn **duy nhất** (mean-ROI vốn không tham
số, ổn định). Hỏng 3 kiểu: (a) **collapse về đều** → thoái hoá thành mean, mất lợi ích; (b) **over-focus**
vào 1 ô nhiễu → variance cao; (c) **bất ổn train** (softmax bão hoà). Vì `region_feat` nó sinh ra **nuôi
cả disease head LẪN T-MLP**, pool tồi đầu độc cả hai module.

**Cách giải quyết:**
- **Scaled dot-product** (chia `√d`) — bắt buộc, tránh softmax bão hoà.
- **Init để khởi đầu ≈ mean:** q/k/v nhỏ/near-identity → α ban đầu gần đều (≈ mean-pool, baseline đã biết
  tốt), rồi học **sharpen dần**. Điểm xuất phát ổn định.
- **Mask đúng:** mỗi query vùng chỉ attend ô trong box (gán `-inf` ngoài box, **trước** softmax). Vùng
  không có box hợp lệ → output 0 + `region_mask=0`; **guard NaN** khi softmax trên tập rỗng.
- **Single-head để bắt đầu** — đơn giản + **giữ α là attribution đọc được** (multi-head phải bình quân,
  mờ tính faithful). Multi-head để sau.
- **Regularize α nhẹ:** entropy regularization nhỏ hoặc dropout trên α để khỏi collapse vào 1 ô. Hoặc
  thêm temperature.
- **Residual + LN** kiểu transformer (LN trước, residual) — ổn định.
- **Theo dõi để chẩn:** giám sát entropy/độ-sparse của α. α đứng yên ở đều → không lợi ích (thoái hoá
  mean); α collapse 1 ô sớm → quá sắc, overfit nhiễu. Muốn **sharpen từ từ**.
- **Giữ mean-ROI làm fallback/baseline** (đã ở §8 ablation): attention-pool bất ổn/thua thì lùi về known-good.

> Thành thật: attention-pool vs mean là **canh bạc thực nghiệm** — có thể cứu focal nhưng thêm rủi ro bất
> ổn. Ablation §13 sẽ phán; set up cẩn thận rồi so.

### 7.3 — Chuẩn hoá + regularization (đòn bẩy #3, vệ sinh — thấp hơn vì data lớn)

**Vì sao vẫn cần:** head MLP nhỏ + class-weight nặng (§7.1) làm động lực huấn luyện nhiễu; chuẩn hoá ổn
định, regularization chặn overfit lớp hiếm (ít dương → dễ học vẹt).

**Cách giải quyết:**
- **LayerNorm, KHÔNG BatchNorm.** Lý do cụ thể: (a) "batch" vùng thay đổi thành phần do mask; (b) LN
  chuẩn hoá per-sample, tránh rò rỉ qua trục 29-vùng/batch. BN rắc rối với mask + batch hiệu dụng nhỏ.
  (Bạn đã có LN trong neck/MLPHead — giữ.)
- **Dropout vừa phải** (0.1–0.3) lớp ẩn MLP. Đừng quá tay (tín hiệu y khoa tinh tế).
- **Weight decay** AdamW ~1e-4…1e-2 (đã dùng AdamW).
- **Chuẩn hoá feature cache** nếu scale BioViL-T dao động mạnh (hoặc dựa vào LN đầu vào).
- **Early-stop/checkpoint theo macro-F1**, KHÔNG theo loss (loss bị lớp đa số chi phối).
- **Đừng over-regularize:** data 200k lớn → overfit ít đáng lo hơn *collapse về lớp đa số* (imbalance).
  Ưu tiên §7.1 ≫ §7.3.

### 7.4 — Metric đánh giá (gắn kèm, bắt buộc đổi)

- **Macro-F1 + per-class** cho cả M3 và M4 — KHÔNG dùng accuracy (M4) hay chỉ micro/AUROC (M3): chúng bị
  lớp đa số lừa. Per-class để thấy đòn bẩy §7.1 có nâng được lớp hiếm không.
- M4: thêm support 3 lớp (đã có trong `tkan_metrics`); accuracy ≈ prior "stable" là **cờ đỏ**, không phải thành công.
- Giữ AUROC (`multilabel_auroc`) như phụ trợ, nhưng quyết định checkpoint theo macro-F1.

---

## 8. KAN = ablation (giữ, đừng bỏ)

Không bỏ KAN. Giữ `KANHead` qua `build_head(head_type=...)` để **chạy so sánh có kiểm soát**:
- Ablation head: `--head mlp` (default) vs `--head kan`.
- Ablation pooling: `mean-roi` vs `attention` vs `predict-then-pool(aux)`.
- Ablation global: on/off.

→ "Novelty" của pipeline được chống lưng bằng bằng chứng. KAN chỉ thành "ngôi sao" nếu (a) thắng trên
dữ liệu, hoặc (b) sau này làm **concept bottleneck** (lúc đó hàm cạnh KAN trên concept có tên = interpretability thật — xem ghi chú §11).

---

## 9. Grounding cho M5 (cập nhật)

Bản MLP này cấp cho M5 các tín hiệu faithful (không sinh tự do):
- **"Vùng nào"** = `softmax_r(region_logit[r,d])` (exact, từ LSE) — như kế hoạch M5.
- **"Vùng con nào trong vùng"** = `α[r,·]` của attention-pool (faithful, vì là trọng số pool thật) — **mới**.
- **Confidence** = σ/softmax sau temperature scaling (calibrate trên val).
- **Tiến triển** = readout argmax T-MLP (1:1), guard temporal khi không có prior.

> Vẫn KHÔNG dùng gradient-saliency/LIME/SHAP. `α` thay thế vai trò "saliency" mà vẫn faithful.

---

## 10. Chi phí train (200k ảnh, 4090)

**Yếu tố quyết định: cache feature BioViL-T** (`CachedFeatureSource`), không phải MLP.

| Giai đoạn | Ước lượng (4090) | Ghi chú |
|-----------|------------------|---------|
| Precompute feature BioViL-T (1 lần) | ~vài chục phút → ~2 giờ | Chặn bởi decode JPEG + ghi đĩa (~40GB fp16, `[196,512]`/ảnh); GPU forward nhanh |
| Train C-MLP trên feature cache | **~vài phút/epoch**; cả run ~**1–3 giờ** | I/O-bound; head tí hon; nhanh hơn YOLO (~70'/epoch) nhiều cấp độ |
| Train T-MLP | ≤ C-MLP | Train trên *cặp*, ít mẫu hiệu dụng hơn |

⚠️ Nếu **không** cache mà chạy encoder live mỗi step → mỗi epoch ≈ thời gian precompute (~giờ), cả run
thành **ngày**. → cache là bắt buộc. **Cần đo thực tế:** throughput precompute (img/s), tốc độ đọc đĩa
(NVMe vs HDD — đang NTFS), batch size.

(Con số trên là ước lượng có biên độ; phụ thuộc throughput chính xác encoder + I/O, đo rồi mới chốt.)

---

## 11. File phải đụng (so với `src/phase_3/` + `phase_4/`)

| File | Thay đổi |
|------|----------|
| `roi_pool.py` | Thêm **attention-pool** (query vùng × key/value ô, mask box) trả `(region_pooled, region_mask, α)`; giữ mean-roi cho ablation |
| `model.py` | `CKAN`→ thêm **đường global** (GAP + GlobalHead + gate combine); `forward` trả thêm `global_logits`, `α`; mặc định head `mlp` |
| `heads.py` | Không đổi (đã có `MLPHead`/`KANHead`/`build_head`); thêm `SharedCellHead` nếu làm predict-then-pool aux |
| `losses.py` | Thêm `pos_weight`/focal cho masked BCE (M3); class-weight CE (M4) |
| `config.py` | Cờ `--head mlp|kan`, `--pool mean|attn|cell`, `--global on|off`, `--pos-weight`, `--proj-dim` |
| `tkan.py`/`tkan_*` (phase_4) | Đổi head default MLP; input contract **giữ nguyên** (`region_feat[29,128]`, `region_in_dim=412`) |

---

## 12. Khác biệt so với `phase_3_4.md` gốc (diff rõ ràng)

- Pooling: ~~mean ROI-pool~~ → **attention-pool** (mean giữ làm ablation).
- Head mặc định: ~~KAN ở reasoning layer~~ → **MLP** (KAN = ablation).
- ~~Chỉ đường vùng~~ → **+ nhánh global** gộp ở image-logit (cho finding quan hệ).
- Neck: **giữ** (512→128), lý do đổi từ "vì KAN" → "feature gọn cho T-MLP + chuẩn hoá".
- Grounding M5: **+ α** (attribution nội-vùng faithful) bên cạnh softmax-LSE mức vùng.
- Loss: imbalance handling (pos_weight/focal, class-weight) từ TODO → **bắt buộc**.
- T-MLP: input contract **không đổi**; chỉ KAN→MLP head.

---

## 13. Ablation cần chạy (để chốt bằng dữ liệu, không bằng cảm giác)

- [ ] head: `mlp` vs `kan` (parity, cùng param/FLOPs nếu được)
- [ ] pool: `mean` vs `attention` vs `predict-then-pool(aux)`
- [ ] global: on vs off (xem cardiomegaly / enlarged cardiomediastinum cải thiện không)
- [ ] imbalance: pos_weight vs focal vs none
- [ ] (sau) concept bottleneck go/no-go: đo F1 concept-từ-ảnh trên silver MIMIC trước khi cân nhắc ante-hoc

---

## 14. TL;DR

Đường vùng **attention-pool** (cứu focal + cho α faithful) → **neck 512→128** (giữ) → **MLP head**;
song song **nhánh global** (cứu cardiomegaly) gộp ở image-logit. **T-MLP nhận `region_feat[29,128]`**
như cũ, head MLP, contract 412 không đổi. **KAN giữ làm ablation.** Xử lý **imbalance** là điểm sống còn.
Train rẻ nếu **cache feature** (vài phút/epoch); precompute BioViL-T là phần tốn một-lần (~giờ).
