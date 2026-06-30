# VERA — Phase 2 (YOLO 29-region detector): kết quả train + audit B1

> Ghi lại kết quả + nhận xét để dùng khi viết bài. Số đo trên **val silver MIMIC**
> (gold giữ ngoài). Detector: yolov8m, imgsz 448, fraction 0.25 (~37k/148k ảnh train),
> 30 epoch, 2×T4, 4.78h. Liên quan: `docs/critical_yolo.md`, `docs/VERA_methodology_concerns.md` (B1).
> Engine đo: `phase_2/audit_yolo.py` (report đầy đủ ở `dhint:phase2_runs/det29/audit/audit_report.json`).

## TL;DR (chốt)
- **Detector dùng được.** mAP50 **0.931**, mAP50-95 **0.694** — *vượt* baseline cũ (full data + yolov8l + 50 epoch = ~0.68) dù chỉ 1/4 data + model nhỏ hơn ⇒ **data không phải đòn bẩy**; trần do **độ phân giải + bản chất task**.
- **Nghi vấn B1 ("YOLO chỉ đoán template, không nhìn ảnh") bị phủ định bằng số:** YOLO vượt static-prior **+0.377 IoU** toàn cục, và **gap nới rộng tới +0.490 ở tầng ảnh bất thường nhất** — nó thêm giá trị nhiều nhất đúng chỗ quan trọng nhất.
- **Còn lại một phép quyết định, hoãn:** oracle gold-box vs detector-box ở M3 (cần M1+M3) — phép trả lời "lỗi box có lan xuống M3 không".

---

## 1. Train
- Config: yolov8m · imgsz 448 · batch 32 (2×T4 DDP) · fraction 0.25 · 30 epoch · cos_lr · aug an toàn giải phẫu (no mosaic/mixup).
- **Hội tụ, KHÔNG underfit.** Train loss còn dốc nhẹ tới epoch 30 nhưng **val loss + mọi val metric (P/R/mAP50/mAP50-95) đã phẳng từ ~epoch 18-20**; cos_lr đã anneal về ~0. Thêm epoch ở cùng config ≈ không lên (chớm overfit nhẹ). Đòn bẩy nếu muốn +điểm là **imgsz 640** (vùng nhỏ), không phải epoch/data.
- Overall val: **P 0.934 · R 0.889 · mAP50 0.931 · mAP50-95 0.694**.

### Per-class (chọn lọc) — vùng to mạnh, landmark nhỏ kéo xuống
| Vùng | mAP50-95 | ghi chú |
|---|---|---|
| right lung / left lung | 0.904 / 0.874 | xuất sắc |
| left/right upper lung zone | 0.861 / 0.858 | |
| abdomen | 0.854 | |
| right/left lower lung | 0.795 / 0.745 | |
| mediastinum | 0.792 | |
| cardiac silhouette | 0.712 | |
| **carina** | **0.279** | điểm chia khí quản — nhỏ, recall thấp (0.555) |
| cavoatrial junction | 0.419 | |
| left/right costophrenic angle | 0.430 / 0.501 | góc nhỏ, hay mờ |
| right atrium | 0.533 | |
| svc / trachea | 0.628 / 0.620 | |

Số gộp 0.694 = (phổi/thuỳ/trung thất ~0.8-0.9) trộn (carina/cavoatrial/CP-angle ~0.28-0.5). Mấy vùng M3 thật sự pool từ đó (phổi, thuỳ, trung thất, tim) đều rất tốt.

---

## 2. Audit B1 — 4 phép từ `docs/critical_yolo.md`

### Phép 1 — Static-prior baseline ✅ vượt rõ
Template = box trung bình mỗi vùng (bỏ ảnh hoàn toàn). YOLO phải vượt → chứng minh "có nhìn".
- **Overall IoU: YOLO 0.807 vs static 0.430 → gap +0.377.**
- Gap LỚN NHẤT ở vùng nhỏ (template vô dụng, YOLO gánh): carina +0.489 (0.564 vs 0.075), CP-angle phải +0.553 (0.672 vs 0.119), cavoatrial +0.455, svc +0.467, right atrium +0.436.
- Vùng to: gap nhỏ hơn (template vốn đã khá) nhưng IoU tuyệt đối cao: right lung 0.920 (static 0.694), abdomen 0.886, spine 0.853, mediastinum 0.856.
- **Dòng cho paper:** "the detector beats a static anatomical prior by **+0.38 IoU** overall, and by **+0.45–0.55** on the small landmark regions where the prior is near-useless."

### Phép 2 — Phân tầng theo độ bất thường giải phẫu ✅ kết quả mạnh nhất
Chia 2000 ảnh val thành 4 tầng theo độ lệch GT-box khỏi template (proxy cho tim to / tràn dịch / xẹp phổi…).
| Tầng | atypicality | YOLO IoU | static IoU | gap |
|---|---|---|---|---|
| Q1 điển hình | 0.408 | 0.854 | 0.592 | +0.262 |
| Q2 | 0.519 | 0.826 | 0.481 | +0.344 |
| Q3 | 0.610 | 0.801 | 0.390 | +0.411 |
| **Q4 bất thường** | 0.751 | **0.740** | **0.249** | **+0.490** |
- static **sụp mạnh** theo độ bất thường (0.592 → 0.249); YOLO **degrade nhẹ** (0.854 → 0.740).
- **Gap NỚI RỘNG** từ +0.26 (Q1) lên **+0.49 (Q4)** → YOLO thêm giá trị nhiều nhất *đúng ở ca bất thường* — phủ định trực tiếp nỗi lo "detector sụp ở đúng ca cần định vị nhất". Đây là bằng chứng mạnh nhất rằng nó đọc nội dung ảnh, không trả template.

### Phép 4 — Perturbation/deletion 🟡 trung tính (test yếu nhất)
Che đen pixel 1 vùng → đo IoU(box trước, box sau). **mean = 0.816** (120 lượt).
- Đọc: box dịch ~18% IoU khi mất nội dung vùng → **một phần content-driven, một phần context-driven**. Che 1 vùng vẫn còn toàn bộ giải phẫu xung quanh để định vị (như bác sĩ định vị tim qua cấu trúc lân cận) → dùng context là hợp lệ, KHÔNG phải "bỏ qua ảnh".
- Đây là phép mơ hồ nhất (lẫn "pixel vùng này" với "context toàn ảnh"); bằng chứng sạch nằm ở phép 1+2. 0.816 **không mâu thuẫn**, chỉ là tín hiệu mềm.

### Phép 3 — Oracle gold-box vs detector-box ở M3 ⏸ HOÃN
Phép *quyết định*: chạy M3 hai lần (gold-box vs detector-box), so macro-F1. Chênh nhỏ ⇒ lỗi box **không** phải nút thắt hạ nguồn ⇒ detector này đủ. Cần M1 features + M3 nên chưa chạy được. **Đây là việc phải làm khi có M3.**

---

## 3. Verdict theo từng nghi vấn
| Nghi vấn (B1) | Trạng thái | Bằng chứng |
|---|---|---|
| "Chỉ đoán template, không nhìn ảnh" | **Bác bỏ** | static-prior gap +0.377; gap +0.45-0.55 ở vùng nhỏ |
| "Sụp ở ca giải phẫu bất thường" | **Bác bỏ** | Q4: YOLO 0.740 vs static 0.249, gap +0.490 (gap nới rộng theo độ bất thường) |
| "Box có bỏ qua nội dung ảnh không" | Trung tính | perturbation 0.816 (content + context), không mâu thuẫn |
| "Lỗi box lan xuống M3 bao nhiêu" | **Chưa đo** | oracle ablation ở M3 — hoãn tới khi có M3 |

## 4. Hệ quả + việc tiếp
- **Dùng `best.pt` này sinh detector-box cho pipeline** (train+infer M3 đều trên detector-box — B1, khớp phân phối). val mAP đã là ước lượng cho full-MIMIC vì val là split held-out.
- **Nếu muốn +điểm** (đặc biệt carina/cavoatrial/CP-angle): warm-start từ `best.pt` ở **imgsz 640**, ~10-15 epoch — không phải thêm epoch/data. Cân nhắc *sau* khi oracle ablation cho biết có cần không.
- **Phải làm khi có M3:** oracle gold-vs-detector (phép 3) — phép chốt cuối cùng. Nếu chênh macro-F1 nhỏ ⇒ 0.694 là quá đủ, đóng concern B1.
- Số "sạch" để báo cáo (B2) cần đo lại trên **gold người-gán** (dựng dataset gold riêng từ 784 id) — hiện audit chạy trên val silver.

> Nhận xét tổng: detector này **vượt baseline cũ, nhìn ảnh thật, mạnh nhất ở ca khó** — đủ tin cậy để đi tiếp. Chỉ còn oracle-ablation-ở-M3 là mảnh cuối để đóng hoàn toàn nghi vấn YOLO.
