# Phase 2 — Tổng kết tiến độ (Module 2: Scene Graph)

> Cập nhật: 2026-06-11. Tài liệu này ghi lại **tất cả những gì đã làm ở Phase 2** —
> code, artifact, trạng thái train, hạ tầng resilience, và những gì còn lại.
> Bổ trợ cho `CLAUDE.md` (lệnh) và `docs/pseudo_scene_graph.md` (kiến trúc nhánh LLM).

Phase 2 = **Module 2** của pipeline KAN-TRaCE: bản đồ giải phẫu có cấu trúc cho ảnh CXR.
Gồm **2 nhánh** chạy độc lập, hội tụ ở định dạng ImaGenome `*_SceneGraph.json`:

| Nhánh | Vai trò | File | Trạng thái |
|-------|---------|------|------------|
| **BBOX** (không gian) | Detector phát hiện 29 vùng giải phẫu | `0_`→`3_` | 🟡 YOLOv8l đang train (~4/100) |
| **ATTRIBUTE/RELATIONSHIP** (ngữ nghĩa) | LLM bóc tách finding theo vùng từ report → pseudo scene graph | `4_`→`7_` + `sg_lib.py` | 🟡 vocab + SFT corpus xong; finetune/inference chờ |

Mọi script đứng độc lập (không import package), **chạy từ project root**.

---

## A. Nền tảng chung — 29 lớp giải phẫu & quy ước toạ độ

- **29 lớp giải phẫu**, thứ tự **alphabetical**, là nguồn chân lý duy nhất — định nghĩa ở
  `CLASS_NAMES` trong [src/phase_2/0_prepare_dataset.py](../src/phase_2/0_prepare_dataset.py)
  và mirror trong [src/phase_2/dataset.yaml](../src/phase_2/dataset.yaml). Hai chỗ phải khớp.
- **`--short-side 512`** = scale resize chuẩn xuyên suốt preprocess + train.
  `width_resized = round(width * 512 / min(width, height))`.
- Scene graph JSON: `x1/y1/x2/y2` ở không gian **resized 512-short-side**; `original_x1..` ở
  pixel gốc. Bbox `(0,0,0,0)` là **sentinel** (ngoài center-crop) → **lọc bỏ**.

---

## B. NHÁNH BBOX — Detector pipeline

### B.1 — `0_prepare_dataset.py` ✅ ĐÃ CHẠY
Build dataset YOLO từ `data/mimic_metadata_final.jsonl` + scene graph silver/gold.
- Parse mỗi scene graph → YOLO label lines (class_id + bbox chuẩn hoá), **bỏ qua sentinel
  `(0,0,0,0)`** và box suy biến (w/h ≤ 0).
- Đặt ảnh vào `dataset/images/{split}/` bằng **hardlink → symlink → copy** (`--link-mode auto`)
  để khỏi tốn thêm disk trên cùng ổ NTFS.
- Split `valid` được chuẩn hoá về `val`.

**Kết quả thực tế (đếm trên đĩa):**

| Split | Số ảnh | Nhãn |
|-------|--------|------|
| train | ~296k ảnh đã link / **148,242** file nhãn | `dataset/labels/train` |
| val   | 42,670 | `dataset/labels/val` |
| test  | 42,194 | `dataset/labels/test` |

> ~148k ảnh train có scene graph (số batch khớp: 12,354 batch × batch≈12 ≈ 148k).

### B.2 — `1_train_yolo.py` 🟡 ĐANG TRAIN
Train **YOLOv8l** (`yolov8l.pt`). Cấu hình chốt:
- `imgsz=1024`, `amp=True`, `batch≈12` (4090), `epochs=100`, `patience=15`, `cache=disk`,
  `save_period=10`.
- **Augmentation giải phẫu-an toàn:** `mosaic=0.0`, `mixup=0.0` (tắt để không xé cấu trúc
  giải phẫu); chỉ giữ `degrees=2.0`, `perspective=0.0005`.
- `resolve_batch_size()` tự co batch theo VRAM (≥24GB→16, ≥20→12, ...).
- **`ProgressWriter`** (callback ultralytics) ghi `yolo_status.txt` — thanh tiến độ live,
  hoạt động bất kể launch kiểu gì. Nhánh `--resume` khôi phục epoch/optimizer/args từ
  `last.pt` (không truyền lại hyperparams).

**Trạng thái train hiện tại (2026-06-11):**

| Epoch | mAP50 | mAP50-95 | P | R | phút/epoch |
|------:|------:|---------:|----:|----:|-----------:|
| 1 | 0.834 | 0.517 | 0.846 | 0.795 | 87.0 |
| 2 | 0.863 | 0.569 | 0.876 | 0.821 | 74.2 |
| 3 | 0.863 | 0.577 | 0.884 | 0.811 | 71.7 |
| 4 | đang chạy (~85% epoch) | — | — | — | ~67 |

- Đang ở **epoch 4/100**, ETA toàn run ~115 giờ. mAP50 ≈ 0.86 ngay từ epoch 2 (vùng giải
  phẫu dễ học vì nhất quán không gian).

### B.3 — `2_train_rtdetr.py` ⬜ CHƯA CHẠY
Train **RT-DETR-l** (`rtdetr-l.pt`) — encoder transformer nặng hơn, `batch=8` (4090).
Cùng dataset/augmentation. Là detector thay thế/ensemble; chờ YOLO xong.

### B.4 — `3_pseudo_labeling.py` ⬜ CHƯA CHẠY
Dùng `best.pt` để pseudo-label NIH/CheXplus → file YOLO `.txt`. Tự nhận YOLO vs RT-DETR
theo tên weight; resume được (bỏ qua ảnh đã có nhãn); giữ box lớn nhất nếu vùng trùng.
Output `.txt` này là **input của `7_`**.

---

## C. NHÁNH ATTRIBUTE/RELATIONSHIP — Pseudo Scene-Graph LLM

Ý tưởng: finetune LLM local trên ImaGenome **silver** để học `(report + danh sách vùng) →
finding JSON gọn theo vùng`, rồi áp lên ~300k ảnh NIH/CheXplus sinh `*_SceneGraph.json` đầy
đủ ImaGenome-compatible. Logic chung ở [src/phase_2/sg_lib.py](../src/phase_2/sg_lib.py).
Cài thêm: `pip install -e ".[sg]"`.

### C.0 — `sg_lib.py` ✅ thư viện lõi (round-trip đã verify)
- `compact_target_from_scene_graph()`: silver `attributes[]` → target gọn
  `{vùng: [{phrase, relations, temporal, severity, texture, comparison}]}`.
- `assemble_scene_graph()`: target gọn → full ImaGenome JSON (objects + attributes +
  attributes_ids + cue arrays). **Chỉ gắn finding vào vùng detector thực sự phát hiện.**
- `bbox_resized_and_original()`: YOLO-normalized box → pixel ở cả 2 không gian (512 + gốc).
- `dump_compact()` / `parse_compact()`: serialize ↔ parse chịu được code-fence/prose.
- `SYSTEM_PROMPT`, `build_user_prompt()`: prompt dùng chung cho `5_` và `7_`.
- **`CUE_FIELDS`** = `temporal_cues, severity_cues, texture_cues, comparison_cues` —
  `comparison_cues` (improved/worsened/new/resolved/stable) là **tín hiệu giám sát dự kiến
  cho Module 4 (T-KAN)**, được carry xuyên suốt.

### C.1 — `4_extract_sg_vocab.py` ✅ ĐÃ CHẠY → `sg_vocab.json`
Quét toàn bộ scene graph silver → vocab quan hệ có kiểm soát (single source of truth cho
`5_` và `7_`). Lọc theo `--min-count 5`, `--top-per-region 60`, giới hạn về 29 lớp detector.

**Kết quả:**
- **140 relations** giữ lại.
- **27/29 vùng** có finding.
- Cue values: `temporal=4, severity=4, texture=10, comparison=3`.
- Prefix quan hệ: `anatomicalfinding`, `nlp`, `tubesandlines`, `disease`,
  `technicalassessment`, `device`.
- Kèm `rel2id` (relation → CUI hay gặp nhất) và `region_synsets`.

### C.2 — `5_build_sft_dataset.py` ✅ ĐÃ CHẠY → `data/sg_sft/`
Mỗi hàng metadata có `scene_path` → 1 mẫu chat (system / user=report+vùng / assistant=target
gọn). "Vùng khả dụng" = silver objects ∩ 29 lớp detector (khớp menu lúc inference).
- Split train/val theo **crc32 hash của `patient_id`** (chống leakage).
- **Downsample mẫu âm** (`--keep-empty-frac 0.1`) để model không học ra `{}`.

**Kết quả:** **207,682 train** + **4,144 val** (khớp `wc -l`).

### C.3 — `6_finetune_sg_llm.py` ⬜ CHƯA CHẠY (chờ YOLO xong, dùng chung GPU)
**QLoRA** finetune `Qwen/Qwen2.5-7B-Instruct`:
- 4-bit **nf4** + double-quant, compute `bfloat16`, `device_map=auto` (vừa 1×24GB).
- LoRA `r=16, alpha=32, dropout=0.05` trên `q/k/v/o/gate/up/down_proj`.
- **Completion-only loss** (`DataCollatorForCompletionOnlyLM`, response template
  `<|im_start|>assistant\n`) — chỉ học phần assistant.
- 2 epoch, `lr=2e-4` cosine, `max_len=2048`, grad-accum 8, gradient checkpointing.

### C.4 — `7_build_pseudo_scene_graph.py` ⬜ CHƯA CHẠY
Inference: với mỗi ảnh NIH/CheXplus, đọc box YOLO `.txt` (từ `3_`) + report → LLM finetuned
map finding lên vùng → `assemble_scene_graph()` ra `*_SceneGraph.json` đầy đủ.
- Generation theo **batch, left-pad, greedy** (`do_sample=False`).
- Output được **validate/snap vào vocab** → bỏ hallucination ngoài vocab.
- `--update-metadata` ghi metadata mới có `scene_path`.

---

## D. Hạ tầng resilience (train không người trông) — `scripts/`

Viết bằng PowerShell, **ASCII-only** (PS5.1 đọc UTF-8 lỗi). Lý do tồn tại: train chạy nhiều
ngày, từng **chết lặng 3 lần** khi launch coupled với phiên Claude.

| Script | Vai trò |
|--------|---------|
| [resume_yolo_on_boot.ps1](../scripts/resume_yolo_on_boot.ps1) | Resume từ `last.pt`; set env vars + **prepend DLL dir của env `chex` vào PATH** (nếu thiếu, DataLoader worker chết `WinError 1114` khi load `shm.dll`); guard double-launch & case 100-epoch-done. Đăng ký Scheduled Task onlogon để tự resume sau reboot. |
| [watchdog_yolo.ps1](../scripts/watchdog_yolo.ps1) | Chạy liên tục. Xử lý 2 kiểu chết: (1) process biến mất → relaunch sau 15s; (2) treo (status đứng yên >15 phút, có guard tránh nhầm lúc validation) → kill + relaunch. PID→`logs/watchdog.pid`, log→`logs/watchdog.log`, dừng bằng tạo `logs/watchdog.stop`. |
| [log_yolo_progress.py](../scripts/log_yolo_progress.py) | Mirror metric per-epoch từ `results.csv` → `yolo_progress.log`. |

**Môi trường train bắt buộc:** conda env **`chex` (Python 3.12.13, torch 2.6.0+cu124)** —
KHÔNG dùng Python 3.13 (bug dataclass làm vỡ `import torch`). Env vars:
`KMP_DUPLICATE_LIB_OK=TRUE`, `PYTHONNOUSERSITE=1`. **Launch DETACHED**
(`Start-Process -WindowStyle Hidden`) để không bị kéo chết theo phiên.

**File tiến độ:**
- `yolo_status.txt` — thanh tiến độ live (callback trong `1_train_yolo.py`).
- `yolo_progress.log` — metric per-epoch mirror từ `results.csv`.

⚠️ ultralytics chỉ resume ở **ranh giới epoch** → crash giữa epoch mất nguyên epoch (~1h).
Mid-epoch resume đã bàn nhưng hoãn (stock ultralytics không hỗ trợ, phải vá `_do_train`).

---

## E. Còn lại của Phase 2

1. ⏳ Chờ YOLOv8l xong 100 epoch → lấy `best.pt`.
2. ⬜ (tuỳ chọn) Train RT-DETR-l (`2_`) để so/ensemble.
3. ⬜ `3_pseudo_labeling.py` — pseudo-label NIH/CheXplus bằng `best.pt`.
4. ⬜ `6_finetune_sg_llm.py` — QLoRA Qwen2.5-7B trên 207.7k SFT (dùng GPU sau khi YOLO nhả).
5. ⬜ `7_build_pseudo_scene_graph.py` — sinh `*_SceneGraph.json` cho ~300k ảnh, cập nhật metadata.

Xong Phase 2, `comparison_cues` trong pseudo scene graph sẵn sàng làm tín hiệu giám sát cho
**Module 4 (T-KAN)** ở phase sau.
