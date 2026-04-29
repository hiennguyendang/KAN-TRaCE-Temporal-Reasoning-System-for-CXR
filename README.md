# MyChex - Chest X-ray Dataset Engineering Platform

Hệ thống quản lý, tiền xử lý, và phân tích dữ liệu cho ba dataset hình ảnh ngực y tế lớn: **MIMIC-CXR**, **CheXplus**, và **NIH ChestXray14**.

**Ngôn ngữ**: Tiếng Việt (docs) + Python 3.10+  
**Cập nhật**: 29 April 2026

---

## 📋 Tổng quan

MyChex cung cấp:

✅ **Preprocessing pipeline**: Tải, resize, convert ảnh từ PNG → JPG (quality=100)  
✅ **Metadata standardization**: Xây dựng JSONL metadata chuẩn hóa cho training/inference  
✅ **Scene graph extraction**: Trích RadGraph JSON → CSV scene graph (CheXplus)  
✅ **Patient-level splitting**: Phân chia bệnh nhân theo quality tier (diamond/gold/silver)  
✅ **Data loaders**: Registry-based loaders tích hợp sẵn cho ba dataset  

---

## 📁 Cấu trúc workspace

Xem chi tiết tại [WORKSPACE_STRUCTURE.md](WORKSPACE_STRUCTURE.md)

```
MyChex/
├── metadata/               # Xây dựng JSONL metadata chuẩn
├── preprocess/             # Preprocessing pipeline (hình ảnh)
├── scene_graph/            # Scene graph extraction + splits
├── src/mychex/             # Core package (loaders, utils)
├── scripts/                # Scripts độc lập (manifest, crawl, v.v.)
├── configs/                # YAML config (đường dẫn, tham số)
├── docs/                   # Tài liệu (notes, roadmap)
├── notebooks/              # Jupyter notebooks
├── artifacts/              # Output (manifest, reports)
├── logs/                   # Log files
└── tests/                  # Unit tests
```

---

## ⚡ Quick Start

### 1. Setup môi trường

```bash
# Tạo virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate.bat

# Cài đặt package
pip install -e .
```

### 2. Cấu hình dataset

```bash
# Tạo file cấu hình từ template
copy configs\datasets.example.yaml configs\datasets.yaml
```

Cập nhật đường dẫn trong `configs/datasets.yaml`:

```yaml
datasets:
	mimic:
		root: C:\Users\dhint\CHEX-DATA\MIMIC-CXR
	chexplus:
		root: C:\Users\dhint\CHEX-DATA\CHEXPLUS
	nih:
		root: C:\Users\dhint\CHEX-DATA\NIH-CHESTXRAY14
```

### 3. Load dữ liệu (Python)

```python
from mychex.data.registry import DatasetRegistry

# Load CheXplus
chexplus_dataset = DatasetRegistry.load("chexplus")
sample = chexplus_dataset[0]
print(sample.image_path, sample.labels, sample.report)

# Load MIMIC
mimic_dataset = DatasetRegistry.load("mimic")
sample = mimic_dataset[1000]

# Load NIH
nih_dataset = DatasetRegistry.load("nih_chestxray14")
```

---

## 🔄 Main Pipelines

### A. MIMIC-CXR Pipeline

**Đầu vào**: Google Drive (rclone)  
**Đầu ra**: JPG images + `mimic_metadata.jsonl`

**Các bước**:

1. **Tải từ Drive** (`preprocess/mimic.bat`):
	 ```bash
	 # Tải pack-by-pack, xử lý, checkpoint
	 .\preprocess\mimic.bat
	 ```
	 - Checkpoint: `MIMIC-CXR/checkpoints/processed_packs.json`
	 - Resume an toàn khi rerun

2. **Xây dựng metadata** (`metadata/build_mimic_metadata.py`):
	 ```bash
	 python metadata/build_mimic_metadata.py
	 ```
	 - Output: `MIMIC-CXR/metadata.jsonl`

**Stats**: ~400k hình ảnh, ~60k bệnh nhân

---

### B. CheXplus Pipeline

**Đầu vào**: PNG images + CSV labels + RadGraph JSON  
**Đầu ra**: JPG images + metadata + scene graph + splits

**Các bước**:

1. **Xây dựng metadata JSONL**:
	 ```bash
	 python metadata/build_chexplus_metadata.py \
		 --images-source "C:\Users\dhint\CHEX-DATA\CHEXPLUS\chexplus_metadata.jsonl" \
		 --labels-json "C:\Users\dhint\CHEX-DATA\CHEXPLUS\chexbert_labels\findings_fixed.json"
	 ```
	 - Output: `CHEXPLUS/metadata/chexplus_unified.jsonl`

2. **Trích scene graph từ RadGraph**:
	 ```bash
	 python scene_graph/extract_scene_graph.py
	 ```
	 - Input: `section_findings.json`, `section_impression.json`
	 - Output: `CHEXPLUS/metadata/chexplus_scene_graph.csv` (2.8M rows)
	 - **Đặc biệt**: `study1` → `temporal_status` trống

3. **Phân chia bệnh nhân (splits + tiers)**:
	 ```bash
	 python scene_graph/build_chexplus_splits.py
	 ```
	 - Output:
		 - `splits.csv` (patient-level split assignment)
		 - `split_score_value_counts.csv`, `split_tier_value_counts.csv`
		 - `split_value_counts.csv`

**Tier logic**:

- **Diamond**: `max_spatial_score >= 2` AND `max_temporal_score >= 5` → 6,359 bệnh nhân
- **Gold**: `max_temporal_score >= 1` OR `max_spatial_score >= 4` → 47,308 bệnh nhân
- **Silver**: else → 11,035 bệnh nhân

**Split assignment** (29 April 2026):

- 3,000 diamond → **test**
- 3,359 diamond → **val**
- 58,343 non-diamond → **train**

**Stats**: 64,702 bệnh nhân, 2.8M scene graph rows

---

### C. NIH ChestXray14 Pipeline

**Đầu vào**: Kaggle dataset  
**Đầu ra**: JPG images + metadata JSONL

**Các bước**:

1. **Tải từ Kaggle**:
	 ```bash
	 .\scripts\download_nih_chestxray14.ps1
	 ```

2. **Xử lý ảnh**:
	 ```bash
	 python preprocess/nih_main.py
	 ```

3. **Xây dựng metadata**:
	 ```bash
	 python metadata/build_nih_metadata.py  # Nếu cần
	 ```

---

## 📊 Dataset Stats (hiện tại)

| Dataset | Bệnh nhân | Hình ảnh | Metadata | Scene Graph | Splits |
|---------|----------|---------|----------|-------------|--------|
| MIMIC-CXR | ~60k | ~400k | ✓ JSONL | - | - |
| CheXplus | 64.7k | 223k | ✓ JSONL | ✓ (2.8M rows) | ✓ (train/val/test) |
| NIH | ~4.8k | ~112k | ✓ | - | - |

---

## 📋 Metadata Schema

### JSONL Records

**Trường**:
- `dataset`: "mimic" | "chexplus" | "nih"
- `image_id`: Định danh ảnh duy nhất
- `image_path`: Đường dẫn đến ảnh JPG
- `patient_id`: ID bệnh nhân
- `study_id`: ID khám bệnh
- `labels`: Dict labels {label_name: {0, 1, -100}}
- `report`: Text báo cáo (text)
- `view`: Loại view (PA, AP, LAT, v.v.)

**Loại bỏ từ export**: `bboxes`, `findings` (chỉ dùng nội bộ)

### Scene Graph CSV (CheXplus)

**Cột**:
- `patient_id`, `study_id`, `study_key`
- `source_section`, `source_view`
- `anatomy`, `observation`, `presence`
- `temporal_status`, `bboxes`

**Luật**:
- Dedup: theo `(patient_id, study_id, anatomy, observation)`, ưu tiên Impression
- `study1` → `temporal_status` trống
- `bboxes` = `[]`

### Splits CSV (CheXplus)

**Cột**: `patient_id`, `split`, `patient_tier`, `max_spatial_score`, `max_temporal_score`

**Split**: "train" | "val" | "test"  
**Tier**: "diamond" | "gold" | "silver"

---

## 🔧 Cấu hình

### `configs/datasets.yaml`

```yaml
datasets:
	mimic:
		root: C:\Users\dhint\CHEX-DATA\MIMIC-CXR
	chexplus:
		root: C:\Users\dhint\CHEX-DATA\CHEXPLUS
	nih:
		root: C:\Users\dhint\CHEX-DATA\NIH-CHESTXRAY14
```

### `preprocess/config.py` (xử lý ảnh)

```python
RESIZE_TARGET = 512           # Kích thước target
JPEG_QUALITY = 100            # Quality 0-100
JPEG_SUBSAMPLING = 0          # 0=4:4:4 (no subsampling)
IMAGE_FORMAT = "JPG"          # Output format
```

---

## 📈 Workflow rebuild metadata

Nếu cần rebuild từ đầu:

```bash
# 1. Scene graph từ RadGraph JSON
python scene_graph/extract_scene_graph.py

# 2. Splits + tiers từ scene graph
python scene_graph/build_chexplus_splits.py

# 3. MIMIC metadata
python metadata/build_mimic_metadata.py

# 4. CheXplus metadata
python metadata/build_chexplus_metadata.py
```

---

## 🔍 Checkpoint & Resume

### MIMIC Processing

- **File**: `MIMIC-CXR/checkpoints/processed_packs.json`
- **Mục đích**: Theo dõi pack đã xử lý
- **Behavior**: Khi rerun, bỏ qua các pack đã hoàn thành
- **Manual reset**: Xóa file để reset

### Batch Logging

- **File**: `MIMIC-CXR/logs/mimic_batch.log`
- **Mục đích**: Ghi log từng batch xử lý

---

## 🎯 Development Notes

### Hiện trạng

- ✓ MIMIC: Tải + xử lý + metadata
- ✓ CheXplus: Metadata + Scene graph + Splits
- ✓ NIH: Tải + xử lý
- ✓ Data loaders: Registry-based loading

### TODO/Mở rộng

- [ ] Multi-modal dataset objects (image + report)
- [ ] Label harmonization cross-dataset
- [ ] Experiment tracking (MLflow/WandB)
- [ ] Fine-tune splits (stratified by modality)
- [ ] Augmentation pipeline

---

## 📝 File quan trọng

- [WORKSPACE_STRUCTURE.md](WORKSPACE_STRUCTURE.md) - Chi tiết cấu trúc
- [METADATA.md](METADATA.md) - Schema metadata
- [docs/dataset_ingestion_notes.md](docs/dataset_ingestion_notes.md) - Download notes
- [docs/framework_roadmap.md](docs/framework_roadmap.md) - Roadmap

---

## 👤 Notes

- **Windows environment**: Các scripts sử dụng PowerShell (.ps1) và batch (.bat)
- **rclone**: Dùng để sync Google Drive
- **RadGraph**: JSON input cho scene graph extraction
- **Seed**: Split random dùng seed=42 để lặp lại được

---

**Các câu hỏi?** Xem `docs/` hoặc file code docstrings.
