# Unified 21-Dimensional Label Conversion

## Tổng Quan

Script `unify_labels_21dim.py` chuyển đổi labels từ format 14 chiều (dataset-specific) sang format unified 21 chiều cho PyTorch multi-label classification.

### Thống Kê Xử Lý
- **MIMIC**: 227,833 records ✓
- **CheXplus**: 190,962 records ✓
- **NIH**: 47,121 records ✓
- **Tổng**: 465,916 records (100% thành công, 0 skipped)

### Định Dạng Output
- **Labels**: Mảng Python list chứa 21 float values
- **Kiểu dữ liệu**: `float` (không phải `int`)
- **Ignore value**: `-100.0` (PyTorch convention)
- **Positive label**: `1.0`
- **Negative label**: `0.0`

---

## Vấn Đề Gốc & Giải Pháp

### Vấn Đề 1: Giá Trị Labels Hiện Tại Không Chính Xác
**Trạng thái cũ**:
- MIMIC/CheXplus: Lưu integer `[0, 1]` từ CSV (-1 bị bỏ qua)
- NIH: Lưu integer `[0, 1]` từ binary vector

**Vấn đề**:
- Giá trị `-1` (uncertain) từ CheXpert format bị mất
- Format không phù hợp với PyTorch (nên là float)
- Không có ignore_index (-100.0) cho missing data

### Giải Pháp:
- Convert sang **float** (1.0, 0.0, -100.0)
- Ánh xạ từng chỉ số từ 14-dim sang 21-dim theo từng dataset
- Gán `-100.0` cho:
  - Giá trị `-1` (uncertain) từ MIMIC/CheXplus
  - Labels không được map vào 21-dim space

---

## Chi Tiết Mapping

### Unified 21-Label Dictionary
```
Indices 0-6 (Phổ biến 7 bệnh):
  0: Atelectasis              | 4: Pleural Effusion
  1: Cardiomegaly            | 5: Pneumonia
  2: Consolidation           | 6: Pneumothorax
  3: Edema

Indices 7-13 (MIMIC/CheXplus specific - 7 bệnh):
  7: Enlarged Cardiomediastinum | 11: No Finding
  8: Fracture                    | 12: Pleural Other
  9: Lung Lesion                 | 13: Support Devices
  10: Lung Opacity

Indices 14-20 (NIH specific - 7 bệnh):
  14: Infiltration           | 18: Fibrosis
  15: Mass                   | 19: Pleural Thickening
  16: Nodule                 | 20: Hernia
  17: Emphysema
```

### MIMIC → Unified (14 → 21)
```
MIMIC[0]  "Atelectasis"              → UNIFIED[0]
MIMIC[1]  "Cardiomegaly"             → UNIFIED[1]
MIMIC[2]  "Consolidation"            → UNIFIED[2]
MIMIC[3]  "Edema"                    → UNIFIED[3]
MIMIC[4]  "Enlarged Cardiomediastinum" → UNIFIED[7]
MIMIC[5]  "Fracture"                 → UNIFIED[8]
MIMIC[6]  "Lung Lesion"              → UNIFIED[9]
MIMIC[7]  "Lung Opacity"             → UNIFIED[10]
MIMIC[8]  "No Finding"               → UNIFIED[11]
MIMIC[9]  "Pleural Effusion"         → UNIFIED[4]
MIMIC[10] "Pleural Other"            → UNIFIED[12]
MIMIC[11] "Pneumonia"                → UNIFIED[5]
MIMIC[12] "Pneumothorax"             → UNIFIED[6]
MIMIC[13] "Support Devices"          → UNIFIED[13]
```

### CheXplus → Unified
- **Giống MIMIC**: Cùng 14-label structure, dùng mapping `MIMIC_TO_UNIFIED`

### NIH → Unified (14 → 21)
```
NIH[0]  "Atelectasis"              → UNIFIED[0]
NIH[1]  "Cardiomegaly"             → UNIFIED[1]
NIH[2]  "Effusion" (= Pleural Effusion) → UNIFIED[4]
NIH[3]  "Infiltration"             → UNIFIED[14]
NIH[4]  "Mass"                     → UNIFIED[15]
NIH[5]  "Nodule"                   → UNIFIED[16]
NIH[6]  "Pneumonia"                → UNIFIED[5]
NIH[7]  "Pneumothorax"             → UNIFIED[6]
NIH[8]  "Consolidation"            → UNIFIED[2]
NIH[9]  "Edema"                    → UNIFIED[3]
NIH[10] "Emphysema"                → UNIFIED[17]
NIH[11] "Fibrosis"                 → UNIFIED[18]
NIH[12] "Pleural Thickening"       → UNIFIED[19]
NIH[13] "Hernia"                   → UNIFIED[20]
```

---

## Logic Xử Lý Value Mapping

### MIMIC & CheXplus (CheXpert Format)
```python
if original_value == 1:
    unified_labels[unified_idx] = 1.0          # Positive (disease present)
elif original_value == 0:
    unified_labels[unified_idx] = 0.0          # Negative (no disease)
elif original_value == -1:
    unified_labels[unified_idx] = -100.0       # Uncertain → Ignore
else:
    unified_labels[unified_idx] = -100.0       # Unknown → Ignore
```

### NIH (Binary Format)
```python
if original_value == 1:
    unified_labels[unified_idx] = 1.0          # Positive (finding present)
elif original_value == 0:
    unified_labels[unified_idx] = 0.0          # Negative (no finding)
else:
    unified_labels[unified_idx] = -100.0       # Unknown → Ignore
```

### Labels Không Được Map
Indices không được map từ original dataset sẽ được gán:
```python
unified_labels[unmapped_idx] = -100.0  # Ignore_index
```

Ví dụ:
- MIMIC/CheXplus: Indices 14-20 (NIH-specific) = `-100.0`
- NIH: Indices 7-13 (MIMIC-specific) = `-100.0`

---

## File Paths & Output

### Input Files (14-dim JSONL)
```
C:\Users\dhint\CHEX-DATA\MIMIC-CXR\metadata\mimic_metadata_normalized.jsonl
C:\Users\dhint\CHEX-DATA\CHEXPLUS\metadata\chexplus_metadata_normalized.jsonl
C:\Users\dhint\CHEX-DATA\NIH\metadata\nih_metadata_with_reports_normalized.jsonl
```

### Output Files (21-dim JSONL)
```
C:\Users\dhint\CHEX-DATA\MIMIC-CXR\metadata\mimic_metadata_unified_21dim.jsonl
C:\Users\dhint\CHEX-DATA\CHEXPLUS\metadata\chexplus_metadata_unified_21dim.jsonl
C:\Users\dhint\CHEX-DATA\NIH\metadata\nih_metadata_unified_21dim.jsonl
```

### Record Structure
```json
{
  "image_id": "p12345_s67890_abc123",
  "image_path": "/path/to/image.png",
  "dataset": "mimic",
  "labels": [0.0, 0.0, 1.0, 0.0, -100.0, ..., -100.0],  // 21 floats
  "bboxes": [],
  "findings": "...",
  "report": "...",
  "patient_id": "12345",
  "study_id": "67890",
  "study_time": "..."
}
```

---

## PyTorch Multi-Label Classification Usage

### Tạo Dataset với Ignore Index
```python
import torch
import torch.nn as nn

class MedicalImageDataset:
    def __getitem__(self, idx):
        record = ...  # Load from unified JSONL
        labels = torch.tensor(record['labels'], dtype=torch.float32)
        # Input: [21] float tensor
        # Values: {-100.0, 0.0, 1.0}
        return image, labels

# Loss function
criterion = nn.BCEWithLogitsLoss(reduction='none')

# Mask ignored labels during loss computation
def masked_loss(logits, targets):
    loss = criterion(logits, targets)
    mask = (targets != -100.0).float()
    return (loss * mask).mean()
```

---

## Cải Tiến So Với Cách Cũ

| Khía Cạnh | Cũ | Mới |
|-----------|-----|-----|
| **Dimensionality** | 14-dim | 21-dim |
| **Data Type** | int | float |
| **Uncertainty Handling** | Bỏ qua (-1) | -100.0 (ignore_index) |
| **Missing Labels** | N/A | -100.0 |
| **Dataset Coverage** | Chỉ 14 bệnh chung | 21 bệnh (7 chung + 7 riêng MIMIC + 7 riêng NIH) |
| **Multi-Label Compatibility** | Giới hạn | Đầy đủ (PyTorch standard) |
| **Cross-Dataset Training** | Khó (khác schema) | Dễ (unified schema) |

---

## Chạy Script

### Cấu Hình Paths
Edit biến ở đầu file `unify_labels_21dim.py`:
```python
INPUT_FILES = {
    "mimic": "...",
    "chexplus": "...",
    "nih": "..."
}

OUTPUT_FILES = {
    "mimic": "...",
    "chexplus": "...",
    "nih": "..."
}
```

### Chạy
```bash
python metadata/unify_labels_21dim.py
```

### Output Log
```
================================================================================
LABEL UNIFICATION: 14-dim -> 21-dim
================================================================================
Processing MIMIC dataset: ...
  Processed 50000 records...
  Processed 100000 records...
  ...
✓ MIMIC: 227833 records processed, 0 skipped
✓ CHEXPLUS: 190962 records processed, 0 skipped
✓ NIH: 47121 records processed, 0 skipped
================================================================================
TOTAL: 465916 records processed, 0 skipped
================================================================================
```

---

## Xác Thực

```python
import json
from pathlib import Path

p = Path("C:/Users/.../mimic_metadata_unified_21dim.jsonl")
with p.open('r') as f:
    record = json.loads(next(f))

labels = record['labels']
assert len(labels) == 21, "Must have 21 labels"
assert all(isinstance(l, float) for l in labels), "Must be float"
assert all(l in {-100.0, 0.0, 1.0} for l in labels), "Invalid label value"
print("✓ Validation passed!")
```

---

## Ghi Chú

1. **Ignore Index**: `-100.0` là convention của PyTorch. Loss functions sẽ bỏ qua labels này.
2. **Mapping Bảo Toàn**: Mỗi bệnh từ 14-label space được map sang exact position trong 21-label space.
3. **Dataset-Specific Labels**: MIMIC/CheXplus có 7 labels không có trong NIH (indices 7-13), NIH có 7 labels không có trong MIMIC (indices 14-20).
4. **Shared Labels**: 7 bệnh phổ biến (indices 0-6) là giao tuyến của cả 3 datasets.
