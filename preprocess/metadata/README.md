# Metadata Builders

This folder contains standalone builders for standardized JSONL metadata.

## MIMIC

Default inputs:
- Dataset root: `C:\Users\dhint\CHEX-DATA\MIMIC-CXR`
- Images root: `C:\Users\dhint\CHEX-DATA\MIMIC-CXR\images`
- Reports root: `C:\Users\dhint\CHEX-DATA\MIMIC-CXR\REPORT__MIMIC`
- Labels CSV: `C:\Users\dhint\CHEX-DATA\MIMIC-CXR\mimic-cxr-2.0.0-chexpert.csv`
- Study metadata CSV: `C:\Users\dhint\CHEX-DATA\MIMIC-CXR\mimic-cxr-2.0.0-metadata.csv`

Run:

```bash
python metadata/build_mimic_metadata.py
```

Outputs:
- `C:\Users\dhint\CHEX-DATA\MIMIC-CXR\metadata\mimic_metadata.jsonl`
- `C:\Users\dhint\CHEX-DATA\MIMIC-CXR\metadata\mimic_label_map.json`
- `C:\Users\dhint\CHEX-DATA\MIMIC-CXR\metadata\mimic_metadata_summary.json`

## CheXplus

Inputs:
- Image manifest: JSON, JSONL, or TXT containing image entries/paths (no recursive scan)
- CSV mapping: `df_chexpert_plus_240401.csv`
- Labels JSON/JSONL: one row per image-level label record

Run:

```bash
python metadata/build_chexplus_metadata.py --images-source <images-manifest> --csv-path <df_chexpert_plus_240401.csv> --labels-json <labels-json>
```

Output:
- `C:\Users\dhint\CHEX-DATA\CHEXPLUS\metadata\chexplus_unified.jsonl`
- `C:\Users\dhint\CHEX-DATA\CHEXPLUS\metadata\chexplus_metadata_summary.json`

Behavior:
- Joins by `(patient_id, study_id, view)`
- Uses CSV `section_findings` + `section_impression` to compose report:
	`FINDINGS:\n<findings>\n\nIMPRESSION:\n<impression>`
- Missing findings/impression are treated as empty strings
- Skips rows missing CSV key or missing labels
- Writes `bboxes` as `[]`
- Normalizes output paths to the configured `processed/images` root
- Parses identifiers only from supplied strings; no folder scanning and no index-based matching

## NIH

Inputs:
- Metadata JSON: `metadata.json` (image_id, patient_id, study_id, image_path)
- Labels CSV: `Data_Entry_2017.csv` (`Image Index`, `Finding Labels`)

Run:

```bash
python metadata/build_nih_metadata.py
```

Outputs:
- `C:\Users\dhint\CHEX-DATA\NIH\metadata\nih_metadata.jsonl`
- `C:\Users\dhint\CHEX-DATA\NIH\metadata\nih_label_map.json`
- `C:\Users\dhint\CHEX-DATA\NIH\metadata\nih_metadata_summary.json`

Behavior:
- No report (`report` is always `""`)
- No bounding boxes (`bboxes` is always `[]`)
- Joins metadata and labels by `(patient_id, study_id)` parsed from IDs
- Converts NIH `Finding Labels` pipe list to fixed 14-label vector
- `No Finding` maps to all-zero label vector
