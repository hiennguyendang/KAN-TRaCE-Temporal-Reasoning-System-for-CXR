from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

# Rare disease indices based on the 21 unified labels mapping:
# 8: Fracture, 9: Lung Lesion, 12: Pleural Other
# 15: Mass, 16: Nodule, 17: Emphysema, 18: Fibrosis, 19: Pleural Thickening, 20: Hernia
RARE_DISEASE_INDICES = {8, 9, 12, 15, 16, 17, 18, 19, 20}
NO_FINDING_INDEX = 11  # "No Finding" unified label index


def get_patient_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Computes aggregated stats for a single patient from all their records."""
    unique_studies = set()
    has_rare_disease = False
    has_scene_graph = False
    max_report_len = 0
    all_normal_or_no_finding = True

    for r in records:
        # Study count
        study_id = r.get("study_id")
        if study_id:
            unique_studies.add(study_id)

        # Labels parsing
        labels = r.get("labels", [])
        if labels and len(labels) == 21:
            # Check for rare disease
            for idx in RARE_DISEASE_INDICES:
                if labels[idx] == 1.0:
                    has_rare_disease = True
            
            # Check if patient has any abnormal finding
            # If any index other than NO_FINDING_INDEX (11) and Support Devices (13) is positive:
            has_active_disease = False
            for idx, val in enumerate(labels):
                if idx not in {NO_FINDING_INDEX, 13} and val == 1.0:
                    has_active_disease = True
            if has_active_disease:
                all_normal_or_no_finding = False

        # Scene graph check
        bboxes = r.get("bboxes", [])
        if bboxes and len(bboxes) > 0:
            has_scene_graph = True

        # Report length check
        report = r.get("report", "")
        max_report_len = max(max_report_len, len(report))

    return {
        "study_count": len(unique_studies),
        "has_rare_disease": has_rare_disease,
        "has_scene_graph": has_scene_graph,
        "max_report_length": max_report_len,
        "has_only_no_finding": all_normal_or_no_finding,
    }


def compute_patient_score(
    stats: dict[str, Any], A: int = 150, B: int = 2, C: int = 80
) -> float:
    """Computes patient score based on the criteria rules."""
    score = 0.0

    if stats["has_rare_disease"]:
        score += 100.0

    if stats["has_scene_graph"]:
        score += 80.0

    if stats["max_report_length"] > A:
        score += 50.0

    if stats["study_count"] >= B:
        score += 30.0

    if stats["has_only_no_finding"] and stats["max_report_length"] < C:
        score -= 50.0

    return score


def load_records_from_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def generate_mock_jsonl(path: Path, dataset_name: str, num_records: int = 100):
    """Generates mock unified 21-dimensional records for testing/placeholders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(num_records):
        patient_id = f"p{10000 + i // 3}"  # 3 images per patient on average
        study_id = f"s{20000 + i // 2}"
        image_id = f"im_{dataset_name}_{i}"
        
        # 21-dimensional labels
        labels = [0.0] * 21
        # Randomly assign diseases
        if random.random() < 0.2:
            labels[NO_FINDING_INDEX] = 1.0
        else:
            # Maybe add a rare disease
            if random.random() < 0.15:
                rare_idx = random.choice(list(RARE_DISEASE_INDICES))
                labels[rare_idx] = 1.0
            # Maybe add common diseases
            for common_idx in [0, 1, 2, 3, 4, 5, 6]:
                if random.random() < 0.2:
                    labels[common_idx] = 1.0

        # Scene graph bboxes
        bboxes = []
        if random.random() < 0.4:
            bboxes = [{"label": "heart", "bbox": [10, 20, 50, 60]}]

        # Reports
        report_templates = [
            "FINDINGS: The lungs are clear. Cardiomegaly is present. IMPRESSION: Cardiomegaly without acute infiltration.",
            "FINDINGS: Minimal atelectasis. No pneumothorax. IMPRESSION: Stable chest.",
            "FINDINGS: Clear chest. IMPRESSION: Normal.",
        ]
        report = random.choice(report_templates)
        if random.random() < 0.2:
            # long report
            report = report * 3

        records.append({
            "image_id": image_id,
            "image_path": f"/mock_path/{dataset_name}/{image_id}.png",
            "dataset": dataset_name,
            "labels": labels,
            "bboxes": bboxes,
            "findings": report.split(";")[0],
            "report": report,
            "patient_id": patient_id,
            "study_id": study_id,
        })

    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Patient-level dataset selection and splitting")
    parser.add_argument("--mimic-meta", type=Path, default=Path(r"C:\Users\dhint\CHEX-DATA\MIMIC-CXR\metadata\mimic_metadata_unified_21dim.jsonl"))
    parser.add_argument("--chexplus-meta", type=Path, default=Path(r"C:\Users\dhint\CHEX-DATA\CHEXPLUS\metadata\chexplus_metadata_unified_21dim.jsonl"))
    parser.add_argument("--nih-meta", type=Path, default=Path(r"C:\Users\dhint\CHEX-DATA\NIH\metadata\nih_metadata_unified_21dim.jsonl"))
    parser.add_argument("--output-split-path", type=Path, default=Path("./selected_patient_splits.json"))
    
    # Thresholds
    parser.add_argument("-A", type=int, default=150, help="Min report length for positive weight")
    parser.add_argument("-B", type=int, default=2, help="Min studies for temporal sequence weight")
    parser.add_argument("-C", type=int, default=80, help="Max report length for negative normal weight")
    
    # Selection parameters
    parser.add_argument("--num-patients", type=int, default=5000, help="Target number of patients to select")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate-mocks", action="store_true", help="Generate mock files if actual metadata is missing")

    args = parser.parse_args()
    random.seed(args.seed)

    # Check for paths and generate mocks if requested/missing
    paths_exist = args.mimic_meta.exists() and args.chexplus_meta.exists() and args.nih_meta.exists()
    if not paths_exist:
        print("[WARNING] Actual metadata files not found at default paths.")
        if args.generate_mocks:
            print("[INFO] Generating mock metadata placeholders...")
            mock_dir = Path("./mock_metadata")
            args.mimic_meta = mock_dir / "mimic_mock.jsonl"
            args.chexplus_meta = mock_dir / "chexplus_mock.jsonl"
            args.nih_meta = mock_dir / "nih_mock.jsonl"
            generate_mock_jsonl(args.mimic_meta, "mimic", 500)
            generate_mock_jsonl(args.chexplus_meta, "chexplus", 400)
            generate_mock_jsonl(args.nih_meta, "nih", 300)
            print(f"[SUCCESS] Mock files written to: {mock_dir.resolve()}")
        else:
            print("[ERROR] Cannot proceed without metadata. Run with --generate-mocks to use synthetic placeholder data.")
            return

    # Load all records
    print("Loading records...")
    records = []
    for name, path in [("MIMIC-CXR", args.mimic_meta), ("CheXplus", args.chexplus_meta), ("NIH", args.nih_meta)]:
        rec_list = load_records_from_jsonl(path)
        print(f"  Loaded {len(rec_list)} records from {name}")
        records.extend(rec_list)

    # Group records by patient_id
    patient_records: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        pid = r.get("patient_id")
        if pid:
            patient_records.setdefault(pid, []).append(r)

    print(f"Total unique patients across datasets: {len(patient_records)}")

    # Score patients
    scored_patients = []
    for pid, p_recs in patient_records.items():
        stats = get_patient_stats(p_recs)
        score = compute_patient_score(stats, A=args.A, B=args.B, C=args.C)
        scored_patients.append({
            "patient_id": pid,
            "score": score,
            "stats": stats,
            "records": p_recs,
        })

    # Sort by score descending
    scored_patients.sort(key=lambda x: x["score"], reverse=True)

    # Print top scoring samples details
    print("\n--- Top Scoring Patients Samples ---")
    for i in range(min(5, len(scored_patients))):
        p = scored_patients[i]
        print(f"Patient ID: {p['patient_id']} | Score: {p['score']} | Stats: {p['stats']}")

    # Select target patients count
    target_count = min(args.num_patients, len(scored_patients))
    selected = scored_patients[:target_count]
    print(f"\nSelected top {target_count} patients based on scores.")

    # Shuffle selected patients list to assign to splits randomly (patient-disjoint)
    random.shuffle(selected)

    num_train = int(target_count * args.train_ratio)
    num_val = int(target_count * args.val_ratio)
    
    train_patients = selected[:num_train]
    val_patients = selected[num_train:num_train + num_val]
    test_patients = selected[num_train + num_val:]

    print(f"Split results (patient-disjoint):")
    print(f"  Train: {len(train_patients)} patients")
    print(f"  Val: {len(val_patients)} patients")
    print(f"  Test: {len(test_patients)} patients")

    # Serialize output mapping image_id to split
    split_mapping = {
        "train": [],
        "val": [],
        "test": []
    }
    
    for split_name, patient_list in [("train", train_patients), ("val", val_patients), ("test", test_patients)]:
        for p in patient_list:
            for r in p["records"]:
                split_mapping[split_name].append({
                    "image_id": r["image_id"],
                    "image_path": r["image_path"],
                    "patient_id": r["patient_id"],
                    "study_id": r["study_id"],
                    "dataset": r["dataset"],
                    "labels": r["labels"],
                    "report": r["report"],
                })

    args.output_split_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_split_path.open("w", encoding="utf-8") as f:
        json.dump(split_mapping, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Saved split mapping json file to: {args.output_split_path.resolve()}")


if __name__ == "__main__":
    main()
