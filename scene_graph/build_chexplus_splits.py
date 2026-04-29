"""
Build CheXplus patient-level split scores from scene graph CSV.

Outputs to C:/Users/dhint/CHEX-DATA/CHEXPLUS/metadata:
- splits.csv
- split_score_value_counts.csv
- split_tier_value_counts.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    metadata_dir = Path(r"C:\Users\dhint\CHEX-DATA\CHEXPLUS\metadata")
    scene_graph_csv = metadata_dir / "chexplus_scene_graph.csv"
    splits_csv = metadata_dir / "splits.csv"
    counts_csv = metadata_dir / "split_score_value_counts.csv"
    tier_counts_csv = metadata_dir / "split_tier_value_counts.csv"
    split_counts_csv = metadata_dir / "split_value_counts.csv"

    df = pd.read_csv(scene_graph_csv)

    for col in ["patient_id", "study_id", "anatomy", "presence", "temporal_status"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Spatial score per study:
    # count rows where anatomy is not null and presence in {present, absent}
    anatomy_ok = df["anatomy"].notna() & df["anatomy"].astype(str).str.strip().ne("")
    presence_ok = df["presence"].fillna("").astype(str).str.lower().isin(["present", "absent"])
    df["spatial_flag"] = (anatomy_ok & presence_ok).astype(int)

    # Temporal score per study:
    # count rows where temporal_status is not null/non-empty
    temporal_ok = df["temporal_status"].notna() & df["temporal_status"].astype(str).str.strip().ne("")
    df["temporal_flag"] = temporal_ok.astype(int)

    study_scores = (
        df.groupby(["patient_id", "study_id"], as_index=False)
        .agg(spatial_score=("spatial_flag", "sum"), temporal_score=("temporal_flag", "sum"))
    )

    patient_scores = (
        study_scores.groupby("patient_id", as_index=False)
        .agg(max_spatial_score=("spatial_score", "max"), max_temporal_score=("temporal_score", "max"))
        .sort_values("patient_id")
    )

    def classify_tier(row: pd.Series) -> str:
        if row["max_spatial_score"] >= 2 and row["max_temporal_score"] >= 5:
            return "diamond"
        if row["max_temporal_score"] >= 1 or row["max_spatial_score"] >= 4:
            return "gold"
        return "silver"

    patient_scores["patient_tier"] = patient_scores.apply(classify_tier, axis=1)

    patient_scores["split"] = "train"

    diamond_mask = patient_scores["patient_tier"].eq("diamond")
    diamond_patients = patient_scores.loc[diamond_mask].sort_values("patient_id")
    if len(diamond_patients) < 3000:
        raise ValueError(f"Expected at least 3000 diamond patients, found {len(diamond_patients)}")

    test_patients = diamond_patients.sample(n=3000, random_state=42)
    patient_scores.loc[test_patients.index, "split"] = "test"
    patient_scores.loc[diamond_mask & ~patient_scores.index.isin(test_patients.index), "split"] = "val"

    patient_scores = patient_scores[["patient_id", "split", "patient_tier", "max_spatial_score", "max_temporal_score"]]

    patient_scores.to_csv(splits_csv, index=False)

    spatial_counts = (
        patient_scores["max_spatial_score"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("score")
        .reset_index(name="count")
    )
    spatial_counts.insert(0, "metric", "max_spatial_score")

    temporal_counts = (
        patient_scores["max_temporal_score"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("score")
        .reset_index(name="count")
    )
    temporal_counts.insert(0, "metric", "max_temporal_score")

    pd.concat([spatial_counts, temporal_counts], ignore_index=True).to_csv(counts_csv, index=False)

    tier_counts = (
        patient_scores["patient_tier"]
        .value_counts(dropna=False)
        .rename_axis("patient_tier")
        .reset_index(name="count")
        .sort_values("patient_tier")
    )
    tier_counts.to_csv(tier_counts_csv, index=False)

    split_counts = (
        patient_scores["split"]
        .value_counts(dropna=False)
        .rename_axis("split")
        .reset_index(name="count")
        .sort_values("split")
    )
    split_counts.to_csv(split_counts_csv, index=False)

    print(f"Wrote {splits_csv}")
    print(f"Wrote {counts_csv}")
    print(f"Wrote {tier_counts_csv}")
    print(f"Wrote {split_counts_csv}")
    print(f"Patients: {len(patient_scores)}")
    print("Tier counts:\n", tier_counts.to_string(index=False))
    print("Split counts:\n", split_counts.to_string(index=False))


if __name__ == "__main__":
    main()
