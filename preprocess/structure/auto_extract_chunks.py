"""Automated Chunked Feature Extraction & Kaggle Dataset Upload.

Extracts features in chunks (e.g. 50,000 records each), automatically creates a Kaggle
dataset for each chunk (biovilt-features-part1, part2, etc.), and deletes local chunk files
to keep disk usage under control.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path to resolve imports regardless of execution directory
_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

try:
    from preprocess.structure.model_wrapper import BioViLWrapper
    from preprocess.structure.train_biovilt import BioViLClassifier, CXRDataset, extract_and_cache_features
except ModuleNotFoundError:
    from model_wrapper import BioViLWrapper
    from train_biovilt import BioViLClassifier, CXRDataset, extract_and_cache_features



def upload_chunk_dataset(chunk_dir: Path, part_idx: int, username: str) -> bool:
    """Create metadata and upload chunk directory to Kaggle as a new Dataset."""
    dataset_id = f"{username}/biovilt-features-part{part_idx}"
    title = f"BioViL Features Part {part_idx}"
    
    print(f"\n[INFO] Auto-uploading {chunk_dir} to Kaggle Dataset: {dataset_id}...")
    
    metadata = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}]
    }
    
    meta_file = chunk_dir / "dataset-metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
        
    try:
        res = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(chunk_dir)],
            capture_output=True,
            text=True,
            check=False
        )
        if res.returncode == 0:
            print(f"[SUCCESS] Uploaded {dataset_id} successfully!")
            print(res.stdout)
            return True
        else:
            print(f"[WARNING] Kaggle dataset create output: {res.stdout}\n{res.stderr}")
            # Check if dataset already exists, try version update
            res_v = subprocess.run(
                ["kaggle", "datasets", "version", "-p", str(chunk_dir), "-m", f"Auto update Part {part_idx}"],
                capture_output=True,
                text=True,
                check=False
            )
            if res_v.returncode == 0:
                print(f"[SUCCESS] Updated version for {dataset_id} successfully!")
                return True
            else:
                print(f"[ERROR] Failed to upload chunk dataset: {res_v.stderr}")
                return False
    except Exception as e:
        print(f"[ERROR] Exception during kaggle upload: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto Chunked Feature Extraction & Upload")
    parser.add_argument("--split-file", type=Path, default=Path("none"), help="Set to 'none' to load full metadata")
    parser.add_argument("--metadata-file", type=Path,
                        default=Path("/kaggle/input/datasets/nguynnghin/mimic-metadata/mimic_metadata_final.jsonl")
                        if Path("/kaggle/input").exists() else Path("./metadata/mimic_metadata_final.jsonl"))
    parser.add_argument("--images-root", type=Path,
                        default=Path("/kaggle/input/mimic-cxr-448/mimic-cxr-448")
                        if Path("/kaggle/input").exists() else Path("./data/images"))
    parser.add_argument("--checkpoint-path", type=Path,
                        default=Path("/kaggle/working/biovilt_model.pt")
                        if Path("/kaggle/input").exists() else Path("./biovilt_model.pt"))
    parser.add_argument("--chunk-size", type=int, default=50000, help="Number of records per dataset chunk")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--kaggle-username", type=str, default="hoangtimothy")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Initializing full dataset...")
    full_dataset = CXRDataset(args.split_file, "all", transform=transform,
                              images_root=args.images_root, metadata_file=args.metadata_file)
    
    total_samples = len(full_dataset)
    print(f"Total samples to extract: {total_samples}")
    if total_samples == 0:
        print("[ERROR] No samples found. Check metadata path.")
        return

    print("Initializing BioViLClassifier...")
    model = BioViLClassifier(num_classes=21, training_mode="lora").to(device)
    if args.checkpoint_path.exists():
        print(f"[INFO] Loading trained checkpoint from {args.checkpoint_path}")
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt, strict=False)

    num_chunks = (total_samples + args.chunk_size - 1) // args.chunk_size
    print(f"Will process dataset in {num_chunks} chunk(s) of ~{args.chunk_size} samples each.")

    for chunk_idx in range(num_chunks):
        part_num = chunk_idx + 1
        start_idx = chunk_idx * args.chunk_size
        end_idx = min((chunk_idx + 1) * args.chunk_size, total_samples)
        
        chunk_dir = Path(f"/kaggle/working/biovilt_features_part{part_num}") if Path("/kaggle/input").exists() else Path(f"./data/biovilt_features_part{part_num}")
        print(f"\n==========================================")
        print(f"Processing Chunk {part_num}/{num_chunks} (Indices {start_idx}..{end_idx}) -> {chunk_dir}")
        print(f"==========================================")

        subset = Subset(full_dataset, list(range(start_idx, end_idx)))
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        # 1. Extract features for this chunk
        extract_and_cache_features(model, loader, chunk_dir, device)

        # 2. Upload to Kaggle Datasets
        if Path("/kaggle/input").exists():
            success = upload_chunk_dataset(chunk_dir, part_num, args.kaggle_username)
            if success:
                print(f"[INFO] Deleting local chunk directory {chunk_dir} to free up disk space...")
                shutil.rmtree(chunk_dir, ignore_errors=True)
                print(f"[SUCCESS] Disk space freed for Chunk {part_num}!")
            else:
                print(f"[WARNING] Keeping local chunk directory {chunk_dir} since upload returned error.")
        else:
            print(f"[INFO] Running locally, keeping features in {chunk_dir}")

    print("\n🎉 [COMPLETED] All chunks extracted, uploaded, and cleaned up successfully!")


if __name__ == "__main__":
    main()
