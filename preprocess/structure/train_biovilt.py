import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path to resolve imports regardless of execution directory
_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any


import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm

# Import custom wrapper
try:
    from preprocess.structure.model_wrapper import BioViLWrapper
except ModuleNotFoundError:
    try:
        from model_wrapper import BioViLWrapper
    except ModuleNotFoundError:
        from .model_wrapper import BioViLWrapper

# Ignore index value for BCE loss
IGNORE_INDEX = -100.0



class CXRDataset(Dataset):
    """
    Chest X-ray Dataset loading from split JSON mapping file or metadata JSONL file,
    resolving image files dynamically against images_root (e.g. Kaggle datasets).
    """
    def __init__(
        self,
        split_file: Path | None = None,
        split_name: str = "train",
        transform: Any = None,
        images_root: Path | None = None,
        metadata_file: Path | None = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.images_root = Path(images_root) if images_root else None
        self.records = []

        # 1. Try loading from split JSON (skip if split_file is 'none' or None)
        if split_file and str(split_file).lower() != "none" and Path(split_file).exists():
            with Path(split_file).open("r", encoding="utf-8") as f:
                data = json.load(f)
                if split_name == "all":
                    self.records = data.get("train", []) + data.get("val", []) + data.get("test", [])
                else:
                    self.records = data.get(split_name, [])


        # 2. Fallback to metadata JSONL if split_file records are empty or missing
        if not self.records and metadata_file:
            meta_path = Path(metadata_file)
            if not meta_path.exists():
                kaggle_input = Path("/kaggle/input")
                if kaggle_input.exists():
                    found = [p for p in kaggle_input.rglob("*.jsonl") if "metadata" in p.name.lower()]
                    if not found:
                        found = list(kaggle_input.rglob("*.jsonl"))
                    if found:
                        meta_path = found[0]
                        print(f"[INFO] Auto-located metadata file on Kaggle: {meta_path}")

            if meta_path.exists():
                import hashlib
                print(f"[INFO] Loading records directly from metadata file: {meta_path} for split: {split_name}")
                with meta_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line.strip())
                        if split_name == "all":
                            self.records.append(rec)
                        elif "split" in rec and rec["split"] is not None:
                            if rec["split"] == split_name:
                                self.records.append(rec)
                        else:
                            # Deterministic patient-disjoint split: 90% train, 10% val
                            pid = str(rec.get("patient_id", rec.get("subject_id", rec.get("image_id", ""))))
                            h_val = int(hashlib.md5(pid.encode()).hexdigest(), 16) % 10
                            rec_split = "val" if h_val == 9 else "train"
                            if rec_split == split_name:
                                self.records.append(rec)
            else:
                print(f"[ERROR] Metadata file not found at: {metadata_file}")



        if not self.records:
            print(f"[WARNING] No records found for split '{split_name}'. Running dataset in empty mode.")

        self.warned_missing = False

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, rec: dict[str, Any]) -> Path | None:
        """Resolve physical image path from record metadata."""
        raw_path = rec.get("image_path", "")
        if raw_path and os.path.exists(raw_path):
            return Path(raw_path)

        if not self.images_root or not self.images_root.exists():
            return None

        # Try joining raw_path with images_root
        if raw_path:
            p = self.images_root / raw_path.lstrip("/")
            if p.exists():
                return p

        # MIMIC-CXR folder hierarchy: p10/p10000032/s50414267/<dicom_id>.jpg
        pid = str(rec.get("patient_id", rec.get("subject_id", "")))
        sid = str(rec.get("study_id", ""))
        did = str(rec.get("dicom_id", rec.get("image_id", "")))

        if pid and sid and did:
            group_folder = f"p{pid[:2]}" if not pid.startswith("p") else pid[:3]
            for ext in [".jpg", ".png"]:
                p = self.images_root / group_folder / pid / f"s{sid}" / f"{did}{ext}"
                if p.exists():
                    return p
                p2 = self.images_root / pid / f"s{sid}" / f"{did}{ext}"
                if p2.exists():
                    return p2

        return None

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        rec = self.records[idx]
        dicom_id = str(rec.get("dicom_id", rec.get("image_id", f"idx_{idx}")))
        labels_list = rec.get("labels", rec.get("labels_14", [0.0] * 21))

        # Pad or slice labels to 21 elements
        if len(labels_list) < 21:
            labels_list = list(labels_list) + [IGNORE_INDEX] * (21 - len(labels_list))
        else:
            labels_list = labels_list[:21]

        resolved_path = self._resolve_image_path(rec)
        img = None
        if resolved_path:
            try:
                img = Image.open(resolved_path).convert("RGB")
            except Exception:
                pass

        if img is None:
            if not self.warned_missing:
                print(f"[INFO] Some image files missing. Using dummy tensor placeholders. Sample ID: {dicom_id}")
                self.warned_missing = True
            img = Image.new("RGB", (512, 512), color="gray")

        if self.transform:
            img = self.transform(img)

        labels = torch.tensor(labels_list, dtype=torch.float32)
        return img, labels, dicom_id


class LoRALinear(nn.Module):
    """Wraps a linear layer with Low-Rank Adaptation (LoRA)."""
    def __init__(self, linear_layer: nn.Linear, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        self.lora_A = nn.Parameter(torch.randn(in_features, r) / r)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


class BioViLClassifier(nn.Module):
    """
    Classifier wrapping BioViLWrapper and adding a multi-label classification head.
    Preserves both patch_features [196, 512] and global CLS token [512].
    """
    def __init__(self, num_classes: int = 21, training_mode: str = "linear_probe", pretrained_weights_path: str | None = None) -> None:
        super().__init__()
        self.image_encoder = BioViLWrapper(pretrained_weights_path=pretrained_weights_path, pretrained_imagenet=True)
        self.classifier = nn.Linear(128, num_classes)
        self.training_mode = training_mode
        self._configure_gradients()


    def _configure_gradients(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

        if self.training_mode == "linear_probe":
            print("[INFO] Setting up model for LINEAR PROBING. Freezing backbone.")
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        elif self.training_mode == "lora":
            print("[INFO] Setting up model for LoRA PEFT.")
            for param in self.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = True
            self.image_encoder.global_projector = LoRALinear(self.image_encoder.global_projector, r=8, alpha=16.0)
            self.image_encoder.global_projector.lora_A.requires_grad = True
            self.image_encoder.global_projector.lora_B.requires_grad = True
        elif self.training_mode == "partial_unfreeze":
            print("[INFO] Setting up model for PARTIAL UNFREEZING.")
            for param in self.image_encoder.backbone.parameters():
                param.requires_grad = False
            for param in self.image_encoder.backbone.layer4.parameters():
                param.requires_grad = True
            for param in self.image_encoder.patch_projector.parameters():
                param.requires_grad = True
            for param in self.image_encoder.global_projector.parameters():
                param.requires_grad = True
            for param in self.classifier.parameters():
                param.requires_grad = True
        elif self.training_mode == "full":
            print("[INFO] Setting up model for FULL FINE-TUNING.")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (batch_size, num_classes)
            patch_features: (batch_size, 196, 512)
            cls_token: (batch_size, 512) global visual CLS token
        """
        patch_features, global_emb, cls_token = self.image_encoder(x)
        logits = self.classifier(global_emb)
        return logits, patch_features, cls_token


def compute_masked_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = nn.BCEWithLogitsLoss(reduction='none')
    loss = bce(logits, targets)
    mask = (targets != IGNORE_INDEX).float()
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_idx: int = 0
) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch_idx+1}", leave=False)
    for imgs, labels, _ in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _, _ = model(imgs)
        loss = compute_masked_loss(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(len(dataloader), 1)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _, _ = model(imgs)
            loss = compute_masked_loss(logits, labels)
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def extract_and_cache_features(
    model: nn.Module,
    dataloader: DataLoader,
    output_dir: Path,
    device: torch.device
) -> None:
    """Extract and cache features (.pt files) for downstream C-MLP (Phase 3).

    Saves a float16 tensor of shape [197, 512] for each study:
    - Token 0: CLS token [1, 512]
    - Tokens 1..196: spatial patch features [196, 512]
    """
    model.eval()
    output_dir = Path(output_dir)
    
    is_zip = output_dir.suffix == ".zip"
    if is_zip:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        import zipfile
        import io
        print(f"\n[INFO] Extracting & caching BioViL-T features to ZIP file: {output_dir}")
        
        # Build set of existing names in zip if file already exists
        existing = set()
        if output_dir.exists():
            try:
                with zipfile.ZipFile(output_dir, "r") as zf:
                    existing = set(zf.namelist())
            except Exception:
                pass
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[INFO] Extracting & caching BioViL-T features (with CLS token) to: {output_dir}")

    count = 0
    skipped = 0
    pbar = tqdm(dataloader, desc="Caching Features", leave=True)
    
    zf = None
    if is_zip:
        zf = zipfile.ZipFile(output_dir, "a", compression=zipfile.ZIP_STORED)

    try:
        for imgs, _, dicom_ids in pbar:
            # Check if all files in batch already exist to skip GPU pass
            batch_needed = False
            for did in dicom_ids:
                if is_zip:
                    if f"{did}.pt" not in existing:
                        batch_needed = True
                        break
                else:
                    if not (output_dir / f"{did}.pt").exists():
                        batch_needed = True
                        break

            if not batch_needed:
                skipped += len(dicom_ids)
                pbar.set_postfix(saved=count, skipped=skipped)
                continue

            imgs = imgs.to(device)
            _, patch_features, cls_tokens = model(imgs)

            B = imgs.size(0)
            for b in range(B):
                did = dicom_ids[b]
                if is_zip:
                    fname = f"{did}.pt"
                    if fname in existing:
                        skipped += 1
                        continue
                    
                    cls_t = cls_tokens[b].unsqueeze(0)  # [1, 512]
                    patches = patch_features[b]         # [196, 512]
                    combined_feat = torch.cat([cls_t, patches], dim=0).half().cpu()
                    
                    buf = io.BytesIO()
                    torch.save(combined_feat, buf)
                    zf.writestr(fname, buf.getvalue())
                    existing.add(fname)
                    count += 1
                else:
                    out_path = output_dir / f"{did}.pt"
                    if out_path.exists():
                        skipped += 1
                        continue

                    cls_t = cls_tokens[b].unsqueeze(0)  # [1, 512]
                    patches = patch_features[b]         # [196, 512]
                    combined_feat = torch.cat([cls_t, patches], dim=0).half().cpu()

                    torch.save(combined_feat, out_path)
                    count += 1
            pbar.set_postfix(saved=count, skipped=skipped)
    finally:
        if zf is not None:
            zf.close()

    print(f"[SUCCESS] Cached features for {count} studies to {output_dir}")



def main() -> None:
    parser = argparse.ArgumentParser(description="BioViL-T Fine-Tuning & Feature Extraction")
    parser.add_argument("--split-file", type=Path, default=Path("./selected_patient_splits.json"))
    parser.add_argument("--metadata-file", type=Path,
                        default=Path("/kaggle/input/mimic-metadata/mimic_metadata_final.jsonl")
                        if Path("/kaggle/input").exists() else Path("./metadata/mimic_metadata_final.jsonl"))
    parser.add_argument("--images-root", type=Path,
                        default=Path("/kaggle/input/mimic-cxr-448/mimic-cxr-448")
                        if Path("/kaggle/input").exists() else Path("./data/images"))
    parser.add_argument("--feature-cache-dir", type=Path, default=Path("/kaggle/working/biovilt_features")
                        if Path("/kaggle/input").exists() else Path("./data/feature_cache"))
    parser.add_argument(
        "--mode",
        type=str,
        default="lora",
        choices=["linear_probe", "lora", "partial_unfreeze", "full"],
        help="Training mode: lora (recommended on Kaggle), linear_probe, partial_unfreeze, full"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--extract-features", action="store_true",
                        help="Extract & save [197, 512] feature cache after training")
    parser.add_argument("--checkpoint-path", type=Path,
                        default=Path("/kaggle/working/biovilt_model.pt")
                        if Path("/kaggle/input").exists() else Path("./biovilt_model.pt"),
                        help="Path to save/load trained model checkpoint")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("Initializing datasets...")
    train_dataset = CXRDataset(args.split_file, "train", transform=transform,
                               images_root=args.images_root, metadata_file=args.metadata_file)
    val_dataset = CXRDataset(args.split_file, "val", transform=transform,
                             images_root=args.images_root, metadata_file=args.metadata_file)

    print(f"Dataset stats: Train samples={len(train_dataset)}, Val samples={len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Initializing BioViLClassifier in mode: {args.mode}")
    model = BioViLClassifier(num_classes=21, training_mode=args.mode).to(device)

    # Load checkpoint if exists
    if args.checkpoint_path and Path(args.checkpoint_path).exists():
        print(f"[INFO] Loading trained model weights from checkpoint: {args.checkpoint_path}")
        try:
            ckpt = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(ckpt, strict=False)
            print("[SUCCESS] Checkpoint weights loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: Total = {total_params:,} | Trainable = {trainable_params:,} ({trainable_params/total_params:.2%})")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    if len(train_dataset) > 0 and args.epochs > 0:
        print("\nStarting training loop...")
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch_idx=epoch)
            val_loss = evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

        # Save trained model checkpoint
        if args.checkpoint_path:
            args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f"[SUCCESS] Saved trained model checkpoint to: {args.checkpoint_path}")

    if args.extract_features:
        full_dataset = CXRDataset(args.split_file, "all", transform=transform,
                                  images_root=args.images_root, metadata_file=args.metadata_file)
        full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        extract_and_cache_features(model, full_loader, args.feature_cache_dir, device)

    print("\n[SUCCESS] Phase 1 run completed successfully!")


if __name__ == "__main__":
    main()

