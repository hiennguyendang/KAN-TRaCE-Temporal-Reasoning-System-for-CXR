from __future__ import annotations

import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from typing import Any

# Import our custom wrapper
try:
    from preprocess.structure.model_wrapper import BioViLWrapper
except ModuleNotFoundError:
    from preprocess.structure.model_wrapper import BioViLWrapper

# Ignore index value for BCE loss
IGNORE_INDEX = -100.0


class CXRDataset(Dataset):
    """
    Chest X-ray Dataset loading from the generated split JSON mapping file.
    """
    def __init__(self, split_file: Path, split_name: str, transform: Any = None) -> None:
        super().__init__()
        self.transform = transform
        self.records = []
        
        if split_file.exists():
            with split_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                self.records = data.get(split_name, [])
        else:
            print(f"[WARNING] Split file {split_file} not found. Running dataset in empty mode.")
            
        self.warned_missing = False

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        image_path = rec.get("image_path", "")
        labels_list = rec.get("labels", [0.0] * 21)
        
        # Load image
        img = None
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception as e:
                pass
                
        if img is None:
            if not self.warned_missing:
                print(f"[INFO] Some image files are missing (e.g. {image_path}). Using random dummy tensors for validation.")
                self.warned_missing = True
            # Create a dummy image
            img = Image.new("RGB", (512, 512), color="gray")

        if self.transform:
            img = self.transform(img)
            
        labels = torch.tensor(labels_list, dtype=torch.float32)
        return img, labels


# Custom Native LoRA implementation for PyTorch to avoid external dependencies
class LoRALinear(nn.Module):
    """Wraps a linear layer with Low-Rank Adaptation (LoRA)."""
    def __init__(self, linear_layer: nn.Linear, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.linear = linear_layer
        
        # Freeze base weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        # LoRA adapters
        self.lora_A = nn.Parameter(torch.randn(in_features, r) / r)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


class BioViLClassifier(nn.Module):
    """
    Classifier wrapping BioViLWrapper and adding a multi-label classification head.
    Supports 4 training modes: linear_probe, lora, partial_unfreeze, and full.
    """
    def __init__(self, num_classes: int = 21, training_mode: str = "linear_probe") -> None:
        super().__init__()
        self.image_encoder = BioViLWrapper()
        
        # Binary Classification head for 21 unified categories
        # Takes the 128-dimensional global visual embedding as input
        self.classifier = nn.Linear(128, num_classes)
        self.training_mode = training_mode
        self._configure_gradients()

    def _configure_gradients(self) -> None:
        # First, unfreeze everything
        for param in self.parameters():
            param.requires_grad = True
            
        if self.training_mode == "linear_probe":
            print("[INFO] Setting up model for LINEAR PROBING. Freezing visual encoder backbone and projection layers.")
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            # Only self.classifier is trainable
            
        elif self.training_mode == "lora":
            print("[INFO] Setting up model for LoRA (PEFT). Freezing visual encoder and injecting low-rank adapters.")
            # Freeze everything first
            for param in self.parameters():
                param.requires_grad = False
            
            # Unfreeze classifier head
            for param in self.classifier.parameters():
                param.requires_grad = True
                
            # Wrap global projector with LoRA
            self.image_encoder.global_projector = LoRALinear(self.image_encoder.global_projector, r=8, alpha=16.0)
            
            # Make sure lora parameters require grad
            for param in self.image_encoder.global_projector.lora_A, self.image_encoder.global_projector.lora_B:
                param.requires_grad = True
                
        elif self.training_mode == "partial_unfreeze":
            print("[INFO] Setting up model for PARTIAL UNFREEZING. Freezing early ResNet layers, keeping layer4 + projectors trainable.")
            # Freeze all backbone layers first
            for param in self.image_encoder.backbone.parameters():
                param.requires_grad = False
                
            # Unfreeze layer4 of ResNet backbone
            for param in self.image_encoder.backbone.layer4.parameters():
                param.requires_grad = True
                
            # Keep projection layers trainable
            for param in self.image_encoder.patch_projector.parameters():
                param.requires_grad = True
            for param in self.image_encoder.global_projector.parameters():
                param.requires_grad = True
            for param in self.classifier.parameters():
                param.requires_grad = True
                
        elif self.training_mode == "full":
            print("[INFO] Setting up model for FULL FINE-TUNING. Unfreezing 100% of parameters.")
            # All remains trainable
            
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract features
        patch_features, global_emb = self.image_encoder(x)
        # Classify using global embedding
        logits = self.classifier(global_emb)
        return logits, patch_features


def compute_masked_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes binary cross entropy loss masking out values set to IGNORE_INDEX (-100)."""
    # Raw BCE loss per label element
    bce = nn.BCEWithLogitsLoss(reduction='none')
    loss = bce(logits, targets)
    
    # Create mask: 1.0 for valid targets, 0.0 for ignored ones
    mask = (targets != IGNORE_INDEX).float()
    
    # Apply mask and compute mean over valid elements
    masked_loss_val = (loss * mask).sum() / (mask.sum() + 1e-8)
    return masked_loss_val


def train_one_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: optim.Optimizer, 
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(imgs)
        loss = compute_masked_loss(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _ = model(imgs)
            loss = compute_masked_loss(logits, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main() -> None:
    parser = argparse.ArgumentParser(description="BioViL-T Fine-Tuning strategies")
    parser.add_argument("--split-file", type=Path, default=Path("./selected_patient_splits.json"))
    parser.add_argument(
        "--mode", 
        type=str, 
        default="linear_probe", 
        choices=["linear_probe", "lora", "partial_unfreeze", "full"],
        help="Training mode: linear_probe (fastest), lora (efficient), partial_unfreeze (moderate), full (slowest)"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Standard transformations for ResNet-50
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("Initializing datasets...")
    train_dataset = CXRDataset(args.split_file, "train", transform=transform)
    val_dataset = CXRDataset(args.split_file, "val", transform=transform)

    print(f"Dataset stats: Train samples={len(train_dataset)}, Val samples={len(val_dataset)}")
    if len(train_dataset) == 0:
        print("[ERROR] Train split contains 0 records. Please run select_dataset.py first to populate the splits.")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    print(f"Initializing BioViLClassifier in mode: {args.mode}")
    model = BioViLClassifier(num_classes=21, training_mode=args.mode)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: Total = {total_params:,} | Trainable = {trainable_params:,} ({trainable_params/total_params:.2%})")

    # Set optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Training Loop
    print("\nStarting training loop...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

    print("\n[SUCCESS] Training simulation completed successfully!")


if __name__ == "__main__":
    main()
