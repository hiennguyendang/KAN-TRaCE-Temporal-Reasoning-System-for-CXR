"""Training loop for Phase 3 C-MLP.

Key design decisions:
- Early-stop / checkpoint by macro-F1 (NOT loss — loss is dominated by majority class).
- AdamW with weight decay.
- pos_weight computed once from training labels.
- Gradient clipping for stability (attention-pool can spike).
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import parse_args
from .constants import MASK_VALUE, NUM_DISEASES
from .dataset import CachedFeatureDataset, load_metadata_jsonl
from .losses import build_loss, compute_pos_weight_log, compute_pos_weight_inverse
from .metrics import compute_all_metrics
from .model import CMLP

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "feats": torch.stack([b["feats"] for b in batch]),
        "boxes": torch.stack([b["boxes"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def _compute_pos_weight(
    train_records: list[dict],
    strategy: str,
    device: torch.device,
) -> torch.Tensor | None:
    """Compute pos_weight from training labels."""
    if strategy == "none":
        return None

    all_labels = []
    for rec in train_records:
        lab = rec.get("labels_14")
        if lab is not None:
            all_labels.append(lab)
    if not all_labels:
        return None

    labels_t = torch.tensor(all_labels, dtype=torch.long)
    if strategy == "log":
        pw = compute_pos_weight_log(labels_t)
    elif strategy == "inverse":
        pw = compute_pos_weight_inverse(labels_t)
    else:
        return None

    logger.info("pos_weight (%s): %s", strategy, pw.tolist())
    return pw.to(device)


@torch.no_grad()
def evaluate(
    model: CMLP,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate on validation set. Returns dict with loss + all metrics."""
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        feats = batch["feats"].to(device)
        boxes = batch["boxes"].to(device)
        labels = batch["labels"].to(device)

        out = model(feats, boxes)
        loss = loss_fn(out.image_logits, labels)

        total_loss += loss.item()
        n_batches += 1
        all_logits.append(out.image_logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_all_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def train(args=None):
    """Main training entry point."""
    if args is None:
        args = parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log"),
        ],
    )

    # --- Load data ---
    logger.info("Loading metadata from %s", args.metadata)
    train_records = load_metadata_jsonl(
        args.metadata, split="train", splits_json=args.splits
    )
    val_records = load_metadata_jsonl(
        args.metadata, split="val", splits_json=args.splits
    )
    logger.info("Train: %d, Val: %d", len(train_records), len(val_records))

    train_ds = CachedFeatureDataset(train_records, args.feature_cache, args.bbox_dir)
    val_ds = CachedFeatureDataset(val_records, args.feature_cache, args.bbox_dir)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # --- Model ---
    model = CMLP(
        pool_type=args.pool,
        head_type=args.head,
        use_global=(args.global_branch == "on"),
        proj_dim=args.proj_dim,
        head_hidden=args.head_hidden,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %s", f"{n_params:,}")

    # --- Loss ---
    pos_weight = _compute_pos_weight(train_records, args.pos_weight, device)
    loss_fn = build_loss(
        loss_type=args.loss,
        pos_weight=pos_weight,
        focal_gamma=args.focal_gamma,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # --- Resume ---
    start_epoch = 0
    best_f1 = 0.0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        logger.info("Resumed from epoch %d, best_f1=%.4f", start_epoch, best_f1)

    # --- Training loop ---
    patience_counter = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            feats = batch["feats"].to(device)
            boxes = batch["boxes"].to(device)
            labels = batch["labels"].to(device)

            out = model(feats, boxes)
            loss = loss_fn(out.image_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # --- Validate ---
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        val_f1 = val_metrics["macro_f1"]

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_macro_f1=%.4f | "
            "val_macro_auroc=%.4f | %.1fs",
            epoch + 1, args.epochs, train_loss, val_metrics["loss"],
            val_f1, val_metrics.get("macro_auroc", 0.0), elapsed,
        )

        # --- Checkpoint ---
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_f1": max(best_f1, val_f1),
            "val_metrics": val_metrics,
            "args": vars(args),
        }
        torch.save(ckpt, output_dir / "last.pt")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(ckpt, output_dir / "cmlp_best.pt")
            logger.info("  ★ New best macro-F1: %.4f", best_f1)
            patience_counter = 0
        else:
            patience_counter += 1

        # Log per-class F1 every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            for name in [f"{n}_f1" for n in
                         ["No Finding", "Cardiomegaly", "Pneumonia",
                          "Consolidation", "Lung Lesion", "Fracture"]]:
                if name in val_metrics:
                    logger.info("    %s = %.4f", name, val_metrics[name])

        # Early stopping
        if patience_counter >= args.patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, args.patience)
            break

    logger.info("Training complete. Best macro-F1: %.4f", best_f1)

    # Save final metrics
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)


if __name__ == "__main__":
    train()
