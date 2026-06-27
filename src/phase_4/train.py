"""Training loop for Phase 4 T-MLP."""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import parse_args
from .dataset import TemporalPairDataset, build_temporal_pairs
from .losses import build_loss
from .metrics import confusion_matrix_3class, progression_metrics
from .model import TMLP

from phase_3.constants import MASK_VALUE

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    all_logits, all_targets, all_masks = [], [], []
    total_loss, n = 0.0, 0

    for batch in loader:
        pf = batch["prior_feat"].to(device)
        cf = batch["curr_feat"].to(device)
        pl = batch["prior_labels"].to(device)
        cl = batch["curr_labels"].to(device)
        pm = batch["prior_mask"].to(device)
        cm_ = batch["curr_mask"].to(device)
        prog = batch["prog_labels"].to(device)

        out = model(pf, cf, pl, cl, pm, cm_)
        loss = loss_fn(out.prog_logits, prog, out.pair_mask)

        total_loss += loss.item()
        n += 1
        all_logits.append(out.prog_logits.cpu())
        all_targets.append(prog.cpu())
        all_masks.append(out.pair_mask.cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    masks = torch.cat(all_masks)

    metrics = progression_metrics(logits, targets, masks)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def train(args=None):
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

    # --- Build pairs ---
    logger.info("Building temporal pairs...")
    train_pairs = build_temporal_pairs(args.metadata, "train", args.splits)
    val_pairs = build_temporal_pairs(args.metadata, "val", args.splits)
    logger.info("Train pairs: %d, Val pairs: %d", len(train_pairs), len(val_pairs))

    # We need a directory of region_feat files from C-MLP inference.
    # For now, assume it's at output_dir/../phase_3/region_feats/
    feat_dir = args.output_dir.parent / "phase_3" / "region_feats"
    if not feat_dir.exists():
        logger.warning("region_feat dir not found: %s — using zeros", feat_dir)
        feat_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TemporalPairDataset(train_pairs, feat_dir, args.bbox_dir)
    val_ds = TemporalPairDataset(val_pairs, feat_dir, args.bbox_dir)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # --- Model ---
    model = TMLP(
        use_hadamard=args.use_hadamard,
        use_labels=not args.no_labels,
        hidden_dim=args.hidden_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("T-MLP params: %s (in_dim=%d)", f"{n_params:,}", model.in_dim)

    # --- Loss ---
    loss_fn = build_loss(weight_strategy=args.class_weight)

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
        logger.info("Resumed from epoch %d", start_epoch)

    # --- Train ---
    patience_counter = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        t0 = time.time()

        for batch in train_loader:
            pf = batch["prior_feat"].to(device)
            cf = batch["curr_feat"].to(device)
            pl = batch["prior_labels"].to(device)
            cl = batch["curr_labels"].to(device)
            pm = batch["prior_mask"].to(device)
            cm_ = batch["curr_mask"].to(device)
            prog = batch["prog_labels"].to(device)

            out = model(pf, cf, pl, cl, pm, cm_)
            loss = loss_fn(out.prog_logits, prog, out.pair_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        val_metrics = evaluate(model, val_loader, loss_fn, device)
        val_f1 = val_metrics["macro_f1"]

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_macro_f1=%.4f | "
            "improved=%.3f stable=%.3f worsened=%.3f | %.1fs",
            epoch + 1, args.epochs, train_loss, val_metrics["loss"],
            val_f1,
            val_metrics.get("improved_f1", 0),
            val_metrics.get("stable_f1", 0),
            val_metrics.get("worsened_f1", 0),
            elapsed,
        )

        if val_metrics.get("is_trivial", False):
            logger.warning("  ⚠ Model appears trivial (accuracy ≈ stable proportion)")

        # Checkpoint
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
            torch.save(ckpt, output_dir / "tmlp_best.pt")
            logger.info("  ★ New best macro-F1: %.4f", best_f1)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    logger.info("Training complete. Best macro-F1: %.4f", best_f1)
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)


if __name__ == "__main__":
    train()
