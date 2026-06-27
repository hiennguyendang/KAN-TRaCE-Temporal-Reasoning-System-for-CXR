"""C-MLP: Region-aware disease classification model.

Architecture (from phase_3_4_mlp_update.md §4):

    feats [B,196,512]  +  boxes [B,29,4]
      │
      ├─ ĐƯỜNG VÙNG (regional path)
      │   attention-pool → region_pooled [B,29,512]
      │   Neck(512→128)  → region_feat [B,29,128]    ← shared with T-MLP
      │   DiseaseHead     → region_logits [B,29,14]
      │   LSE aggregate   → region_image_logits [B,14]
      │
      └─ ĐƯỜNG GLOBAL (global path)
          GAP all 196 cells → global_feat [B,512]
          GlobalHead        → global_logits [B,14]

    COMBINE: image_logits = region_image_logits + gate ⊙ global_logits
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .constants import ENCODER_DIM, NUM_DISEASES, NUM_REGIONS, PROJ_DIM
from .heads import DiseaseGate, GlobalHead, build_head
from .neck import Neck
from .roi_pool import build_pool


@dataclass
class CMLPOutput:
    """Structured output of the C-MLP model."""

    image_logits: torch.Tensor       # [B, 14]
    region_logits: torch.Tensor      # [B, 29, 14]
    region_feat: torch.Tensor        # [B, 29, 128]  — feed T-MLP
    region_mask: torch.Tensor        # [B, 29]  bool
    alpha: torch.Tensor              # [B, 29, 196]  — faithful attribution
    global_logits: torch.Tensor | None  # [B, 14]  (None if global branch off)
    region_image_logits: torch.Tensor   # [B, 14]  — before global fusion


class CMLP(nn.Module):
    """C-MLP: region-aware multi-label disease classifier.

    Parameters
    ----------
    pool_type : 'attn' or 'mean'
    head_type : 'mlp' (default) or 'kan' (ablation)
    use_global : bool
        Whether to enable the global branch.
    encoder_dim : int
        Patch feature dimension (512 for BioViL-T).
    proj_dim : int
        Neck projection dimension (128).
    head_hidden : int
        Hidden dim in disease head (64).
    num_diseases : int
        Number of disease classes (14).
    """

    def __init__(
        self,
        pool_type: str = "attn",
        head_type: str = "mlp",
        use_global: bool = True,
        encoder_dim: int = ENCODER_DIM,
        proj_dim: int = PROJ_DIM,
        head_hidden: int = 64,
        num_diseases: int = NUM_DISEASES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_global = use_global

        # --- Regional path ---
        self.pool = build_pool(pool_type, d_model=encoder_dim)
        self.neck = Neck(encoder_dim=encoder_dim, proj_dim=proj_dim)
        self.disease_head = build_head(
            head_type=head_type,
            in_dim=proj_dim,
            hidden_dim=head_hidden,
            num_classes=num_diseases,
            dropout=dropout,
        )

        # --- Global path ---
        if use_global:
            self.global_head = GlobalHead(
                in_dim=encoder_dim,
                hidden_dim=proj_dim,
                num_classes=num_diseases,
                dropout=dropout,
            )
            self.gate = DiseaseGate(num_classes=num_diseases)

    def _lse_aggregate(
        self,
        region_logits: torch.Tensor,  # [B, 29, 14]
        region_mask: torch.Tensor,    # [B, 29] bool
    ) -> torch.Tensor:
        """Log-Sum-Exp aggregation over regions (masked).

        LSE is a smooth max: image_logit[d] = log Σ_{r:mask} exp(region_logit[r,d]).
        Gradient = softmax(region_logit[r,d]) → exact region attribution.
        """
        # Mask invalid regions by setting logits to -inf
        masked_logits = region_logits.clone()
        mask_expanded = region_mask.unsqueeze(-1).expand_as(region_logits)  # [B,29,14]
        masked_logits[~mask_expanded] = float("-inf")

        # LSE over region dim (dim=1)
        # torch.logsumexp handles -inf correctly (returns -inf if all -inf)
        image_logits = torch.logsumexp(masked_logits, dim=1)  # [B, 14]

        # If ALL regions masked for a sample, logsumexp returns -inf.
        # Replace with 0 (no evidence) to avoid NaN in loss.
        image_logits = image_logits.clamp(min=-100.0)

        return image_logits

    def forward(
        self,
        feats: torch.Tensor,   # [B, 196, 512]
        boxes: torch.Tensor,   # [B, 29, 4]  normalised x1y1x2y2
    ) -> CMLPOutput:
        # --- Regional path ---
        region_pooled, region_mask, alpha = self.pool(feats, boxes)
        # region_pooled: [B, 29, 512]

        region_feat = self.neck(region_pooled)        # [B, 29, 128]
        region_logits = self.disease_head(region_feat)  # [B, 29, 14]

        # Zero out logits for invalid regions
        region_logits = region_logits * region_mask.unsqueeze(-1).float()

        # LSE aggregate → image-level logits
        region_image_logits = self._lse_aggregate(region_logits, region_mask)  # [B, 14]

        # --- Global path ---
        global_logits = None
        if self.use_global:
            global_feat = feats.mean(dim=1)  # GAP: [B, 512]
            global_logits = self.global_head(global_feat)  # [B, 14]
            gated_global = self.gate(global_logits)
            image_logits = region_image_logits + gated_global
        else:
            image_logits = region_image_logits

        return CMLPOutput(
            image_logits=image_logits,
            region_logits=region_logits,
            region_feat=region_feat,
            region_mask=region_mask,
            alpha=alpha,
            global_logits=global_logits,
            region_image_logits=region_image_logits,
        )
