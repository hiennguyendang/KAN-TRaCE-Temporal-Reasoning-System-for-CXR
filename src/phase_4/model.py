"""T-MLP: Temporal progression classification model.

Architecture (from phase_3_4_mlp_update.md §5):

    Siamese C-MLP (shared weights):
        prior_feat [29,128] ; curr_feat [29,128]
        merged = curr - prior [29,128]

    Per-region concat:
        [prior ; curr ; merged ; prior_labels ; curr_labels]
        = 128×3 + 14×2 = 412 dim

    Head = MLP [412 → 128 → 42]  →  prog_logits [B,29,14,3]
        masked CE (ignore_index=-100, skip where pair_mask=0)

    3 classes: 0=improved, 1=stable, 2=worsened
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from phase_3.constants import NUM_DISEASES, NUM_PROG, NUM_REGIONS, PROJ_DIM


@dataclass
class TMLPOutput:
    """Structured output of the T-MLP model."""
    prog_logits: torch.Tensor     # [B, 29, 14, 3]
    pair_mask: torch.Tensor       # [B, 29, 14]  bool — 1 = valid pair


class TemporalHead(nn.Module):
    """MLP head for temporal progression: in_dim → hidden → 14*3."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_diseases: int = NUM_DISEASES,
        num_prog: int = NUM_PROG,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_diseases = num_diseases
        self.num_prog = num_prog
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_diseases * num_prog),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 29, in_dim] → [B, 29, 14, 3]."""
        out = self.net(x)  # [B, 29, 14*3]
        B, R, _ = out.shape
        return out.view(B, R, self.num_diseases, self.num_prog)


class TMLP(nn.Module):
    """T-MLP: Temporal progression model.

    Takes region_feat from C-MLP for both prior and current studies,
    computes difference, concatenates with disease labels, and predicts
    3-class progression per (region, disease).

    Parameters
    ----------
    proj_dim : int
        Dimension of region_feat from C-MLP neck (128).
    num_diseases : int
        Number of disease classes (14).
    hidden_dim : int
        Hidden dimension in temporal head (128).
    use_hadamard : bool
        If True, add element-wise product to concat (+proj_dim dim).
    use_labels : bool
        If True, concat prior/current disease labels (+2*num_diseases).
    """

    def __init__(
        self,
        proj_dim: int = PROJ_DIM,
        num_diseases: int = NUM_DISEASES,
        num_regions: int = NUM_REGIONS,
        hidden_dim: int = 128,
        use_hadamard: bool = False,
        use_labels: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.num_diseases = num_diseases
        self.use_hadamard = use_hadamard
        self.use_labels = use_labels

        # Compute input dimension
        in_dim = proj_dim * 3  # prior + curr + merged(=diff)
        if use_hadamard:
            in_dim += proj_dim  # element-wise product
        if use_labels:
            in_dim += num_diseases * 2  # prior_labels + curr_labels

        self.in_dim = in_dim
        self.head = TemporalHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_diseases=num_diseases,
            dropout=dropout,
        )

    def forward(
        self,
        prior_feat: torch.Tensor,     # [B, 29, 128]
        curr_feat: torch.Tensor,      # [B, 29, 128]
        prior_labels: torch.Tensor,   # [B, 29, 14] or [B, 14]
        curr_labels: torch.Tensor,    # [B, 29, 14] or [B, 14]
        prior_mask: torch.Tensor,     # [B, 29] bool — prior region valid
        curr_mask: torch.Tensor,      # [B, 29] bool — current region valid
    ) -> TMLPOutput:
        B, R, D = prior_feat.shape

        # Expand image-level labels to per-region if needed
        if prior_labels.dim() == 2:  # [B, 14]
            prior_labels = prior_labels.unsqueeze(1).expand(B, R, -1)
        if curr_labels.dim() == 2:
            curr_labels = curr_labels.unsqueeze(1).expand(B, R, -1)

        # Merged = difference
        merged = curr_feat - prior_feat  # [B, 29, 128]

        # Concat components
        parts = [prior_feat, curr_feat, merged]
        if self.use_hadamard:
            parts.append(prior_feat * curr_feat)
        if self.use_labels:
            # Clamp labels: replace -100 with 0 for feature input
            pl = prior_labels.float().clamp(min=0)
            cl = curr_labels.float().clamp(min=0)
            parts.append(pl)
            parts.append(cl)

        region_in = torch.cat(parts, dim=-1)  # [B, 29, in_dim]

        # Predict progression
        prog_logits = self.head(region_in)  # [B, 29, 14, 3]

        # Pair mask: both prior and current regions must be valid
        pair_mask = prior_mask & curr_mask  # [B, 29]
        # Expand to [B, 29, 14]
        pair_mask_expanded = pair_mask.unsqueeze(-1).expand(B, R, self.num_diseases)

        return TMLPOutput(
            prog_logits=prog_logits,
            pair_mask=pair_mask_expanded,
        )
