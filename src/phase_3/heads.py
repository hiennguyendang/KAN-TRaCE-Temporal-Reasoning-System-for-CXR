"""Classification heads: MLPHead, KANHead (ablation), GlobalHead.

Design rationale (from phase_3_4_mlp_update.md):
- MLPHead is default: Linear→LN→GELU→Linear, shared across 29 regions.
- KANHead kept for ablation (FastKAN stub — swap in real impl later).
- GlobalHead: for the global branch (image-level features → 14 diseases).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .constants import NUM_DISEASES, PROJ_DIM


# ---------------------------------------------------------------------------
# MLPHead — default disease head (shared across 29 regions)
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """MLP disease head: in_dim → hidden → num_classes.

    Applied identically to each region (shared weights = applied on region axis).
    """

    def __init__(
        self,
        in_dim: int = PROJ_DIM,
        hidden_dim: int = 64,
        num_classes: int = NUM_DISEASES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim] → [..., num_classes]."""
        return self.net(x)


# ---------------------------------------------------------------------------
# KANHead — ablation stub
# ---------------------------------------------------------------------------

class KANHead(nn.Module):
    """Kolmogorov-Arnold Network head (placeholder for ablation).

    Uses simple B-spline–inspired architecture.  Replace with FastKAN / pykan
    when running the KAN ablation experiment.
    """

    def __init__(
        self,
        in_dim: int = PROJ_DIM,
        hidden_dim: int = 64,
        num_classes: int = NUM_DISEASES,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Placeholder: same topology as MLP but with SiLU (approximating
        # learnable activation).  Swap for real KAN layers when needed.
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# GlobalHead — global branch (image-level GAP → disease logits)
# ---------------------------------------------------------------------------

class GlobalHead(nn.Module):
    """Global branch head: encoder_dim → hidden → num_classes."""

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 128,
        num_classes: int = NUM_DISEASES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_dim] → [B, num_classes]."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Gate — learned per-disease gate for combining regional + global logits
# ---------------------------------------------------------------------------

class DiseaseGate(nn.Module):
    """Learned sigmoid gate per disease for regional+global fusion."""

    def __init__(self, num_classes: int = NUM_DISEASES):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(num_classes))  # init 0 → sigmoid=0.5

    def forward(self, global_logits: torch.Tensor) -> torch.Tensor:
        """Returns gated global logits: sigmoid(gate) * global_logits."""
        return torch.sigmoid(self.gate) * global_logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_head(
    head_type: str = "mlp",
    in_dim: int = PROJ_DIM,
    hidden_dim: int = 64,
    num_classes: int = NUM_DISEASES,
    dropout: float = 0.1,
) -> nn.Module:
    """Build a disease classification head.

    Parameters
    ----------
    head_type : 'mlp' (default) or 'kan' (ablation).
    """
    cls = {"mlp": MLPHead, "kan": KANHead}
    if head_type not in cls:
        raise ValueError(f"Unknown head type: {head_type!r}")
    return cls[head_type](
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    )
