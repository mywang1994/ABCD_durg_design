from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .pocket_context import PocketContext


@dataclass(frozen=True)
class AnchorSelection:
    positions: torch.Tensor
    features: torch.Tensor
    scores: torch.Tensor
    indices: torch.Tensor
    candidate_scores: torch.Tensor
    mode: str = "learned_topk"


class LearnedAnchorScorer(nn.Module):
    """Score invariant pocket features with w^T sigma(W_u u_j + W_c h_c).
    Keep all boundary ties, so selection may exceed num_anchors."""

    def __init__(self, cond_dim: int, hidden: int, num_anchors: int = 4):
        super().__init__()
        if int(num_anchors) != num_anchors or num_anchors < 1:
            raise ValueError("num_anchors must be a positive integer")
        self.num_anchors = int(num_anchors)
        self.environment_projection = nn.Linear(cond_dim, hidden, bias=False)
        self.task_projection = nn.Linear(cond_dim, hidden, bias=False)
        self.score_projection = nn.Linear(hidden, 1, bias=False)

    def forward(self, pocket: PocketContext, hc: torch.Tensor) -> AnchorSelection:
        if not isinstance(pocket, PocketContext):
            raise TypeError("Learned anchors require a typed PocketContext")
        n = pocket.positions.shape[0]
        if n < 1 or pocket.positions.shape != (n, 3):
            raise ValueError("Learned anchors require nonempty (N,3) pocket positions")
        if pocket.node_features.shape != (n, self.environment_projection.in_features):
            raise ValueError("Pocket features do not match the anchor scorer")
        hidden = self.environment_projection(pocket.node_features)
        hidden = hidden + self.task_projection(hc.reshape(-1))
        scores = self.score_projection(F.silu(hidden)).squeeze(-1)
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("Anchor scores must be finite")
        with torch.no_grad():
            boundary = scores.topk(min(n, self.num_anchors)).values[-1]
            selected = (scores >= boundary).nonzero(as_tuple=True)[0]
            selected = selected[torch.argsort(scores[selected], descending=True, stable=True)]
        hard = F.one_hot(selected, num_classes=n).to(dtype=scores.dtype)
        soft = scores.softmax(dim=0).expand(len(selected), -1)
        # An exact-zero residual preserves the selected coordinates in the forward pass.
        selection = hard + (soft - soft.detach())
        coordinate_dtype = torch.float64 if pocket.positions.dtype == torch.float64 else torch.float32
        positions = selection.to(dtype=coordinate_dtype) @ pocket.positions.to(dtype=coordinate_dtype)
        features = selection @ pocket.node_features
        selected_scores = selection @ scores
        return AnchorSelection(positions, features, selected_scores, selected, scores)
