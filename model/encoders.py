from __future__ import annotations

import torch
import torch.nn as nn

from config import POCKET_HIDDEN, POCKET_LAYERS
from .equivariant_layers import EGNNStack, eidx_adj, pocket_edges
from .pocket_context import PocketContext


class PocketEncoder(nn.Module):
    """Typed pocket graph -> invariant summary and optional atom-level context."""

    def __init__(
        self,
        out_dim: int,
        feat_dim: int = 0,
        *,
        hidden_nf: int = POCKET_HIDDEN,
        n_layers: int = POCKET_LAYERS,
        radius: float = 5.0,
        max_num_neighbors: int = 32,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.feat_dim = feat_dim
        self.radius = float(radius)
        self.max_num_neighbors = int(max_num_neighbors)

        in_node_nf = max(1, feat_dim)
        self.input_proj = nn.Linear(in_node_nf, hidden_nf)
        self.stack = EGNNStack(
            hidden=hidden_nf,
            num_layers=n_layers,
            edge_dim=0,
            update_coors=False,
            only_sparse_neighbors=True,
            norm_coors=True,
            coor_weights_clamp_value=2.0,
            num_nearest_neighbors=0,
            valid_radius=self.radius,
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, out_dim),
        )
        self.node_readout = nn.Linear(hidden_nf, out_dim)

    def _encode_one(self, pos, mask, feat, return_context):
        if pos.ndim != 2 or pos.shape[-1] != 3 or pos.shape[0] == 0:
            raise ValueError("Pocket coordinates must be nonempty (N,3)")
        if not pos.is_floating_point():
            raise ValueError("Pocket coordinates must have floating-point dtype")
        n = pos.shape[0]
        if feat is not None and feat.shape != (n, self.feat_dim):
            raise ValueError(f"feat expected ({n},{self.feat_dim}), got {tuple(feat.shape)}")
        if mask is not None:
            if mask.shape != (n,) or not bool(torch.isfinite(mask).all()):
                raise ValueError("Pocket mask must be finite and have shape (N,)")
            valid = mask.to(device=pos.device)
            valid = valid if valid.dtype == torch.bool else valid > .5
            if not bool(valid.any()):
                raise ValueError("Pocket mask must select at least one atom")
            # Exclude padding before graph construction, including NaN padding.
            pos = pos[valid]
            feat = feat.to(device=pos.device)[valid] if feat is not None else None
        if not bool(torch.isfinite(pos).all()):
            raise ValueError("Pocket coordinates must be finite")
        n = pos.shape[0]
        device = pos.device
        dtype = pos.dtype
        if feat is None:
            if self.feat_dim != 0:
                raise ValueError("feat is required when feat_dim > 0")
            h = torch.ones(n, 1, device=device, dtype=dtype)
        else:
            if not bool(torch.isfinite(feat).all()):
                raise ValueError("Pocket features must be finite")
            h = feat.to(device=device, dtype=dtype)

        edge_index = pocket_edges(pos, None, self.radius, max_num_neighbors=self.max_num_neighbors)
        h = self.input_proj(h)
        adj = eidx_adj(n, edge_index, device)
        h, _ = self.stack(h, pos, adj_mat=adj)
        summary = self.readout(h.mean(dim=0))
        if return_context:
            return PocketContext(positions=pos, node_features=self.node_readout(h), summary=summary)
        return summary

    def forward(self, coords, mask=None, feat=None, return_context=False):
        """Return a summary tensor, or one context (a list for batched input)."""
        if coords.ndim == 2:
            return self._encode_one(coords, mask, feat, return_context)
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(f"coords expected (B,N,3) or (N,3), got {tuple(coords.shape)}")
        batch = coords.shape[0]
        batch_out = [
            self._encode_one(
                coords[i],
                mask[i] if mask is not None else None,
                feat[i] if feat is not None else None,
                return_context,
            )
            for i in range(batch)
        ]
        return batch_out if return_context else torch.stack(batch_out, dim=0)


class TaskEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, task_feat: torch.Tensor) -> torch.Tensor:
        return self.net(task_feat)
