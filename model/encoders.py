from __future__ import annotations

import torch
import torch.nn as nn

from config import POCKET_HIDDEN, POCKET_LAYERS
from .equivariant_layers import EGNNStack, eidx_adj, pocket_edges


class PocketEncoder(nn.Module):
    """Pocket 3D points -> h_P via radius graph + EGNN (coords frozen at readout)."""

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

    def _encode_one(self, pos, mask, feat):
        n = pos.shape[0]
        device = pos.device
        dtype = pos.dtype
        if feat is None:
            if self.feat_dim != 0:
                raise ValueError("feat is required when feat_dim > 0")
            h = torch.ones(n, 1, device=device, dtype=dtype)
        else:
            if feat.shape != (n, self.feat_dim):
                raise ValueError(f"feat expected ({n},{self.feat_dim}), got {tuple(feat.shape)}")
            h = feat.to(dtype=dtype)

        node_mask = None
        if mask is not None:
            m = mask.to(device=device)
            if m.dtype != torch.bool:
                m = m > 0.5
            node_mask = m.float().view(n, 1)

        edge_index = pocket_edges(pos, mask, self.radius, max_num_neighbors=self.max_num_neighbors)
        h = self.input_proj(h)
        adj = eidx_adj(n, edge_index, device)
        h, _ = self.stack(h, pos, adj_mat=adj, mask=node_mask)

        if node_mask is not None:
            denom = node_mask.sum().clamp(min=1.0)
            graph_h = (h * node_mask).sum(dim=0) / denom
        else:
            graph_h = h.mean(dim=0)
        return self.readout(graph_h)

    def forward(self, coords, *, mask=None, feat=None):
        if coords.ndim == 2:
            return self._encode_one(coords, mask, feat)
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(f"coords expected (B,N,3) or (N,3), got {tuple(coords.shape)}")
        batch = coords.shape[0]
        batch_out = [
            self._encode_one(
                coords[i],
                mask[i] if mask is not None else None,
                feat[i] if feat is not None else None,
            )
            for i in range(batch)
        ]
        return torch.stack(batch_out, dim=0)


class TaskEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, task_feat: torch.Tensor) -> torch.Tensor:
        return self.net(task_feat)
