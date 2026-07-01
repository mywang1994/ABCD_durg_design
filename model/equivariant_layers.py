from __future__ import annotations

import networkx as nx
import torch
import torch.nn as nn
from e3nn import o3
from egnn_pytorch import EGNN

from config import HIDDEN, N_LAYERS

try:
    from torch_geometric.nn import radius_graph as _pyg_radius_graph

    _HAS_PYG = True
except ImportError:
    _pyg_radius_graph = None
    _HAS_PYG = False


def has_pyg() -> bool:
    return _HAS_PYG


def radius_edges(
    pos: torch.Tensor,
    radius: float,
    *,
    max_num_neighbors: int = 32,
    loop: bool = False,
    batch: torch.Tensor | None = None,
) -> torch.Tensor:
    if pos.ndim != 2 or pos.shape[-1] != 3:
        raise ValueError(f"pos expected (N,3), got {tuple(pos.shape)}")
    if _HAS_PYG:
        return _pyg_radius_graph(
            pos,
            float(radius),
            batch=batch,
            max_num_neighbors=int(max_num_neighbors),
            loop=bool(loop),
        )

    n = int(pos.shape[0])
    device = pos.device
    if n <= 1:
        idx = torch.arange(n, device=device, dtype=torch.long)
        return torch.stack([idx, idx], dim=0)
    dist = torch.cdist(pos, pos)
    mask = (dist <= float(radius)) & (dist > 0.0)
    src, dst = mask.nonzero(as_tuple=True)
    if src.numel() == 0:
        idx = torch.arange(n, device=device, dtype=torch.long)
        return torch.stack([idx, idx], dim=0)
    return torch.stack([src, dst], dim=0)


def pocket_edges(
    pos: torch.Tensor,
    mask: torch.Tensor | None,
    r_max: float,
    *,
    max_num_neighbors: int = 64,
    add_self_loops: bool = True,
) -> torch.Tensor:
    n = int(pos.shape[0])
    device = pos.device
    edge_index = radius_edges(pos, r_max, max_num_neighbors=max_num_neighbors, loop=False)
    if mask is not None:
        m = mask.to(device=device)
        if m.dtype != torch.bool:
            m = m > 0.5
        m = m.view(-1)
        keep = m[edge_index[0]] & m[edge_index[1]]
        edge_index = edge_index[:, keep]

    if edge_index.shape[1] == 0 and add_self_loops:
        idx = torch.arange(n, device=device, dtype=torch.long)
        edge_index = torch.stack([idx, idx], dim=0)

    if add_self_loops:
        sl = torch.stack([torch.arange(n, device=device), torch.arange(n, device=device)], dim=0)
        edge_index = torch.cat([edge_index, sl], dim=1)
        edge_index = torch.unique(edge_index, dim=1)

    if edge_index.shape[1] == 0:
        idx = torch.arange(n, device=device, dtype=torch.long)
        edge_index = torch.stack([idx, idx], dim=0)
    return edge_index


def eidx_adj(n: int, edge_index: torch.Tensor, device: torch.device, *, symmetric: bool = True) -> torch.Tensor:
    adj = torch.zeros(n, n, dtype=torch.bool, device=device)
    row, col = edge_index
    adj[row, col] = True
    if symmetric:
        adj[col, row] = True
    return adj


def eidx_dense(n: int, edge_index: torch.Tensor, edge_attr: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    e = int(edge_attr.shape[-1])
    dense = torch.zeros(1, n, n, e, device=device, dtype=dtype)
    row, col = edge_index
    dense[0, row, col] = edge_attr.to(device=device, dtype=dtype)
    if row.numel() > 0:
        dense[0, col, row] = edge_attr.to(device=device, dtype=dtype)
    return dense


class CondMix(nn.Module):
    def __init__(self, hidden: int, cond_dim: int):
        super().__init__()
        in_ir = o3.Irreps(f"{hidden}x0e + {cond_dim}x0e + {cond_dim}x0e")
        out_ir = o3.Irreps(f"{hidden}x0e")
        self.lin = o3.Linear(in_ir, out_ir)
        self.act = nn.SiLU()

    def forward(self, h: torch.Tensor, hP: torch.Tensor, hc: torch.Tensor) -> torch.Tensor:
        n = int(h.shape[0])
        pocket_tile = hP.view(1, -1).expand(n, -1)
        task_tile = hc.view(1, -1).expand(n, -1)
        mixed = self.lin(torch.cat([h, pocket_tile, task_tile], dim=-1))
        return self.act(mixed + h)


class CoordDelta(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor, R: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        edge_w = self.edge_mlp(torch.cat([h[row], h[col]], dim=-1))
        unit = R[col] - R[row]
        unit = unit / unit.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        msg = edge_w * unit
        dR = torch.zeros_like(R)
        dR.index_add_(0, row, msg)
        return dR


class EGNNStack(nn.Module):
    def __init__(
        self,
        hidden: int,
        num_layers: int,
        edge_dim: int,
        *,
        update_coors: bool = True,
        num_nearest_neighbors: int = 0,
        only_sparse_neighbors: bool = True,
        norm_coors: bool = True,
        coor_weights_clamp_value: float | None = 2.0,
        dropout: float = 0.0,
        valid_radius: float = float("inf"),
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EGNN(
                    dim=int(hidden),
                    edge_dim=int(edge_dim),
                    num_nearest_neighbors=int(num_nearest_neighbors),
                    only_sparse_neighbors=bool(only_sparse_neighbors),
                    update_coors=bool(update_coors),
                    update_feats=True,
                    norm_coors=bool(norm_coors),
                    coor_weights_clamp_value=coor_weights_clamp_value,
                    dropout=float(dropout),
                    valid_radius=float(valid_radius),
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, h, x, *, adj_mat=None, edges=None, mask=None):
        h_b = h.unsqueeze(0)
        x_b = x.unsqueeze(0)
        batch_mask = None
        if mask is not None:
            if mask.ndim == 2 and mask.shape[-1] == 1:
                batch_mask = (mask.squeeze(-1) > 0.5).unsqueeze(0)
            else:
                batch_mask = (mask > 0.5).unsqueeze(0)

        for layer in self.layers:
            kwargs: dict = {}
            if adj_mat is not None:
                kwargs["adj_mat"] = adj_mat
            if edges is not None:
                kwargs["edges"] = edges
            if batch_mask is not None:
                kwargs["mask"] = batch_mask
            h_b, x_b = layer(h_b, x_b, **kwargs)
        return h_b.squeeze(0), x_b.squeeze(0)


def _eidx_from_graph(mol_graph: nx.Graph, n: int, device: torch.device) -> torch.Tensor:
    src: list[int] = []
    dst: list[int] = []
    for u, v in mol_graph.edges():
        ui, vi = int(u), int(v)
        if 0 <= ui < n and 0 <= vi < n:
            src.extend([ui, vi])
            dst.extend([vi, ui])
    if not src:
        idx = torch.arange(n, device=device, dtype=torch.long)
        return torch.stack([idx, idx], dim=0)
    return torch.stack(
        [torch.tensor(src, device=device, dtype=torch.long), torch.tensor(dst, device=device, dtype=torch.long)],
        dim=0,
    )


def _bond_attr(edge_index: torch.Tensor, mol_graph: nx.Graph | None, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    e = int(edge_index.shape[1])
    bond = torch.zeros((e, 1), device=device, dtype=dtype)
    if mol_graph is None:
        return bond
    bonds = {tuple(sorted((int(u), int(v)))) for u, v in mol_graph.edges()}
    row, col = edge_index
    for k in range(e):
        i, j = int(row[k].item()), int(col[k].item())
        key = (i, j) if i <= j else (j, i)
        if key in bonds:
            bond[k, 0] = 1.0
    return bond


class MPNNCore(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        hidden: int = HIDDEN,
        num_layers: int = N_LAYERS,
        num_rbf: int = 16,
        rbf_max_dist: float = 6.0,
        radius_cutoff: float = 6.0,
        use_radius_edges: bool = True,
        chirality_aware: bool = False,
        chirality_strength: float = 0.25,
        mpnn_sub_layers: int = 2,
        attention: bool = True,
        coords_range: float = 15.0,
        norm_constant: float = 1.0,
        normalization_factor: float = 100.0,
        reflection_equiv: bool = True,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.radius_cutoff = float(radius_cutoff)
        self.use_radius_edges = bool(use_radius_edges)
        self.chirality_aware = bool(chirality_aware)
        self.chirality_strength = float(chirality_strength)

        cond_edge_dim = 4
        self.in_edge_nf = 1 + cond_edge_dim
        self.cond_mixer = CondMix(self.hidden, int(cond_dim))
        self.cond_edge_proj = nn.Sequential(nn.Linear(2 * int(cond_dim), cond_edge_dim), nn.SiLU())
        self.stack = EGNNStack(
            hidden=self.hidden,
            num_layers=self.num_layers,
            edge_dim=self.in_edge_nf,
            update_coors=True,
            only_sparse_neighbors=True,
            norm_coors=True,
            coor_weights_clamp_value=2.0,
            num_nearest_neighbors=0,
            valid_radius=self.radius_cutoff,
        )
        self.coord_refine = CoordDelta(self.hidden) if self.chirality_aware else None
        self.struct_head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, 1),
        )

    def _build_edges(self, R, mol_graph, n, device, dtype):
        if self.use_radius_edges:
            edge_index = radius_edges(R, self.radius_cutoff, max_num_neighbors=32, loop=False)
        else:
            edge_index = (
                _eidx_from_graph(mol_graph, n, device)
                if mol_graph is not None
                else radius_edges(R, self.radius_cutoff, max_num_neighbors=32, loop=False)
            )
        bond = _bond_attr(edge_index, mol_graph, n, device, dtype)
        return edge_index, bond

    def forward_from_node_emb(self, node_emb, R, hP, hc, device, mol_graph=None):
        n = int(node_emb.shape[0])
        dtype = node_emb.dtype
        h = node_emb.to(device=device, dtype=dtype)
        coords0 = R.to(device=device, dtype=dtype)
        x = coords0.clone()

        if n == 0:
            z = torch.zeros(0, 3, device=device, dtype=dtype)
            return {"dR": z, "edit_intensity": torch.tensor(0.0, device=device, dtype=dtype), "node_emb": h}

        pocket_flat = hP.to(device=device, dtype=dtype).view(-1)
        task_flat = hc.to(device=device, dtype=dtype).view(-1)
        h = self.cond_mixer(h, pocket_flat, task_flat)

        edge_index, bond = self._build_edges(coords0, mol_graph, n, device, dtype)
        row, col = edge_index
        pocket_at_edge = pocket_flat.view(1, -1).expand(n, -1)
        task_at_edge = task_flat.view(1, -1).expand(n, -1)
        cond_e = self.cond_edge_proj(torch.cat([pocket_at_edge[col], task_at_edge[col]], dim=-1))
        edge_attr = torch.cat([bond, cond_e], dim=-1)
        node_mask = torch.ones((n, 1), device=device, dtype=dtype)

        adj = eidx_adj(n, edge_index, device)
        edges = eidx_dense(n, edge_index, edge_attr, device, dtype)
        h, x = self.stack(h, x, adj_mat=adj, edges=edges, mask=node_mask)

        dR = x - coords0
        if self.coord_refine is not None:
            dR = dR + self.chirality_strength * self.coord_refine(h, coords0, edge_index)

        edit_intensity = self.struct_head(h.mean(dim=0)).abs().squeeze()
        return {"dR": dR, "edit_intensity": edit_intensity, "node_emb": h}


def build_core(cond_dim: int, core_kwargs: dict | None = None):
    ck = dict(core_kwargs or {})
    ck.pop("backbone", None)
    return MPNNCore(cond_dim, **ck)
