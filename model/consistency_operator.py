from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from config import ABCMConfig, HIDDEN, N_LAYERS, SUB_LAYERS
from mol_state import MolState
from .edge_diffusion import EdgeNet, adj_ste, predict_adj, sample_adj
from .equivariant_layers import build_core


def _to_torch(x, device):
    return torch.from_numpy(x).to(device=device, dtype=torch.float32)


def _adj_from_graph(mol_graph: nx.Graph, n: int, device, dtype) -> torch.Tensor:
    adj = torch.zeros((n, n), device=device, dtype=dtype)
    for u, v in mol_graph.edges():
        ui, vi = int(u), int(v)
        if 0 <= ui < n and 0 <= vi < n:
            adj[ui, vi] = 1.0
            adj[vi, ui] = 1.0
    return adj


def _graph_from_adj(adj: torch.Tensor, old_graph: nx.Graph, threshold: float = 0.5) -> nx.Graph:
    n = int(adj.shape[0])
    G = nx.Graph()
    for i in range(n):
        attrs = dict(old_graph.nodes[i]) if old_graph.has_node(i) else {}
        G.add_node(i, **attrs)
    for i in range(n):
        for j in range(i + 1, n):
            if float(adj[i, j].item()) >= threshold:
                G.add_edge(i, j)
    return G


def state_from_step(X: MolState, step_out: dict, *, edge_threshold: float = 0.5) -> MolState:
    R_pred = step_out["R_pred"]
    node_feat_pred = step_out["node_feat_pred"]
    adj_soft = step_out.get("adj_soft")
    with torch.no_grad():
        R_np = R_pred.detach().cpu().numpy().astype(np.float32)
        feat_np = node_feat_pred.detach().cpu().numpy().astype(np.float32)
        if adj_soft is not None and adj_soft.numel() > 0:
            G = _graph_from_adj(adj_soft, X.G, threshold=edge_threshold)
        else:
            G = X.G.copy()
    return MolState(G=G, R=R_np, node_feat=feat_np)


def adj_carry(step_out: dict) -> torch.Tensor | None:
    adj = step_out.get("adj_soft")
    if adj is None or adj.numel() == 0:
        return None
    return adj


class GroupReadout(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.member_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.attn = nn.Linear(hidden, 1)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, member_summaries: torch.Tensor) -> torch.Tensor:
        hidden = self.member_mlp(member_summaries)
        weights = torch.softmax(self.attn(hidden).squeeze(-1), dim=0)
        summary = (weights.unsqueeze(-1) * hidden).sum(dim=0)
        return self.out(summary)


class Operator(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        cond_dim: int,
        hidden=HIDDEN,
        num_layers=N_LAYERS,
        num_rbf=16,
        rbf_max_dist=6.0,
        radius_cutoff=6.0,
        use_radius_edges=True,
        chirality_aware=False,
        chirality_strength=0.25,
        step_scale=0.1,
        core=None,
        edge_denoiser=None,
        edge_T=20,
        edge_time_dim=64,
        edge_threshold=0.5,
    ):
        super().__init__()
        self.hidden = hidden
        self.cond_dim = int(cond_dim)
        self.step_scale = step_scale
        self.node_feat_dim = int(node_feat_dim)
        self.edge_T = int(edge_T)
        self.edge_threshold = float(edge_threshold)
        cond2 = 2 * cond_dim
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim + cond2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.group_readout = GroupReadout(hidden)
        self.group_to_hp = nn.Linear(hidden, self.cond_dim)
        self.group_to_hc = nn.Linear(hidden, self.cond_dim)
        self.core = core or build_core(
            cond_dim,
            dict(
                hidden=hidden,
                num_layers=num_layers,
                num_rbf=num_rbf,
                rbf_max_dist=rbf_max_dist,
                radius_cutoff=radius_cutoff,
                use_radius_edges=use_radius_edges,
                chirality_aware=chirality_aware,
                chirality_strength=chirality_strength,
                mpnn_sub_layers=SUB_LAYERS,
            ),
        )
        self.edge_denoiser = edge_denoiser or EdgeNet(
            node_dim=hidden,
            cond_dim=cond_dim,
            time_dim=edge_time_dim,
            hidden=hidden,
        )
        self.node_feat_head = nn.Linear(hidden, self.node_feat_dim)

    @property
    def use_radius_edges(self) -> bool:
        return bool(self.core.use_radius_edges)

    def forward(self, X, hP, hc, device, *, R_in=None, node_feat_in=None):
        R = _to_torch(X.R, device) if R_in is None else R_in.to(device=device, dtype=torch.float32)
        node_feat = _to_torch(X.node_feat, device) if node_feat_in is None else node_feat_in.to(device=device, dtype=torch.float32)
        n = node_feat.shape[0]

        pocket_tile = hP.to(device=device, dtype=torch.float32).view(1, -1).repeat(n, 1)
        task_tile = hc.to(device=device, dtype=torch.float32).view(1, -1).repeat(n, 1)
        node_emb = self.node_proj(torch.cat([node_feat, pocket_tile, task_tile], dim=-1))

        out = self.core.forward_from_node_emb(
            node_emb, R, hP, hc, device, mol_graph=X.G if not self.core.use_radius_edges else None
        )
        return {"dR": out["dR"], "edit_intensity": out["edit_intensity"], "node_emb": out["node_emb"]}

    def _member_summary(self, node_emb: torch.Tensor) -> torch.Tensor:
        return torch.cat([node_emb.mean(dim=0), node_emb.std(dim=0).clamp(min=1e-6)], dim=-1)

    def _primary_from_group(self, group: list[MolState], X_star: MolState | None) -> MolState:
        if not group:
            raise ValueError("empty mapping group")
        if X_star is None:
            return group[0]
        from utils.distances import d_mcm

        return min(group, key=lambda x: d_mcm(x, X_star))

    def encode_group_context(self, group, hP, hc, device):
        summaries = []
        for X in group:
            out = self.forward(X, hP=hP, hc=hc, device=device)
            summaries.append(self._member_summary(out["node_emb"]))
        group_vec = self.group_readout(torch.stack(summaries, dim=0))
        hP_base = hP.to(device=device, dtype=group_vec.dtype).view(-1)
        hc_base = hc.to(device=device, dtype=group_vec.dtype).view(-1)
        return hP_base + self.group_to_hp(group_vec), hc_base + self.group_to_hc(group_vec), group_vec

    def forward_group_step_trainable(
        self,
        group,
        hP,
        hc,
        device=None,
        *,
        X_star: MolState | None = None,
        R_in=None,
        node_feat_in=None,
        adj_carry: torch.Tensor | None = None,
    ):
        if device is None:
            device = next(self.parameters()).device
        if not group:
            z = torch.zeros(0, 3, device=device, dtype=torch.float32)
            return {
                "dR": z,
                "edit_intensity": torch.tensor(0.0, device=device, dtype=torch.float32),
                "node_emb": torch.zeros(0, self.hidden, device=device, dtype=torch.float32),
                "R_pred": z,
                "adj_soft": torch.zeros((0, 0), device=device, dtype=torch.float32),
                "adj_ste": torch.zeros((0, 0), device=device, dtype=torch.float32),
                "node_feat_pred": torch.zeros(0, self.node_feat_dim, device=device, dtype=torch.float32),
                "group_vec": torch.zeros(self.hidden, device=device, dtype=torch.float32),
                "group_size": 0,
            }

        hP_ctx, hc_ctx, group_vec = self.encode_group_context(group, hP, hc, device)
        primary = self._primary_from_group(group, X_star)
        step_out = self.forward_step_trainable(
            primary,
            hP_ctx,
            hc_ctx,
            device=device,
            R_in=R_in,
            node_feat_in=node_feat_in,
            adj_carry=adj_carry,
        )
        step_out["group_vec"] = group_vec
        step_out["group_size"] = len(group)
        step_out["primary"] = primary
        return step_out

    def forward_step_trainable(
        self,
        X,
        hP,
        hc,
        device=None,
        *,
        R_in=None,
        node_feat_in=None,
        adj_carry: torch.Tensor | None = None,
    ):
        if device is None:
            device = next(self.parameters()).device
        Rn = _to_torch(X.R, device) if R_in is None else R_in.to(device=device, dtype=torch.float32)
        out = self.forward(X, hP=hP, hc=hc, device=device, R_in=Rn, node_feat_in=node_feat_in)
        n = int(out["node_emb"].shape[0])
        R_pred = Rn + float(self.step_scale) * out["dR"]
        adj_soft = torch.zeros((0, 0), device=device, dtype=out["node_emb"].dtype)
        adj_ste_out = adj_soft
        if n > 0:
            dtype = out["node_emb"].dtype
            if adj_carry is not None and adj_carry.numel() > 0:
                adj_init = adj_ste(adj_carry.to(device=device, dtype=dtype), threshold=self.edge_threshold)
            else:
                adj_init = _adj_from_graph(X.G, n, device, dtype)
            adj_soft = predict_adj(
                self.edge_denoiser,
                out["node_emb"],
                hP,
                hc,
                T=self.edge_T,
                device=device,
                dtype=dtype,
                adj_init=adj_init,
            )
            adj_ste_out = adj_ste(adj_soft, threshold=self.edge_threshold)
        node_feat_pred = self.node_feat_head(out["node_emb"]) if n > 0 else out["node_emb"]
        return {**out, "R_pred": R_pred, "adj_soft": adj_soft, "adj_ste": adj_ste_out, "node_feat_pred": node_feat_pred}

    def step(self, X, hP, hc):
        device = next(self.parameters()).device
        out = self.forward(X, hP=hP, hc=hc, device=device)
        dR = out["dR"].detach()
        node_emb = out["node_emb"].detach()
        R_new = (X.R + self.step_scale * dR.cpu().numpy()).astype(np.float32)
        n = X.num_nodes()
        if n == 0:
            return X.copy_with(R=R_new)

        dtype = node_emb.dtype
        adj_init = _adj_from_graph(X.G, n, device, dtype)
        adj = sample_adj(
            self.edge_denoiser,
            node_emb,
            hP,
            hc,
            T=self.edge_T,
            device=device,
            dtype=dtype,
            adj_init=adj_init,
        )
        G_new = _graph_from_adj(adj, X.G, threshold=self.edge_threshold)
        node_feat_new = self.node_feat_head(node_emb).detach().cpu().numpy().astype(np.float32)
        return X.copy_with(G=G_new, R=R_new, node_feat=node_feat_new)

    def group_step_trainable_infer(self, group, hP, hc, device=None, *, X_star=None, edge_threshold=None):
        if device is None:
            device = next(self.parameters()).device
        if not group:
            raise ValueError("empty mapping group for group inference")
        thr = float(self.edge_threshold if edge_threshold is None else edge_threshold)
        with torch.no_grad():
            step_out = self.forward_group_step_trainable(group, hP, hc, device=device, X_star=X_star)
            primary = step_out.get("primary") or self._primary_from_group(group, X_star)
            return state_from_step(primary, step_out, edge_threshold=thr)

    def group_step(self, group, hP, hc, *, X_star=None, X_current=None):
        device = next(self.parameters()).device
        if not group:
            raise ValueError("empty mapping group for group inference")
        with torch.no_grad():
            hP_ctx, hc_ctx, _ = self.encode_group_context(group, hP, hc, device)
            primary = X_current if X_current is not None else self._primary_from_group(group, X_star)
            return self.step(primary, hP_ctx, hc_ctx)


@dataclass(frozen=True)
class Route:
    X_init: MolState
    final: MolState
    trajectory: list[MolState]
    init_index: int = -1
    group_states: tuple[MolState, ...] = ()
    group_size: int = 0


def rollout(model, X0, hP, hc, cfg: ABCMConfig, *, target=None, early_stop=True):
    del cfg, target, early_stop  # single-step generation (leap-like)
    model.eval()
    with torch.no_grad():
        X1 = model.step(X0, hP=hP, hc=hc)
    traj = [X0, X1]
    return X1, traj


def group_rollout(
    model,
    group,
    hP,
    hc,
    cfg: ABCMConfig,
    *,
    target=None,
    X_star=None,
    early_stop=True,
    use_trainable_infer=True,
):
    del cfg, early_stop  # single-step generation (leap-like)
    model.eval()
    basin_ref = X_star or target
    members = list(group)
    if not members:
        raise ValueError("empty group for group_rollout")
    primary0 = model._primary_from_group(members, basin_ref)
    device = next(model.parameters()).device
    with torch.no_grad():
        if use_trainable_infer and hasattr(model, "group_step_trainable_infer"):
            X = model.group_step_trainable_infer(members, hP, hc, device=device, X_star=basin_ref)
        else:
            X = model.group_step(members, hP, hc, X_star=basin_ref)
    traj = [primary0, X]
    return X, traj


def _sample_pool_indices(n_pool: int, num_samples: int, seed: int | None) -> list[int]:
    if n_pool <= 0 or num_samples <= 0:
        return []
    rng = random.Random(seed)
    if num_samples <= n_pool:
        return rng.sample(range(n_pool), num_samples)
    return [rng.randrange(n_pool) for _ in range(num_samples)]


def infer_candidates(model, pool, hP, hc, cfg, num_samples=8, X_star=None, seed=0, early_stop=True):
    if not pool:
        return []
    out: list[MolState] = []
    for idx in _sample_pool_indices(len(pool), num_samples, seed):
        final, _ = rollout(model, pool[idx], hP, hc, cfg, target=X_star, early_stop=early_stop)
        out.append(final)
    return out


def infer_routes(model, pool, hP, hc, cfg, num_starts, X_star=None, seed=0, early_stop=True):
    if not pool:
        return []
    routes: list[Route] = []
    for idx in _sample_pool_indices(len(pool), num_starts, seed):
        X0 = pool[idx]
        final, traj = rollout(model, X0, hP, hc, cfg, target=X_star, early_stop=early_stop)
        routes.append(Route(X_init=X0, final=final, trajectory=traj, init_index=idx))
    return routes


def infer_grp_routes(
    model,
    pool,
    hP,
    hc,
    cfg,
    num_starts,
    X_star=None,
    seed=0,
    early_stop=True,
    *,
    tau=None,
    d_edit=None,
    g_min=None,
    g_max=None,
    use_trainable_infer=True,
):
    from dataset.sampling import MapCfg, MapSampler, _key

    if not pool or num_starts <= 0:
        return []
    if X_star is None:
        raise ValueError("infer_grp_routes requires X_star for group sampling")

    sampler_cfg = MapCfg(
        tau=float(tau if tau is not None else getattr(cfg, "tau", 0.92)),
        d_edit=int(d_edit if d_edit is not None else cfg.delta_edit),
        g_min=int(g_min if g_min is not None else getattr(cfg, "grp_min", 2)),
        g_max=int(g_max if g_max is not None else getattr(cfg, "grp_max", 4)),
    )
    sampler = MapSampler(sampler_cfg)
    rng = random.Random(seed)
    routes: list[Route] = []
    seen: set[tuple] = set[tuple]()
    n_try = 0
    budget = max(1, int(num_starts) * sampler_cfg.max_tries)

    while len(routes) < int(num_starts) and n_try < budget:
        n_try += 1
        group = sampler.sample_one(pool, X_star, rng)
        if group is None:
            continue
        sig = tuple[tuple, ...](sorted(_key(s) for s in group.states))
        if sig in seen:
            continue
        seen.add(sig)
        members = list(group.states)
        primary = model._primary_from_group(members, X_star)
        final, traj = group_rollout(
            model,
            members,
            hP,
            hc,
            cfg,
            target=X_star,
            X_star=X_star,
            early_stop=early_stop,
            use_trainable_infer=use_trainable_infer,
        )
        init_idx = -1
        primary_key = _key(primary)
        for j, ms in enumerate(pool):
            if _key(ms) == primary_key:
                init_idx = j
                break
        routes.append(
            Route(
                X_init=primary,
                final=final,
                trajectory=traj,
                init_index=init_idx,
                group_states=tuple(members),
                group_size=len(members),
            )
        )
    return routes
