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
from utils.chemistry import decode_graph, decode_joint_graph, graph_node_features
from .edge_diffusion import EdgeNet
from .equivariant_layers import build_core
from .pocket_context import PocketInteraction, summary_of, with_summary
from .joint_decoder import ChemistryCoupledDecoder, ChemistryHeads, OutputSlots
from .state_distance import GraphStateEncoder


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


def state_from_step(X: MolState, step_out: dict, *, edge_threshold: float = 0.5) -> MolState:
    """Decode predicted chemistry, independent of the source graph's atom types."""
    with torch.no_grad():
        G = decode_joint_graph(step_out) if "edge_logits" in step_out else decode_graph(
            step_out["atom_logits"], step_out["charge_logits"],
            step_out["hydrogen_logits"], step_out["bond_logits"],
            step_out["adj_soft"], threshold=edge_threshold,
        )
        R = step_out["R_pred"].detach().cpu().numpy().astype(np.float32)
        features = graph_node_features(G, feat_dim=int(X.node_feat.shape[1]))
    return MolState(G=G, R=R, node_feat=features)


def adj_carry(step_out: dict) -> torch.Tensor | None:
    adj = step_out.get("adj_ste", step_out.get("adj_soft"))
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
        max_nodes=128,
        joint_steps=3,
        pocket_neighbors=32,
        pocket_step_scale=0.2,
    ):
        super().__init__()
        self.hidden = hidden
        self.max_nodes = int(max_nodes)
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be positive")
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
        self.graph_encoder = GraphStateEncoder(hidden)
        self.chemistry_heads = ChemistryHeads(hidden)
        self.count_head = nn.Sequential(
            nn.Linear(hidden + cond2, hidden), nn.SiLU(), nn.Linear(hidden, self.max_nodes),
        )
        self.output_slots = OutputSlots(hidden, self.cond_dim, self.max_nodes)
        self.pocket_interaction = PocketInteraction(hidden, self.cond_dim, neighbors=pocket_neighbors,
                                                   step_scale=pocket_step_scale)
        self.joint_decoder = ChemistryCoupledDecoder(hidden, self.cond_dim, steps=joint_steps,
                                                     neighbors=pocket_neighbors, step_scale=pocket_step_scale)

    @property
    def use_radius_edges(self) -> bool:
        return bool(self.core.use_radius_edges)

    def forward(self, X, hP, hc, device, *, R_in=None, node_feat_in=None):
        R = _to_torch(X.R, device) if R_in is None else R_in.to(device=device, dtype=torch.float32)
        node_feat = _to_torch(X.node_feat, device) if node_feat_in is None else node_feat_in.to(device=device, dtype=torch.float32)
        n = node_feat.shape[0]

        pocket_summary = summary_of(hP).to(device=device, dtype=torch.float32)
        pocket_tile = pocket_summary.view(1, -1).repeat(n, 1)
        task_tile = hc.to(device=device, dtype=torch.float32).view(1, -1).repeat(n, 1)
        node_emb = self.node_proj(torch.cat([node_feat, pocket_tile, task_tile], dim=-1))
        node_emb, cross_R = self.pocket_interaction(node_emb, R, hP)
        out = self.core.forward_from_node_emb(
            node_emb, cross_R, pocket_summary, hc, device, mol_graph=X.G
        )
        return {"dR": cross_R - R + out["dR"], "edit_intensity": out["edit_intensity"], "node_emb": out["node_emb"]}

    def _member_summary(self, node_emb: torch.Tensor) -> torch.Tensor:
        return torch.cat([node_emb.mean(dim=0), node_emb.std(dim=0, unbiased=False).clamp(min=1e-6)], dim=-1)

    def _primary_from_group(self, group: list[MolState], X_star: MolState | None) -> MolState:
        """Display member; full-group decoding uses every member."""
        if not group:
            raise ValueError("empty mapping group")
        return group[0]

    def _encode_group(self, group, hP, hc, device):
        if not group:
            raise ValueError("empty mapping group")
        embeddings = [self.forward(X, hP=hP, hc=hc, device=device)["node_emb"] for X in group]
        summaries = [self._member_summary(nodes) for nodes in embeddings]
        group_vec = self.group_readout(torch.stack(summaries, dim=0))
        return embeddings, group_vec

    def _group_conditions(self, hP, hc, group_vec, device):
        hP_base = summary_of(hP).to(device=device, dtype=group_vec.dtype).view(-1)
        hc_base = hc.to(device=device, dtype=group_vec.dtype).view(-1)
        return with_summary(hP, hP_base + self.group_to_hp(group_vec)), hc_base + self.group_to_hc(group_vec)

    def encode_group_context(self, group, hP, hc, device):
        _, group_vec = self._encode_group(group, hP, hc, device)
        hP_ctx, hc_ctx = self._group_conditions(hP, hc, group_vec, device)
        return hP_ctx, hc_ctx, group_vec

    def forward_group_step_trainable(
        self,
        group,
        hP,
        hc,
        device=None,
        *,
        X_star: MolState | None = None,
        adj_carry: torch.Tensor | None = None,
        num_nodes: int | None = None,
        predict_count: bool = True,
        group_context: bool = True,
        slot_noise=None,
    ):
        """Decode the full member set; targets are never read by this forward.
        Training supplies num_nodes explicitly; otherwise the count is predicted."""
        if device is None:
            device = next(self.parameters()).device
        if num_nodes is None and not predict_count:
            raise ValueError("Group teacher forcing requires explicit num_nodes")
        embeddings, group_vec = self._encode_group(group, hP, hc, device)
        source_emb = torch.cat(embeddings, dim=0)
        source_coords = torch.cat([_to_torch(X.R, device) for X in group], dim=0)
        log_weights = torch.cat([nodes.new_full((len(nodes),), -np.log(len(nodes))) for nodes in embeddings])
        member_mean = torch.stack([nodes.mean(0) for nodes in embeddings]).mean(0)
        count_logits = self.count_head(torch.cat([member_mean, summary_of(hP).reshape(-1), hc.reshape(-1)]))
        hP_ctx, hc_ctx = self._group_conditions(hP, hc, group_vec, device) if group_context else (hP, hc)
        step_out = self._decode_sources(
            source_emb, source_coords, hP_ctx, hc_ctx, count_logits,
            num_nodes=num_nodes, adj_carry=adj_carry, source_log_weights=log_weights,
            slot_noise=slot_noise,
        )
        step_out["group_vec"] = group_vec
        step_out["group_size"] = len(group)
        step_out["primary"] = self._primary_from_group(group, None)
        return step_out

    def forward_step_trainable(
        self, X, hP, hc, device=None, *, R_in=None, node_feat_in=None,
        adj_carry: torch.Tensor | None = None, num_nodes: int | None = None,
        predict_count: bool = False, decode_adjacency: bool = True,
    ):
        """Decode graph then geometry; num_nodes overrides the output count.
        Without num_nodes, predict_count=True predicts a count; False keeps the input size.
        decode_adjacency is accepted but does not affect decoding."""
        if device is None:
            device = next(self.parameters()).device
        Rn = _to_torch(X.R, device) if R_in is None else R_in.to(device=device, dtype=torch.float32)
        encoded = self.forward(X, hP=hP, hc=hc, device=device, R_in=Rn, node_feat_in=node_feat_in)
        source_emb = encoded["node_emb"]
        if source_emb.shape[0] == 0:
            raise ValueError("Cannot map an empty molecular input; use the seed branch")
        hp = summary_of(hP)
        count_logits = self.count_head(torch.cat([source_emb.mean(dim=0), hp.reshape(-1), hc.reshape(-1)]))
        if num_nodes is None and not predict_count:
            num_nodes = int(source_emb.shape[0])
        return self._decode_sources(source_emb, Rn, hP, hc, count_logits,
                                    num_nodes=num_nodes, adj_carry=adj_carry)

    def _decode_sources(self, source_emb, source_coords, hP, hc, count_logits, *,
                        num_nodes=None, adj_carry=None, source_log_weights=None, slot_noise=None):
        device = source_emb.device
        hp = summary_of(hP)
        n = int(count_logits.argmax().item()) + 1 if num_nodes is None else int(num_nodes)
        slot_emb, R0 = self.output_slots(source_emb, source_coords, hp, hc, n,
                                        source_log_weights=source_log_weights, noise=slot_noise)
        decoded = self.core.forward_from_node_emb(slot_emb, R0, hp, hc, device, mol_graph=None)
        hidden = decoded["node_emb"]
        R_pred = R0 + float(self.step_scale) * decoded["dR"]
        # Output slots have no atom correspondence to the input graph.
        adj_init = None
        if adj_carry is not None:
            if tuple(adj_carry.shape) != (n, n):
                raise ValueError("Carried adjacency does not match the requested output count")
            # Carry five-class decisions and gradients without binary rethresholding.
            adj_init = adj_carry.to(device=device, dtype=hidden.dtype)
        joint = self.joint_decoder(hidden, R_pred, hP, hc, self.chemistry_heads,
                                   self.edge_denoiser, adj_init=adj_init)
        return {
            **decoded, **joint, "R0": R0,
            "node_feat_pred": self.node_feat_head(joint["node_emb"]), "count_logits": count_logits,
        }

    def step(self, X, hP, hc, *, num_nodes: int | None = None):
        """Generate a molecule with a predicted or specified atom count."""
        with torch.no_grad():
            out = self.forward_step_trainable(
                X, hP, hc, num_nodes=num_nodes, predict_count=True, decode_adjacency=False,
            )
            return state_from_step(X, out, edge_threshold=self.edge_threshold)

    def group_step_trainable_infer(self, group, hP, hc, device=None, *, X_star=None, edge_threshold=None):
        thr = float(self.edge_threshold if edge_threshold is None else edge_threshold)
        with torch.no_grad():
            step_out = self.forward_group_step_trainable(group, hP, hc, device=device, X_star=X_star, predict_count=True)
            return state_from_step(step_out["primary"], step_out, edge_threshold=thr)

    def group_step(self, group, hP, hc, *, X_star=None, X_current=None, num_nodes=None,
                   group_context=True):
        with torch.no_grad():
            out = self.forward_group_step_trainable(group, hP, hc, num_nodes=num_nodes,
                                                     group_context=group_context)
            return state_from_step(out["primary"], out, edge_threshold=self.edge_threshold)


@dataclass(frozen=True)
class Route:
    X_init: MolState
    final: MolState
    trajectory: list[MolState]
    init_index: int = -1
    group_states: tuple[MolState, ...] = ()
    group_size: int = 0


def rollout(model, X0, hP, hc, cfg: ABCMConfig, *, target=None, early_stop=True):
    del cfg, target, early_stop  # One generation step; these arguments do not affect it.
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
):
    del cfg, early_stop  # One generation step; these arguments do not affect it.
    model.eval()
    members = list(group)
    if not members:
        raise ValueError("empty group for group_rollout")
    primary0 = model._primary_from_group(members, None)
    device = next(model.parameters()).device
    with torch.no_grad():
        X = model.group_step_trainable_infer(members, hP, hc, device=device)
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


def is_recoverable_generation_error(error):
    """Molecular failures may be reported per sample; device/resource failures abort."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return False
    message = str(error).lower()
    if any(token in message for token in ("out of memory", "cuda error", "cudnn", "cublas", "device-side assert")):
        return False
    return isinstance(error, (ValueError, RuntimeError))


def _report_generation_error(error, on_error, *, init_index, group_size):
    recoverable = is_recoverable_generation_error(error)
    if on_error is not None:
        on_error({"pool_init_index": init_index, "group_size": group_size,
                  "error_type": type(error).__name__, "error": str(error), "recoverable": recoverable})
    if on_error is None or not recoverable:
        raise error


def infer_routes(model, pool, hP, hc, cfg, num_starts, X_star=None, seed=0, early_stop=True, *, on_error=None, on_route=None):
    if not pool:
        return []
    routes: list[Route] = []
    for idx in _sample_pool_indices(len(pool), num_starts, seed):
        X0 = pool[idx]
        try:
            final, traj = rollout(model, X0, hP, hc, cfg, target=X_star, early_stop=early_stop)
        except (ValueError, RuntimeError) as error:
            _report_generation_error(error, on_error, init_index=idx, group_size=0)
            continue
        route = Route(X_init=X0, final=final, trajectory=traj, init_index=idx)
        routes.append(route)
        if on_route is not None:
            on_route(route)
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
    on_error=None,
    on_route=None,
):
    from dataset.sampling import MapCfg, MapSampler, _key

    if not pool or num_starts <= 0:
        return []
    if X_star is None:
        raise ValueError("infer_grp_routes requires X_star for group sampling")

    sampler_cfg = MapCfg(
        tau=float(tau if tau is not None else getattr(cfg, "tau", 0.92)),
        d_edit=int(d_edit if d_edit is not None else cfg.delta_edit),
        g_min=int(g_min if g_min is not None else getattr(cfg, "grp_min", 3)),
        g_max=int(g_max if g_max is not None else getattr(cfg, "grp_max", 6)),
        distance_config=getattr(cfg, "molecular_distance", None),
        max_tries=int(getattr(cfg, "group_max_tries", 50)),
    )
    sampler = MapSampler(sampler_cfg)
    rng = random.Random(seed)
    routes: list[Route] = []
    seen: set[tuple] = set[tuple]()
    n_try = 0
    n_attempted = 0
    budget = max(1, int(num_starts) * sampler_cfg.max_tries)

    while n_attempted < int(num_starts) and n_try < budget:
        n_try += 1
        group = sampler.sample_one(pool, X_star, rng)
        if group is None:
            continue
        sig = tuple[tuple, ...](sorted(_key(s) for s in group.states))
        if sig in seen:
            continue
        seen.add(sig)
        n_attempted += 1
        members = list(group.states)
        init_idx = -1
        try:
            primary = model._primary_from_group(members, None)
            primary_key = _key(primary)
            for j, ms in enumerate(pool):
                if _key(ms) == primary_key:
                    init_idx = j
                    break
            final, traj = group_rollout(
                model, members, hP, hc, cfg, target=X_star, X_star=X_star,
                early_stop=early_stop,
            )
        except (ValueError, RuntimeError) as error:
            _report_generation_error(error, on_error, init_index=init_idx, group_size=len(members))
            continue
        route = Route(
            X_init=primary,
            final=final,
            trajectory=traj,
            init_index=init_idx,
            group_states=tuple(members),
            group_size=len(members),
        )
        routes.append(route)
        if on_route is not None:
            on_route(route)
    return routes
