from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from config import HIDDEN, N_LAYERS, PRIOR_HIDDEN, SUB_LAYERS
from mol_state import MolState
from .consistency_operator import Operator
from .edge_diffusion import EdgeNet, sample_adj
from .equivariant_layers import build_core


class Prior(nn.Module):
    def __init__(self, cond_dim: int, z_dim: int, hidden: int | None = None):
        super().__init__()
        h = int(hidden if hidden is not None else PRIOR_HIDDEN)
        self.mu = nn.Sequential(nn.Linear(2 * cond_dim, h), nn.SiLU(), nn.Linear(h, z_dim))
        self.logvar = nn.Sequential(nn.Linear(2 * cond_dim, h), nn.SiLU(), nn.Linear(h, z_dim))

    def forward(self, hP: torch.Tensor, hc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([hP, hc], dim=-1)
        mu = self.mu(x)
        logvar = self.logvar(x).clamp(-10.0, 10.0)
        return mu, logvar

    def sample(self, hP: torch.Tensor, hc: torch.Tensor, *, num_samples: int = 1) -> torch.Tensor:
        mu, logvar = self.forward(hP, hc)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn((num_samples, mu.shape[-1]), device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(0) + eps * std.unsqueeze(0)

    def rsample(self, hP: torch.Tensor, hc: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.forward(hP, hc)
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std


class SeedGeom(nn.Module):
    def __init__(
        self,
        core,
        z_dim,
        cond_dim,
        num_nodes,
        node_feat_dim=16,
        step_scale=0.1,
        r0_scale=0.15,
        edge_T=20,
        edge_time_dim=64,
        hidden=None,
        edge_denoiser=None,
        node_feat_head=None,
    ):
        super().__init__()
        self.core = core
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.step_scale = step_scale
        self.r0_scale = r0_scale
        self.edge_T = int(edge_T)
        h = int(hidden or core.hidden)
        cond2 = 2 * cond_dim
        seed_in = z_dim + cond2 + 3 + 3
        self.seed_node_mlp = nn.Sequential(nn.Linear(seed_in, h), nn.SiLU(), nn.Linear(h, h))
        self._edge_net_mod = edge_denoiser if edge_denoiser is not None else EdgeNet(
            node_dim=h, cond_dim=cond_dim, time_dim=edge_time_dim, hidden=h
        )
        self._feat_head_mod = node_feat_head if node_feat_head is not None else nn.Linear(h, node_feat_dim)

    @property
    def edge_denoiser(self):
        return self._edge_net_mod

    def _edge_net(self):
        return self._edge_net_mod

    def _feat_head(self):
        return self._feat_head_mod

    def forward_trainable(self, z, anchors, hP, hc):
        device = z.device
        dtype = z.dtype
        n = self.num_nodes
        z_flat = z.reshape(-1)
        if z_flat.shape[0] != self.z_dim:
            raise ValueError(f"z dim {z_flat.shape[0]} != z_dim {self.z_dim}")

        R0 = torch.randn((n, 3), device=device, dtype=dtype) * float(self.r0_scale)
        anchor_mean = (
            anchors.to(device=device, dtype=dtype).mean(dim=0)
            if anchors is not None and anchors.numel() > 0
            else torch.zeros(3, device=device, dtype=dtype)
        )
        seed_in = torch.cat(
            [
                z_flat.unsqueeze(0).expand(n, -1),
                hP.reshape(-1).unsqueeze(0).expand(n, -1),
                hc.reshape(-1).unsqueeze(0).expand(n, -1),
                R0,
                anchor_mean.unsqueeze(0).expand(n, -1),
            ],
            dim=-1,
        )
        node_emb_in = self.seed_node_mlp(seed_in)
        out = self.core.forward_from_node_emb(node_emb_in, R0, hP, hc, device, mol_graph=None)
        hidden = out["node_emb"]
        R1 = R0 + float(self.step_scale) * out["dR"]
        return {"R": R1, "node_feat": self._feat_head()(hidden), "node_emb": hidden, "R0": R0}

    def forward(self, z, anchors, hP, hc):
        device = z.device
        dtype = z.dtype
        n = self.num_nodes
        z_flat = z.view(-1)
        if z_flat.shape[0] != self.z_dim:
            raise ValueError(f"z dim {z_flat.shape[0]} != z_dim {self.z_dim}")

        R0 = torch.randn((n, 3), device=device, dtype=dtype) * float(self.r0_scale)
        anchor_mean = (
            anchors.to(device=device, dtype=dtype).mean(dim=0)
            if anchors is not None and anchors.numel() > 0
            else torch.zeros(3, device=device, dtype=dtype)
        )
        seed_in = torch.cat(
            [
                z_flat.unsqueeze(0).expand(n, -1),
                hP.reshape(-1).unsqueeze(0).expand(n, -1),
                hc.reshape(-1).unsqueeze(0).expand(n, -1),
                R0,
                anchor_mean.unsqueeze(0).expand(n, -1),
            ],
            dim=-1,
        )
        node_emb = self.seed_node_mlp(seed_in)
        out = self.core.forward_from_node_emb(node_emb, R0, hP, hc, device, mol_graph=None)
        R1 = (R0 + float(self.step_scale) * out["dR"]).detach()
        hidden = out["node_emb"]
        adj = sample_adj(self._edge_net(), hidden, hP, hc, T=self.edge_T, device=device, dtype=dtype)

        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if float(adj[i, j].item()) >= 0.5:
                    G.add_edge(i, j)

        node_feat = self._feat_head()(hidden).detach().cpu().numpy().astype(np.float32)
        return MolState(G=G, R=R1.detach().cpu().numpy().astype(np.float32), node_feat=node_feat)


@dataclass(frozen=True)
class SeedBranch:
    prior: Prior
    seed: SeedGeom

    def sample_X0(self, *, hP, hc, anchors=None, num_samples=1) -> list[MolState]:
        z = self.prior.sample(hP, hc, num_samples=num_samples)
        return [self.seed.forward(z[i], anchors, hP, hc) for i in range(num_samples)]


def build_model(
    node_feat_dim,
    cond_dim,
    z_dim,
    num_nodes,
    shared_core=True,
    core_kwargs=None,
    edge_T=20,
    prior_hidden: int | None = None,
):
    core_kw = dict(
        hidden=HIDDEN,
        num_layers=N_LAYERS,
        num_rbf=16,
        rbf_max_dist=6.0,
        radius_cutoff=6.0,
        use_radius_edges=True,
        mpnn_sub_layers=SUB_LAYERS,
    )
    if core_kwargs:
        core_kw.update(core_kwargs)
    core_kw.pop("backbone", None)

    core_op = build_core(cond_dim, core_kw)
    core_seed = core_op if shared_core else build_core(cond_dim, core_kw)

    op = Operator(
        node_feat_dim,
        cond_dim,
        hidden=core_kw["hidden"],
        num_layers=core_kw["num_layers"],
        num_rbf=core_kw["num_rbf"],
        rbf_max_dist=core_kw["rbf_max_dist"],
        radius_cutoff=core_kw["radius_cutoff"],
        use_radius_edges=core_kw["use_radius_edges"],
        core=core_op,
        edge_T=edge_T,
    )
    seed = SeedGeom(
        core_seed,
        z_dim,
        cond_dim=cond_dim,
        num_nodes=num_nodes,
        node_feat_dim=node_feat_dim,
        step_scale=float(op.step_scale),
        hidden=core_kw["hidden"],
        edge_T=edge_T,
        edge_denoiser=op.edge_denoiser,
        node_feat_head=op.node_feat_head,
    )
    prior = Prior(cond_dim, z_dim, hidden=prior_hidden)
    return op, seed, prior
