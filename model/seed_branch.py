from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from config import HIDDEN, N_LAYERS, PRIOR_HIDDEN, SUB_LAYERS
from mol_state import MolState
from utils.chemistry import decode_joint_graph, graph_node_features
from .consistency_operator import Operator
from .edge_diffusion import EdgeNet
from .equivariant_layers import build_core
from .joint_decoder import ChemistryCoupledDecoder, ChemistryHeads, OutputSlots
from .learned_anchor import AnchorSelection, LearnedAnchorScorer
from .pocket_context import PocketContext, summary_of


class Prior(nn.Module):
    """Conditional Gaussian for training and generation.
    Uses no target-conditioned posterior or KL term."""

    def __init__(self, cond_dim: int, z_dim: int, hidden: int | None = None):
        super().__init__()
        h = int(hidden if hidden is not None else PRIOR_HIDDEN)
        self.mu = nn.Sequential(nn.Linear(2 * cond_dim, h), nn.SiLU(), nn.Linear(h, z_dim))
        self.logvar = nn.Sequential(nn.Linear(2 * cond_dim, h), nn.SiLU(), nn.Linear(h, z_dim))

    def forward(self, hP, hc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([summary_of(hP), hc], dim=-1)
        mu = self.mu(x)
        logvar = self.logvar(x).clamp(-10.0, 10.0)
        return mu, logvar

    @staticmethod
    def sample_gaussian(mu, logvar, *, noise=None):
        if mu.shape != logvar.shape:
            raise ValueError("Gaussian mean and log variance must have matching shapes")
        noise = torch.randn_like(mu) if noise is None else noise
        if noise.shape != mu.shape:
            raise ValueError("Gaussian noise must match the mean shape")
        return mu + noise * torch.exp(0.5 * logvar)

    def sample(self, hP, hc: torch.Tensor, *, num_samples: int = 1) -> torch.Tensor:
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        mu, logvar = self.forward(hP, hc)
        return self.sample_gaussian(mu.unsqueeze(0).expand(num_samples, -1),
                                    logvar.unsqueeze(0).expand(num_samples, -1))

    def rsample(self, hP, hc: torch.Tensor) -> torch.Tensor:
        return self.sample_gaussian(*self.forward(hP, hc))


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
        chemistry_heads=None,
        max_nodes=128,
        joint_decoder=None,
        joint_steps=3,
        pocket_neighbors=32,
        pocket_step_scale=0.2,
        num_anchors=4,
    ):
        super().__init__()
        self.core = core
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.num_nodes = int(num_nodes)
        self.max_nodes = int(max_nodes)
        if not 1 <= self.num_nodes <= self.max_nodes:
            raise ValueError("Configured seed num_nodes must lie in [1, max_nodes]")
        self.node_feat_dim = node_feat_dim
        self.step_scale = step_scale
        self.r0_scale = r0_scale
        self.edge_T = int(edge_T)
        h = int(hidden or core.hidden)
        cond2 = 2 * cond_dim
        seed_in = z_dim + cond2 + 3 + 3
        self.seed_node_mlp = nn.Sequential(nn.Linear(seed_in, h), nn.SiLU(), nn.Linear(h, h))
        self.pocket_source_mlp = nn.Sequential(
            nn.Linear(cond_dim + 3, h), nn.SiLU(), nn.Linear(h, h),
        )
        self.latent_condition = nn.Linear(z_dim, cond_dim)
        self.output_slots = OutputSlots(h, cond_dim, self.max_nodes, noise_scale=float(r0_scale))
        self.anchor_scorer = LearnedAnchorScorer(cond_dim, h, num_anchors=num_anchors)
        self._edge_net_mod = edge_denoiser if edge_denoiser is not None else EdgeNet(
            node_dim=h, cond_dim=cond_dim, time_dim=edge_time_dim, hidden=h
        )
        self._feat_head_mod = node_feat_head if node_feat_head is not None else nn.Linear(h, node_feat_dim)
        self.chemistry_heads = chemistry_heads if chemistry_heads is not None else ChemistryHeads(h)
        self.count_head = nn.Sequential(nn.Linear(z_dim + cond2, h), nn.SiLU(), nn.Linear(h, self.max_nodes))
        self.joint_decoder = joint_decoder if joint_decoder is not None else ChemistryCoupledDecoder(
            h, cond_dim, steps=joint_steps, neighbors=pocket_neighbors,
            step_scale=pocket_step_scale,
        )

    @property
    def edge_denoiser(self):
        return self._edge_net_mod

    def _edge_net(self):
        return self._edge_net_mod

    def _feat_head(self):
        return self._feat_head_mod

    def _count_logits(self, z, hP, hc):
        return self.count_head(torch.cat([z.reshape(-1), summary_of(hP).reshape(-1), hc.reshape(-1)]))

    def select_anchors(self, anchors, hP, hc) -> AnchorSelection:
        """Select learned pocket anchors or use caller-supplied coordinates."""
        summary = summary_of(hP)
        device = summary.device
        coordinate_dtype = torch.float64 if summary.dtype == torch.float64 else torch.float32
        if anchors is None or anchors.numel() == 0:
            if isinstance(hP, PocketContext):
                return self.anchor_scorer(hP, hc)
            # A tensor summary without anchors uses the origin.
            positions = torch.zeros((1, 3), device=device, dtype=coordinate_dtype)
            features = summary.reshape(1, -1)
            return AnchorSelection(positions, features, summary.new_zeros(1),
                                   torch.empty(0, device=device, dtype=torch.long),
                                   summary.new_empty(0), "summary_only_test_fallback")
        positions = anchors.to(device=device, dtype=coordinate_dtype).reshape(-1, 3)
        if not bool(torch.isfinite(positions).all()):
            raise ValueError("Explicit anchor coordinates must be finite")
        if isinstance(hP, PocketContext):
            distances = torch.cdist(positions, hP.positions.to(dtype=coordinate_dtype),
                                    compute_mode="donot_use_mm_for_euclid_dist")
            weights = (-distances / 6.).softmax(-1).to(dtype=hP.node_features.dtype)
            features = weights @ hP.node_features
        else:
            features = summary.reshape(1, -1).expand(len(positions), -1)
        return AnchorSelection(positions, features, summary.new_zeros(len(positions)),
                               torch.empty(0, device=device, dtype=torch.long),
                               summary.new_empty(0), "explicit_override")

    def _initial_slots(self, z, anchors, hP, hc, num_nodes, *, selection=None):
        """Use latent-conditioned attention over anchors to initialize output slots."""
        device, dtype = z.device, z.dtype
        coordinate_dtype = dtype if dtype in (torch.float32, torch.float64) else torch.float32
        summary = summary_of(hP).to(device=device, dtype=dtype).reshape(-1)
        selection = self.select_anchors(anchors, hP, hc) if selection is None else selection
        source_coords = selection.positions.to(device=device, dtype=coordinate_dtype)
        source_features = selection.features.to(device=device, dtype=dtype)
        # Feature weighting trains scores even when top-k selects every candidate.
        if selection.mode == "learned_topk":
            source_features = source_features * (1. + selection.scores.sigmoid().unsqueeze(-1))
        anchor_mean = source_coords.mean(dim=0)
        distances = torch.linalg.vector_norm(source_coords - anchor_mean, dim=-1, keepdim=True) / 6.0
        source_geometry = torch.cat([distances, distances.square(), torch.exp(-distances)], dim=-1)
        source_emb = self.pocket_source_mlp(torch.cat([source_features, source_geometry], dim=-1))
        slot_hidden, coords = self.output_slots(
            source_emb, source_coords, summary + self.latent_condition(z), hc,
            num_nodes,
        )
        return slot_hidden, coords, anchor_mean, summary

    def forward_trainable(self, z, anchors, hP, hc, *, num_nodes: int | None = None):
        """Use num_nodes or the configured training size; also predict count logits."""
        device, dtype = z.device, z.dtype
        n = self.num_nodes if num_nodes is None else int(num_nodes)
        if not 1 <= n <= self.max_nodes:
            raise ValueError(f"num_nodes must lie in [1, {self.max_nodes}], got {n}")
        z_flat = z.reshape(-1)
        if z_flat.shape[0] != self.z_dim:
            raise ValueError(f"z dim {z_flat.shape[0]} != z_dim {self.z_dim}")
        count_logits = self._count_logits(z_flat, hP, hc)
        selection = self.select_anchors(anchors, hP, hc)
        slot_hidden, R0, anchor_mean, summary = self._initial_slots(
            z_flat, anchors, hP, hc, n, selection=selection,
        )
        dtype = R0.dtype
        local_R = R0 - anchor_mean
        radial = torch.linalg.vector_norm(local_R, dim=-1, keepdim=True)
        radial_features = torch.cat([radial, radial.square(), radial.pow(3)], dim=-1)
        if selection.positions.numel() > 0:
            distances = torch.cdist(R0, selection.positions.to(device=device, dtype=R0.dtype),
                                    compute_mode="donot_use_mm_for_euclid_dist")
            anchor_features = torch.stack([
                distances.min(dim=-1).values, distances.mean(dim=-1),
                distances.std(dim=-1, unbiased=False),
            ], dim=-1)
        else:
            anchor_features = torch.zeros((n, 3), device=device, dtype=dtype)
        seed_in = torch.cat([
            z_flat.unsqueeze(0).expand(n, -1), summary.unsqueeze(0).expand(n, -1),
            hc.reshape(-1).unsqueeze(0).expand(n, -1), radial_features, anchor_features,
        ], dim=-1)
        hidden_in = slot_hidden + self.seed_node_mlp(seed_in)
        out = self.core.forward_from_node_emb(hidden_in, R0, summary, hc, device, mol_graph=None)
        hidden = out["node_emb"]
        R1 = R0 + float(self.step_scale) * out["dR"]
        joint = self.joint_decoder(hidden, R1, hP, hc, self.chemistry_heads, self._edge_net())
        features = self._feat_head()(joint["node_emb"])
        return {
            **joint, "R": joint["R_pred"], "node_feat": features,
            "node_feat_pred": features, "R0": R0, "count_logits": count_logits,
            "anchor_positions": selection.positions, "anchor_scores": selection.scores,
            "anchor_candidate_scores": selection.candidate_scores,
            "anchor_indices": selection.indices, "anchor_mode": selection.mode,
        }

    def forward(self, z, anchors, hP, hc, *, num_nodes: int | None = None):
        with torch.no_grad():
            n = int(num_nodes) if num_nodes is not None else int(
                torch.distributions.Categorical(logits=self._count_logits(z, hP, hc)).sample().item()
            ) + 1
            out = self.forward_trainable(z, anchors, hP, hc, num_nodes=n)
            graph = decode_joint_graph(out)
            return MolState(
                G=graph, R=out["R"].detach().cpu().numpy().astype(np.float32),
                node_feat=graph_node_features(graph, self.node_feat_dim),
            )


@dataclass(frozen=True)
class SeedBranch:
    prior: Prior
    seed: SeedGeom

    def sample_X0(self, *, hP, hc, anchors=None, num_samples=1, num_nodes=None) -> list[MolState]:
        z = self.prior.sample(hP, hc, num_samples=num_samples)
        return [self.seed.forward(z[i], anchors, hP, hc, num_nodes=num_nodes) for i in range(num_samples)]


def build_model(
    node_feat_dim,
    cond_dim,
    z_dim,
    num_nodes,
    shared_core=True,
    core_kwargs=None,
    edge_T=20,
    prior_hidden: int | None = None,
    max_nodes: int = 128,
    joint_steps: int = 3,
    pocket_neighbors: int = 32,
    pocket_step_scale: float = 0.2,
    num_anchors: int = 4,
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
        max_nodes=max_nodes,
        joint_steps=joint_steps,
        pocket_neighbors=pocket_neighbors,
        pocket_step_scale=pocket_step_scale,
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
        chemistry_heads=op.chemistry_heads,
        max_nodes=max_nodes,
        joint_decoder=op.joint_decoder,
        num_anchors=num_anchors,
    )
    prior = Prior(cond_dim, z_dim, hidden=prior_hidden)
    return op, seed, prior
