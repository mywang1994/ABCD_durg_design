from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from model.pocket_context import PocketInteraction, summary_of
from utils.chemistry import ATOMIC_NUMBERS, BOND_TYPES, EXPLICIT_HS, FORMAL_CHARGES


class ChemistryHeads(nn.Module):
    """Predict node chemistry and bond order conditional on a connected pair."""

    def __init__(self, hidden: int):
        super().__init__()
        self.atom_head = nn.Linear(hidden, len(ATOMIC_NUMBERS))
        self.charge_head = nn.Linear(hidden, len(FORMAL_CHARGES))
        self.hydrogen_head = nn.Linear(hidden, len(EXPLICIT_HS))
        self.bond_head = nn.Sequential(
            nn.Linear(2 * hidden + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, len(BOND_TYPES)),
        )

    def forward(self, node_emb: torch.Tensor, coords: torch.Tensor) -> dict:
        n = node_emb.shape[0]
        left = node_emb[:, None, :].expand(n, n, -1)
        right = node_emb[None, :, :].expand(n, n, -1)
        distances = torch.linalg.vector_norm(coords[:, None] - coords[None, :], dim=-1, keepdim=True)
        pair = torch.cat([left + right, (left - right).abs(), distances], dim=-1)
        return {
            "atom_logits": self.atom_head(node_emb),
            "charge_logits": self.charge_head(node_emb),
            "hydrogen_logits": self.hydrogen_head(node_emb),
            "bond_logits": self.bond_head(pair),
        }


class OutputSlots(nn.Module):
    """Initialize slots from permutation-invariant attention and isotropic noise.
    Coordinates are translation equivariant and rotation equivariant in distribution;
    pathwise rotation equivariance requires rotating the sampled noise."""

    def __init__(self, hidden: int, cond_dim: int, max_nodes: int, noise_scale: float = 0.2):
        super().__init__()
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        self.max_nodes = int(max_nodes)
        self.noise_scale = float(noise_scale)
        self.queries = nn.Parameter(torch.randn(max_nodes, hidden) / math.sqrt(hidden))
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.condition = nn.Linear(2 * cond_dim, hidden)
        self.mix = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, source_emb, source_coords, hP, hc, num_nodes: int, *, noise=None,
                source_log_weights=None):
        n = int(num_nodes)
        if not 1 <= n <= self.max_nodes:
            raise ValueError(f"num_nodes must be in [1, {self.max_nodes}], got {n}")
        if source_emb.shape[0] < 1 or source_coords.shape != (source_emb.shape[0], 3):
            raise ValueError("OutputSlots requires a nonempty source set with matching coordinates")
        cond = self.condition(torch.cat([hP.reshape(-1), hc.reshape(-1)]))
        queries = self.queries[:n] + cond
        logits = queries @ self.key(source_emb).T / math.sqrt(source_emb.shape[-1])
        if source_log_weights is not None:
            logits = logits + source_log_weights
        weights = torch.softmax(logits, dim=-1)
        context = weights @ self.value(source_emb)
        hidden = self.mix(torch.cat([queries, context], dim=-1))
        if noise is None:
            noise = torch.randn((n, 3), device=source_coords.device, dtype=source_coords.dtype)
        if noise.shape != (n, 3):
            raise ValueError(f"slot noise must have shape ({n}, 3)")
        coords = weights @ source_coords + self.noise_scale * noise
        return hidden, coords


def _masked_edge_logits(logits: torch.Tensor, diagonal: torch.Tensor) -> torch.Tensor:
    # Finite diagonal logits keep softmax defined for singleton graphs.
    no_self = logits.new_tensor([0.0, -1.0e4, -1.0e4, -1.0e4, -1.0e4])
    return torch.where(diagonal[..., None], no_self, logits)


def _map_straight_through(logits: torch.Tensor) -> torch.Tensor:
    """Forward: categorical MAP; backward: softmax surrogate derivative."""
    probabilities = torch.softmax(logits, dim=-1)
    labels = probabilities.argmax(dim=-1)
    hard = F.one_hot(labels, num_classes=logits.shape[-1]).to(probabilities.dtype)
    # An exact-zero residual preserves binary states at low precision.
    return hard + (probabilities - probabilities.detach())


class GraphConditionedGeometryLayer(nn.Module):
    """Scalar attention with SE(3)-equivariant vector updates, conditioned on a fixed graph.
    Self attention normalizes singleton graphs without displacing them."""

    def __init__(self, hidden: int, cond_dim: int, step_scale: float, neighbors: int):
        super().__init__()
        self.num_heads = math.gcd(hidden, 4)
        self.head_dim = hidden // self.num_heads
        chemical_dim = len(ATOMIC_NUMBERS) + len(FORMAL_CHARGES) + len(EXPLICIT_HS)
        self.chemistry_embedding = nn.Linear(chemical_dim, hidden)
        self.condition = nn.Linear(2 * cond_dim, hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.register_buffer("radial_centers", torch.linspace(0.0, 10.0, 16))
        self.edge_attention_bias = nn.Sequential(
            nn.Linear(5 + 16, hidden), nn.SiLU(), nn.Linear(hidden, self.num_heads),
        )
        self.edge_value = nn.Linear(5 + 16, hidden)
        self.attention_out = nn.Linear(hidden, hidden)
        self.attention_norm = nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.SiLU(),
                                         nn.Linear(2 * hidden, hidden))
        self.feedforward_norm = nn.LayerNorm(hidden)
        self.coordinate_gate = nn.Sequential(
            nn.Linear(2 * hidden + 5 + 16, hidden), nn.SiLU(),
            nn.Linear(hidden, self.num_heads), nn.Tanh(),
        )
        self.pocket_interaction = PocketInteraction(
            hidden, cond_dim, neighbors=neighbors, step_scale=step_scale,
        )
        self.step_scale = float(step_scale)

    def forward(self, hidden, coords, graph, hP, hc):
        n, width = hidden.shape
        coordinate_dtype = torch.float64 if coords.dtype == torch.float64 else torch.float32
        coords = coords.to(dtype=coordinate_dtype)
        chemical_state = torch.cat([
            graph[key] for key in ("atom_state", "charge_state", "hydrogen_state")
        ], dim=-1)
        condition = self.condition(torch.cat([summary_of(hP).reshape(-1), hc.reshape(-1)]))
        contextual = hidden + self.chemistry_embedding(chemical_state) + condition
        q, k, v = self.qkv(contextual).reshape(n, 3, self.num_heads, self.head_dim).unbind(dim=1)
        relative = coords[:, None, :] - coords[None, :, :]
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        radial = torch.exp(-((distance - self.radial_centers.to(distance)) / (10.0 / 15.0)).square())
        edge_features = torch.cat([graph["edge_state"], radial.to(hidden.dtype)], dim=-1)
        logits = torch.einsum("ihd,jhd->ijh", q, k) / math.sqrt(self.head_dim)
        logits = logits + self.edge_attention_bias(edge_features)
        attention = torch.softmax(logits.to(coordinate_dtype), dim=1)
        values = v[None, :, :, :] + self.edge_value(edge_features).reshape(n, n, self.num_heads, self.head_dim)
        attended = (attention[..., None].to(values.dtype) * values).sum(dim=1).reshape(n, width)
        hidden_next = self.attention_norm(hidden + self.attention_out(attended))
        hidden_next = self.feedforward_norm(hidden_next + self.feedforward(hidden_next))
        left, right = contextual[:, None, :], contextual[None, :, :]
        gates = self.coordinate_gate(torch.cat([
            left + right, (left - right).abs(), edge_features,
        ], dim=-1)).to(coordinate_dtype)
        coefficients = (attention * gates).mean(dim=-1, keepdim=True)
        # Bounded directions and gates limit the displacement to step_scale.
        displacement = (coefficients * relative / (1.0 + distance)).sum(dim=1)
        coords_next = coords + self.step_scale * displacement
        return self.pocket_interaction(hidden_next, coords_next, hP)


class ChemistryCoupledDecoder(nn.Module):
    """Predict typed G once, then decode R with ``steps`` geometry attention layers.
    MAP one-hot graph states carry softmax surrogate gradients."""

    def __init__(self, hidden: int, cond_dim: int, steps: int = 3,
                 neighbors: int = 32, step_scale: float = 0.2):
        super().__init__()
        if isinstance(steps, bool) or int(steps) != steps or int(steps) < 1:
            raise ValueError("geometry attention steps must be a positive integer")
        if hidden < 1 or cond_dim < 1:
            raise ValueError("hidden and cond_dim must be positive")
        if not 0 <= step_scale < float("inf"):
            raise ValueError("step_scale must be finite and nonnegative")
        self.steps = int(steps)
        self.graph_condition = nn.Linear(2 * cond_dim, hidden)
        self.graph_pocket_interaction = PocketInteraction(
            hidden, cond_dim, neighbors=neighbors, step_scale=0.0,
        )
        self.stages = nn.ModuleList([
            GraphConditionedGeometryLayer(hidden, cond_dim, step_scale, neighbors)
            for _ in range(self.steps)
        ])

    def predict_graph(self, hidden, coords, hP, hc, chemistry_heads, edge_denoiser, adj_init=None):
        n = int(hidden.shape[0])
        if n < 1 or coords.shape != (n, 3):
            raise ValueError("graph decoding requires nonempty nodes and matching (N, 3) coordinates")
        diagonal = torch.eye(n, device=hidden.device, dtype=torch.bool)
        if adj_init is None:
            adjacency = hidden.new_full((n, n), 0.5).masked_fill(diagonal, 0.0)
        else:
            if adj_init.shape != (n, n):
                raise ValueError("adj_init must have shape (N, N)")
            adjacency = adj_init.to(device=hidden.device, dtype=hidden.dtype).clamp(0.0, 1.0)
            adjacency = ((adjacency + adjacency.T) * 0.5).masked_fill(diagonal, 0.0)
        hidden = hidden + self.graph_condition(torch.cat([summary_of(hP).reshape(-1), hc.reshape(-1)]))
        # Graph-stage pocket attention updates embeddings only.
        hidden, _ = self.graph_pocket_interaction(hidden, coords, hP)
        chemistry = chemistry_heads(hidden, coords)
        t = torch.zeros(1, device=hidden.device, dtype=torch.long)
        presence = edge_denoiser(hidden, adjacency, t, hP, hc).masked_fill(diagonal, 0.0)
        edge_logits = torch.cat([
            torch.zeros_like(presence[..., None]),
            presence[..., None] + F.log_softmax(chemistry["bond_logits"], dim=-1),
        ], dim=-1)
        edge_logits = _masked_edge_logits((edge_logits + edge_logits.transpose(0, 1)) * 0.5, diagonal)
        probabilities = torch.softmax(edge_logits, dim=-1)
        edge_state = _map_straight_through(edge_logits)
        return {
            "graph_node_emb": hidden,
            "atom_logits": chemistry["atom_logits"],
            "charge_logits": chemistry["charge_logits"],
            "hydrogen_logits": chemistry["hydrogen_logits"],
            "bond_logits": edge_logits[..., 1:].masked_fill(diagonal[..., None], 0.0),
            "edge_logits": edge_logits,
            "atom_state": _map_straight_through(chemistry["atom_logits"]),
            "charge_state": _map_straight_through(chemistry["charge_logits"]),
            "hydrogen_state": _map_straight_through(chemistry["hydrogen_logits"]),
            "edge_state": edge_state,
            "adj_soft": probabilities[..., 1:].sum(dim=-1).masked_fill(diagonal, 0.0),
            "adj_ste": edge_state[..., 1:].sum(dim=-1).masked_fill(diagonal, 0.0),
        }

    def forward_geometry(self, hidden, coords, graph, hP, hc):
        """Decode coordinates without changing or re-predicting graph labels."""
        for stage in self.stages:
            hidden, coords = stage(hidden, coords, graph, hP, hc)
        return {"node_emb": hidden, "R_pred": coords}

    def forward(self, hidden, coords, hP, hc, chemistry_heads, edge_denoiser, adj_init=None):
        graph = self.predict_graph(hidden, coords, hP, hc, chemistry_heads, edge_denoiser, adj_init=adj_init)
        geometry = self.forward_geometry(graph["graph_node_emb"], coords, graph, hP, hc)
        return {**graph, **geometry}


GraphThenGeometryDecoder = ChemistryCoupledDecoder
