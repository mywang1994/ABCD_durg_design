
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from utils.chemistry import ATOMIC_NUMBERS, EXPLICIT_HS, FORMAL_CHARGES, atom_targets, bond_targets


class GraphStateEncoder(nn.Module):
    """Frozen MPNN with composition, mean typed degree and log count.
    The representation is not an injective graph identity test."""

    def __init__(self, hidden: int = 64, layers: int = 2):
        super().__init__()
        if hidden < 1 or layers < 1:
            raise ValueError("Graph encoder hidden and layers must be positive")
        self.hidden = int(hidden)
        self.chemical_dim = len(ATOMIC_NUMBERS) + len(FORMAL_CHARGES) + len(EXPLICIT_HS)
        self.node_in = nn.Linear(self.chemical_dim, self.hidden)
        self.neighbor_projections = nn.ModuleList(nn.Linear(self.hidden, self.hidden, bias=False)
                                                  for _ in range(layers))
        self.edge_projections = nn.ModuleList(nn.Linear(4, self.hidden, bias=False) for _ in range(layers))
        self.updates = nn.ModuleList(nn.Sequential(nn.Linear(2 * self.hidden, self.hidden), nn.SiLU(),
                                                  nn.Linear(self.hidden, self.hidden)) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(self.hidden) for _ in range(layers))
        self.readout = nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.SiLU(),
                                     nn.Linear(self.hidden, self.hidden))
        self.requires_grad_(False)

    @staticmethod
    def prediction_probabilities(out):
        names = ("atom", "charge", "hydrogen")
        nodes = torch.cat([out[name + "_state"] if name + "_state" in out else out[name + "_logits"].softmax(-1)
                           for name in names], dim=-1)
        if "edge_state" in out:
            edges = out["edge_state"]
        elif "edge_logits" in out:
            edges = out["edge_logits"].softmax(-1)
        else:
            # Combine adjacency and conditional bond probabilities into five classes.
            adjacency = out["adj_soft"].clamp(0., 1.)
            conditional = out["bond_logits"].softmax(-1)
            edges = torch.cat([(1. - adjacency)[..., None], adjacency[..., None] * conditional], dim=-1)
        return nodes, edges

    def forward(self, nodes, edges):
        n = int(nodes.shape[0])
        if n < 1 or nodes.shape != (n, self.chemical_dim) or edges.shape != (n, n, 5):
            raise ValueError("Graph representation requires nonempty typed node/edge probabilities")
        if not bool(torch.isfinite(nodes).all()) or not bool(torch.isfinite(edges).all()):
            raise ValueError("Graph representation requires finite probabilities")
        dtype = next(self.parameters()).dtype
        nodes = nodes.to(dtype=dtype)
        diagonal = torch.eye(n, dtype=torch.bool, device=nodes.device)
        typed_bonds = ((edges[..., 1:] + edges[..., 1:].transpose(0, 1)) * .5).to(dtype=dtype)
        typed_bonds = typed_bonds.masked_fill(diagonal[..., None], 0.)
        adjacency = typed_bonds.sum(-1)
        hidden = self.node_in(nodes)
        denominator = max(n - 1, 1)
        for neighbor, edge, update, norm in zip(self.neighbor_projections, self.edge_projections,
                                              self.updates, self.norms):
            # Sum before the bias-free projection to avoid an N x N x hidden tensor.
            messages = adjacency @ neighbor(hidden) + edge(typed_bonds.sum(dim=1))
            hidden = norm(hidden + update(torch.cat([hidden, messages / denominator], dim=-1)))
        features = self.readout(hidden.mean(0))
        statistics = torch.cat([nodes.mean(0), typed_bonds.sum((0, 1)) / n,
                                nodes.new_tensor([float(n)]).log1p()])
        return torch.cat([features, statistics])

    def from_prediction(self, out):
        return self(*self.prediction_probabilities(out))

    def from_state(self, state, *, device=None, dtype=None):
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        labels = atom_targets(state, device)
        nodes = torch.cat([F.one_hot(labels[key], num_classes=len(vocabulary)).to(dtype=dtype)
                           for key, vocabulary in (("atom", ATOMIC_NUMBERS), ("charge", FORMAL_CHARGES),
                                                   ("hydrogen", EXPLICIT_HS))], dim=-1)
        edges = F.one_hot(bond_targets(state, device), num_classes=5).to(dtype=dtype)
        return self(nodes, edges)
