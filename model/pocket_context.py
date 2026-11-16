
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn


@dataclass(frozen=True)
class PocketContext:
    positions: torch.Tensor
    node_features: torch.Tensor
    summary: torch.Tensor

    def to(self, *args, **kwargs) -> "PocketContext":
        return PocketContext(*(value.to(*args, **kwargs) for value in
                               (self.positions, self.node_features, self.summary)))


def summary_of(pocket: PocketContext | torch.Tensor) -> torch.Tensor:
    return pocket.summary if isinstance(pocket, PocketContext) else pocket


def with_summary(pocket: PocketContext | torch.Tensor, summary: torch.Tensor):
    return replace(pocket, summary=summary) if isinstance(pocket, PocketContext) else summary


class PocketInteraction(nn.Module):
    """Attend to nearest pocket atoms, including boundary ties; keep the pocket fixed.
    Ligand displacement is bounded by step_scale angstroms per step.
    Tensor summaries return the inputs unchanged."""

    def __init__(self, hidden: int, cond_dim: int, neighbors: int = 32, step_scale: float = .2):
        super().__init__()
        if (int(hidden) != hidden or int(cond_dim) != cond_dim or hidden < 1 or cond_dim < 1
                or int(neighbors) != neighbors or neighbors < 1):
            raise ValueError("hidden, cond_dim and neighbors must be positive integers")
        if not 0. <= step_scale < float("inf"):
            raise ValueError("step_scale must be finite and nonnegative")
        self.hidden = int(hidden)
        self.cond_dim = int(cond_dim)
        self.neighbors = int(neighbors)
        self.step_scale = float(step_scale)
        self.register_buffer("rbf_centers", torch.linspace(0., 20., 16))
        self.pocket_projection = nn.Linear(cond_dim, hidden)
        self.message = nn.Sequential(nn.Linear(2 * hidden + 16, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU())
        self.attention = nn.Linear(hidden, 1)
        self.node_update = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                         nn.Linear(hidden, hidden))
        self.coordinate_weight = nn.Linear(hidden, 1)

    def forward(self, hidden: torch.Tensor, R: torch.Tensor,
                hP: PocketContext | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(hP, torch.Tensor):
            return hidden, R
        if not isinstance(hP, PocketContext):
            raise TypeError("hP must be PocketContext or a tensor summary")
        n = R.shape[0]
        if R.shape != (n, 3) or n == 0 or hidden.shape != (n, self.hidden):
            raise ValueError("Ligand coordinates/features must be nonempty matching (N,3)/(N,hidden)")
        m = hP.positions.shape[0]
        if (hP.positions.shape != (m, 3) or m == 0
                or hP.node_features.shape != (m, self.cond_dim)):
            raise ValueError("PocketContext must contain matching nonempty positions and node features")
        if R.device != hidden.device or hP.positions.device != R.device or hP.node_features.device != R.device:
            raise ValueError("Ligand and pocket tensors must be on the same device")
        if not R.is_floating_point() or not hP.positions.is_floating_point():
            raise ValueError("Ligand and pocket coordinates must have floating-point dtype")
        # Keep coordinate arithmetic at fp32 or higher under mixed precision.
        coordinate_dtype = torch.float64 if R.dtype == torch.float64 else torch.float32
        coordinates = R.to(dtype=coordinate_dtype)
        positions = hP.positions.to(dtype=coordinate_dtype)
        with torch.no_grad():
            distances = torch.cdist(coordinates, positions, compute_mode="donot_use_mm_for_euclid_dist")
            boundary = distances.topk(min(self.neighbors, m), dim=1, largest=False).values[:, -1:]
            ligand_index, pocket_index = (distances <= boundary).nonzero(as_tuple=True)
        relative = positions[pocket_index] - coordinates[ligand_index]
        squared_distance = relative.square().sum(dim=-1)
        distance = (squared_distance + 1e-12).sqrt()
        radial = torch.exp(-((distance[:, None] - self.rbf_centers.to(distance)) / (20. / 15.)) ** 2)
        pocket_hidden = self.pocket_projection(hP.node_features.to(dtype=hidden.dtype))
        messages = self.message(torch.cat((hidden[ligand_index], pocket_hidden[pocket_index],
                                           radial.to(dtype=hidden.dtype)), dim=-1))
        # fp32 normalization avoids unstable scatter softmax in fp16/bfloat16.
        logits = self.attention(messages).squeeze(-1).to(dtype=coordinate_dtype) - distance / 5.
        maxima = logits.new_full((n,), -torch.inf).scatter_reduce(0, ligand_index, logits, reduce="amax")
        unnormalized = (logits - maxima[ligand_index]).exp()
        totals = logits.new_zeros(n).index_add(0, ligand_index, unnormalized)
        attention = unnormalized / totals[ligand_index].clamp_min(torch.finfo(logits.dtype).tiny)
        pooled = messages.new_zeros((n, self.hidden)).index_add(
            0, ligand_index, messages * attention[:, None].to(dtype=messages.dtype))
        updated_hidden = hidden + self.node_update(pooled)
        direction = relative / (1. + squared_distance[:, None]).sqrt()
        gates = self.coordinate_weight(messages).tanh().to(dtype=coordinate_dtype)
        displacement = coordinates.new_zeros((n, 3)).index_add(
            0, ligand_index, attention[:, None] * gates * direction)
        return updated_hidden, coordinates + self.step_scale * displacement
