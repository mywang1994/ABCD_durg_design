from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import HIDDEN
from model.pocket_context import summary_of


def _sym_no_diag(adj):
    sym = (adj + adj.T) * 0.5
    sym = sym.clone()
    sym.fill_diagonal_(0.0)
    return sym


def _sample_symmetric(probabilities):
    """Sample each undirected pair once to keep edges binary."""
    upper = torch.triu((torch.rand_like(probabilities) < probabilities).to(probabilities.dtype), diagonal=1)
    return upper + upper.T


def adj_ste(adj_soft: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    adj = _sym_no_diag(adj_soft)
    adj_hard = (adj >= float(threshold)).to(dtype=adj.dtype)
    return adj_hard - adj.detach() + adj


def time_emb(t, dim):
    half = dim // 2
    t = t.float().unsqueeze(-1)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=t.dtype) / max(half - 1, 1))
    args = t * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class EdgeNet(nn.Module):
    def __init__(self, node_dim, cond_dim, time_dim=64, hidden=HIDDEN):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.global_proj = nn.Linear(2 * cond_dim, hidden)
        # Shared node context couples edge predictions through neighbors and degrees.
        self.neighbor_proj = nn.Linear(node_dim, node_dim, bias=False)
        self.adjacency_context = nn.Sequential(
            nn.Linear(node_dim + 1, node_dim), nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        pair_in_dim = 2 * node_dim + hidden + 1
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_emb, adj_t, t, hP, hc):
        n = node_emb.shape[0]
        t_flat = t.view(-1).long()
        time_sin = time_emb(t_flat, self.time_dim).to(dtype=node_emb.dtype)
        time_h = self.time_mlp(time_sin).view(1, 1, -1).expand(n, n, -1)
        global_h = self.global_proj(torch.cat([summary_of(hP).reshape(-1), hc.reshape(-1)], dim=-1)).view(1, 1, -1).expand(n, n, -1)
        cond = time_h + global_h
        adjacency = _sym_no_diag(adj_t.clamp(0.0, 1.0))
        denominator = max(n - 1, 1)
        neighbors = adjacency @ self.neighbor_proj(node_emb) / denominator
        degree = adjacency.sum(dim=-1, keepdim=True) / denominator
        contextual_nodes = node_emb + self.adjacency_context(torch.cat([neighbors, degree], dim=-1))
        left = contextual_nodes.unsqueeze(1).expand(n, n, -1)
        right = contextual_nodes.unsqueeze(0).expand(n, n, -1)
        adj_in = adjacency.unsqueeze(-1)
        pair = torch.cat([left, right, cond, adj_in], dim=-1)
        logits = self.pair_mlp(pair).squeeze(-1)
        logits = (logits + logits.T) * 0.5
        logits = logits.clone()
        logits.fill_diagonal_(-float("inf"))
        return logits


def mask_fwd(x0, t, T, mask_eps=0.04):
    if T <= 0:
        raise ValueError("T must be >= 1")
    n = x0.shape[0]
    device, dtype = x0.device, x0.dtype
    p_mask = float(t + 1) / float(T) * (1.0 - mask_eps)
    p_mask = min(max(p_mask, 0.0), 1.0)
    noise = _sample_symmetric(torch.full_like(x0, 0.5))
    u = torch.rand(n, n, device=device, dtype=dtype)
    mask = (u < p_mask).to(dtype)
    upper = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
    m = mask * upper.float()
    m = m + m.T
    x_t = x0 * (1.0 - m) + noise * m
    return _sym_no_diag(x_t)


def _denoise_adj(denoiser, node_emb, hP, hc, T, device, dtype, adj_init, *, sample):
    """Shared reverse schedule; only the discrete sampling policy differs."""
    n = int(node_emb.shape[0])
    if T < 1:
        raise ValueError("T must be >= 1")
    if adj_init is not None:
        adj = _sym_no_diag(adj_init.to(device=device, dtype=dtype).clone())
    else:
        adj = torch.full((n, n), 0.5, device=device, dtype=dtype)
        if sample:
            adj = _sample_symmetric(adj)
        else:
            adj.fill_diagonal_(0.0)
    for step in range(T - 1, -1, -1):
        t = torch.tensor([step], device=device, dtype=torch.long)
        logits = denoiser(node_emb, adj, t, hP, hc)
        p_clean = _sym_no_diag(torch.sigmoid(logits))
        blend = (T - step) / float(T)
        adj = _sym_no_diag(adj * (1.0 - blend) + p_clean * blend)
        if sample:
            adj = _sample_symmetric(adj) if step > 0 else (adj >= 0.5).to(dtype)
            adj = _sym_no_diag(adj)
    return adj


def predict_adj(denoiser, node_emb, hP, hc, T, device, dtype=torch.float32, adj_init=None):
    return _denoise_adj(denoiser, node_emb, hP, hc, T, device, dtype, adj_init, sample=False)


def sample_adj(denoiser, node_emb, hP, hc, T, device, dtype=torch.float32, adj_init=None):
    return _denoise_adj(denoiser, node_emb, hP, hc, T, device, dtype, adj_init, sample=True)


def edge_loss(denoiser, x0, node_emb, hP, hc, T):
    device = x0.device
    if int(T) < 1:
        raise ValueError("T must be >= 1")
    if x0.shape[0] <= 1:
        return node_emb.sum() * 0.0
    t = int(torch.randint(0, T, (1,), device=device).item())
    target = _sym_no_diag(x0.clamp(0.0, 1.0))
    with torch.no_grad():
        noisy = mask_fwd(target, t, T)
    t_tensor = torch.tensor([t], device=device, dtype=torch.long)
    logits = denoiser(node_emb, noisy, t_tensor, hP, hc)
    upper = torch.triu(torch.ones_like(target, dtype=torch.bool), diagonal=1)
    return F.binary_cross_entropy_with_logits(logits[upper], target[upper])
