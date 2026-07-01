from __future__ import annotations

import networkx as nx
import numpy as np
import torch

from mol_state import MolState
from utils.pkt_fit import pkt_fit


def probe_at(pos, atomic_num, feat_dim):
    p = np.asarray(pos, dtype=np.float32).reshape(3,)
    G = nx.Graph()
    G.add_node(0, atomic_num=int(atomic_num))
    R = p.reshape(1, 3)
    node_feat = np.zeros((1, int(feat_dim)), dtype=np.float32)
    return MolState(G=G, R=R, node_feat=node_feat)


def score_sites(sites, pocket, wt, clash_r, probe_z, feat_dim):
    sites_np = np.asarray(sites, dtype=np.float64)
    if sites_np.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if sites_np.ndim == 1:
        sites_np = sites_np.reshape(1, -1)
    if sites_np.shape[-1] != 3:
        raise ValueError(f"sites must be (J,3), got {sites_np.shape}")

    n_sites = sites_np.shape[0]
    scores = np.empty((n_sites,), dtype=np.float64)
    for j in range(n_sites):
        X = probe_at(sites_np[j], atomic_num=probe_z, feat_dim=feat_dim)
        out = pkt_fit(X, pocket, wt=wt, clash_r=clash_r)
        scores[j] = out.tot
    return scores


def pick_anchors(scores, k):
    if isinstance(scores, np.ndarray):
        rank = torch.from_numpy(scores.astype(np.float64))
    else:
        rank = scores.detach().float().view(-1)
    if rank.numel() == 0:
        return torch.zeros((0,), dtype=torch.long)
    k = min(int(k), int(rank.shape[0]))
    return torch.topk(rank, k=k, dim=0).indices


def top_anchors(sites, pocket, k, score_kw):
    if isinstance(sites, torch.Tensor):
        sites_np = sites.detach().float().cpu().numpy()
    else:
        sites_np = np.asarray(sites, dtype=np.float64)
    scores = score_sites(sites_np, pocket, **score_kw)
    idx = pick_anchors(scores, k)
    return torch.from_numpy(sites_np[idx.numpy()].astype(np.float32))
