from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import coo_matrix
from utils.chemistry import ATOMIC_NUMBERS, EXPLICIT_HS, FORMAL_CHARGES, atom_targets, bond_targets


def match_target(out, target, cfg):
    """Target indices in output-slot order."""
    R = out["R_pred"]
    n = target.num_nodes()
    if R.shape != (n, 3) or n == 0:
        raise ValueError("Supervised output must contain exactly the target atom count")
    transport, _ = coordinate_transport(out, target, cfg)
    return transport.argmax(-1)


def chemical_loss(out, target, cfg, *, target_embedding=None, order=None):
    """Typed structure supervision and valence regularization, before w_chem.
    Coordinate MSE is reported separately and excluded from this loss."""
    R = out["R_pred"]
    n = target.num_nodes()
    if R.shape != (n, 3) or n == 0:
        raise ValueError("Supervised output must contain exactly the target atom count")
    order = match_target(out, target, cfg) if order is None else order
    labels = {k: v[order] for k, v in atom_targets(target, R.device).items()}
    bonds = bond_targets(target, R.device)[order][:, order]
    ref = torch.as_tensor(target.R, device=R.device, dtype=R.dtype)[order]
    feat = torch.as_tensor(target.node_feat, device=R.device, dtype=R.dtype)[order]
    zero = R.sum() * 0
    parts = {
        "atom": F.cross_entropy(out["atom_logits"], labels["atom"]),
        "charge": F.cross_entropy(out["charge_logits"], labels["charge"]),
        "hydrogen": F.cross_entropy(out["hydrogen_logits"], labels["hydrogen"]),
        "geometry": F.mse_loss(R, ref),
        "feature": F.mse_loss(out["node_feat_pred"], feat),
    }
    tri = torch.triu(torch.ones((n, n), dtype=torch.bool, device=R.device), diagonal=1)
    adjacency_logits = torch.logsumexp(out["edge_logits"][..., 1:], dim=-1) - out["edge_logits"][..., 0]
    parts["adjacency"] = F.binary_cross_entropy_with_logits(
        adjacency_logits[tri], (bonds[tri] > 0).to(R.dtype)
    ) if n > 1 else zero
    bonded = tri & (bonds > 0)
    parts["bond"] = F.cross_entropy(out["bond_logits"][bonded], bonds[bonded]-1) if bonded.any() else zero
    parts["bond_length"] = F.mse_loss(
        torch.cdist(R, R, compute_mode="donot_use_mm_for_euclid_dist")[bonded],
        torch.cdist(ref, ref, compute_mode="donot_use_mm_for_euclid_dist")[bonded],
    ) if bonded.any() else zero
    count = out["count_logits"].reshape(1, -1)
    if n > count.shape[-1]:
        raise ValueError("Target count exceeds model max_nodes")
    parts["count"] = F.cross_entropy(count, torch.tensor([n-1], device=R.device))
    parts["embedding"] = zero if target_embedding is None else torch.linalg.vector_norm(
        out["node_emb"].mean(0)-target_embedding.detach().mean(0)
    ) / out["node_emb"].shape[-1]**0.5
    categorical = cfg.w_adjacency*parts["adjacency"] + sum(getattr(cfg, "w_"+k)*parts[k] for k in (
        "atom", "charge", "hydrogen", "bond", "count"
    ))
    parts["categorical"] = categorical
    reference_valence = R.new_tensor([0., 1., 2., 3., 1.])[bonds].sum(-1)
    reference_valence = reference_valence + R.new_tensor(EXPLICIT_HS)[labels["hydrogen"]]
    parts["valence"] = valence_penalty(out, labels=labels, reference_valence=reference_valence) if cfg.w_valence > 0 else zero
    parts["total"] = (categorical + cfg.w_bond_length*parts["bond_length"]
                      + cfg.w_feature*parts["feature"] + cfg.w_embedding*parts["embedding"]
                      + cfg.w_valence*parts["valence"])
    return parts, order


def _target_coordinates_and_elements(target, reference):
    if isinstance(target, dict):
        elements = target["atom_state"] if "atom_state" in target else target["atom_logits"].softmax(-1)
        return target["R_pred"], elements
    coordinates = torch.as_tensor(target.R, device=reference.device, dtype=reference.dtype)
    labels = atom_targets(target, reference.device)["atom"]
    probabilities = F.one_hot(labels, num_classes=len(ATOMIC_NUMBERS)).to(dtype=reference.dtype)
    return coordinates, probabilities


def coordinate_transport(out, target, cfg):
    """Hungarian matching for equal counts; balanced transport otherwise.
    Row/column masses are 1/N and 1/M, covering every atom. Matching is detached;
    coordinate gradients flow through all pairs with positive mass."""
    R = out["R_pred"]
    S, target_elements = _target_coordinates_and_elements(target, R)
    n, m = len(R), len(S)
    if n == 0 or m == 0 or R.shape != (n, 3) or S.shape != (m, 3):
        raise ValueError("State distance requires nonempty coordinate sets of shape (N, 3)")
    if R.device != S.device:
        raise ValueError("Both state coordinates must be on the same device")
    if not bool(torch.isfinite(R).all()) or not bool(torch.isfinite(S).all()):
        raise ValueError("State coordinates must be finite")
    with torch.no_grad():
        source_elements = out["atom_state"] if "atom_state" in out else out["atom_logits"].softmax(-1)
        geometric_cost = torch.cdist(R, S, compute_mode="donot_use_mm_for_euclid_dist").square() / 3.
        chemical_cost = .5 * (source_elements[:, None, :] - target_elements[None, :, :]).square().sum(-1)
        cost = cfg.match_position_weight * geometric_cost + cfg.match_atom_weight * chemical_cost
        if not bool(torch.isfinite(cost).all()):
            raise ValueError("Nonfinite state-distance correspondence cost")
        cpu_cost = cost.detach().to(device="cpu", dtype=torch.float64).numpy()
        if n == m:
            rows, columns = linear_sum_assignment(cpu_cost)
            matrix = np.zeros((n, m), dtype=np.float64)
            matrix[rows, columns] = 1. / n
        else:
            variables = np.arange(n * m)
            constraint_rows = np.concatenate([np.repeat(np.arange(n), m),
                                              n + np.tile(np.arange(m), n)])
            constraint_columns = np.tile(variables, 2)
            constraints = coo_matrix((np.ones(2 * n * m), (constraint_rows, constraint_columns)),
                                     shape=(n + m, n * m)).tocsr()
            masses = np.concatenate([np.full(n, 1. / n), np.full(m, 1. / m)])
            solution = linprog(cpu_cost.reshape(-1), A_eq=constraints, b_eq=masses,
                               bounds=(0., None), method="highs")
            if not solution.success or solution.x is None:
                raise RuntimeError(f"Atom correspondence transport failed: {solution.message}")
            matrix = solution.x.reshape(n, m)
            if (not np.isfinite(matrix).all() or matrix.min() < -1e-8
                    or not np.allclose(matrix.sum(1), 1. / n, atol=1e-7, rtol=0)
                    or not np.allclose(matrix.sum(0), 1. / m, atol=1e-7, rtol=0)):
                raise RuntimeError("Atom correspondence transport returned invalid marginals")
        transport = torch.as_tensor(matrix, device=R.device, dtype=R.dtype)
    return transport, S


def paper_state_distance(out, target, cfg, graph_encoder):
    """Graph-embedding Euclidean distance plus lambda_geom-weighted coordinate MSE.
    Molecular targets stop gradients; both predicted states retain gradients."""
    transport, S = coordinate_transport(out, target, cfg)
    first = graph_encoder.from_prediction(out)
    if isinstance(target, dict):
        second = graph_encoder.from_prediction(target)
    else:
        with torch.no_grad():
            second = graph_encoder.from_state(target, device=out["R_pred"].device)
    graph = torch.linalg.vector_norm(first - second)
    pair_squared = (out["R_pred"][:, None, :] - S[None, :, :]).square().mean(-1)
    geometry = (transport * pair_squared).sum()
    return {"graph": graph, "geometry": geometry, "total": graph + cfg.lambda_geom * geometry}, transport


def aligned_edge_target(target, order, dtype):
    labels = bond_targets(target, order.device)[order][:, order]
    return (labels > 0).to(dtype)


def valence_penalty(out, *, labels=None, reference_valence=None):
    """Pair-normalized SmoothL1 excess with fixed valence capacity.
    Reference valence sets a capacity floor; aromatic bonds count as one."""
    adj = out["adj_soft"]
    eye = torch.eye(adj.shape[0], device=adj.device, dtype=torch.bool)
    adj = adj.masked_fill(eye, 0)
    order = out["bond_logits"].softmax(-1) @ adj.new_tensor([1., 2., 3., 1.])
    hydrogen = out["hydrogen_logits"].softmax(-1) @ adj.new_tensor([0., 1., 2., 3., 4.])
    valence = (adj*order).sum(-1) + hydrogen
    neutral_caps = {1:1, 5:3, 6:4, 7:3, 8:2, 9:1, 14:4, 15:5, 16:6, 17:1, 35:1, 53:1}
    caps = []
    for z in ATOMIC_NUMBERS:
        row = []
        for charge in FORMAL_CHARGES:
            cap = neutral_caps[z]
            if z == 7 and charge == 1: cap = 4
            if z == 8 and charge == -1: cap = 1
            if z == 8 and charge == 1: cap = 3
            if z == 5 and charge == -1: cap = 4
            row.append(cap)
        caps.append(row)
    limits = adj.new_tensor(caps)
    pair_scale = max(adj.shape[0] - 1, 1)
    if labels is not None:
        capacity = limits[labels["atom"], labels["charge"]]
        if reference_valence is not None:
            capacity = torch.maximum(capacity, reference_valence.detach())
        excess = torch.relu(valence - capacity)
        return F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=1., reduction="none").mean() / pair_scale
    joint = (out["atom_logits"].softmax(-1).unsqueeze(-1)
             * out["charge_logits"].softmax(-1).unsqueeze(-2)).detach()
    excess = torch.relu(valence[:, None, None] - limits)
    penalty = F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=1., reduction="none")
    return (joint * penalty).sum((-1, -2)).mean() / pair_scale


def basin_consistency_loss(a, b, cfg, *, graph_encoder):
    """Joint state distance between two group predictions."""
    return paper_state_distance(a, b, cfg, graph_encoder)[0]["total"]


def seed_neighborhood_loss(out, target, cfg, *, graph_encoder):
    """Joint state distance with balanced transport for unequal atom counts."""
    return paper_state_distance(out, target, cfg, graph_encoder)[0]["total"]
