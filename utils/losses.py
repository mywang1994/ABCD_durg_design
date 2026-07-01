from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from mol_state import MolState
from utils.pkt_fit import PktCtx, PktWt, pkt_fit_train

from dataset.sampling import TrajChain
from dataset.sampling import MapGroup

'''
Losses for C_theta: match pool teacher locally +X* globally 
Optional: edit limit, pkt fit, edge BCE, basin group; seed handled in train.py.

'''






# Graph adjacency


def adjacency_from_molstate(X: MolState, n: int, device, dtype) -> torch.Tensor:
    adj = torch.zeros((n, n), device=device, dtype=dtype)
    nn = min(n, X.num_nodes())
    for u, v in X.G.edges():
        u, v = int(u), int(v)
        if u < nn and v < nn:
            adj[u, v] = 1.0
            adj[v, u] = 1.0
    return adj


# Geometry


def _coord_mse(Rp, Rr) -> torch.Tensor:
    n = min(int(Rp.shape[0]), int(Rr.shape[0]))
    if n == 0:
        return torch.tensor(0.0, device=Rp.device, dtype=torch.float32)
    return torch.mean((Rp[:n] - Rr[:n]) ** 2)


def _kabsch_sq(P, Q) -> torch.Tensor:
    n = min(int(P.shape[0]), int(Q.shape[0]))
    if n == 0:
        return torch.tensor(0.0, device=P.device, dtype=P.dtype)
    P = P[:n]
    Q = Q[:n]
    P0 = P - P.mean(dim=0, keepdim=True)
    Q0 = Q - Q.mean(dim=0, keepdim=True)
    if n < 2:
        return torch.mean((P0 - Q0) ** 2)
    H = P0.T @ Q0
    U, _, Vh = torch.linalg.svd(H)
    R = U @ Vh
    if torch.det(R) < 0:
        Vh_adj = Vh.clone()
        Vh_adj[-1, :] *= -1.0
        R = U @ Vh_adj
    P_al = P0 @ R
    return torch.mean((P_al - Q0) ** 2)


def _geom_dist(Rp, Rr, *, mode: str = "mse", use_kabsch: bool = False) -> torch.Tensor:
    if use_kabsch or mode == "kabsch":
        return _kabsch_sq(Rp, Rr)
    return _coord_mse(Rp, Rr)


# Graph embedding CG


def _node_emb(
    model: torch.nn.Module,
    X: MolState,
    *,
    hP: torch.Tensor,
    hc: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    import numpy as np

    node_feat = torch.from_numpy(np.asarray(X.node_feat, dtype=np.float32)).to(device=device)
    R = torch.from_numpy(np.asarray(X.R, dtype=np.float32)).to(device=device)
    n = int(node_feat.shape[0])
    hP2 = hP.to(device=device, dtype=torch.float32).reshape(1, -1).expand(n, -1)
    hc2 = hc.to(device=device, dtype=torch.float32).reshape(1, -1).expand(n, -1)
    cat_in = torch.cat([node_feat, hP2, hc2], dim=-1)
    nodes_in = model.node_proj(cat_in)
    g0 = X.G if not model.core.use_radius_edges else None
    out = model.core.forward_from_node_emb(nodes_in, R, hP, hc, device, mol_graph=g0)
    return out["node_emb"]


def _gr_emb_dist(
    model: torch.nn.Module,
    pred_node_emb: torch.Tensor,
    X_ref: MolState,
    *,
    hP: torch.Tensor,
    hc: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """d_G: MSE between mean predicted emb and mean reference emb (reference path no grad)."""
    with torch.no_grad():
        ref_emb = _node_emb(model, X_ref, hP=hP, hc=hc, device=device)
        ref_mu = ref_emb.mean(dim=0)
    pred_mu = pred_node_emb.mean(dim=0)
    return F.mse_loss(pred_mu, ref_mu)


def _gr_adj_dist(
    adj_pred: torch.Tensor,
    X_ref: MolState,
    *,
    device: torch.device,
) -> torch.Tensor:
    """d_G (topology): BCE between predicted soft adjacency and reference graph."""
    n = min(int(adj_pred.shape[0]), X_ref.num_nodes())
    if n <= 1:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    adj_ref = adjacency_from_molstate(X_ref, n, device, adj_pred.dtype)
    triu = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
    logits = torch.logit(adj_pred[triu].clamp(1e-4, 1.0 - 1e-4))
    return F.binary_cross_entropy_with_logits(logits, adj_ref[triu])


# Public losses (Eq. 6–8, 16–17)


@dataclass(frozen=True)
class LossParts:
    dg: torch.Tensor
    dg_emb: torch.Tensor
    dg_adj: torch.Tensor
    dr: torch.Tensor
    tot: torch.Tensor


def d_state(
    model: torch.nn.Module,
    Xn: MolState,
    R_pred: torch.Tensor,
    X_ref: MolState,
    *,
    hP: torch.Tensor,
    hc: torch.Tensor,
    w_g: float,
    w_gr: float,
    device: torch.device,
    pred_node_emb: torch.Tensor | None = None,
    adj_pred: torch.Tensor | None = None,
    coord_loss: str = "mse",
    use_kabsch: bool = False,
) -> LossParts:
    """d_state ≈ w_gr·d_G + w_g·d_R; pass pred_emb to skip extra forward."""
    R_ref = torch.from_numpy(X_ref.R).to(device=device, dtype=torch.float32)
    z = torch.tensor(0.0, device=device, dtype=torch.float32)
    dg_emb = z
    dg_adj = z
    if w_gr > 0:
        emb = pred_node_emb
        if emb is None:
            emb = _node_emb(model, Xn, hP=hP, hc=hc, device=device)
        dg_emb = _gr_emb_dist(model, emb, X_ref, hP=hP, hc=hc, device=device)
        if adj_pred is not None and adj_pred.numel() > 0:
            dg_adj = _gr_adj_dist(adj_pred, X_ref, device=device)
        dg = dg_emb + dg_adj
    else:
        dg = z
    dr = _geom_dist(R_pred, R_ref, mode=coord_loss, use_kabsch=use_kabsch)
    tot = float(w_gr) * dg + float(w_g) * dr
    return LossParts(dg=dg, dg_emb=dg_emb, dg_adj=dg_adj, dr=dr, tot=tot)


def loss_local(
    model: torch.nn.Module,
    Xn: MolState,
    X_teacher: MolState,
    *,
    hP: torch.Tensor,
    hc: torch.Tensor,
    w_g: float,
    w_gr: float,
    device: torch.device,
    step_scale: float = 0.1,
    step_out: dict[str, torch.Tensor] | None = None,
    coord_loss: str = "mse",
    use_kabsch: bool = False,
) -> torch.Tensor:
    """Eq. 7: d_state(C_θ(X_n|·), X*_{n+1}) with joint G+R step."""
    if step_out is None:
        step_out = model.forward_step_trainable(Xn, hP=hP, hc=hc, device=device)

    R_pred = step_out["R_pred"]
    node_emb = step_out.get("node_emb")
    adj_soft = step_out.get("adj_soft")

    parts = d_state(
        model,
        Xn,
        R_pred,
        X_teacher,
        hP=hP,
        hc=hc,
        w_g=w_g,
        w_gr=w_gr,
        device=device,
        pred_node_emb=node_emb,
        adj_pred=adj_soft,
        coord_loss=coord_loss,
        use_kabsch=use_kabsch,
    )
    return parts.tot


def loss_global(
    model: torch.nn.Module,
    Xn: MolState,
    X_star: MolState,
    *,
    hP: torch.Tensor,
    hc: torch.Tensor,
    w_g: float,
    w_gr: float,
    device: torch.device,
    step_scale: float = 0.1,
    step_out: dict[str, torch.Tensor] | None = None,
    coord_loss: str = "mse",
    use_kabsch: bool = False,
) -> torch.Tensor:
    if step_out is None:
        step_out = model.forward_step_trainable(Xn, hP=hP, hc=hc, device=device)

    R_pred = step_out["R_pred"]
    node_emb = step_out.get("node_emb")
    adj_soft = step_out.get("adj_soft")

    parts = d_state(
        model,
        Xn,
        R_pred,
        X_star,
        hP=hP,
        hc=hc,
        w_g=w_g,
        w_gr=w_gr,
        device=device,
        pred_node_emb=node_emb,
        adj_pred=adj_soft,
        coord_loss=coord_loss,
        use_kabsch=use_kabsch,
    )
    return parts.tot


def loss_seed(
    X0: MolState,
    X_seed_star: MolState,
    *,
    w_g: float = 1.0,
    w_f: float = 0.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    dev = device or torch.device("cpu")
    R0 = torch.from_numpy(X0.R).to(device=dev, dtype=torch.float32)
    Rs = torch.from_numpy(X_seed_star.R).to(device=dev, dtype=torch.float32)
    loss = float(w_g) * _coord_mse(R0, Rs)
    if w_f > 0:
        f0 = torch.from_numpy(X0.node_feat).to(device=dev, dtype=torch.float32).mean(dim=0)
        fs = torch.from_numpy(X_seed_star.node_feat).to(device=dev, dtype=torch.float32).mean(dim=0)
        loss = loss + float(w_f) * F.mse_loss(f0, fs)
    return loss


def loss_init(
    X0: MolState,
    X_star: MolState,
    *,
    w_g: float = 1.0,
    w_f: float = 0.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    return loss_seed(X0, X_star, w_g=w_g, w_f=w_f, device=device)


def tensor_alignment_to_molstate(
    R_pred: torch.Tensor,
    node_feat_pred: torch.Tensor,
    X_ref: MolState,
    *,
    w_g: float,
    w_f: float,
    device: torch.device,
    coord_loss: str = "mse",
    use_kabsch: bool = False,
) -> torch.Tensor:
    #like loss_seed 
    R_ref = torch.from_numpy(np.asarray(X_ref.R, dtype=np.float32)).to(device=device, dtype=R_pred.dtype)
    n = min(int(R_pred.shape[0]), int(R_ref.shape[0]))
    if n == 0:
        return torch.tensor(0.0, device=device, dtype=R_pred.dtype)
    Rp = R_pred[:n]
    Rr = R_ref[:n]
    if use_kabsch or coord_loss == "kabsch":
        lc = _kabsch_sq(Rp, Rr)
    else:
        lc = F.mse_loss(Rp, Rr)
    loss = float(w_g) * lc
    if w_f > 0 and node_feat_pred is not None:
        fp = node_feat_pred[:n].mean(dim=0)
        fr = torch.from_numpy(np.asarray(X_ref.node_feat, dtype=np.float32)).to(device=device, dtype=R_pred.dtype)[:n].mean(
            dim=0
        )
        loss = loss + float(w_f) * F.mse_loss(fp, fr)
    return loss


def _edge_thr(m) -> float:
    return float(getattr(m, "edge_threshold", 0.5))


def _edit_pen(st, Xn, d_edit, dev) -> torch.Tensor:
    adj = st.get("adj_soft")
    if adj is None or adj.numel() == 0:
        return torch.tensor(0.0, device=dev, dtype=torch.float32)
    n = min(int(adj.shape[0]), Xn.num_nodes())
    if n <= 1:
        return torch.tensor(0.0, device=dev, dtype=torch.float32)
    adj0 = adjacency_from_molstate(Xn, n, dev, adj.dtype)
    tri = torch.triu(torch.ones(n, n, device=dev, dtype=torch.bool), diagonal=1)
    n_edit = torch.sum(torch.abs(adj[tri] - adj0[tri]))
    return torch.relu(n_edit - float(d_edit)) ** 2


def _should_detach(i, n, bptt) -> bool:
    if int(bptt) < 0:
        return False
    return i < n - int(bptt)


def forward_step_with_carry(
    model: torch.nn.Module,
    X: MolState,
    *,
    hP: torch.Tensor,
    hc: torch.Tensor,
    device: torch.device,
    step_idx: int,
    R_carry: torch.Tensor | None = None,
    feat_carry: torch.Tensor | None = None,
    adj_carry: torch.Tensor | None = None,
    use_carry: bool = True,
) -> dict:
    use_tensor_carry = bool(use_carry) and int(step_idx) > 0
    use_adj_carry = use_tensor_carry and adj_carry is not None and adj_carry.numel() > 0
    return model.forward_step_trainable(
        X,
        hP=hP,
        hc=hc,
        device=device,
        R_in=R_carry if use_tensor_carry else None,
        node_feat_in=feat_carry if use_tensor_carry else None,
        adj_carry=adj_carry if use_adj_carry else None,
    )


def advance_carry_state(
    X: MolState,
    step_out: dict,
    *,
    edge_threshold: float,
    detach: bool,
) -> tuple[MolState, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    from model.consistency_operator import state_from_step

    X_next = state_from_step(X, step_out, edge_threshold=edge_threshold)
    R_next = step_out["R_pred"]
    feat_next = step_out["node_feat_pred"]
    adj_next = step_out.get("adj_soft")
    if detach:
        R_next = R_next.detach()
        feat_next = feat_next.detach()
        if adj_next is not None:
            adj_next = adj_next.detach()
    return X_next, R_next, feat_next, adj_next


def rollout_k_carry_trainable(
    model: torch.nn.Module,
    X0: MolState,
    *,
    hP: torch.Tensor,
    hc: torch.Tensor,
    device: torch.device,
    steps: int,
    carry_bptt_steps: int = 1,
) -> dict:
    """K-step C_theta carry rollout (R/feat/adj STE); returns final step_out."""
    n = max(1, int(steps))
    edge_thr = _edge_thr(model)
    X = X0
    R_carry: torch.Tensor | None = None
    feat_carry: torch.Tensor | None = None
    adj_carry: torch.Tensor | None = None
    step_out: dict = {}
    for step_idx in range(n):
        step_out = forward_step_with_carry(
            model,
            X,
            hP=hP,
            hc=hc,
            device=device,
            step_idx=step_idx,
            R_carry=R_carry,
            feat_carry=feat_carry,
            adj_carry=adj_carry,
            use_carry=True,
        )
        if step_idx < n - 1:
            detach = _should_detach(step_idx, n, carry_bptt_steps)
            X, R_carry, feat_carry, adj_carry = advance_carry_state(
                X,
                step_out,
                edge_threshold=edge_thr,
                detach=detach,
            )
    return step_out


def rollout_k_trainable(
    model: torch.nn.Module,
    X0: MolState,
    *,
    hP: torch.Tensor,
    hc: torch.Tensor,
    device: torch.device,
    steps: int,
    step_scale: float,
    carry_bptt_steps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Apply C_theta for k carry steps; returns final R_pred, node_emb, adj_soft."""
    del step_scale  # kept for call-site compatibility; step scale lives on the model
    step_out = rollout_k_carry_trainable(
        model,
        X0,
        hP=hP,
        hc=hc,
        device=device,
        steps=steps,
        carry_bptt_steps=carry_bptt_steps,
    )
    adj = step_out.get("adj_soft")
    return step_out["R_pred"], step_out["node_emb"], adj if adj is not None and adj.numel() > 0 else None


def loss_fixpoint(
    model: torch.nn.Module,
    starts: list[MolState],
    *,
    hP: torch.Tensor,
    hc: torch.Tensor,
    device: torch.device,
    steps: int,
    step_scale: float,
    w_g: float,
    w_gr: float,
    carry_bptt_steps: int = 1,
    X_star: MolState | None = None,
    star_align: float = 0.5,
) -> torch.Tensor:
    """Encourage k-step carry outputs from different starts to cluster (basin fixpoint)."""
    if len(starts) < 2 or int(steps) < 1:
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    coord_means: list[torch.Tensor] = []
    emb_mu: list[torch.Tensor] = []
    adj_mu: list[torch.Tensor] = []
    for x0 in starts:
        Rk, embk, adjk = rollout_k_trainable(
            model, x0, hP=hP, hc=hc, device=device, steps=steps,
            step_scale=step_scale, carry_bptt_steps=carry_bptt_steps,
        )
        coord_means.append(Rk.mean(dim=0))
        emb_mu.append(embk.mean(dim=0))
        if adjk is not None and w_gr > 0:
            tri = torch.triu(torch.ones_like(adjk, dtype=torch.bool), diagonal=1)
            adj_mu.append(adjk[tri].mean())

    C = torch.stack(coord_means, dim=0)
    E = torch.stack(emb_mu, dim=0)
    c_mu = C.mean(dim=0)
    e_mu = E.mean(dim=0)
    l_geom = F.mse_loss(C, c_mu.unsqueeze(0).expand_as(C))
    l_gr = F.mse_loss(E, e_mu.unsqueeze(0).expand_as(E))
    if adj_mu:
        A = torch.stack(adj_mu, dim=0)
        a_mu = A.mean()
        l_gr = l_gr + F.mse_loss(A, a_mu.expand_as(A))
    loss = float(w_g) * l_geom + float(w_gr) * l_gr
    if X_star is not None and float(star_align) > 0.0:
        R_star = torch.from_numpy(np.asarray(X_star.R, dtype=np.float32)).to(device=device, dtype=C.dtype)
        if R_star.shape[0] > 0:
            star_mu = R_star.mean(dim=0)
            loss = loss + float(star_align) * F.mse_loss(C, star_mu.unsqueeze(0).expand_as(C))
    return loss


def loss_pkt_fit(R_pred, ctx, *, wt: PktWt | None = None, tgt: float | None = None) -> torch.Tensor:
    out = pkt_fit_train(R_pred, ctx, wt=wt)
    if tgt is None:
        return -out.tot
    m = out.tot.new_tensor(float(tgt))
    return torch.relu(m - out.tot)


def loss_operator_edge_diffusion(
    model: torch.nn.Module,
    step_out: dict,
    X_ref: MolState,
    *,
    hP: torch.Tensor,
    hc: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Auxiliary edge-mask diffusion BCE for C_theta edge_denoiser (target = X_ref graph)."""
    from model.edge_diffusion import edge_loss

    denoiser = getattr(model, "edge_denoiser", None)
    if denoiser is None:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    emb = step_out.get("node_emb")
    if emb is None or emb.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    n = int(emb.shape[0])
    T = int(getattr(model, "edge_T", 20))
    x0 = adjacency_from_molstate(X_ref, n, device, emb.dtype)
    return edge_loss(denoiser, x0, emb, hP, hc, T)


@dataclass(frozen=True)
class ChainLoss:
    tot: torch.Tensor
    loc: torch.Tensor
    glob: torch.Tensor
    edit: torch.Tensor
    pocket: torch.Tensor
    edge: torch.Tensor
    n: int


def loss_basin_step(
    model: torch.nn.Module,
    chain: TrajChain,
    *,
    hP: torch.Tensor,
    hc: torch.Tensor,
    b_global: float,
    w_g: float,
    w_gr: float,
    delta_edit: int,
    device: torch.device,
    step_scale: float = 0.1,
    coord_loss: str = "mse",
    use_kabsch: bool = False,
    pkt_ctx: PktCtx | None = None,
    w_pocket: float = 0.0,
    pocket_tgt: float | None = None,
    w_edge_op: float = 0.0,
) -> ChainLoss:
    """Leap-like single step: C_theta(X_n) toward pool teacher and X* (no carry rollout)."""
    z = torch.tensor(0.0, device=device, dtype=torch.float32)
    if chain.n < 1:
        return ChainLoss(tot=z, loc=z, glob=z, edit=z, pocket=z, edge=z, n=0)

    Xn, X_t = chain.steps[0]
    st = model.forward_step_trainable(Xn, hP=hP, hc=hc, device=device)
    ll = loss_local(
        model, Xn, X_t, hP=hP, hc=hc, w_g=w_g, w_gr=w_gr, device=device,
        step_scale=step_scale, step_out=st, coord_loss=coord_loss, use_kabsch=use_kabsch,
    )
    lg = loss_global(
        model, Xn, chain.X_star, hP=hP, hc=hc, w_g=w_g, w_gr=w_gr, device=device,
        step_scale=step_scale, step_out=st, coord_loss=coord_loss, use_kabsch=use_kabsch,
    )
    ep = _edit_pen(st, Xn, delta_edit, device)
    L = ll + float(b_global) * lg + 0.1 * ep
    pk = z
    ed = z
    if pkt_ctx is not None and float(w_pocket) > 0.0:
        lp = loss_pkt_fit(st["R_pred"], pkt_ctx, tgt=pocket_tgt)
        L = L + float(w_pocket) * lp
        pk = lp
    if float(w_edge_op) > 0.0:
        le = loss_operator_edge_diffusion(model, st, X_t, hP=hP, hc=hc, device=device)
        L = L + float(w_edge_op) * le
        ed = le
    return ChainLoss(tot=L, loc=ll, glob=lg, edit=ep, pocket=pk, edge=ed, n=1)


@dataclass(frozen=True)
class GrpLoss:
    tot: torch.Tensor
    tgt: torch.Tensor
    basin: torch.Tensor
    edit: torch.Tensor
    pocket: torch.Tensor
    edge: torch.Tensor
    gsz: int


def d_state_from_step_outs(
    out_a: dict,
    out_b: dict,
    *,
    w_g: float,
    w_gr: float,
    device: torch.device,
    coord_loss: str = "mse",
    use_kabsch: bool = False,
) -> torch.Tensor:
    """d_state between two C_theta predictions (handles unequal node counts via pooled reps)."""
    zero = torch.tensor(0.0, device=device, dtype=torch.float32)
    emb_a = out_a.get("node_emb")
    emb_b = out_b.get("node_emb")
    R_a = out_a.get("R_pred")
    R_b = out_b.get("R_pred")
    if emb_a is None or emb_b is None or R_a is None or R_b is None:
        return zero
    if emb_a.numel() == 0 or emb_b.numel() == 0:
        return zero

    d_g = F.mse_loss(emb_a.mean(dim=0), emb_b.mean(dim=0))
    adj_a = out_a.get("adj_soft")
    adj_b = out_b.get("adj_soft")
    if (
        w_gr > 0
        and adj_a is not None
        and adj_b is not None
        and adj_a.numel() > 0
        and adj_b.numel() > 0
        and adj_a.shape == adj_b.shape
    ):
        triu = torch.triu(torch.ones_like(adj_a, dtype=torch.bool), diagonal=1)
        logits_a = torch.logit(adj_a[triu].clamp(1e-4, 1.0 - 1e-4))
        logits_b = torch.logit(adj_b[triu].clamp(1e-4, 1.0 - 1e-4))
        d_g = d_g + F.mse_loss(logits_a, logits_b)

    if int(R_a.shape[0]) == int(R_b.shape[0]) and int(R_a.shape[0]) > 0:
        d_r = _geom_dist(R_a, R_b, mode=coord_loss, use_kabsch=use_kabsch)
    else:
        d_r = F.mse_loss(R_a.mean(dim=0), R_b.mean(dim=0))
    return float(w_gr) * d_g + float(w_g) * d_r


def loss_basin_group(
    model: torch.nn.Module,
    group_a: MapGroup,
    group_b: MapGroup,
    *,
    hP: torch.Tensor,
    hc: torch.Tensor,
    w_g: float,
    w_gr: float,
    device: torch.device,
    coord_loss: str = "mse",
    use_kabsch: bool = False,
) -> torch.Tensor:
    """Eq. 10: E[d_state(C_theta(S_a), C_theta(S_b))] for groups from the same basin."""
    out_a = model.forward_group_step_trainable(
        list(group_a.states),
        hP,
        hc,
        device=device,
        X_star=group_a.X_star,
    )
    out_b = model.forward_group_step_trainable(
        list(group_b.states),
        hP,
        hc,
        device=device,
        X_star=group_b.X_star,
    )
    return d_state_from_step_outs(
        out_a,
        out_b,
        w_g=w_g,
        w_gr=w_gr,
        device=device,
        coord_loss=coord_loss,
        use_kabsch=use_kabsch,
    )


def loss_mapping_group(
    model: torch.nn.Module,
    group: MapGroup,
    *,
    hP: torch.Tensor,
    hc: torch.Tensor,
    w_g: float,
    w_gr: float,
    delta_edit: int,
    device: torch.device,
    coord_loss: str = "mse",
    use_kabsch: bool = False,
    pkt_ctx: PktCtx | None = None,
    w_pocket: float = 0.0,
    pocket_tgt: float | None = None,
    w_edge_op: float = 0.0,
    group_b: MapGroup | None = None,
    w_basin: float = 0.0,
) -> GrpLoss:
    """Many-to-one group loss: L_target on C_theta(S_k)->X* plus optional L_basin_group."""
    z = torch.tensor(0.0, device=device, dtype=torch.float32)
    sts = list(group.states)
    if not sts:
        return GrpLoss(tot=z, tgt=z, basin=z, edit=z, pocket=z, edge=z, gsz=0)

    st = model.forward_group_step_trainable(sts, hP, hc, device=device, X_star=group.X_star)
    pri = st.get("primary", sts[0])
    lt = d_state(
        model, pri, st["R_pred"], group.X_star, hP=hP, hc=hc, w_g=w_g, w_gr=w_gr, device=device,
        pred_node_emb=st.get("node_emb"), adj_pred=st.get("adj_soft"),
        coord_loss=coord_loss, use_kabsch=use_kabsch,
    ).tot
    ep = _edit_pen(st, pri, delta_edit, device)
    L = lt + 0.1 * ep
    lb = z
    if group_b is not None and float(w_basin) > 0.0:
        st_b = model.forward_group_step_trainable(list(group_b.states), hP, hc, device=device, X_star=group_b.X_star)
        lb = d_state_from_step_outs(st, st_b, w_g=w_g, w_gr=w_gr, device=device, coord_loss=coord_loss, use_kabsch=use_kabsch)
        L = L + float(w_basin) * lb

    pk = z
    ed = z
    if pkt_ctx is not None and float(w_pocket) > 0.0:
        lp = loss_pkt_fit(st["R_pred"], pkt_ctx, tgt=pocket_tgt)
        L = L + float(w_pocket) * lp
        pk = lp
    if float(w_edge_op) > 0.0:
        le = loss_operator_edge_diffusion(model, st, group.X_star, hP=hP, hc=hc, device=device)
        L = L + float(w_edge_op) * le
        ed = le

    return GrpLoss(tot=L, tgt=lt, basin=lb, edit=ep, pocket=pk, edge=ed, gsz=len(sts))


def loss_trajectory_chain(
    model: torch.nn.Module,
    chain: TrajChain,
    *,
    hP: torch.Tensor,
    hc: torch.Tensor,
    b_global: float,
    w_g: float,
    w_gr: float,
    delta_edit: int,
    device: torch.device,
    step_scale: float = 0.1,
    coord_loss: str = "mse",
    use_kabsch: bool = False,
    chain_global_ramp: bool = True,
    pkt_ctx: PktCtx | None = None,
    w_pocket: float = 0.0,
    pocket_tgt: float | None = None,
    carry_rollout: bool = True,
    carry_bptt_steps: int = 1,
    w_edge_op: float = 0.0,
) -> ChainLoss:
    """Apply C_theta along a short trajectory; pool teachers supervise each step."""
    if chain.n < 1:
        z = torch.tensor(0.0, device=device, dtype=torch.float32)
        return ChainLoss(tot=z, loc=z, glob=z, edit=z, pocket=z, edge=z, n=0)

    n = chain.n
    loc_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
    glob_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
    edit_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
    pk_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
    ed_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
    L = torch.tensor(0.0, device=device, dtype=torch.float32)

    eth = _edge_thr(model)
    here = chain.steps[0][0]
    R_c: torch.Tensor | None = None
    f_c: torch.Tensor | None = None
    a_c: torch.Tensor | None = None

    for i, (Xn_pool, X_t) in enumerate(chain.steps):
        Xn = here if carry_rollout else Xn_pool
        st = forward_step_with_carry(
            model, Xn, hP=hP, hc=hc, device=device,
            step_idx=i if carry_rollout else 0,
            R_carry=R_c, feat_carry=f_c, adj_carry=a_c, use_carry=carry_rollout,
        )
        ll = loss_local(
            model, Xn, X_t, hP=hP, hc=hc, w_g=w_g, w_gr=w_gr, device=device,
            step_scale=step_scale, step_out=st, coord_loss=coord_loss, use_kabsch=use_kabsch,
        )
        lg = loss_global(
            model, Xn, chain.X_star, hP=hP, hc=hc, w_g=w_g, w_gr=w_gr, device=device,
            step_scale=step_scale, step_out=st, coord_loss=coord_loss, use_kabsch=use_kabsch,
        )
        ep = _edit_pen(st, Xn, delta_edit, device)
        b_step = float(b_global) * float(i + 1) / float(n) if chain_global_ramp and n > 1 else float(b_global)
        step_L = ll + b_step * lg + 0.1 * ep
        if pkt_ctx is not None and float(w_pocket) > 0.0:
            lp = loss_pkt_fit(st["R_pred"], pkt_ctx, tgt=pocket_tgt)
            step_L = step_L + float(w_pocket) * lp
            pk_sum = pk_sum + lp
        if float(w_edge_op) > 0.0:
            le = loss_operator_edge_diffusion(model, st, X_t, hP=hP, hc=hc, device=device)
            step_L = step_L + float(w_edge_op) * le
            ed_sum = ed_sum + le
        L = L + step_L
        loc_sum = loc_sum + ll
        glob_sum = glob_sum + lg
        edit_sum = edit_sum + ep

        if carry_rollout and i < n - 1:
            detach = _should_detach(i, n, carry_bptt_steps)
            here, R_c, f_c, a_c = advance_carry_state(Xn, st, edge_threshold=eth, detach=detach)

    inv = 1.0 / float(n)
    return ChainLoss(
        tot=L * inv, loc=loc_sum * inv, glob=glob_sum * inv, edit=edit_sum * inv,
        pocket=pk_sum * inv, edge=ed_sum * inv, n=n,
    )
