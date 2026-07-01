from __future__ import annotations
from dataclasses import dataclass
import networkx as nx
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from mol_state import MolState
#distances functions


'''
A toolbox for measuring how similar two MolStates are  
— mainly used for state-pool sampling, teacher selection, 
generation ranking, and MolState.distance_to
'''



@dataclass(frozen=True)
class McmWeights:

    w_topo: float = 0.35
    w_geom: float = 0.45
    w_feat: float = 0.2
    fp_radius: int = 2
    fp_nbits: int = 2048
    geom_rmsd_cap_a: float = 5.0
    feat_l2_cap: float = 5.0


def _node_atomic_num(G, idx):
    return int(G.nodes[idx].get("atomic_num", 6))


def molstate_to_mol(state):
    n = state.num_nodes()
    if n == 0:
        return None
    rw = Chem.RWMol()
    for i in range(n):
        rw.AddAtom(Chem.Atom(_node_atomic_num(state.G, i)))
    seen: set[tuple[int, int]] = set()
    for u, v in state.G.edges():
        a, b = int(u), int(v)
        if a > b:
            a, b = b, a
        if (a, b) in seen:
            continue
        seen.add((a, b))
        rw.AddBond(a, b, Chem.BondType.SINGLE)
    try:
        mol = rw.GetMol()
    except Exception:
        return None
    conf = Chem.Conformer(n)
    for i in range(n):
        p = state.R[i]
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(float(p[0]), float(p[1]), float(p[2])))
    mol.AddConformer(conf, assignId=True)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


def _morgan_fp(mol: Chem.Mol, *, radius: int, nbits: int):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def tanimoto_molstate(X, Y, radius=2, nbits=2048):
    mx, my = molstate_to_mol(X), molstate_to_mol(Y)
    if mx is None or my is None:
        return None
    try:
        f1 = _morgan_fp(mx, radius=radius, nbits=nbits)
        f2 = _morgan_fp(my, radius=radius, nbits=nbits)
        return float(DataStructs.TanimotoSimilarity(f1, f2))
    except Exception:
        return None


def _kabsch_rotate(P, Q):

    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    h = Pc.T @ Qc
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    return (Pc @ r) + Q.mean(axis=0)


def rmsd_aligned_same_index(X, Y):

    n = X.num_nodes()
    if n == 0 or Y.num_nodes() != n:
        return float("inf")
    p = X.R.astype(np.float64)
    q = Y.R.astype(np.float64)
    pal = _kabsch_rotate(p, q)
    return float(np.sqrt(np.mean(np.sum((pal - q) ** 2, axis=1))))


def geom_distance_molstate(X, Y):

    n1, n2 = X.num_nodes(), Y.num_nodes()
    if n1 == 0 or n2 == 0:
        return 0.0
    if n1 == n2:
        return rmsd_aligned_same_index(X, Y)
    n_match = min(n1, n2)
    d = float(np.sqrt(np.mean(np.sum((X.R[:n_match] - Y.R[:n_match]) ** 2, axis=1))))
    return d + 0.4 * abs(n1 - n2)


def feat_mean_l2(X, Y):
    return float(np.linalg.norm(X.node_feat.mean(axis=0) - Y.node_feat.mean(axis=0)))


def d_mcm(X, Y, weights=None):

    weights = weights or McmWeights()
    fp_sim = tanimoto_molstate(X, Y, radius=weights.fp_radius, nbits=weights.fp_nbits)
    if fp_sim is not None:
        d_topo = 1.0 - fp_sim
    else:
        e1 = {tuple(sorted(e)) for e in X.G.edges()}
        e2 = {tuple(sorted(e)) for e in Y.G.edges()}
        d_topo = min(1.0, len(e1.symmetric_difference(e2)) / max(1, max(len(e1), len(e2))))
    rmsd = geom_distance_molstate(X, Y)
    if not np.isfinite(rmsd):
        rmsd = float(np.sqrt(coordinate_mse_min_nodes(X, Y)))
    d_geom = min(rmsd / max(weights.geom_rmsd_cap_a, 1e-6), 1.0)
    df = feat_mean_l2(X, Y)
    d_feat = min(df / max(weights.feat_l2_cap, 1e-6), 1.0)
    return float(weights.w_topo * d_topo + weights.w_geom * d_geom + weights.w_feat * d_feat)


def edge_set_edit_count(G1, G2):

    e1 = {tuple(sorted(e)) for e in G1.edges()}
    e2 = {tuple(sorted(e)) for e in G2.edges()}
    return int(len(e1.symmetric_difference(e2)))


def graph_heuristic_distance(X, Y):

    edge_term = abs(X.G.number_of_edges() - Y.G.number_of_edges())
    feat_term = float(np.linalg.norm(X.node_feat.mean(axis=0) - Y.node_feat.mean(axis=0)))
    return float(edge_term + 0.1 * feat_term)


def coordinate_mse_min_nodes(X, Y):

    n = min(X.num_nodes(), Y.num_nodes())
    if n == 0:
        return 0.0
    return float(np.mean((X.R[:n] - Y.R[:n]) ** 2))


def edit_distance_molstate(X, Y):

    n1, n2 = X.num_nodes(), Y.num_nodes()
    if n1 == n2 and n1 > 0:
        edge_delta = edge_set_edit_count(X.G, Y.G)
        atom_delta = sum(
            1
            for i in range(n1)
            if _node_atomic_num(X.G, i) != _node_atomic_num(Y.G, i)
        )
        return int(edge_delta + atom_delta)
    base = abs(n1 - n2)
    fp_sim = tanimoto_molstate(X, Y)
    if fp_sim is not None:
        topo = int(round((1.0 - fp_sim) * max(n1, n2, 1)))
        return int(base + topo)
    return int(base + abs(X.G.number_of_edges() - Y.G.number_of_edges()))
