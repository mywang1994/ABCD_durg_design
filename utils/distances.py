from __future__ import annotations
from dataclasses import dataclass
import networkx as nx
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from mol_state import MolState
from utils.molecular_distance import MolecularDistanceConfig

McmWeights = MolecularDistanceConfig


def _node_atomic_num(G, idx):
    if "atomic_num" not in G.nodes[idx]:
        raise ValueError(f"Atom {idx} is missing atomic_num")
    return int(G.nodes[idx]["atomic_num"])


def molstate_to_mol(state):
    """Return None for invalid states; missing chemistry is never filled in."""
    from utils.chemistry import graph_to_mol

    n = state.num_nodes()
    coords = np.asarray(state.R)
    if n == 0 or coords.shape != (n, 3) or not np.isfinite(coords).all():
        return None
    if state.G.number_of_nodes() != n:
        return None
    try:
        mol = graph_to_mol(state.G, sanitize=True)
        conf = Chem.Conformer(n)
        for i, p in enumerate(coords):
            conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(float(p[0]), float(p[1]), float(p[2])))
        mol.AddConformer(conf, assignId=True)
    except (ValueError, RuntimeError, KeyError, TypeError, OverflowError):
        return None
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
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = u @ vt
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
    """Four-component distance; invalid molecules are excluded with infinity."""
    from utils.molecular_distance import molecular_distance
    cfg = MolecularDistanceConfig(**weights) if isinstance(weights, dict) else weights
    try:
        return molecular_distance(X, Y, cfg)
    except ValueError:
        return float("inf")


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
    """Permutation-aware typed edit-script upper bound; see utils.edit_distance."""
    from utils.edit_distance import molecular_edit_distance
    return molecular_edit_distance(X, Y)
