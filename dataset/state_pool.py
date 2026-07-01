from __future__ import annotations

#attraction-basin state pool builder


import random
from pathlib import Path

import networkx as nx
import numpy as np
from rdkit import Chem

from .protein_data import CrossDockComplex
from .pool_builders import RXN_TPL, build_geo, build_local, build_pharm, build_rxn, build_scaffold
from .pool_core import (
    CrossDockState,
    LocalMMPConfig,
    PoolCfg,
    PoolCounts,
    StatePool,
    StatePoolKind,
    _local_mmp,
    load_reference_mol_from_sdf,
    mol_to_molstate,
    mol_to_smiles,
    tanimoto_to_reference,
)
from utils.distances import edit_distance_molstate
from mol_state import MolState

__all__ = [
    "CrossDockState",
    "StatePool",
    "PoolCfg",
    "LocalMMPConfig",
    "PoolCounts",
    "StatePoolKind",
    "load_reference_mol_from_sdf",
    "mol_to_molstate",
    "mol_to_smiles",
    "tanimoto_to_reference",
    "build_geo",
    "build_scaffold",
    "build_local",
    "build_pharm",
    "build_rxn",
    "RXN_TPL",
    "build_crossdock_bundle",
    "build_state_pool_test",
    "filter_pool_by_edit_delta",
]


def _passes_similarity(ref_mol, mol, cfg: PoolCfg) -> bool:
    try:
        t = tanimoto_to_reference(ref_mol, mol)
        return cfg.t_min <= t <= cfg.t_max
    except Exception:
        return True


def _passes_pkt_fit(
    st: CrossDockState,
    receptor_pdb: Path,
    cfg: PoolCfg,
    ref_mol,
) -> bool:
    if cfg.pocket_min is None:
        return True
    from utils.pkt_fit import pkt_fit_stub

    pocket: dict = {"receptor_pdb": receptor_pdb}
    if ref_mol is not None:
        pocket["reference_mol"] = ref_mol
    out = pkt_fit_stub(st.X, pocket=pocket)
    return out.tot >= cfg.pocket_min


def _need_tanimoto_filter(item: CrossDockState, ref_smi: str) -> bool:
    if item.pool_kind == "geo":
        return False
    if item.smiles is None:
        return False
    if item.smiles == ref_smi:
        return False
    return True


def _passes_filters(item: CrossDockState, ref_mol, ref_smi: str, receptor_pdb: Path, cfg: PoolCfg) -> bool:
    if _need_tanimoto_filter(item, ref_smi):
        m = Chem.MolFromSmiles(item.smiles or "")
        if m is None:
            return False
        if not _passes_similarity(ref_mol, m, cfg):
            return False
    if not _passes_pkt_fit(item, receptor_pdb, cfg, ref_mol):
        return False
    return True


def _backfill_kind_shortfalls(
    buckets: dict[str, list[CrossDockState]],
    *,
    counts: PoolCounts,
    ref_mol,
    complex_id: str,
    rng: random.Random,
    cfg: PoolCfg,
    ref_smi: str,
    receptor_pdb: Path,
) -> None:
    order = ("local", "pharm", "scaffold", "rxn", "geo")
    smiles_seen: set[str] = set()
    for lst in buckets.values():
        for st in lst:
            if st.smiles:
                smiles_seen.add(st.smiles)
    try:
        smiles_seen.add(ref_smi)
    except Exception:
        pass

    for kind in order:
        target = getattr(counts, kind, 0)
        slot = buckets.setdefault(kind, [])
        short = target - len(slot)
        if short <= 0:
            continue

        def _pull_from_candidates(candidates: list[CrossDockState], tag: str) -> None:
            nonlocal short
            for cand in candidates:
                if short <= 0:
                    break
                if not _passes_filters(cand, ref_mol, ref_smi, receptor_pdb, cfg):
                    continue
                smi = cand.smiles
                if smi is not None and smi in smiles_seen:
                    continue
                slot.append(
                    CrossDockState(
                        X=cand.X,
                        pool_kind=kind,
                        complex_id=complex_id,
                        source_note=f"{tag}:{cand.source_note}",
                        smiles=smi,
                    )
                )
                if smi is not None:
                    smiles_seen.add(smi)
                short -= 1

        n_try = min(256, max(short * 4, short + 16, 32))
        local_try = build_local(
            ref_mol,
            complex_id,
            n_try,
            rng,
            max(cfg.local_tries, n_try * 2),
            _local_mmp(cfg),
        )
        rng.shuffle(local_try)
        _pull_from_candidates(local_try, "backfill_local")

        if short > 0:
            n_geo = min(256, max(short * 4, short + 16, 32))
            geo_try = build_geo(ref_mol, complex_id, n_geo, rng, cfg.geo_sigma)
            rng.shuffle(geo_try)
            _pull_from_candidates(geo_try, "backfill_geo")


def build_crossdock_bundle(
    cx: CrossDockComplex,
    cfg: PoolCfg | None = None,
) -> StatePool:
    if not isinstance(cx, CrossDockComplex):
        raise TypeError("cx must be CrossDockComplex")

    cfg = cfg or PoolCfg()
    rng = random.Random(cfg.seed)

    ref_mol = load_reference_mol_from_sdf(cx.reference_ligand_sdf)
    X_star = mol_to_molstate(ref_mol)
    ref_smi = mol_to_smiles(ref_mol)
    want = cfg.counts

    built: dict[str, list[CrossDockState]] = {
        "local": build_local(ref_mol, cx.complex_id, want.local, rng, cfg.local_tries, _local_mmp(cfg)),
        "pharm": build_pharm(ref_mol, cx.complex_id, want.pharm, rng, cx.receptor_pdb, cx.reference_ligand_sdf),
        "scaffold": build_scaffold(ref_mol, cx.complex_id, want.scaffold, rng),
        "rxn": build_rxn(ref_mol, cid=cx.complex_id, n=want.rxn, rng=rng),
        "geo": build_geo(ref_mol, cx.complex_id, want.geo, rng, cfg.geo_sigma),
    }

    picked: list[CrossDockState] = []
    for items in built.values():
        for item in items:
            if _passes_filters(item, ref_mol, ref_smi, cx.receptor_pdb, cfg):
                picked.append(item)

    if cfg.max_pool is not None and len(picked) > cfg.max_pool:
        rng.shuffle(picked)
        picked = picked[: cfg.max_pool]

    grouped: dict[str, list[CrossDockState]] = {}
    for s in picked:
        grouped.setdefault(str(s.pool_kind), []).append(s)

    _backfill_kind_shortfalls(
        grouped,
        counts=want,
        ref_mol=ref_mol,
        complex_id=cx.complex_id,
        rng=rng,
        cfg=cfg,
        ref_smi=ref_smi,
        receptor_pdb=cx.receptor_pdb,
    )

    return StatePool(complex_id=cx.complex_id, X_star=X_star, by_kind=grouped)


class build_state_pool_test:

    @staticmethod
    def make_target_state(num_nodes: int = 12, feat_dim: int = 16) -> MolState:
        G = nx.cycle_graph(num_nodes)
        R = np.random.randn(num_nodes, 3).astype(np.float32)
        node_feat = np.random.randn(num_nodes, feat_dim).astype(np.float32)
        return MolState(G=G, R=R, node_feat=node_feat)

    @staticmethod
    def _perturb_graph(G: nx.Graph, max_edge_flips: int = 2) -> nx.Graph:
        H = G.copy()
        n = H.number_of_nodes()
        flips = random.randint(1, max_edge_flips)
        for _ in range(flips):
            if random.random() < 0.5 and H.number_of_edges() > 0:
                e = random.choice(list(H.edges()))
                H.remove_edge(*e)
            else:
                u = random.randrange(n)
                v = random.randrange(n)
                if u != v:
                    H.add_edge(u, v)
        return H

    @staticmethod
    def _perturb_geom(R: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        return (R + sigma * np.random.randn(*R.shape)).astype(np.float32)

    @staticmethod
    def build_pool(X_star: MolState, pool_size: int = 128) -> list[MolState]:
        out: list[MolState] = []
        for _ in range(pool_size):
            Gp = build_state_pool_test._perturb_graph(X_star.G, max_edge_flips=2)
            Rp = build_state_pool_test._perturb_geom(X_star.R, sigma=0.2)
            fp = (X_star.node_feat + 0.05 * np.random.randn(*X_star.node_feat.shape)).astype(np.float32)
            out.append(MolState(G=Gp, R=Rp, node_feat=fp))
        return out


def filter_pool_by_edit_delta(mol_states: list[MolState], X_ref: MolState, delta_edit: int) -> list[MolState]:
    keep: list[MolState] = []
    for X in mol_states:
        if edit_distance_molstate(X, X_ref) <= delta_edit:
            keep.append(X)
    return keep
