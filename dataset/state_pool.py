from __future__ import annotations



import random
from pathlib import Path

import numpy as np
from rdkit import Chem

from .protein_data import CrossDockComplex
from .sampling import is_valid
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

POOL_CAP_POLICY = "preserve_minimum_scaffolds_then_uniform_remainder_v1"

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
        return False


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


def _passes_filters(item: CrossDockState, ref_mol, ref_smi: str, receptor_pdb: Path, cfg: PoolCfg,
                    diagnostics=None) -> bool:
    if not is_valid(item.X):
        return False
    if not is_valid(item.X, require_geometry=True):
        if diagnostics is not None:
            diagnostics["pool_geometry_rejected"] = diagnostics.get("pool_geometry_rejected", 0) + 1
        return False
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
    diagnostics: dict[str, int],
) -> None:
    """Fill capacity while preserving each candidate's actual source.
    Source quotas may fall short; geometric/local replacements cannot count as pharm/scaffold/reaction states.
    """
    target = sum(getattr(counts, kind) for kind in ("local", "pharm", "scaffold", "rxn", "geo"))
    if cfg.max_pool is not None:
        target = min(target, cfg.max_pool)
    short = target - sum(len(states) for states in buckets.values())
    if short <= 0:
        return
    smiles_seen = {st.smiles for states in buckets.values() for st in states if st.smiles}
    smiles_seen.add(ref_smi)

    def _pull(candidates: list[CrossDockState], tag: str) -> None:
        nonlocal short
        for cand in candidates:
            if short <= 0:
                break
            if not _passes_filters(cand, ref_mol, ref_smi, receptor_pdb, cfg, diagnostics):
                continue
            # Deduplicate structural edits by SMILES; mapping sampling also checks full graph/coordinates.
            if cand.pool_kind != "geo" and cand.smiles in smiles_seen:
                continue
            buckets.setdefault(str(cand.pool_kind), []).append(cand.with_meta_note(f"{tag}:{cand.source_note}"))
            if cand.smiles:
                smiles_seen.add(cand.smiles)
            short -= 1

    n_try = min(256, max(short * 4, short + 16, 32))
    local_try = build_local(ref_mol, complex_id, n_try, rng,
                            max(cfg.local_tries, n_try * 2), _local_mmp(cfg), diagnostics=diagnostics)
    rng.shuffle(local_try)
    _pull(local_try, "backfill_local")
    if short > 0:
        _pull(build_geo(ref_mol, complex_id, short, rng, cfg.geo_sigma), "backfill_geo")


def build_crossdock_bundle(
    cx: CrossDockComplex,
    cfg: PoolCfg | None = None,
) -> StatePool:
    if not isinstance(cx, CrossDockComplex):
        raise TypeError("cx must be CrossDockComplex")

    cfg = cfg or PoolCfg()
    if (cfg.max_pool is not None and (isinstance(cfg.max_pool, bool)
            or not isinstance(cfg.max_pool, int) or cfg.max_pool < 0)):
        raise ValueError("max_pool must be a nonnegative integer or None")
    rng = random.Random(cfg.seed)

    ref_mol = load_reference_mol_from_sdf(cx.reference_ligand_sdf, require_3d=True)
    X_star = mol_to_molstate(ref_mol)
    ref_smi = mol_to_smiles(ref_mol)
    want = cfg.counts
    diagnostics: dict[str, int] = {}

    built: dict[str, list[CrossDockState]] = {
        "local": build_local(ref_mol, cx.complex_id, want.local, rng, cfg.local_tries, _local_mmp(cfg), diagnostics=diagnostics),
        "pharm": build_pharm(ref_mol, cx.complex_id, want.pharm, rng, cx.receptor_pdb, cx.reference_ligand_sdf,
                             diagnostics=diagnostics),
        "scaffold": build_scaffold(ref_mol, cx.complex_id, want.scaffold, rng, diagnostics=diagnostics),
        "rxn": build_rxn(ref_mol, cid=cx.complex_id, n=want.rxn, rng=rng, diagnostics=diagnostics),
        "geo": build_geo(ref_mol, cx.complex_id, want.geo, rng, cfg.geo_sigma),
    }

    picked: list[CrossDockState] = []
    for items in built.values():
        for item in items:
            if _passes_filters(item, ref_mol, ref_smi, cx.receptor_pdb, cfg, diagnostics):
                picked.append(item)

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
        diagnostics=diagnostics,
    )

    # Cap after all builders and backfills to enforce the final pool limit.
    picked = [state for states in grouped.values() for state in states]
    if cfg.max_pool is not None and len(picked) > cfg.max_pool:
        scaffolds = [(index, sum(int(attrs["atomic_num"]) > 1 for _, attrs in state.X.G.nodes(data=True)))
                     for index, state in enumerate(picked) if state.pool_kind == "scaffold"]
        minimum = min((count for _, count in scaffolds), default=None)
        simplest = [index for index, count in scaffolds if count == minimum]
        reserved = rng.sample(simplest, min(cfg.max_pool, len(simplest)))
        remaining = [index for index in range(len(picked)) if index not in reserved]
        selected = reserved + rng.sample(remaining, cfg.max_pool - len(reserved))
        diagnostics["pool_cap_removed"] = len(picked) - len(selected)
        diagnostics["pool_cap_minimum_scaffolds_retained"] = len(reserved)
        picked = [picked[index] for index in selected]
    grouped = {}
    for state in picked:
        grouped.setdefault(str(state.pool_kind), []).append(state)

    return StatePool(complex_id=cx.complex_id, X_star=X_star, by_kind=grouped, build_rejections=diagnostics)


class build_state_pool_test:

    @staticmethod
    def make_target_state(num_nodes: int = 12, feat_dim: int = 16) -> MolState:
        """Small chemically typed fixture, not a molecular-generation dataset."""
        if int(num_nodes) < 1:
            raise ValueError("num_nodes must be positive")
        n = int(num_nodes)
        smiles = "C" if n == 1 else "CO" if n == 2 else "C" * (n - 3) + "C(=O)N"
        return mol_to_molstate(Chem.MolFromSmiles(smiles), node_feat_dim=feat_dim)

    @staticmethod
    def build_pool(X_star: MolState, pool_size: int = 128) -> list[MolState]:
        """Distinct rigid poses for synthetic training-path checks."""
        return [X_star.copy_with(R=(X_star.R + .2 * np.random.randn(1, 3)).astype(np.float32))
                for _ in range(pool_size)]


def filter_pool_by_edit_delta(mol_states: list[MolState], X_ref: MolState, delta_edit: int) -> list[MolState]:
    keep: list[MolState] = []
    for X in mol_states:
        if edit_distance_molstate(X, X_ref) <= delta_edit:
            keep.append(X)
    return keep
