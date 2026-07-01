from __future__ import annotations

import gzip
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors

from mol_state import MolState

StatePoolKind = Literal["local", "pharm", "scaffold", "rxn", "geo", "target"]


def _default_mmp_db() -> Path:
    return Path(__file__).resolve().parent / "mmp.db"


@dataclass
class CrossDockState:
    X: MolState
    pool_kind: StatePoolKind | str
    complex_id: str
    source_note: str = ""
    smiles: str | None = None

    def with_meta_note(self, note: str) -> CrossDockState:
        return CrossDockState(
            X=self.X,
            pool_kind=self.pool_kind,
            complex_id=self.complex_id,
            source_note=note,
            smiles=self.smiles,
        )


@dataclass
class StatePool:
    complex_id: str
    X_star: MolState
    by_kind: dict[str, list[CrossDockState]] = field(default_factory=dict)

    def all_states(self) -> list[CrossDockState]:
        out: list[CrossDockState] = []
        for k in ("local", "pharm", "scaffold", "rxn", "geo"):
            out.extend(self.by_kind.get(k, []))
        return out

    def all_molstates(self) -> list[MolState]:
        return [s.X for s in self.all_states()]


@dataclass
class LocalMMPConfig:
    fragment_db_path: Path | str | None = field(default_factory=_default_mmp_db)
    mmp_engine_root: Path | str | None = None
    substitution_module: str | None = None
    radius: int = 3
    mutate_max_replacements: int | None = 128
    mutate_min_size: int = 0
    mutate_max_size: int = 10
    mutate_min_inc: int = -2
    mutate_max_inc: int = 2
    grow_from_db: bool = False
    grow_max_replacements: int | None = 32
    grow_min_atoms: int = 1
    grow_max_atoms: int = 3
    protected_atom_indices: tuple[int, ...] | None = None
    fallback_prune_max_heavy: int = 4


@dataclass
class PoolCounts:
    local: int = 15
    pharm: int = 10
    scaffold: int = 10
    rxn: int = 5
    geo: int = 15


@dataclass
class PoolCfg:
    seed: int = 0
    counts: PoolCounts = field(default_factory=PoolCounts)
    geo_sigma: float = 0.15
    local_tries: int = 128
    t_min: float = 0.15
    t_max: float = 0.98
    pocket_min: float | None = None
    max_pool: int | None = 512
    mmp_db: Path | str | None = None
    local_mmp: LocalMMPConfig | None = None


def _local_mmp(cfg: PoolCfg) -> LocalMMPConfig:
    mmp_cfg = cfg.local_mmp if cfg.local_mmp is not None else LocalMMPConfig()
    if cfg.mmp_db is None:
        return mmp_cfg
    return replace(mmp_cfg, fragment_db_path=Path(cfg.mmp_db))


def load_reference_mol_from_sdf(sdf_path: Path | str, *, sanitize: bool = True):
    path = Path(sdf_path)

    def _iter_supplier():
        if path.suffix.lower() == ".gz" or str(path).lower().endswith(".sdf.gz"):
            with gzip.open(path, "rb") as fh:
                suppl = Chem.ForwardSDMolSupplier(fh, removeHs=False, sanitize=sanitize)
                for m in suppl:
                    yield m
        else:
            suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=sanitize)
            for m in suppl:
                yield m

    for m in _iter_supplier():
        if m is not None:
            return m
    raise ValueError(f"No valid molecule in SDF: {path}")


def mol_to_molstate(mol, conf_id: int = -1, *, node_feat_dim: int = 16) -> MolState:
    m = Chem.Mol(mol)
    if m.GetNumConformers() == 0:
        AllChem.EmbedMolecule(m, randomSeed=0xC0FFEE)
        AllChem.MMFFOptimizeMolecule(m)
    if conf_id < 0:
        conf_id = m.GetNumConformers() - 1
    conf = m.GetConformer(conf_id)
    n = m.GetNumAtoms()
    R = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        p = conf.GetAtomPosition(i)
        R[i, 0] = float(p.x)
        R[i, 1] = float(p.y)
        R[i, 2] = float(p.z)

    G = nx.Graph()
    for atom in m.GetAtoms():
        G.add_node(atom.GetIdx(), atomic_num=int(atom.GetAtomicNum()))
    for bond in m.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.add_edge(int(a), int(b))

    atom_rows: list[list[float]] = []
    for atom in m.GetAtoms():
        atom_rows.append(
            [
                float(atom.GetAtomicNum()) / 100.0,
                float(atom.GetDegree()) / 6.0,
                float(atom.GetIsAromatic()),
                float(atom.GetTotalNumHs()) / 4.0,
                float(atom.GetFormalCharge()) / 4.0,
                float(atom.IsInRing()),
                float(atom.GetHybridization()),
                float(atom.GetChiralTag()),
            ]
        )
    raw = np.asarray(atom_rows, dtype=np.float32)
    if raw.shape[1] < node_feat_dim:
        padding = np.zeros((n, node_feat_dim - raw.shape[1]), dtype=np.float32)
        node_feat = np.concatenate([raw, padding], axis=1)
    else:
        node_feat = raw[:, :node_feat_dim]

    return MolState(G=G, R=R, node_feat=node_feat)


def mol_to_smiles(mol, *, canonical: bool = True) -> str:
    return Chem.MolToSmiles(mol, canonical=canonical)


def morgan_fp(mol, radius: int = 2, nbits: int = 2048):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def tanimoto_to_reference(ref_mol, mol, *, radius: int = 2, nbits: int = 2048) -> float:
    f1 = morgan_fp(ref_mol, radius=radius, nbits=nbits)
    f2 = morgan_fp(mol, radius=radius, nbits=nbits)
    return float(DataStructs.TanimotoSimilarity(f1, f2))


def ensure_conformer(mol: Chem.Mol) -> None:
    if mol.GetNumConformers() > 0:
        return
    AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
    if mol.GetNumConformers() == 0:
        return
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
