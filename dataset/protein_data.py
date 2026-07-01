from __future__ import annotations

import csv
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from mol_state import MolState

_WATER = frozenset({"HOH", "H2O", "WAT", "TIP", "TIP3", "SOL"})


@dataclass(frozen=True)
class CrossDockComplex:

    complex_id: str
    receptor_pdb: Path
    reference_ligand_sdf: Path
    pocket_fn_rel: str | None = None
    ligand_fn_rel: str | None = None


@dataclass(frozen=True)
class CrossdockTypesRow:

    label: int
    pK: float
    rmsd_to_crystal: float | None
    receptor_rel: str
    ligand_rel: str
    vina_score: float | None
    raw_line: str


def _resolve(p: str | Path, root: Path | None) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    if root is None:
        return path.resolve()
    return (root / path).resolve()


def parse_crossdock_types_line(line: str) -> CrossdockTypesRow | None:
    s = line.strip()
    if not s or s.startswith("#"):
        return None

    comment_score: float | None = None
    if "#" in s:
        main, comment = s.split("#", 1)
        main = main.strip()
        m = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", comment)
        if m:
            try:
                comment_score = float(m.group(0))
            except ValueError:
                comment_score = None
    else:
        main = s

    parts = main.split()
    if len(parts) >= 5:
        label = int(float(parts[0]))
        pK = float(parts[1])
        try:
            rmsd = float(parts[2])
        except ValueError:
            rmsd = None
        rec, lig = parts[3], parts[4]
        return CrossdockTypesRow(
            label=label,
            pK=pK,
            rmsd_to_crystal=rmsd,
            receptor_rel=rec,
            ligand_rel=lig,
            vina_score=comment_score,
            raw_line=line.rstrip("\n"),
        )
    if len(parts) == 4:
        label = int(float(parts[0]))
        pK = float(parts[1])
        rec, lig = parts[2], parts[3]
        return CrossdockTypesRow(
            label=label,
            pK=pK,
            rmsd_to_crystal=None,
            receptor_rel=rec,
            ligand_rel=lig,
            vina_score=comment_score,
            raw_line=line.rstrip("\n"),
        )
    return None


def iter_crossdock_types(types_file: Path | str) -> Iterator[CrossdockTypesRow]:
    path = Path(types_file)
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            row = parse_crossdock_types_line(line)
            if row is not None:
                yield row


def complexes_from_types_file(
    types_file: Path | str,
    *,
    data_root: Path | str | None = None,
    complex_id_prefix: str = "",
) -> list[CrossDockComplex]:
    root = Path(data_root).resolve() if data_root is not None else None
    types_path = Path(types_file).resolve()
    complexes: list[CrossDockComplex] = []
    for row in iter_crossdock_types(types_path):
        rec = _resolve(row.receptor_rel, root)
        lig = _resolve(row.ligand_rel, root)
        safe_id = f"{complex_id_prefix}{rec.stem}__{lig.stem}".strip("_")
        complexes.append(
            CrossDockComplex(
                complex_id=safe_id,
                receptor_pdb=rec,
                reference_ligand_sdf=lig,
                pocket_fn_rel=row.receptor_rel,
                ligand_fn_rel=row.ligand_rel,
            )
        )
    return complexes


def load_complexes_from_index_csv(
    csv_path: Path | str,
    *,
    dataset_root: Path | str | None = None,
    encoding: str = "utf-8-sig",
) -> list[CrossDockComplex]:
    """load complexes from csv"""
    root = Path(dataset_root).resolve() if dataset_root is not None else None
    csv_path = Path(csv_path)

    rows: list[CrossDockComplex] = []
    with csv_path.open(newline="", encoding=encoding) as f:
        reader = csv.DictReader(f)
        required = {"complex_id", "receptor_pdb", "ligand_sdf"}
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        if not required.issubset({h.strip() for h in reader.fieldnames if h}):
            raise ValueError(f"CSV must contain columns {sorted(required)}, got {reader.fieldnames}")

        for raw in reader:
            cid = (raw.get("complex_id") or "").strip()
            if not cid:
                continue
            rec = _resolve(str(raw["receptor_pdb"]).strip(), root)
            lig = _resolve(str(raw["ligand_sdf"]).strip(), root)

            rows.append(
                CrossDockComplex(
                    complex_id=cid,
                    receptor_pdb=rec,
                    reference_ligand_sdf=lig,
                )
            )
    return rows


def discover_pairs_by_suffix(
    dataset_root: Path | str,
    *,
    receptor_suffix: str = "_receptor.pdb",
    ligand_suffix: str = "_ligand.sdf",
) -> list[CrossDockComplex]:
    """receptor-ligand files"""
    root = Path(dataset_root).resolve()
    rec_files = sorted(root.glob(f"*{receptor_suffix}"))
    pairs: list[CrossDockComplex] = []
    for rec in rec_files:
        stem = rec.name[: -len(receptor_suffix)]
        lig = root / f"{stem}{ligand_suffix}"
        if not lig.is_file():
            continue
        pairs.append(
            CrossDockComplex(
                complex_id=stem,
                receptor_pdb=rec.resolve(),
                reference_ligand_sdf=lig.resolve(),
            )
        )
    return pairs


def _pick_uff_reference_ligand(pocket_dir: Path) -> Path | None:
    uff_sdfs = sorted(pocket_dir.glob("*_uff*.sdf"))
    if uff_sdfs:
        return uff_sdfs[0]
    lig_pdbs = sorted(pocket_dir.glob("*_lig.pdb"))
    if lig_pdbs:
        return lig_pdbs[0]
    return None


def complex_from_pocket_subdir(
    pocket_dir: Path | str,
    *,
    complex_id: str | None = None,
) -> CrossDockComplex | None:
    d = Path(pocket_dir).resolve()
    rec_files = sorted(d.glob("*_rec.pdb"))
    if not rec_files:
        return None
    rec = rec_files[0]
    ref_lig = _pick_uff_reference_ligand(d)
    if ref_lig is None:
        return None
    cid = complex_id or d.name
    return CrossDockComplex(
        complex_id=cid,
        receptor_pdb=rec.resolve(),
        reference_ligand_sdf=ref_lig.resolve(),
    )


def discover_pocket_subdirs(
    pocket_dataset_root: Path | str,
    *,
    max_complexes: int | None = None,
) -> list[CrossDockComplex]:

    root = Path(pocket_dataset_root).resolve()
    found: list[CrossDockComplex] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        cx = complex_from_pocket_subdir(child)
        if cx is None:
            continue
        found.append(cx)
        if max_complexes is not None and len(found) >= max_complexes:
            break
    return found




def ligand_sdf_from_pocket10_pdb(pocket_pdb: Path) -> Path:
    p = Path(pocket_pdb)
    if not p.name.endswith("_pocket10.pdb"):
        raise ValueError(f"Expected *_pocket10.pdb path, got {p}")
    return p.with_name(p.name.replace("_pocket10.pdb", ".sdf"))


def discover_crossdock_pocket10(
    pocket10_root: Path | str,
    *,
    max_clusters: int | None = None,
    max_pairs_per_cluster: int | None = None,
) -> list[CrossDockComplex]:
    root = Path(pocket10_root).resolve()
    pairs: list[CrossDockComplex] = []
    clusters = sorted(p for p in root.iterdir() if p.is_dir())
    for ci, cluster_dir in enumerate(clusters):
        if max_clusters is not None and ci >= max_clusters:
            break
        pocket_pdbs = sorted(cluster_dir.glob("*_pocket10.pdb"))
        n_done = 0
        for pocket_pdb in pocket_pdbs:
            if max_pairs_per_cluster is not None and n_done >= max_pairs_per_cluster:
                break
            try:
                lig = ligand_sdf_from_pocket10_pdb(pocket_pdb)
            except ValueError:
                continue
            if not lig.is_file():
                continue
            cid = f"{cluster_dir.name}__{lig.stem}"
            pairs.append(
                CrossDockComplex(
                    complex_id=cid,
                    receptor_pdb=pocket_pdb.resolve(),
                    reference_ligand_sdf=lig.resolve(),
                )
            )
            n_done += 1
    return pairs


def load_crossdock_index_pkl(index_pkl: Path | str) -> list[Any]:
    path = Path(index_pkl)
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"index.pkl must be a list, got {type(data)}")
    return data


def complexes_from_crossdock_index_pkl(
    pocket10_root: Path | str,
    index_pkl: Path | str | None = None,
) -> list[CrossDockComplex]:
    root = Path(pocket10_root).resolve()
    pkl = Path(index_pkl).resolve() if index_pkl is not None else root / "index.pkl"
    index = load_crossdock_index_pkl(pkl)
    complexes: list[CrossDockComplex] = []

    for i, row in enumerate(index):
        if not row:
            continue
        pocket_fn = row[0]
        ligand_fn = row[1]
        if pocket_fn is None:
            continue
        pocket_fn_s = str(pocket_fn)
        ligand_fn_s = str(ligand_fn)
        rec = (root / pocket_fn_s).resolve()
        lig = (root / ligand_fn_s).resolve()
        cluster = Path(pocket_fn_s).parts[0] if Path(pocket_fn_s).parts else "item"
        cid = f"{cluster}__{Path(ligand_fn_s).stem}__{i}"

        complexes.append(
            CrossDockComplex(
                complex_id=cid,
                receptor_pdb=rec,
                reference_ligand_sdf=lig,
                pocket_fn_rel=pocket_fn_s,
                ligand_fn_rel=ligand_fn_s,
            )
        )
    return complexes


def iter_sdfs(sdf_path: Path) -> Iterator[Path]:
    p = Path(sdf_path)
    if p.is_dir():
        yield from sorted(p.glob("*.sdf"))
    else:
        yield p


def pocket_coords(pdb_path, max_atoms=None, seed=0):
    path = Path(pdb_path)
    raw: list[list[float]] = []

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(line) < 54:
                continue
            if line[0:6] not in ("ATOM  ", "HETATM"):
                continue
            if line[17:20].strip().upper() in _WATER:
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            elem = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not elem and len(line) >= 16:
                aname = line[12:16].strip().upper()
                if aname.startswith(("H", "D")):
                    elem = "H"
            if elem in ("H", "D"):
                continue
            raw.append([x, y, z])

    if not raw:
        return np.zeros((0, 3), dtype=np.float32)

    coords = np.asarray(raw, dtype=np.float32)
    n = coords.shape[0]
    if max_atoms is not None and n > max_atoms:
        idx = np.random.default_rng(seed).choice(n, size=max_atoms, replace=False)
        coords = coords[idx]
    return coords


def task_vec(X: MolState, dim: int):
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive")
    chunks = [
        np.asarray(X.R.mean(axis=0), dtype=np.float32),
        np.asarray(X.R.std(axis=0), dtype=np.float32),
        np.asarray(X.node_feat.mean(axis=0), dtype=np.float32),
        np.asarray([np.log(float(X.num_nodes()) + 1.0)], dtype=np.float32),
    ]
    flat = np.concatenate([p.ravel() for p in chunks])
    task = np.zeros(dim, dtype=np.float32)
    n = min(dim, int(flat.shape[0]))
    task[:n] = flat[:n]
    return task
