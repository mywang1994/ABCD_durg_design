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
TASK_VECTOR_REPRESENTATION = "rigid_invariant_v2"

# Category order, including unknown atoms, is part of the checkpoint schema.
PROTEIN_ELEMENTS = (
    "C", "N", "O", "S", "P", "SE", "F", "CL", "BR", "I", "B", "SI",
    "NA", "MG", "K", "CA", "MN", "FE", "CO", "NI", "CU", "ZN", "UNK",
)
PROTEIN_RESIDUES = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK",
)
PROTEIN_FEATURE_DIM = len(PROTEIN_ELEMENTS) + len(PROTEIN_RESIDUES) + 1
PROTEIN_FEATURE_SCHEMA = "element23_residue21_backbone_v1"
_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})
_RESIDUE_ALIASES = {"MSE": "MET", "HID": "HIS", "HIE": "HIS", "HIP": "HIS"}


def protein_atom_features(elements, residue_names, atom_names) -> np.ndarray:
    """Encode one-hot elements (23), one-hot residues (21), and a recognized-residue backbone flag.
    Use UNK for nonstandard residues except MSE and common histidine names; do not infer charges.
    """
    elements, residue_names, atom_names = list(elements), list(residue_names), list(atom_names)
    if not (len(elements) == len(residue_names) == len(atom_names)):
        raise ValueError("Element, residue, and atom-name arrays must have the same length")
    result = np.zeros((len(elements), PROTEIN_FEATURE_DIM), dtype=np.float32)
    element_indices = {name: i for i, name in enumerate(PROTEIN_ELEMENTS)}
    residue_indices = {name: i for i, name in enumerate(PROTEIN_RESIDUES)}
    for i, (element, residue, atom_name) in enumerate(zip(elements, residue_names, atom_names)):
        element, residue, atom_name = (str(value).strip().upper() for value in (element, residue, atom_name))
        residue = _RESIDUE_ALIASES.get(residue, residue)
        result[i, element_indices.get(element, len(PROTEIN_ELEMENTS) - 1)] = 1.
        result[i, len(PROTEIN_ELEMENTS) + residue_indices.get(residue, len(PROTEIN_RESIDUES) - 1)] = 1.
        result[i, -1] = float(residue in residue_indices and residue != "UNK" and atom_name in _BACKBONE_ATOMS)
    return result


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
    # PDB lacks typed ligand bonds; never pass it to the SDF reader or infer bond orders.
    uff_sdfs = sorted(pocket_dir.glob("*_uff*.sdf"))
    if uff_sdfs:
        return uff_sdfs[0]
    lig_sdfs = sorted(set(pocket_dir.glob("*_lig.sdf")) | set(pocket_dir.glob("*_ligand.sdf")))
    if lig_sdfs:
        return lig_sdfs[0]
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


def _pdb_element(line: str, residue: str) -> str:
    explicit = line[76:78].strip().upper()
    if explicit:
        return explicit
    name_field = line[12:16]
    name = name_field.strip().upper()
    letters = "".join(character for character in name if character.isalpha())
    if not letters:
        return "UNK"
    if residue == "MSE" and letters == "SE":
        return "SE"
    # A protein CA is alpha carbon; left-aligned HETATM CA is a calcium ion.
    # Leading digits in names such as 1HG1 indicate hydrogens, not mercury.
    if (name_field[0].isspace() or name_field[0].isdigit()
            or residue in PROTEIN_RESIDUES[:-1] or residue in _RESIDUE_ALIASES):
        return letters[0]
    two_letter_elements = frozenset({
        "HE", "LI", "BE", "NE", "NA", "MG", "AL", "SI", "CL", "AR", "CA", "SC", "TI", "CR",
        "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR",
        "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN", "SB", "TE", "XE",
        "CS", "BA", "LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER",
        "TM", "YB", "LU", "HF", "TA", "RE", "OS", "IR", "PT", "AU", "HG", "TL", "PB", "BI",
        "PO", "AT", "RN", "FR", "RA", "AC", "TH", "PA", "NP", "PU", "AM", "CM", "BK", "CF",
        "ES", "FM", "MD", "NO", "LR", "RF", "DB", "SG", "BH", "HS", "MT", "DS", "RG", "CN",
        "NH", "FL", "MC", "LV", "TS", "OG",
    })
    return letters[:2] if letters[:2] in two_letter_elements else letters[0]


def read_pdb_heavy_atoms(pdb_path) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Read the first PDB model; choose altloc by highest occupancy, then blank/A/lexical code.
    Omit water, H/D/T, zero occupancy, and invalid coordinates; keep cofactors and metals.
    Return float64 coordinates, actual elements (not UNK), residue names, and atom names.
    """
    selected: dict[tuple[str, ...], tuple[tuple[float, int, str], list[float], str, str, str]] = {}
    in_first_model = False
    with Path(pdb_path).open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            record = line[0:6]
            if record == "MODEL ":
                if in_first_model:
                    break
                in_first_model = True
                continue
            if record == "ENDMDL":
                break
            if record not in ("ATOM  ", "HETATM") or len(line) < 54:
                continue
            residue = line[17:20].strip().upper()
            if residue in _WATER:
                continue
            try:
                xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                occupancy = float(line[54:60].strip() or "1")
            except ValueError:
                continue
            if not np.isfinite(xyz).all() or not np.isfinite(occupancy) or occupancy <= 0.:
                continue
            element = _pdb_element(line, residue)
            if element in ("H", "D", "T"):
                continue
            atom_name = line[12:16].strip().upper()
            alt = line[16:17].strip()
            priority = (-occupancy, 0 if not alt else 1 if alt == "A" else 2, alt)
            key = (record, line[21:22], line[22:26], line[26:27], residue, atom_name)
            if key not in selected or priority < selected[key][0]:
                selected[key] = (priority, xyz, element, residue, atom_name)
    if not selected:
        return np.zeros((0, 3), dtype=np.float64), (), (), ()
    rows = list(selected.values())
    return (np.asarray([row[1] for row in rows], dtype=np.float64),
            tuple(row[2] for row in rows), tuple(row[3] for row in rows), tuple(row[4] for row in rows))


def pocket_atoms(pdb_path, max_atoms=None, seed=0) -> tuple[np.ndarray, np.ndarray]:
    """Read a typed heavy-atom pocket with shared filtering/subsampling indices."""
    if max_atoms is not None and (int(max_atoms) != max_atoms or max_atoms < 1):
        raise ValueError("max_atoms must be a positive integer or None")
    coords, elements, residues, atom_names = read_pdb_heavy_atoms(pdb_path)
    coords = coords.astype(np.float32)
    features = protein_atom_features(elements, residues, atom_names)
    if max_atoms is not None and len(coords) > max_atoms:
        indices = np.random.default_rng(seed).choice(len(coords), size=int(max_atoms), replace=False)
        coords, features = coords[indices], features[indices]
    return coords, features


def pocket_coords(pdb_path, max_atoms=None, seed=0):
    return pocket_atoms(pdb_path, max_atoms=max_atoms, seed=seed)[0]


def task_vec(X: MolState, dim: int):
    """Encode rigid-motion and atom-order invariant molecular conditioning.
    Use centroid-distance mean/population std/max, descending population-covariance eigenvalues,
    feature means, and log(N + 1); the checkpoint tag identifies this representation.
    """
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive")
    coords = np.asarray(X.R, dtype=np.float64)
    features = np.asarray(X.node_feat, dtype=np.float64)
    n = X.num_nodes()
    if (n < 1 or coords.shape != (n, 3) or features.ndim != 2 or features.shape[0] != n
            or not np.isfinite(coords).all() or not np.isfinite(features).all()):
        raise ValueError("task_vec requires finite nonempty coordinates and matching node features")
    centered = coords - coords.mean(axis=0)
    radii = np.linalg.norm(centered, axis=1)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / n)[::-1].clip(min=0.0)
    chunks = [
        np.asarray([radii.mean(), radii.std(), radii.max()], dtype=np.float32),
        np.asarray(eigenvalues, dtype=np.float32),
        np.asarray(features.mean(axis=0), dtype=np.float32),
        np.asarray([np.log(float(n) + 1.0)], dtype=np.float32),
    ]
    flat = np.concatenate([p.ravel() for p in chunks])
    task = np.zeros(dim, dtype=np.float32)
    n = min(dim, int(flat.shape[0]))
    task[:n] = flat[:n]
    return task
