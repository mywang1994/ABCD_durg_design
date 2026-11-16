"""CrossDocked splits match Pocket2Mol/TargetDiff filename pairs before filtering.
Processed LMDB row IDs are not file indices. Raw PDB/SDF poses share a coordinate frame.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from dataset.protein_data import CrossDockComplex

MANIFEST_SCHEMA = "abcd_crossdock_files_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprints(spec: CrossDockComplex) -> dict:
    return {
        field: {"sha256": _sha256(path), "size": path.stat().st_size}
        for field, path in (("receptor_pdb", spec.receptor_pdb),
                            ("ligand_sdf", spec.reference_ligand_sdf))
    }


def _relative_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Split filenames must be nonempty strings, not processed LMDB integer IDs")
    # CrossDocked names are POSIX-relative even when a manifest is read on Windows.
    name = value.replace("\\", "/")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or ":" in name or str(path) == ".":
        raise ValueError(f"Expected a relative dataset filename without '..': {value!r}")
    return str(path)


def _validate_named_splits(data: Any) -> dict[str, list[tuple[str, str]]]:
    if not isinstance(data, dict) or not data:
        raise ValueError("A filename split must be a nonempty mapping of split names to filename pairs")
    parsed: dict[str, list[tuple[str, str]]] = {}
    owner: dict[tuple[str, str], str] = {}
    for part, rows in data.items():
        if not isinstance(part, str) or not isinstance(rows, (list, tuple)):
            raise ValueError("Split names must be strings and each split must contain a list of filename pairs")
        pairs = []
        seen = set()
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise ValueError("Expected (pocket_filename, ligand_filename) pairs; integer LMDB splits cannot be reused")
            pair = (_relative_name(row[0]), _relative_name(row[1]))
            if not pair[0].lower().endswith(".pdb") or not pair[1].lower().endswith((".sdf", ".sd", ".sdf.gz", ".sd.gz")):
                raise ValueError(f"Split entry must identify a pocket PDB and typed ligand SDF: {pair}")
            if pair in seen:
                raise ValueError(f"Duplicate filename pair in {part!r}: {pair}")
            if pair in owner:
                raise ValueError(f"Train/test split overlap: the same pair appears in {owner[pair]!r} and {part!r}: {pair}")
            owner[pair] = part
            seen.add(pair)
            pairs.append(pair)
        parsed[part] = pairs
    return parsed


def read_crossdock_named_split(split_file: Path | str) -> dict[str, list[tuple[str, str]]]:
    """Read the public filename split without unpickling arbitrary Python classes."""
    path = Path(split_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CrossDocked filename split does not exist: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        import torch

        # Official name pairs need only the restricted loader; never fall back to arbitrary unpickling.
        data = torch.load(path, map_location="cpu", weights_only=True)
    return _validate_named_splits(data)


def _complex(root: Path, pair: tuple[str, str]) -> CrossDockComplex:
    pocket, ligand = pair
    identifier = hashlib.sha256((pocket + "\0" + ligand).encode()).hexdigest()[:12]
    # Keep relative names for IDs and metadata; resolve symlinks for file access.
    return CrossDockComplex(
        complex_id=f"{Path(ligand).stem}__{identifier}",
        receptor_pdb=(root / pocket).resolve(),
        reference_ligand_sdf=(root / ligand).resolve(),
        pocket_fn_rel=pocket,
        ligand_fn_rel=ligand,
    )


def _inventory(root: Path, splits: dict[str, list[tuple[str, str]]]):
    availability = {}
    available = {}
    # Resolved pairs expose train/test overlap through symlinks.
    resolved_owner = {}
    for part, pairs in splits.items():
        kept = []
        missing = []
        aliases = 0
        seen_resolved = set()
        for pair in pairs:
            rec, lig = root / pair[0], root / pair[1]
            missing_fields = []
            if not rec.is_file():
                missing_fields.append("receptor_pdb")
            if not lig.is_file():
                missing_fields.append("ligand_sdf")
            if missing_fields:
                missing.append({"pocket_filename": pair[0], "ligand_filename": pair[1], "missing": missing_fields})
                continue
            resolved = (str(rec.resolve()), str(lig.resolve()))
            if resolved in resolved_owner and resolved_owner[resolved] != part:
                raise ValueError(f"Train/test split overlap through file aliases: {pair} also belongs to {resolved_owner[resolved]!r}")
            resolved_owner[resolved] = part
            if resolved in seen_resolved:
                aliases += 1
            seen_resolved.add(resolved)
            kept.append(pair)
        available[part] = kept
        availability[part] = {
            "requested": len(pairs), "available": len(kept), "missing_count": len(missing),
            "same_split_file_aliases": aliases, "missing_examples": missing[:20],
        }
    return available, availability


def load_crossdock_split(
    pocket10_root: Path | str,
    split_file: Path | str,
    *,
    split: str = "train",
    allow_missing: bool = False,
) -> tuple[list[CrossDockComplex], dict]:
    """Select raw pairs by official filenames, never by another LMDB's row order."""
    root = Path(pocket10_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CrossDocked pocket directory does not exist: {root}")
    source = Path(split_file).expanduser().resolve()
    named = read_crossdock_named_split(source)
    if split not in named:
        raise ValueError(f"Split {split!r} is absent; available names: {sorted(named)}")
    available, availability = _inventory(root, named)
    selected = availability[split]
    if selected["missing_count"] and not allow_missing:
        raise FileNotFoundError(
            f"CrossDocked split {split!r}: {selected['missing_count']} of {selected['requested']} pairs missing. "
            f"First: {selected['missing_examples'][0]}. Use allow_missing explicitly only when documenting exclusions."
        )
    if not available[split]:
        raise ValueError(f"CrossDocked split {split!r} has no available file pairs; no other split is substituted")
    report = {
        "source": "filename_split", "dataset_root": str(root), "split_file": str(source),
        "split_sha256": _sha256(source), "split": split, "availability": availability,
        "selected_count": len(available[split]), "allow_missing": bool(allow_missing),
        "structure_validation": "not_run", "subset": False,
    }
    return [_complex(root, pair) for pair in available[split]], report


def validate_crossdock_complex(
    spec: CrossDockComplex,
    *,
    max_nodes: int = 128,
    min_pocket_atoms: int = 4,
) -> dict:
    """Validate a training target without embedding, recentering, or redocking it."""
    import numpy as np
    from rdkit import Chem

    from dataset.pool_core import load_reference_mol_from_sdf, mol_to_molstate
    from dataset.protein_data import read_pdb_heavy_atoms
    from dataset.sampling import is_valid
    from utils.chemistry import atom_targets, bond_targets

    if max_nodes < 1 or min_pocket_atoms < 1:
        raise ValueError("max_nodes and min_pocket_atoms must be positive")
    mol = load_reference_mol_from_sdf(spec.reference_ligand_sdf, require_3d=True)
    # Validate heavy atoms without embedding a missing reference pose.
    if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        raise ValueError("reference_contains_explicit_hydrogen_nodes")
    n = mol.GetNumAtoms()
    if not 1 <= n <= max_nodes:
        raise ValueError(f"ligand_heavy_atom_count={n} exceeds supported range [1,{max_nodes}]")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("disconnected_reference_ligand")
    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    if coords.shape != (n, 3) or not np.isfinite(coords).all():
        raise ValueError("nonfinite_reference_coordinates")
    if n > 1 and np.max(np.linalg.norm(coords - coords.mean(axis=0), axis=1)) < 1e-6:
        raise ValueError("collapsed_reference_coordinates")
    state = mol_to_molstate(mol)
    if not is_valid(state):
        raise ValueError("invalid_reference_molecular_state")
    atom_targets(state)
    bond_targets(state)
    pocket, _, _, _ = read_pdb_heavy_atoms(spec.receptor_pdb)
    if len(pocket) < min_pocket_atoms:
        raise ValueError(f"pocket_heavy_atom_count={len(pocket)} below minimum {min_pocket_atoms}")
    return {
        "ligand_heavy_atoms": n, "pocket_heavy_atoms": len(pocket),
        "coordinate_frame": "unchanged_source_PDB_SDF", "reference_3d": True,
    }


def prepare_crossdock_manifest(
    pocket10_root: Path | str,
    split_file: Path | str,
    output: Path | str,
    *,
    split: str = "train",
    allow_missing: bool = False,
    validate_structures: bool = True,
    max_complexes: int | None = None,
    max_nodes: int = 128,
    min_pocket_atoms: int = 4,
) -> dict:
    """Audit the source split and write accepted/rejected records without editing data."""
    destination = Path(output).expanduser().resolve()
    source_split = Path(split_file).expanduser().resolve()
    if destination == source_split:
        raise ValueError("The output manifest must not overwrite a source structure or split file")
    if destination.suffix.lower() != ".json":
        raise ValueError("The output manifest must be a .json file; raw structures, databases and split files must not be overwritten")
    specs, report = load_crossdock_split(pocket10_root, split_file, split=split, allow_missing=allow_missing)
    all_named = read_crossdock_named_split(source_split)
    source_root = Path(report["dataset_root"])
    protected = {source_split, *((source_root / name).resolve()
                                for pairs in all_named.values() for pair in pairs for name in pair)}
    if destination in protected:
        raise ValueError("The output manifest must not overwrite any source split structure, including held-out files")
    if max_complexes is not None:
        if max_complexes < 1:
            raise ValueError("max_complexes must be positive")
        selected = specs[:max_complexes]
    else:
        selected = specs
    records, rejected = [], []
    validation_started = time.monotonic()
    for item_index, spec in enumerate(selected, start=1):
        record = {
            "complex_id": spec.complex_id, "split": split,
            "receptor_pdb": spec.pocket_fn_rel, "ligand_sdf": spec.ligand_fn_rel,
        }
        fingerprints = _source_fingerprints(spec)
        record["source_files"] = fingerprints
        if validate_structures:
            try:
                record["validation"] = validate_crossdock_complex(
                    spec, max_nodes=max_nodes, min_pocket_atoms=min_pocket_atoms,
                )
            except (ValueError, RuntimeError, OSError, IndexError, OverflowError) as exc:
                rejected.append({**record, "reason": f"{type(exc).__name__}: {exc}"})
            else:
                if _source_fingerprints(spec) != fingerprints:
                    raise RuntimeError(f"Source structures changed during validation: {spec.complex_id}; prepare again")
                records.append(record)
        else:
            records.append(record)
        if item_index % 1000 == 0:
            print(f"Prepared {item_index}/{len(selected)}: accepted={len(records)}, "
                  f"rejected={len(rejected)}, elapsed={time.monotonic() - validation_started:.1f}s",
                  file=sys.stderr, flush=True)
    report.update({
        "source_available_count": len(specs), "inspected_count": len(selected),
        "selected_count": len(records), "rejected_count": len(rejected),
        "subset": len(selected) < len(specs), "selection_order": "first_N_in_source_split_order",
        "structure_validation": "complete_for_inspected_subset" if validate_structures else "not_run",
        "max_nodes": max_nodes, "min_pocket_atoms": min_pocket_atoms,
        "validation_seconds": round(time.monotonic() - validation_started, 3),
        "rejection_reasons": dict(Counter(item["reason"] for item in rejected)),
    })
    manifest = {
        "schema": MANIFEST_SCHEMA, "dataset_root": report["dataset_root"],
        "split_file": report["split_file"], "split_sha256": report["split_sha256"],
        "prepared_split": split, "records": records, "rejected": rejected, "report": report,
    }
    if _sha256(source_split) != report["split_sha256"]:
        raise RuntimeError("Source filename split changed during preparation; prepare again")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def load_crossdock_manifest(
    manifest_path: Path | str,
    *,
    split: str = "train",
    dataset_root: Path | str | None = None,
    split_file: Path | str | None = None,
) -> tuple[list[CrossDockComplex], dict]:
    """Read exactly the prepared subset; never silently replace it with directory discovery."""
    path = Path(manifest_path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Expected {MANIFEST_SCHEMA} manifest: {path}")
    if data.get("prepared_split") != split:
        raise ValueError(f"Manifest was prepared for {data.get('prepared_split')!r}, not requested split {split!r}")
    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest contains no accepted training pairs")
    root = Path(dataset_root if dataset_root is not None else data["dataset_root"]).expanduser().resolve()
    named = {split: []}
    for record in records:
        if not isinstance(record, dict) or record.get("split") != split:
            raise ValueError("A manifest record belongs to a different split")
        named[split].append((record.get("receptor_pdb"), record.get("ligand_sdf")))
    parsed = _validate_named_splits(named)
    source_split = Path(split_file if split_file is not None else data.get("split_file", "")).expanduser().resolve()
    if not source_split.is_file():
        raise FileNotFoundError("The manifest's source filename split is unavailable; provide split_file with the same SHA256 or prepare again")
    if _sha256(source_split) != data.get("split_sha256"):
        raise ValueError("Source filename split changed since manifest preparation; prepare again")
    source_names = read_crossdock_named_split(source_split)
    allowed = set(source_names.get(split, ()))
    if any(pair not in allowed for pair in parsed[split]):
        raise ValueError("Prepared manifest includes pairs outside its declared source split")
    # Recheck every source partition for retargeted symlinks after preparation or relocation.
    _, source_availability = _inventory(root, source_names)
    available, availability = _inventory(root, parsed)
    if availability[split]["missing_count"]:
        raise FileNotFoundError(f"Prepared manifest structures are missing: {availability[split]['missing_examples'][:3]}")
    specs = [_complex(root, pair) for pair in available[split]]
    for record, spec in zip(records, specs):
        fingerprint = record.get("source_files")
        if not isinstance(fingerprint, dict) or set(fingerprint) != {"receptor_pdb", "ligand_sdf"}:
            raise ValueError("Manifest has no source-structure fingerprints; prepare again with the current script")
        if _source_fingerprints(spec) != fingerprint:
            raise ValueError(f"Source structure content changed since preparation: {spec.complex_id}; prepare again")
    report = dict(data.get("report", {}))
    report.update({
        "source": "prepared_manifest", "manifest_path": str(path), "manifest_sha256": _sha256(path),
        "dataset_root": str(root), "split": split, "selected_count": len(available[split]),
        "manifest_file_validation": availability[split],
        "source_split_rechecked": True, "source_structure_hashes_rechecked": True,
        "resolved_split_file": str(source_split), "source_availability_rechecked": source_availability,
    })
    return specs, report
