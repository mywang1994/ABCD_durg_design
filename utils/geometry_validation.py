"""Report molecular geometry without changing coordinates or stereochemistry."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom


GEOMETRY_VALIDATION_SCHEMA = "rdkit_bounds_geometry_v1"
_CHECK_NAMES = ("coordinates", "bond_lengths", "bond_angles", "internal_clashes",
                "aromatic_ring_planarity", "double_bond_planarity", "receptor_clashes")


@dataclass(frozen=True)
class GeometryValidationConfig:
    bond_tolerance: float = .25
    angle_tolerance: float = .25
    clash_tolerance: float = .30
    planar_tolerance_angstrom: float = .25
    receptor_clash_fraction: float = .75

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("bond_tolerance", "angle_tolerance", "clash_tolerance"):
            if getattr(self, name) >= 1:
                raise ValueError(f"{name} must be less than one")
        if not 0 < self.receptor_clash_fraction <= 1:
            raise ValueError("receptor_clash_fraction must be in (0, 1]")


def _check(status, *, count=0, failures=(), failed_count=None, reason=None):
    failures = list(failures)
    failed_count = len(failures) if failed_count is None else int(failed_count)
    return {"status": status, "passed": status == "passed" if status in ("passed", "failed", "error") else None,
            "checked_count": int(count), "failed_count": failed_count,
            "failures": failures[:20], "failures_truncated": max(0, failed_count - min(len(failures), 20)),
            "reason": reason}


def not_evaluated_geometry(reason, *, config=None, chemical_valid=None):
    cfg = GeometryValidationConfig() if config is None else config
    return {"schema": GEOMETRY_VALIDATION_SCHEMA, "chemical_valid": chemical_valid, "geometry_valid": None,
            "intramolecular_valid": None, "receptor_valid": None, "status": "not_evaluated",
            "thresholds": asdict(cfg), "checks": {name: _check("not_evaluated", reason=reason) for name in _CHECK_NAMES},
            "failure_reasons": [str(reason)], "receptor_requested": False,
            "scope": "Heavy-atom geometry; no energy, stereo identity, metal coordination or coordinate repair."}


def summarize_geometry(reports, attempted=None):
    reports = list(reports)
    attempted = len(reports) if attempted is None else attempted
    if isinstance(attempted, bool) or not isinstance(attempted, int) or attempted < len(reports):
        raise ValueError("attempted must be an integer at least as large as the report count")
    counts = {"schema": GEOMETRY_VALIDATION_SCHEMA, "attempted": attempted, "rate_denominator": "all_attempts"}
    for name in ("chemical", "geometry", "intramolecular", "receptor"):
        valid = sum(report.get(name + "_valid") is True for report in reports)
        invalid = sum(report.get(name + "_valid") is False for report in reports)
        counts.update({name + "_valid": valid, name + "_invalid": invalid,
                       name + "_evaluated": valid + invalid, name + "_not_evaluated": attempted - valid - invalid})
    counts["geometry_valid_rate"] = counts["geometry_valid"] / attempted if attempted else None
    evaluated = counts["geometry_evaluated"]
    counts["geometry_valid_rate_among_evaluated"] = counts["geometry_valid"] / evaluated if evaluated else None
    failures = Counter(name for report in reports for name, check in report.get("checks", {}).items()
                       if check.get("status") in ("failed", "error"))
    counts["check_failures"] = dict(sorted(failures.items()))
    return counts


def _pair_check(pairs, distances, bounds, tolerance, *, lower_only=False):
    if not pairs:
        return _check("not_applicable", reason="no_relevant_atom_pairs")
    failures = []
    for i, j in pairs:
        lower = float(bounds[j, i]) * (1 - tolerance)
        upper = None if lower_only else float(bounds[i, j]) * (1 + tolerance)
        value = float(distances[i, j])
        if value < lower or (upper is not None and value > upper):
            failures.append({"atoms": [i, j], "distance_angstrom": value,
                             "lower_bound_angstrom": lower, "upper_bound_angstrom": upper,
                             "reason": "too_short" if value < lower else "too_long"})
    return _check("failed" if failures else "passed", count=len(pairs), failures=failures)


def _plane_check(groups, positions, threshold):
    if not groups:
        return _check("not_applicable", reason="no_relevant_planar_groups")
    failures = []
    maximum = 0.
    for group in groups:
        centered = positions[list(group)] - positions[list(group)].mean(0)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        deviation = float(np.abs(centered @ vectors[-1]).max())
        if not math.isfinite(deviation):
            raise ValueError("Nonfinite plane fit")
        maximum = max(maximum, deviation)
        if deviation > threshold:
            failures.append({"atoms": list(group), "maximum_deviation_angstrom": deviation,
                             "reason": "not_planar"})
    result = _check("failed" if failures else "passed", count=len(groups), failures=failures)
    result["maximum_deviation_angstrom"] = maximum
    return result


def load_receptor_geometry(pdb_path):
    """Read heavy atoms with the repository's first-model/altloc policy."""
    from dataset.protein_data import read_pdb_heavy_atoms

    positions, elements, _, _ = read_pdb_heavy_atoms(pdb_path)
    periodic = Chem.GetPeriodicTable()
    element_numbers = {periodic.GetElementSymbol(z).upper(): z for z in range(1, 119)}
    numbers = np.asarray([element_numbers.get(element.upper(), 0) for element in elements], dtype=np.int64)
    if not len(positions):
        raise ValueError("Receptor has no usable heavy atoms")
    return positions, numbers


def _receptor_check(positions, atom_numbers, receptor_positions, receptor_numbers, threshold):
    receptor_positions = np.asarray(receptor_positions, dtype=np.float64)
    receptor_numbers = np.asarray(receptor_numbers)
    if (receptor_positions.ndim != 2 or receptor_positions.shape[1:] != (3,)
            or receptor_numbers.shape != (len(receptor_positions),) or not len(receptor_positions)
            or not np.isfinite(receptor_positions).all() or not np.isfinite(receptor_numbers).all()
            or not np.equal(receptor_numbers, np.floor(receptor_numbers)).all()
            or (receptor_numbers < 1).any() or (receptor_numbers > 118).any()):
        raise ValueError("Receptor requires finite (N,3) positions and matching atomic numbers in [1,118]")
    receptor_numbers = receptor_numbers.astype(np.int64)
    keep = receptor_numbers > 1
    receptor_positions, receptor_numbers = receptor_positions[keep], receptor_numbers[keep]
    if not len(receptor_positions):
        raise ValueError("Receptor has no heavy atoms")
    periodic = Chem.GetPeriodicTable()
    ligand_radii = np.asarray([periodic.GetRvdw(int(z)) for z in atom_numbers])
    receptor_radii = np.asarray([periodic.GetRvdw(int(z)) for z in receptor_numbers])
    failures, failed_count, minimum = [], 0, math.inf
    count = len(positions) * len(receptor_positions)
    receptor_ids = np.flatnonzero(keep)
    for start in range(0, len(receptor_positions), 4096):
        xyz = receptor_positions[start:start + 4096]
        distances = np.linalg.norm(positions[:, None] - xyz[None], axis=-1)
        limits = threshold * (ligand_radii[:, None] + receptor_radii[None, start:start + 4096])
        if not np.isfinite(distances).all():
            raise ValueError("Nonfinite receptor distances")
        minimum = min(minimum, float(distances.min()))
        overlap = distances < limits
        failed_count += int(np.count_nonzero(overlap))
        indices = np.argwhere(overlap)[:20 - len(failures)] if len(failures) < 20 else ()
        for i, j in indices:
            failures.append({"ligand_heavy_atom": int(i), "receptor_atom": int(receptor_ids[start + j]),
                             "distance_angstrom": float(distances[i, j]),
                             "lower_bound_angstrom": float(limits[i, j]), "reason": "vdw_overlap"})
    result = _check("failed" if failed_count else "passed", count=count, failures=failures, failed_count=failed_count)
    result["minimum_distance_angstrom"] = minimum
    return result


def evaluate_geometry(mol, *, receptor_positions=None, receptor_atomic_numbers=None, config=None):
    """Check a copy of an RDKit molecule; unavailable checks never count as passes."""
    cfg = GeometryValidationConfig() if config is None else config
    result = not_evaluated_geometry("chemical_validation_required", config=cfg, chemical_valid=False)
    checks = result["checks"]
    receptor_requested = receptor_positions is not None or receptor_atomic_numbers is not None
    result.update(receptor_requested=receptor_requested, failure_reasons=[])
    try:
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError("missing_or_empty_molecule")
        molecule = Chem.Mol(mol)
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_CLEANUP)
        if len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError("disconnected_molecule")
        if any(not 1 <= atom.GetAtomicNum() <= 118 for atom in molecule.GetAtoms()):
            raise ValueError("unsupported_atomic_number")
    except (ValueError, RuntimeError, TypeError, AttributeError) as error:
        result["failure_reasons"] = [f"chemical_validation: {error}"]
        return result
    result["chemical_valid"] = True
    try:
        if molecule.GetNumConformers() == 0:
            raise ValueError("missing_conformer")
        if not molecule.GetConformer().Is3D():
            raise ValueError("conformer_not_marked_3d")
        positions = np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float64)
        if positions.shape != (molecule.GetNumAtoms(), 3) or not np.isfinite(positions).all():
            raise ValueError("nonfinite_or_malformed_coordinates")
    except (ValueError, RuntimeError) as error:
        checks["coordinates"] = _check("failed", failures=[{"reason": str(error)}])
        result.update(status="failed", geometry_valid=False, intramolecular_valid=False,
                      failure_reasons=[f"coordinates: {error}"])
        return result
    checks["coordinates"] = _check("passed", count=molecule.GetNumAtoms())
    heavy = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    if not heavy:
        result["failure_reasons"] = ["no_heavy_atoms_to_evaluate"]
        return result
    names = ("bond_lengths", "bond_angles", "internal_clashes")
    try:
        bounds = rdDistGeom.GetMoleculeBoundsMatrix(molecule, set15bounds=True, scaleVDW=True,
                                                   doTriangleSmoothing=True, useMacrocycle14config=False)
        topology = Chem.GetDistanceMatrix(molecule)
        with np.errstate(over="ignore", invalid="ignore"):
            distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
        if not np.isfinite(bounds).all() or not np.isfinite(distances).all():
            raise ValueError("nonfinite_distance_bounds")
        pairs = [(i, j) for index, i in enumerate(heavy) for j in heavy[index + 1:]]
        for name, length, tolerance in zip(names, (1, 2, None),
                                           (cfg.bond_tolerance, cfg.angle_tolerance, cfg.clash_tolerance)):
            selected = [(i, j) for i, j in pairs if topology[i, j] == length] if length else [
                (i, j) for i, j in pairs if topology[i, j] >= 3]
            checks[name] = _pair_check(selected, distances, bounds, tolerance, lower_only=length is None)
    except (ValueError, RuntimeError) as error:
        for name in names:
            checks[name] = _check("error", reason=str(error))
    aromatic = {tuple(sorted(ring)) for ring in molecule.GetRingInfo().AtomRings()
                if len(ring) in (5, 6) and all(molecule.GetAtomWithIdx(i).GetIsAromatic()
                    and molecule.GetAtomWithIdx(i).GetHybridization() == Chem.HybridizationType.SP2 for i in ring)}
    double = set()
    for bond in molecule.GetBonds():
        ends = (bond.GetBeginAtom(), bond.GetEndAtom())
        if (bond.GetBondType() == Chem.BondType.DOUBLE and not bond.GetIsAromatic()
                and all(atom.GetHybridization() == Chem.HybridizationType.SP2 for atom in ends)):
            group = {atom.GetIdx() for atom in ends}
            group.update(neighbor.GetIdx() for atom in ends for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() > 1)
            if len(group) >= 4:
                double.add(tuple(sorted(group)))
    for name, groups in (("aromatic_ring_planarity", aromatic), ("double_bond_planarity", double)):
        try:
            checks[name] = _plane_check(sorted(groups), positions, cfg.planar_tolerance_angstrom)
        except (ValueError, np.linalg.LinAlgError) as error:
            checks[name] = _check("error", reason=str(error))
    checks["aromatic_ring_planarity"]["scope"] = "Five- and six-membered sp2 aromatic rings"
    if receptor_requested:
        try:
            numbers = [molecule.GetAtomWithIdx(i).GetAtomicNum() for i in heavy]
            checks["receptor_clashes"] = _receptor_check(positions[heavy], numbers, receptor_positions,
                receptor_atomic_numbers, cfg.receptor_clash_fraction)
        except (ValueError, RuntimeError, TypeError) as error:
            checks["receptor_clashes"] = _check("error", reason=str(error))
    else:
        checks["receptor_clashes"] = _check("not_requested", reason="receptor_not_provided")
    failed = [name for name, check in checks.items() if check["status"] in ("failed", "error")]
    result["intramolecular_valid"] = not any(name != "receptor_clashes" for name in failed)
    result["receptor_valid"] = checks["receptor_clashes"]["passed"] if receptor_requested else None
    result.update(status="failed" if failed else "passed", geometry_valid=not failed,
                  failure_reasons=[f"{name}: {checks[name]['reason'] or str(checks[name]['failed_count']) + ' violations'}" for name in failed])
    return result
