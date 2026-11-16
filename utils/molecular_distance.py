from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Integral, Real

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Geometry import Point3D


@dataclass(frozen=True)
class MolecularDistanceConfig:

    w_topo: float = 0.25
    w_scaffold: float = 0.25
    w_pharm: float = 0.25
    w_geom: float = 0.25
    fp_radius: int = 2
    fp_nbits: int = 2048

    def __post_init__(self):
        weights = (self.w_topo, self.w_scaffold, self.w_pharm, self.w_geom)
        if any(not isinstance(w, Real) or not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Molecular-distance weights must be finite and nonnegative")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("Molecular-distance weights must sum to one")
        if not isinstance(self.fp_radius, Integral) or isinstance(self.fp_radius, bool) or self.fp_radius < 0:
            raise ValueError("fp_radius must be a nonnegative integer")
        if not isinstance(self.fp_nbits, Integral) or isinstance(self.fp_nbits, bool) or self.fp_nbits < 1:
            raise ValueError("fp_nbits must be a positive integer")


def _validated_mol(value):
    try:
        if isinstance(value, Chem.Mol):
            mol = Chem.Mol(value)
        else:
            from utils.distances import molstate_to_mol

            n = value.num_nodes()
            coords = np.asarray(value.R)
            if n < 1 or coords.shape != (n, 3) or not np.isfinite(coords).all():
                raise ValueError("MolState must have finite coordinates of shape (N, 3), N > 0")
            if set(value.G.nodes()) != set(range(n)):
                raise ValueError("MolState graph nodes must match coordinate indices 0..N-1")
            mol = molstate_to_mol(value)
            if mol is None:
                raise ValueError("MolState could not be converted to a molecule")
        if mol.GetNumAtoms() < 1:
            raise ValueError("Empty molecules have no molecular distance")
        if any(atom.GetAtomicNum() < 1 for atom in mol.GetAtoms()):
            raise ValueError("Dummy atoms have no supported molecular distance")
        # Reject neutral pentavalent nitrogen without changing its charge.
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_CLEANUP)
        mol = Chem.RemoveHs(mol)
        if mol.GetNumAtoms() < 1 or mol.GetNumConformers() < 1:
            raise ValueError("A nonempty heavy-atom molecule with a conformer is required")
        coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
        if coords.shape != (mol.GetNumAtoms(), 3) or not np.isfinite(coords).all():
            raise ValueError("Conformer coordinates must be finite and match the atoms")
        return mol
    except Exception as exc:
        raise ValueError(f"Cannot evaluate molecular distance: {exc}") from exc


@lru_cache(maxsize=32)
def _fingerprint_generators(radius, nbits):
    topo = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(nbits))
    pharm = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius),
        fpSize=int(nbits),
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
    )
    return topo, pharm


def _fingerprint_distance(left, right):
    return 1.0 - float(DataStructs.TanimotoSimilarity(left, right))


def _scaffold_fingerprint(mol, generator):
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return None
    return generator.GetFingerprint(scaffold)


def _scaffold_distance(left, right, generator):
    a = _scaffold_fingerprint(left, generator)
    b = _scaffold_fingerprint(right, generator)
    if a is None or b is None:
        return 0.0 if a is None and b is None else 1.0
    return _fingerprint_distance(a, b)


def _usr_descriptor(mol):
    coords = mol.GetConformer().GetPositions()
    points = [Point3D(float(x), float(y), float(z)) for x, y, z in coords]
    # GetUSR rejects one or two atoms; these lower-level functions support them.
    distributions = rdMolDescriptors.GetUSRDistributions(points)
    return rdMolDescriptors.GetUSRFromDistributions(distributions)


def _bounded_distance(value):
    if not math.isfinite(value) or value < -1e-9 or value > 1.0 + 1e-9:
        raise ValueError("A molecular-distance component is nonfinite or outside [0, 1]")
    return min(1.0, max(0.0, float(value)))


def molecular_distance_components(X, Y, config=None):
    """Components in [0, 1] for valid molecules with coordinates in angstroms.
    pharm uses 2D feature-Morgan fingerprints; geom = 1 - USR score (12 moments).
    Removable hydrogens are excluded; invalid chemistry raises an error."""
    cfg = config if config is not None else MolecularDistanceConfig()
    if not isinstance(cfg, MolecularDistanceConfig):
        raise TypeError("config must be a MolecularDistanceConfig")
    left, right = _validated_mol(X), _validated_mol(Y)
    topo, pharm = _fingerprint_generators(cfg.fp_radius, cfg.fp_nbits)
    try:
        components = {
            "topo": _fingerprint_distance(topo.GetFingerprint(left), topo.GetFingerprint(right)),
            "scaffold": _scaffold_distance(left, right, topo),
            "pharm": _fingerprint_distance(pharm.GetFingerprint(left), pharm.GetFingerprint(right)),
            "geom": 1.0 - float(rdMolDescriptors.GetUSRScore(_usr_descriptor(left), _usr_descriptor(right))),
        }
        return {name: _bounded_distance(value) for name, value in components.items()}
    except Exception as exc:
        raise ValueError(f"Cannot calculate molecular-distance components: {exc}") from exc


def molecular_distance(X, Y, config=None):
    """Return the configured weighted sum in [0, 1], or raise on invalid input."""
    cfg = config if config is not None else MolecularDistanceConfig()
    parts = molecular_distance_components(X, Y, cfg)
    return _bounded_distance(
        cfg.w_topo * parts["topo"]
        + cfg.w_scaffold * parts["scaffold"]
        + cfg.w_pharm * parts["pharm"]
        + cfg.w_geom * parts["geom"]
    )
