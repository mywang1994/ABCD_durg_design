from __future__ import annotations

from functools import lru_cache
from itertools import product
import math
from numbers import Integral

from rdkit import Chem
from rdkit.Chem import rdFMCS


_MAX_SEARCH_ATOMS = 256
_MAX_MATCHES = 64


def _canonical_chemistry(value):
    """Normalize removable H; retain isotopes and stereo; ignore atom maps and pose."""
    try:
        if isinstance(value, Chem.Mol):
            mol = Chem.Mol(value)
            # Reject neutral pentavalent nitrogen without changing its charge.
            flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_CLEANUP
            Chem.SanitizeMol(mol, sanitizeOps=flags)
        else:
            # Lazy import avoids a cycle with utils.distances.
            from utils.distances import molstate_to_mol

            mol = molstate_to_mol(value)
            if mol is None:
                return None
        if not mol.GetNumAtoms() or any(a.GetAtomicNum() < 1 for a in mol.GetAtoms()):
            return None
        mol = Chem.RemoveHs(mol)
        if not mol.GetNumAtoms():
            return None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except (ValueError, RuntimeError, TypeError, AttributeError, KeyError, OverflowError):
        return None


def _atom_label(atom):
    # A tuple substitution costs one. CIP labels avoid neighbor-order ambiguity.
    return (
        atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsotope(),
        atom.GetNumExplicitHs(), atom.GetNumRadicalElectrons(),
        atom.GetIsAromatic(),
        atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "",
        int(atom.GetChiralTag()) if not atom.HasProp("_CIPCode") else 0,
    )


def _bond_label(bond):
    return str(bond.GetBondType()), int(bond.GetStereo())


def _edges(mol):
    return {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))): _bond_label(bond)
        for bond in mol.GetBonds()
    }


def _mapping_cost(left, right, atom_labels_left, atom_labels_right,
                  edges_left, edges_right, mapping):
    cost = left.GetNumAtoms() + right.GetNumAtoms() - 2 * len(mapping)
    cost += sum(atom_labels_left[i] != atom_labels_right[j] for i, j in mapping.items())
    accounted_right = set()
    for (i, j), label in edges_left.items():
        if i not in mapping or j not in mapping:
            cost += 1  # Atom deletion does not include incident edge costs.
            continue
        target_edge = tuple(sorted((mapping[i], mapping[j])))
        if target_edge not in edges_right:
            cost += 1
        else:
            accounted_right.add(target_edge)
            cost += label != edges_right[target_edge]
    cost += len(edges_right) - len(accounted_right)
    return int(cost)


@lru_cache(maxsize=4096)
def _distance_from_smiles(left_smi, right_smi, timeout, max_matches):
    left, right = Chem.MolFromSmiles(left_smi), Chem.MolFromSmiles(right_smi)
    if left is None or right is None:
        return math.inf
    if left_smi == right_smi:
        return 0
    # Delete-all/insert-all bounds the cost even when MCS search fails.
    best = left.GetNumAtoms() + right.GetNumAtoms() + left.GetNumBonds() + right.GetNumBonds()
    if max(left.GetNumAtoms(), right.GetNumAtoms()) > _MAX_SEARCH_ATOMS:
        return best
    try:
        result = rdFMCS.FindMCS(
            [left, right], maximizeBonds=False, timeout=timeout,
            atomCompare=rdFMCS.AtomCompare.CompareAny,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            matchValences=False, matchChiralTag=False,
            ringMatchesRingOnly=False, completeRingsOnly=False,
        )
        # A timed-out MCS may still provide a partial correspondence.
        query = Chem.MolFromSmarts(result.smartsString) if result.smartsString else None
        if query is None or query.GetNumAtoms() == 0:
            return best
        matches_left = left.GetSubstructMatches(query, uniquify=False, useChirality=False, maxMatches=max_matches)
        matches_right = right.GetSubstructMatches(query, uniquify=False, useChirality=False, maxMatches=max_matches)
        atoms_left = [_atom_label(a) for a in left.GetAtoms()]
        atoms_right = [_atom_label(a) for a in right.GetAtoms()]
        edges_left, edges_right = _edges(left), _edges(right)
        for a, b in product(matches_left, matches_right):
            best = min(best, _mapping_cost(
                left, right, atoms_left, atoms_right, edges_left, edges_right, dict(zip(a, b)),
            ))
    except (ValueError, RuntimeError, TypeError, OverflowError):
        pass
    return int(best)


def molecular_edit_distance(X, Y, timeout=1, max_matches=16):
    """Symmetric edit-script upper bound; invalid inputs return infinity.
    Timeout is in seconds; at most max_matches correspondences per input are used.
    Above 256 atoms, use the delete-all/insert-all bound."""
    if isinstance(timeout, bool) or not isinstance(timeout, Integral) or timeout < 1:
        raise ValueError("timeout must be a positive integer number of seconds")
    if (isinstance(max_matches, bool) or not isinstance(max_matches, Integral)
            or not 1 <= max_matches <= _MAX_MATCHES):
        raise ValueError(f"max_matches must be an integer in [1, {_MAX_MATCHES}]")
    left, right = _canonical_chemistry(X), _canonical_chemistry(Y)
    if left is None or right is None:
        return math.inf
    a, b = sorted((left, right))
    return _distance_from_smiles(a, b, int(timeout), int(max_matches))
