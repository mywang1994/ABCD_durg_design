"""Scaffold Tree states retain reference coordinates and use ScaffoldGraph 1.1.2.
Its tree_frags_from_mol API applies Schuffenhauer rules and scheme-4 ring fragmentation:
https://github.com/UCLCheminformatics/ScaffoldGraph/blob/master/scaffoldgraph/tree.py
The library removes stereochemistry; fragments are validated and the hierarchy capped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem


SCAFFOLD_TREE_BACKEND = "scaffoldgraph_original_rules_v1"
_SOURCE_INDEX = "_abcd_scaffold_source_atom_index"


@dataclass(frozen=True)
class ScaffoldTreeFragment:
    mol: Chem.Mol
    depth: int
    num_rings: int
    rule: str
    source_atom_indices: tuple[int, ...]
    backend_version: str

    @property
    def source_note(self) -> str:
        return (f"scaffoldgraph_tree@{self.backend_version}|depth={self.depth}"
                f"|rings={self.num_rings}|rule={self.rule}|pose=retained_reference")


def _load_scaffold_backend():
    try:
        from scaffoldgraph import __version__, tree_frags_from_mol
    except ImportError as exc:
        raise RuntimeError(
            "Scaffold Tree state construction requires ScaffoldGraph==1.1.2 "
            "and its dependencies. Install with: pip install ScaffoldGraph==1.1.2."
        ) from exc
    return tree_frags_from_mol, __version__


def _record(diagnostics, reason):
    if diagnostics is not None:
        diagnostics[reason] = diagnostics.get(reason, 0) + 1


def scaffold_tree_fragments(reference, *, max_states=None, diagnostics=None):
    """Return unique levels in the reference frame: depth 0 is Murcko; later levels remove rings.
    Private atom properties track source indices through fragmentation; acyclic inputs yield no states.
    Caps keep the simplest level, full Murcko when at least two fit, and evenly spaced intermediate levels.
    """
    if max_states is not None and (
            isinstance(max_states, bool) or not isinstance(max_states, int) or max_states < 1):
        raise ValueError("max_states must be a positive integer or None")
    if reference is None or reference.GetNumAtoms() == 0:
        raise ValueError("Scaffold Tree requires a nonempty reference molecule")
    source = Chem.Mol(reference)
    Chem.SanitizeMol(source)
    if len(Chem.GetMolFrags(source)) != 1:
        raise ValueError("Scaffold Tree requires a connected reference molecule")
    if source.GetNumConformers() == 0:
        raise ValueError("Scaffold Tree requires reference complex coordinates")
    source_conformer = source.GetConformer()
    coordinates = np.asarray(source_conformer.GetPositions(), dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError("Scaffold Tree reference coordinates must be finite")

    for atom in source.GetAtoms():
        atom.SetIntProp(_SOURCE_INDEX, atom.GetIdx())
        # Atom-map labels are bookkeeping, not a scaffold-priority criterion.
        atom.SetAtomMapNum(0)
    heavy = Chem.RemoveHs(source)
    if heavy.GetRingInfo().NumRings() == 0:
        return []
    generate_tree, backend_version = _load_scaffold_backend()
    try:
        levels = generate_tree(heavy)
    except (ValueError, RuntimeError) as exc:
        raise ValueError("ScaffoldGraph could not construct the reference scaffold tree") from exc

    result, seen = [], set()
    for depth, fragment in enumerate(levels):
        candidate = Chem.Mol(fragment)
        try:
            Chem.SanitizeMol(candidate)
            if (candidate.GetNumHeavyAtoms() < 3 or len(Chem.GetMolFrags(candidate)) != 1
                    or any(atom.GetNumRadicalElectrons() for atom in candidate.GetAtoms())):
                _record(diagnostics, "scaffold_tree_invalid_fragment")
                continue
            if not all(atom.HasProp(_SOURCE_INDEX) for atom in candidate.GetAtoms()):
                raise ValueError("ScaffoldGraph lost the source atom mapping")
            indices = tuple(atom.GetIntProp(_SOURCE_INDEX) for atom in candidate.GetAtoms())
            if (len(set(indices)) != len(indices)
                    or any(index < 0 or index >= source.GetNumAtoms() for index in indices)
                    or any(atom.GetAtomicNum() != source.GetAtomWithIdx(index).GetAtomicNum()
                           for atom, index in zip(candidate.GetAtoms(), indices))):
                raise ValueError("ScaffoldGraph returned an invalid source atom mapping")
            smiles = Chem.MolToSmiles(candidate, canonical=True)
            if smiles in seen:
                continue
            rings = candidate.GetRingInfo().NumRings()
            if rings < 1:
                _record(diagnostics, "scaffold_tree_invalid_fragment")
                continue
            conf = Chem.Conformer(candidate.GetNumAtoms())
            conf.Set3D(source_conformer.Is3D())
            for atom_index, source_index in enumerate(indices):
                conf.SetAtomPosition(atom_index, coordinates[source_index])
            candidate.RemoveAllConformers()
            candidate.AddConformer(conf, assignId=True)
            for atom in candidate.GetAtoms():
                atom.ClearProp(_SOURCE_INDEX)
            rule = candidate.GetProp("prioritization_rule") if candidate.HasProp("prioritization_rule") else "murcko"
            result.append(ScaffoldTreeFragment(candidate, depth, rings, rule, indices, backend_version))
            seen.add(smiles)
        except (ValueError, RuntimeError):
            _record(diagnostics, "scaffold_tree_invalid_fragment_or_mapping")

    if max_states is not None and len(result) > max_states:
        if max_states == 1:
            return [result[-1]]
        selected = np.linspace(0, len(result) - 1, max_states, dtype=int)
        return [result[int(index)] for index in selected]
    return result
