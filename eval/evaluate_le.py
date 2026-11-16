"""Dock SMILES or SDF molecules to one receptor and average LE = -Vina/heavy atoms."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.evaluate_properties import default_output, iter_molecules, numeric_summary, source_path, write_results

SETTINGS = json.loads((ROOT / "configs/defaults.json").read_text())["evaluation"]["docking"]


def prepare_receptor(source, destination):
    source = Path(source)
    if source.suffix.lower() == ".pdbqt":
        return source
    from meeko import MoleculePreparation, PDBQTWriterLegacy, Polymer, ResidueChemTemplates
    polymer = Polymer.from_pdb_string(
        source.read_text(), ResidueChemTemplates.create_from_defaults(),
        MoleculePreparation(), default_altloc="A")
    rigid, _ = PDBQTWriterLegacy.write_from_polymer(polymer)
    destination = Path(destination)
    destination.write_text(rigid)
    return destination


def receptor_coordinates(path):
    coordinates = [list(map(float, (line[30:38], line[38:46], line[46:54])))
                   for line in Path(path).read_text().splitlines()
                   if line.startswith(("ATOM  ", "HETATM")) and
                   line.split()[-1] not in ("H", "HD", "HS")]
    return np.asarray(coordinates, dtype=float)


def docking_box(points, *, center=None, box_size=None):
    points = np.asarray(points, dtype=float)
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2 if center is None else np.asarray(center, dtype=float)
    if box_size is None:
        half_extent = np.maximum(np.abs(hi - center), np.abs(lo - center))
        box_size = np.maximum(2 * (half_extent + SETTINGS["box_padding_per_side"]),
                              SETTINGS["minimum_box_size"])
    box_size = np.asarray(box_size, dtype=float)
    return center.tolist(), box_size.tolist()


def reference_coordinates(path):
    record, = iter_molecules(str(path))
    molecule = record["mol"]
    if not molecule.GetNumConformers() or not molecule.GetConformer().Is3D():
        raise ValueError("Reference requires a 3D pose in the receptor coordinate frame")
    return np.asarray(molecule.GetConformer().GetPositions())[
        [atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()]]


def ligand_pdbqt(molecule, center, seed):
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    molecule = Chem.AddHs(molecule, addCoords=True)
    if not molecule.GetNumConformers() or not molecule.GetConformer().Is3D():
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = seed
        parameters.numThreads = 1
        if AllChem.EmbedMolecule(molecule, parameters) != 0:
            raise ValueError("ETKDGv3 could not generate a 3D conformer")
    conformer = molecule.GetConformer()
    positions = np.asarray(conformer.GetPositions())
    heavy = [atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()]
    positions += np.asarray(center) - positions[heavy].mean(axis=0)
    for index, position in enumerate(positions):
        conformer.SetAtomPosition(index, position)
    setups = MoleculePreparation().prepare(molecule)
    text, success, error = PDBQTWriterLegacy.write_string(setups[0])
    if not success:
        raise ValueError(f"Ligand preparation failed: {error}")
    return text


def dock_records(records, engine, *, center, exhaustiveness, seed):
    rows = []
    for record in records:
        heavy_atoms = record["mol"].GetNumHeavyAtoms()
        engine.set_ligand_from_string(ligand_pdbqt(record["mol"], center, seed))
        engine.dock(exhaustiveness=exhaustiveness, n_poses=SETTINGS["n_poses"])
        affinity = float(engine.energies(n_poses=1)[0][0])
        rows.append({"index": record["index"], "smiles": record["smiles"],
                     "heavy_atoms": heavy_atoms, "vina": affinity, "le": -affinity / heavy_atoms})
        if record["index"] % 25 == 0:
            print(f"Processed {record['index']} molecules", file=sys.stderr, flush=True)
    return rows


def evaluate(source, receptor, *, smiles_column="SMILES", reference_sdf=None,
             center=None, box_size=None, exhaustiveness=SETTINGS["exhaustiveness"],
             cpu=SETTINGS["cpu"], seed=SETTINGS["seed"]):
    import meeko
    import vina
    receptor = Path(receptor).resolve()
    with tempfile.TemporaryDirectory(prefix="abcd_le_") as work:
        prepared = prepare_receptor(receptor, Path(work) / "receptor.pdbqt")
        points = reference_coordinates(reference_sdf) if reference_sdf else receptor_coordinates(prepared)
        center, box_size = docking_box(points, center=center, box_size=box_size)
        engine = vina.Vina(sf_name=SETTINGS["scoring_function"], cpu=cpu, seed=seed, verbosity=0)
        engine.set_receptor(str(prepared))
        # Build all atom-type maps once for the complete ligand set.
        engine.compute_vina_maps(center=center, box_size=box_size)
        rows = dock_records(iter_molecules(source, smiles_column), engine, center=center,
                            exhaustiveness=exhaustiveness, seed=seed)
    summary = {
        "input": str(source), "receptor": str(receptor),
        "input_count": len(rows),
        "metrics": {name: numeric_summary(row[name] for row in rows) for name in ("vina", "le")},
        "parameters": {"mode": "redock", "scoring_function": SETTINGS["scoring_function"],
                       "center": center, "box_size": box_size,
                       "box_reference": str(reference_sdf) if reference_sdf else "receptor heavy atoms",
                       "exhaustiveness": exhaustiveness, "n_poses": SETTINGS["n_poses"], "cpu": cpu, "seed": seed,
                       "receptor_preparation": "supplied PDBQT" if receptor.suffix.lower() == ".pdbqt"
                           else SETTINGS["receptor_preparation"],
                       "ligand_preparation": "Meeko; ETKDGv3 for missing/2D conformers; recenter before redocking",
                       "le": "-best Vina affinity (kcal/mol) / heavy-atom count",
                       "averaging": "All records, duplicates retained; mean per molecule, SD ddof=1"},
        "versions": {"rdkit": rdBase.rdkitVersion, "vina": vina.__version__, "meeko": meeko.__version__},
    }
    return rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="SMILES string, SMILES CSV, .smi/.txt file, or SDF")
    parser.add_argument("--receptor", type=Path, required=True, help="CrossDocked pocket PDB or prepared PDBQT")
    parser.add_argument("--reference-sdf", type=Path, help="Bound ligand defining the common search box")
    parser.add_argument("--center", type=float, nargs=3, help="Search-box center in Angstrom")
    parser.add_argument("--box-size", type=float, nargs=3, help="Search-box side lengths in Angstrom")
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument("--exhaustiveness", type=int, default=SETTINGS["exhaustiveness"])
    parser.add_argument("--cpu", type=int, default=SETTINGS["cpu"])
    parser.add_argument("--seed", type=int, default=SETTINGS["seed"])
    parser.add_argument("--output", type=Path, help="Output CSV; a matching JSON stores the summary")
    args = parser.parse_args(argv)
    output = args.output or default_output(args.input, "_le")
    destinations = {output.resolve(), output.with_suffix(".json").resolve()}
    inputs = {path.resolve() for path in (args.receptor, args.reference_sdf, source_path(args.input))
              if path is not None}
    if output.suffix.lower() != ".csv" or len(destinations) != 2 or inputs & destinations:
        parser.error("Output must be a .csv whose CSV/JSON paths differ from all input files")
    rows, summary = evaluate(
        args.input, args.receptor, smiles_column=args.smiles_column, reference_sdf=args.reference_sdf,
        center=args.center, box_size=args.box_size, exhaustiveness=args.exhaustiveness,
        cpu=args.cpu, seed=args.seed)
    write_results(rows, summary, output)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return summary


if __name__ == "__main__":
    main()
