"""Calculate SA, QED and a six-point Lipinski-like score from SMILES or SDF."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer


SETTINGS = json.loads((Path(__file__).resolve().parents[1] /
                       "configs/defaults.json").read_text(encoding="utf-8"))["evaluation"]["properties"]
LIMITS = SETTINGS["lipinski"]
LIPINSKI_RULES = {name: f"<= {LIMITS[name + '_max']}"
                  for name in ("hbd", "hba", "logp", "rotatable_bonds", "tpsa")}
LIPINSKI_RULES["mw"] = f"< {LIMITS['mw_upper_exclusive']}"


def source_path(source):
    path = Path(source)
    try:
        return path if path.is_file() else None
    except OSError:
        return None


def default_output(source, suffix):
    path = source_path(source)
    return (path.with_name(path.stem + suffix + ".csv") if path is not None
            else Path("smiles" + suffix + ".csv"))


def _raw_molecules(source, smiles_column):
    path = source_path(source)
    if path is None:
        yield Chem.MolFromSmiles(str(source))
    elif path.suffix.lower() == ".sdf":
        with path.open("rb") as handle:
            yield from Chem.ForwardSDMolSupplier(handle, removeHs=False)
    elif path.suffix.lower() == ".mol":
        yield Chem.MolFromMolFile(str(path), removeHs=False)
    elif path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            column = next(name for name in reader.fieldnames
                          if name.strip().casefold() == smiles_column.strip().casefold())
            yield from (Chem.MolFromSmiles(row[column]) for row in reader)
    else:
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip() and not line.lstrip().startswith("#"):
                    yield Chem.MolFromSmiles(line.split()[0])


def iter_molecules(source, smiles_column="SMILES"):
    """Remove ordinary explicit H and atom maps; retain duplicate records."""
    for index, molecule in enumerate(_raw_molecules(source, smiles_column), 1):
        molecule = Chem.RemoveHs(molecule)
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        yield {"index": index, "smiles": Chem.MolToSmiles(molecule, isomericSmiles=True), "mol": molecule}


def lipinski_score(descriptors):
    return sum((descriptors["hbd"] <= LIMITS["hbd_max"], descriptors["hba"] <= LIMITS["hba_max"],
                descriptors["mw"] < LIMITS["mw_upper_exclusive"], descriptors["logp"] <= LIMITS["logp_max"],
                descriptors["rotatable_bonds"] <= LIMITS["rotatable_bonds_max"],
                descriptors["tpsa"] <= LIMITS["tpsa_max"]))


def molecular_properties(molecule):
    values = {
        "hbd": Lipinski.NumHDonors(molecule), "hba": Lipinski.NumHAcceptors(molecule),
        "mw": Descriptors.MolWt(molecule), "logp": Crippen.MolLogP(molecule),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(
            molecule, rdMolDescriptors.NumRotatableBondsOptions.Strict),
        "tpsa": rdMolDescriptors.CalcTPSA(molecule), "heavy_atoms": molecule.GetNumHeavyAtoms(),
    }
    return {"sa": sascorer.calculateScore(molecule), "qed": QED.qed(molecule),
            "lipinski": lipinski_score(values), **values}


def numeric_summary(values):
    values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Summary values must be finite")
    return {"count": len(values), "mean": statistics.mean(values) if values else None,
            "sd": statistics.stdev(values) if len(values) > 1 else None}


def write_results(rows, summary, output):
    output = Path(output)
    if output.suffix.lower() != ".csv":
        raise ValueError("Output filename must end in .csv")
    sidecar = output.with_suffix(".json")
    source = source_path(summary["input"])
    destinations = (output.resolve(), sidecar.resolve())
    if len(set(destinations)) != 2 or (source is not None and source.resolve() in destinations):
        raise ValueError("Outputs must differ from each other and the input")
    text = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fields = rows[0].keys()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    sidecar.write_text(text, encoding="utf-8")


def evaluate(source, *, smiles_column="SMILES"):
    rows = [{"index": record["index"], "smiles": record["smiles"], **molecular_properties(record["mol"])}
            for record in iter_molecules(source, smiles_column)]
    summary = {
        "input": str(source), "input_count": len(rows),
        "metrics": {name: numeric_summary(row[name] for row in rows)
                    for name in ("sa", "qed", "lipinski")},
        "parameters": {"sa": "RDKit SA_Score, raw 1-10; lower is easier",
                       "qed": "RDKit QED, average descriptor weights; 0-1",
                       "lipinski": {"range": [0, 6], "rules": LIPINSKI_RULES,
                                    "rotatable_bonds": "RDKit Strict"},
                       "preprocessing": "Remove ordinary explicit H and atom maps; retain duplicates, fragments, charges, isotopes and stereochemistry",
                       "averaging": "Arithmetic mean over all records; SD uses ddof=1"},
        "versions": {"rdkit": rdBase.rdkitVersion},
    }
    return rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="SMILES string, SMILES CSV, .smi/.txt file, or SDF")
    parser.add_argument("--smiles-column", default="SMILES", help="CSV column (case-insensitive)")
    parser.add_argument("--output", type=Path, help="Output CSV; a matching JSON stores the summary")
    args = parser.parse_args(argv)
    rows, summary = evaluate(args.input, smiles_column=args.smiles_column)
    write_results(rows, summary, args.output or default_output(args.input, "_properties"))
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return summary


if __name__ == "__main__":
    main()
