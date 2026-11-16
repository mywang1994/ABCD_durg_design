import argparse
import csv
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.SimDivFilters import rdSimDivPickers

SETTINGS = json.loads((Path(__file__).resolve().parents[1] /
                       "configs/defaults.json").read_text(encoding="utf-8"))["evaluation"]


def bottleneck(fingerprints):
    if len(fingerprints) < 2:
        raise ValueError("Bottleneck requires at least two molecules")
    nearest = np.full(len(fingerprints), np.inf)
    for i in range(1, len(fingerprints)):
        distances = np.asarray(DataStructs.BulkTanimotoSimilarity(
            fingerprints[i], fingerprints[:i], returnDistance=True))
        nearest[i] = distances.min()
        np.minimum(nearest[:i], distances, out=nearest[:i])
    return float(nearest.sum())


def evaluate_csv(path, *, smiles_column="SMILES", threshold=SETTINGS["circles"]["threshold"],
                 seed=SETTINGS["circles"]["seed"]):
    if not 0 <= threshold <= 1 or not 0 <= seed < 2**31:
        raise ValueError("Require threshold in [0,1] and seed in [0,2**31)")
    path = Path(path).resolve()
    smiles, input_count = set(), 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        column = next(name for name in reader.fieldnames
                      if name.strip().casefold() == smiles_column.strip().casefold())
        for input_count, row in enumerate(reader, start=1):
            molecule = Chem.MolFromSmiles(row[column])
            for atom in molecule.GetAtoms():
                atom.SetAtomMapNum(0)
            smiles.add(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    smiles = sorted(smiles)
    molecules = [Chem.MolFromSmiles(value) for value in smiles]
    maccs = [MACCSkeys.GenMACCSKeys(molecule) for molecule in molecules]
    bottleneck_score = bottleneck(maccs)
    generator = rdFingerprintGenerator.GetMorganGenerator(**SETTINGS["morgan"])
    morgan = [generator.GetFingerprint(molecule) for molecule in molecules]
    picks, _ = rdSimDivPickers.MaxMinPicker().LazyBitVectorPickWithThreshold(
        morgan, len(morgan), len(morgan), float(threshold), seed=int(seed))
    return {
        "input_csv": str(path), "input_count": input_count, "unique_count": len(smiles),
        "bottleneck_maccs": bottleneck_score, "circles_morgan_approx": len(picks),
        "parameters": {"bottleneck": {**SETTINGS["bottleneck"],
                                      "definition": "sum_i min_{j != i}(1 - Tanimoto(fp_i, fp_j))"},
                       "morgan": SETTINGS["morgan"],
                       "circles": {"threshold": threshold, "seed": seed,
                                   "algorithm": "RDKit MaxMin approximation; lower bound on maximum #Circles",
                                   "pairwise_rule": "1 - Tanimoto > threshold", "exact_maximum_proven": False},
                       "preprocessing": "Remove atom maps; deduplicate and sort canonical isomeric SMILES"},
        "versions": {"rdkit": rdBase.rdkitVersion, "numpy": np.__version__},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path, help="One evaluation set with a SMILES column")
    parser.add_argument("--smiles-column", default="SMILES", help="Column name (case-insensitive)")
    parser.add_argument("--threshold", type=float, default=SETTINGS["circles"]["threshold"],
                        help="Morgan distance threshold for #Circles (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=SETTINGS["circles"]["seed"],
                        help="MaxMin initialization seed (default: %(default)s)")
    parser.add_argument("--output", type=Path, help="JSON result path; default: <input_stem>_diversity.json")
    args = parser.parse_args(argv)
    output = args.output or args.csv_file.with_name(args.csv_file.stem + "_diversity.json")
    if output.resolve() == args.csv_file.resolve():
        raise ValueError("Output must differ from the input CSV")
    result = evaluate_csv(args.csv_file, smiles_column=args.smiles_column,
                          threshold=args.threshold, seed=args.seed)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text, end="")
    return result


if __name__ == "__main__":
    main()
