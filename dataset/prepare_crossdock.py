"""Prepare a filename-based CrossDocked training manifest, keeping raw poses intact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset.crossdock_index import prepare_crossdock_manifest


def main(argv=None):
    root = Path(__file__).resolve().parent / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pocket10-root", type=Path, default=root / "crossdocked_v1.1_rmsd1.0_pocket10")
    parser.add_argument("--split-file", type=Path, default=root / "split_by_name.pt")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-complexes", type=int, help="Prepare the first N available pairs and record the subset size")
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--min-pocket-atoms", type=int, default=4)
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing pairs and record their counts and examples")
    parser.add_argument("--files-only", action="store_true", help="Check filenames and split membership only; skip structure validation")
    args = parser.parse_args(argv)
    result = prepare_crossdock_manifest(
        args.pocket10_root, args.split_file, args.output, split=args.split,
        allow_missing=args.allow_missing, validate_structures=not args.files_only,
        max_complexes=args.max_complexes, max_nodes=args.max_nodes, min_pocket_atoms=args.min_pocket_atoms,
    )
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    print(f"Manifest: {args.output.resolve()}")
    if not result["records"]:
        raise SystemExit("No accepted pairs; inspect rejected entries in the written manifest")


if __name__ == "__main__":
    main()
