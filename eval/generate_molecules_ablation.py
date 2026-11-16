from __future__ import annotations

from pathlib import Path
import sys

# Resolve shared modules when launched outside the repository.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import generate_molecules as generation


def parse_args(argv=None):
    parser = generation.build_parser()
    parser.description = "Compare inference conditions using the shared generation pipeline without training or modifying checkpoints."
    parser.set_defaults(output_dir=ROOT / "output" / "eval")
    parser.add_argument("--group-size", type=int,
                        help="Fix the number of members per inference group; defaults to the group-size range saved in the checkpoint.")
    parser.add_argument("--group-context", choices=("on", "off"), default="on",
                        help="off disables pooled group conditioning while retaining all sampled members; this is not a retrained pointwise baseline.")
    parser.add_argument("--exclude-state-types", nargs="+",
                        choices=generation.STATE_POOL_TYPES, default=[],
                        help="Pool mode only: remove states by their actual source without refilling the pool; the pool size may change.")
    return generation.validate_args(parser.parse_args(argv), parser)


def main(argv=None):
    args = parse_args(argv)
    return generation.run_generation(
        args, group_size=args.group_size, group_context=args.group_context == "on",
        exclude_state_types=tuple(args.exclude_state_types), evaluation_entry=True,
    )


if __name__ == "__main__":
    main()
