from __future__ import annotations

import argparse
import json

from .dataset import export_demo_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SymPy tracing demo datasets.")
    parser.add_argument("output_dir", help="Directory where dataset artifacts will be written.")
    parser.add_argument("--num-expr", type=int, default=4, help="Number of expression seeds to generate.")
    parser.add_argument("--num-solve", type=int, default=4, help="Number of solve seeds to generate.")
    args = parser.parse_args()

    summary = export_demo_corpus(args.output_dir, num_expr=args.num_expr, num_solve=args.num_solve)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
