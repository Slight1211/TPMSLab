"""Run from the source directory after installation; output must be new."""

import argparse
from pathlib import Path

from tpmslab import Config, generate, run_comsol, save_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--problem", choices=("flow", "elasticity"), default="flow")
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--import-only", action="store_true")
    parser.add_argument("--preferences-directory", type=Path)
    args = parser.parse_args()
    flow = args.problem == "flow"
    config = Config(resolution=args.resolution, snap_alpha=.25,
                    domain_mode="solid_fluid" if flow else "solid",
                    edge_interpolation="implicit_root" if flow else "linear")
    folder = save_model(generate(config, progress=print), args.out)
    result = run_comsol(folder, solve=not args.import_only, progress=print,
                        preferences_directory=args.preferences_directory)
    print(result["output_directory"])


if __name__ == "__main__":
    main()
