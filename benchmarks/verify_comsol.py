"""Optional licensed COMSOL workflow; not part of the open test suite."""

import argparse
import json
from pathlib import Path
from tpmslab import Config, generate, save_model
from tpmslab.comsol import build_mph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for family in ("Gyroid", "Primitive Schwartz", "Diamond"):
        for strategy in ("pulling", "quality_fan"):
            folder = args.out / (family.replace(" ", "_") + "_" + strategy)
            mesh = generate(Config(family=family, resolution=12, quality_strategy=strategy))
            save_model(mesh, folder)
            result = build_mph(
                folder, mesh["report"], solve=True, progress=lambda msg: print(msg, flush=True)
            )
            rows.append({"family": family, "strategy": strategy, **result})
            (args.out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf8")
            print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
