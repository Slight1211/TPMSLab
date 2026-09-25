"""Reproduce licensed COMSOL solid/fluid import and fixed-wall flow examples."""

import argparse
import json
from pathlib import Path
from tpmslab import Config, generate, save_model
from tpmslab.comsol import build_mph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for family in ("Gyroid", "Primitive Schwartz", "Diamond"):
        folder = args.out / family.replace(" ", "_")
        mesh = generate(Config(family=family, domain_mode="solid_fluid", resolution=12))
        save_model(mesh, folder)
        result = build_mph(
            folder, mesh["report"], solve=True, progress=lambda s: print(s, flush=True)
        )
        rows.append(dict(family=family, **result))
        (args.out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
