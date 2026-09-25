"""Single-run paired mesh benchmarks. Run on an otherwise idle machine for timing."""

import argparse
import csv
import json
import platform
import importlib.metadata
from pathlib import Path
from tpmslab import Config, generate
from tpmslab.model import calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for family in ["Gyroid", "Primitive Schwartz", "Diamond"]:
        for profile in ["uniform", "linear"]:
            for resolution in [12, 20, 28]:
                paired = []
                for strategy in ["pulling", "quality_fan"]:
                    c = Config(
                        family=family,
                        gradient=profile,
                        resolution=resolution,
                        density_start=0.4 if profile == "uniform" else 0.25,
                        density_end=0.5,
                        quality_strategy=strategy,
                    )
                    calibration(c.expression, c.mode)  # exclude CDF warmup from timing
                    report = generate(c)["report"]
                    row = {
                        "family": family,
                        "gradient": profile,
                        "resolution": resolution,
                        "strategy": strategy,
                        "tetrahedra": report["tetrahedra"],
                        "volume_mm3": report["volume_mm3"],
                        "density": report["actual_density"],
                        "q_min": report["quality_quantiles"]["min"],
                        "q_p01": report["quality_quantiles"]["p01"],
                        "q_median": report["quality_quantiles"]["median"],
                        "poor_count": report["low_quality_count_below_001"],
                        "poor_volume_fraction": report["low_quality_volume_fraction_below_001"],
                        "improved_cells": report["improved_cut_cells"],
                        "seconds": report["elapsed_seconds"],
                        "boundary_sha256": report["boundary_sha256"],
                    }
                    rows.append(row)
                    paired.append(row)
                    print(
                        f"{family} {profile} N={resolution} {strategy}: {row['tetrahedra']} tets",
                        flush=True,
                    )
                    (args.out / "results.json").write_text(
                        json.dumps(rows, indent=2), encoding="utf8"
                    )
                assert paired[0]["boundary_sha256"] == paired[1]["boundary_sha256"]
                assert abs(paired[0]["volume_mm3"] - paired[1]["volume_mm3"]) < 1e-9
                assert paired[1]["q_min"] >= paired[0]["q_min"] * (1 - 1e-9)
    with (args.out / "results.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "dependencies": {
                    name: importlib.metadata.version(name)
                    for name in ("tpmslab", "numpy", "scipy", "trimesh")
                },
                "platform": platform.platform(),
                "timing": "single measurement; CDF prewarmed; not a speed comparison",
            },
            indent=2,
        ),
        encoding="utf8",
    )


if __name__ == "__main__":
    main()
