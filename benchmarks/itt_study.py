"""Stage-0 go/no-go study for template inheritance + iso-surface transport (ITT).

For every benchmark design of quality_vs_gmsh.py and every resolution:
  dev7     generate(snapped_root) + optimize_mesh (defaults)        -- online
  template generate at the mid density (uniform) + hard optimize      -- offline, reported separately
  itt_raw  template transported onto the graded design (no repair)  -- online
  itt_r1   itt_raw + optimize_mesh(rounds=1)                         -- online
  itt_r6   itt_raw + optimize_mesh(defaults)                         -- online
Metrics: solid-phase p01 / min / median, all-cell min, inversions, solid
volume error vs the 256^3 exact integral (pp of box volume), seconds.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from itt_prototype import transport  # noqa: E402
from tpmslab import Config, generate, optimize_mesh  # noqa: E402
from itt_prototype import tetra_metrics  # noqa: E402  (signed volumes)

spec = importlib.util.spec_from_file_location("qvg", HERE / "quality_vs_gmsh.py")
qvg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qvg)

HARD = dict(rounds=30, quality_threshold=1.0, focus_quantile=0.2, max_ring=9, smooth_sweeps=4, min_round_gain=0.0)


def metrics(model, points, config, reference):
    volumes, quality = tetra_metrics(points[model["tetra"]])
    solid = model["phase_ids"] == 1
    return {"inverted": int((volumes <= 0).sum()), "q_min_all": float(quality.min()),
            "q_min": float(quality[solid].min()), "q_p01": float(np.quantile(quality[solid], .01)),
            "q_median": float(np.median(quality[solid])), "tetrahedra": int(solid.sum()),
            "volume_error_pp": float((volumes[solid].sum() - reference) / np.prod(config.size) * 100)}


def retarget(model, points, config):
    out = dict(model)
    out["points"] = points
    out["report"] = dict(model["report"])
    out["report"]["config"] = config.to_dict()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--designs", nargs="+", default=list(qvg.DESIGNS))
    parser.add_argument("--reference-samples", type=int, default=256)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.designs:
        design = qvg.DESIGNS[name]
        reference = qvg.reference_volume(Config(**design), args.reference_samples)
        mid = (design["density_start"] + design["density_end"]) / 2
        for res in args.resolutions:
            common = dict(resolution=res, domain_mode="solid_fluid", snap_alpha=0.25, edge_interpolation="implicit_root")
            target = Config(**design, **common)
            ref = Config(**{**design, "density_start": mid, "density_end": mid}, gradient="uniform", **common)

            def record(variant, model, points, seconds, **extra):
                row = {"design": name, "resolution": res, "variant": variant, "seconds": round(seconds, 3),
                       **metrics(model, points, target, reference), **extra}
                rows.append(row)
                print(json.dumps(row), flush=True)

            t0 = time.perf_counter()
            direct = generate(target)
            t1 = time.perf_counter()
            optimized = optimize_mesh(direct)
            t2 = time.perf_counter()
            record("dev7_generate", direct, direct["points"], t1 - t0)
            record("dev7", optimized, optimized["points"], t2 - t0)

            t0 = time.perf_counter()
            template = optimize_mesh(generate(ref), **HARD)
            offline = time.perf_counter() - t0
            record("template_offline", template, template["points"], offline)

            t0 = time.perf_counter()
            points, residual = transport(template, ref, target, increments=1, stiffen=1.0)
            t1 = time.perf_counter()
            record("itt_raw", template, points, t1 - t0, residual=float(residual.max()))
            moved = retarget(template, points, target)
            t1 = time.perf_counter()
            r1 = optimize_mesh(moved, rounds=1)
            t2 = time.perf_counter()
            record("itt_r1", r1, r1["points"], (t2 - t1) + (t1 - t0) - 0.0)
            t1 = time.perf_counter()
            r6 = optimize_mesh(moved)
            t2 = time.perf_counter()
            record("itt_r6", r6, r6["points"], (t2 - t1) + (rows[-2]["seconds"]))
            (args.out / "itt_study.json").write_text(json.dumps(rows, indent=1), encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
