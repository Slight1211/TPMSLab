"""Side-by-side generation time and quality: BCC vs Kuhn (both TPMS Lab) vs Gmsh.

Levels pair a Kuhn resolution with the BCC resolution of similar element count
(about 5/6). Per level and design:
  tpms_kuhn      generate (lattice=kuhn, snap 0.25, implicit root) + optimize_mesh
  tpms_bcc       generate (lattice=bcc,  snap 0.30, implicit root) + optimize_mesh
  gmsh_fixed     TPMS BCC generation supplies the solid boundary; Gmsh (Delaunay +
                 Netgen optimise) fills the interior without touching it
  gmsh_surface   Gmsh reparametrises/remeshes the BCC solid surface STL, then meshes
Each row stores the solid-phase quality, exact-field volume error, P1 k_eff and
the end-to-end seconds (Gmsh rows include the TPMS generation that supplies the
boundary). Per-cell qualities and one mid-plane slice are saved for the figures.
"""
import argparse
import csv
import sys
import json
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quality_vs_gmsh as qvg  # noqa: E402  (importable by name for Gmsh child processes)
from tpmslab import Config, generate, optimize_mesh, __version__  # noqa: E402
from tpmslab.quality import tetra_metrics  # noqa: E402

LEVELS = [(10, 8), (12, 10), (16, 13), (20, 17)]
FIELDS = ["design", "level", "mesher", "resolution", "h_mm", "tetrahedra", "generation_seconds",
          "total_seconds", "q_min", "q_p01", "q_median", "q_below_01_fraction",
          "volume_error_pp", "k_eff_over_bulk", "error"]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--designs", nargs="+", default=list(qvg.DESIGNS))
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(len(LEVELS))))
    parser.add_argument("--surface-timeout", type=int, default=300)
    parser.add_argument("--bcc-boundary-alpha", type=float, default=None)
    parser.add_argument("--bcc-effort", default="standard")
    parser.add_argument("--bcc-envelope", type=float, default=None, help="surface_envelope in mm for the BCC optimisation")
    parser.add_argument("--only", nargs="+", default=None, help="subset of meshers to run (tpms_kuhn tpms_bcc gmsh)")
    parser.add_argument("--keep", default="Gyroid_central:2", help="design:level whose cells are saved")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    transport = qvg.load_transport()
    path = args.out / "crosscompare.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        csv.DictWriter(stream, fieldnames=FIELDS).writeheader()
    keep_design, keep_level = args.keep.split(":")

    def record(row):
        with path.open("a", newline="", encoding="utf8") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore").writerow(row)
        print(json.dumps({k: row.get(k) for k in ("design", "level", "mesher", "tetrahedra", "q_min", "q_p01",
                                                   "q_median", "volume_error_pp", "total_seconds", "error")}),
              flush=True)

    for name in args.designs:
        design = qvg.DESIGNS[name]
        reference = qvg.reference_volume(Config(**design), 256)
        for level in args.levels:
            kuhn_res, bcc_res = LEVELS[level]
            saved = {}
            base = dict(**design, domain_mode="solid_fluid", edge_interpolation="implicit_root")
            runs = (("tpms_kuhn", Config(**base, resolution=kuhn_res, snap_alpha=0.25)),
                    ("tpms_bcc", Config(**base, resolution=bcc_res, lattice="bcc", snap_alpha=0.3,
                                        snap_alpha_boundary=args.bcc_boundary_alpha)))
            for mesher, config in runs:
                if args.only and mesher not in args.only and mesher != "tpms_bcc":
                    continue
                h = config.size[2] / (config.cells[2] * config.resolution)
                common = {"design": name, "level": level, "resolution": config.resolution, "h_mm": h}
                t0 = time.perf_counter()
                model = generate(config)
                t1 = time.perf_counter()
                optimized = optimize_mesh(model, effort=args.bcc_effort if mesher == "tpms_bcc" else "standard",
                                          surface_envelope=args.bcc_envelope if mesher == "tpms_bcc" else None)
                t2 = time.perf_counter()
                solid = optimized["phase_ids"] == 1
                record({**common, "mesher": mesher, "generation_seconds": round(t1 - t0, 3),
                        "total_seconds": round(t2 - t0, 3),
                        **qvg.quality_row(optimized["points"], optimized["tetra"][solid], config, reference, transport)})
                saved[mesher] = (optimized["points"], optimized["tetra"][solid])
                if mesher == "tpms_bcc":
                    bcc_model, bcc_seconds, bcc_h, bcc_config = model, t1 - t0, h, config
            common = {"design": name, "level": level, "resolution": bcc_config.resolution, "h_mm": bcc_h}
            if args.only and "gmsh" not in args.only:
                continue
            with tempfile.TemporaryDirectory(prefix="tpms_") as temporary:
                np.save(Path(temporary) / "points.npy", bcc_model["points"])
                np.save(Path(temporary) / "faces.npy", bcc_model["solid_boundary"])
                try:
                    t0 = time.perf_counter()
                    gp, gt = qvg.gmsh_isolated("fixed", {"points": str(Path(temporary) / "points.npy"),
                                                         "faces": str(Path(temporary) / "faces.npy")}, bcc_h, 900)
                    seconds = time.perf_counter() - t0 + bcc_seconds
                    record({**common, "mesher": "gmsh_fixed", "generation_seconds": round(seconds, 3),
                            "total_seconds": round(seconds, 3),
                            **qvg.quality_row(gp, gt, bcc_config, reference, transport)})
                    saved["gmsh_fixed"] = (gp, gt)
                except Exception as exc:  # noqa: BLE001
                    record({**common, "mesher": "gmsh_fixed", "error": str(exc)[:160]})
                stl = Path(temporary) / "surface.stl"
                bcc_model["surface"].export(str(stl))
                try:
                    t0 = time.perf_counter()
                    gp, gt = qvg.gmsh_isolated("surface", {"stl": str(stl)}, bcc_h, args.surface_timeout)
                    seconds = time.perf_counter() - t0 + bcc_seconds
                    record({**common, "mesher": "gmsh_surface", "generation_seconds": round(seconds, 3),
                            "total_seconds": round(seconds, 3),
                            **qvg.quality_row(gp, gt, bcc_config, reference, transport)})
                    saved["gmsh_surface"] = (gp, gt)
                except Exception as exc:  # noqa: BLE001
                    record({**common, "mesher": "gmsh_surface", "error": str(exc)[:160]})
            if name == keep_design and str(level) == keep_level:
                arrays = {}
                for mesher, (p, t) in saved.items():
                    arrays[f"{mesher}_points"] = p
                    arrays[f"{mesher}_tetra"] = t
                    arrays[f"{mesher}_quality"] = tetra_metrics(p[t])[1]
                np.savez_compressed(args.out / f"cells_{name}_L{level}.npz", **arrays)
    (args.out / "summary.json").write_text(json.dumps({"generator_version": __version__, "levels": LEVELS,
                                                         "designs": args.designs}, indent=2), encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
