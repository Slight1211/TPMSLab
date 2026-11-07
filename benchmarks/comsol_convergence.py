"""COMSOL mesh-convergence study (runs only on a Windows machine with licensed COMSOL; verified on 6.2.0.290).

For each design, resolution and generator variant it generates the two-phase
mesh and (separately) the solid-only mesh, builds the MPH through the existing
bridge and runs the two demos already defined in the package:

* pore phase: fixed-wall creeping flow, inlet 0.01 Pa at z=0, outlet 0 Pa at
  z=Lz, no slip elsewhere -> outlet flux Q and an apparent Darcy permeability
  K = Q * mu * Lz / (Lx*Ly*dp);
* solid phase: linear elasticity, bottom fixed, top z displacement -0.001*Lz
  -> strain energy and maximum displacement.

Every run keeps its own directory with the COMSOL logs, MPH and verification
record. Results are appended to comsol_convergence.csv after each run so a
crash or timeout loses nothing. Nothing is deleted or overwritten.

Example (PowerShell, from the source directory, after Run-ComsolConvergence.cmd
created the environment):
  .\\.venv\\Scripts\\python.exe benchmarks\\comsol_convergence.py --out ..\\comsol_convergence_20261030 --resolutions 8 12 16
"""

import argparse
import csv
import json
import hashlib
import os
import platform
import re
import time
import traceback
from uuid import uuid4
from pathlib import Path

import numpy as np

from tpmslab import Config, generate, save_model, __version__
from tpmslab.comsol import build_mph, detect_comsol

DESIGNS = {
    "Gyroid_central": dict(family="Gyroid", density_start=.25, density_end=.50),
    "Diamond_low": dict(family="Diamond", density_start=.15, density_end=.30),
    "Primitive_Schwartz_central": dict(family="Primitive Schwartz", density_start=.25, density_end=.50),
}
FIELDS = ["design", "resolution", "h_mm", "variant", "problem", "snap_alpha", "snap_target",
          "edge_interpolation", "element_order", "comsol_version", "config_sha256", "comsol_folder", "attempt_id", "tetrahedra", "vertices", "generation_seconds",
          "comsol_seconds", "solid_density", "permeability_m2", "outlet_flux_m3_s", "relative_flux_imbalance",
          "strain_energy_J", "max_displacement_m", "comsol_solution_time_s", "folder", "status", "error"]


def comsol_solution_time(folder):
    """Best-effort parse of COMSOL's own solver timing from the batch log."""
    try:
        text = (Path(folder) / "comsol.log").read_text(encoding="utf8", errors="replace")
    except OSError:
        return None
    times = [float(x) for x in re.findall(r"Solution time:\s*([0-9.]+)\s*s", text)]
    return max(times) if times else None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16, 24])
    parser.add_argument("--designs", nargs="+", choices=list(DESIGNS), default=list(DESIGNS))
    parser.add_argument("--max-tetrahedra", type=int, default=1_000_000)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--skip-elasticity", action="store_true")
    parser.add_argument("--skip-flow", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip successful cases in this study; retain failed attempts")
    parser.add_argument("--preferences-directory", type=Path, help="Explicit trusted COMSOL preference files")
    parser.add_argument("--element-order", type=int, choices=(1, 2), default=1, help="1: linear (verified default); 2: quadratic displacement / P2+P1 flow")
    parser.add_argument("--np", type=int, default=2, dest="processes", help="COMSOL -np processes")
    args = parser.parse_args(argv)
    if args.skip_flow and args.skip_elasticity:
        parser.error("At least one problem must be enabled.")
    if any(r < 8 for r in args.resolutions) or len(set(args.resolutions)) != len(args.resolutions):
        parser.error("resolutions must be distinct integers of at least 8")
    if not 1 <= args.max_tetrahedra <= 1_000_000 or not 1 <= args.timeout <= 3600:
        parser.error("max-tetrahedra must be in [1,1000000] and timeout in [1,3600]")
    if detect_comsol() is None:
        raise SystemExit("COMSOL was not found. Set COMSOL_BIN to the folder containing comsolbatch.exe.")
    args.out.mkdir(parents=True, exist_ok=args.resume)
    csv_path = args.out / "comsol_convergence.csv"
    study = {"generator_version": __version__, "resolutions": args.resolutions, "designs": args.designs,
             "skip_flow": args.skip_flow, "skip_elasticity": args.skip_elasticity,
             "max_tetrahedra": args.max_tetrahedra, "timeout": args.timeout,
             "element_order": args.element_order}
    if args.preferences_directory is not None:
        pref = args.preferences_directory.resolve()
        study["preferences_directory"] = str(pref)
        study["preferences_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in (pref / "comsol.prefs", pref / "comsolserver.prefs") if p.is_file()}
    environment_path = args.out / "environment.json"
    rows = []
    if args.resume:
        if json.loads(environment_path.read_text(encoding="utf8")).get("study_options") != study:
            parser.error("Study options or generator version differ from the saved study.")
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    else:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
        environment_path.write_text(json.dumps({
        "generator_version": __version__, "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "comsol_bin": str(detect_comsol()), "resolutions": args.resolutions,
        "designs": args.designs, "study_options": study}, indent=2), encoding="utf8")

    def row_key(row):
        return tuple(str(row[k]) for k in ("design", "resolution", "variant", "problem"))

    latest = {row_key(row): row for row in rows}

    def record(row):
        with csv_path.open("a", newline="", encoding="utf8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow({k: row.get(k, "") for k in FIELDS})
            f.flush()
            os.fsync(f.fileno())
        latest[row_key(row)] = row
        print(json.dumps({k: row.get(k) for k in ("design", "resolution", "variant", "problem", "status", "error")}), flush=True)

    for name in args.designs:
        design = DESIGNS[name]
        for res in args.resolutions:
            for variant in ("baseline", "snapped_root"):
                extra = {} if variant == "baseline" else {"snap_alpha": 0.25, "edge_interpolation": "implicit_root"}
                problems = []
                if not args.skip_flow:
                    problems.append(("flow", dict(domain_mode="solid_fluid", **extra)))
                if not args.skip_elasticity:
                    # Elasticity uses the solid-only generator; root projection needs two phases,
                    # so the snapped solid-only mesh uses snapping alone.
                    problems.append(("elasticity", dict(domain_mode="solid", **{k: v for k, v in extra.items() if k == "snap_alpha"})))
                for problem, mode in problems:
                    config = Config(resolution=res, **design, **mode)
                    actual_variant = "snapped" if variant == "snapped_root" and problem == "elasticity" else variant
                    key = (name, str(res), actual_variant, problem)
                    if latest.get(key, {}).get("status") == "ok":
                        continue
                    h = config.size[2] / (config.cells[2] * res)
                    folder = args.out / f"{name}_r{res}_{actual_variant}_{problem}"
                    if folder.exists():
                        folder = folder.with_name(folder.name + "_retry_" + uuid4().hex[:12])
                    row = {"design": name, "resolution": res, "h_mm": h, "variant": actual_variant, "problem": problem,
                           "folder": folder.name, "snap_alpha": config.snap_alpha, "snap_target": config.snap_target,
                           "edge_interpolation": config.edge_interpolation,
                           "element_order": args.element_order,
                           "config_sha256": hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()}
                    model = None
                    try:
                        t0 = time.perf_counter()
                        model = generate(config)
                        row["generation_seconds"] = round(time.perf_counter() - t0, 2)
                        row["tetrahedra"] = model["report"]["tetrahedra"]
                        row["vertices"] = model["report"]["vertices"]
                        row["solid_density"] = model["report"].get("actual_density") if problem == "elasticity" else \
                            model["report"]["solid_volume_mm3"] / float(np.prod(config.size))
                        save_model(model, folder)
                        t0 = time.perf_counter()
                        result = build_mph(folder, model["report"], solve=True, progress=print,
                                           max_solve_tetrahedra=args.max_tetrahedra, build_timeout_s=args.timeout,
                                           preferences_directory=args.preferences_directory,
                                           element_order=args.element_order, processes=args.processes)
                        row["comsol_version"] = result.get("comsol_version")
                        row["comsol_seconds"] = round(time.perf_counter() - t0, 1)
                        row["comsol_folder"] = str(Path(result["output_directory"]).relative_to(args.out.resolve()))
                        row["attempt_id"] = result["attempt_id"]
                        row["comsol_solution_time_s"] = comsol_solution_time(result["output_directory"])
                        if problem == "flow":
                            q = result["outlet_flux_m3_s"]
                            lx, ly, lz = (s * 1e-3 for s in config.size)
                            row["outlet_flux_m3_s"] = q
                            row["relative_flux_imbalance"] = result["relative_flux_imbalance"]
                            row["permeability_m2"] = q * 0.001 * lz / (lx * ly * 0.01)
                        else:
                            row["strain_energy_J"] = result["strain_energy_J"]
                            row["max_displacement_m"] = result["max_displacement_m"]
                        row["status"] = "ok"
                    except Exception as exc:  # noqa: BLE001
                        row["status"] = "failed"
                        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                        if model is not None and (model_report := model["report"].get("comsol_run")):
                            row["comsol_folder"] = model_report["output_directory"]
                            row["attempt_id"] = model_report["attempt_id"]
                        with (args.out / "errors.log").open("a", encoding="utf8") as stream:
                            stream.write(f"{folder.name}\n{traceback.format_exc()}\n")
                    record(row)
    print("Done:", csv_path)
    return int(any(row.get("status") != "ok" for row in latest.values()))


if __name__ == "__main__":
    raise SystemExit(main())
