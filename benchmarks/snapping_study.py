"""Paired study of pre-cut vertex snapping on the 12 conditioning-study designs.

Variants per design: baseline (snap_alpha=0), snapped (snap_alpha=ALPHA),
snapped + interior conditioning, and snapped + implicit-root edge projection. Every number is recomputed from the exported
arrays; optimizer or generator flags are not trusted. Reference density is a
dense midpoint sample of the implicit indicator, and surface fidelity is the
first-order distance of interface nodes to the nearest implicit sheet surface.

As in conditioning_study.py, each design is one mesh; cells are not independent
samples, and the scalar diffusion patch check is a consistency check only.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditioning_study import cases, diffusion_patch  # noqa: E402

from tpmslab import Config, condition_mesh, generate, save_model, __version__  # noqa: E402
from tpmslab.model import field_and_threshold  # noqa: E402
from tpmslab.verification import audit_interface  # noqa: E402

PHASE = {1: "solid", 2: "fluid"}
VARIANTS = ("baseline", "snapped", "snapped_conditioned", "snapped_root")


def implicit_values(config, xyz):
    value, threshold = field_and_threshold(config, xyz[:, 0], xyz[:, 1], xyz[:, 2])
    value = np.broadcast_to(np.asarray(value, dtype=float), len(xyz))
    threshold = np.broadcast_to(np.asarray(threshold, dtype=float), len(xyz))
    if config.mode == "sheet":
        return np.stack((threshold - value, threshold + value))
    if config.mode == "solid_above":
        return np.stack((threshold + value,))
    return np.stack((threshold - value,))


def reference_density(config, samples=240):
    axes = [(np.arange(samples) + 0.5) / samples * L for L in config.size]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = np.all(implicit_values(config, grid) >= 0, axis=0)
    return float(inside.mean())


def surface_fidelity(config, model):
    """First-order distance of interface nodes to the nearest implicit surface (mm)."""
    nodes = np.unique(np.asarray(model["interface"]))
    xyz = np.asarray(model["points"])[nodes]
    values = implicit_values(config, xyz)
    h = max(config.size) * 1e-6
    gradient = np.zeros((values.shape[0], len(xyz), 3))
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = h
        gradient[:, :, axis] = (implicit_values(config, xyz + shift)
                                - implicit_values(config, xyz - shift)) / (2 * h)
    norm = np.maximum(np.linalg.norm(gradient, axis=2), 1e-12)
    distance = np.min(np.abs(values) / norm, axis=0)
    return {"interface_nodes": int(len(nodes)),
            "mean_distance_mm": float(distance.mean()),
            "rms_distance_mm": float(np.sqrt(np.mean(distance ** 2))),
            "p99_distance_mm": float(np.quantile(distance, 0.99)),
            "max_distance_mm": float(distance.max())}


def recomputed_stats(model):
    points = np.asarray(model["points"])
    tetra = np.asarray(model["tetra"])
    phases = np.asarray(model["phase_ids"])
    domains = np.asarray(model["domains"])
    p = points[tetra]
    a, b, c = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
    signed = (a[:, 0] * (b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1])
              + a[:, 1] * (b[:, 2] * c[:, 0] - b[:, 0] * c[:, 2])
              + a[:, 2] * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])) / 6
    if np.any(signed <= 0):
        raise ValueError("Recomputation found a nonpositive cell")
    lengths = sum(np.sum((p[:, i] - p[:, j]) ** 2, axis=1)
                  for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    quality = 12 * (3 * signed) ** (2 / 3) / lengths
    per_phase = {}
    for phase in np.unique(phases):
        q = quality[phases == phase]
        per_phase[PHASE[int(phase)]] = {
            "tetrahedra": int(len(q)), "volume_mm3": float(signed[phases == phase].sum()),
            "poor_count_below_001": int(np.sum(q < 0.01)),
            "poor_count_below_01": int(np.sum(q < 0.1)),
            "minimum": float(q.min()), "p01": float(np.quantile(q, 0.01)),
            "p10": float(np.quantile(q, 0.10)), "median": float(np.median(q)),
        }
    return {"vertices": int(len(points)), "tetrahedra": int(len(tetra)),
            "total_volume_mm3": float(signed.sum()),
            "domain_count": int(len(np.unique(domains))),
            "domain_volumes_mm3": {int(d): float(signed[domains == d].sum()) for d in np.unique(domains)},
            "solid_components": int(len(np.unique(domains[phases == 1]))),
            "fluid_components": int(len(np.unique(domains[phases == 2]))),
            "phases": per_phase}


def box_boundary_check(model, size):
    """Every outer-box boundary node must still lie exactly on its box face."""
    points = np.asarray(model["points"])
    boundary = np.asarray(model["boundary"])
    ids = np.asarray(model["boundary_ids"])
    tol = max(size) * 1e-9
    worst = 0.0
    for axis in range(3):
        for side, value in enumerate((0.0, size[axis])):
            face_nodes = np.unique(boundary[ids == 2 + axis * 2 + side])
            if len(face_nodes):
                worst = max(worst, float(np.abs(points[face_nodes, axis] - value).max()))
    inside = np.all((points >= -tol) & (points <= np.asarray(size) + tol))
    return {"max_box_face_deviation_mm": worst, "all_nodes_inside_box": bool(inside),
            "valid": bool(inside and worst < tol)}


def evaluate(name, config, model, elapsed, extra):
    stats = recomputed_stats(model)
    record = {"variant": name, "config": config.to_dict(), "generation_seconds": elapsed,
              "recomputed": stats, "interface": audit_interface(model),
              "box": box_boundary_check(model, config.size),
              "surface_fidelity": surface_fidelity(config, model),
              "patch": diffusion_patch(model["points"], model["tetra"], config.size)}
    record.update(extra)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--save-meshes", action="store_true")
    parser.add_argument("--reference-samples", type=int, default=240)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, base_config in cases():
        print("Case " + name, flush=True)
        reference = reference_density(base_config, args.reference_samples)
        variants = {}
        t0 = time.perf_counter()
        baseline = generate(base_config)
        variants["baseline"] = evaluate("baseline", base_config, baseline, time.perf_counter() - t0, {})
        snapped_config = Config(**{**base_config.to_dict(), "snap_alpha": args.alpha})
        t0 = time.perf_counter()
        snapped = generate(snapped_config)
        snap_info = snapped["report"]["vertex_snapping"]
        variants["snapped"] = evaluate("snapped", snapped_config, snapped, time.perf_counter() - t0,
                                       {"vertex_snapping": snap_info})
        t0 = time.perf_counter()
        conditioned = condition_mesh(snapped)
        cond = conditioned["report"]["conditioning"]
        variants["snapped_conditioned"] = evaluate(
            "snapped_conditioned", snapped_config, conditioned, time.perf_counter() - t0,
            {"vertex_snapping": snap_info,
             "conditioning": {k: cond[k] for k in ("accepted", "rolled_back", "failure_reasons",
                                                   "retained_moves", "elapsed_seconds")}})
        root_config = Config(**{**snapped_config.to_dict(), "edge_interpolation": "implicit_root"})
        t0 = time.perf_counter()
        snapped_root = generate(root_config)
        variants["snapped_root"] = evaluate(
            "snapped_root", root_config, snapped_root, time.perf_counter() - t0,
            {"vertex_snapping": snapped_root["report"]["vertex_snapping"],
             "edge_root": {k: snapped_root["report"][k] for k in
                           ("edge_root_blend", "edge_root_fully_projected_fraction",
                            "edge_root_unbracketed_cuts")}})
        record = {"case": name, "reference_density_samples": args.reference_samples,
                  "reference_density": reference, "variants": variants}
        (args.out / (name + ".json")).write_text(json.dumps(record, indent=2), encoding="utf8")
        if args.save_meshes:
            save_model(baseline, args.out / (name + "_baseline"))
            save_model(snapped, args.out / (name + "_snapped"))
            save_model(conditioned, args.out / (name + "_snapped_conditioned"))
            save_model(snapped_root, args.out / (name + "_snapped_root"))
        box_volume = float(np.prod(base_config.size))
        for variant, data in variants.items():
            for phase, stats in data["recomputed"]["phases"].items():
                rows.append({
                    "case": name, "variant": variant, "phase": phase,
                    "tetrahedra": stats["tetrahedra"], "poor_below_001": stats["poor_count_below_001"],
                    "poor_below_01": stats["poor_count_below_01"], "min": stats["minimum"],
                    "p01": stats["p01"], "p10": stats["p10"], "median": stats["median"],
                    "phase_volume_mm3": stats["volume_mm3"],
                    "solid_density_mesh": data["recomputed"]["phases"]["solid"]["volume_mm3"] / box_volume,
                    "solid_density_reference": reference,
                    "surface_rms_mm": data["surface_fidelity"]["rms_distance_mm"],
                    "surface_max_mm": data["surface_fidelity"]["max_distance_mm"],
                    "interface_valid": data["interface"]["valid"],
                    "box_valid": data["box"]["valid"],
                    "patch_valid": data["patch"]["valid"],
                    "cg_iterations": data["patch"]["jacobi_cg_iterations"],
                    "generation_s": round(data["generation_seconds"], 3),
                    "snapped_vertices": data.get("vertex_snapping", {}).get("snapped_vertices", 0),
                    "conditioning_accepted": data.get("conditioning", {}).get("accepted", ""),
                })
        print(json.dumps({v: {p: s["poor_count_below_001"] for p, s in d["recomputed"]["phases"].items()}
                          for v, d in variants.items()}), flush=True)
    with (args.out / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    def total(variant, key):
        return int(sum(r[key] for r in rows if r["variant"] == variant))

    summary = {"generator_version": __version__, "snap_alpha": args.alpha,
               "python": platform.python_version(), "platform": platform.platform(),
               "dependencies": {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "trimesh")},
               "case_count": len(cases()),
               "poor_below_001": {v: total(v, "poor_below_001") for v in VARIANTS},
               "tetrahedra": {v: total(v, "tetrahedra") for v in VARIANTS},
               "all_interface_valid": all(r["interface_valid"] for r in rows),
               "all_box_valid": all(r["box_valid"] for r in rows),
               "all_patch_valid": all(r["patch_valid"] for r in rows),
               "scope": "12 selected paired designs; descriptive cell totals; scalar patch check only"}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in args.out.iterdir() if f.is_file()}
    (args.out / "sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
