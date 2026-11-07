"""Paired conditioning study and an exact linear-FE diffusion patch check.

The patch is a numerical consistency/linear-solver check on the complete box,
not a permeability, FSI or elasticity-accuracy validation. Each case is one mesh;
cells within a mesh are not independent experimental replicates.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg, LinearOperator

from tpmslab import Config, condition_mesh, generate, save_model, __version__
from tpmslab.verification import audit_interface


def diffusion_patch(points, tetra, size):
    """P1 Laplace on a homogeneous box: u=z/Lz, zero lateral normal flux."""
    p = points[tetra]
    edges = np.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2)
    signed = np.linalg.det(edges) / 6
    if np.any(signed <= 0):
        raise ValueError("Independent patch check found a nonpositive tetrahedron")
    inverse = np.linalg.inv(edges)
    gradients = np.concatenate((-inverse.sum(axis=1)[:, None], inverse), axis=1)
    local = signed[:, None, None] * np.einsum("nik,njk->nij", gradients, gradients)
    matrix = coo_matrix((local.ravel(),
                         (np.repeat(tetra, 4, axis=1).ravel(), np.tile(tetra, (1, 4)).ravel())),
                        shape=(len(points), len(points))).tocsr()
    boundary = np.isclose(points[:, 2], 0, rtol=0, atol=max(size) * 1e-10) | np.isclose(
        points[:, 2], size[2], rtol=0, atol=max(size) * 1e-10)
    free = np.flatnonzero(~boundary)
    fixed = np.flatnonzero(boundary)
    u = np.zeros(len(points))
    u[fixed] = points[fixed, 2] / size[2]
    reduced = matrix[free][:, free]
    rhs = -matrix[free][:, fixed] @ u[fixed]
    diagonal = reduced.diagonal()
    if np.any(diagonal <= 0):
        raise ValueError("Nonpositive stiffness diagonal")
    preconditioner = LinearOperator(reduced.shape, matvec=lambda x: x / diagonal)
    iterations = 0

    def count(_):
        nonlocal iterations
        iterations += 1

    started = time.perf_counter()
    solution, info = cg(reduced, rhs, M=preconditioner, rtol=1e-9, atol=0, maxiter=2000, callback=count)
    solve_seconds = time.perf_counter() - started
    u[free] = solution
    exact_energy = float(np.prod(size) / size[2]**2)
    energy = float(u @ (matrix @ u))
    error = float(np.max(np.abs(u - points[:, 2] / size[2])))
    residual = float(np.linalg.norm(reduced @ solution - rhs) / np.linalg.norm(rhs))
    valid = info == 0 and error < 1e-6 and residual < 2e-9 and abs(energy / exact_energy - 1) < 1e-7
    return {"valid": bool(valid), "cg_info": int(info), "jacobi_cg_iterations": iterations,
            "solve_seconds": solve_seconds, "relative_residual": residual,
            "maximum_nodal_error": error, "dissipation": energy,
            "exact_dissipation": exact_energy,
            "scope": "homogeneous full-box scalar diffusion; not solid elasticity or pore flow"}


def independent_geometry(before, after):
    fixed = np.unique(before["boundary"])
    result = {"fixed_boundary_nodes": bool(np.array_equal(before["points"][fixed], after["points"][fixed]))}
    result["same_connectivity_labels"] = all(np.array_equal(before[k], after[k])
                                             for k in ("tetra", "domains", "phase_ids", "interface", "boundary", "boundary_ids"))
    domain_volumes = []
    for model in (before, after):
        p = model["points"][model["tetra"]]
        a, b, c = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
        # Component form, independent of the conditioner and FE determinant routine.
        v = (a[:, 0] * (b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1])
             + a[:, 1] * (b[:, 2] * c[:, 0] - b[:, 0] * c[:, 2])
             + a[:, 2] * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])) / 6
        if np.any(v <= 0):
            raise ValueError("Independent geometry check found a nonpositive cell")
        domain_volumes.append({int(d): float(v[model["domains"] == d].sum()) for d in np.unique(model["domains"])})
    result["maximum_domain_volume_relative_drift"] = max(
        abs(domain_volumes[1][d] / value - 1) for d, value in domain_volumes[0].items())
    result["interface"] = audit_interface(after)
    result["valid"] = (result["fixed_boundary_nodes"] and result["same_connectivity_labels"]
                       and result["maximum_domain_volume_relative_drift"] < 1e-10
                       and result["interface"]["valid"])
    return result


def cases():
    designs = []
    for family in ("Gyroid", "Primitive Schwartz", "Diamond"):
        for label, start, end in (("low", .15, .30), ("central", .25, .50), ("high", .40, .65)):
            designs.append((family.replace(" ", "_") + "_" + label,
                            Config(family=family, density_start=start, density_end=end,
                                   resolution=12, domain_mode="solid_fluid")))
    designs.extend([
        ("Gyroid_n16", Config(resolution=16, domain_mode="solid_fluid")),
        ("Primitive_shifted", Config(family="Primitive Schwartz", phase_degrees=(17, 31, 47),
                                     resolution=12, domain_mode="solid_fluid")),
        ("Gyroid_Diamond_blend", Config(combine="blend", family_secondary="Diamond",
                                       resolution=12, domain_mode="solid_fluid")),
    ])
    return designs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-meshes", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows, outcomes = [], []
    for name, config in cases():
        print("Case " + name, flush=True)
        baseline = generate(config)
        conditioned = condition_mesh(baseline)
        e = conditioned["report"]["conditioning"]
        geometry = independent_geometry(baseline, conditioned)
        patch_before = diffusion_patch(baseline["points"], baseline["tetra"], config.size)
        patch_after = diffusion_patch(conditioned["points"], conditioned["tetra"], config.size)
        record = {"case": name, "config": config.to_dict(), "conditioning": e,
                  "geometry": geometry, "patch_before": patch_before, "patch_after": patch_after}
        (args.out / (name + ".json")).write_text(json.dumps(record, indent=2), encoding="utf8")
        if args.save_meshes:
            save_model(baseline, args.out / (name + "_baseline"))
            save_model(conditioned, args.out / (name + "_conditioned"))
        for phase in e["before_by_phase"]:
            before, after = e["before_by_phase"][phase], e["after_by_phase"][phase]
            rows.append({"case": name, "phase": "solid" if phase == "1" else "fluid",
                         "accepted": e["accepted"], "tetrahedra": before["tetrahedra"],
                         "poor_before": before["poor_count_below_001"], "poor_after": after["poor_count_below_001"],
                         "min_before": before["minimum"], "min_after": after["minimum"],
                         "p01_before": before["p01"], "p01_after": after["p01"],
                         "median_before": before["median"], "median_after": after["median"],
                         "conditioning_s": e["elapsed_seconds"], "geometry_passed": geometry["valid"],
                         "patch_before_passed": patch_before["valid"], "patch_after_passed": patch_after["valid"],
                         "cg_before": patch_before["jacobi_cg_iterations"], "cg_after": patch_after["jacobi_cg_iterations"]})
        outcomes.append({"case": name, "accepted": e["accepted"], "rolled_back": e["rolled_back"],
                         "geometry_passed": geometry["valid"],
                         "patch_passed": patch_before["valid"] and patch_after["valid"]})
        print(json.dumps(outcomes[-1]), flush=True)
    with (args.out / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"generator_version": __version__, "python": platform.python_version(),
               "dependencies": {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "trimesh")},
               "outcomes": outcomes, "case_count": len(outcomes),
               "poor_count_before": sum(r["poor_before"] for r in rows),
               "poor_count_after": sum(r["poor_after"] for r in rows),
               "accepted_cases": sum(o["accepted"] for o in outcomes),
               "scope": "12 selected paired meshes, no independent replication; cell totals are descriptive, not an inferential sample size"}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in args.out.iterdir() if f.is_file()}
    (args.out / "sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf8")
    print(json.dumps(summary), flush=True)
    if not all(o["geometry_passed"] and o["patch_passed"] for o in outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
