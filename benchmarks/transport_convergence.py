"""Mesh-convergence study of effective transport coefficients on both phases.

For each design, resolution and generator variant, the solid phase and the pore
phase are each solved with P1 finite elements for steady diffusion
(Laplace) with u=0 on z=0, u=1 on z=Lz and zero flux elsewhere, including the
solid/fluid interface. The dissipation E = int |grad u|^2 gives the effective
coefficient relative to the bulk value: k_eff/k_bulk = E / (Lx*Ly/Lz).

Variants: baseline (snap_alpha=0) and snapped_root (snap_alpha=0.25 +
edge_interpolation="implicit_root"). Optionally the solid phase is also
re-meshed by Gmsh (reparametrized surface, size = lattice spacing) from the
baseline STL as an external cross-check with the same solver.

Richardson extrapolation uses the three finest resolutions. Nothing here is a
COMSOL, Stokes, elasticity or FSI validation; it is the simplest phase-wise
elliptic problem that the two-phase mesh is meant to feed.
"""

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg, LinearOperator

from tpmslab import Config, generate, save_model, __version__
from tpmslab.convergence import richardson as richardson, summarize_transport
from tpmslab.io import file_sha256

DESIGNS = {
    "Gyroid_central": dict(family="Gyroid", density_start=.25, density_end=.50),
    "Diamond_low": dict(family="Diamond", density_start=.15, density_end=.30),
    "Primitive_Schwartz_central": dict(family="Primitive Schwartz", density_start=.25, density_end=.50),
}


def effective_coefficient(points, tetra, size, rtol=1e-10):
    """P1 Laplace on one phase; returns dict with k_eff/k_bulk and solver data."""
    points = np.asarray(points, dtype=float)
    tetra = np.asarray(tetra)
    used = np.unique(tetra)
    remap = np.full(len(points), -1)
    remap[used] = np.arange(len(used))
    pts = points[used]
    tet = remap[tetra]
    tol = max(size) * 1e-9
    bottom = np.abs(pts[:, 2]) < tol
    top = np.abs(pts[:, 2] - size[2]) < tol
    # Drop components that touch no Dirichlet node (singular, zero flux anyway).
    adjacency = np.concatenate([tet[:, [0, j]] for j in (1, 2, 3)])
    graph = coo_matrix((np.ones(len(adjacency)), (adjacency[:, 0], adjacency[:, 1])),
                       shape=(len(pts), len(pts)))
    n_comp, label = connected_components(graph, directed=False)
    anchored = np.zeros(n_comp, dtype=bool)
    anchored[np.unique(label[bottom | top])] = True
    keep_cells = anchored[label[tet[:, 0]]]
    tet = tet[keep_cells]
    p = pts[tet]
    edges = np.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2)
    signed = np.linalg.det(edges) / 6
    if np.any(signed <= 0):
        raise ValueError("nonpositive tetrahedron")
    inverse = np.linalg.inv(edges)
    gradients = np.concatenate((-inverse.sum(axis=1)[:, None], inverse), axis=1)
    local = signed[:, None, None] * np.einsum("nik,njk->nij", gradients, gradients)
    matrix = coo_matrix((local.ravel(), (np.repeat(tet, 4, axis=1).ravel(), np.tile(tet, (1, 4)).ravel())),
                        shape=(len(pts), len(pts))).tocsr()
    active = np.zeros(len(pts), dtype=bool)
    active[np.unique(tet)] = True
    fixed = (bottom | top) & active
    free = np.flatnonzero(active & ~fixed)
    u = np.zeros(len(pts))
    u[top & active] = 1.0
    reduced = matrix[free][:, free]
    rhs = -matrix[free][:, np.flatnonzero(fixed)] @ u[fixed]
    diagonal = reduced.diagonal()
    pre = LinearOperator(reduced.shape, matvec=lambda x: x / diagonal)
    iterations = 0

    def count(_):
        nonlocal iterations
        iterations += 1

    t0 = time.perf_counter()
    sol, info = cg(reduced, rhs, M=pre, rtol=rtol, atol=0, maxiter=20000, callback=count)
    seconds = time.perf_counter() - t0
    if info != 0:
        raise RuntimeError(f"CG did not converge ({info})")
    u[free] = sol
    energy = float(u @ (matrix @ u))
    bulk = size[0] * size[1] / size[2]
    # Outward flux through the top face from the stiffness rows of the top nodes.
    flux_top = float((matrix[np.flatnonzero(top & active)] @ u).sum())
    return {"k_eff_over_bulk": energy / bulk, "flux_top_over_bulk": flux_top / bulk,
            "energy": energy, "unknowns": int(len(free)), "tetrahedra_used": int(len(tet)),
            "dropped_cells": int((~keep_cells).sum()), "components": int(n_comp),
            "cg_iterations": iterations, "solve_seconds": seconds,
            "phase_volume_mm3": float(signed.sum())}


def gmsh_solid(stl, size_h):
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.merge(str(stl))
        gmsh.model.mesh.classifySurfaces(np.radians(40), True, True, np.radians(180))
        gmsh.model.mesh.createGeometry()
        gmsh.option.setNumber("Mesh.MeshSizeMin", size_h * 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size_h)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in gmsh.model.getEntities(2)])
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.mesh.generate(3)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        index = {int(t): i for i, t in enumerate(tags)}
        _, _, enodes = gmsh.model.mesh.getElements(3)
        tets = np.array([[index[int(n)] for n in enodes[0][i:i + 4]] for i in range(0, len(enodes[0]), 4)])
        points = coords.reshape(-1, 3)
    finally:
        gmsh.finalize()
    # Orient positively.
    p = points[tets]
    s = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
    tets[s < 0] = tets[s < 0][:, [1, 0, 2, 3]]
    return points, tets


FIELDS = ["design", "resolution", "h_mm", "variant", "phase", "generation_seconds", "mesh_tetrahedra",
          "mesh_vertices", "k_eff_over_bulk", "flux_top_over_bulk", "energy", "unknowns", "tetrahedra_used",
          "dropped_cells", "components", "cg_iterations", "solve_seconds", "phase_volume_mm3", "error"]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16, 24])
    parser.add_argument("--designs", nargs="+", choices=list(DESIGNS), default=list(DESIGNS))
    parser.add_argument("--gmsh", action="store_true", help="also solve the solid phase on a Gmsh remesh")
    parser.add_argument("--save-meshes", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume this version's checkpointed study")
    parser.add_argument("--summarize-csv", type=Path, help="Recompute extrapolation from an existing CSV; do not solve")
    args = parser.parse_args(argv)
    if len(set(args.resolutions)) != len(args.resolutions) or any(r < 8 for r in args.resolutions):
        parser.error("resolutions must be distinct integers of at least 8")
    if args.resume and args.summarize_csv:
        parser.error("--resume and --summarize-csv cannot be combined")
    args.out.mkdir(parents=True, exist_ok=args.resume)
    csv_path = args.out / "transport_convergence.csv"
    study_path = args.out / "study.json"
    study = {"generator_version": __version__, "designs": args.designs,
             "resolutions": args.resolutions, "gmsh": args.gmsh, "save_meshes": args.save_meshes}
    rows = []
    if args.summarize_csv:
        with args.summarize_csv.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        shutil.copyfile(args.summarize_csv, csv_path)
        write_summary(args.out, rows, study, source_csv=str(args.summarize_csv.resolve()))
        return
    if args.resume:
        if json.loads(study_path.read_text(encoding="utf8")) != study:
            parser.error("Study options or generator version differ from the saved study.")
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    else:
        study_path.write_text(json.dumps(study, indent=2), encoding="utf8")
        with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writeheader()

    def row_key(row):
        return tuple(str(row[k]) for k in ("design", "resolution", "variant", "phase"))

    completed = {row_key(r) for r in rows if not r.get("error") and r.get("k_eff_over_bulk") not in (None, "")}

    def record(row):
        with csv_path.open("a", newline="", encoding="utf8") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        rows.append(row)
        if not row.get("error"):
            completed.add(row_key(row))

    for name in args.designs:
        design = DESIGNS[name]
        for res in args.resolutions:
            base = Config(resolution=res, domain_mode="solid_fluid", **design)
            variants = {"baseline": base,
                        "snapped_root": Config(**{**base.to_dict(), "snap_alpha": 0.25,
                                                  "edge_interpolation": "implicit_root"})}
            h = base.size[2] / (base.cells[2] * res)
            for variant, config in variants.items():
                need_phase = [label for label in ("solid", "fluid")
                              if (name, str(res), variant, label) not in completed]
                need_gmsh = args.gmsh and variant == "baseline" and (
                    name, str(res), "gmsh_remesh_solid", "solid") not in completed
                if not need_phase and not need_gmsh:
                    continue
                print(f"{name} res={res} {variant}", flush=True)
                t0 = time.perf_counter()
                model = generate(config)
                gen_s = time.perf_counter() - t0
                if args.save_meshes:
                    destination = args.out / f"{name}_r{res}_{variant}"
                    if not destination.exists():
                        save_model(model, destination)
                    elif json.loads((destination / "report.json").read_text(encoding="utf8"))["mesh_sha256"] != model["report"]["mesh_sha256"]:
                        raise ValueError(f"Saved mesh differs from regenerated mesh: {destination}")
                phases = model["phase_ids"]
                for phase, label in ((1, "solid"), (2, "fluid")):
                    if label not in need_phase:
                        continue
                    result = effective_coefficient(model["points"], model["tetra"][phases == phase], config.size)
                    record({"design": name, "resolution": res, "h_mm": h, "variant": variant,
                                 "phase": label, "generation_seconds": round(gen_s, 2),
                                 "mesh_tetrahedra": int(np.sum(phases == phase)),
                                 "mesh_vertices": int(len(model["points"])), **result})
                    print("   ", label, json.dumps({k: result[k] for k in ("k_eff_over_bulk", "cg_iterations", "unknowns")}), flush=True)
                if need_gmsh:
                    stl = args.out / f"_gmsh_{name}_r{res}.stl"
                    model["surface"].export(str(stl))
                    try:
                        t0 = time.perf_counter()
                        gp, gt = gmsh_solid(stl, h)
                        gmsh_s = time.perf_counter() - t0
                        result = effective_coefficient(gp, gt, config.size)
                        record({"design": name, "resolution": res, "h_mm": h, "variant": "gmsh_remesh_solid",
                                     "phase": "solid", "generation_seconds": round(gmsh_s, 2),
                                     "mesh_tetrahedra": int(len(gt)), "mesh_vertices": int(len(gp)), **result})
                        print("    gmsh solid", json.dumps({k: result[k] for k in ("k_eff_over_bulk", "cg_iterations")}), flush=True)
                    except Exception as exc:  # noqa: BLE001
                        record({"design": name, "resolution": res, "h_mm": h, "variant": "gmsh_remesh_solid",
                                     "phase": "solid", "error": str(exc)[:160]})
                    finally:
                        stl.unlink(missing_ok=True)
    write_summary(args.out, rows, study)


def write_summary(out, rows, study, source_csv=None):
    extrapolation = summarize_transport(rows)
    summary = {"generator_version": __version__, "python": platform.python_version(), "platform": platform.platform(),
               "dependencies": {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "trimesh")},
               **study, "richardson": extrapolation, "source_csv": source_csv,
               "extrapolation_method": "nonuniform three-grid power-law fit",
               "problem": "P1 steady diffusion per phase, u=0 at z=0, u=1 at z=Lz, zero flux elsewhere; k_eff/k_bulk = dissipation / (Lx*Ly/Lz)",
               "scope": "phase-wise scalar elliptic consistency and convergence; not Stokes, elasticity, FSI or COMSOL"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf8")
    hashes = {f.name: file_sha256(f) for f in out.iterdir() if f.is_file() and f.name != "sha256.json"}
    (out / "sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf8")
    print(json.dumps(extrapolation, indent=1))


if __name__ == "__main__":
    main()
