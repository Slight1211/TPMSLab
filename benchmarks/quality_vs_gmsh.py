"""Element-quality and accuracy comparison: TPMS Lab variants versus a Gmsh remesh.

For each design and resolution the solid phase is produced four ways:

* ``baseline``       - TPMS Lab, linear cuts (snap_alpha=0)
* ``snapped_root``   - TPMS Lab, snap_alpha=0.25 + edge_interpolation="implicit_root"
* ``snapped_root_conditioned`` - the same plus bounded interior conditioning
* ``snapped_root_optimized``   - the same plus cut-layer reconnection, interior
  smoothing and implicit-surface sliding of boundary vertices (``optimize_mesh``)
* ``gmsh_<variant>`` - Gmsh (Delaunay + Netgen optimisation) re-meshing the
  *same* boundary surface exported by that TPMS Lab variant, so both meshes
  bound exactly the same polyhedron and differ only in the interior tetrahedra.

Reported per mesh: tetrahedra, mean-ratio quality (min, p01, median, share
below 0.01 and 0.1), solid volume against a dense midpoint sampling of the
exact implicit field, generation seconds, and the P1 effective diffusion
coefficient from ``transport_convergence.effective_coefficient``.

This is a like-for-like interior-quality comparison on a fixed boundary; it
is not a claim about Gmsh's own surface meshing, nTop, or any commercial tool.
"""

import argparse
import csv
import importlib.util
import json
import multiprocessing
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np

from tpmslab import Config, generate, condition_mesh, optimize_mesh, __version__
from tpmslab.model import raw_field
from tpmslab.quality import tetra_metrics

HERE = Path(__file__).resolve().parent
DESIGNS = {
    "Gyroid_central": dict(family="Gyroid", density_start=.25, density_end=.50),
    "Diamond_low": dict(family="Diamond", density_start=.15, density_end=.30),
    "Primitive_Schwartz_central": dict(family="Primitive Schwartz", density_start=.25, density_end=.50),
    "IWP_sheet": dict(family="IWP", density_start=.30, density_end=.45),
}
FIELDS = ["design", "resolution", "h_mm", "variant", "mesher", "tetrahedra", "vertices", "generation_seconds",
          "q_min", "q_p01", "q_median", "q_below_001_fraction", "q_below_01_fraction",
          "solid_volume_mm3", "reference_volume_mm3", "volume_error_pp", "k_eff_over_bulk",
          "cg_iterations", "unknowns", "error"]


def load_transport():
    spec = importlib.util.spec_from_file_location("transport_convergence", HERE / "transport_convergence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_volume(config, samples=256):
    """Dense midpoint integration of the exact implicit solid indicator."""
    axes = [(np.arange(samples) + 0.5) * L / samples for L in config.size]
    total = 0.0
    for z in axes[2]:
        X, Y = np.meshgrid(axes[0], axes[1], indexing="ij")
        total += float(np.count_nonzero(raw_field(config, X, Y, np.full_like(X, z)) >= 0))
    return total * float(np.prod(config.size)) / samples ** 3


def gmsh_fixed_boundary(points, faces, size_h):
    """Tetrahedralise the interior of a closed triangle surface without touching it.

    The surface triangles are handed to Gmsh as a discrete surface (no
    reparametrisation, no surface remeshing), so the resulting volume mesh has
    exactly the given boundary; only interior nodes and tetrahedra are new.
    """
    import gmsh
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    used = np.unique(faces)
    remap = np.full(len(points), -1)
    remap[used] = np.arange(len(used))
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        surface = gmsh.model.addDiscreteEntity(2)
        node_tags = np.arange(1, len(used) + 1)
        gmsh.model.mesh.addNodes(2, surface, node_tags, points[used].ravel())
        tri = remap[faces] + 1
        gmsh.model.mesh.addElementsByType(surface, 2, np.arange(1, len(tri) + 1), tri.ravel())
        loop = gmsh.model.geo.addSurfaceLoop([surface])
        volume = gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", size_h * 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size_h)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.mesh.generate(3)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        index = {int(t): i for i, t in enumerate(tags)}
        _, _, enodes = gmsh.model.mesh.getElements(3, volume)
        tets = np.array([[index[int(n)] for n in enodes[0][i:i + 4]] for i in range(0, len(enodes[0]), 4)])
        out = coords.reshape(-1, 3)
        # The first len(used) tags are the input surface nodes; verify they did not move.
        kept = np.array([index[int(t)] for t in node_tags])
        if not np.array_equal(out[kept], points[used]):
            raise RuntimeError("Gmsh moved boundary nodes")
    finally:
        gmsh.finalize()
    p = out[tets]
    s = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
    tets[s < 0] = tets[s < 0][:, [1, 0, 2, 3]]
    return out, tets


def _gmsh_worker(mode, payload, size_h, out_path):
    import numpy as np  # noqa: F811 - fresh interpreter
    transport = load_transport()
    if mode == "fixed":
        points, faces = np.load(payload["points"]), np.load(payload["faces"])
        gp, gt = gmsh_fixed_boundary(points, faces, size_h)
    else:
        gp, gt = transport.gmsh_solid(payload["stl"], size_h)
    np.savez(out_path, points=gp, tets=gt)


def gmsh_isolated(mode, payload, size_h, timeout_s=900):
    """Run a Gmsh mode in a child process so a Gmsh crash or hang cannot kill the study."""
    with tempfile.TemporaryDirectory(prefix="gmsh_") as temporary:
        out_path = Path(temporary) / "mesh.npz"
        context = multiprocessing.get_context("spawn")
        process = context.Process(target=_gmsh_worker, args=(mode, payload, size_h, str(out_path)))
        process.start()
        process.join(timeout_s)
        if process.is_alive():
            process.kill()
            process.join()
            raise RuntimeError(f"Gmsh {mode} mode exceeded {timeout_s} s")
        if process.exitcode != 0:
            raise RuntimeError(f"Gmsh {mode} mode exited with code {process.exitcode} (crash or error)")
        with np.load(out_path) as data:
            return data["points"], data["tets"]


def quality_row(points, tetra, config, reference, transport):
    volumes, quality = tetra_metrics(points[tetra])
    solid = float(volumes.sum())
    row = {"tetrahedra": int(len(tetra)), "vertices": int(len(np.unique(tetra))),
           "q_min": float(quality.min()), "q_p01": float(np.quantile(quality, .01)),
           "q_median": float(np.median(quality)),
           "q_below_001_fraction": float(np.mean(quality < .01)),
           "q_below_01_fraction": float(np.mean(quality < .1)),
           "solid_volume_mm3": solid, "reference_volume_mm3": reference,
           "volume_error_pp": (solid - reference) / float(np.prod(config.size)) * 100}
    result = transport.effective_coefficient(points, tetra, config.size)
    row.update(k_eff_over_bulk=result["k_eff_over_bulk"], cg_iterations=result["cg_iterations"],
               unknowns=result["unknowns"])
    return row


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--designs", nargs="+", choices=list(DESIGNS), default=list(DESIGNS))
    parser.add_argument("--reference-samples", type=int, default=256)
    parser.add_argument("--no-gmsh", action="store_true")
    parser.add_argument("--gmsh-modes", nargs="+", choices=("fixed", "surface"), default=["fixed", "surface"],
                        help="fixed: same boundary, interior only; surface: Gmsh also reparametrises/remeshes the surface (can be slow or fail on complex sheets)")
    parser.add_argument("--gmsh-timeout", type=int, default=900, help="seconds per Gmsh run (child process)")
    parser.add_argument("--resume", action="store_true", help="skip (design, resolution, variant, mesher) rows already in the CSV")
    parser.add_argument("--condition-nodes", type=int, default=4096)
    parser.add_argument("--condition-sweeps", type=int, default=3)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    transport = load_transport()
    csv_path = args.out / "quality_vs_gmsh.csv"
    rows = []
    if args.resume and csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = [r for r in csv.DictReader(stream) if not r.get("error")]
    else:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writeheader()
    done = {(r["design"], str(r["resolution"]), r["variant"], r["mesher"]) for r in rows}

    def finished(design, res, variant, mesher):
        return (design, str(res), variant, mesher) in done

    def record(row):
        rows.append(row)
        with csv_path.open("a", newline="", encoding="utf8") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writerow({k: row.get(k, "") for k in FIELDS})
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({k: row.get(k) for k in ("design", "resolution", "variant", "mesher", "tetrahedra",
                                                   "q_p01", "q_below_001_fraction", "volume_error_pp",
                                                   "k_eff_over_bulk", "generation_seconds", "error")}), flush=True)

    for name in args.designs:
        design = DESIGNS[name]
        reference = reference_volume(Config(**design), args.reference_samples)
        for res in args.resolutions:
            base = Config(resolution=res, domain_mode="solid_fluid", **design)
            h = base.size[2] / (base.cells[2] * res)
            variants = {
                "baseline": base,
                "snapped_root": Config(**{**base.to_dict(), "snap_alpha": 0.25, "edge_interpolation": "implicit_root"}),
            }
            for variant, config in variants.items():
                common = {"design": name, "resolution": res, "h_mm": h}
                wanted = [(variant, "tpmslab")]
                if variant == "snapped_root":
                    wanted += [("snapped_root_conditioned", "tpmslab"), ("snapped_root_optimized", "tpmslab")]
                if not args.no_gmsh:
                    wanted += [(variant, "gmsh_fixed_boundary")] if "fixed" in args.gmsh_modes else []
                    wanted += [(variant, "gmsh_remeshed_surface")] if "surface" in args.gmsh_modes else []
                if all(finished(name, res, v, m) for v, m in wanted):
                    continue
                t0 = time.perf_counter()
                model = generate(config)
                seconds = time.perf_counter() - t0
                solid = model["phase_ids"] == 1
                if not finished(name, res, variant, "tpmslab"):
                    try:
                        record({**common, "variant": variant, "mesher": "tpmslab", "generation_seconds": round(seconds, 2),
                                **quality_row(model["points"], model["tetra"][solid], config, reference, transport)})
                    except Exception as exc:  # noqa: BLE001
                        record({**common, "variant": variant, "mesher": "tpmslab", "error": str(exc)[:160]})
                if variant == "snapped_root" and not finished(name, res, "snapped_root_conditioned", "tpmslab"):
                    t0 = time.perf_counter()
                    conditioned = condition_mesh(model, max_nodes=args.condition_nodes, sweeps=args.condition_sweeps)
                    extra = time.perf_counter() - t0
                    try:
                        record({**common, "variant": "snapped_root_conditioned", "mesher": "tpmslab",
                                "generation_seconds": round(seconds + extra, 2),
                                **quality_row(conditioned["points"], conditioned["tetra"][solid], config, reference, transport)})
                    except Exception as exc:  # noqa: BLE001
                        record({**common, "variant": "snapped_root_conditioned", "mesher": "tpmslab", "error": str(exc)[:160]})
                if variant == "snapped_root" and not finished(name, res, "snapped_root_optimized", "tpmslab"):
                    t0 = time.perf_counter()
                    try:
                        optimized = optimize_mesh(model)
                        extra = time.perf_counter() - t0
                        osolid = optimized["phase_ids"] == 1
                        record({**common, "variant": "snapped_root_optimized", "mesher": "tpmslab",
                                "generation_seconds": round(seconds + extra, 2),
                                **quality_row(optimized["points"], optimized["tetra"][osolid], config, reference, transport)})
                    except Exception as exc:  # noqa: BLE001
                        record({**common, "variant": "snapped_root_optimized", "mesher": "tpmslab", "error": str(exc)[:160]})
                if not args.no_gmsh and "fixed" in args.gmsh_modes and not finished(name, res, variant, "gmsh_fixed_boundary"):
                    # (a) identical boundary: Gmsh only fills the interior
                    try:
                        t0 = time.perf_counter()
                        with tempfile.TemporaryDirectory(prefix="tpms_") as temporary:
                            np.save(Path(temporary) / "points.npy", model["points"])
                            np.save(Path(temporary) / "faces.npy", model["solid_boundary"])
                            gp, gt = gmsh_isolated("fixed", {"points": str(Path(temporary) / "points.npy"),
                                                             "faces": str(Path(temporary) / "faces.npy")}, h,
                                                   args.gmsh_timeout)
                        gseconds = time.perf_counter() - t0
                        record({**common, "variant": variant, "mesher": "gmsh_fixed_boundary",
                                "generation_seconds": round(gseconds, 2),
                                **quality_row(gp, gt, config, reference, transport)})
                    except Exception as exc:  # noqa: BLE001
                        record({**common, "variant": variant, "mesher": "gmsh_fixed_boundary", "error": str(exc)[:160]})
                if not args.no_gmsh and "surface" in args.gmsh_modes and not finished(name, res, variant, "gmsh_remeshed_surface"):
                    # (b) Gmsh also reparametrises and remeshes the surface
                    stl = args.out / f"_{name}_r{res}_{variant}.stl"
                    model["surface"].export(str(stl))
                    try:
                        t0 = time.perf_counter()
                        gp, gt = gmsh_isolated("surface", {"stl": str(stl)}, h, args.gmsh_timeout)
                        gseconds = time.perf_counter() - t0
                        record({**common, "variant": variant, "mesher": "gmsh_remeshed_surface",
                                "generation_seconds": round(gseconds, 2),
                                **quality_row(gp, gt, config, reference, transport)})
                    except Exception as exc:  # noqa: BLE001
                        record({**common, "variant": variant, "mesher": "gmsh_remeshed_surface", "error": str(exc)[:160]})
                    finally:
                        stl.unlink(missing_ok=True)
    summary = {"generator_version": __version__, "python": platform.python_version(), "platform": platform.platform(),
               "designs": args.designs, "resolutions": args.resolutions, "reference_samples": args.reference_samples,
               "rows": rows,
               "scope": "same boundary polyhedron for every mesher; interior quality, exact-field volume error and P1 diffusion coefficient"}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
