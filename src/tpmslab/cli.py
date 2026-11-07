"""Command-line entry points; core commands do not import Flask or COMSOL."""

import argparse
import json
import os
from pathlib import Path
from . import Config, generate, list_families, save_model, condition_mesh, optimize_mesh, quality_audit, __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tpmslab", description="Direct TPMS solid volume meshes")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("families", help="List built-in formulas and aliases")
    init = sub.add_parser("init-config", help="Write a default JSON configuration")
    init.add_argument("path", type=Path)
    make = sub.add_parser("generate", help="Generate and validate a volume mesh")
    make.add_argument("--config", type=Path)
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--mph", action="store_true")
    make.add_argument("--solve", action="store_true")
    make.add_argument("--snap-alpha", type=float, default=None, help="Snap lattice vertices within this edge fraction (0-0.5) of a cut point onto the surface before clipping; overrides snap_alpha in the config")
    make.add_argument("--snap-alpha-boundary", type=float, default=None, help="Snap threshold for lattice vertices on the outer box (default: same as --snap-alpha; 0.35 recommended with --lattice bcc)")
    make.add_argument("--lattice", choices=("kuhn", "bcc"), default=None, help="Background lattice: kuhn (6 tetrahedra per voxel, default) or bcc (body-centred cubic, congruent tetrahedra of mean ratio 0.952; about twice the cells per voxel, so use about 5/6 of the resolution)")
    make.add_argument("--snap-target", choices=("implicit", "linear"), default=None, help="Snap onto the exact implicit root (default) or the linear-cut point")
    make.add_argument("--periodic-faces", action="store_true", help="Identical node sets and triangulations on opposite box faces along every axis where the design repeats (for periodic boundary conditions)")
    make.add_argument("--optimize", action="store_true", help="Cut-layer reconnection, interior smoothing and implicit-surface sliding of boundary nodes (recommended with --snap-alpha 0.25)")
    make.add_argument("--optimize-threshold", type=float, default=0.5, help="Cells below this mean-ratio quality are targeted by --optimize")
    make.add_argument("--optimize-rounds", type=int, default=6)
    make.add_argument("--surface-envelope", type=float, default=None, help="Let interface nodes leave the exact surface by at most this distance in mm when it improves quality (default: exact surface)")
    make.add_argument("--optimize-effort", choices=("standard", "high"), default="standard", help="high: target all cells below quality 0.7 (slower, higher p01)")
    make.add_argument("--condition", action="store_true", help="Improve interior nodes with fixed boundaries and an audited quality gate")
    make.add_argument("--condition-max-nodes", type=int, default=256)
    make.add_argument("--condition-sweeps", type=int, default=2)
    make.add_argument("--min-quality", type=float, default=0.0, help="Reject export if any phase has a lower mean-ratio quality")
    make.add_argument("--max-poor-fraction", type=float, default=1.0, help="Reject if the fraction of Q<0.01 cells in any phase exceeds this value")
    run = sub.add_parser("comsol", help="Run a saved export in COMSOL; each attempt is preserved")
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--import-only", action="store_true", help="Create and reopen MPH without solving")
    run.add_argument("--material", type=Path, help="Elasticity material JSON with E, nu, rho (SI units)")
    for target in (make, run):
        # The same COMSOL launch options apply to `generate --mph/--solve` and `comsol`.
        target.add_argument("--max-tetrahedra", type=int, default=180_000, help="Demo solver limit on volume elements")
        target.add_argument("--timeout", type=int, default=600, help="COMSOL import/solve/reopen timeout in seconds (1-3600)")
        target.add_argument("--preferences-directory", type=Path, help="Explicit trusted COMSOL preference files; copied per attempt")
        target.add_argument("--element-order", type=int, choices=(1, 2), default=1, help="1: linear (verified default); 2: quadratic displacement / P2+P1 flow (not yet covered by shipped evidence)")
        target.add_argument("--np", type=int, default=2, dest="processes", help="COMSOL -np processes")
    quality = sub.add_parser("remesh-quality", help="Optionally remesh a verified two-phase export in COMSOL (not yet exercised in shipped evidence)")
    quality.add_argument("--source", type=Path, required=True)
    quality.add_argument("--out", type=Path, required=True)
    quality.add_argument("--volume-size", type=float, default=7.0)
    quality.add_argument("--face-size", type=float, default=0.0)
    quality.add_argument("--temp-directory", type=Path)
    quality.add_argument("--preferences-directory", type=Path, help="Explicit trusted COMSOL preference files; copied next to the remesh")
    quality.add_argument("--np", type=int, default=2, dest="processes", help="COMSOL -np processes")
    web = sub.add_parser("web", help="Start the optional local browser interface")
    web.add_argument("--output", type=Path, default=Path.cwd() / "tpmslab-output")
    web.add_argument("--port", type=int, default=0)
    web.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "families":
            print(json.dumps(list_families(), indent=2))
            return 0
        if args.command == "init-config":
            with args.path.open("x", encoding="utf8") as stream:
                json.dump(Config().to_dict(), stream, indent=2)
            return 0
        if args.command == "remesh-quality":
            from .comsol_remesh import remesh_in_comsol

            evidence = remesh_in_comsol(
                args.source, args.out, volume_size=args.volume_size,
                face_size=args.face_size, temp_directory=args.temp_directory,
                preferences_directory=args.preferences_directory, processes=args.processes,
            )
            print(json.dumps(evidence, indent=2))
            return 0
        if args.command == "comsol":
            from .comsol import run_comsol

            material = json.loads(args.material.read_text(encoding="utf-8-sig")) if args.material else None
            result = run_comsol(args.source, solve=not args.import_only, material=material,
                                max_solve_tetrahedra=args.max_tetrahedra,
                                build_timeout_s=args.timeout, preferences_directory=args.preferences_directory,
                                element_order=args.element_order, processes=args.processes,
                                progress=print)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0
        if args.command == "web":
            os.environ["TPMSLAB_OUTPUT_DIR"] = str(args.output.resolve())
            try:
                from .web import serve
            except ImportError as exc:
                raise RuntimeError('Install the web extra: pip install "tpmslab[web]"') from exc
            serve(args.port, args.no_browser)
            return 0
        data = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
        if not isinstance(data, dict):
            raise ValueError("Parameters must be a JSON object")
        request_conditioning = data.pop("condition", False)
        request_optimization = data.pop("optimize", False)
        if type(request_conditioning) is not bool or type(request_optimization) is not bool:
            raise ValueError("condition and optimize must be booleans")
        request_conditioning = args.condition or request_conditioning
        request_optimization = args.optimize or request_optimization
        if args.snap_alpha is not None:
            data["snap_alpha"] = args.snap_alpha
        if args.snap_target is not None:
            data["snap_target"] = args.snap_target
        if args.snap_alpha_boundary is not None:
            data["snap_alpha_boundary"] = args.snap_alpha_boundary
        if args.lattice is not None:
            data["lattice"] = args.lattice
        if args.periodic_faces:
            data["periodic_faces"] = True
        config = Config.from_dict(data)
        if args.out.exists():
            raise FileExistsError(f"Output already exists: {args.out}")
        model = generate(config, progress=print)
        if request_optimization:
            model = optimize_mesh(model, quality_threshold=args.optimize_threshold,
                                  rounds=args.optimize_rounds, effort=args.optimize_effort,
                                  surface_envelope=args.surface_envelope, progress=print)
        elif request_conditioning:
            model = condition_mesh(model, max_nodes=args.condition_max_nodes,
                                   sweeps=args.condition_sweeps, progress=print)
        if request_conditioning or request_optimization or args.min_quality != 0 or args.max_poor_fraction != 1:
            model["report"]["quality_audit"] = quality_audit(
                model, minimum_quality=args.min_quality,
                max_poor_fraction=args.max_poor_fraction,
            )
        save_model(model, args.out)
        if args.mph or args.solve:
            from .comsol import build_mph

            build_mph(args.out, model["report"], solve=args.solve, progress=print,
                      max_solve_tetrahedra=args.max_tetrahedra, build_timeout_s=args.timeout,
                      preferences_directory=args.preferences_directory,
                      element_order=args.element_order, processes=args.processes)
        print(str(args.out.resolve()))
        return 0
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(2, f"tpmslab: {exc}\n")


if __name__ == "__main__":
    main()
