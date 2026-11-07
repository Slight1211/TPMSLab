"""BCC versus Kuhn background lattice on the quality_vs_gmsh designs.

For each design and Kuhn resolution r the BCC lattice is run at
round(r * 5 / 6), which gives a comparable element count (BCC has twice the
tetrahedra per voxel). Both use vertex snapping onto the exact surface and
implicit-root edge cuts; the optimised rows use ``optimize_mesh`` defaults.
Reported per row: solid-phase quality, exact-field volume error, P1 k_eff
(transport_convergence.effective_coefficient) and wall-clock seconds.
"""
import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("qvg", HERE / "quality_vs_gmsh.py")
qvg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qvg)
from tpmslab import Config, generate, optimize_mesh, __version__  # noqa: E402

FIELDS = ["design", "kuhn_resolution", "lattice", "resolution", "snap_alpha", "stage", "h_mm",
          "tetrahedra_all", "generation_seconds", "total_seconds"] + [
          f for f in qvg.FIELDS if f not in ("design", "resolution", "h_mm", "variant", "mesher",
                                             "generation_seconds", "error")] + ["error"]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--designs", nargs="+", default=list(qvg.DESIGNS))
    parser.add_argument("--bcc-alpha", type=float, nargs="+", default=[0.4])
    parser.add_argument("--kuhn-alpha", type=float, default=0.25)
    parser.add_argument("--skip-kuhn", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    transport = qvg.load_transport()
    path = args.out / "bcc_vs_kuhn.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        csv.DictWriter(stream, fieldnames=FIELDS).writeheader()
    for name in args.designs:
        design = qvg.DESIGNS[name]
        reference = qvg.reference_volume(Config(**design), 256)
        for r in args.resolutions:
            runs = [] if args.skip_kuhn else [("kuhn", r, args.kuhn_alpha)]
            runs += [("bcc", max(8, round(r * 5 / 6)), a) for a in args.bcc_alpha]
            for lattice, res, alpha in runs:
                config = Config(**design, resolution=res, domain_mode="solid_fluid", lattice=lattice,
                                snap_alpha=alpha, edge_interpolation="implicit_root")
                h = config.size[2] / (config.cells[2] * res)
                common = {"design": name, "kuhn_resolution": r, "lattice": lattice, "resolution": res,
                          "snap_alpha": alpha, "h_mm": h}
                try:
                    t0 = time.perf_counter()
                    model = generate(config)
                    t1 = time.perf_counter()
                    optimized = optimize_mesh(model)
                    t2 = time.perf_counter()
                    for stage, m, total in (("generated", model, t1 - t0), ("optimized", optimized, t2 - t0)):
                        solid = m["phase_ids"] == 1
                        row = {**common, "stage": stage, "tetrahedra_all": int(len(m["tetra"])),
                               "generation_seconds": round(t1 - t0, 3), "total_seconds": round(total, 3),
                               **qvg.quality_row(m["points"], m["tetra"][solid], config, reference, transport)}
                        row.pop("resolution", None)
                        row["resolution"] = res
                        with path.open("a", newline="", encoding="utf8") as stream:
                            csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore").writerow(row)
                        print(json.dumps({k: row.get(k) for k in ("design", "kuhn_resolution", "lattice", "resolution",
                                                                   "snap_alpha", "stage", "tetrahedra", "q_min", "q_p01",
                                                                   "q_median", "volume_error_pp", "k_eff_over_bulk",
                                                                   "total_seconds")}), flush=True)
                except Exception as exc:  # noqa: BLE001
                    with path.open("a", newline="", encoding="utf8") as stream:
                        csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore").writerow(
                            {**common, "error": str(exc)[:200]})
                    print(json.dumps({**common, "error": str(exc)[:200]}), flush=True)
    (args.out / "summary.json").write_text(json.dumps({"generator_version": __version__, "designs": args.designs,
                                                         "resolutions": args.resolutions}, indent=2), encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
