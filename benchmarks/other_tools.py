"""Other meshers on the bcc_crosscompare designs and density levels.

  tetgen     TetGen 1.6 (python `tetgen`), switches pq1.414/10 + maximum volume
             of a regular tetrahedron with edge h, on the TPMS Lab BCC solid
             boundary (Steiner points may split boundary faces; geometry kept)
  ftetwild   fTetWild (python `pytetwild`), target edge h, envelope 1e-3 of the
             bounding-box diagonal, single thread, same boundary as input
  microgen   microgen 1.2.1 Tpms.generate_grid_vtk("sheet") + triangulate: VTK
             clip of a structured hexahedral grid (its native volume output),
             same field, phase and graded threshold; resolution chosen for a
             similar element count. Runs in its own venv (needs numpy<2).
Seconds for tetgen/ftetwild include the TPMS Lab BCC generation that supplies
the boundary. Rows are appended in the bcc_crosscompare CSV format.
"""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quality_vs_gmsh as qvg  # noqa: E402
from bcc_crosscompare import FIELDS, LEVELS  # noqa: E402
from tpmslab import Config, generate  # noqa: E402
from tpmslab.model import field_and_threshold  # noqa: E402
from tpmslab.quality import signed_volumes  # noqa: E402

MICROGEN_RES = [11, 13, 15, 19]


def oriented(points, tets):
    tets = np.asarray(tets, dtype=np.int64).copy()
    s = signed_volumes(points[tets])
    tets[s < 0] = tets[s < 0][:, [1, 0, 2, 3]]
    return tets[np.abs(s) > 0]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--designs", nargs="+", default=list(qvg.DESIGNS))
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(len(LEVELS))))
    parser.add_argument("--tools", nargs="+", default=["tetgen", "ftetwild", "microgen"])
    parser.add_argument("--microgen-python", default="/home/claude/venv_microgen/bin/python")
    parser.add_argument("--microgen-worker", default=str(HERE / "microgen_worker.py"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    transport = qvg.load_transport()
    path = args.out / "other_tools.csv"
    if not path.is_file():
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writeheader()

    def record(row):
        with path.open("a", newline="", encoding="utf8") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore").writerow(row)
        print(json.dumps({k: row.get(k) for k in ("design", "level", "mesher", "tetrahedra", "q_min", "q_p01",
                                                   "q_median", "volume_error_pp", "total_seconds", "error")}), flush=True)

    for name in args.designs:
        design = qvg.DESIGNS[name]
        reference = qvg.reference_volume(Config(**design), 256)
        z = np.linspace(0, 5, 2001)
        base = Config(**design)
        _, q = field_and_threshold(base, np.zeros_like(z), np.zeros_like(z), z)
        with tempfile.TemporaryDirectory(prefix="other_") as temporary:
            table = Path(temporary) / "q.npz"
            np.savez(table, z=z, q=np.broadcast_to(q, z.shape), phase_degrees=np.array(base.phase_degrees))
            for level in args.levels:
                _, bcc_res = LEVELS[level]
                config = Config(**design, resolution=bcc_res, lattice="bcc", snap_alpha=0.3,
                                domain_mode="solid_fluid", edge_interpolation="implicit_root")
                h = config.size[2] / bcc_res
                t0 = time.perf_counter()
                model = generate(config)
                gen = time.perf_counter() - t0
                faces = np.asarray(model["solid_boundary"])
                used = np.unique(faces)
                remap = -np.ones(len(model["points"]), np.int64)
                remap[used] = np.arange(len(used))
                v, f = model["points"][used], remap[faces].astype(np.int32)
                for tool in args.tools:
                    common = {"design": name, "level": level, "mesher": tool, "h_mm": h}
                    try:
                        if tool == "tetgen":
                            import tetgen
                            t0 = time.perf_counter()
                            nodes, elem, _, _ = tetgen.TetGen(v, f).tetrahedralize(
                                switches=f"pq1.414/10Qa{0.118 * h ** 3:.8f}")
                            seconds = time.perf_counter() - t0 + gen
                            points, tets, resolution = np.asarray(nodes), oriented(nodes, elem), bcc_res
                        elif tool == "ftetwild":
                            import pytetwild
                            t0 = time.perf_counter()
                            nodes, elem = pytetwild.tetrahedralize(v.astype(np.float64), f, edge_length_abs=h,
                                                                   epsilon=1e-3, num_threads=1, quiet=True)
                            seconds = time.perf_counter() - t0 + gen
                            points, tets, resolution = np.asarray(nodes), oriented(nodes, elem), bcc_res
                        else:
                            resolution = MICROGEN_RES[level]
                            out = Path(temporary) / "mg.npz"
                            t0 = time.perf_counter()
                            subprocess.run([args.microgen_python, args.microgen_worker, base.family, str(table),
                                            str(resolution), str(out)], check=True, capture_output=True, timeout=900)
                            with np.load(out) as data:
                                points, seconds = data["points"], float(data["seconds"])
                                tets = oriented(points, data["tets"])
                        record({**common, "resolution": resolution, "generation_seconds": round(seconds, 3),
                                "total_seconds": round(seconds, 3),
                                **qvg.quality_row(points, tets, config, reference, transport)})
                    except Exception as exc:  # noqa: BLE001
                        record({**common, "error": f"{type(exc).__name__}: {str(exc)[:150]}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
