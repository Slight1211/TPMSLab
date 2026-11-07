"""Wall-resolution proxy for ITT vs dev7 and a density-offset sweep.

layers proxy = (solid vertices not on the interface or box) / (interface vertices)
measured in the thin (bottom 20 %) and thick (top 20 %) bands of a z-graded
sheet; a direct lattice mesher loses interior vertices where the wall is thin.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from itt_prototype import fixed_mask, transport, transport_adaptive  # noqa: E402
from itt_study import HARD  # noqa: E402
from tpmslab import Config, generate, optimize_mesh  # noqa: E402
from itt_prototype import tetra_metrics  # noqa: E402


def layers(model, points, size):
    tetra = model["tetra"]
    solid_vertices = np.unique(tetra[model["phase_ids"] == 1])
    surface = np.unique(model["interface"])
    on_box = fixed_mask(points, size).any(axis=1)
    interior = np.setdiff1d(solid_vertices, surface)
    interior = interior[~on_box[interior]]
    z = points[:, 2] / size[2]
    out = {}
    for band, (a, b) in {"thin": (0.0, 0.2), "thick": (0.8, 1.0)}.items():
        si = np.count_nonzero((z[interior] >= a) & (z[interior] <= b))
        ss = np.count_nonzero((z[surface] >= a) & (z[surface] <= b))
        out[band] = round(si / max(ss, 1), 3)
    return out


def main():
    rows = []
    for family, d0, d1 in (("Gyroid", .10, .50), ("Gyroid", .25, .50)):
        for res in (12, 16):
            common = dict(family=family, resolution=res, domain_mode="solid_fluid", snap_alpha=.25,
                          edge_interpolation="implicit_root")
            target = Config(density_start=d0, density_end=d1, **common)
            mid = (d0 + d1) / 2
            ref = Config(density_start=mid, density_end=mid, gradient="uniform", **common)
            size = np.asarray(target.size)
            direct = optimize_mesh(generate(target))
            template = optimize_mesh(generate(ref), **HARD)
            moved, log = transport_adaptive(template, ref, target, guard=0.05)
            repaired = optimize_mesh(moved, rounds=1)
            for name, m in (("dev7", direct), ("template", template), ("itt_transport", moved), ("itt_r1", repaired)):
                v, q = tetra_metrics(m["points"][m["tetra"]])
                s = m["phase_ids"] == 1
                row = {"family": family, "density": [d0, d1], "resolution": res, "variant": name,
                       "inverted": int((v <= 0).sum()), "q_p01": round(float(np.quantile(q[s], .01)), 4),
                       "q_min": round(float(q.min()), 4), **layers(m, m["points"], size)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    # density-offset sweep: template at 0.375, uniform targets at 0.375 +/- delta
    common = dict(family="Gyroid", resolution=12, domain_mode="solid_fluid", snap_alpha=.25,
                  edge_interpolation="implicit_root", gradient="uniform")
    ref = Config(density_start=.375, density_end=.375, **common)
    template = optimize_mesh(generate(ref), **HARD)
    for delta in (-.2, -.125, -.0625, -.03, .03, .0625, .125, .2):
        target = Config(density_start=.375 + delta, density_end=.375 + delta, **common)
        P, _ = transport(template, ref, target)
        v, q = tetra_metrics(P[template["tetra"]])
        s = template["phase_ids"] == 1
        row = {"sweep_delta": delta, "inverted": int((v <= 0).sum()), "q_p01": round(float(np.quantile(q[s], .01)), 4),
               "q_min": round(float(q.min()), 4)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
