"""Density realization and exact grid-aligned slab volumes for reference specimens."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from tpmslab import Config, generate
from tpmslab.model import raw_field
from tpmslab.verification import audit_interface


def load_mesh(folder):
    a = np.load(Path(folder) / "mesh.npz")
    return dict(
        points=a["points_mm"],
        tetra=a["tetrahedra"],
        phase_ids=a["phase_ids"],
        interface=a["interface"],
        domains=a["domain_ids"],
    )


def geometry_record(model, config):
    p = model["points"][model["tetra"]]
    v = (
        np.abs(
            np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
        )
        / 6
    )
    solid = model["phase_ids"] == 1
    n = config.resolution * config.cells[2]
    z = np.clip(np.floor(p[:, :, 2].mean(axis=1) / config.size[2] * n).astype(int), 0, n - 1)
    bins = np.bincount(z[solid], weights=v[solid], minlength=n)
    expected = (config.density_start + config.density_end) / 2
    result = dict(
        family=config.family,
        resolution=config.resolution,
        target_average=expected,
        mesh_density=float(v[solid].sum() / np.prod(config.size)),
        slab_z_mm=((np.arange(n) + 0.5) * config.size[2] / n).tolist(),
        slab_density=(bins / (np.prod(config.size) / n)).tolist(),
        audit=audit_interface(model),
    )
    result["relative_target_deviation"] = (result["mesh_density"] - expected) / expected
    return result


def field_reference(config, n):
    # Independent midpoint integration of the unlinearized implicit field;
    # it uses the same threshold calibration, not the tetrahedral mesh.
    x = (np.arange(n) + 0.5) * config.size[0] / n
    y = (np.arange(n) + 0.5) * config.size[1] / n
    count = 0
    for z in (np.arange(n) + 0.5) * config.size[2] / n:
        count += np.count_nonzero(raw_field(config, x[:, None], y[None, :], z) >= 0)
    return count / n**3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for family in ["Gyroid", "Primitive Schwartz", "Diamond"]:
        c = Config(family=family, domain_mode="solid_fluid", resolution=12)
        model = load_mesh(args.source / family.replace(" ", "_")) if args.source else generate(c)
        r = geometry_record(model, c)
        r["implicit_midpoint_128"] = field_reference(c, 128)
        r["implicit_midpoint_256"] = field_reference(c, 256)
        records.append(r)
        for lo, hi in [(0.125, 0.375), (0.375, 0.625)]:
            cc = replace(c, density_start=lo, density_end=hi)
            m = generate(cc)
            records.append(geometry_record(m, cc))
        print("DENSITY", family, flush=True)
    (args.out / "density_results.json").write_text(json.dumps(records, indent=2), "utf8")


if __name__ == "__main__":
    main()
