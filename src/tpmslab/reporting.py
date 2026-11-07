"""Geometry statistics reconstructed from the final exported linear cells."""

import numpy as np


def density_profile(points, tetra, volumes, solid_mask, config):
    """Centroid-binned slab fractions, conserving the total solid volume.

    These are approximate spatial bins, not exact tetrahedron/slab intersections.
    ``slab_count`` makes the weighting explicit when bins have unequal widths.
    """
    axis = "xyz".index(config.axis)
    n_axis = int(config.cells[axis] * config.resolution)
    centroids = np.asarray(points)[np.asarray(tetra)].mean(axis=1)[:, axis]
    slab = np.clip((centroids / config.size[axis] * n_axis).astype(int), 0, n_axis - 1)
    solid_mask = np.asarray(solid_mask, dtype=bool)
    solid_by_slab = np.bincount(slab[solid_mask], weights=np.asarray(volumes)[solid_mask], minlength=n_axis)
    slab_volume = float(np.prod(config.size)) / n_axis
    return [{"position_mm": float((ix.mean() + 0.5) * config.size[axis] / n_axis),
             "sampled_density": float(solid_by_slab[ix].sum() / (slab_volume * len(ix))),
             "slab_count": int(len(ix))}
            for ix in np.array_split(np.arange(n_axis), min(10, n_axis))]


def refresh_geometry_statistics(model, config, volumes):
    """Update the report from final geometry after reconnection or smoothing."""
    report = model["report"]
    points, tetra = model["points"], model["tetra"]
    phases = model.get("phase_ids", np.ones(len(tetra), dtype=int))
    solid = phases == 1
    solid_volume = float(volumes[solid].sum())
    box_volume = float(np.prod(config.size))
    report.update(volume_mm3=solid_volume, actual_density=solid_volume / box_volume,
                  tetrahedra=int(len(tetra)), vertices=int(len(points)),
                  triangles=int(len(model["boundary"])),
                  bounds_mm=[points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                  density_profile=density_profile(points, tetra, volumes, solid, config),
                  density_profile_method="centroid-binned slabs of final solid cells")
    if "phase_ids" in model:
        fluid_volume = float(volumes[phases == 2].sum())
        report.update(solid_volume_mm3=solid_volume, fluid_volume_mm3=fluid_volume,
                      porosity=fluid_volume / box_volume,
                      total_mesh_volume_mm3=float(volumes.sum()),
                      partition_volume_error_mm3=abs(float(volumes.sum()) - box_volume),
                      interface_triangles=int(len(model["interface"])))
        tri = points[model["interface"]]
        report["interface_area_mm2"] = float(np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2)
    for entry in report.get("domain_map", []):
        mask = model["domains"] == entry["id"]
        entry["tetrahedra"] = int(mask.sum())
        entry["volume_mm3"] = float(volumes[mask].sum())


def wall_resolution(config, points, tetra, solid_mask, max_samples=20000):
    """Sheet walls only: estimated wall thickness 2q/|grad f| against the lattice spacing.

    Returns None for other modes. ``cells_across_p05`` < 1 means the thinnest
    5 % of the wall is narrower than one background cell: refine the resolution.
    """
    if config.mode != "sheet":
        return None
    from .model import field_and_threshold

    cells = np.asarray(tetra)[np.asarray(solid_mask, dtype=bool)]
    if len(cells) == 0:
        return None
    step = max(1, len(cells) // max_samples)
    xyz = np.asarray(points)[cells[::step]].mean(axis=1)
    h = float(max(config.size)) * 1e-5
    grad = np.zeros_like(xyz)
    for axis in range(3):
        e = np.zeros(3)
        e[axis] = h
        fp, _ = field_and_threshold(config, *(xyz + e).T)
        fm, _ = field_and_threshold(config, *(xyz - e).T)
        grad[:, axis] = (np.asarray(fp) - np.asarray(fm)) / (2 * h)
    _, q = field_and_threshold(config, *xyz.T)
    thickness = 2 * np.broadcast_to(q, len(xyz)) / np.maximum(np.linalg.norm(grad, axis=1), 1e-12)
    spacing = float(min(np.asarray(config.size) / (np.asarray(config.cells) * config.resolution)))
    p05, median = np.quantile(thickness, [0.05, 0.5])
    return {"method": "2 q / |grad f| at sampled solid-cell centroids (first-order sheet thickness)",
            "spacing_mm": spacing, "thickness_mm_p05": float(p05), "thickness_mm_median": float(median),
            "cells_across_p05": float(p05 / spacing), "cells_across_median": float(median / spacing)}
