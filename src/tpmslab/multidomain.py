"""Complementary solid/fluid meshes sharing exact cut-edge identities."""

from dataclasses import replace
import hashlib
import time
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .volume import generate_volume, TET_FACES, EDGES, boundary_digest


def generate_solid_fluid(config, progress):
    from . import __version__
    start = time.monotonic()
    single = replace(config, domain_mode="solid")
    progress("Generating solid domains and complementary pore-fluid domains...")
    parts = [generate_volume(single, progress)]
    for side in range(2 if config.mode == "sheet" else 1):
        part = generate_volume(single, progress, _fluid_side=side)
        if part is not None:
            parts.append(part)
    points, lookup, cells, phases = [], {}, [], []
    for part_index, part in enumerate(parts):
        remap = []
        for key, point in zip(part["vertex_keys"], part["points"]):
            # Interior fan vertices belong to a single clipped region.
            key = (*key, part_index) if key[0] == 2 else key
            if key not in lookup:
                lookup[key] = len(points)
                points.append(point)
            elif not np.allclose(points[lookup[key]], point, rtol=0, atol=max(config.size) * 1e-12):
                raise ValueError("Solid/fluid cut vertex coordinates disagree")
            remap.append(lookup[key])
        cells.append(np.asarray(remap)[part["tetra"]])
        phases.extend([1 if part_index == 0 else 2] * len(part["tetra"]))
    points = np.asarray(points)
    tetra = np.concatenate(cells).astype(np.int32)
    phases = np.asarray(phases, dtype=np.int32)
    faces = tetra[:, TET_FACES].reshape(-1, 3)
    _, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if counts.max() > 2:
        raise ValueError("Nonmanifold solid/fluid partition")
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    exterior = order[starts[counts == 1]]
    pairs = starts[counts == 2]
    a, b = order[pairs], order[pairs + 1]
    interface_pair = phases[a // 4] != phases[b // 4]
    # Consistent positive tetrahedron orientation implies opposite face normals.
    fa, fb = faces[a], faces[b]
    na = np.cross(points[fa[:, 1]] - points[fa[:, 0]], points[fa[:, 2]] - points[fa[:, 0]])
    nb = np.cross(points[fb[:, 1]] - points[fb[:, 0]], points[fb[:, 2]] - points[fb[:, 0]])
    if np.any(np.einsum("ij,ij->i", na, nb) >= 0):
        raise ValueError("Overlapping or inconsistently oriented adjacent cells")
    interface = np.where(phases[a[interface_pair] // 4] == 1, a[interface_pair], b[interface_pair])
    if not len(interface):
        raise ValueError("No solid-fluid interface generated")
    exterior_faces = faces[exterior]
    side_ids = np.zeros(len(exterior), dtype=np.int32)
    tol = max(config.size) * 1e-9
    for axis in range(3):
        for side, value in enumerate((0, config.size[axis])):
            mask = np.all(np.abs(points[exterior_faces, axis] - value) < tol, axis=1)
            side_ids[mask] = 2 + 2 * axis + side
    if np.any(side_ids == 0):
        raise ValueError("Solid/fluid partition has an internal crack or unmatched interface")
    side_ids += np.where(phases[exterior // 4] == 2, 10, 0)
    boundary = np.concatenate([exterior_faces, faces[interface]])
    boundary_ids = np.r_[side_ids, np.full(len(interface), 8)].astype(np.int32)
    same = ~interface_pair
    graph = coo_matrix(
        (np.ones(np.sum(same)), (a[same] // 4, b[same] // 4)), shape=(len(tetra), len(tetra))
    )
    components, labels = connected_components(graph, directed=False)
    # Number all solid components first, then all fluid components.
    _, representatives = np.unique(labels, return_index=True)
    raw = sorted(range(components), key=lambda k: (int(phases[representatives[k]]), k))
    mapping = np.empty(components, dtype=np.int32)
    mapping[raw] = np.arange(1, components + 1)
    domains = mapping[labels]
    p = points[tetra]
    volumes = (
        np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6
    )
    expected = float(np.prod(config.size))
    if np.any(volumes <= 0) or not np.isclose(volumes.sum(), expected, rtol=1e-10):
        raise ValueError("Solid and fluid do not form a positive-volume partition of the box")
    domain_map = []
    for d in np.unique(domains):
        mask = domains == d
        phase = "solid" if phases[mask][0] == 1 else "fluid"
        touching = sorted(set(map(int, side_ids[domains[exterior // 4] == d])))
        domain_map.append(
            dict(
                id=int(d),
                nastran_pid=100 + int(d),
                phase=phase,
                tetrahedra=int(mask.sum()),
                volume_mm3=float(volumes[mask].sum()),
                exterior_boundary_ids=touching,
            )
        )
    quality = (
        12
        * (3 * volumes) ** (2 / 3)
        / sum(np.sum((p[:, a] - p[:, b]) ** 2, axis=1) for a, b in EDGES)
    )
    solid_volume = float(volumes[phases == 1].sum())
    fluid_volume = float(volumes[phases == 2].sum())
    solid_faces = np.concatenate([exterior_faces[phases[exterior // 4] == 1], faces[interface]])
    fluid_faces = np.concatenate(
        [exterior_faces[phases[exterior // 4] == 2], faces[interface][:, [0, 2, 1]]]
    )
    surface = trimesh.Trimesh(points.copy(), solid_faces, process=False)
    surface.remove_unreferenced_vertices()
    fluid_surface = trimesh.Trimesh(points.copy(), fluid_faces, process=False)
    fluid_surface.remove_unreferenced_vertices()
    report = dict(parts[0]["report"])
    report.update(
        config=config.to_dict(),
        representation="conforming complementary solid/fluid tetrahedral domains",
        vertices=len(points),
        bounds_mm=[points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        tetrahedra=len(tetra),
        triangles=len(boundary),
        volume_components=components,
        solid_components=sum(x["phase"] == "solid" for x in domain_map),
        fluid_components=sum(x["phase"] == "fluid" for x in domain_map),
        domain_map=domain_map,
        volume_mm3=solid_volume,
        solid_volume_mm3=solid_volume,
        fluid_volume_mm3=fluid_volume,
        total_mesh_volume_mm3=float(volumes.sum()),
        actual_density=solid_volume / expected,
        porosity=fluid_volume / expected,
        partition_volume_error_mm3=float(abs(volumes.sum() - expected)),
        interface_triangles=len(interface),
        interface_conforming=True,
        interface_invalid_triangles=int(np.sum(counts[inverse[interface]] != 2)),
        interface_area_mm2=float(np.linalg.norm(na[interface_pair], axis=1).sum() / 2),
        boundary_tag_names={
            **{
                str(2 + i): "solid_" + name
                for i, name in enumerate(("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"))
            },
            **{
                str(12 + i): "fluid_" + name
                for i, name in enumerate(("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"))
            },
            "8": "solid_fluid_interface",
        },
        minimum_tetra_volume_mm3=float(volumes.min()),
        quality_quantiles=dict(
            zip(
                ("min", "p01", "median", "max"), map(float, np.quantile(quality, [0, 0.01, 0.5, 1]))
            )
        ),
        low_quality_count_below_001=int(np.sum(quality < 0.01)),
        low_quality_volume_fraction_below_001=float(volumes[quality < 0.01].sum() / expected),
        improved_cut_cells=sum(part["report"]["improved_cut_cells"] for part in parts),
        minimum_selected_local_quality_ratio=None,
        mesh_sha256=hashlib.sha256(
            points.astype("<f8").tobytes()
            + tetra.astype("<i4").tobytes()
            + domains.astype("<i4").tobytes()
        ).hexdigest(),
        boundary_sha256=boundary_digest(points, boundary),
        schema_version=2,
        generator_version=__version__,
        elapsed_seconds=round(time.monotonic() - start, 3),
    )
    report["warnings"] = [w for w in report["warnings"] if "static elasticity demo" not in w]
    report["warnings"] += [
        "Solid and fluid fill the same rectangular box only; external flow regions and inlet buffers are not included.",
        "The solid-fluid interface is conforming with shared nodes. Check fluid boundary layers and flow mesh convergence for the intended conditions.",
        "The fluid may contain disconnected channels or closed pores. Select inlets and outlets by connected domain.",
    ]
    progress(
        f"Two-phase partition verified: {report['solid_components']} solid domains, {report['fluid_components']} fluid domains, and {len(interface):,} shared interface triangles."
    )
    return dict(
        points=points,
        tetra=tetra,
        boundary=boundary,
        boundary_ids=boundary_ids,
        domains=domains,
        phase_ids=phases,
        interface=faces[interface],
        solid_boundary=solid_faces,
        fluid_boundary=fluid_faces,
        surface=surface,
        fluid_surface=fluid_surface,
        report=report,
    )
