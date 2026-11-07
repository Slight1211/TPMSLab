"""Complementary solid/fluid meshes sharing exact cut-edge identities."""

from dataclasses import replace
import hashlib
import time
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .volume import generate_volume, sample_lattice, TET_FACES, EDGES, boundary_digest
from .model import field_and_threshold
from .quality import unique_rows
from .reporting import density_profile


def generate_solid_fluid(config, progress):
    from . import __version__
    start = time.monotonic()
    single = replace(config, domain_mode="solid", edge_interpolation="linear")
    progress("Generating solid domains and complementary pore-fluid domains...")
    # Field sampling, calibration, the sampling audit and vertex snapping are
    # deterministic, so they run once and are shared by all complementary passes.
    lattice = sample_lattice(single, progress)
    parts = [generate_volume(single, progress, _lattice=lattice)]
    for side in range(2 if config.mode == "sheet" else 1):
        part = generate_volume(single, progress, _fluid_side=side, _lattice=lattice)
        if part is not None:
            parts.append(part)
    points, lookup, cells, phases = [], {}, [], []
    coordinate_tolerance = max(config.size) * 1e-12
    for part_index, part in enumerate(parts):
        remap = []
        duplicate_existing, duplicate_incoming = [], []
        for key, point in zip(part["vertex_keys"], part["points"]):
            # Interior fan vertices belong to a single clipped region.
            key = (*key, part_index) if key[0] == 2 else key
            if key not in lookup:
                lookup[key] = len(points)
                points.append(point)
            else:
                duplicate_existing.append(points[lookup[key]])
                duplicate_incoming.append(point)
            remap.append(lookup[key])
        # The tolerance applies independently to every coordinate. One array
        # check preserves that rule without a numpy call for every shared node.
        if duplicate_existing and not np.allclose(
            duplicate_existing, duplicate_incoming, rtol=0, atol=coordinate_tolerance
        ):
            raise ValueError("Solid/fluid cut vertex coordinates disagree")
        cells.append(np.asarray(remap)[part["tetra"]])
        phases.extend([1 if part_index == 0 else 2] * len(part["tetra"]))
    points = np.asarray(points)
    tetra = np.concatenate(cells).astype(np.int32)
    phases = np.asarray(phases, dtype=np.int32)
    edge_blend = None
    fully_projected_fraction = None
    root_cut_count = 0
    unbracketed = 0
    if config.edge_interpolation in ("implicit_root", "quadratic_root"):
        n = np.asarray(config.cells) * config.resolution
        shape = n + 1
        spacing = np.asarray(config.size) / n

        lattice_xyz = parts[0]["lattice_xyz"]

        def grid_points(indices):
            if lattice_xyz is not None:
                return np.asarray(lattice_xyz[indices], dtype=float)
            i = indices // (shape[1] * shape[2])
            j = (indices // shape[2]) % shape[1]
            k = indices % shape[2]
            return np.stack((i, j, k), axis=-1) * spacing

        def implicit_batch(xyz, sides):
            value, threshold = field_and_threshold(config, *xyz.T)
            value = np.broadcast_to(value, len(xyz))
            threshold = np.broadcast_to(threshold, len(xyz))
            if config.mode == "sheet":
                return np.where(sides == 0, threshold - value, threshold + value)
            return threshold + value if config.mode == "solid_above" else threshold - value

        projected = points.copy()
        cut_keys = [key for key in lookup if key[0] == 1]
        cut_nodes = np.asarray([lookup[key] for key in cut_keys], dtype=int)
        root_cut_count = len(cut_keys)
        indices = np.asarray([[key[1], key[2]] for key in cut_keys], dtype=np.int64).reshape(-1, 2)
        sides = np.asarray([key[3] for key in cut_keys], dtype=np.int8)
        if cut_keys and config.edge_interpolation == "quadratic_root":
            # Three batched field evaluations define a quadratic along every cut edge.
            # This avoids iterative scalar root calls while retaining a bracketed fallback.
            left, right = grid_points(indices[:, 0]), grid_points(indices[:, 1])
            direction = right - left
            linear = points[cut_nodes]
            t = np.einsum("ij,ij->i", linear - left, direction) / np.einsum(
                "ij,ij->i", direction, direction
            )
            g0 = implicit_batch(left, sides)
            g1 = implicit_batch(right, sides)
            gm = implicit_batch(linear, sides)
            slope = g1 - g0
            with np.errstate(divide="ignore", invalid="ignore"):
                a = (gm - g0 - slope * t) / (t * (t - 1))
                b = slope - a
                discriminant = b * b - 4 * a * g0
                square_root = np.sqrt(np.maximum(discriminant, 0))
                root_a = (-b + square_root) / (2 * a)
                root_b = (-b - square_root) / (2 * a)
                secant = t - gm / slope
            valid_a = (discriminant >= 0) & (root_a >= 0) & (root_a <= 1)
            valid_b = (discriminant >= 0) & (root_b >= 0) & (root_b <= 1)
            choose_a = valid_a & (~valid_b | (np.abs(root_a - t) <= np.abs(root_b - t)))
            roots = np.where(choose_a, root_a, np.where(valid_b, root_b, secant))
            roots = np.where(np.isfinite(roots), roots, t)
            roots = np.clip(roots, 0, 1)
            projected[cut_nodes] = left + roots[:, None] * direction
        elif cut_keys:
            # Vectorised bracketed bisection on the true field along every cut
            # edge (the same scheme as snapping.warp_vertices). Snapped lattice
            # values are linear/implicit samples; keep the linear cut when the
            # true field does not bracket a root.
            left, right = grid_points(indices[:, 0]), grid_points(indices[:, 1])
            g0, g1 = implicit_batch(left, sides), implicit_batch(right, sides)
            bracketed = (g0 < 0) & (g1 > 0) | (g1 < 0) & (g0 > 0)
            unbracketed = int(np.sum(~bracketed))
            if np.any(bracketed):
                b_left, b_right = left[bracketed], right[bracketed]
                b_sides = sides[bracketed]
                direction = b_right - b_left
                # Start from the linear cut: when the field crosses zero more
                # than once along an edge, take the root on the side of the
                # linear crossing (the same choice quadratic_root makes).
                linear = points[cut_nodes[bracketed]]
                t = np.clip(np.einsum("ij,ij->i", linear - b_left, direction)
                            / np.einsum("ij,ij->i", direction, direction), 0.0, 1.0)
                gt = implicit_batch(b_left + direction * t[:, None], b_sides)
                sign_left = np.sign(g0[bracketed])
                left_half = np.sign(gt) != sign_left  # crossing lies in [0, t]
                lo = np.where(left_half, 0.0, t)
                hi = np.where(left_half, t, 1.0)
                for _ in range(48):  # 2**-48 of an edge: below the 1e-11 xtol used before
                    mid = 0.5 * (lo + hi)
                    g = implicit_batch(b_left + direction * mid[:, None], b_sides)
                    same = np.sign(g) == sign_left
                    lo = np.where(same, mid, lo)
                    hi = np.where(same, hi, mid)
                root = np.where(gt == 0, t, 0.5 * (lo + hi))
                projected[cut_nodes[bracketed]] = b_left + direction * root[:, None]
        delta = projected - points
        incident = [[] for _ in range(len(points))]
        for ti, vertices in enumerate(tetra):
            for vertex in vertices:
                incident[vertex].append(ti)
        accepted = []
        for node in cut_nodes:
            adjacent = tetra[incident[node]]
            rows, cols = np.where(adjacent == node)
            local = points[adjacent]
            blend = 0.0
            for candidate in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125,
                              0.015625, 0.0078125, 0.00390625):
                p = local.copy()
                p[rows, cols] = points[node] + candidate * delta[node]
                signed = np.einsum("ij,ij->i", p[:, 1] - p[:, 0],
                                   np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6
                if np.all(signed > 0):
                    blend = candidate
                    points[node] += blend * delta[node]
                    break
            accepted.append(blend)
        if accepted:
            edge_blend = float(np.mean(accepted))
            fully_projected_fraction = float(np.mean(np.asarray(accepted) == 1.0))
    faces = tetra[:, TET_FACES].reshape(-1, 3)
    _, inverse, counts = unique_rows(np.sort(faces, axis=1), return_inverse=True, return_counts=True)
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
    # Slab density profile from the final solid cells (after optional root
    # projection), binned by centroid along the gradient axis.
    profile = density_profile(points, tetra, volumes, phases == 1, config)
    report.update(
        config=config.to_dict(),
        edge_root_blend=edge_blend,
        edge_root_cut_count=root_cut_count,
        edge_root_fully_projected_fraction=fully_projected_fraction,
        edge_root_unbracketed_cuts=int(unbracketed),
        density_profile=profile,
        density_profile_method="centroid-binned slabs of final solid cells",
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
