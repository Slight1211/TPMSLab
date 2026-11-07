"""Direct conforming cut-tetrahedral solid generation; no STL-to-CAD conversion.
Two linear halfspaces clip each background tetrahedron for sheet structures.
Shared edge vertices and canonical polygon fans preserve face conformity.
"""

import itertools
import os
import time
import hashlib
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .model import field_and_threshold
from .quality import choose_fan, unique_rows
from .snapping import warp_vertices
from .sampling import audit_custom_sampling

PATTERN = np.array(
    [[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]]
)
EDGES = list(itertools.combinations(range(4), 2))
TET_FACES = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
# A clipped vertex belongs to a tetrahedron face when none of its supporting
# corners are outside that face. These four masks do not depend on the cell.
_FACE_EXCLUSION_MASKS = tuple(~sum(1 << j for j in face) for face in TET_FACES)
# Explicit pre-flight limit on the background lattice. The default allows
# roughly 2.5e8 lattice points (about 8 GiB of lattice arrays); it can be
# raised with the TPMSLAB_MAX_LATTICE_BYTES environment variable.
DEFAULT_MAX_LATTICE_BYTES = 8 * 1024**3


def estimate_lattice_bytes(config):
    """Rough peak size of the sampling-stage arrays for ``config``.

    Counts the lattice coordinates, the inequality fields, the background
    tetrahedra and the corner-index scratch arrays; the clipping stage adds
    Python-object overhead proportional to the number of cut cells.
    """
    n = np.asarray(config.cells, dtype=np.int64) * int(config.resolution)
    points = int(np.prod(n + 1))
    voxels = int(np.prod(n))
    sides = 2 if config.mode == "sheet" else 1
    if getattr(config, "lattice", "kuhn") == "bcc":
        points += voxels  # cell centres (box-face centres are lower order)
    lattice = points * 8 * (3 + sides) + points * 8 * 2  # xyz + fields + f/q samples
    tets_per_voxel = 12 if getattr(config, "lattice", "kuhn") == "bcc" else 6
    cells = voxels * (8 * 3 + 8 * 8 + 8 * 8) + voxels * tets_per_voxel * 4 * 8  # ijk, corners, ids, background
    return int(lattice + cells)


def bcc_background(n, size):
    """Body-centred cubic tetrahedralisation of the box, conforming to its faces.

    Points: the (n+1)^3 voxel corners (same order as the Kuhn lattice), then the
    n^3 voxel centres, then one centre per voxel face on the box boundary.
    Every pair of face-adjacent voxels contributes four congruent tetrahedra
    (both centres + one edge of the shared square; mean ratio 0.952, dihedral
    angles 60 and 90 degrees). A boundary square is coned from its voxel centre
    through the face centre (four half tetrahedra, mean ratio 0.800).
    Returns (extra_xyz, background, owner_voxel); background is positively oriented.
    """
    n = np.asarray(n, dtype=np.int64)
    size = np.asarray(size, dtype=float)
    shape = n + 1
    spacing = size / n
    n_corner = int(np.prod(shape))
    n_voxel = int(np.prod(n))

    def corner(i, j, k):
        return (i * shape[1] + j) * shape[2] + k

    def voxel(i, j, k):
        return (i * n[1] + j) * n[2] + k

    ijk = np.indices(tuple(n)).reshape(3, -1).T
    centres = (ijk + 0.5) * spacing
    extra = [centres]
    next_id = n_corner + n_voxel
    tets, owners = [], []
    for a in range(3):
        b, c = [x for x in range(3) if x != a]
        # Square of the voxel face normal to axis a at offset 0 or 1, cyclic order.
        def square(cells, offset):
            base = cells.copy()
            base[:, a] += offset
            out = []
            for db, dc in ((0, 0), (1, 0), (1, 1), (0, 1)):
                q = base.copy()
                q[:, b] += db
                q[:, c] += dc
                out.append(corner(q[:, 0], q[:, 1], q[:, 2]))
            return np.stack(out, axis=1)
        # interior faces between voxel (i) and (i + e_a)
        left = ijk[ijk[:, a] < n[a] - 1]
        right = left.copy()
        right[:, a] += 1
        c1 = n_corner + voxel(*left.T)
        c2 = n_corner + voxel(*right.T)
        sq = square(left, 1)
        for e in range(4):
            tets.append(np.stack([c1, c2, sq[:, e], sq[:, (e + 1) % 4]], axis=1))
            owners.append(voxel(*(left if e % 2 == 0 else right).T))
        # boundary faces at both ends of axis a
        for side in (0, 1):
            cells = ijk[ijk[:, a] == (0 if side == 0 else n[a] - 1)]
            centre = n_corner + voxel(*cells.T)
            fc_xyz = (cells + 0.5) * spacing
            fc_xyz[:, a] = 0.0 if side == 0 else size[a]
            fc = next_id + np.arange(len(cells))
            next_id += len(cells)
            extra.append(fc_xyz)
            sq = square(cells, side)
            for e in range(4):
                tets.append(np.stack([centre, fc, sq[:, e], sq[:, (e + 1) % 4]], axis=1))
                owners.append(voxel(*cells.T))
    extra_xyz = np.concatenate(extra)
    background = np.concatenate(tets).astype(np.int64)
    owner_voxel = np.concatenate(owners).astype(np.int64)
    return extra_xyz, background, owner_voxel


def max_lattice_bytes():
    raw = os.environ.get("TPMSLAB_MAX_LATTICE_BYTES")
    if raw:
        try:
            value = int(float(raw))
        except ValueError as exc:
            raise ValueError("TPMSLAB_MAX_LATTICE_BYTES must be a number of bytes") from exc
        if value < 1:
            raise ValueError("TPMSLAB_MAX_LATTICE_BYTES must be positive")
        return value
    return DEFAULT_MAX_LATTICE_BYTES


def sample_lattice(config, progress=lambda text: None):
    """Sample the field, build the background lattice and apply optional snapping.

    The result is shared by the solid pass and the complementary fluid passes
    of ``solid_fluid`` generation, so the (deterministic) field evaluation,
    calibration, sampling audit and vertex snapping run once per design.
    """
    config.validate()
    estimate = estimate_lattice_bytes(config)
    limit = max_lattice_bytes()
    if estimate > limit:
        raise ValueError(
            f"The background lattice for cells={tuple(config.cells)} and resolution={config.resolution} "
            f"is estimated at {estimate / 1024**3:.1f} GiB, above the limit of {limit / 1024**3:.1f} GiB. "
            "Reduce cells or resolution, or raise TPMSLAB_MAX_LATTICE_BYTES explicitly."
        )
    progress("Sampling the surface function and density field...")
    n = np.array(config.cells) * config.resolution
    shape = n + 1
    spacing = np.array(config.size) / n
    axes = [np.linspace(0, L, int(k) + 1) for L, k in zip(config.size, n)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    xyz = grid.reshape(-1, 3)
    f, q = field_and_threshold(config, grid[..., 0], grid[..., 1], grid[..., 2])
    f = np.broadcast_to(f, tuple(shape)).ravel()
    q = np.broadcast_to(q, tuple(shape)).ravel()
    bcc = config.lattice == "bcc"
    if bcc:
        extra_xyz, bcc_tets, owner_voxel = bcc_background(n, config.size)
        fe, qe = field_and_threshold(config, extra_xyz[:, 0], extra_xyz[:, 1], extra_xyz[:, 2])
        f = np.concatenate([f, np.broadcast_to(fe, len(extra_xyz))])
        q = np.concatenate([q, np.broadcast_to(qe, len(extra_xyz))])
        xyz = np.concatenate([xyz, extra_xyz])
    if config.mode == "sheet":
        fields = np.array([q - f, q + f])
    elif config.mode == "solid_above":
        fields = np.array([q + f])
    else:
        fields = np.array([q - f])
    periodic = []
    if config.periodic_faces:
        from .periodic import face_partners, periodic_axes

        periodic = periodic_axes(config)
        if not periodic:
            raise ValueError("periodic_faces=True but the design field does not repeat across any pair of box "
                             "faces (needs whole cells and a threshold that does not vary along that axis).")
        tol = float(np.max(config.size)) * 1e-9
        partners = []
        for axis in periodic:
            low, high = face_partners(xyz, axis, np.asarray(config.size, float), tol)
            fields[:, high] = fields[:, low]
            partners.append((axis, low, high))
    eps = max(1.0, float(np.abs(fields).max())) * 1e-12
    fields[np.abs(fields) < eps] = 0
    ijk = np.indices(tuple(n)).reshape(3, -1).T
    offsets = np.array([[j & 1, (j >> 1) & 1, (j >> 2) & 1] for j in range(8)])
    corners = ijk[:, None, :] + offsets
    ids = (corners[:, :, 0] * shape[1] + corners[:, :, 1]) * shape[2] + corners[:, :, 2]
    sampling_audit = audit_custom_sampling(config, fields, ids, ijk, spacing)
    if bcc:
        p = xyz[bcc_tets]
        signed = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
        bcc_tets[signed < 0] = bcc_tets[signed < 0][:, [1, 0, 2, 3]]
        background = bcc_tets
    else:
        background = ids[:, PATTERN].reshape(-1, 4)
        owner_voxel = np.arange(len(background)) // len(PATTERN)
    # Optional pre-cut vertex snapping. It runs on the full inequality set before
    # any fluid-side sign flip, so complementary passes see identical lattices.
    snap_info = {"vertex_snap_alpha": 0.0, "snapped_vertices": 0,
                 "snap_reverted_vertices": 0, "snap_candidates": 0}
    if config.snap_alpha > 0:
        progress("Snapping lattice vertices that lie close to cut points...")
        implicit = None
        if config.snap_target == "implicit":
            def implicit(pts):
                value, threshold = field_and_threshold(config, pts[:, 0], pts[:, 1], pts[:, 2])
                value = np.broadcast_to(np.asarray(value, dtype=float), len(pts))
                threshold = np.broadcast_to(np.asarray(threshold, dtype=float), len(pts))
                if config.mode == "sheet":
                    return np.stack((threshold - value, threshold + value))
                if config.mode == "solid_above":
                    return np.stack((threshold + value,))
                return np.stack((threshold - value,))
        xyz, fields, snap_info = warp_vertices(
            xyz, fields, background, config.size, config.snap_alpha, implicit=implicit,
            alpha_boundary=config.snap_alpha_boundary,
            periodic_partners=partners if periodic else None,
        )
        snap_info["snap_target"] = config.snap_target
    return {"n": n, "shape": shape, "spacing": spacing, "xyz": xyz, "fields": fields,
            "eps": eps, "background": background, "sampling_audit": sampling_audit,
            "snap_info": snap_info, "owner_voxel": owner_voxel, "periodic_axes": periodic}


def face_components(tetra, n_points=None):
    """Connected components through shared faces (not vertex or edge contact).

    Returns ``(count, labels_per_tetrahedron)``. Two cells touching only at a
    vertex or an edge belong to different components; this matches the
    two-phase partition and is what a conforming FE model actually couples.
    """
    tetra = np.asarray(tetra)
    faces = np.sort(tetra[:, TET_FACES].reshape(-1, 3), axis=1)
    _, inverse, counts = unique_rows(faces, return_inverse=True, return_counts=True)
    return _components_from_faces(len(tetra), inverse, counts)


def _components_from_faces(n_tetra, inverse, counts):
    """Reuse canonical face incidence for connectivity after boundary checks."""
    order = np.argsort(inverse.ravel(), kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    pairs = starts[counts == 2]
    a, b = order[pairs] // 4, order[pairs + 1] // 4
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n_tetra, n_tetra))
    return connected_components(graph, directed=False)


def generate_volume(config, progress=lambda text: None, *, _fluid_side=None, _lattice=None):
    model = _generate_volume(config, progress, _fluid_side=_fluid_side, _lattice=_lattice)
    if _fluid_side is None and _lattice is None and model is not None:
        from .reporting import wall_resolution

        solid = model["phase_ids"] == 1 if "phase_ids" in model else np.ones(len(model["tetra"]), dtype=bool)
        info = wall_resolution(config, model["points"], model["tetra"], solid)
        if info is not None:
            model["report"]["wall_resolution"] = info
            if info["cells_across_p05"] < 1.0:
                model["report"].setdefault("warnings", []).append(
                    f"The thinnest 5% of the sheet wall is about {info['thickness_mm_p05']:.3g} mm, "
                    f"{info['cells_across_p05']:.2f} background cells across; raise the resolution "
                    "if the wall response (bending, conduction through the wall) matters.")
    if config.periodic_faces and _fluid_side is None and _lattice is None and model is not None:
        from .periodic import enforce_periodic, periodic_axes

        info = enforce_periodic(model, periodic_axes(config), config.size)
        progress(f"Periodic faces matched on axes {', '.join(info['axes'])}.")
    return model


def _generate_volume(config, progress=lambda text: None, *, _fluid_side=None, _lattice=None):
    from . import __version__
    config.validate()
    if config.domain_mode == "solid_fluid" and _fluid_side is None:
        from .multidomain import generate_solid_fluid

        return generate_solid_fluid(config, progress)
    start = time.monotonic()
    lattice = _lattice if _lattice is not None else sample_lattice(config, progress)
    n, spacing = lattice["n"], lattice["spacing"]
    xyz, eps, background = lattice["xyz"], lattice["eps"], lattice["background"]
    fields = np.array(lattice["fields"], copy=True)
    sampling_audit = lattice["sampling_audit"]
    snap_info = dict(lattice["snap_info"])
    cut_sides = list(range(len(fields)))
    if _fluid_side is not None:
        cut_sides = [_fluid_side]
        fields = -fields[[_fluid_side]]
    inside = np.all(fields >= 0, axis=0)
    full = inside[background].all(axis=1)
    possible = np.all(fields[:, background].max(axis=2) > 0, axis=0)
    active = np.flatnonzero(possible & ~full)
    nodes = xyz.tolist()
    keys = [(0, int(i), 0, 0) for i in range(len(xyz))]
    cuts = {}
    # Adjacent clipped cells can present a shared polygon in exactly the same
    # vertex order. Reuse its SVD-derived order only in that exact case; a
    # permutation can change SVD orientation and must retain the original path.
    face_orders = {}
    cells = background[full].tolist()
    owners = np.flatnonzero(full).tolist()
    improved_cells = 0
    local_quality_gain = []
    progress(f"Directly clipping {len(active):,} boundary tetrahedra...")
    for iteration, bi in enumerate(active):
        tet = background[bi]
        verts = []
        supports = []
        caps = []
        for i, vi in enumerate(tet):
            if inside[vi]:
                verts.append(int(vi))
                supports.append(1 << i)
                caps.append(set(np.flatnonzero(fields[:, vi] == 0)))
        for a, b in EDGES:
            va, vb = int(tet[a]), int(tet[b])
            for side in range(len(fields)):
                fa, fb = fields[side, va], fields[side, vb]
                if not ((fa < 0 < fb) or (fb < 0 < fa)):
                    continue
                fraction = fa / (fa - fb)
                interpolated = fields[:, va] * (1 - fraction) + fields[:, vb] * fraction
                if np.any(interpolated < -eps * 4):
                    continue
                key = (1, min(va, vb), max(va, vb), cut_sides[side])
                if key not in cuts:
                    cuts[key] = len(nodes)
                    nodes.append((xyz[va] * (1 - fraction) + xyz[vb] * fraction).tolist())
                    keys.append(key)
                verts.append(cuts[key])
                supports.append((1 << a) | (1 << b))
                caps.append({side})
        if len(verts) < 4:
            continue
        local = np.array([nodes[v] for v in verts])
        anchor = min(verts, key=lambda vi: keys[vi])
        face_sets = [
            [i for i, s in enumerate(supports) if not s & exclusion]
            for exclusion in _FACE_EXCLUSION_MASKS
        ]
        face_sets += [[i for i, c in enumerate(caps) if side in c] for side in range(len(fields))]
        boundary_tris = []
        for face_index, face in enumerate(face_sets):
            if len(face) < 3:
                continue
            if len(face) > 3:
                # The first four faces belong to the background tetrahedron;
                # cap faces belong to one cell and cannot benefit from caching.
                face_key = tuple(verts[i] for i in face) if face_index < 4 else None
                order = face_orders.pop(face_key, None) if face_key is not None else None
                if order is None:
                    p = local[face]
                    centered = p - p.mean(axis=0)
                    _, _, basis = np.linalg.svd(centered, full_matrices=False)
                    angles = np.arctan2(centered @ basis[1], centered @ basis[0])
                    order = np.argsort(angles)
                    if face_key is not None:
                        face_orders[face_key] = order
                face = [face[i] for i in order]
            smallest = min(range(len(face)), key=lambda j: keys[verts[face[j]]])
            face = face[smallest:] + face[:smallest]
            for j in range(1, len(face) - 1):
                boundary_tris.append([verts[face[0]], verts[face[j]], verts[face[j + 1]]])
        # A canonical face triangulation is shared by both candidate interiors.
        fan, new_point, gain = choose_fan(
            nodes, anchor, verts, boundary_tris, config.quality_strategy
        )
        if new_point is not None:
            nodes.append(new_point.tolist())
            keys.append((2, int(bi), 0, 0))
            improved_cells += 1
            local_quality_gain.append(gain)
        cells.extend(fan)
        owners.extend([int(bi)] * len(fan))
        if iteration and iteration % 10000 == 0:
            progress(f"Boundary clipping: {iteration:,} / {len(active):,}")
    if not cells:
        if _fluid_side is not None:
            return None
        raise ValueError("No volume elements were generated. Adjust the expression, density, or resolution.")
    points = np.array(nodes)
    tetra = np.array(cells, dtype=np.int32)
    owners = np.array(owners)
    p = points[tetra]
    signed = (
        np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6
    )
    nonzero = signed != 0
    tetra = tetra[nonzero]
    owners = owners[nonzero]
    signed = signed[nonzero]
    tetra[signed < 0] = tetra[signed < 0][:, [1, 0, 2, 3]]
    used = np.unique(tetra)
    remap = np.full(len(points), -1, dtype=np.int32)
    remap[used] = np.arange(len(used))
    points = points[used]
    tetra = remap[tetra]
    volumes = np.abs(signed)
    progress("Checking volume elements, shared faces, and material connectivity...")
    allfaces = tetra[:, TET_FACES].reshape(-1, 3)
    _, first, inverse, counts = unique_rows(
        np.sort(allfaces, axis=1), return_index=True, return_inverse=True, return_counts=True
    )
    if counts.max() > 2:
        raise ValueError("A non-manifold volume mesh was detected. Export was rejected.")
    boundary = allfaces[first[counts == 1]]
    surface = trimesh.Trimesh(points.copy(), boundary.copy(), process=False)
    surface.remove_unreferenced_vertices()
    ec = np.bincount(surface.edges_unique_inverse)
    if not surface.is_watertight or not surface.is_winding_consistent or np.any(ec != 2):
        raise ValueError("The boundary closure check failed. Increase the resolution or adjust parameters; no analysis-ready volume was exported.")
    if abs(surface.volume - volumes.sum()) > max(volumes.sum() * 1e-8, 1e-10):
        raise ValueError("Volume-mesh and boundary volumes do not match. Export was rejected.")
    # Material connectivity through shared faces. Vertex- or edge-only contact
    # does not transmit load in a conforming model and must not count as one body.
    components, labels = _components_from_faces(len(tetra), inverse, counts)
    domains = labels + 1
    adjacency = np.concatenate([tetra[:, [0, j]] for j in (1, 2, 3)])
    vertex_components, _ = connected_components(coo_matrix(
        (np.ones(len(adjacency)), (adjacency[:, 0], adjacency[:, 1])),
        shape=(len(points), len(points))), directed=False)
    pinched_contacts = int(components - vertex_components)
    p = points[tetra]
    edge_sum = sum(np.sum((p[:, a] - p[:, b]) ** 2, axis=1) for a, b in EDGES)
    quality = 12 * (3 * volumes) ** (2 / 3) / edge_sum
    pid = np.full(len(boundary), 8, dtype=int)
    tol = max(config.size) * 1e-9
    for ax in range(3):
        for side, value in enumerate((0, config.size[ax])):
            pid[np.all(np.abs(points[boundary, ax] - value) < tol, axis=1)] = 2 + ax * 2 + side
    per_cube = np.bincount(lattice["owner_voxel"][owners], weights=volumes, minlength=int(np.prod(n))).reshape(
        tuple(n)
    ) / np.prod(spacing)
    profile = []
    k = "xyz".index(config.axis)
    for ix in np.array_split(np.arange(n[k]), min(10, n[k])):
        profile.append(
            {
                "position_mm": float((ix.mean() + 0.5) * spacing[k]),
                "sampled_density": float(np.take(per_cube, ix, axis=k).mean()),
            }
        )
    warnings = [
        "Directly clipped tetrahedra define the computational volume; this is not editable smooth STEP/NURBS geometry.",
        "The surface and gradient are linearly approximated within background tetrahedra. Increase the resolution to check geometry and simulation response stability.",
        "Local density is calibrated using periodic cells. Finite boundaries and rapid grading may cause the measured volume fraction to differ from the target.",
    ]
    if config.mode == "sheet":
        warnings.append("Sheet structures are defined by two isosurfaces and do not imply constant physical wall thickness.")
    if config.combine == "blend":
        warnings.append("Combined TPMS uses a spatial blend of implicit fields with interpolated fixed-weight periodic calibration. Connectivity, physical layer density and mesh quality require separate checks.")
    if components > 1:
        warnings.append(
            f"There are {components} disconnected material domains requiring separate constraints; the static elasticity demo supports only connected material."
        )
    if pinched_contacts > 0:
        warnings.append(
            f"Material domains touch only at vertices or edges in {pinched_contacts} place(s); such contacts transmit no load and are counted as separate domains."
        )
    if quality.min() < 0.01:
        warnings.append("Low-quality cut elements are present. Inspect the quality distribution; solving the demo does not establish mesh convergence.")
    missed = sampling_audit.get("missed_void_voxels", 0) + sampling_audit.get("missed_solid_voxels", 0)
    if missed:
        warnings.append(
            f"Off-grid probes found {missed} voxel(s) whose interior disagrees with the lattice corners; the field is under-resolved at this resolution. Increase the resolution before drawing geometric conclusions."
        )
    report = {
        "config": config.to_dict(),
        "sampling_audit": sampling_audit,
        "units": "mm",
        "expression": config.expression,
        "representation": "conforming cut-tetrahedral solid domains (no STL-to-CAD)",
        "vertices": len(points),
        "tetrahedra": len(tetra),
        "triangles": len(boundary),
        "watertight": True,
        "winding_consistent": True,
        "nonmanifold_edges": 0,
        "boundary_edges": 0,
        "volume_components": int(components),
        "connectivity": "shared faces",
        "vertex_connected_components": int(vertex_components),
        "pinched_contacts": pinched_contacts,
        "exportable": True,
        "volume_mm3": float(volumes.sum()),
        "actual_density": float(volumes.sum() / np.prod(config.size)),
        "domain_volume_mm3": float(np.prod(config.size)),
        "bounds_mm": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "cell_size_mm": (np.array(config.size) / config.cells).tolist(),
        "grid_spacing_mm": spacing.tolist(),
        "minimum_tetra_volume_mm3": float(volumes.min()),
        "quality_quantiles": dict(
            zip(
                ["min", "p01", "median", "max"], map(float, np.quantile(quality, [0, 0.01, 0.5, 1]))
            )
        ),
        "density_profile": profile,
        "warnings": warnings,
        "comsol_verified": False,
        "comsol_solved": False,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "removed_nonzero_volume_mm3": 0.0,
        "quality_strategy": config.quality_strategy,
        "vertex_snapping": snap_info,
        "improved_cut_cells": improved_cells,
        "minimum_selected_local_quality_ratio": float(min(local_quality_gain))
        if local_quality_gain
        else None,
        "low_quality_count_below_001": int(np.sum(quality < 0.01)),
        "low_quality_volume_fraction_below_001": float(
            volumes[quality < 0.01].sum() / volumes.sum()
        ),
        "lattice_bytes_estimate": estimate_lattice_bytes(config),
        "mesh_sha256": hashlib.sha256(
            np.asarray(points, dtype="<f8").tobytes() + np.asarray(tetra, dtype="<i4").tobytes()
        ).hexdigest(),
        "boundary_sha256": boundary_digest(points, boundary),
        "schema_version": 1,
        "generator_version": __version__,
    }
    if config.combine == "blend":
        report["combined_design"] = {
            "operation": "blend", "primary_family": config.family,
            "secondary_family": config.family_secondary,
            "primary_expression": config.expression,
            "secondary_expression": config.expression_secondary,
            "weight_definition": "(1-w)*primary_field + w*secondary_field",
            "axis": config.blend_axis, "profile": config.blend_profile,
            "transition_bounds_fraction": [config.blend_start, config.blend_end],
            "transition_bounds_mm": [v*config.size["xyz".index(config.blend_axis)]
                                     for v in (config.blend_start, config.blend_end)],
            "calibration": "fixed-weight periodic quantiles with linear weight interpolation",
            "calibration_weight_samples": config.blend_calibration_steps,
        }
    progress("The solid volume mesh passed the topology checks.")
    return {
        "vertex_keys": [keys[i] for i in used],
        # Lattice coordinates after optional snapping; root projection must use
        # these rather than coordinates rebuilt from grid indices.
        "lattice_xyz": xyz if (config.snap_alpha > 0 or config.lattice == "bcc") else None,
        "points": points,
        "tetra": tetra,
        "boundary": boundary,
        "boundary_ids": pid,
        "domains": domains,
        "surface": surface,
        "report": report,
    }


def write_nastran(model, path):
    with open(path, "w", encoding="ascii", newline="\n") as stream:
        stream.write(
            "$ TPMS Lab direct volume mesh. Length unit: mm.\nBEGIN BULK\nMAT1,1,1.0,,0.3,1.0\n"
        )
        for i in np.unique(model["domains"]):
            stream.write(f"PSOLID,{100 + int(i)},1\n")
        for i in np.unique(model["boundary_ids"]):
            stream.write(f"PSHELL,{i},1,0.1\n")
        for i, p in enumerate(model["points"], 1):
            stream.write(f"GRID,{i},," + ",".join(f"{x:.15g}" for x in p) + "\n")
        for i, (tet, domain) in enumerate(zip(model["tetra"], model["domains"]), 1):
            stream.write(
                f"CTETRA,{i},{100 + int(domain)}," + ",".join(str(v + 1) for v in tet) + "\n"
            )
        offset = len(model["tetra"])
        for i, (tri, pid) in enumerate(zip(model["boundary"], model["boundary_ids"]), offset + 1):
            stream.write(f"CTRIA3,{i},{pid}," + ",".join(str(v + 1) for v in tri) + "\n")
        stream.write("ENDDATA\n")


def boundary_digest(points, faces):
    """Orientation-independent digest of boundary triangles, with 1e-11 mm rounding.

    Vectorised form of: round each triangle's corners, order them by (x, y, z),
    then sort the triangles lexicographically over their nine coordinates.
    Both sorts are stable, as Python's ``sorted`` was, so the bytes hashed are
    unchanged.
    """
    faces = np.asarray(faces)
    if len(faces) == 0:
        return hashlib.sha256(b"").hexdigest()
    xyz = np.round(np.asarray(points)[faces], 11)                      # (F, 3, 3)
    order = np.lexsort((xyz[..., 2], xyz[..., 1], xyz[..., 0]), axis=-1)
    rows = np.take_along_axis(xyz, order[..., None], axis=1).reshape(len(faces), 9)
    rows = rows[np.lexsort(rows.T[::-1])]
    return hashlib.sha256(np.ascontiguousarray(rows, dtype="<f8").tobytes()).hexdigest()
