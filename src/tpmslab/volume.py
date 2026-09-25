"""Direct conforming cut-tetrahedral solid generation; no STL-to-CAD conversion.
Two linear halfspaces clip each background tetrahedron for sheet structures.
Shared edge vertices and canonical polygon fans preserve face conformity.
"""

import itertools
import time
import hashlib
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .model import calibration, evaluate, target_density
from .quality import choose_fan

PATTERN = np.array(
    [[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]]
)
EDGES = list(itertools.combinations(range(4), 2))
TET_FACES = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]


def generate_volume(config, progress=lambda text: None, *, _fluid_side=None):
    config.validate()
    if config.domain_mode == "solid_fluid" and _fluid_side is None:
        from .multidomain import generate_solid_fluid

        return generate_solid_fluid(config, progress)
    start = time.monotonic()
    progress("采样曲面函数与密度场…")
    n = np.array(config.cells) * config.resolution
    shape = n + 1
    spacing = np.array(config.size) / n
    axes = [np.linspace(0, L, int(k) + 1) for L, k in zip(config.size, n)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    xyz = grid.reshape(-1, 3)
    f = evaluate(
        config.expression,
        *(
            grid[..., i] * 2 * np.pi * config.cells[i] / config.size[i]
            + np.deg2rad(config.phase_degrees[i])
            for i in range(3)
        ),
    )
    f = np.broadcast_to(f, tuple(shape)).ravel()
    probability, samples = calibration(config.expression, config.mode)
    rho = np.broadcast_to(
        target_density(config, grid[..., 0], grid[..., 1], grid[..., 2]), tuple(shape)
    ).ravel()
    q = np.interp(rho, probability, samples)
    if config.mode == "sheet":
        fields = np.array([q - f, q + f])
    elif config.mode == "solid_above":
        fields = np.array([q + f])
    else:
        fields = np.array([q - f])
    eps = max(1.0, float(np.abs(fields).max())) * 1e-12
    cut_sides = list(range(len(fields)))
    if _fluid_side is not None:
        cut_sides = [_fluid_side]
        fields = -fields[[_fluid_side]]
    fields[np.abs(fields) < eps] = 0
    ijk = np.indices(tuple(n)).reshape(3, -1).T
    offsets = np.array([[j & 1, (j >> 1) & 1, (j >> 2) & 1] for j in range(8)])
    corners = ijk[:, None, :] + offsets
    ids = (corners[:, :, 0] * shape[1] + corners[:, :, 1]) * shape[2] + corners[:, :, 2]
    background = ids[:, PATTERN].reshape(-1, 4)
    inside = np.all(fields >= 0, axis=0)
    full = inside[background].all(axis=1)
    possible = np.all(fields[:, background].max(axis=2) > 0, axis=0)
    active = np.flatnonzero(possible & ~full)
    nodes = xyz.tolist()
    keys = [(0, int(i), 0, 0) for i in range(len(xyz))]
    cuts = {}
    cells = background[full].tolist()
    owners = np.flatnonzero(full).tolist()
    improved_cells = 0
    local_quality_gain = []
    progress(f"直接裁切 {len(active):,} 个边界四面体…")
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
            [i for i, s in enumerate(supports) if not s & ~sum(1 << j for j in face)]
            for face in TET_FACES
        ]
        face_sets += [[i for i, c in enumerate(caps) if side in c] for side in range(len(fields))]
        boundary_tris = []
        for face in face_sets:
            if len(face) < 3:
                continue
            if len(face) > 3:
                p = local[face]
                center = p.mean(axis=0)
                _, _, basis = np.linalg.svd(p - center, full_matrices=False)
                angles = np.arctan2((p - center) @ basis[1], (p - center) @ basis[0])
                face = [face[i] for i in np.argsort(angles)]
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
            progress(f"边界裁切：{iteration:,} / {len(active):,}")
    if not cells:
        if _fluid_side is not None:
            return None
        raise ValueError("没有生成体单元，请调整公式、密度或分辨率。")
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
    progress("检查体单元、共享面与材料连通性…")
    allfaces = tetra[:, TET_FACES].reshape(-1, 3)
    _, first, counts = np.unique(
        np.sort(allfaces, axis=1), axis=0, return_index=True, return_counts=True
    )
    if counts.max() > 2:
        raise ValueError("检测到非流形体网格，拒绝导出。")
    boundary = allfaces[first[counts == 1]]
    surface = trimesh.Trimesh(points.copy(), boundary.copy(), process=False)
    surface.remove_unreferenced_vertices()
    ec = np.bincount(surface.edges_unique_inverse)
    if not surface.is_watertight or not surface.is_winding_consistent or np.any(ec != 2):
        raise ValueError("边界闭合检查失败，请提高分辨率或调整参数；没有输出可分析实体。")
    if abs(surface.volume - volumes.sum()) > max(volumes.sum() * 1e-8, 1e-10):
        raise ValueError("体网格与边界体积不一致，拒绝导出。")
    adjacency = np.concatenate([tetra[:, [0, j]] for j in (1, 2, 3)])
    graph = coo_matrix(
        (np.ones(len(adjacency)), (adjacency[:, 0], adjacency[:, 1])),
        shape=(len(points), len(points)),
    )
    components, labels = connected_components(graph, directed=False)
    domains = labels[tetra[:, 0]] + 1
    p = points[tetra]
    edge_sum = sum(np.sum((p[:, a] - p[:, b]) ** 2, axis=1) for a, b in EDGES)
    quality = 12 * (3 * volumes) ** (2 / 3) / edge_sum
    pid = np.full(len(boundary), 8, dtype=int)
    tol = max(config.size) * 1e-9
    for ax in range(3):
        for side, value in enumerate((0, config.size[ax])):
            pid[np.all(np.abs(points[boundary, ax] - value) < tol, axis=1)] = 2 + ax * 2 + side
    per_cube = np.bincount(owners // 6, weights=volumes, minlength=int(np.prod(n))).reshape(
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
        "直接裁切四面体定义实体计算域；不是可编辑的光滑 STEP/NURBS 几何。",
        "曲面与梯度在背景四面体内作线性近似。须提高采样数检查几何及仿真收敛性。",
        "局部密度由周期单胞校准；有限外形及快速梯度可能使实际体积分数偏离目标。",
    ]
    if config.mode == "sheet":
        warnings.append("片状结构按双等值面定义，不等于恒定物理壁厚。")
    if components > 1:
        warnings.append(
            f"有 {components} 个独立材料域，需分别施加约束；静力学演示仅支持单连通材料。"
        )
    if quality.min() < 0.01:
        warnings.append("存在低质量裁切单元，请检查质量分布；演示求解不代表网格已收敛。")
    report = {
        "config": config.to_dict(),
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
        "improved_cut_cells": improved_cells,
        "minimum_selected_local_quality_ratio": float(min(local_quality_gain))
        if local_quality_gain
        else None,
        "low_quality_count_below_001": int(np.sum(quality < 0.01)),
        "low_quality_volume_fraction_below_001": float(
            volumes[quality < 0.01].sum() / volumes.sum()
        ),
        "mesh_sha256": hashlib.sha256(
            np.asarray(points, dtype="<f8").tobytes() + np.asarray(tetra, dtype="<i4").tobytes()
        ).hexdigest(),
        "boundary_sha256": boundary_digest(points, boundary),
        "schema_version": 1,
        "generator_version": "0.3.0rc1",
    }
    progress("实体体网格已通过拓扑检查。")
    return {
        "vertex_keys": [keys[i] for i in used],
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
    """Orientation-independent digest of boundary triangles, with 1e-11 mm rounding."""
    rows = []
    for face in faces:
        xyz = np.round(points[face], 11)
        order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
        rows.append(tuple(xyz[order].ravel()))
    return hashlib.sha256(np.asarray(sorted(rows), dtype="<f8").tobytes()).hexdigest()
