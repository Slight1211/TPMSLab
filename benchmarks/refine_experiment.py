"""EXPERIMENT (not part of the package; see evidence/dev9): wall-thickness-adaptive refinement of a generated (two-phase) mesh.

A uniform background lattice gives the thin end of a graded sheet the same
element size as the thick end, so thin walls are crossed by fewer elements.
``refine_thin_walls`` splits, in passes, the longest edge of every solid cell
whose longest edge exceeds ``local wall thickness / layers``. Splitting an
edge splits every cell around it, so the mesh stays conforming (both phases
share the new node); a split edge of the solid/fluid interface gets its new
node on the exact level set (Newton steps, box planes pinned), otherwise the
straight midpoint is used. A projected node that would invert a cell falls
back to the midpoint. Run ``optimize_mesh`` afterwards.

Wall thickness of a sheet |f| <= q is estimated as 2 q / |grad f| at the cell
centroid (the exact distance between the two level sets to first order).
Only ``mode="sheet"`` is supported.
"""

from __future__ import annotations

import numpy as np

from tpmslab.model import Config, field_and_threshold
from tpmslab.quality import signed_volumes
from tpmslab.surface_smoothing import _level_set
from tpmslab.volume import TET_FACES

_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def wall_thickness(config, xyz, h=None):
    """2 q / |grad f| of a sheet design at points ``xyz`` (n, 3), in mm."""
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    h = float(max(config.size)) * 1e-5 if h is None else h
    grad = np.zeros_like(xyz)
    for axis in range(3):
        e = np.zeros(3)
        e[axis] = h
        fp, _ = field_and_threshold(config, *(xyz + e).T)
        fm, _ = field_and_threshold(config, *(xyz - e).T)
        grad[:, axis] = (np.asarray(fp) - np.asarray(fm)) / (2 * h)
    _, q = field_and_threshold(config, *xyz.T)
    norm = np.linalg.norm(grad, axis=1)
    return 2 * np.broadcast_to(q, len(xyz)) / np.maximum(norm, 1e-12)


def _project(config, x, side, pin_mask, pin_value, iterations=12):
    for _ in range(iterations):
        g = _level_set(config, x[None], side)[0]
        if abs(g) < 1e-11:
            return x, True
        grad = np.zeros(3)
        h = float(max(config.size)) * 1e-6
        for axis in np.flatnonzero(~pin_mask):
            e = np.zeros(3)
            e[axis] = h
            grad[axis] = (_level_set(config, (x + e)[None], side)[0] - _level_set(config, (x - e)[None], side)[0]) / (2 * h)
        n2 = grad @ grad
        if n2 < 1e-30:
            return x, False
        x = x - g / n2 * grad
        x[pin_mask] = pin_value[pin_mask]
    return x, abs(_level_set(config, x[None], side)[0]) < 1e-9


def refine_thin_walls(model, *, layers=3.0, passes=3, max_new_nodes=None, progress=None):
    """Return a refined copy of a ``mode="sheet"`` model (see module docstring)."""
    from tpmslab.optimize import _rebuild_boundary

    config = Config.from_dict(model["report"]["config"])
    if config.mode != "sheet":
        raise ValueError("refine_thin_walls currently supports mode='sheet' only")
    progress = progress or (lambda message: None)
    size = np.asarray(config.size, dtype=float)
    tol = float(size.max()) * 1e-9
    points = [p for p in np.asarray(model["points"], dtype=float)]
    cells = {i: list(map(int, t)) for i, t in enumerate(np.asarray(model["tetra"]))}
    domains = {i: int(d) for i, d in enumerate(np.asarray(model["domains"]))}
    phases = {i: int(p) for i, p in enumerate(np.asarray(model.get("phase_ids", np.ones(len(cells), int))))}
    interface = {tuple(sorted(map(int, f))) for f in np.asarray(model.get("interface", np.zeros((0, 3), int)))}
    if not interface:
        interface = {tuple(sorted(map(int, f))) for f, i in zip(np.asarray(model["boundary"]),
                                                               np.asarray(model["boundary_ids"])) if int(i) == 8}
    on_surface = set(v for f in interface for v in f)
    edge_cells = {}
    for cid, t in cells.items():
        for a, b in _EDGES:
            edge_cells.setdefault((min(t[a], t[b]), max(t[a], t[b])), set()).add(cid)
    next_cell = len(cells)
    splits = projected = 0
    budget = max_new_nodes if max_new_nodes is not None else 10 * len(points)
    for pass_index in range(passes):
        P = np.asarray(points)
        solid_ids = np.array([c for c, p in phases.items() if p == 1])
        T = np.array([cells[c] for c in solid_ids])
        centroid = P[T].mean(axis=1)
        target = wall_thickness(config, centroid) / layers
        lengths = np.stack([np.linalg.norm(P[T[:, a]] - P[T[:, b]], axis=1) for a, b in _EDGES], axis=1)
        longest = lengths.argmax(axis=1)
        marked = lengths.max(axis=1) > target
        edges = {}
        for t, k, flag, ln in zip(T[marked], longest[marked], marked[marked], lengths.max(axis=1)[marked]):
            a, b = _EDGES[k]
            edges[(min(t[a], t[b]), max(t[a], t[b]))] = ln
        if not edges:
            break
        # Longest edges first; an edge whose cells were already split this pass is skipped.
        for edge in sorted(edges, key=edges.get, reverse=True):
            if edge not in edge_cells or splits >= budget:
                continue
            a, b = edge
            pa, pb = points[a], points[b]
            mid = 0.5 * (pa + pb)
            new = mid
            faces = [tuple(sorted((a, b, w))) for w in {w for c in edge_cells[edge] for w in cells[c]} - {a, b}]
            surface_edge = any(f in interface for f in faces)
            if surface_edge:
                pin = ((np.abs(pa) < tol) & (np.abs(pb) < tol)) | ((np.abs(pa - size) < tol) & (np.abs(pb - size) < tol))
                pin_value = np.where(np.abs(pa) < tol, 0.0, size)
                residual = [abs(_level_set(config, pa[None], s)[0]) + abs(_level_set(config, pb[None], s)[0]) for s in (0, 1)]
                x, ok = _project(config, mid.copy(), int(np.argmin(residual)), pin, pin_value)
                if ok and np.linalg.norm(x - mid) < 0.5 * np.linalg.norm(pb - pa):
                    new = x
            m = len(points)
            old = list(edge_cells[edge])
            children = []
            for cid in old:
                t = cells[cid]
                c1 = [m if v == a else v for v in t]
                c2 = [m if v == b else v for v in t]
                children.append((cid, c1, c2))
            def valid(node):
                coords = np.array([[node if v == m else points[v] for v in c] for _, c1, c2 in children
                                   for c in (c1, c2)])
                orient = np.array([signed_volumes(np.array([points[v] for v in cells[cid]])[None])[0]
                                   for cid, _, _ in children for _ in (0, 1)])
                return bool(np.all(signed_volumes(coords) * np.sign(orient) > 0))

            ok = valid(new)
            if not ok and new is not mid:
                new = mid
                ok = valid(new)
            if not ok:
                continue
            if new is not mid:
                projected += 1
            points.append(new)
            if surface_edge:
                on_surface.add(m)
                for f in faces:
                    if f in interface:
                        interface.discard(f)
                        w = (set(f) - {a, b}).pop()
                        interface.add(tuple(sorted((a, m, w))))
                        interface.add(tuple(sorted((m, b, w))))
            for cid, c1, c2 in children:
                for x, y in _EDGES:
                    key = (min(cells[cid][x], cells[cid][y]), max(cells[cid][x], cells[cid][y]))
                    edge_cells[key].discard(cid)
                    if not edge_cells[key]:
                        del edge_cells[key]
                cells[cid] = c1
                cells[next_cell] = c2
                domains[next_cell], phases[next_cell] = domains[cid], phases[cid]
                for cid2 in (cid, next_cell):
                    t = cells[cid2]
                    for x, y in _EDGES:
                        edge_cells.setdefault((min(t[x], t[y]), max(t[x], t[y])), set()).add(cid2)
                next_cell += 1
            splits += 1
        progress(f"Thin-wall refinement pass {pass_index + 1}: {splits} edge splits so far")
    P = np.asarray(points)
    order = sorted(cells)
    tetra = np.array([cells[c] for c in order], dtype=np.int64)
    dom = np.array([domains[c] for c in order], dtype=np.int32)
    pha = np.array([phases[c] for c in order], dtype=np.int32)
    s = signed_volumes(P[tetra])
    tetra[s < 0] = tetra[s < 0][:, [1, 0, 2, 3]]
    # Exterior ids from box planes and phase; interface id 8.
    low, high = np.abs(P) < tol, np.abs(P - size) < tol
    faces = tetra[:, TET_FACES].reshape(-1, 3)
    owner = np.repeat(np.arange(len(tetra)), 4)
    exterior = {}
    for f, o in zip(faces, owner):
        common_low = low[f].all(axis=0)
        common_high = high[f].all(axis=0)
        if common_low.any() or common_high.any():
            axis = int(np.flatnonzero(common_low | common_high)[0])
            side = 0 if common_low[axis] else 1
            exterior[tuple(sorted(map(int, f)))] = 2 + 2 * axis + side + (10 if pha[o] == 2 else 0)
    two_phase = "phase_ids" in model
    arrays = _rebuild_boundary(tetra.astype(np.int32), pha if two_phase else None, exterior)
    out = {k: v for k, v in model.items() if k not in ("surface", "fluid_surface", "vertex_keys", "lattice_xyz")}
    out.update(arrays, points=P, tetra=tetra.astype(np.int32), domains=dom)
    if two_phase:
        out["phase_ids"] = pha
    import copy

    out["report"] = copy.deepcopy(model["report"])
    out["report"]["thin_wall_refinement"] = {"layers": layers, "passes": passes, "edge_splits": int(splits),
                                             "projected_surface_nodes": int(projected),
                                             "tetrahedra_before": int(len(model["tetra"])),
                                             "tetrahedra_after": int(len(tetra))}
    for key, faces_key in (("surface", "solid_boundary"), ("fluid_surface", "fluid_boundary")):
        if faces_key in out:
            import trimesh

            mesh = trimesh.Trimesh(P.copy(), np.asarray(out[faces_key]).copy(), process=False)
            mesh.remove_unreferenced_vertices()
            out[key] = mesh
    if "boundary" in arrays and "solid_boundary" not in arrays:
        import trimesh

        mesh = trimesh.Trimesh(P.copy(), np.asarray(arrays["boundary"]).copy(), process=False)
        mesh.remove_unreferenced_vertices()
        out["surface"] = mesh
    return out
