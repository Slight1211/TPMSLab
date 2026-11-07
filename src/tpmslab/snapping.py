"""Pre-cut background-vertex snapping (isosurface-stuffing style warping).

Before a background tetrahedron is clipped, any lattice vertex that lies within
``alpha`` of a cut point along one of its incident lattice edges is moved onto
that cut point and declared to lie exactly on the corresponding isosurface.
The cut point is then shared with the vertex instead of producing a sliver.

This follows the warping step of Labelle & Shewchuk, "Isosurface stuffing"
(SIGGRAPH 2007), applied to the background lattice (Kuhn or BCC).
No dihedral-angle guarantee is claimed (with lattice="bcc" the background is
the body-centred cubic lattice used by isosurface stuffing, but a single alpha
and the box-face half tetrahedra fall outside its proof); the step is a
source-level reduction of thin cut cells, to be judged by the exported arrays.

Rules:
* A vertex on an outer box face/edge/corner may only move along a lattice edge
  that stays within that face/edge, so the box boundary is never dented.
* Each vertex is moved at most once, to its closest qualifying cut point.
* Field values at a moved vertex are linearly interpolated along the edge and
  the snapped side is set to exactly zero.
* If any background tetrahedron would lose positive orientation, the moves of
  its vertices are reverted and the check is repeated until stable.
* The transformation is deterministic in the configuration, so complementary
  solid and fluid passes produce identical vertices and cut keys.
"""

import numpy as np

from .quality import unique_rows

_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _signed_volumes(xyz, tets):
    p = xyz[tets]
    return np.einsum("ij,ij->i", p[:, 1] - p[:, 0],
                     np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6


def warp_vertices(xyz, fields, background, size, alpha, *, tol=None, implicit=None, alpha_boundary=None,
                  periodic_partners=None):
    """Return (xyz, fields, info) with qualifying lattice vertices snapped.

    ``xyz``: (N, 3) lattice coordinates; ``fields``: (S, N) inequality values
    (material where all >= 0); ``background``: (T, 4) lattice tetrahedra with
    positive orientation; ``size``: box extents; ``alpha`` in [0, 0.5).
    ``implicit``: optional callable mapping (M, 3) points to (S, M) true
    inequality values. When given, each snapped vertex is placed on the exact
    root of its side along the lattice edge (bisection) and its other field
    values are the true samples there, instead of the linear-cut point and
    linearly interpolated values.
    ``alpha_boundary``: optional threshold for vertices on the outer box
    (faces, edges); there the cut layer meets the box at the surface/face
    angle and a larger value removes more short edges. Default: ``alpha``.
    ``periodic_partners``: list of ``(axis, low, high)`` lattice index arrays of
    paired nodes on opposite periodic faces; their snaps are made identical
    (same target up to the box length) or reverted together.
    """
    xyz = np.array(xyz, dtype=float)
    fields = np.array(fields, dtype=float)
    size = np.asarray(size, dtype=float)
    info = {"vertex_snap_alpha": float(alpha), "snapped_vertices": 0,
            "snap_reverted_vertices": 0, "snap_candidates": 0}
    if alpha <= 0:
        return xyz, fields, info
    if not 0 < alpha < 0.5:
        raise ValueError("snap_alpha must be in [0, 0.5)")
    if tol is None:
        tol = float(size.max()) * 1e-9
    n_vertices = len(xyz)
    original_signed = _signed_volumes(xyz, background)
    if np.any(original_signed <= 0):
        raise ValueError("Background lattice must be positively oriented before snapping")

    edges = unique_rows(np.sort(background[:, _EDGES].reshape(-1, 2), axis=1))
    a, b = edges[:, 0], edges[:, 1]
    on_face = (np.abs(xyz) < tol) | (np.abs(xyz - size) < tol)
    same = np.abs(xyz[a] - xyz[b]) < tol
    ok_a = np.all(~on_face[a] | same, axis=1)
    ok_b = np.all(~on_face[b] | same, axis=1)

    # Candidate list: (vertex, distance fraction, edge index, side)
    if alpha_boundary is None:
        alpha_a = alpha_b = alpha
    else:
        if not 0 <= alpha_boundary < 0.5:
            raise ValueError("snap_alpha_boundary must be in [0, 0.5)")
        alpha_a = np.where(on_face[a].any(axis=1), float(alpha_boundary), float(alpha))
        alpha_b = np.where(on_face[b].any(axis=1), float(alpha_boundary), float(alpha))
        info["vertex_snap_alpha_boundary"] = float(alpha_boundary)
    cand_vertex, cand_dist, cand_edge, cand_side = [], [], [], []
    for side in range(fields.shape[0]):
        fa, fb = fields[side, a], fields[side, b]
        cut = ((fa < 0) & (fb > 0)) | ((fb < 0) & (fa > 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(cut, fa / (fa - fb), np.nan)
        near_a = cut & ok_a & (t < alpha_a)
        near_b = cut & ok_b & (t > 1 - alpha_b)
        for mask, vertex, dist in ((near_a, a, t), (near_b, b, 1 - t)):
            idx = np.flatnonzero(mask)
            cand_vertex.append(vertex[idx])
            cand_dist.append(dist[idx])
            cand_edge.append(idx)
            cand_side.append(np.full(len(idx), side))
    cand_vertex = np.concatenate(cand_vertex) if cand_vertex else np.zeros(0, int)
    info["snap_candidates"] = int(len(cand_vertex))
    if len(cand_vertex) == 0:
        return xyz, fields, info
    cand_dist = np.concatenate(cand_dist)
    cand_edge = np.concatenate(cand_edge)
    cand_side = np.concatenate(cand_side)
    order = np.lexsort((cand_dist, cand_vertex))
    cand_vertex, cand_dist = cand_vertex[order], cand_dist[order]
    cand_edge, cand_side = cand_edge[order], cand_side[order]
    first = np.r_[True, cand_vertex[1:] != cand_vertex[:-1]]
    sel_vertex = cand_vertex[first]
    sel_dist = cand_dist[first]
    sel_edge = cand_edge[first]
    sel_side = cand_side[first]

    # Target positions and field values, all from original data.
    ea, eb = a[sel_edge], b[sel_edge]
    fa = fields[sel_side, ea]
    fb = fields[sel_side, eb]
    t = fa / (fa - fb)
    if implicit is None:
        target_xyz = xyz[ea] * (1 - t)[:, None] + xyz[eb] * t[:, None]
        target_fields = fields[:, ea] * (1 - t) + fields[:, eb] * t
    else:
        # Bracketed bisection on the true field along each edge; lattice samples
        # are exact so the bracket [0, 1] always has opposite signs.
        lo = np.zeros(len(t))
        hi = np.ones(len(t))
        sign_a = np.sign(fa)
        pa, pb = xyz[ea], xyz[eb]
        rows = np.arange(len(t))
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            g = implicit(pa + (pb - pa) * mid[:, None])[sel_side, rows]
            same = np.sign(g) == sign_a
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
        root = 0.5 * (lo + hi)
        target_xyz = pa + (pb - pa) * root[:, None]
        target_fields = implicit(target_xyz)
        info["snap_root_mean_shift_fraction"] = float(np.mean(np.abs(root - t)))
    target_fields[sel_side, np.arange(len(sel_vertex))] = 0.0

    # Vertex -> incident background tetrahedra (CSR style), used for pinch checks.
    flat = background.ravel()
    order_t = np.argsort(flat, kind="stable")
    starts = np.searchsorted(flat[order_t], np.arange(n_vertices + 1))
    tet_of = order_t // 4

    def incident(v):
        return tet_of[starts[v]:starts[v + 1]]

    def pinched(u, v, side, f):
        """True if material {f[side] >= 0} meets itself only along lattice edge (u, v)."""
        ring = np.intersect1d(incident(u), incident(v))
        if len(ring) < 2:
            return False
        others = [[int(w) for w in background[ti] if w != u and w != v] for ti in ring]
        # Walk the ring through shared third vertices.
        count = {}
        for pair in others:
            for w in pair:
                count[w] = count.get(w, 0) + 1
        ends = [w for w, c in count.items() if c == 1]
        closed = not ends
        unused = set(range(len(ring)))
        current = 0
        if not closed:
            current = next(i for i in unused if any(count[w] == 1 for w in others[i]))
            entry = next(w for w in others[current] if count[w] == 1)
        else:
            entry = others[0][0]
        sequence = [entry]
        while True:
            unused.discard(current)
            exit_w = next(w for w in others[current] if w != entry)
            sequence.append(exit_w)
            nxt = [i for i in unused if exit_w in others[i]]
            if not nxt:
                break
            current, entry = nxt[0], exit_w
        if closed:
            sequence = sequence[:-1]  # last equals first
        values = np.array([f[side, w] for w in sequence])

        def runs(flags):
            flags = list(flags)
            if not any(flags):
                return 0
            if closed:
                # Cyclic ring: index -1 wraps to the last entry.
                return sum(1 for i in range(len(flags)) if flags[i] and not flags[i - 1])
            # Open chain (edge on the outer box): a leading run starts at index 0.
            return sum(1 for i in range(len(flags))
                       if flags[i] and (i == 0 or not flags[i - 1]))

        # Both the material (>= 0) and its complement (<= 0) must stay manifold
        # along the edge; a pinch of either makes the shared boundary non-manifold.
        return runs(values > 0) >= 2 or runs(values < 0) >= 2

    active = np.ones(len(sel_vertex), dtype=bool)
    pair_rows = []
    if periodic_partners:
        position = np.full(n_vertices, -1)
        position[sel_vertex] = np.arange(len(sel_vertex))
        for axis, low, high in periodic_partners:
            il, ih = position[low], position[high]
            lone = (il >= 0) != (ih >= 0)
            active[il[lone & (il >= 0)]] = False
            active[ih[lone & (ih >= 0)]] = False
            both = (il >= 0) & (ih >= 0)
            il, ih = il[both], ih[both]
            target_xyz[ih] = target_xyz[il]
            target_xyz[ih, axis] = size[axis]
            target_fields[:, ih] = target_fields[:, il]
            pair_rows.append((il, ih))

    def symmetrise():
        for il, ih in pair_rows:
            keep = active[il] & active[ih]
            active[il] = keep
            active[ih] = keep

    reverted_orientation = 0
    reverted_pinch = 0
    for _ in range(64):
        changed = False
        symmetrise()
        new_xyz = xyz.copy()
        new_xyz[sel_vertex[active]] = target_xyz[active]
        signed = _signed_volumes(new_xyz, background)
        bad = signed <= original_signed * 1e-6
        if np.any(bad):
            bad_vertices = np.unique(background[bad])
            revert = active & np.isin(sel_vertex, bad_vertices)
            if np.any(revert):
                reverted_orientation += int(revert.sum())
                active &= ~revert
                changed = True
                continue
        new_fields = fields.copy()
        new_fields[:, sel_vertex[active]] = target_fields[:, active]
        snapped_mask = np.zeros(n_vertices, dtype=bool)
        snapped_mask[sel_vertex[active]] = True
        dist_lookup = np.full(n_vertices, np.inf)
        dist_lookup[sel_vertex[active]] = sel_dist[active]
        to_revert = set()
        # A background tetrahedron whose four vertices all lie on one isosurface
        # carries an identically-zero linear field and would be claimed by both
        # the solid and the complementary fluid. Release its farthest snap.
        zero_tet = np.all(new_fields[:, background] == 0, axis=2)  # (S, T)
        for side_i, tet_i in zip(*np.nonzero(zero_tet)):
            verts = background[tet_i]
            cand = [v for v in verts if snapped_mask[v]]
            if cand:
                to_revert.add(int(max(cand, key=lambda v: dist_lookup[v])))
        zero_a = new_fields[:, a] == 0
        zero_b = new_fields[:, b] == 0
        shared_zero = zero_a & zero_b & (snapped_mask[a] | snapped_mask[b])[None, :]
        for side_i, edge_i in zip(*np.nonzero(shared_zero)):
            u, v = int(a[edge_i]), int(b[edge_i])
            if u in to_revert or v in to_revert:
                continue
            if pinched(u, v, side_i, new_fields):
                if snapped_mask[u] and snapped_mask[v]:
                    to_revert.add(u if dist_lookup[u] >= dist_lookup[v] else v)
                else:
                    to_revert.add(u if snapped_mask[u] else v)
        if to_revert:
            revert = active & np.isin(sel_vertex, list(to_revert))
            reverted_pinch += int(revert.sum())
            active &= ~revert
            changed = True
        if not changed:
            break
    else:
        raise ValueError("Vertex snapping did not converge")
    if np.any(signed <= 0):
        raise ValueError("Vertex snapping produced a nonpositive background tetrahedron")
    moved = sel_vertex[active]
    xyz[moved] = target_xyz[active]
    fields[:, moved] = target_fields[:, active]
    info["snapped_vertices"] = int(len(moved))
    info["snap_reverted_orientation"] = int(reverted_orientation)
    info["snap_reverted_pinch"] = int(reverted_pinch)
    info["snap_reverted_vertices"] = int(reverted_orientation + reverted_pinch)
    info["snapped_vertex_fraction"] = float(len(moved) / n_vertices)
    return xyz, fields, info
