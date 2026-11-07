"""Conservative candidate selection without moving any boundary vertex.

This is a local heuristic, not a minimum-angle guarantee or global optimizer.
Canonical polygon triangles are identical in both candidates. A centroid fan
is accepted only if its worst mean-ratio quality strictly improves and its
signed-volume magnitude sum agrees with the pulling fan.
"""

import numpy as np


def cross(a, b):
    """Row-wise cross product without numpy's axis bookkeeping (fast for small arrays)."""
    out = np.empty_like(a)
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


def signed_volumes(points):
    """Signed volumes of (N, 4, 3) tetrahedra."""
    p = np.asarray(points, dtype=float)
    return np.einsum("ij,ij->i", p[:, 1] - p[:, 0], cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6


def tetra_metrics(points):
    p = np.asarray(points, dtype=float)
    volumes = np.abs(signed_volumes(p))
    lengths = sum(
        np.sum((p[:, a] - p[:, b]) ** 2, axis=1)
        for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    )
    quality = np.divide(
        12 * (3 * volumes) ** (2 / 3), lengths, out=np.zeros_like(volumes), where=lengths > 0
    )
    return volumes, quality


def unique_rows(rows, *, return_index=False, return_inverse=False, return_counts=False):
    """``np.unique(rows, axis=0, ...)`` for small non-negative integer rows.

    Rows are packed into one int64 key (lexicographic order is preserved
    because every entry is smaller than the radix), so the result -- unique
    rows, first indices, inverse and counts -- is identical to numpy's
    row-wise unique, which is several times slower. Falls back to numpy when
    the packed key would overflow.
    """
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.dtype.kind not in "iu" or len(rows) == 0 or rows.min() < 0:
        return np.unique(rows, axis=0, return_index=return_index, return_inverse=return_inverse,
                         return_counts=return_counts)
    radix = int(rows.max()) + 1
    if radix ** rows.shape[1] >= 2 ** 62:
        return np.unique(rows, axis=0, return_index=return_index, return_inverse=return_inverse,
                         return_counts=return_counts)
    key = np.zeros(len(rows), dtype=np.int64)
    for column in range(rows.shape[1]):
        key = key * radix + rows[:, column].astype(np.int64)
    result = np.unique(key, return_index=True, return_inverse=return_inverse, return_counts=return_counts)
    first = result[1]
    out = [rows[first]]
    if return_index:
        out.append(first)
    if return_inverse:
        out.append(result[2].reshape(-1))
    if return_counts:
        out.append(result[-1])
    return out[0] if len(out) == 1 else tuple(out)


def tet_scalar(P, t):
    """(signed volume, mean-ratio quality) of one tetrahedron in pure Python.

    ``P`` is a sequence of 3-tuples/lists (e.g. ``points.tolist()``) and ``t``
    four indices. For the 1-10 cell stars touched by local mesh operations
    this is ~10x faster than the vectorised numpy path, whose cost is all
    call overhead at that size.
    """
    a, b, c, d = P[t[0]], P[t[1]], P[t[2]], P[t[3]]
    return _tet_xyz(a, b, c, d)


def _tet_xyz(a, b, c, d):
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    wx, wy, wz = d[0] - a[0], d[1] - a[1], d[2] - a[2]
    volume = (ux * (vy * wz - vz * wy) + uy * (vz * wx - vx * wz) + uz * (vx * wy - vy * wx)) / 6
    bcx, bcy, bcz = c[0] - b[0], c[1] - b[1], c[2] - b[2]
    bdx, bdy, bdz = d[0] - b[0], d[1] - b[1], d[2] - b[2]
    cdx, cdy, cdz = d[0] - c[0], d[1] - c[1], d[2] - c[2]
    lengths = (ux * ux + uy * uy + uz * uz + vx * vx + vy * vy + vz * vz + wx * wx + wy * wy + wz * wz
               + bcx * bcx + bcy * bcy + bcz * bcz + bdx * bdx + bdy * bdy + bdz * bdz
               + cdx * cdx + cdy * cdy + cdz * cdz)
    if lengths <= 0:
        return volume, 0.0
    return volume, 12 * (3 * abs(volume)) ** (2 / 3) / lengths


def star_scalar(P, tets, vertex=None, x=None):
    """(min signed volume, min quality) over ``tets``; optionally with ``vertex`` moved to ``x``."""
    vmin = qmin = float("inf")
    for t in tets:
        corners = [x if (vertex is not None and i == vertex) else P[i] for i in t]
        v, q = _tet_xyz(*corners)
        if v < vmin:
            vmin = v
        if q < qmin:
            qmin = q
    return vmin, qmin


def star_trials(points, star, vertex, trials):
    """Min signed volume and min quality of a star for each trial position of ``vertex``.

    ``points`` (N, 3) array, ``star`` (k, 4) cell indices containing ``vertex``,
    ``trials`` (m, 3). Returns two arrays of length m. One vectorised pass over
    m*k tetrahedra; used when several candidate positions are scored at once.
    """
    star = np.asarray(star)
    trials = np.asarray(trials, dtype=float)
    base = points[star]                                   # (k, 4, 3)
    p = np.broadcast_to(base, (len(trials),) + base.shape).copy()
    rows, cols = np.nonzero(star == vertex)
    p[:, rows, cols] = trials[:, None, :]
    a = p[..., 1, :] - p[..., 0, :]
    b = p[..., 2, :] - p[..., 0, :]
    c = p[..., 3, :] - p[..., 0, :]
    volume = (a[..., 0] * (b[..., 1] * c[..., 2] - b[..., 2] * c[..., 1])
              + a[..., 1] * (b[..., 2] * c[..., 0] - b[..., 0] * c[..., 2])
              + a[..., 2] * (b[..., 0] * c[..., 1] - b[..., 1] * c[..., 0])) / 6
    d1 = p[..., 2, :] - p[..., 1, :]
    d2 = p[..., 3, :] - p[..., 1, :]
    d3 = p[..., 3, :] - p[..., 2, :]
    lengths = sum((x * x).sum(axis=-1) for x in (a, b, c, d1, d2, d3))
    quality = 12 * (3 * np.abs(volume)) ** (2 / 3) / np.where(lengths > 0, lengths, np.inf)
    return volume.min(axis=1), quality.min(axis=1)


def ideal_point(P, tet, vertex):
    """Position of ``vertex`` that makes ``tet`` regular over its fixed opposite face.

    The point lies on the face normal through the face centroid, on the side
    where the vertex currently is, at the height of a regular tetrahedron
    with the face's mean edge length.
    """
    a, b, c = (P[i] for i in tet if i != vertex)
    x = P[vertex]
    cx, cy, cz = (a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    norm = (nx * nx + ny * ny + nz * nz) ** 0.5
    if norm == 0:
        return list(x)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    if (x[0] - cx) * nx + (x[1] - cy) * ny + (x[2] - cz) * nz < 0:
        nx, ny, nz = -nx, -ny, -nz
    edge = ((ux * ux + uy * uy + uz * uz) ** 0.5 + (vx * vx + vy * vy + vz * vz) ** 0.5
            + ((c[0] - b[0]) ** 2 + (c[1] - b[1]) ** 2 + (c[2] - b[2]) ** 2) ** 0.5) / 3
    h = edge * (2 / 3) ** 0.5
    return [cx + h * nx, cy + h * ny, cz + h * nz]


def worst_tet(P, tets):
    """The tetrahedron of ``tets`` with the lowest mean-ratio quality."""
    return min(tets, key=lambda t: _tet_xyz(*(P[i] for i in t))[1])


_DOT_ORDERS = {
    "02_1": lambda u0, u1, u2, c0, c1, c2: (u0 * c0 + u2 * c2) + u1 * c1,
    "01_2": lambda u0, u1, u2, c0, c1, c2: (u0 * c0 + u1 * c1) + u2 * c2,
    "12_0": lambda u0, u1, u2, c0, c1, c2: (u1 * c1 + u2 * c2) + u0 * c0,
}


def _detect_dot_order(samples=4096):
    """Name of the 3-term dot-product order that reproduces this numpy's
    ``einsum('ij,ij->i')`` bit for bit, or None. Detected once at import so
    the scalar fan test below takes exactly the decisions of the numpy path
    on this machine and numpy build."""
    rng = np.random.default_rng(20261105)
    u = rng.standard_normal((samples, 3)) * rng.uniform(1e-3, 1e3, (samples, 1))
    c = rng.standard_normal((samples, 3)) * rng.uniform(1e-3, 1e3, (samples, 1))
    reference = np.einsum("ij,ij->i", u, c).tolist()
    rows = list(zip(u.tolist(), c.tolist()))
    for name, dot in _DOT_ORDERS.items():
        if all(dot(*x, *y) == r for (x, y), r in zip(rows, reference)):
            return name
    return None


_DOT = _DOT_ORDERS.get(_detect_dot_order())


def _volume_length_exact(a, b, c, d):
    """(|volume|, summed squared edge length) of one tetrahedron with the same
    floating-point operations, in the same order, as :func:`tetra_metrics`
    (requires ``_DOT``)."""
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    wx, wy, wz = d[0] - a[0], d[1] - a[1], d[2] - a[2]
    volume = abs(_DOT(ux, uy, uz, vy * wz - vz * wy, vz * wx - vx * wz, vx * wy - vy * wx) / 6)
    lengths = 0
    for p, q in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
        x, y, z = p[0] - q[0], p[1] - q[1], p[2] - q[2]
        lengths += x * x + y * y + z * z
    return volume, lengths


def _fan_metrics(nodes, tets, point=None):
    """|volumes| and qualities of a small fan, bit-identical to ``tetra_metrics``.

    Volumes and edge sums are scalar (bit-exact once ``_DOT`` is detected);
    the power and division stay in numpy because numpy's vectorised ``pow``
    can differ from the C library's by one ulp."""
    if _DOT is None:
        if point is None:
            xyz = np.array([[nodes[i] for i in tet] for tet in tets])
        else:
            xyz = np.array([[point, *[nodes[i] for i in tri]] for tri in tets])
        return tetra_metrics(xyz)
    if point is None:
        pairs = [_volume_length_exact(*(nodes[i] for i in tet)) for tet in tets]
    else:
        pairs = [_volume_length_exact(point, *(nodes[i] for i in tri)) for tri in tets]
    return _fan_pair_metrics(pairs)


def _fan_pair_metrics(pairs):
    volumes, lengths = np.array(pairs).T
    quality = np.divide(12 * (3 * volumes) ** (2 / 3), lengths, out=np.zeros_like(volumes), where=lengths > 0)
    return volumes, quality


def choose_fan(nodes, anchor, vertices, triangles, strategy):
    baseline = [[anchor, *tri] for tri in triangles if anchor not in tri]
    if strategy == "pulling" or not baseline:
        return baseline, None, None
    # Scalar arithmetic on the current polytope only (3-10 cells): the numpy
    # path spent nearly all its time in call overhead at this size.
    # Most candidates are rejected. Scalar powers cheaply screen only clear
    # rejections; all close decisions and every accepted fan still use the
    # original numpy arithmetic. The margin is much larger than libm/numpy
    # rounding differences for finite mean-ratio qualities in [0, 1].
    baseline_pairs = None
    if _DOT is not None:
        baseline_pairs = [_volume_length_exact(*(nodes[i] for i in tet)) for tet in baseline]
        if any(v <= 0 for v, _ in baseline_pairs):
            return baseline, None, None
        approximate_baseline = min(12 * (3 * v) ** (2 / 3) / length if length > 0 else 0.0
                                   for v, length in baseline_pairs)
        if approximate_baseline > 0.3 + 1e-12:
            return baseline, None, None
        # Keep the exact quality threshold when the scalar screen is too close.
        if approximate_baseline >= 0.3 - 1e-12:
            vb, qb = _fan_pair_metrics(baseline_pairs)
            if qb.min() >= 0.3:
                return baseline, None, None
    else:
        vb, qb = _fan_metrics(nodes, baseline)
        if np.any(vb <= 0) or qb.min() >= 0.3:
            return baseline, None, None
    point = np.mean([nodes[i] for i in vertices], axis=0)
    if baseline_pairs is not None:
        point_list = point.tolist()
        candidate_pairs = [_volume_length_exact(point_list, *(nodes[i] for i in tri)) for tri in triangles]
        if any(v <= 0 for v, _ in candidate_pairs):
            return baseline, None, None
        approximate_candidate = min(12 * (3 * v) ** (2 / 3) / length if length > 0 else 0.0
                                    for v, length in candidate_pairs)
        if approximate_candidate < approximate_baseline * (1 + 1e-6) - 1e-12:
            return baseline, None, None
        vb, qb = _fan_pair_metrics(baseline_pairs)
        vc, qc = _fan_pair_metrics(candidate_pairs)
    else:
        vc, qc = _fan_metrics(nodes, triangles, point.tolist())
    if np.any(vb <= 0) or qb.min() >= 0.3:
        return baseline, None, None
    if np.any(vc <= 0) or qc.min() <= qb.min() * (1 + 1e-6):
        return baseline, None, None
    if not np.isclose(vc.sum(), vb.sum(), rtol=1e-10, atol=0):
        return baseline, None, None
    return [[len(nodes), *tri] for tri in triangles], point, float(qc.min() / qb.min())
