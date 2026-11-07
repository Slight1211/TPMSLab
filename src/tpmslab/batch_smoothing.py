"""Vectorised maximin vertex smoothing on independent vertex sets.

Moving one vertex only changes the tetrahedra of its star. Vertices whose
stars share no tetrahedron can therefore be moved simultaneously without
interacting. Candidates are split into such independent batches (worst
quality first: a vertex joins the current batch when it has the highest
priority among all remaining candidates of every incident cell), and every
trial position of every vertex in a batch is scored in one numpy pass. This
replaces one small numpy call per vertex and per trial (all call overhead).

Acceptance is exactly the rule of the scalar loops it replaces: a move is
taken only if no incident cell inverts and the worst incident mean-ratio
quality strictly improves.
"""

from __future__ import annotations

import numpy as np

from .quality import unique_rows

LOCAL_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_CHUNK = 1024


def incidence(tetra, n_points):
    """CSR incidence: cells of vertex v are ``cells[starts[v]:starts[v + 1]]``."""
    flat = np.asarray(tetra).ravel()
    order = np.argsort(flat, kind="stable")
    starts = np.searchsorted(flat[order], np.arange(n_points + 1))
    return order // 4, starts


def neighbours(tetra, n_points):
    """CSR vertex adjacency through mesh edges (each neighbour once)."""
    tetra = np.asarray(tetra)
    edges = np.sort(tetra[:, LOCAL_EDGES].reshape(-1, 2), axis=1)
    edges = unique_rows(edges)
    both = np.concatenate([edges, edges[:, ::-1]])
    both = both[np.lexsort((both[:, 1], both[:, 0]))]
    starts = np.searchsorted(both[:, 0], np.arange(n_points + 1))
    return both[:, 1], starts


def padded(values, starts, rows):
    """(len(rows), K) gather of CSR rows with a validity mask."""
    lo, hi = starts[rows], starts[rows + 1]
    width = int((hi - lo).max()) if len(rows) else 0
    index = lo[:, None] + np.arange(max(width, 1))[None, :]
    mask = index < hi[:, None]
    return values[np.where(mask, index, 0)], mask


def independent_batches(candidates, cell_ids, cell_starts):
    """Split ``candidates`` (in priority order) into vertex sets with disjoint stars."""
    remaining = np.asarray(candidates)
    batches = []
    n_cells = int(cell_ids.max()) + 1 if len(cell_ids) else 0
    while len(remaining):
        priority = np.arange(len(remaining))
        cells, mask = padded(cell_ids, cell_starts, remaining)
        best = np.full(n_cells, len(remaining), dtype=np.int64)
        np.minimum.at(best, cells[mask], np.broadcast_to(priority[:, None], cells.shape)[mask])
        own = np.where(mask, best[cells] == priority[:, None], True).all(axis=1)
        batches.append(remaining[own])
        remaining = remaining[~own]
    return batches


def volume_quality(p):
    """Signed volume and mean-ratio quality of tetrahedra ``p[..., 4, 3]``."""
    a = p[..., 1, :] - p[..., 0, :]
    b = p[..., 2, :] - p[..., 0, :]
    c = p[..., 3, :] - p[..., 0, :]
    volume = (a[..., 0] * (b[..., 1] * c[..., 2] - b[..., 2] * c[..., 1])
              + a[..., 1] * (b[..., 2] * c[..., 0] - b[..., 0] * c[..., 2])
              + a[..., 2] * (b[..., 0] * c[..., 1] - b[..., 1] * c[..., 0])) / 6
    d = p[..., 2, :] - p[..., 1, :]
    e = p[..., 3, :] - p[..., 1, :]
    f = p[..., 3, :] - p[..., 2, :]
    lengths = sum((x * x).sum(axis=-1) for x in (a, b, c, d, e, f))
    quality = 12 * (3 * np.abs(volume)) ** (2 / 3) / np.where(lengths > 0, lengths, np.inf)
    return volume, quality


def star_state(points, tetra, vertices, cell_ids, cell_starts, *, phase_ids=None, phase_minima=None):
    """Padded stars of ``vertices``: cells (B, K, 4), mask (B, K), current min quality (B,),
    index of the worst cell (B,)."""
    star, mask = padded(cell_ids, cell_starts, vertices)
    cells = tetra[star]
    _, quality = volume_quality(points[cells])
    quality = np.where(mask, quality, np.inf)
    if phase_ids is not None and phase_minima is not None:
        star_phases = np.asarray(phase_ids)[star]
        for phase in np.unique(star_phases[mask]):
            phase_minima[int(phase)] = np.where(mask & (star_phases == phase), quality, np.inf).min(axis=1)
    return cells, mask, quality.min(axis=1), quality.argmin(axis=1)


def score_trials(points, cells, mask, vertices, trials, *, phase_ids=None, phase_minima=None):
    """Min signed volume and min quality over each star for trials (B, T, 3).

    Only the vertex itself moves, so for each star cell the signed volume is
    affine and the summed squared edge length quadratic in its position x:
    V(x) = (x . n - c) / 6 and L(x) = L0 + 3|x|^2 - 2 x . s, with n, c, L0, s
    from the three fixed corners. All trials are then dot products, without
    materialising (B, T, K, 4, 3) coordinates. Optional ``phase_ids`` are
    per-star cell labels (B, K); ``phase_minima`` receives per-phase trial
    minima (B, T), reusing the same computed quality array. The ordinary
    two-array return value is unchanged.
    """
    base = points[cells]                                            # (B, K, 4, 3)
    slot = np.argmax(cells == vertices[:, None, None], axis=2)      # (B, K) index of the vertex
    order = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    fixed = np.take_along_axis(base, order[slot][..., None], axis=2)  # (B, K, 3, 3)
    a, b, c = fixed[..., 0, :], fixed[..., 1, :], fixed[..., 2, :]
    # Moving the vertex from slot i to the front is a permutation of parity i,
    # and det[p1 - p0, p2 - p0, p3 - p0] = -(p0 - p1) . ((p2 - p1) x (p3 - p1)).
    sign = np.where(slot % 2 == 0, -1.0, 1.0)
    normal = np.cross(b - a, c - a) * sign[..., None]               # (B, K, 3)
    offset = np.einsum("bkd,bkd->bk", a, normal)
    fixed_lengths = ((b - a) ** 2).sum(-1) + ((c - a) ** 2).sum(-1) + ((c - b) ** 2).sum(-1)
    corner_sum = a + b + c
    corner_sq = (a * a).sum(-1) + (b * b).sum(-1) + (c * c).sum(-1)
    volume = (np.einsum("btd,bkd->btk", trials, normal) - offset[:, None, :]) / 6
    lengths = (fixed_lengths + corner_sq)[:, None, :] + 3 * (trials * trials).sum(-1)[..., None] \
        - 2 * np.einsum("btd,bkd->btk", trials, corner_sum)
    quality = 12 * (3 * np.abs(volume)) ** (2 / 3) / np.where(lengths > 0, lengths, np.inf)
    volume = np.where(mask[:, None], volume, np.inf)
    quality = np.where(mask[:, None], quality, np.inf)
    if phase_ids is not None and phase_minima is not None:
        phase_ids = np.asarray(phase_ids)
        for phase in np.unique(phase_ids[mask]):
            selected = mask & (phase_ids == phase)
            phase_minima[int(phase)] = np.where(selected[:, None], quality, np.inf).min(axis=2)
    return volume.min(axis=2), quality.min(axis=2)


def ideal_points(points, cells, worst, vertices):
    """For each vertex, the point that makes its worst incident cell regular
    over the opposite (fixed) face, on the vertex's side of that face."""
    tet = cells[np.arange(len(vertices)), worst]                    # (B, 4)
    others = np.sort(np.where(tet == vertices[:, None], np.iinfo(tet.dtype).max, tet), axis=1)[:, :3]
    a, b, c = points[others[:, 0]], points[others[:, 1]], points[others[:, 2]]
    centroid = (a + b + c) / 3
    normal = np.cross(b - a, c - a)
    norm = np.linalg.norm(normal, axis=1)
    safe = np.where(norm > 0, norm, 1.0)
    normal = normal / safe[:, None]
    side = np.sign(np.einsum("ij,ij->i", points[vertices] - centroid, normal))
    normal *= np.where(side == 0, 1.0, side)[:, None]
    edge = (np.linalg.norm(b - a, axis=1) + np.linalg.norm(c - a, axis=1) + np.linalg.norm(c - b, axis=1)) / 3
    ideal = centroid + (edge * np.sqrt(2 / 3))[:, None] * normal
    return np.where((norm > 0)[:, None], ideal, points[vertices])


def neighbour_means(points, vertices, nb_ids, nb_starts):
    nb, mask = padded(nb_ids, nb_starts, vertices)
    total = np.where(mask[..., None], points[nb], 0.0).sum(axis=1)
    return total / np.maximum(mask.sum(axis=1), 1)[:, None]


def choose(vmin, qmin, current, *, valid=None, chord=None, chord_rows=None, gain_floor=0.1):
    """Index of the accepted trial per vertex, or -1.

    Default: the trial with the highest worst quality. For rows flagged in
    ``chord_rows``: among trials that reach ``gain_floor`` of the best gain,
    the one with the smallest ``chord`` value."""
    ok = (vmin > 0) & (qmin > (current * (1 + 1e-6))[:, None])
    if valid is not None:
        ok &= valid
    pick = np.where(ok, qmin, -np.inf).argmax(axis=1)
    if chord is not None and chord_rows is not None and np.any(chord_rows):
        gain = np.where(ok, qmin - current[:, None], -np.inf)
        best_gain = gain.max(axis=1)
        eligible = ok & (gain >= gain_floor * best_gain[:, None])
        by_chord = np.where(eligible, chord, np.inf).argmin(axis=1)
        pick = np.where(chord_rows, by_chord, pick)
    return np.where(ok.any(axis=1), pick, -1)


def chunks(array, size=_CHUNK):
    for start in range(0, len(array), size):
        yield slice(start, start + size)
