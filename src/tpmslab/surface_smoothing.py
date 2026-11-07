"""Surface-constrained smoothing of boundary vertices on the exact implicit field.

Cells whose four vertices all lie on the boundary cannot be improved by
interior smoothing or by reconnections that keep every boundary face. The
generator, unlike a mesher that only receives triangles, still knows the
implicit field that produced the surface, so a boundary vertex can slide:

* a vertex on the solid/fluid interface moves within the tangent plane and is
  re-projected onto the zero level set (Newton steps along the field
  gradient), so it stays *on the exact implicit surface*;
* a vertex on one outer box face moves within that face, on a box edge along
  the edge, and a box corner never moves; a vertex on both a box face and the
  interface satisfies both constraints.

A move is accepted only if every incident tetrahedron (in both phases) stays
positively oriented and the worst quality of that star strictly improves.
Boundary and interface connectivity are unchanged, so the two phases remain
conforming; phase volumes change slightly because the chordal boundary
follows the surface more closely, and the caller recomputes them.
"""

from __future__ import annotations

import numpy as np

from .model import field_and_threshold
from .quality import tetra_metrics

_LOCAL_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _level_set(config, xyz, side):
    """Inequality value g (>= 0 inside material) of ``side`` at points xyz."""
    xyz = np.atleast_2d(xyz)
    value, threshold = field_and_threshold(config, xyz[:, 0], xyz[:, 1], xyz[:, 2])
    value = np.broadcast_to(np.asarray(value, dtype=float), len(xyz))
    threshold = np.broadcast_to(np.asarray(threshold, dtype=float), len(xyz))
    if config.mode == "sheet":
        return threshold - value if side == 0 else threshold + value
    return threshold + value if config.mode == "solid_above" else threshold - value


def classify_boundary_vertices(points, boundary, boundary_ids, size, tol):
    """Return (planes, on_surface): per-vertex box-plane list and implicit-surface flag."""
    n = len(points)
    planes = [[] for _ in range(n)]
    on_surface = np.zeros(n, dtype=bool)
    for face, pid in zip(np.asarray(boundary), np.asarray(boundary_ids)):
        pid = int(pid)
        if pid == 8:
            on_surface[face] = True
            continue
        axis_side = pid - 2 if pid < 10 else pid - 12
        axis, side = axis_side // 2, axis_side % 2
        value = size[axis] if side else 0.0
        for v in face:
            if (axis, value) not in planes[v]:
                planes[v].append((axis, value))
    return planes, on_surface


def surface_smooth(points, tetra, boundary, boundary_ids, config, *, quality_threshold=0.35,
                   sweeps=2, step_fractions=(1.0, 0.5, 0.25, 0.125), newton_iterations=6,
                   chord_tolerance=None, chord_select=True, chord_gain_floor=0.1, adjacency=None,
                   phase_ids=None, frozen=None, envelope=None):
    """Slide boundary vertices along the box faces and the implicit surface.

    Returns ``(points, info)``; ``points`` is a new array. Only vertices whose
    incident worst quality is below ``quality_threshold`` are considered.
    Candidate positions for all vertices and all step fractions are projected
    onto the level set in one vectorised Newton iteration per sweep; moves are
    then accepted on independent vertex batches (stars sharing no cell), each
    scored in one vectorised pass against the current positions. Chord errors
    are also refreshed for each bounded batch, after neighbouring batches
    have moved. ``chord_tolerance=None`` ranks candidates without a hard chord
    limit; zero enforces non-increase up to the numerical comparison tolerance.
    With ``phase_ids``, moves at a material interface must also preserve the
    minimum quality of each phase in the affected star.
    ``envelope`` (mm, optional): surface vertices may also take the unprojected
    trial positions when their first-order distance |g|/|grad g| to the level
    set stays within ``envelope``; the hard chord limit is then not applied
    (chord error still ranks candidates).
    """
    from . import batch_smoothing as bs

    points = np.array(points, dtype=float)
    tetra = np.asarray(tetra)
    if phase_ids is not None:
        phase_ids = np.asarray(phase_ids)
        if phase_ids.shape != (len(tetra),) or phase_ids.dtype.kind not in "iu":
            raise ValueError("phase_ids must contain one integer label per tetrahedron")
    size = np.asarray(config.size, dtype=float)
    tol = float(size.max()) * 1e-9
    field_tol = 1e-9
    h = float(size.max()) * 1e-6
    planes, on_surface = classify_boundary_vertices(points, boundary, boundary_ids, size, tol)
    boundary_vertex = np.zeros(len(points), dtype=bool)
    boundary_vertex[np.unique(np.asarray(boundary))] = True
    if adjacency is None:
        adjacency = (*bs.incidence(tetra, len(points)), *bs.neighbours(tetra, len(points)))
    cell_ids, cell_starts, nb_ids, nb_starts = adjacency
    sides = 2 if config.mode == "sheet" else 1
    # Implicit-surface triangles incident to each vertex, for the chord-error guard.
    surface_faces = np.asarray(boundary)[np.asarray(boundary_ids) == 8]
    faces_of = [[] for _ in range(len(points))]
    for fi, face in enumerate(surface_faces):
        for v in face:
            faces_of[v].append(fi)
    # Per-vertex plane mask (True where a coordinate is pinned) and pinned values.
    pinned = np.zeros((len(points), 3), dtype=bool)
    pinned_value = np.zeros((len(points), 3))
    for v, items in enumerate(planes):
        for axis, value in items:
            pinned[v, axis] = True
            pinned_value[v, axis] = value
    if frozen is not None:
        # Nodes on periodic faces keep their (paired) positions.
        pinned[frozen] = True
        pinned_value[frozen] = points[frozen]
    corner = pinned.sum(axis=1) >= 3

    def project(x, vertex_ids, side_ids):
        """Vectorised Newton projection of rows of x onto their level sets."""
        x = x.copy()
        pin = pinned[vertex_ids]
        pin_value = pinned_value[vertex_ids]
        x[pin] = pin_value[pin]
        ok = np.ones(len(x), dtype=bool)
        active = np.ones(len(x), dtype=bool)
        for _ in range(newton_iterations):
            g = np.empty(len(x))
            for s_ in range(sides):
                mask = active & (side_ids == s_)
                if np.any(mask):
                    g[mask] = _level_set(config, x[mask], s_)
            converged = np.abs(g) < field_tol
            active &= ~converged
            if not np.any(active):
                break
            grad = np.zeros((len(x), 3))
            for s_ in range(sides):
                mask = active & (side_ids == s_)
                if not np.any(mask):
                    continue
                xm = x[mask]
                for axis in range(3):
                    offset = np.zeros(3)
                    offset[axis] = h
                    grad[mask, axis] = (_level_set(config, xm + offset, s_) - _level_set(config, xm - offset, s_)) / (2 * h)
            grad[pin] = 0.0
            norm2 = np.einsum("ij,ij->i", grad, grad)
            stuck = active & (norm2 < 1e-30)
            ok &= ~stuck
            active &= ~stuck
            step = np.zeros(len(x))
            step[active] = g[active] / norm2[active]
            x -= step[:, None] * grad
            x[pin] = pin_value[pin]
        if np.any(active):
            g = np.empty(len(x))
            for s_ in range(sides):
                mask = active & (side_ids == s_)
                if np.any(mask):
                    g[mask] = _level_set(config, x[mask], s_)
            ok[active] &= np.abs(g[active]) < field_tol
        return x, ok

    attempted = accepted = phase_guard_rejected = 0
    moved = set()
    for _ in range(sweeps):
        _, quality = tetra_metrics(points[tetra])
        severity = np.full(len(points), np.inf)
        np.minimum.at(severity, tetra.ravel(), np.repeat(quality, 4))
        candidates = np.flatnonzero(boundary_vertex & ~corner & (severity < quality_threshold))
        candidates = candidates[np.argsort(severity[candidates], kind="stable")]
        if len(candidates) == 0:
            break
        # Which level set each surface candidate lies on (nearest side), vectorised.
        xc = points[candidates]
        residual = np.stack([np.abs(_level_set(config, xc, s_)) for s_ in range(sides)])
        side_ids = residual.argmin(axis=0)
        if envelope:
            gnorm = np.stack([_gradient_norm(config, xc, s_, h) for s_ in range(sides)])
            near = (residual / np.maximum(gnorm, 1e-30)).min(axis=0) <= envelope * (1 + 1e-9)
            surface_candidate = on_surface[candidates] & ((residual.min(axis=0) < 1e-6) | near)
        else:
            surface_candidate = on_surface[candidates] & (residual.min(axis=0) < 1e-6)
        keep = ~on_surface[candidates] | surface_candidate
        candidates, side_ids = candidates[keep], side_ids[keep]
        surface_candidate = surface_candidate[keep]
        # Candidate directions, constrained to the box planes: neighbour
        # centroid (Laplacian) and the ideal point of the worst incident cell.
        x0 = points[candidates]
        ideal = np.empty_like(x0)
        laplace = bs.neighbour_means(points, candidates, nb_ids, nb_starts)
        for part in bs.chunks(candidates):
            cells, _, _, worst = bs.star_state(points, tetra, candidates[part], cell_ids, cell_starts)
            ideal[part] = bs.ideal_points(points, cells, worst, candidates[part])
        direction = np.stack((laplace - x0, ideal - x0))  # (2, n, 3)
        direction[:, pinned[candidates]] = 0.0
        # All step fractions and directions at once: trial index = (fraction, direction).
        fractions = np.asarray(step_fractions)
        trial = (points[candidates][None, None] + fractions[:, None, None, None] * direction[None])
        n_trials = len(fractions) * 2
        trial = trial.reshape(-1, 3)
        trial_vertex = np.tile(candidates, n_trials)
        trial_side = np.tile(side_ids, n_trials)
        trial_surface = np.tile(surface_candidate, n_trials)
        valid = np.ones(len(trial), dtype=bool)
        raw_trial = trial.copy()
        if np.any(trial_surface):
            projected, ok = project(trial[trial_surface], trial_vertex[trial_surface], trial_side[trial_surface])
            trial[trial_surface] = projected
            valid[trial_surface] = ok
        if envelope and np.any(trial_surface):
            # Second trial set: the same steps without projection, kept when the
            # vertex stays within the envelope of its level set.
            pin_r = pinned[trial_vertex]
            raw_trial[pin_r] = pinned_value[trial_vertex][pin_r]
            raw_valid = np.zeros(len(raw_trial), dtype=bool)
            xs = raw_trial[trial_surface]
            sides_s = trial_side[trial_surface]
            dist = np.full(len(xs), np.inf)
            for s_ in range(sides):
                m_ = sides_s == s_
                if np.any(m_):
                    dist[m_] = np.abs(_level_set(config, xs[m_], s_)) / np.maximum(
                        _gradient_norm(config, xs[m_], s_, h), 1e-30)
            raw_valid[trial_surface] = dist <= envelope
            trial = np.concatenate([trial, raw_trial])
            valid = np.concatenate([valid, raw_valid])
            trial_vertex = np.concatenate([trial_vertex, trial_vertex])
            trial_side = np.concatenate([trial_side, trial_side])
            n_trials *= 2
        pin = pinned[trial_vertex]
        trial[pin] = pinned_value[trial_vertex][pin]
        valid &= np.all(trial >= -tol, axis=1) & np.all(trial <= size + tol, axis=1)
        trial = trial.reshape(n_trials, len(candidates), 3)
        valid = valid.reshape(n_trials, len(candidates))
        moved_this_sweep = 0
        row = np.full(len(points), -1, dtype=np.int64)
        row[candidates] = np.arange(len(candidates))
        trial_by_row = np.ascontiguousarray(trial.transpose(1, 0, 2))      # (n, T, 3)
        valid_by_row = valid.T
        attempted += int(np.sum(np.any(direction != 0, axis=(0, 2))))
        for batch in bs.independent_batches(candidates, cell_ids, cell_starts):
            for part in bs.chunks(batch):
                vertices = batch[part]
                rows = row[vertices]
                current_phases = {}
                if phase_ids is None:
                    cells, mask, current, _ = bs.star_state(points, tetra, vertices, cell_ids, cell_starts)
                else:
                    cells, mask, current, _ = bs.star_state(
                        points, tetra, vertices, cell_ids, cell_starts,
                        phase_ids=phase_ids, phase_minima=current_phases)
                trials = trial_by_row[rows]
                mixed = (sum(np.isfinite(v).astype(int) for v in current_phases.values()) > 1
                         if current_phases else np.zeros(len(vertices), dtype=bool))
                trial_phases = {}
                if np.any(mixed):
                    star, _ = bs.padded(cell_ids, cell_starts, vertices)
                    vmin, qmin = bs.score_trials(
                        points, cells, mask, vertices, trials,
                        phase_ids=phase_ids[star], phase_minima=trial_phases)
                else:
                    vmin, qmin = bs.score_trials(points, cells, mask, vertices, trials)
                batch_valid = valid_by_row[rows].copy()
                phase_ok = np.ones(batch_valid.shape, dtype=bool)
                for phase, proposed in trial_phases.items():
                    present = mixed & np.isfinite(current_phases[phase])
                    phase_ok[present] &= proposed[present] >= current_phases[phase][present, None] - 1e-12
                phase_guard_rejected += int(np.sum(batch_valid & ~phase_ok))
                batch_valid &= phase_ok
                batch_surface = surface_candidate[rows]
                batch_chord = None
                # Chord evaluations cannot rescue an inverted or non-improving
                # trial. Remove such rows before the expensive field queries.
                viable = batch_valid & (vmin > 0) & (qmin > (current * (1 + 1e-6))[:, None])
                guard_surface = batch_surface & viable.any(axis=1)
                if ((chord_tolerance is not None or chord_select) and not envelope
                        and np.any(guard_surface) and len(surface_faces)):
                    # Different independent batches can share surface triangles.
                    # Evaluate their geometry against the positions immediately
                    # before acceptance, just as score_trials does for quality.
                    # Restricting the guard to this chunk also bounds its large
                    # (trials, vertices, incident faces, 3, 3) temporary array.
                    keep, delta = _chord_guard(
                        points, surface_faces, faces_of, vertices, side_ids[rows],
                        guard_surface, trials.transpose(1, 0, 2), config, h,
                        chord_tolerance)
                    batch_valid &= keep.T
                    if chord_select:
                        batch_chord = delta.T
                pick = bs.choose(vmin, qmin, current, valid=batch_valid,
                                 chord=batch_chord,
                                 chord_rows=batch_surface if chord_select else None,
                                 gain_floor=chord_gain_floor)
                take = pick >= 0
                points[vertices[take]] = trials[np.flatnonzero(take), pick[take]]
                moved_this_sweep += int(take.sum())
                moved.update(map(int, vertices[take]))
        accepted += moved_this_sweep
        if moved_this_sweep == 0:
            break
    info = {"method": "tangential Laplacian step re-projected onto the implicit level set / box faces",
            "attempted": int(attempted), "accepted": int(accepted), "quality_threshold": quality_threshold,
            "phase_guard_rejected_trials": int(phase_guard_rejected),
            "moved_vertices": sorted(moved)}
    return points, info


def _gradient_norm(config, x, side, h):
    """|grad g| at points ``x`` (n, 3) by central differences, per level-set side."""
    side = np.broadcast_to(side, len(x))
    out = np.empty(len(x))
    for s_ in np.unique(side):
        mask = side == s_
        xm = x[mask]
        grad = np.empty((len(xm), 3))
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = h
            grad[:, axis] = (_level_set(config, xm + offset, s_) - _level_set(config, xm - offset, s_)) / (2 * h)
        out[mask] = np.linalg.norm(grad, axis=1)
    return out


def _chord_error(tri, side_of_tri, config, h, grad_norm=None):
    """Area-weighted distance from triangle centroids to the level set.

    ``tri`` is (..., 3, 3). Returns (...) values of area * |g| / |grad g| at the
    centroid: a first-order estimate of how far the flat triangle departs from
    the exact surface, i.e. its local contribution to the phase-volume error.
    ``grad_norm`` (broadcastable to the leading shape) supplies |grad g| from a
    nearby point, which saves six field evaluations per centroid; the field
    varies slowly on the scale of one triangle, and every caller compares
    values that share the same normalisation.
    """
    shape = tri.shape[:-2]
    centroid = tri.mean(axis=-2).reshape(-1, 3)
    side = np.broadcast_to(side_of_tri, shape).reshape(-1)
    area = 0.5 * np.linalg.norm(np.cross(tri[..., 1, :] - tri[..., 0, :], tri[..., 2, :] - tri[..., 0, :]), axis=-1)
    g = np.empty(len(centroid))
    for s_ in np.unique(side):
        mask = side == s_
        g[mask] = _level_set(config, centroid[mask], s_)
    if grad_norm is None:
        norm = _gradient_norm(config, centroid, side, h)
    else:
        norm = np.broadcast_to(grad_norm, shape).reshape(-1)
    distance = np.abs(g) / np.maximum(norm, 1e-300)
    return area * distance.reshape(shape)


def _chord_guard(points, surface_faces, faces_of, candidates, side_ids, surface_candidate, trial, config, h,
                 tolerance=0.0):
    """(n_trials, n_candidates) mask: a surface move must not increase the
    chord error of the vertex's incident surface triangles (a move that keeps
    the vertex on the level set can still make the flat surface deviate more)."""
    n_trials, n_cand = trial.shape[:2]
    keep = np.ones((n_trials, n_cand), dtype=bool)
    error = np.zeros((n_trials, n_cand))
    rows = np.flatnonzero(surface_candidate)
    counts = np.array([len(faces_of[candidates[j]]) for j in rows], dtype=np.int64)
    active = counts > 0
    rows, counts = rows[active], counts[active]
    if not len(rows):
        return keep, error
    # Pack only actual incident faces. A moving triangle's centroid and
    # area normal depend affinely on its moving vertex, so no array of full
    # triangle coordinates is needed for every trial, nor any padded faces.
    owners = np.repeat(np.arange(len(rows)), counts)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    face_ids = np.concatenate([faces_of[candidates[j]] for j in rows])
    tri_ids = surface_faces[face_ids]
    moving_ids = candidates[rows][owners]
    fixed_ids = tri_ids[tri_ids != moving_ids[:, None]].reshape(-1, 2)
    a, b = points[fixed_ids[:, 0]], points[fixed_ids[:, 1]]
    x0 = points[moving_ids]
    side = side_ids[rows][owners]
    # One gradient per candidate vertex, shared by its triangles and trials.
    norm = _gradient_norm(config, points[candidates[rows]], side_ids[rows], h)[owners]

    def face_errors(centres, normals):
        leading = centres.shape[:-1]
        flat = centres.reshape(-1, 3)
        which_side = np.broadcast_to(side, leading).reshape(-1)
        g = np.empty(len(flat))
        for s in np.unique(side):
            take = which_side == s
            g[take] = _level_set(config, flat[take], s)
        return .5 * np.linalg.norm(normals, axis=-1) * np.abs(g.reshape(leading)) / np.maximum(norm, 1e-300)

    normal = np.cross(a - x0, b - x0)
    current = np.add.reduceat(face_errors((a + b + x0) / 3, normal), starts)
    replacement = trial[:, rows][:, owners]
    normals = normal[None] + np.cross((b - a)[None], replacement - x0[None])
    after = np.add.reduceat(face_errors((a[None] + b[None] + replacement) / 3, normals), starts, axis=1)
    error[:, rows] = after - current[None]
    if tolerance is not None:
        keep[:, rows] = after <= current[None] * (1 + tolerance + 1e-6) + 1e-15
    return keep, error


def surface_distances(points, vertices, config, h=None):
    """First-order distance |g|/|grad g| of ``vertices`` to the nearest level set (mm)."""
    xyz = np.asarray(points)[np.asarray(vertices)]
    h = float(max(config.size)) * 1e-6 if h is None else h
    sides = 2 if config.mode == "sheet" else 1
    values = np.stack([np.abs(_level_set(config, xyz, s)) / np.maximum(_gradient_norm(config, xyz, s, h), 1e-30)
                       for s in range(sides)])
    return values.min(axis=0)


def surface_residuals(points, vertices, config):
    """Absolute level-set residual of ``vertices`` (nearest side)."""
    xyz = np.asarray(points)[np.asarray(vertices)]
    sides = 2 if config.mode == "sheet" else 1
    values = np.stack([np.abs(_level_set(config, xyz, s)) for s in range(sides)])
    return values.min(axis=0)
