"""Periodic-matching outer faces for homogenisation (COMSOL periodic conditions).

With ``Config(periodic_faces=True)`` every box axis along which the design
field repeats exactly (TPMS with an integer number of cells and a threshold
that does not vary along that axis; checked numerically) gets identical node
sets and triangulations on its two opposite faces:

* generation copies the field samples of low-face lattice points to the
  matching high-face points, so cuts and snapping decisions coincide;
* afterwards the high-face node coordinates are set exactly to their low-face
  partner + box length (root-finding differs only by rounding there);
* ``optimize_mesh`` leaves nodes on periodic faces in place and skips
  reconnections that would change those face triangulations.

``check_periodic`` independently re-derives the node pairing and the
face-triangle correspondence (including solid/fluid boundary ids) and is
stored in ``report["periodic"]``.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .model import raw_field


def periodic_axes(config, samples=96, seed=0):
    """Axes along which raw_field(x) == raw_field(x + L e_axis) on the faces."""
    rng = np.random.default_rng(seed)
    size = np.asarray(config.size, dtype=float)
    axes = []
    for axis in range(3):
        p = rng.random((samples, 3)) * size
        lo, hi = p.copy(), p.copy()
        lo[:, axis] = 0.0
        hi[:, axis] = size[axis]
        a = raw_field(config, *lo.T)
        b = raw_field(config, *hi.T)
        scale = max(1.0, float(np.abs(a).max()))
        if np.max(np.abs(a - b)) <= 1e-9 * scale:
            axes.append(axis)
    return axes


def face_partners(points, axis, size, tol):
    """(low, high) index arrays of nodes paired across ``axis``; raises if unpaired."""
    low = np.flatnonzero(np.abs(points[:, axis]) < tol)
    high = np.flatnonzero(np.abs(points[:, axis] - size[axis]) < tol)
    other = [a for a in range(3) if a != axis]
    if len(low) != len(high):
        raise ValueError(f"Periodic axis {'xyz'[axis]}: {len(low)} low-face nodes but {len(high)} high-face nodes")
    if len(low) == 0:
        return low, high
    scale = 1.0 / max(tol * 100, 1e-12)
    key_low = np.round(points[low][:, other] * scale).astype(np.int64)
    key_high = np.round(points[high][:, other] * scale).astype(np.int64)
    order_low = np.lexsort(key_low.T[::-1])
    order_high = np.lexsort(key_high.T[::-1])
    if not np.array_equal(key_low[order_low], key_high[order_high]):
        diff = np.abs(points[low][order_low][:, other] - points[high][order_high][:, other]).max()
        raise ValueError(f"Periodic axis {'xyz'[axis]}: face nodes do not pair (max offset {diff:.3e} mm)")
    return low[order_low], high[order_high]


def check_periodic(model, axes, size, tol=None):
    """Verify node pairing and face-triangle correspondence on every periodic axis."""
    points = np.asarray(model["points"], dtype=float)
    size = np.asarray(size, dtype=float)
    tol = float(size.max()) * 1e-7 if tol is None else tol
    boundary = np.asarray(model["boundary"])
    ids = np.asarray(model["boundary_ids"])
    report = {"axes": ["xyz"[a] for a in axes], "per_axis": {}}
    for axis in axes:
        low, high = face_partners(points, axis, size, tol)
        other = [a for a in range(3) if a != axis]
        offset = float(np.abs(points[low][:, other] - points[high][:, other]).max()) if len(low) else 0.0
        partner = np.full(len(points), -1)
        partner[high] = low
        lo_id = {2 + 2 * axis, 12 + 2 * axis}
        hi_id = {3 + 2 * axis, 13 + 2 * axis}
        low_tris = {(tuple(sorted(map(int, f))), int(i) - (2 + 2 * axis)) for f, i in zip(boundary, ids) if int(i) in lo_id}
        high_tris = {(tuple(sorted(int(partner[v]) for v in f)), int(i) - (3 + 2 * axis))
                     for f, i in zip(boundary, ids) if int(i) in hi_id}
        report["per_axis"]["xyz"[axis]] = {"paired_nodes": int(len(low)), "max_in_plane_offset_mm": offset,
                                           "face_triangles": int(len(low_tris)),
                                           "triangulations_match": bool(low_tris == high_tris)}
    report["matched"] = all(v["triangulations_match"] and v["max_in_plane_offset_mm"] == 0.0
                            for v in report["per_axis"].values())
    return report


def enforce_periodic(model, axes, size):
    """Copy low-face coordinates (+ L) onto paired high-face nodes; refresh hashes; return report."""
    from .volume import boundary_digest

    points = np.asarray(model["points"], dtype=float)
    size = np.asarray(size, dtype=float)
    tol = float(size.max()) * 1e-7
    for axis in axes:
        low, high = face_partners(points, axis, size, tol)
        other = [a for a in range(3) if a != axis]
        points[np.ix_(high, other)] = points[np.ix_(low, other)]
        points[high, axis] = size[axis]
    model["points"] = points
    report = check_periodic(model, axes, size, tol)
    if not report["matched"]:
        raise ValueError("Periodic faces could not be matched: " + str(report["per_axis"]))
    rep = model["report"]
    if "mesh_sha256" in rep:
        rep["mesh_sha256"] = hashlib.sha256(
            points.astype("<f8").tobytes() + np.asarray(model["tetra"]).astype("<i4").tobytes()
            + np.asarray(model["domains"]).astype("<i4").tobytes()).hexdigest()
    if "boundary_sha256" in rep:
        rep["boundary_sha256"] = boundary_digest(points, model["boundary"])
    rep["periodic"] = report
    return report


def pair_smooth(points, tetra, frozen_pairs, config, *, quality_threshold=0.5, sweeps=2,
                fractions=(1.0, 0.5, 0.25), on_surface=None):
    """Move paired periodic-face nodes together (same in-plane displacement).

    ``frozen_pairs``: list of (axis, low, high) node index arrays. A pair moves
    towards the mean of the in-plane neighbours of both copies (neighbours of
    the high copy are shifted by -L); interface nodes are re-projected onto
    their level set inside the face plane. A move is kept when no cell of
    either star inverts and the worst quality over both stars improves.
    Returns (points, accepted).
    """
    from .quality import signed_volumes, tetra_metrics
    from .surface_smoothing import _level_set

    points = np.array(points, dtype=float)
    tetra = np.asarray(tetra)
    size = np.asarray(config.size, dtype=float)
    tol = float(size.max()) * 1e-9
    n = len(points)
    order = np.argsort(tetra.ravel(), kind="stable")
    starts = np.searchsorted(tetra.ravel()[order], np.arange(n + 1))
    cell_of = order // 4
    sides = 2 if config.mode == "sheet" else 1
    on_plane = (np.abs(points) < tol) | (np.abs(points - size) < tol)
    accepted = 0

    def star(v):
        return cell_of[starts[v]:starts[v + 1]]

    for _ in range(sweeps):
        _, q = tetra_metrics(points[tetra])
        worst = np.full(n, np.inf)
        np.minimum.at(worst, tetra.ravel(), np.repeat(q, 4))
        moved = 0
        for axis, low, high in frozen_pairs:
            other = [a for a in range(3) if a != axis]
            for lo, hi in zip(low, high):
                current = min(worst[lo], worst[hi])
                if current >= quality_threshold:
                    continue
                pin = on_plane[lo].copy()
                pin[axis] = True
                if pin.sum() >= 3 or any(on_plane[lo, a] for a, _, _ in frozen_pairs if a != axis):
                    continue  # nodes on two periodic planes have more than one image
                cells = np.concatenate([star(lo), star(hi)])
                nb = []
                for v, shift in ((lo, 0.0), (hi, -size[axis])):
                    for c in star(v):
                        for w in tetra[c]:
                            if w != v and abs(points[w, axis] - points[v, axis]) < tol:
                                p = points[w].copy()
                                p[axis] += shift
                                nb.append(p)
                if not nb:
                    continue
                target = np.mean(nb, axis=0)
                surface_side = None
                if on_surface is not None and on_surface[lo]:
                    residual = [abs(_level_set(config, points[lo][None], s)[0]) for s in range(sides)]
                    surface_side = int(np.argmin(residual))
                best = None
                for f in fractions:
                    x = points[lo].copy()
                    x[~pin] += f * (target - points[lo])[~pin]
                    if surface_side is not None:
                        for _ in range(8):
                            g = _level_set(config, x[None], surface_side)[0]
                            grad = np.zeros(3)
                            h = float(size.max()) * 1e-6
                            for a_ in np.flatnonzero(~pin):
                                e = np.zeros(3)
                                e[a_] = h
                                grad[a_] = (_level_set(config, (x + e)[None], surface_side)[0]
                                            - _level_set(config, (x - e)[None], surface_side)[0]) / (2 * h)
                            nrm = grad @ grad
                            if nrm < 1e-30:
                                break
                            x = x - g / nrm * grad
                        if abs(_level_set(config, x[None], surface_side)[0]) > 1e-9:
                            continue
                    if np.any(x[other] < -tol) or np.any(x[other] > size[other] + tol):
                        continue
                    trial = points.copy()
                    trial[lo] = x
                    trial[hi] = x
                    trial[hi, axis] = size[axis]
                    if np.any(signed_volumes(trial[tetra[cells]]) <= 0):
                        continue
                    score = tetra_metrics(trial[tetra[cells]])[1].min()
                    if score > current * (1 + 1e-9) and (best is None or score > best[0]):
                        best = (score, trial[lo].copy(), trial[hi].copy())
                if best is not None:
                    points[lo], points[hi] = best[1], best[2]
                    for c in cells:
                        q_c = tetra_metrics(points[tetra[c]][None])[1][0]
                        for w in tetra[c]:
                            worst[w] = min(worst[w], q_c) if w not in (lo, hi) else worst[w]
                    worst[lo] = worst[hi] = best[0]
                    moved += 1
        accepted += moved
        if moved == 0:
            break
    return points, accepted
