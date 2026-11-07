"""Stage-0 prototype: template inheritance + iso-surface transport (ITT).

A reference mesh (uniform density) is moved onto a target design whose
threshold differs (e.g. a density gradient): interface vertices are projected
onto the target level set by Newton steps, box-face vertices slide in their
planes, and the displacement is extended into both phases by a stiffened
Laplacian.  Nothing here is wired into the package yet.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from tpmslab.model import Config, field_and_threshold
from tpmslab.quality import signed_volumes, tetra_metrics as _abs_metrics


def tetra_metrics(points):
    """Signed volumes (negative = inverted) and mean-ratio quality."""
    v, q = _abs_metrics(points)
    return signed_volumes(points), q


def _branch(config, P, sign=None):
    """phi(x) whose zero set is the solid boundary branch (negative inside solid)."""
    f, q = field_and_threshold(config, P[:, 0], P[:, 1], P[:, 2])
    f = np.broadcast_to(f, q.shape) if np.ndim(f) else np.full(len(P), f)
    if config.mode == "sheet":
        if sign is None:
            sign = np.where(f >= 0, 1.0, -1.0)
        return sign * f - q, sign
    if config.mode == "solid_above":
        return -f - q, sign
    return f - q, sign


def _gradient(config, P, sign, step):
    g = np.empty_like(P)
    for c in range(3):
        e = np.zeros(3)
        e[c] = step
        g[:, c] = (_branch(config, P + e, sign)[0] - _branch(config, P - e, sign)[0]) / (2 * step)
    return g


def fixed_mask(points, size, tol=1e-9):
    size = np.asarray(size, float)
    return (np.abs(points) < tol * size) | (np.abs(points - size) < tol * size)


def project_interface(points, vertices, reference, target, iterations=8):
    """Newton-project ``vertices`` from the reference level set onto the target one."""
    size = np.asarray(target.size, float)
    P = points[vertices].copy()
    fixed = fixed_mask(P, size)
    _, sign = _branch(reference, P)
    step = 1e-6 * float(size.max())
    phi, _ = _branch(target, P, sign)
    for _ in range(iterations):
        g = _gradient(target, P, sign, step)
        g[fixed] = 0.0
        n2 = np.einsum("ij,ij->i", g, g)
        ok = n2 > 1e-20
        delta = np.zeros_like(P)
        delta[ok] = (phi[ok] / n2[ok])[:, None] * g[ok]
        # Damped Newton with backtracking on |phi|.
        t = np.ones(len(P))
        for _ in range(8):
            trial = np.clip(P - t[:, None] * delta, 0.0, size)
            new, _ = _branch(target, trial, sign)
            worse = np.abs(new) > np.abs(phi)
            if not worse.any():
                break
            t[worse] *= 0.5
        P, phi = trial, new
        if np.abs(phi).max() < 1e-12:
            break
    residual = np.abs(phi)
    return P, residual


def _edges(tetra):
    pairs = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    e = np.sort(tetra[:, pairs].reshape(-1, 2), axis=1)
    owner = np.repeat(np.arange(len(tetra)), 6)
    return e, owner


def extend(points, tetra, dirichlet_values, dirichlet_vertices, size, *, weights=None):
    """Hierarchical harmonic extension: box edges (1-D), then box faces (2-D), then the volume.

    A free vertex only couples to neighbours lying on all of its box planes, so
    edge vertices are interpolated monotonically along their edge and face
    vertices within their face before the volume solve sees them as Dirichlet.
    """
    n = len(points)
    e, owner = _edges(tetra)
    w = np.ones(len(e)) if weights is None else weights[owner]
    size = np.asarray(size, float)
    tol = 1e-9 * size
    low = np.abs(points) < tol
    high = np.abs(points - size) < tol
    bits = (low.astype(np.int64) * np.array([1, 4, 16])).sum(1) + (high.astype(np.int64) * np.array([2, 8, 32])).sum(1)
    count = low.sum(1) + high.sum(1)
    fixed = low | high
    known = np.zeros(n, bool)
    known[dirichlet_vertices] = True
    known |= count >= 3
    disp = np.zeros_like(points)
    disp[dirichlet_vertices] = dirichlet_values
    a = np.r_[e[:, 0], e[:, 1]]
    b = np.r_[e[:, 1], e[:, 0]]
    ww = np.r_[w, w]
    for level in (2, 1, 0):
        free = (count == level) & ~known
        if not free.any():
            continue
        sel = free[a] & ((bits[b] & bits[a]) == bits[a])
        ra, rb, rw = a[sel], b[sel], ww[sel]
        index = -np.ones(n, np.int64)
        fid = np.flatnonzero(free)
        index[fid] = np.arange(len(fid))
        diag = np.bincount(index[ra], weights=rw, minlength=len(fid))
        inner = free[rb]
        M = sp.coo_matrix((-rw[inner], (index[ra[inner]], index[rb[inner]])), shape=(len(fid), len(fid))).tocsr()
        M = M + sp.diags(np.where(diag > 0, diag, 1.0))
        rhs = np.zeros((len(fid), 3))
        np.add.at(rhs, index[ra[~inner]], rw[~inner, None] * disp[rb[~inner]])
        solved = spla.splu(M.tocsc()).solve(rhs)
        solved[fixed[fid]] = 0.0
        disp[fid] = solved
        known |= free
    return disp


def transport(model, reference, target, *, increments=4, stiffen=1.0):
    """Return new points of ``model`` moved from ``reference`` onto ``target``."""
    points = np.asarray(model["points"], float)
    tetra = np.asarray(model["tetra"], np.int64)
    size = np.asarray(target.size, float)
    surface = np.unique(np.asarray(model["interface"]))
    # Box-face boundary triangles that belong to neither phase interface keep their plane.
    goal, residual = project_interface(points, surface, reference, target)
    total = goal - points[surface]
    current = points.copy()
    for k in range(increments):
        vol, _ = tetra_metrics(current[tetra])
        weights = None
        if stiffen:
            weights = (vol.mean() / np.maximum(vol, 1e-30 * vol.mean())) ** stiffen
        boundary = points[surface] + total * (k + 1) / increments
        disp = extend(current, tetra, boundary - current[surface], surface, size, weights=weights)
        current = current + disp
        current[surface] = boundary
    return current, residual


def _u(config, P):
    """Mode-normalised field u (solid where u < q) and threshold q at P."""
    f, q = field_and_threshold(config, P[:, 0], P[:, 1], P[:, 2])
    f = np.asarray(f, float)
    if config.mode == "sheet":
        u = np.abs(f)
    elif config.mode == "solid_above":
        u = -f
    else:
        u = f
    return u, np.broadcast_to(q, u.shape)


def level_transport(points, reference, target, *, iterations=30, u_range=None, max_move=None):
    """Move every vertex to the target iso-value of its remapped reference value.

    Reference value u0 with threshold q0 maps piecewise-linearly so that the
    extremes of u stay fixed and q0 -> q(y): layers between the extremum and
    the surface are stretched proportionally (constant layer count across walls).
    """
    size = np.asarray(target.size, float)
    P = np.asarray(points, float).copy()
    fixed = fixed_mask(P, size)
    u0, q0 = _u(reference, P)
    lo, hi = u_range if u_range is not None else (u0.min(), u0.max())
    lo = min(lo, u0.min())
    hi = max(hi, u0.max())
    below = u0 <= q0
    frac = np.where(below, (u0 - lo) / np.maximum(q0 - lo, 1e-30), (u0 - q0) / np.maximum(hi - q0, 1e-30))
    step = 1e-6 * float(size.max())

    def residual(Y):
        u, q = _u(target, Y)
        goal = np.where(below, lo + frac * (q - lo), q + frac * (hi - q))
        return u - goal

    phi = residual(P)
    start = P.copy()
    for _ in range(iterations):
        g = np.empty_like(P)
        for c in range(3):
            e = np.zeros(3)
            e[c] = step
            g[:, c] = (residual(P + e) - residual(P - e)) / (2 * step)
        g[fixed] = 0.0
        n2 = np.einsum("ij,ij->i", g, g)
        ok = n2 > 1e-12
        delta = np.zeros_like(P)
        delta[ok] = (phi[ok] / n2[ok])[:, None] * g[ok]
        t = np.ones(len(P))
        for _ in range(8):
            trial = np.clip(P - t[:, None] * delta, 0.0, size)
            if max_move is not None:
                d = trial - start
                n = np.linalg.norm(d, axis=1)
                over = n > max_move
                trial[over] = start[over] + d[over] * (max_move / n[over])[:, None]
            new = residual(trial)
            worse = np.abs(new) > np.abs(phi) + 1e-15
            if not worse.any():
                break
            t[worse] *= 0.5
        P, phi = trial, new
        if np.abs(phi).max() < 1e-12:
            break
    return P, np.abs(phi)


def _interpolated(reference, target, s):
    """Design halfway (fraction s) between a uniform reference and the target."""
    mid = reference.density_start
    data = {**target.to_dict(),
            "density_start": mid + (target.density_start - mid) * s,
            "density_end": mid + (target.density_end - mid) * s}
    if s == 0.0:
        data["gradient"] = "uniform"
    return Config(**data)


def _retarget(model, points, config):
    out = dict(model)
    out["points"] = points
    out["report"] = dict(model["report"])
    out["report"]["config"] = config.to_dict()
    return out


def transport_adaptive(model, reference, target, *, guard=0.08, first_step=1.0, min_step=1 / 64,
                       repair=None, progress=None):
    """Homotopy transport s: 0 -> 1 with local repair whenever a step would drop below ``guard``.

    Returns (model on target, log).  ``repair`` defaults to one optimize_mesh round.
    """
    from tpmslab import optimize_mesh

    repair = repair or (lambda m: optimize_mesh(m, rounds=1))
    log = []
    s, ds = 0.0, first_step
    current_cfg = reference
    m = model
    repaired_here = False
    while s < 1.0 - 1e-12:
        step = min(ds, 1.0 - s)
        nxt = _interpolated(reference, target, s + step)
        surface = np.unique(np.asarray(m["interface"]))
        goal, residual = project_interface(m["points"], surface, current_cfg, nxt, iterations=30)
        vol, _ = tetra_metrics(m["points"][m["tetra"]])
        weights = vol.mean() / np.maximum(vol, 1e-30 * vol.mean())
        disp = extend(m["points"], m["tetra"], goal - m["points"][surface], surface, target.size, weights=weights)
        P = m["points"] + disp
        P[surface] = goal
        v, q = tetra_metrics(P[m["tetra"]])
        # Vertex-local backtracking: vertices of inverted cells lag behind
        # (surface vertices then sit slightly off the target level set).
        for _ in range(10):
            inverted = v <= 0
            if not inverted.any():
                break
            lag = np.unique(m["tetra"][inverted])
            P[lag] = m["points"][lag] + 0.5 * (P[lag] - m["points"][lag])
            v, q = tetra_metrics(P[m["tetra"]])
        _, q_now = tetra_metrics(m["points"][m["tetra"]])
        floor = min(guard, 0.5 * float(q_now.min()))
        worst = float(q.min()) if (v > 0).all() else 0.0
        if worst >= floor or (step <= min_step and worst > 0):
            m = _retarget(m, P, nxt)
            current_cfg = nxt
            s += step
            log.append(("step", round(s, 4), round(worst, 4)))
            ds = min(1.0, step * 2)
            repaired_here = False
        elif not repaired_here:
            m2, _, ops = collapse_pass(m, P, target.size, floor)
            if ops:
                m = m2
                log.append(("collapse", round(s, 4), ops))
            else:
                m = repair(_retarget(m, m["points"], current_cfg))
                log.append(("repair", round(s, 4)))
            repaired_here = True
        elif step > min_step:
            ds = step / 2
            repaired_here = False
        else:
            raise RuntimeError(f"transport stalled at s={s:.4f}")
        if progress:
            progress(log[-1])
    return _retarget(m, m["points"], target), log


def _plane_bits(points, size):
    size = np.asarray(size, float)
    tol = 1e-9 * size
    low = np.abs(points) < tol
    high = np.abs(points - size) < tol
    return (low.astype(np.int64) * np.array([1, 4, 16])).sum(1) + (high.astype(np.int64) * np.array([2, 8, 32])).sum(1)


def rebuild(model, points, tetra, domains, phases, size):
    """Model dict (boundary ids 2..7 solid, 12..17 fluid, 8 interface) from cells."""
    from tpmslab.optimize import _rebuild_boundary
    from tpmslab.volume import TET_FACES

    used = np.unique(tetra)
    remap = -np.ones(len(points), np.int64)
    remap[used] = np.arange(len(used))
    points, tetra = points[used], remap[tetra]
    bits = _plane_bits(points, size)
    faces = tetra[:, TET_FACES].reshape(-1, 3)
    owner = np.repeat(np.arange(len(tetra)), 4)
    common = bits[faces[:, 0]] & bits[faces[:, 1]] & bits[faces[:, 2]]
    exterior = {}
    for f, c, o in zip(faces[common > 0], common[common > 0], owner[common > 0]):
        plane = int(c).bit_length() - 1
        exterior[tuple(sorted(map(int, f)))] = 2 + plane + (10 if phases[o] == 2 else 0)
    arrays = _rebuild_boundary(tetra.astype(np.int32), phases, exterior)
    out = {k: v for k, v in model.items() if k not in ("surface", "fluid_surface")}
    out.update(arrays, points=points, tetra=tetra.astype(np.int32), domains=domains, phase_ids=phases)
    return out


def collapse_pass(model, trial, size, guard, max_ops=200):
    """Collapse non-interface vertices of cells that drop below ``guard`` at ``trial`` positions.

    a -> b is accepted when b lies on every box plane of a, a is not on the
    interface, and every surviving cell of a's star is positive at both the
    current and the trial positions with trial quality above the old star minimum.
    Returns (model, trial points, operations).
    """
    P0 = np.asarray(model["points"], float)
    T = np.asarray(model["tetra"], np.int64).copy()
    dom = np.asarray(model["domains"]).copy()
    ph = np.asarray(model["phase_ids"]).copy()
    surface = np.zeros(len(P0), bool)
    surface[np.unique(model["interface"])] = True
    bits = _plane_bits(P0, size)
    alive = np.ones(len(T), bool)
    incident = [[] for _ in range(len(P0))]
    for c, t in enumerate(T):
        for v in t:
            incident[v].append(c)
    v, q = tetra_metrics(trial[T])
    bad = np.flatnonzero((v <= 0) | (q < guard))
    bad = bad[np.argsort(np.where(v[bad] <= 0, -1.0, q[bad]))]
    ops = 0
    for c in bad:
        if not alive[c] or ops >= max_ops:
            continue
        best = None
        for a in T[c]:
            if surface[a]:
                continue
            star = [k for k in incident[a] if alive[k]]
            old = tetra_metrics(trial[T[star]])[1]
            old_min = float(old.min()) if (tetra_metrics(trial[T[star]])[0] > 0).all() else -1.0
            nbrs = np.unique(T[star])
            for b in nbrs:
                if b == a or (bits[b] & bits[a]) != bits[a]:
                    continue
                keep = [k for k in star if b not in T[k]]
                new = T[keep].copy()
                new[new == a] = b
                v0, _ = tetra_metrics(P0[new])
                v1, q1 = tetra_metrics(trial[new])
                if len(new) == 0 or (v0 <= 0).any() or (v1 <= 0).any():
                    continue
                score = float(q1.min())
                if score > old_min and (best is None or score > best[0]):
                    best = (score, a, b, star, keep, new)
        if best is None:
            continue
        _, a, b, star, keep, new = best
        for k in star:
            alive[k] = False
        for k, t in zip(keep, new):
            T[k] = t
            alive[k] = True
            if k not in incident[b]:
                incident[b].append(k)
        incident[a] = []
        ops += 1
    if ops == 0:
        return model, trial, 0
    used = np.unique(T[alive])
    out = rebuild(model, P0, T[alive], dom[alive], ph[alive], size)
    return out, trial[used], ops
