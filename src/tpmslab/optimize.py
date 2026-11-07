"""Local cut-layer optimisation with per-phase geometry and quality budgets.

The direct cut-tetrahedral generator keeps every polygon face of a clipped
background cell in a canonical triangulation so that neighbouring cells
conform. That constraint leaves little freedom inside each cell and produces
a tail of poorly shaped tetrahedra next to the surface. This module improves
that tail afterwards with the classic local reconnection operations used by
Delaunay-based meshers:

* 2-3 flip: an interior face shared by two tetrahedra is replaced by the edge
  joining their apexes (three tetrahedra);
* 3-2 flip: an interior edge with exactly three incident tetrahedra is
  replaced by the face joining the three ring vertices (two tetrahedra);
* 4-4 flip: an interior edge with four incident tetrahedra is reconnected
  across the other diagonal of its ring (four tetrahedra).

Reconnections improve the worst affected mean-ratio quality and preserve
positive orientation. Optional box-face retriangulation and implicit-surface
moves retain conforming boundaries; interface cap flips can transfer cells
between phases. Each round is checked against per-phase quality and volume
budgets, with rollback to the previous mesh on failure. The result is audited
by :func:`tpmslab.conditioning.quality_audit` before it is returned.
"""

from __future__ import annotations

import copy
import hashlib
import time

import numpy as np
import trimesh

from . import batch_smoothing as bs
from .model import Config
from .quality import _tet_xyz, signed_volumes, tet_scalar, tetra_metrics, unique_rows
from .reporting import refresh_geometry_statistics
from .surface_smoothing import _chord_error, _level_set, surface_distances, surface_residuals, surface_smooth
from .volume import boundary_digest

_LOCAL_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
_LOCAL_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_METRIC_CACHE_LIMIT = 65536


def _signed(points, cells):
    return signed_volumes(points[np.asarray(cells)])


def _quality(points, cells):
    return tetra_metrics(points[np.asarray(cells)])[1]


def _oriented(points, tet):
    """Return ``tet`` reordered to positive orientation, or None if degenerate."""
    v = _signed(points, [tet])[0]
    if v > 0:
        return tuple(tet)
    if v < 0:
        return (tet[1], tet[0], tet[2], tet[3])
    return None


class _Mesh:
    """Dictionary-based incidence for local reconnection."""

    def __init__(self, points, tetra, labels, protected_faces):
        self.points = points  # also refreshes the scalar coordinate list self.P
        self.cells = {}          # id -> (a, b, c, d) positively oriented
        self.label = {}          # id -> domain label
        self.faces = {}          # sorted face -> set(cell ids)
        self.edges = {}          # sorted edge -> set(cell ids)
        self.protected = protected_faces
        self.next_id = 0
        self._bulk_add(tetra, labels)
        self.flips = {"2-3": 0, "3-2": 0, "4-4": 0, "edge_removal_5+": 0}
        self.exterior_id = {}    # sorted exterior face -> boundary id (box faces 2-7 / 12-17, surface 8)

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        self._points = value
        self.P = np.asarray(value, dtype=float).tolist()
        # Reconnection repeatedly scores the same cells at fixed coordinates.
        # A smoothing step refreshes points and invalidates every cached metric.
        self._metric_cache = {}

    def _metric(self, tet):
        key = tuple(tet)
        value = self._metric_cache.get(key)
        if value is None:
            value = tet_scalar(self.P, key)
            if len(self._metric_cache) < _METRIC_CACHE_LIMIT:
                self._metric_cache[key] = value
        return value

    def _bulk_add(self, tetra, labels):
        """Same incidence as calling ``_add`` per cell in order, built with array grouping.

        Owner sets receive their cell ids in increasing order, exactly as the
        sequential insertion did, so set iteration (and every later decision)
        is unchanged.
        """
        tetra = np.asarray(tetra, dtype=np.int64).reshape(-1, 4)
        n = len(tetra)
        rows = tetra.tolist()
        self.cells.update(zip(range(n), map(tuple, rows)))
        self.label.update(zip(range(n), map(int, np.asarray(labels).tolist())))
        self.next_id = n
        if n == 0:
            return
        for table, local in ((self.faces, _LOCAL_FACES), (self.edges, _LOCAL_EDGES)):
            keys = np.sort(tetra[:, np.array(local)], axis=2).reshape(n * len(local), -1)
            owner = np.repeat(np.arange(n), len(local))
            order = np.lexsort(tuple(keys[:, j] for j in range(keys.shape[1] - 1, -1, -1)) + ())
            sorted_keys = keys[order]
            change = np.r_[True, np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)]
            starts = np.flatnonzero(change)
            ends = np.r_[starts[1:], len(order)]
            owners_sorted = owner[order]
            # Stable lexsort keeps owners of one key in increasing cell id.
            unique_keys = list(map(tuple, sorted_keys[starts].tolist()))
            owner_list = owners_sorted.tolist()
            for key, a, b in zip(unique_keys, starts.tolist(), ends.tolist()):
                table[key] = set(owner_list[a:b])

    def _add(self, tet, label):
        cid = self.next_id
        self.next_id += 1
        self.cells[cid] = tet
        self.label[cid] = label
        for f in _LOCAL_FACES:
            self.faces.setdefault(tuple(sorted((tet[f[0]], tet[f[1]], tet[f[2]]))), set()).add(cid)
        for e in _LOCAL_EDGES:
            self.edges.setdefault(tuple(sorted((tet[e[0]], tet[e[1]]))), set()).add(cid)
        return cid

    def _remove(self, cid):
        tet = self.cells.pop(cid)
        self.label.pop(cid)
        for f in _LOCAL_FACES:
            key = tuple(sorted((tet[f[0]], tet[f[1]], tet[f[2]])))
            owners = self.faces[key]
            owners.discard(cid)
            if not owners:
                del self.faces[key]
        for e in _LOCAL_EDGES:
            key = tuple(sorted((tet[e[0]], tet[e[1]])))
            owners = self.edges[key]
            owners.discard(cid)
            if not owners:
                del self.edges[key]

    def interior_face(self, key):
        return key not in self.protected and len(self.faces.get(key, ())) == 2

    def interior_edge(self, key):
        """An edge is interior when none of its incident faces is protected."""
        owners = self.edges.get(key, ())
        for cid in owners:
            tet = self.cells[cid]
            for f in _LOCAL_FACES:
                face = (tet[f[0]], tet[f[1]], tet[f[2]])
                if key[0] in face and key[1] in face and tuple(sorted(face)) in self.protected:
                    return False
        return True

    def _oriented(self, tet):
        v, q = self._metric(tet)
        if v > 0:
            return tuple(tet), q
        if v < 0:
            return (tet[1], tet[0], tet[2], tet[3]), q
        return None, 0.0

    def _try_replace(self, old_ids, new_tets, kind, improvement, *, oriented=None):
        old_cells = [self.cells[c] for c in old_ids]
        old_volume = 0.0
        old_min = float("inf")
        for t in old_cells:
            v, q = self._metric(t)
            old_volume += v
            old_min = min(old_min, q)
        if oriented is None:
            oriented = []
            for t in new_tets:
                tet, _ = self._oriented(t)
                if tet is None:
                    return False
                oriented.append(tet)
        new_volume = 0.0
        new_min = float("inf")
        for t in oriented:
            v, q = self._metric(t)
            if v <= 0:
                return False
            new_volume += v
            new_min = min(new_min, q)
        if new_min <= old_min * improvement:
            return False
        if abs(old_volume - new_volume) > 1e-10 * abs(old_volume):
            return False
        labels = {self.label[c] for c in old_ids}
        if len(labels) != 1:
            return False
        label = labels.pop()
        # New faces must not already exist (would create a non-manifold face).
        old_faces = set()
        for t in old_cells:
            for f in _LOCAL_FACES:
                old_faces.add(tuple(sorted((t[f[0]], t[f[1]], t[f[2]]))))
        for t in oriented:
            for f in _LOCAL_FACES:
                face = tuple(sorted((t[f[0]], t[f[1]], t[f[2]])))
                if face not in old_faces and face in self.faces:
                    return False
        for cid in old_ids:
            self._remove(cid)
        for t in oriented:
            self._add(t, label)
        self.flips[kind] = self.flips.get(kind, 0) + 1
        return True

    def flip_2_3(self, face, improvement):
        owners = self.faces.get(face)
        if owners is None or len(owners) != 2 or face in self.protected:
            return False
        c1, c2 = owners
        a, b, c = face
        d = next(v for v in self.cells[c1] if v not in face)
        e = next(v for v in self.cells[c2] if v not in face)
        if tuple(sorted((d, e))) in self.edges:
            return False
        new = [(d, e, a, b), (d, e, b, c), (d, e, c, a)]
        return self._try_replace([c1, c2], new, "2-3", improvement)

    def _ring(self, edge):
        """Ordered ring vertices around an interior edge, or None."""
        owners = self.edges.get(edge)
        if not owners or not self.interior_edge(edge):
            return None
        d, e = edge
        pairs = []
        for cid in owners:
            others = [v for v in self.cells[cid] if v not in edge]
            if len(others) != 2:
                return None
            pairs.append(others)
        # Walk the ring through shared vertices.
        ring = [pairs[0][0], pairs[0][1]]
        used = {0}
        while len(used) < len(pairs):
            tail = ring[-1]
            found = None
            for i, (u, v) in enumerate(pairs):
                if i in used:
                    continue
                if u == tail:
                    found, nxt = i, v
                elif v == tail:
                    found, nxt = i, u
                if found is not None:
                    break
            if found is None:
                return None
            used.add(found)
            ring.append(nxt)
        if ring[-1] != ring[0]:
            return None
        return ring[:-1]

    def _ring_triangulation(self, ring, d, e, closed):
        """Best maximin triangulation of the polygon ``ring`` (closed by the
        edge ring[-1]-ring[0]) with apexes d and e, as oriented tetrahedra; or
        None. New edges must not already exist in the mesh."""
        n = len(ring)
        best = None
        raw = {}
        # Same scalar arithmetic as _metric/tet_scalar, without per-call
        # cache and tuple overhead (these candidate cells are rarely reused).
        P = self.P
        xyz = [P[v] for v in ring]
        pd, pe = P[d], P[e]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = xyz[i], xyz[j], xyz[k]
                    vd, qd = _tet_xyz(a, b, c, pd)
                    ve, qe = _tet_xyz(a, b, c, pe)
                    raw[i, j, k] = (vd, ve, min(qd, qe))
        for sign in (1.0, -1.0):
            # Quality of triangle (i, j, k): worst of its two tetrahedra, or -1 if inverted.
            tri = {key: (q if vd * sign > 0 and ve * sign < 0 else -1.0) for key, (vd, ve, q) in raw.items()}
            table, choice = {}, {}
            for i in range(n - 1):
                table[i, i + 1] = float("inf")
            for length in range(2, n):
                for i in range(n - length):
                    j = i + length
                    value, arg = -1.0, None
                    for k in range(i + 1, j):
                        candidate = min(table[i, k], table[k, j], tri[i, k, j])
                        if candidate > value:
                            value, arg = candidate, k
                    table[i, j], choice[i, j] = value, arg
            if table[0, n - 1] > 0 and (best is None or table[0, n - 1] > best[0]):
                best = (table[0, n - 1], sign, dict(choice))
        if best is None:
            return None
        _, sign, choice = best
        triangles, stack = [], [(0, n - 1)]
        while stack:
            i, j = stack.pop()
            if j - i < 2:
                continue
            k = choice[i, j]
            triangles.append((ring[i], ring[k], ring[j]))
            stack += [(i, k), (k, j)]
        existing = {tuple(sorted((ring[i], ring[i + 1]))) for i in range(n - 1)}
        if closed:
            existing.add(tuple(sorted((ring[-1], ring[0]))))
        for a, b, c in triangles:
            for u, v in ((a, b), (b, c), (a, c)):
                pair = tuple(sorted((u, v)))
                if pair in self.edges and pair not in existing:
                    return None
        oriented = []
        for a, b, c in triangles:
            oriented.append((a, b, c, d) if sign > 0 else (b, a, c, d))
            oriented.append((b, a, c, e) if sign > 0 else (a, b, c, e))
        return oriented

    def remove_edge(self, edge, improvement, max_ring=7):
        """Shewchuk-style edge removal: replace the n tetrahedra around an
        interior edge (d, e) by 2(n-2) tetrahedra built on the best
        triangulation of its ring polygon (maximin quality, O(n^3) dynamic
        programme). n=3 is the 3-2 flip and n=4 the 4-4 flip."""
        owners = self.edges.get(edge)
        if not owners or not 3 <= len(owners) <= max_ring:
            return False
        ring = self._ring(edge)
        if ring is None or len(ring) != len(owners):
            return False
        oriented = self._ring_triangulation(ring, edge[0], edge[1], closed=True)
        if oriented is None:
            return False
        kind = {3: "3-2", 4: "4-4"}.get(len(ring), "edge_removal_5+")
        return self._try_replace(list(owners), None, kind, improvement, oriented=oriented)

    def remove_boundary_edge(self, edge, improvement, max_ring=7):
        """Edge removal for an edge lying in a flat box face: its two box-face
        triangles (a,b,c) and (a,b,d) are coplanar, so replacing them by
        (a,c,d)+(b,c,d) and re-triangulating the open ring c..d leaves the
        region, every box face and every phase volume exactly unchanged."""
        owners = self.edges.get(edge)
        if not owners or not 2 <= len(owners) <= max_ring:
            return False
        boundary = [f for f in self._faces_of_edge(edge) if f in self.protected]
        if len(boundary) != 2 or any(f not in self.exterior_id for f in boundary):
            return False
        f1, f2 = boundary
        plane = self.exterior_id[f1]
        if self.exterior_id[f2] != plane or plane == 8 or plane == 18:
            return False
        if plane in getattr(self, "frozen_planes", ()):
            return False
        c = (set(f1) - set(edge)).pop()
        d = (set(f2) - set(edge)).pop()
        # Open ring from c to d through the third vertices of the owners.
        links = {}
        for cid in owners:
            u, v = [x for x in self.cells[cid] if x not in edge]
            links.setdefault(u, []).append(v)
            links.setdefault(v, []).append(u)
        if len(links.get(c, ())) != 1 or len(links.get(d, ())) != 1:
            return False
        ring, previous = [c], None
        while ring[-1] != d:
            nxt = [x for x in links[ring[-1]] if x != previous]
            if len(nxt) != 1:
                return False
            previous = ring[-1]
            ring.append(nxt[0])
            if len(ring) > len(owners) + 1:
                return False
        if len(ring) != len(owners) + 1:
            return False
        oriented = self._ring_triangulation(ring, edge[0], edge[1], closed=False)
        if oriented is None:
            return False
        if not self._try_replace(list(owners), None, "boundary_edge", improvement, oriented=oriented):
            return False
        new1 = tuple(sorted((edge[0], c, d)))
        new2 = tuple(sorted((edge[1], c, d)))
        self.protected.difference_update((f1, f2))
        self.protected.update((new1, new2))
        for f in (f1, f2):
            self.exterior_id.pop(f, None)
        self.exterior_id[new1] = self.exterior_id[new2] = plane
        return True

    def _faces_of_edge(self, edge):
        out = set()
        for cid in self.edges.get(edge, ()):
            tet = self.cells[cid]
            for f in _LOCAL_FACES:
                face = (tet[f[0]], tet[f[1]], tet[f[2]])
                if edge[0] in face and edge[1] in face:
                    out.add(tuple(sorted(face)))
        return out

    def cap_flip(self, cid, improvement, chord=None, max_ring=7):
        """Remove an interface cap: a cell whose four vertices lie on the
        phase interface and two of whose faces, (a,b,c) and (a,b,d), are
        interface faces. The cell is handed to the neighbouring phase, which
        turns the interface into (a,c,d)+(b,c,d) -- the other triangulation of
        the same four surface points -- and the now-interior edge (a,b) is
        removed. Kept only if that edge removal improves the worst quality and
        the new interface triangles do not depart further from the level set;
        otherwise everything is restored."""
        tet = self.cells.get(cid)
        if tet is None:
            return False
        faces = [tuple(sorted((tet[f[0]], tet[f[1]], tet[f[2]]))) for f in _LOCAL_FACES]
        interface = [f for f in faces if f in self.protected and len(self.faces.get(f, ())) == 2]
        if len(interface) != 2:
            return False
        if any(f in self.protected or len(self.faces.get(f, ())) != 2 for f in faces if f not in interface):
            return False
        f1, f2 = interface
        ab = tuple(sorted(set(f1) & set(f2)))
        if len(ab) != 2:
            return False
        c = (set(f1) - set(ab)).pop()
        d = (set(f2) - set(ab)).pop()
        u1 = next(iter(self.faces[f1] - {cid}))
        u2 = next(iter(self.faces[f2] - {cid}))
        own, other = self.label[cid], self.label[u1]
        if self.label[u2] != other or other == own:
            return False
        cd = tuple(sorted((c, d)))
        if any(f in self.protected for f in self._faces_of_edge(cd)):
            return False
        if {f for f in self._faces_of_edge(ab) if f in self.protected} != {f1, f2}:
            return False
        new1 = tuple(sorted((ab[0], c, d)))
        new2 = tuple(sorted((ab[1], c, d)))
        if chord is not None and chord([new1, new2]) > chord([f1, f2]) * (1 + 1e-6) + 1e-15:
            return False
        self.label[cid] = other
        self.protected.difference_update((f1, f2))
        self.protected.update((new1, new2))
        if self.remove_edge(ab, improvement, max_ring=max_ring):
            self.flips["cap"] = self.flips.get("cap", 0) + 1
            return True
        self.label[cid] = own
        self.protected.difference_update((new1, new2))
        self.protected.update((f1, f2))
        return False

    @staticmethod
    def _ring_edges(ring):
        n = len(ring)
        return {tuple(sorted((ring[i], ring[(i + 1) % n]))) for i in range(n)}

    # Backwards-compatible names.
    def flip_3_2(self, edge, improvement):
        return len(self.edges.get(edge, ())) == 3 and self.remove_edge(edge, improvement, max_ring=3)

    def flip_4_4(self, edge, improvement):
        return len(self.edges.get(edge, ())) == 4 and self.remove_edge(edge, improvement, max_ring=4)


def _rebuild_boundary(tetra, phases, exterior_id):
    """Re-derive exterior and interface faces from cell incidence after
    boundary-changing operations. Faces are oriented outward from their owning
    cell (the solid cell for interface faces), as in the generators; exterior
    ids come from the operations' bookkeeping (``_Mesh.exterior_id``)."""
    from .volume import TET_FACES

    faces = tetra[:, TET_FACES].reshape(-1, 3)
    keys = np.sort(faces, axis=1)
    _, inverse, counts = unique_rows(keys, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    single = order[starts[counts == 1]]
    exterior = faces[single]
    exterior_ids = np.array([exterior_id[tuple(sorted(map(int, f)))] for f in exterior], dtype=np.int32)
    out = {}
    if phases is None:
        out.update(boundary=exterior.astype(np.int32), boundary_ids=exterior_ids)
        return out
    pairs = starts[counts == 2]
    a, b = order[pairs], order[pairs + 1]
    cross = phases[a // 4] != phases[b // 4]
    interface = faces[np.where(phases[a[cross] // 4] == 1, a[cross], b[cross])]
    out.update(boundary=np.concatenate([exterior, interface]).astype(np.int32),
               boundary_ids=np.r_[exterior_ids, np.full(len(interface), 8)].astype(np.int32),
               interface=interface.astype(np.int32),
               solid_boundary=np.concatenate([exterior[exterior_ids < 10], interface]).astype(np.int32),
               fluid_boundary=np.concatenate([exterior[exterior_ids >= 10], interface[:, [0, 2, 1]]]).astype(np.int32))
    return out


def interior_smooth(points, tetra, locked, *, quality_threshold=0.5, sweeps=2,
                    step_fractions=(1.0, 0.5, 0.25, 0.125), adjacency=None):
    """Maximin smoothing of free vertices on independent batches.

    Each vertex tries two directions (neighbour centroid and the point that
    makes its worst incident cell regular) at ``step_fractions`` and keeps the
    best trial if no incident cell inverts and its worst incident quality
    strictly improves. Vertices whose stars share no cell are moved together
    (see :mod:`tpmslab.batch_smoothing`). ``adjacency`` may pass precomputed
    ``(cell_ids, cell_starts, nb_ids, nb_starts)`` for this ``tetra``.
    """
    from . import batch_smoothing as bs

    points = np.array(points, dtype=float)
    tetra = np.asarray(tetra)
    if adjacency is None:
        adjacency = (*bs.incidence(tetra, len(points)), *bs.neighbours(tetra, len(points)))
    cell_ids, cell_starts, nb_ids, nb_starts = adjacency
    fractions = np.asarray(step_fractions, dtype=float)
    accepted = 0
    for _ in range(sweeps):
        _, quality = bs.volume_quality(points[tetra])
        severity = np.full(len(points), np.inf)
        np.minimum.at(severity, tetra.ravel(), np.repeat(quality, 4))
        candidates = np.flatnonzero(~locked & (severity < quality_threshold))
        candidates = candidates[np.argsort(severity[candidates], kind="stable")]
        moved = 0
        for batch in bs.independent_batches(candidates, cell_ids, cell_starts):
            for part in bs.chunks(batch):
                vertices = batch[part]
                cells, mask, current, worst = bs.star_state(points, tetra, vertices, cell_ids, cell_starts)
                x0 = points[vertices]
                directions = np.stack((bs.neighbour_means(points, vertices, nb_ids, nb_starts) - x0,
                                       bs.ideal_points(points, cells, worst, vertices) - x0), axis=1)
                trials = x0[:, None, None, :] + fractions[None, None, :, None] * directions[:, :, None, :]
                trials = trials.reshape(len(vertices), -1, 3)
                vmin, qmin = bs.score_trials(points, cells, mask, vertices, trials)
                pick = bs.choose(vmin, qmin, current)
                take = pick >= 0
                points[vertices[take]] = trials[np.flatnonzero(take), pick[take]]
                moved += int(take.sum())
        accepted += moved
        if moved == 0:
            break
    return points, accepted


def _phase_quality(volumes, quality, phases):
    """Per-phase geometry/quality gate data, independent of cell count changes."""
    result = {}
    for phase in np.unique(phases):
        mask = phases == phase
        v, q = volumes[mask], quality[mask]
        result[int(phase)] = {"volume": float(v.sum()), "minimum": float(q.min()),
                              "p01": float(np.quantile(q, .01)), "median": float(np.median(q)),
                              "poor_volume_fraction": float(v[q < .01].sum() / v.sum())}
    return result


def optimize_mesh(model, *, quality_threshold=0.5, rounds=6, improvement=1.0 + 1e-9,
                  smooth=True, smooth_nodes=20000, smooth_sweeps=2, max_median_loss=0.15,
                  surface=True, surface_sweeps=2, min_round_gain=0.01, max_ring=7, focus_quantile=0.05,
                  chord_tolerance=0.0, chord_select=True, chord_gain_floor=0.1, cap_flips=True, boundary_flips=True,
                  max_phase_volume_shift_pp=0.3, effort="standard", surface_envelope=None,
                  progress=None):
    """Return a new model whose cut-layer tetrahedra were reconnected and smoothed.

    Domain labels are preserved; box faces may be re-triangulated. Interface
    cap flips transfer cells between phases when ``surface=True``.
    ``quality_threshold`` selects the cells whose faces and edges
    are tried; ``rounds`` alternates reconnections, interior smoothing and,
    with ``surface=True``, sliding of boundary vertices along the exact
    implicit surface and the box faces (see :mod:`tpmslab.surface_smoothing`).
    ``smooth`` is ``True`` (local maximin Laplacian steps), ``"slsqp"``
    (:func:`condition_mesh`, slower) or ``False``.
    With ``surface=False`` every boundary vertex and every phase volume is
    unchanged. The returned ``report["optimization"]`` records what happened;
    ``quality_audit`` runs on the result.

    Each round must preserve per-phase minimum/p01/poor-volume quality and
    respect ``max_median_loss`` relative to the input. Phase volume drift is
    limited to ``max_phase_volume_shift_pp`` percentage points of box volume
    (default 0.3, configurable; None disables this budget). A failed round is
    discarded and the last accepted mesh is returned. ``chord_tolerance=0``
    additionally rejects surface moves that increase the sampled chord error;
    explicit None uses chord error only for ranking. These are user-adjustable
    geometry/quality budgets, not solver-accuracy guarantees.
    Mixed-phase surface vertices additionally preserve the worst incident
    quality of each phase at every accepted move.
    ``effort="high"`` raises ``quality_threshold`` to at least 0.7 (about 3-4x
    the time; on BCC meshes solid p01 rises from about 0.52 to 0.57-0.66).
    ``surface_envelope`` (mm, default None = exact surface): interface vertices
    may leave the implicit surface by at most this first-order distance when
    that improves the local worst quality (fTetWild-style envelope). The
    maximum deviation is reported in ``optimization["surface_deviation_max_mm"]``.
    """
    from .conditioning import _validate, condition_mesh, quality_audit

    if surface_envelope is not None and (not np.isfinite(surface_envelope) or surface_envelope <= 0):
        raise ValueError("surface_envelope must be a positive length in mm or None")
    if effort not in ("standard", "high"):
        raise ValueError("effort must be 'standard' or 'high'")
    if effort == "high":
        # Work on every cell below 0.7 instead of 0.5: on BCC meshes the
        # standard threshold caps the solid p01 near 0.52 (dev10 study).
        quality_threshold = max(quality_threshold, 0.7)

    if not np.isfinite(quality_threshold) or not 0 < quality_threshold <= 1:
        raise ValueError("quality_threshold must lie in (0, 1]")
    if type(rounds) is not int or not 1 <= rounds <= 50:
        raise ValueError("rounds must be an integer from 1 to 50")
    if not np.isfinite(min_round_gain) or min_round_gain < 0:
        raise ValueError("min_round_gain must be a non-negative number")
    if not np.isfinite(max_median_loss) or not 0 <= max_median_loss < 1:
        raise ValueError("max_median_loss must lie in [0, 1)")
    if max_phase_volume_shift_pp is not None and (
            not np.isfinite(max_phase_volume_shift_pp) or max_phase_volume_shift_pp < 0):
        raise ValueError("max_phase_volume_shift_pp must be non-negative or None")
    if chord_tolerance is not None and (not np.isfinite(chord_tolerance) or chord_tolerance < 0):
        raise ValueError("chord_tolerance must be non-negative or None")
    progress = progress or (lambda message: None)
    started = time.monotonic()
    points, tetra, domains, phases, volumes, locked = _validate(model)
    points = points.copy()
    protected = {tuple(sorted(map(int, f))) for f in np.asarray(model["boundary"])}
    if "interface" in model:
        protected |= {tuple(sorted(map(int, f))) for f in np.asarray(model["interface"])}
    phase_of_domain = {int(d): int(phases[domains == d][0]) for d in np.unique(domains)}
    _, before = tetra_metrics(points[tetra])
    initial_quality = _phase_quality(volumes, before, phases)
    retained_quality = initial_quality
    config = Config.from_dict(model["report"]["config"])
    two_phase = "phase_ids" in model and "interface" in model
    sides = 2 if config.mode == "sheet" else 1
    probe = max(config.size) * 1e-6

    arrays = {key: np.asarray(model[key]) for key in ("boundary", "boundary_ids", "interface",
                                                    "solid_boundary", "fluid_boundary") if key in model}

    def chord(faces):
        """Sum of area x distance-to-level-set over interface triangles (see surface_smoothing)."""
        tri = mesh.points[np.asarray(faces)]
        values = np.stack([np.abs(_level_set(config, tri[:, 0], s_)) for s_ in range(sides)])
        # Both triangle pairs compared by cap_flip span the same four surface
        # points, so a common |grad g| normalisation cancels: compare area*|g|.
        return float(_chord_error(tri, values.argmin(axis=0), config, probe, 1.0).sum())
    history = []
    surface_moves = 0
    moved_surface_vertices = set()
    mesh = _Mesh(points, tetra, domains, protected)
    periodic = []
    frozen = None
    if config.periodic_faces:
        from .periodic import periodic_axes

        periodic = periodic_axes(config)
        mesh.frozen_planes = {pid for a in periodic for pid in (2 + 2 * a, 3 + 2 * a, 12 + 2 * a, 13 + 2 * a)}
        size_ = np.asarray(config.size, dtype=float)
        tol_ = float(size_.max()) * 1e-9
        frozen = np.zeros(len(points), dtype=bool)
        for a in periodic:
            frozen |= (np.abs(points[:, a]) < tol_) | (np.abs(points[:, a] - size_[a]) < tol_)
    for face, pid in zip(arrays["boundary"], arrays["boundary_ids"]):
        if not (two_phase and int(pid) == 8):
            mesh.exterior_id[tuple(sorted(map(int, face)))] = int(pid)
    current_points = points
    rollback_reasons = []
    for round_index in range(rounds):
        # Arrays are cheap checkpoints; rebuild dictionary incidence only on rollback.
        checkpoint = (current_points.copy(), np.array(list(mesh.cells.values()), dtype=np.int32),
                      np.array(list(mesh.label.values()), dtype=np.int32), mesh.protected.copy(),
                      mesh.exterior_id.copy(), mesh.flips.copy(), surface_moves,
                      moved_surface_vertices.copy())
        # Candidate faces and edges from the worst cells.
        ids = np.fromiter(mesh.cells.keys(), dtype=np.int64, count=len(mesh.cells))
        _, quality = bs.volume_quality(mesh.points[np.array(list(mesh.cells.values()), dtype=np.int64)])
        # Work on the tail that decides the minimum and p01: cells below the
        # focus quantile (never above quality_threshold).
        threshold = quality_threshold
        if focus_quantile is not None:
            threshold = min(quality_threshold, float(np.quantile(quality, focus_quantile)) * 1.0001)
        selected = quality < threshold
        bad = ids[selected][np.argsort(quality[selected], kind="stable")]  # worst first
        attempted = accepted = 0
        tried = set()  # edges/faces already tried unsuccessfully in this round
        for cid in bad:
            if cid not in mesh.cells:
                continue
            tet = mesh.cells[cid]
            done = False
            for e in _LOCAL_EDGES:
                key = tuple(sorted((tet[e[0]], tet[e[1]])))
                if key in tried:
                    continue
                attempted += 1
                if (mesh.remove_edge(key, improvement, max_ring=max_ring)
                        or (boundary_flips and mesh.remove_boundary_edge(key, improvement, max_ring=max_ring))):
                    accepted += 1
                    done = True
                    break
                tried.add(key)
            if done:
                continue
            for f in _LOCAL_FACES:
                key = tuple(sorted((tet[f[0]], tet[f[1]], tet[f[2]])))
                if key in tried:
                    continue
                attempted += 1
                if mesh.flip_2_3(key, improvement):
                    accepted += 1
                    break
                tried.add(key)
        caps = 0
        if cap_flips and two_phase and surface:
            # Caps hand a sliver between phases, so they belong to the
            # geometry-adjusting mode (surface=True) only.
            for cid in bad:
                if cid in mesh.cells and mesh.cap_flip(cid, improvement, chord=chord, max_ring=max_ring):
                    caps += 1
            accepted += caps
        new_tetra = np.array(list(mesh.cells.values()), dtype=np.int64)
        new_domains = np.array(list(mesh.label.values()), dtype=np.int64)
        new_phases = np.array([phase_of_domain[int(d)] for d in new_domains], dtype=np.int32)
        entry = {"round": round_index + 1, "cap_flips": caps, "focus_threshold": threshold, "candidate_cells": int(len(bad)), "attempted_operations": attempted,
                 "accepted_operations": accepted, "flips": dict(mesh.flips)}
        if smooth == "slsqp":
            staged = dict(model, points=current_points, tetra=new_tetra, domains=new_domains, phase_ids=new_phases)
            staged.update(_rebuild_boundary(new_tetra, new_phases if two_phase else None, mesh.exterior_id))
            staged["report"] = copy.deepcopy(model["report"])
            smoothed = condition_mesh(staged, max_nodes=smooth_nodes, sweeps=smooth_sweeps,
                                      target_quality=min(1.0, 2 * quality_threshold),
                                      max_median_loss=max_median_loss)
            current_points = np.asarray(smoothed["points"], dtype=float)
            mesh.points = current_points
            entry["smoothing_retained_moves"] = smoothed["report"]["conditioning"]["retained_moves"]
        adjacency = ((*bs.incidence(new_tetra, len(current_points)),
                      *bs.neighbours(new_tetra, len(current_points))) if smooth or surface else None)
        if smooth and smooth != "slsqp":
            # Two-direction maximin smoothing (Laplacian + ideal point). The
            # SLSQP refinement of condition_mesh is available as smooth="slsqp";
            # in the 0.3.9.dev4 study it changed no p01 / minimum value.
            current_points, moves = interior_smooth(current_points, new_tetra, locked,
                                                    quality_threshold=threshold, sweeps=smooth_sweeps,
                                                    adjacency=adjacency)
            mesh.points = current_points
            entry["smoothing_retained_moves"] = int(moves)
        if surface:
            # Boundary faces straight from the mesh bookkeeping (orientation is
            # not needed here); the oriented arrays are rebuilt once at the end.
            exterior = list(mesh.exterior_id.items())
            interface = [f for f in mesh.protected if f not in mesh.exterior_id] if two_phase else []
            round_boundary = np.array([f for f, _ in exterior] + interface, dtype=np.int64).reshape(-1, 3)
            round_ids = np.array([i for _, i in exterior] + [8] * len(interface), dtype=np.int64)
            current_points, surface_info = surface_smooth(
                current_points, new_tetra, round_boundary, round_ids, config,
                quality_threshold=threshold, sweeps=surface_sweeps, chord_tolerance=chord_tolerance,
                chord_select=chord_select, chord_gain_floor=chord_gain_floor, adjacency=adjacency,
                phase_ids=new_phases if two_phase else None, frozen=frozen, envelope=surface_envelope)
            mesh.points = current_points
            if periodic:
                from .periodic import face_partners, pair_smooth

                tol_p = float(np.max(config.size)) * 1e-7
                pairs = [(a, *face_partners(current_points, a, np.asarray(config.size, float), tol_p)) for a in periodic]
                on_surface_mask = np.zeros(len(current_points), dtype=bool)
                if len(interface):
                    on_surface_mask[np.unique(np.asarray(interface))] = True
                current_points, pair_moves = pair_smooth(current_points, new_tetra, pairs, config,
                                                         quality_threshold=threshold, on_surface=on_surface_mask)
                mesh.points = current_points
                entry["periodic_pair_moves"] = int(pair_moves)
            entry["surface_moves"] = surface_info["accepted"]
            entry["surface_phase_guard_rejected_trials"] = surface_info.get("phase_guard_rejected_trials", 0)
            surface_moves += surface_info["accepted"]
            moved_surface_vertices.update(surface_info["moved_vertices"])
        round_volumes, q = tetra_metrics(current_points[new_tetra])
        round_phases = np.array([phase_of_domain[int(d)] for d in new_domains])
        candidate_quality = _phase_quality(round_volumes, q, round_phases)
        failures = []
        if not np.isfinite(current_points).all() or np.any(signed_volumes(current_points[new_tetra]) <= 0):
            failures.append("Nonfinite coordinates or nonpositive cells")
        for phase, previous in retained_quality.items():
            row = candidate_quality.get(phase)
            if row is None:
                failures.append(f"Phase {phase} disappeared")
                continue
            if (row["minimum"] < previous["minimum"] - 1e-12
                    or row["p01"] < previous["p01"] - 1e-12
                    or row["poor_volume_fraction"] > previous["poor_volume_fraction"] + 1e-12
                    or row["median"] < initial_quality[phase]["median"] * (1 - max_median_loss) - 1e-12):
                failures.append(f"Phase {phase} quality budget exceeded")
            shift = abs(row["volume"] - initial_quality[phase]["volume"]) / np.prod(config.size) * 100
            if max_phase_volume_shift_pp is not None and shift > max_phase_volume_shift_pp + 1e-10:
                failures.append(f"Phase {phase} volume shift {shift:.6g} pp exceeds {max_phase_volume_shift_pp:.6g} pp")
        entry.update(minimum=float(q.min()), p01=float(np.quantile(q, .01)), poor_below_01=int(np.sum(q < .1)))
        entry.update(retained=not failures, rejection_reasons=failures)
        history.append(entry)
        if failures:
            (current_points, kept_tetra, kept_domains, kept_protected, kept_exterior,
             kept_flips, surface_moves, moved_surface_vertices) = checkpoint
            frozen_planes = getattr(mesh, "frozen_planes", None)
            mesh = _Mesh(current_points, kept_tetra, kept_domains, kept_protected)
            mesh.exterior_id, mesh.flips = kept_exterior, kept_flips
            if frozen_planes is not None:
                mesh.frozen_planes = frozen_planes
            rollback_reasons = failures
            progress("Optimisation round rejected; retained the previous mesh: " + "; ".join(failures))
            break
        retained_quality = candidate_quality
        progress(f"Cut-layer optimisation round {round_index + 1}/{rounds}: {accepted} reconnections, "
                 f"{entry.get('smoothing_retained_moves', 0)} interior moves, {entry.get('surface_moves', 0)} surface moves, "
                 f"min quality {q.min():.3f}, p01 {np.quantile(q, .01):.3f}")
        previous_p01 = history[-2]["p01"] if len(history) > 1 else None
        stalled = previous_p01 is not None and entry["p01"] <= previous_p01 * (1 + min_round_gain)
        if (accepted == 0 and not entry.get("smoothing_retained_moves") and not entry.get("surface_moves")) or stalled:
            break
    new_tetra = np.array(list(mesh.cells.values()), dtype=np.int32)
    new_domains = np.array(list(mesh.label.values()), dtype=np.int32)
    new_phases = np.array([phase_of_domain[int(d)] for d in new_domains], dtype=np.int32)
    result = dict(model)
    result.pop("vertex_keys", None)
    result.pop("lattice_xyz", None)
    result.update(points=current_points, tetra=new_tetra, domains=new_domains)
    if mesh.flips.get("cap", 0) + mesh.flips.get("boundary_edge", 0):
        arrays.update(_rebuild_boundary(new_tetra, new_phases if two_phase else None, mesh.exterior_id))
    result.update({key: value for key, value in arrays.items()})
    if "phase_ids" in model:
        result["phase_ids"] = new_phases
    for key, faces in (("surface", arrays.get("solid_boundary", arrays["boundary"])),
                       ("fluid_surface", arrays.get("fluid_boundary"))):
        if faces is not None:
            surface = trimesh.Trimesh(current_points.copy(), np.asarray(faces).copy(), process=False)
            surface.remove_unreferenced_vertices()
            result[key] = surface
    report = copy.deepcopy(model["report"])
    # Solver evidence belongs to the input mesh, not the optimised geometry.
    report.pop("comsol", None)
    report.pop("comsol_run", None)
    result["report"] = report
    final_volumes, final_quality = tetra_metrics(current_points[new_tetra])
    by_phase = {}
    for phase in np.unique(new_phases):
        mask = new_phases == phase
        by_phase[str(int(phase))] = {"tetrahedra": int(mask.sum()), "volume_mm3": float(final_volumes[mask].sum()),
                                     "minimum": float(final_quality[mask].min()),
                                     "p01": float(np.quantile(final_quality[mask], .01))}
    original_by_phase = {}
    for phase in np.unique(phases):
        mask = phases == phase
        original_by_phase[str(int(phase))] = float(volumes[mask].sum())
    volume_shift_pp = {}
    box_volume = float(np.prod(config.size))
    for phase, row in by_phase.items():
        volume_shift_pp[phase] = (row["volume_mm3"] - original_by_phase[phase]) / box_volume * 100
        if surface_moves == 0 and not mesh.flips.get("cap") and not np.isclose(row["volume_mm3"], original_by_phase[phase], rtol=1e-10, atol=0):
            raise ValueError(f"Phase {phase} volume changed during optimisation")
    if "phase_ids" in model and not np.isclose(final_volumes.sum(), box_volume, rtol=1e-10):
        raise ValueError("Solid and fluid no longer partition the box after optimisation")
    # Every moved vertex that lies on the implicit surface must still be on it.
    # (Linear-cut vertices that never were on the exact surface are never moved.)
    on_surface = set(map(int, np.unique(arrays["boundary"][arrays["boundary_ids"] == 8])))
    checked = sorted(moved_surface_vertices & on_surface)
    residual = surface_residuals(current_points, checked, config) if checked else np.zeros(0)
    deviation = surface_distances(current_points, checked, config) if checked else np.zeros(0)
    if surface_envelope is None:
        if len(residual) and residual.max() > 1e-6:
            raise ValueError("Surface vertices left the implicit surface during optimisation")
    elif len(deviation) and deviation.max() > surface_envelope * 1.01:
        raise ValueError("Surface vertices left the surface envelope during optimisation")
    report.update(
        optimization={
            "method": "cut-layer reconnection with interior smoothing; optional box-face retriangulation "
                      "and budgeted implicit-surface moves/interface cap flips",
            "quality_threshold": quality_threshold, "rounds_run": len(history), "history": history,
            "rounds_retained": sum(row["retained"] for row in history),
            "rolled_back": bool(rollback_reasons), "rollback_reasons": rollback_reasons,
            "max_phase_volume_shift_pp": max_phase_volume_shift_pp,
            "max_median_loss": max_median_loss, "chord_tolerance": chord_tolerance,
            "surface_phase_quality_guard": bool(surface and two_phase),
            "flips": dict(mesh.flips), "tetrahedra_before": int(len(tetra)), "tetrahedra_after": int(len(new_tetra)),
            "minimum_before": float(before.min()), "minimum_after": float(final_quality.min()),
            "p01_before": float(np.quantile(before, .01)), "p01_after": float(np.quantile(final_quality, .01)),
            "poor_below_01_before": int(np.sum(before < .1)), "poor_below_01_after": int(np.sum(final_quality < .1)),
            "by_phase": by_phase, "surface_moves": int(surface_moves),
            "phase_volume_shift_pp": volume_shift_pp,
            "surface_residual_max": float(residual.max()) if len(residual) else None,
            "surface_envelope_mm": surface_envelope,
            "surface_deviation_max_mm": float(deviation.max()) if len(deviation) else 0.0,
            "moved_surface_vertices": int(len(checked)),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "limits": "boundary-constrained cells can remain; no solver-accuracy guarantee",
        },
        tetrahedra=int(len(new_tetra)),
        minimum_tetra_volume_mm3=float(final_volumes.min()),
        quality_quantiles=dict(zip(("min", "p01", "median", "max"),
                                  map(float, np.quantile(final_quality, [0, .01, .5, 1])))),
        low_quality_count_below_001=int(np.sum(final_quality < .01)),
        low_quality_volume_fraction_below_001=float(final_volumes[final_quality < .01].sum() / final_volumes.sum()),
        boundary_sha256=boundary_digest(current_points, arrays["boundary"]),
        mesh_sha256=hashlib.sha256(current_points.astype("<f8").tobytes() + new_tetra.astype("<i4").tobytes()
                                   + new_domains.astype("<i4").tobytes()).hexdigest(),
        comsol_verified=False, comsol_solved=False,
    )
    if periodic:
        from .periodic import check_periodic

        report["periodic"] = check_periodic(result, periodic, config.size)
        if not report["periodic"]["matched"]:
            raise ValueError("Optimisation broke the periodic face correspondence")
    refresh_geometry_statistics(result, config, final_volumes)
    report["warnings"] = [w for w in report.get("warnings", []) if not w.startswith("Low-quality cut elements")]
    if np.any(final_quality < .01):
        report["warnings"].append("Low-quality cut elements remain after optimisation; inspect each phase and check solver convergence.")
    if rollback_reasons:
        report["warnings"].append("An optimisation round exceeded its budget and was rolled back: " + "; ".join(rollback_reasons))
    report["quality_audit"] = quality_audit(result)
    return result
