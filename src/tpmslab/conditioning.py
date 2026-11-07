"""Bounded interior-node maximin smoothing with immutable domain interfaces.

This is an opt-in local mesh conditioner, not a remesher or a convergence
certificate. It keeps all boundary/interface vertices and all connectivity fixed.
"""

from __future__ import annotations

import copy
import hashlib
import time

import numpy as np
from scipy.optimize import minimize
import trimesh

from .quality import tetra_metrics, unique_rows
from .volume import TET_FACES, boundary_digest


def _signed_volumes(points, tetra):
    p = points[tetra]
    return np.einsum(
        "ij,ij->i", p[:, 1] - p[:, 0],
        np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]),
    ) / 6


def _summary(volumes, quality, labels):
    result = {}
    for label in np.unique(labels):
        mask = labels == label
        v, q = volumes[mask], quality[mask]
        result[str(int(label))] = {
            "tetrahedra": int(mask.sum()),
            "volume_mm3": float(v.sum()),
            "minimum": float(q.min()),
            "p01": float(np.quantile(q, .01)),
            "median": float(np.median(q)),
            "poor_count_below_001": int(np.sum(q < .01)),
            "poor_volume_fraction_below_001": float(v[q < .01].sum() / v.sum()),
        }
    return result


def _validate(model):
    points = np.asarray(model["points"], dtype=float)
    raw_tetra = np.asarray(model["tetra"])
    raw_domains = np.asarray(model["domains"])
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or raw_tetra.ndim != 2 or raw_tetra.shape[1] != 4 or not len(raw_tetra)
            or raw_tetra.dtype.kind not in "iu"
            or raw_domains.shape != (len(raw_tetra),) or raw_domains.dtype.kind not in "iu"):
        raise ValueError("Conditioning requires finite points, integer tetrahedra and domain labels")
    tetra = raw_tetra.astype(np.int64, copy=False)
    if np.any(tetra < 0) or np.any(tetra >= len(points)) or np.any(raw_domains <= 0):
        raise ValueError("Invalid tetrahedron index or domain label")
    if len(unique_rows(np.sort(tetra, axis=1))) != len(tetra):
        raise ValueError("Duplicate tetrahedra are not admissible")
    volumes = _signed_volumes(points, tetra)
    if not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError("Conditioning requires strictly positive signed tetrahedron volumes")
    faces = tetra[:, TET_FACES].reshape(-1, 3)
    keys, inverse, counts = unique_rows(np.sort(faces, axis=1), return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Nonmanifold tetrahedron faces are not admissible")
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    paired = starts[counts == 2]
    left, right = order[paired], order[paired + 1]
    a, b = points[faces[left]], points[faces[right]]
    normals_a = np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
    normals_b = np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0])
    if np.any(np.einsum("ij,ij->i", normals_a, normals_b) >= 0):
        raise ValueError("Adjacent cells overlap or have inconsistent orientation")
    domains = raw_domains.astype(np.int64, copy=False)
    phases = np.asarray(model.get("phase_ids", np.ones(len(tetra), dtype=int)))
    if phases.shape != domains.shape or phases.dtype.kind not in "iu" or np.any(phases <= 0):
        raise ValueError("Invalid material phase labels")
    for domain in np.unique(domains):
        if len(np.unique(phases[domains == domain])) != 1:
            raise ValueError("A domain cannot contain more than one material phase")
    # Derive locked vertices from cells, rather than trusting supplied flags.
    exterior = keys[counts == 1]
    different_domain = domains[left // 4] != domains[right // 4]
    interface_keys = keys[inverse[left[different_domain]]]
    constrained = np.concatenate((exterior, interface_keys))
    actual_keys = {tuple(row) for row in constrained}
    declared = np.asarray(model["boundary"])
    if (declared.ndim != 2 or declared.shape[1] != 3 or declared.dtype.kind not in "iu"
            or len(declared) != len(actual_keys)
            or {tuple(row) for row in np.sort(declared, axis=1)} != actual_keys):
        raise ValueError("Declared boundaries must cover every exterior and domain-interface face once")
    if "interface" in model:
        expected = keys[inverse[left[phases[left // 4] != phases[right // 4]]]]
        supplied = np.asarray(model["interface"])
        if (supplied.ndim != 2 or supplied.shape[1] != 3 or supplied.dtype.kind not in "iu"
                or len(supplied) != len(expected)
                or {tuple(row) for row in np.sort(supplied, axis=1)}
                != {tuple(row) for row in expected}):
            raise ValueError("Declared material interface disagrees with cell incidence")
    locked = np.zeros(len(points), dtype=bool)
    locked[np.unique(constrained)] = True
    return points, tetra, domains, phases, volumes, locked


def quality_audit(model, *, minimum_quality=0.0, max_poor_fraction=1.0):
    """Reconstruct topology and quality; optionally reject a user quality limit.

    Quality limits are screening criteria, not FE/CFD accuracy guarantees.
    """
    if (not np.isfinite(minimum_quality) or not 0 <= minimum_quality <= 1
            or not np.isfinite(max_poor_fraction) or not 0 <= max_poor_fraction <= 1):
        raise ValueError("Quality limits must be finite values between zero and one")
    points, tetra, domains, phases, volumes, _ = _validate(model)
    _, quality = tetra_metrics(points[tetra])
    by_phase = _summary(volumes, quality, phases)
    for phase, row in by_phase.items():
        if (row["minimum"] < minimum_quality
                or row["poor_count_below_001"] / row["tetrahedra"] > max_poor_fraction):
            raise ValueError(f"Phase {phase} fails the requested quality limits: {row}")
    return {
        "valid": True, "positive_volumes": True,
        "by_phase": by_phase, "by_domain": _summary(volumes, quality, domains),
        "minimum_quality_required": float(minimum_quality),
        "max_poor_fraction_allowed": float(max_poor_fraction),
        "scope": "linear-cell topology and mean-ratio screening; no global collision or solver accuracy certificate",
    }


def _local_candidate(local, rows, cols, original_volumes, quality):
    """Maximise the smallest star quality using analytic first derivatives."""
    def quantities(y):
        p = local.copy()
        p[rows, cols] = y[:3]
        a, b, c = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
        volume = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
        derivatives = np.stack((np.cross(b, c), np.cross(c, a), np.cross(a, b)), axis=1) / 6
        dv = np.concatenate((-derivatives.sum(axis=1)[:, None], derivatives), axis=1)[rows, cols]
        edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        length_sum = sum(np.sum((p[:, i] - p[:, j]) ** 2, axis=1) for i, j in edges)
        ds = 8 * p[rows, cols] - 2 * p.sum(axis=1)
        safe = np.maximum(volume, 1e-30)
        q = 12 * (3 * safe) ** (2 / 3) / length_sum
        dq = q[:, None] * ((2 / 3) * dv / safe[:, None] - ds / length_sum[:, None])
        return volume, q, dv, dq

    def constraints(y):
        v, q, _, _ = quantities(y)
        return np.r_[q - y[3], v / original_volumes - 1e-8]

    def jacobian(y):
        _, _, dv, dq = quantities(y)
        return np.vstack((
            np.column_stack((dq, -np.ones(len(rows)))),
            np.column_stack((dv / original_volumes[:, None], np.zeros(len(rows)))),
        ))

    solution = minimize(
        lambda y: -y[3], np.r_[np.zeros(3), quality.min()],
        jac=lambda y: np.array([0., 0., 0., -1.]),
        constraints={"type": "ineq", "fun": constraints, "jac": jacobian},
        bounds=[(-.5, .5)] * 3 + [(0., 1.)], method="SLSQP",
        options={"maxiter": 35, "ftol": 1e-9},
    )
    v, q, _, _ = quantities(solution.x)
    return solution.x[:3], v, q, bool(solution.success)


def condition_mesh(model, *, max_nodes=256, sweeps=2, target_quality=.15, max_median_loss=.05,
                   progress=None):
    """Return a new model with qualified interior moves or an explicit rollback.

    Each sweep considers at most ``max_nodes`` vertices, ordered by incident
    worst quality. Limits are explicit and independent of mesh size. Boundaries,
    interface nodes, cell connectivity, domain labels and node count are immutable.
    The final gate forbids a worse phase minimum/p01/poor-cell count, more than
    ``max_median_loss`` relative median loss (default 5%), or any domain-volume
    drift above 1e-10 relative.
    """
    for name, value, high in (("max_nodes", max_nodes, 100000), ("sweeps", sweeps, 20)):
        if type(value) is not int or not 1 <= value <= high:
            raise ValueError(f"{name} must be an integer from 1 to {high}")
    if not np.isfinite(target_quality) or not 0 < target_quality <= 1:
        raise ValueError("target_quality must lie in (0, 1]")
    if not np.isfinite(max_median_loss) or not 0 <= max_median_loss < 1:
        raise ValueError("max_median_loss must lie in [0, 1)")
    started = time.monotonic()
    progress = progress or (lambda message: None)
    source, tetra, domains, phases, volumes, locked = _validate(model)
    points = source.copy()
    _, quality = tetra_metrics(points[tetra])
    before_phase = _summary(volumes, quality, phases)
    before_domain = _summary(volumes, quality, domains)
    incident = [[] for _ in points]
    for ti, vertices in enumerate(tetra):
        for node in vertices:
            if not locked[node]:
                incident[node].append(ti)
    records, attempted, accepted = [], 0, 0
    for sweep in range(sweeps):
        severity = np.full(len(points), np.inf)
        np.minimum.at(severity, tetra.ravel(), np.repeat(quality, 4))
        free = np.flatnonzero(~locked & (severity < target_quality))
        free = free[np.argsort(severity[free], kind="stable")][:max_nodes]
        for node in free:
            adj = np.asarray(incident[node])
            rows, cols = np.where(tetra[adj] == node)
            star = points[tetra[adj]]
            origin = points[node].copy()
            scale = float(np.mean(np.linalg.norm(star.reshape(-1, 3) - origin, axis=1)))
            if not scale > 0:
                continue
            v0, q0 = volumes[adj] / scale**3, quality[adj].copy()
            move, v, q, converged = _local_candidate((star - origin) / scale, rows, cols, v0, q0)
            attempted += 1
            admissible = (
                np.isfinite(move).all() and np.isfinite(q).all()
                and np.all(v > v0 * 1e-8)
                and q.min() > q0.min() * (1 + 1e-6)
                and np.sum(q < .01) <= np.sum(q0 < .01)
                and np.quantile(q, .01) >= np.quantile(q0, .01) - 1e-12
                and np.isclose(v.sum(), v0.sum(), rtol=1e-10, atol=1e-14)
            )
            if admissible:
                points[node] = origin + scale * move
                volumes[adj] = v * scale**3
                quality[adj] = q
                accepted += 1
                records.append({"node": int(node), "sweep": sweep + 1,
                                "minimum_before": float(q0.min()), "minimum_after": float(q.min()),
                                "optimizer_converged": converged})
        progress(f"Interior conditioning sweep {sweep + 1}/{sweeps}: {accepted}/{attempted} moves accepted")
    # Recompute from the final coordinates instead of trusting incremental state.
    final_volumes = _signed_volumes(points, tetra)
    _, final_quality = tetra_metrics(points[tetra])
    after_phase = _summary(final_volumes, final_quality, phases)
    after_domain = _summary(final_volumes, final_quality, domains)
    failures = []
    if not np.array_equal(points[locked], source[locked]):
        failures.append("A constrained vertex moved")
    if not np.isfinite(points).all() or np.any(final_volumes <= 0):
        failures.append("Nonfinite coordinates or nonpositive volume")
    for phase, before in before_phase.items():
        after = after_phase[phase]
        if (after["minimum"] < before["minimum"] - 1e-12
                or after["p01"] < before["p01"] - 1e-12
                or after["poor_count_below_001"] > before["poor_count_below_001"]
                or after["median"] < before["median"] * (1 - max_median_loss)):
            failures.append(f"Phase {phase} fails the quality non-regression gate")
    for domain, before in before_domain.items():
        if not np.isclose(after_domain[domain]["volume_mm3"], before["volume_mm3"], rtol=1e-10, atol=0):
            failures.append(f"Domain {domain} volume changed")
    if failures:
        points = source.copy()
        final_volumes = _signed_volumes(points, tetra)
        _, final_quality = tetra_metrics(points[tetra])
    result = dict(model)
    result.pop("vertex_keys", None)  # Original grid keys no longer locate free vertices.
    result["points"] = points
    result["report"] = copy.deepcopy(model["report"])
    report = result["report"]
    evidence = {
        "method": "bounded interior maximin mean-ratio smoothing",
        "accepted": not failures and accepted > 0, "rolled_back": bool(failures),
        "failure_reasons": failures, "attempted_moves": attempted,
        "candidate_accepted_moves": accepted, "retained_moves": accepted if not failures else 0,
        "max_nodes_per_sweep": max_nodes, "sweeps": sweeps, "target_quality": target_quality,
        "max_median_loss": max_median_loss,
        "locked_vertices": int(locked.sum()), "boundary_and_interface_fixed": True,
        "connectivity_and_labels_fixed": True, "tetrahedra_added": 0,
        "source_mesh_sha256": model["report"].get("mesh_sha256"),
        "before_by_phase": before_phase, "candidate_by_phase": after_phase,
        "after_by_phase": _summary(final_volumes, final_quality, phases),
        "before_by_domain": before_domain,
        "after_by_domain": _summary(final_volumes, final_quality, domains),
        "moves": records, "elapsed_seconds": round(time.monotonic() - started, 6),
        "limits": "Boundary-constrained poor cells can remain. No solver-convergence or smooth-geometry accuracy guarantee.",
    }
    report.update(
        conditioning=evidence,
        minimum_tetra_volume_mm3=float(final_volumes.min()),
        quality_quantiles=dict(zip(("min", "p01", "median", "max"),
                                  map(float, np.quantile(final_quality, [0, .01, .5, 1])))),
        low_quality_count_below_001=int(np.sum(final_quality < .01)),
        low_quality_volume_fraction_below_001=float(final_volumes[final_quality < .01].sum() / final_volumes.sum()),
        boundary_sha256=boundary_digest(points, model["boundary"]),
        mesh_sha256=hashlib.sha256(points.astype("<f8").tobytes()
                                   + tetra.astype("<i4").tobytes()
                                   + domains.astype("<i4").tobytes()).hexdigest(),
        generation_elapsed_seconds=model["report"].get("elapsed_seconds"),
        elapsed_seconds=round(model["report"].get("elapsed_seconds", 0) + evidence["elapsed_seconds"], 3),
        comsol_verified=False, comsol_solved=False,
    )
    for key, faces in (("surface", model.get("solid_boundary", model["boundary"])),
                       ("fluid_surface", model.get("fluid_boundary"))):
        if faces is not None:
            surface = trimesh.Trimesh(points.copy(), faces.copy(), process=False)
            surface.remove_unreferenced_vertices()
            result[key] = surface
    report["warnings"] = [w for w in report.get("warnings", []) if not w.startswith("Low-quality cut elements")]
    if np.any(final_quality < .01):
        report["warnings"].append("Low-quality cut elements remain after conditioning; inspect each phase and check solver convergence.")
    if failures:
        report["warnings"].append("Conditioning candidate failed its audit and was rolled back: " + "; ".join(failures))
    report["quality_audit"] = quality_audit(result)
    return result
