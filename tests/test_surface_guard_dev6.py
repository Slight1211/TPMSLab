"""Check accepted surface moves against current geometry, independently of the guard."""

import numpy as np

from tpmslab import Config, generate
from tpmslab import batch_smoothing as bs
from tpmslab import surface_smoothing as ss
from tpmslab.model import field_and_threshold


def test_surface_guard_uses_current_neighbours_and_bounded_chunks(monkeypatch):
    # This is the real r=8 case whose old, sweep-wide cached guard admitted
    # 77 moves that failed its own chord limit at the moment of acceptance.
    # Observe the actual selected trials; do not replace the selection policy
    # or use _chord_guard as the oracle for its own correctness.
    config = Config(resolution=8, domain_mode="solid_fluid", snap_alpha=.25,
                    edge_interpolation="implicit_root")
    model = generate(config)
    faces = model["boundary"][model["boundary_ids"] == 8]
    surface_vertices = set(map(int, np.unique(faces)))
    incident = {v: faces[np.any(faces == v, axis=1)] for v in surface_vertices}
    original_points = model["points"].copy()
    observations = {"checked": 0, "changed_neighbours": 0, "violations": []}
    scoring = {}
    guard_sizes = []
    score_trials = bs.score_trials
    choose = bs.choose
    guard = ss._chord_guard
    chunks = bs.chunks

    def fields(xyz):
        value, threshold = field_and_threshold(config, *np.asarray(xyz).T)
        return np.stack((threshold - value, threshold + value), axis=-1)

    def observed_score(points, cells, mask, vertices, trials):
        scoring.update(points=points, vertices=vertices, trials=trials)
        return score_trials(points, cells, mask, vertices, trials)

    def observed_choose(*args, **kwargs):
        pick = choose(*args, **kwargs)
        points, vertices, trials = (scoring[key] for key in ("points", "vertices", "trials"))
        selected = [i for i, v in enumerate(vertices) if pick[i] >= 0 and int(v) in surface_vertices]
        if not selected:
            return pick
        positions = points[vertices[selected]]
        sides = np.abs(fields(positions)).argmin(axis=1)
        # The guard uses one common normalisation per vertex for before/after.
        # Independently integrate area * |g| at each actual incident face centre
        # and divide by that vertex's finite-difference gradient magnitude.
        h = max(config.size) * 1e-6
        gradient = np.empty((len(selected), 3))
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = h
            derivatives = (fields(positions + offset) - fields(positions - offset)) / (2 * h)
            gradient[:, axis] = derivatives[np.arange(len(selected)), sides]
        norms = np.linalg.norm(gradient, axis=1)
        for j, i in enumerate(selected):
            vertex = int(vertices[i])
            face_ids = incident[vertex]
            before = points[face_ids]
            after = before.copy()
            after[face_ids == vertex] = trials[i, pick[i]]

            def error(triangles):
                area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                               triangles[:, 2] - triangles[:, 0]), axis=1) / 2
                residual = np.abs(fields(triangles.mean(axis=1))[:, sides[j]])
                return float(np.sum(area * residual / max(norms[j], 1e-300)))

            previous, proposed = error(before), error(after)
            observations["checked"] += 1
            if not np.array_equal(before, original_points[face_ids]):
                observations["changed_neighbours"] += 1
            if proposed > previous * (1 + 1e-6) + 1e-15:
                observations["violations"].append((vertex, previous, proposed))
        return pick

    def observed_guard(points, surface_faces, faces_of, candidates, *args, **kwargs):
        guard_sizes.append(len(candidates))
        return guard(points, surface_faces, faces_of, candidates, *args, **kwargs)

    # Exercise multiple bounded chunks even on this deliberately small mesh.
    monkeypatch.setattr(bs, "chunks", lambda array, size=64: chunks(array, size=min(size, 64)))
    monkeypatch.setattr(bs, "score_trials", observed_score)
    monkeypatch.setattr(bs, "choose", observed_choose)
    monkeypatch.setattr(ss, "_chord_guard", observed_guard)
    _, info = ss.surface_smooth(model["points"], model["tetra"], model["boundary"],
                                model["boundary_ids"], config,
                                quality_threshold=.5, sweeps=2, chord_tolerance=0.0)
    assert observations["checked"] > 0 and info["accepted"] > 0
    assert observations["changed_neighbours"] > 0
    assert not observations["violations"], observations["violations"][:5]
    assert len(guard_sizes) > 1 and max(guard_sizes) <= 64
    assert np.array_equal(model["points"], original_points)


def test_none_is_soft_chord_ranking_and_zero_is_a_hard_limit():
    config = Config(family="Custom", custom="Z", mode="solid_below", gradient="uniform",
                    density_start=.5, phase_degrees=(0, 0, 0))
    points = np.array([[1., 1., 2.5], [2., 1., 2.5], [1., 2., 2.5]])
    faces = np.array([[0, 1, 2]])
    trial = np.array([[[1.1, 1.1, 2.7]]])
    args = (points, faces, [[0], [0], [0]], np.array([0]), np.array([0]),
            np.array([True]), trial, config, 5e-6)
    soft, soft_delta = ss._chord_guard(*args, tolerance=None)
    hard, hard_delta = ss._chord_guard(*args, tolerance=0.0)
    assert soft[0, 0] and not hard[0, 0]
    assert soft_delta[0, 0] > 0
    assert np.array_equal(soft_delta, hard_delta)
