"""Packed chord geometry and per-phase acceptance use independent cell checks."""

import numpy as np

from tpmslab import Config, generate
from tpmslab import batch_smoothing as bs
from tpmslab import surface_smoothing as ss
from tpmslab.quality import tetra_metrics


def test_trial_phase_minima_match_direct_tetrahedra():
    rng = np.random.default_rng(42)
    points = rng.random((20, 3))
    cells = np.array([[[0, 2, 3, 4], [5, 0, 6, 7], [8, 9, 0, 10]],
                      [[1, 11, 12, 13], [14, 1, 15, 16], [17, 18, 1, 19]]])
    mask = np.array([[True, True, True], [True, True, False]])
    vertices = np.array([0, 1])
    phases = np.array([[1, 2, 1], [2, 2, 1]])
    trials = rng.random((2, 5, 3))
    by_phase = {}
    actual = bs.score_trials(points, cells, mask, vertices, trials,
                             phase_ids=phases, phase_minima=by_phase)
    reference = bs.score_trials(points, cells, mask, vertices, trials)
    assert all(np.array_equal(a, b) for a, b in zip(actual, reference))
    for row in range(2):
        for trial_id in range(5):
            xyz = points[cells[row]].copy()
            xyz[cells[row] == vertices[row]] = trials[row, trial_id]
            _, quality = tetra_metrics(xyz)
            for phase, values in by_phase.items():
                selected = mask[row] & (phases[row] == phase)
                expected = quality[selected].min() if np.any(selected) else np.inf
                assert np.isclose(values[row, trial_id], expected, rtol=1e-12, atol=1e-14)


def test_packed_chord_matches_explicit_moved_triangles():
    rng = np.random.default_rng(43)
    points = rng.uniform(.5, 4.5, (7, 3))
    faces = np.array([[0, 1, 3], [0, 3, 4], [0, 4, 2], [2, 4, 5], [2, 5, 6]])
    faces_of = [list(np.flatnonzero(np.any(faces == v, axis=1))) for v in range(len(points))]
    candidates = np.array([0, 1, 2])
    sides = np.array([0, 1, 0])
    trial = points[candidates][None] + rng.normal(scale=.08, size=(4, 3, 3))
    config, h = Config(), 5e-6
    keep, delta = ss._chord_guard(points, faces, faces_of, candidates, sides,
                                  np.ones(3, bool), trial, config, h, tolerance=0.0)
    for row, vertex in enumerate(candidates):
        ids = faces[faces_of[vertex]]
        base = points[ids]
        norm = ss._gradient_norm(config, points[[vertex]], sides[row], h)[0]
        current = ss._chord_error(base, sides[row], config, h, norm).sum()
        for t in range(len(trial)):
            moved = base.copy()
            moved[ids == vertex] = trial[t, row]
            proposed = ss._chord_error(moved, sides[row], config, h, norm).sum()
            assert np.isclose(delta[t, row], proposed - current, rtol=1e-11, atol=1e-13)
            assert keep[t, row] == (proposed <= current * (1 + 1e-6) + 1e-15)


def test_interface_moves_preserve_each_phase_star_minimum(monkeypatch):
    config = Config(resolution=8, domain_mode="solid_fluid", snap_alpha=.25,
                    edge_interpolation="implicit_root")
    mesh = generate(config)
    original_score, original_choose = bs.score_trials, bs.choose
    context, checked, violations = {}, [], []

    def observed_score(points, cells, mask, vertices, trials, **kwargs):
        context.update(points=points, cells=cells, mask=mask, vertices=vertices,
                       trials=trials, phases=kwargs.get("phase_ids"))
        return original_score(points, cells, mask, vertices, trials, **kwargs)

    def observed_choose(*args, **kwargs):
        pick = original_choose(*args, **kwargs)
        phases = context["phases"]
        if phases is None:
            return pick
        for row in np.flatnonzero(pick >= 0):
            mask = context["mask"][row]
            labels = phases[row][mask]
            if len(np.unique(labels)) < 2:
                continue
            cells = context["cells"][row][mask]
            xyz = context["points"][cells]
            _, before = tetra_metrics(xyz)
            moved = xyz.copy()
            moved[cells == context["vertices"][row]] = context["trials"][row, pick[row]]
            _, after = tetra_metrics(moved)
            checked.append(int(context["vertices"][row]))
            for phase in np.unique(labels):
                selected = labels == phase
                if after[selected].min() < before[selected].min() - 2e-12:
                    violations.append((checked[-1], int(phase)))
        return pick

    monkeypatch.setattr(bs, "score_trials", observed_score)
    monkeypatch.setattr(bs, "choose", observed_choose)
    _, info = ss.surface_smooth(mesh["points"], mesh["tetra"], mesh["boundary"],
                                mesh["boundary_ids"], config, quality_threshold=.5,
                                sweeps=1, chord_tolerance=0.0, phase_ids=mesh["phase_ids"])
    assert checked and not violations
    assert info["phase_guard_rejected_trials"] > 0
