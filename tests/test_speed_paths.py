"""Fast paths must reproduce the reference computations they replace."""

import hashlib

import numpy as np
import pytest

from tpmslab import Config, generate
from tpmslab import batch_smoothing as bs
from tpmslab.model import evaluate, FAMILIES
from tpmslab.quality import _fan_metrics, tetra_metrics, unique_rows
from tpmslab.volume import boundary_digest


def test_unique_rows_matches_numpy():
    rng = np.random.default_rng(0)
    rows = np.sort(rng.integers(0, 300, (5000, 3)), axis=1)
    rows = np.vstack([rows, rows[:900]])
    expected = np.unique(rows, axis=0, return_index=True, return_inverse=True, return_counts=True)
    actual = unique_rows(rows, return_index=True, return_inverse=True, return_counts=True)
    for a, b in zip(expected, actual):
        assert np.array_equal(a, b.reshape(a.shape))
    assert np.array_equal(unique_rows(rows), np.unique(rows, axis=0))


def test_fan_metrics_are_bitwise_tetra_metrics():
    rng = np.random.default_rng(1)
    for n in range(1, 11):
        points = rng.random((n, 4, 3)) * rng.uniform(0.01, 5)
        nodes = points.reshape(-1, 3).tolist()
        cells = [[4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3] for k in range(n)]
        v, q = tetra_metrics(points)
        v2, q2 = _fan_metrics(nodes, cells)
        assert np.array_equal(v, v2) and np.array_equal(q, q2)


def reference_digest(points, faces):
    rows = []
    for face in faces:
        xyz = np.round(points[face], 11)
        order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
        rows.append(tuple(xyz[order].ravel()))
    return hashlib.sha256(np.asarray(sorted(rows), dtype="<f8").tobytes()).hexdigest()


def test_boundary_digest_matches_reference_with_ties():
    rng = np.random.default_rng(2)
    points = np.round(rng.random((40, 3)) * 4) / 4
    faces = rng.integers(0, 40, (3000, 3))
    assert boundary_digest(points, faces) == reference_digest(points, faces)
    model = generate(Config(resolution=8, domain_mode="solid_fluid"))
    assert boundary_digest(model["points"], model["boundary"]) == reference_digest(model["points"], model["boundary"])


def test_compiled_expressions_keep_values_and_whitelist():
    rng = np.random.default_rng(3)
    x, y, z = (rng.random(500) * 7 for _ in range(3))
    for item in FAMILIES.values():
        value = evaluate(item["expression"], x, y, z)
        assert value.shape == x.shape and np.all(np.isfinite(value))
    assert np.allclose(evaluate("sin(X)*cos(Y)+Z**2", x, y, z), np.sin(x) * np.cos(y) + z ** 2, rtol=0, atol=0)
    for bad in ("sin", "sin + 1", "__import__('os')", "X.__class__", "open(X)", "[X for X in Y]"):
        with pytest.raises((ValueError, SyntaxError)):
            evaluate(bad, x[:2], y[:2], z[:2])


def test_independent_batches_have_disjoint_stars():
    model = generate(Config(resolution=8))
    tetra = model["tetra"]
    cell_ids, cell_starts = bs.incidence(tetra, len(model["points"]))
    candidates = np.unique(tetra)[::3]
    batches = bs.independent_batches(candidates, cell_ids, cell_starts)
    assert np.array_equal(np.sort(np.concatenate(batches)), np.sort(candidates))
    for batch in batches:
        stars = [set(cell_ids[cell_starts[v]:cell_starts[v + 1]]) for v in batch]
        seen = set()
        for star in stars:
            assert not (star & seen)
            seen |= star


def test_affine_trial_scoring_matches_direct_quality():
    rng = np.random.default_rng(4)
    points = rng.random((60, 3))
    vertices = np.array([0, 1])
    cells = np.array([[[0, 5, 6, 7], [8, 0, 9, 10], [11, 12, 0, 13], [14, 15, 16, 0]],
                      [[1, 17, 18, 19], [20, 1, 21, 22], [23, 24, 1, 25], [26, 27, 28, 1]]])
    mask = np.ones((2, 4), dtype=bool)
    mask[1, 2] = False
    trials = rng.random((2, 6, 3))
    vmin, qmin = bs.score_trials(points, cells, mask, vertices, trials)
    moved = np.where((cells == vertices[:, None, None])[:, None, :, :, None],
                     trials[:, :, None, None, :], points[cells][:, None])
    v, q = bs.volume_quality(moved)
    assert np.allclose(vmin, np.where(mask[:, None], v, np.inf).min(2), rtol=0, atol=1e-15)
    assert np.allclose(qmin, np.where(mask[:, None], q, np.inf).min(2), rtol=0, atol=1e-14)
