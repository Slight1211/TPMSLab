"""Reconnection memoization must never reuse geometry after points move."""
import numpy as np

from tpmslab import Config, generate, optimize_mesh, quality_audit
from tpmslab import optimize as opt
from tpmslab.quality import tet_scalar


def test_metric_cache_is_bounded_and_invalidated_with_geometry(monkeypatch):
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    tetra = np.array([[0, 1, 2, 3]])
    mesh = opt._Mesh(points, tetra, np.array([1]), set())
    monkeypatch.setattr(opt, "_METRIC_CACHE_LIMIT", 1)
    first = mesh._metric((0, 1, 2, 3))
    second = mesh._metric((1, 0, 2, 3))
    assert first == tet_scalar(points.tolist(), (0, 1, 2, 3))
    assert second == tet_scalar(points.tolist(), (1, 0, 2, 3))
    assert first[0] == -second[0] and len(mesh._metric_cache) == 1
    moved = points.copy()
    moved[3, 2] = .5
    mesh.points = moved
    assert not mesh._metric_cache
    assert mesh._metric((0, 1, 2, 3)) == tet_scalar(moved.tolist(), (0, 1, 2, 3))
    assert mesh._metric((0, 1, 2, 3))[0] != first[0]


def test_cached_reconnections_match_uncached_after_interior_smoothing(monkeypatch):
    base = generate(Config(resolution=8, domain_mode="solid_fluid", snap_alpha=.25,
                           edge_interpolation="implicit_root"))
    cached = optimize_mesh(base, surface=False, rounds=2)
    monkeypatch.setattr(opt._Mesh, "_metric", lambda self, tet: tet_scalar(self.P, tet))
    reference = optimize_mesh(base, surface=False, rounds=2)
    for key, value in reference.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(cached[key], value), key
    for key in ("quality_quantiles", "actual_density", "mesh_sha256", "boundary_sha256"):
        assert cached["report"][key] == reference["report"][key]
    assert cached["report"]["optimization"]["flips"] == reference["report"]["optimization"]["flips"]
    assert quality_audit(cached)["valid"]
