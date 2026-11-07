"""Final-array reporting and actual geometry-budget rollback regressions."""

import copy

import numpy as np
import pytest

from tpmslab import Config, generate, optimize_mesh, quality_audit
from tpmslab.quality import tetra_metrics
from tpmslab.reporting import density_profile


@pytest.fixture(scope="module")
def mesh():
    return generate(Config(resolution=8, domain_mode="solid_fluid", snap_alpha=.25,
                           edge_interpolation="implicit_root"))


def test_zero_volume_budget_restores_actual_mesh_and_input_is_unchanged(mesh):
    before = {key: mesh[key].copy() for key in ("points", "tetra", "domains", "boundary", "interface", "phase_ids")}
    candidate = optimize_mesh(mesh, rounds=2, max_phase_volume_shift_pp=0)
    evidence = candidate["report"]["optimization"]
    assert evidence["rolled_back"] and evidence["rounds_retained"] == 0
    assert any("volume shift" in reason for reason in evidence["rollback_reasons"])
    assert evidence["surface_moves"] == 0 and not any(evidence["flips"].values())
    for key, value in before.items():
        assert np.array_equal(candidate[key], value), key
        assert np.array_equal(mesh[key], value), key
    assert quality_audit(candidate)["valid"]


def test_phase_quality_regression_restores_mesh_even_with_positive_cells(mesh, monkeypatch):
    from tpmslab import optimize

    points, tetra, phases = mesh["points"], mesh["tetra"], mesh["phase_ids"]
    volumes, quality = tetra_metrics(points[tetra])
    locked = np.zeros(len(points), dtype=bool)
    locked[np.unique(mesh["boundary"])] = True
    severity = np.full(len(points), np.inf)
    np.minimum.at(severity, tetra.ravel(), np.repeat(quality, 4))
    free = np.flatnonzero(~locked & np.isfinite(severity))
    free = free[np.argsort(severity[free], kind="stable")]

    def signed(xyz):
        return np.einsum("ij,ij->i", xyz[:, 1] - xyz[:, 0],
                         np.cross(xyz[:, 2] - xyz[:, 0], xyz[:, 3] - xyz[:, 0])) / 6

    # Move one interior vertex almost to the first incident face it would
    # cross. Signed volumes vary affinely along this segment, so stopping
    # just before the first zero keeps every cell positive while creating a
    # genuine thin tetrahedron. The enclosing boundary and phase volume stay
    # fixed. This exercises the real quality calculation, not a mocked score.
    degraded = None
    for vertex in free:
        adjacent = np.flatnonzero(np.any(tetra == vertex, axis=1))
        worst = tetra[adjacent[np.argmin(quality[adjacent])]]
        target = points[worst[worst != vertex]].mean(axis=0)
        local = points[tetra[adjacent]]
        endpoint = local.copy()
        moving = tetra[adjacent] == vertex
        endpoint[moving] = target
        start = signed(local)
        slope = signed(endpoint) - start
        shrinking = slope < 0
        if not np.any(shrinking):
            continue
        first_zero = np.min(-start[shrinking] / slope[shrinking])
        position = points[vertex] + first_zero * (1 - 1e-6) * (target - points[vertex])
        trial = local.copy()
        trial[moving] = position
        phase = phases[adjacent[0]]
        assert np.all(phases[adjacent] == phase)
        if (np.all(signed(trial) > 0)
                and tetra_metrics(trial)[1].min() < quality[phases == phase].min() * .9):
            degraded = points.copy()
            degraded[vertex] = position
            break
    assert degraded is not None, "Fixture needs an interior vertex with a positive-volume degrading move"
    actual_volumes = signed(degraded[tetra])
    assert np.all(actual_volumes > 0)
    assert np.array_equal(degraded[locked], points[locked])
    for phase in np.unique(phases):
        mask = phases == phase
        assert actual_volumes[mask].sum() == pytest.approx(volumes[mask].sum(), rel=1e-12)

    calls = []

    def degrading_surface(current_points, current_tetra, *args, **kwargs):
        calls.append(True)
        assert np.array_equal(current_points, points)
        assert np.array_equal(current_tetra, tetra)
        return degraded.copy(), {"accepted": 1, "moved_vertices": [int(vertex)]}

    monkeypatch.setattr(optimize, "surface_smooth", degrading_surface)
    before = {key: mesh[key].copy() for key in
              ("points", "tetra", "domains", "boundary", "interface", "phase_ids")}
    # No cell is below this threshold, so reconnections do not obscure the
    # injected move. Disable the volume budget to isolate quality rejection.
    result = optimize_mesh(mesh, rounds=1, quality_threshold=float(quality.min() / 2),
                           focus_quantile=None, smooth=False, cap_flips=False,
                           boundary_flips=False, max_phase_volume_shift_pp=None)
    evidence = result["report"]["optimization"]
    assert calls == [True]
    assert evidence["rolled_back"] and evidence["rounds_retained"] == 0
    assert any("quality budget exceeded" in reason for reason in evidence["rollback_reasons"])
    assert evidence["surface_moves"] == 0 and not any(evidence["flips"].values())
    for key, value in before.items():
        assert np.array_equal(result[key], value), key
        assert np.array_equal(mesh[key], value), key
    assert quality_audit(result)["valid"]


def test_optimized_profile_and_domain_volumes_match_final_arrays(mesh):
    result = optimize_mesh(mesh, rounds=2)
    report = result["report"]
    volumes, _ = tetra_metrics(result["points"][result["tetra"]])
    solid = result["phase_ids"] == 1
    actual = float(volumes[solid].sum() / 125)
    profile = report["density_profile"]
    weighted = sum(r["sampled_density"] * r["slab_count"] for r in profile) / sum(r["slab_count"] for r in profile)
    assert report["optimization"]["rounds_retained"] > 0
    assert report["actual_density"] == pytest.approx(actual, abs=1e-14)
    assert weighted == pytest.approx(actual, abs=1e-14)
    assert profile != mesh["report"]["density_profile"]
    assert report["triangles"] == len(result["boundary"])
    for domain in report["domain_map"]:
        mask = result["domains"] == domain["id"]
        assert domain["volume_mm3"] == pytest.approx(volumes[mask].sum())
        assert domain["tetrahedra"] == int(mask.sum())
    assert max(map(abs, report["optimization"]["phase_volume_shift_pp"].values())) <= .3 + 1e-10


def test_unequal_slabs_conserve_solid_volume():
    config = Config(resolution=12)
    # Two independently specified tetrahedra fall into the first and last slabs.
    points = np.array([[0, 0, .1], [1, 0, .1], [0, 1, .1], [0, 0, .2],
                       [0, 0, 4.8], [1, 0, 4.8], [0, 1, 4.8], [0, 0, 4.9]])
    tetra = np.arange(8).reshape(2, 4)
    volumes, _ = tetra_metrics(points[tetra])
    profile = density_profile(points, tetra, volumes, np.ones(2, bool), config)
    assert {r["slab_count"] for r in profile} == {1, 2}
    density = sum(r["sampled_density"] * r["slab_count"] for r in profile) / 12
    assert density == pytest.approx(volumes.sum() / 125, abs=1e-16)


def test_nonuniform_blend_honors_configured_volume_budget():
    base = generate(Config(resolution=8, domain_mode="solid_fluid", snap_alpha=.25,
                           edge_interpolation="implicit_root", combine="blend"))
    result = optimize_mesh(base, rounds=2, max_phase_volume_shift_pp=.01)
    evidence = result["report"]["optimization"]
    assert max(map(abs, evidence["phase_volume_shift_pp"].values())) <= .01 + 1e-10
    assert quality_audit(result)["valid"]


def test_slsqp_uses_current_reconnected_boundaries(mesh):
    result = optimize_mesh(mesh, rounds=1, surface=False, smooth="slsqp", smooth_nodes=8, smooth_sweeps=1)
    assert result["report"]["optimization"]["flips"].get("boundary_edge", 0) > 0
    assert quality_audit(result)["valid"]


def test_optimized_geometry_clears_input_solver_evidence(mesh):
    source = dict(mesh, report=copy.deepcopy(mesh["report"]))
    source["report"].update(comsol_verified=True, comsol_solved=True,
                            comsol={"solved": True}, comsol_run={"status": "succeeded"})
    result = optimize_mesh(source, rounds=1, surface=False)
    assert not result["report"]["comsol_verified"] and not result["report"]["comsol_solved"]
    assert "comsol" not in result["report"] and "comsol_run" not in result["report"]
    assert source["report"]["comsol_run"]["status"] == "succeeded"


@pytest.mark.parametrize("kwargs", [
    {"max_phase_volume_shift_pp": -1}, {"max_phase_volume_shift_pp": float("nan")},
    {"max_median_loss": 1}, {"max_median_loss": -1},
    {"chord_tolerance": -1}, {"chord_tolerance": float("nan")},
])
def test_invalid_quality_budgets_are_rejected(mesh, kwargs):
    with pytest.raises(ValueError):
        optimize_mesh(mesh, **kwargs)
