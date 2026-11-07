"""Cut-layer optimisation: reconnection, interior smoothing, implicit-surface sliding."""

import json

import numpy as np
import pytest

from tpmslab import Config, generate, optimize_mesh, condition_mesh
from tpmslab.quality import tetra_metrics
from tpmslab.surface_smoothing import surface_residuals
from tpmslab.verification import audit_interface


def exact_solid_volume(config, samples=128):
    """Midpoint integration of the exact implicit solid indicator."""
    from tpmslab.model import raw_field
    axes = [(np.arange(samples) + 0.5) * L / samples for L in config.size]
    X, Y = np.meshgrid(axes[0], axes[1], indexing="ij")
    inside = sum(np.count_nonzero(raw_field(config, X, Y, np.full_like(X, z)) >= 0) for z in axes[2])
    return inside * float(np.prod(config.size)) / samples ** 3


def sorted_faces(faces):
    return {tuple(sorted(map(int, f))) for f in np.asarray(faces)}


@pytest.fixture(scope="module")
def snapped():
    return generate(Config(resolution=8, domain_mode="solid_fluid", snap_alpha=0.25,
                           edge_interpolation="implicit_root"))


def test_optimisation_improves_tail_and_keeps_conformity(snapped):
    before = snapped["report"]["quality_quantiles"]
    model = optimize_mesh(snapped)
    report = model["report"]
    after = report["quality_quantiles"]
    assert after["p01"] > before["p01"] * 1.3
    assert after["min"] > before["min"]
    assert audit_interface(model)["valid"]
    assert report["quality_audit"]["valid"]
    # Boundary operations may re-triangulate a flat box face or swap the
    # diagonal of an interface quad, but every boundary id keeps its vertices
    # and its triangle count, and box faces keep their exact area.
    assert len(model["boundary"]) == len(snapped["boundary"])
    assert len(model["interface"]) == len(snapped["interface"])
    for pid in np.unique(snapped["boundary_ids"]):
        assert set(np.unique(model["boundary"][model["boundary_ids"] == pid])) == \
            set(np.unique(snapped["boundary"][snapped["boundary_ids"] == pid]))
        assert np.sum(model["boundary_ids"] == pid) == np.sum(snapped["boundary_ids"] == pid)
    # Each flat box face (solid id 2-7 plus fluid id 12-17) keeps its full area.
    for pid in range(2, 8):
        def area(m):
            tri = m["points"][m["boundary"][np.isin(m["boundary_ids"], (pid, pid + 10))]]
            return np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2
        assert area(model) == pytest.approx(25.0, rel=1e-9)
        assert area(snapped) == pytest.approx(25.0, rel=1e-9)
    flips = report["optimization"]["flips"]
    assert flips.get("boundary_edge", 0) > 0 and flips.get("cap", 0) > 0
    # Both phases still partition the box, with positive cells only.
    volumes, _ = tetra_metrics(model["points"][model["tetra"]])
    assert volumes.sum() == pytest.approx(125.0, rel=1e-10)
    p = model["points"][model["tetra"]]
    signed = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
    assert np.all(signed > 0)
    # Surface vertices stayed on the exact implicit surface.
    config = Config(**report["config"])
    surface_vertices = np.unique(model["boundary"][model["boundary_ids"] == 8])
    assert np.quantile(surface_residuals(model["points"], surface_vertices, config), .99) < 1e-6
    # Box faces are still flat.
    for axis in range(3):
        for side, value in enumerate((0.0, 5.0)):
            nodes = np.unique(model["boundary"][np.isin(model["boundary_ids"], (2 + 2 * axis + side, 12 + 2 * axis + side))])
            assert np.allclose(model["points"][nodes, axis], value)
    evidence = report["optimization"]
    assert evidence["surface_moves"] > 0 and sum(evidence["flips"].values()) > 0
    # Surface vertices slide along the exact level set; the flat triangles may
    # move the phase volume slightly, but not away from the exact implicit volume.
    assert abs(evidence["phase_volume_shift_pp"]["1"]) < 0.3
    reference = exact_solid_volume(config)
    before_error = abs(snapped["report"]["solid_volume_mm3"] - reference)
    after_error = abs(report["solid_volume_mm3"] - reference)
    assert after_error <= before_error + 0.02 * 125 / 100
    assert report["domain_map"][0]["tetrahedra"] == int(np.sum(model["domains"] == report["domain_map"][0]["id"]))
    assert report["solid_volume_mm3"] == pytest.approx(volumes[model["phase_ids"] == 1].sum())
    json.dumps(report, allow_nan=False)


def test_surface_disabled_preserves_boundary_nodes_and_volumes(snapped):
    model = optimize_mesh(snapped, surface=False, rounds=2)
    boundary_nodes = np.unique(snapped["boundary"])
    assert np.array_equal(model["points"][boundary_nodes], snapped["points"][boundary_nodes])
    for phase in (1, 2):
        before = tetra_metrics(snapped["points"][snapped["tetra"][snapped["phase_ids"] == phase]])[0].sum()
        after = tetra_metrics(model["points"][model["tetra"][model["phase_ids"] == phase]])[0].sum()
        assert after == pytest.approx(before, rel=1e-10)
    assert model["report"]["optimization"]["surface_moves"] == 0


def test_solid_only_model_can_be_optimised():
    model = generate(Config(resolution=8, snap_alpha=0.25), optimize=True)
    assert "optimization" in model["report"]
    assert model["surface"].is_watertight
    assert model["report"]["quality_audit"]["valid"]
    assert model["report"]["volume_mm3"] == pytest.approx(
        tetra_metrics(model["points"][model["tetra"]])[0].sum())


@pytest.mark.parametrize("kwargs", [dict(quality_threshold=0), dict(quality_threshold=1.5),
                                    dict(rounds=0), dict(rounds=2.0), dict(min_round_gain=-1)])
def test_invalid_optimisation_parameters(snapped, kwargs):
    with pytest.raises(ValueError):
        optimize_mesh(snapped, **kwargs)


def test_condition_median_budget_validation(snapped):
    with pytest.raises(ValueError, match="max_median_loss"):
        condition_mesh(snapped, max_median_loss=1.0)
    evidence = condition_mesh(snapped, max_median_loss=0.2)["report"]["conditioning"]
    assert evidence["max_median_loss"] == 0.2


def test_cli_and_web_accept_optimize(tmp_path, monkeypatch):
    from tpmslab.cli import main
    from tpmslab.web import app, TOKEN

    config = tmp_path / "c.json"
    config.write_text(json.dumps({"resolution": 8, "snap_alpha": 0.25, "optimize": True}))
    assert main(["generate", "--config", str(config), "--out", str(tmp_path / "out")]) == 0
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf8"))
    assert "optimization" in report and report["quality_audit"]["valid"]
    client = app.test_client()
    bad = client.post("/api/generate", json={**Config().to_dict(), "optimize": "yes"},
                      headers={"X-TPMS-Token": TOKEN})
    assert bad.status_code == 400 and "optimize" in bad.json["error"]


def test_edge_removal_on_a_five_tet_ring():
    from tpmslab.optimize import _Mesh
    angles = np.arange(5) * 2 * np.pi / 5
    ring = np.c_[np.cos(angles), np.sin(angles), np.zeros(5)]
    points = np.vstack((ring, [[0, 0, 1.5], [0, 0, -1.5]]))
    d, e = 5, 6
    cells = []
    for i in range(5):
        tet = (i, (i + 1) % 5, d, e)
        p = points[list(tet)]
        if np.dot(p[1] - p[0], np.cross(p[2] - p[0], p[3] - p[0])) < 0:
            tet = (tet[1], tet[0], tet[2], tet[3])
        cells.append(tet)
    mesh = _Mesh(points, np.array(cells), np.ones(5, dtype=int), set())
    _, before = tetra_metrics(points[np.array(cells)])
    assert mesh.remove_edge((d, e), 1.0 + 1e-9)
    new = np.array(list(mesh.cells.values()))
    volumes, after = tetra_metrics(points[new])
    assert len(new) == 6 and (d, e) not in mesh.edges
    assert after.min() > before.min()
    assert volumes.sum() == pytest.approx(tetra_metrics(points[np.array(cells)])[0].sum(), rel=1e-12)
    assert mesh.flips["edge_removal_5+"] == 1


def test_vectorised_star_trials_match_scalar():
    from tpmslab.quality import star_scalar, star_trials
    rng = np.random.default_rng(3)
    points = rng.random((20, 3))
    star = np.array([[0, 1, 2, 3], [4, 0, 5, 6], [7, 8, 0, 9]])
    trials = rng.random((6, 3))
    vmin, qmin = star_trials(points, star, 0, trials)
    reference = np.array([star_scalar(points.tolist(), star.tolist(), 0, x.tolist()) for x in trials])
    assert np.allclose(vmin, reference[:, 0], rtol=0, atol=1e-15)
    assert np.allclose(qmin, reference[:, 1], rtol=0, atol=1e-14)
