import copy
import json

import numpy as np
import pytest
from tpmslab import Config, condition_mesh, generate, quality_audit, save_model
from tpmslab.conditioning import _signed_volumes
from tpmslab.verification import audit_interface


def star_mesh():
    # One distorted interior node in a tetrahedron with an exact fixed boundary.
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                       [0., 0., 1.], [.02, .03, .04]])
    boundary = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    tetra = np.column_stack((np.full(4, 4), boundary))
    return {"points": points, "tetra": tetra, "boundary": boundary,
            "boundary_ids": np.full(4, 8), "domains": np.ones(4, dtype=int),
            "report": {"warnings": [], "elapsed_seconds": 0., "comsol_verified": True}}


def scalar_volumes(points, cells):
    # Separate scalar triple-product expression, independent of conditioner state.
    return np.array([np.dot(points[b] - points[a],
                            np.cross(points[c] - points[a], points[d] - points[a])) / 6
                     for a, b, c, d in cells])


def test_distorted_interior_star_improves_without_refinement():
    source = star_mesh()
    original = copy.deepcopy(source)
    before = quality_audit(source)["by_phase"]["1"]
    result = condition_mesh(source, target_quality=.9)
    after = quality_audit(result)["by_phase"]["1"]
    assert result["report"]["conditioning"]["accepted"]
    assert after["minimum"] > 2 * before["minimum"]
    assert np.array_equal(result["points"][:4], source["points"][:4])
    assert np.array_equal(result["tetra"], source["tetra"])
    assert np.array_equal(source["points"], original["points"])
    assert source["report"] == original["report"]
    assert scalar_volumes(result["points"], result["tetra"]).sum() == pytest.approx(1 / 6)
    assert result["report"]["comsol_verified"] is False


@pytest.fixture(scope="module")
def dual():
    return generate(Config(resolution=8, domain_mode="solid_fluid"))


def test_dual_phase_geometry_labels_and_interface_survive(dual, tmp_path):
    old_points = dual["points"].copy()
    result = condition_mesh(dual, max_nodes=128)
    assert result["report"]["conditioning"]["accepted"]
    assert np.array_equal(dual["points"], old_points)
    for key in ("tetra", "domains", "phase_ids", "boundary", "boundary_ids", "interface"):
        assert np.array_equal(result[key], dual[key])
    fixed = np.unique(dual["boundary"])
    assert np.array_equal(result["points"][fixed], dual["points"][fixed])
    assert audit_interface(result)["valid"]
    old_v = scalar_volumes(old_points, dual["tetra"])
    new_v = scalar_volumes(result["points"], result["tetra"])
    assert np.all(new_v > 0)
    for d in np.unique(dual["domains"]):
        mask = dual["domains"] == d
        assert new_v[mask].sum() == pytest.approx(old_v[mask].sum(), rel=1e-11)
    out = save_model(result, tmp_path / "conditioned")
    stored = np.load(out / "mesh.npz")
    assert audit_interface({"points": stored["points_mm"], "tetra": stored["tetrahedra"],
                            "phase_ids": stored["phase_ids"], "interface": stored["interface"]})["valid"]
    report = json.loads((out / "report.json").read_text())
    assert report["conditioning"]["accepted"]


@pytest.mark.parametrize("mutation", ["negative", "duplicate", "missing_boundary", "missing_interface", "nan", "mixed_domain"])
def test_invalid_mesh_rejected_before_optimization(dual, mutation):
    model = copy.deepcopy(dual)
    if mutation == "negative":
        model["tetra"][0] = model["tetra"][0, [1, 0, 2, 3]]
    elif mutation == "duplicate":
        model["tetra"][1] = model["tetra"][0]
    elif mutation == "missing_boundary":
        model["boundary"] = model["boundary"][:-1]
    elif mutation == "missing_interface":
        model["interface"] = model["interface"][:-1]
    elif mutation == "nan":
        model["points"][0, 0] = np.nan
    else:
        model["domains"][:] = 1
    with pytest.raises(ValueError):
        condition_mesh(model)


def test_optimizer_candidate_is_audited_even_if_it_claims_success(monkeypatch):
    import tpmslab.conditioning as module
    source = star_mesh()
    monkeypatch.setattr(module, "_local_candidate", lambda local, rows, cols, v, q:
                        (np.array([np.nan, 0., 0.]), v, q, True))
    result = condition_mesh(source)
    assert not result["report"]["conditioning"]["accepted"]
    assert np.array_equal(result["points"], source["points"])


def test_failed_global_gate_rolls_back_with_evidence(monkeypatch):
    import tpmslab.conditioning as module
    original_summary = module._summary
    count = 0

    def bad_final_median(v, q, labels):
        nonlocal count
        count += 1
        result = original_summary(v, q, labels)
        if count == 3:  # Candidate phase summary, after all local moves.
            result["1"]["median"] = 0.
        return result

    monkeypatch.setattr(module, "_summary", bad_final_median)
    source = star_mesh()
    result = condition_mesh(source)
    e = result["report"]["conditioning"]
    assert e["rolled_back"] and not e["accepted"] and e["retained_moves"] == 0
    assert e["failure_reasons"]
    assert np.array_equal(result["points"], source["points"])


@pytest.mark.parametrize("kwargs", [{"sweeps": 0}, {"sweeps": True}, {"max_nodes": -1},
                                     {"max_nodes": 1.5}, {"target_quality": np.nan}])
def test_invalid_work_limits(kwargs):
    with pytest.raises(ValueError):
        condition_mesh(star_mesh(), **kwargs)


def test_quality_gate_rejects_before_cli_export(tmp_path):
    from tpmslab.cli import main
    config = tmp_path / "config.json"
    config.write_text(json.dumps(Config(resolution=8, domain_mode="solid_fluid").to_dict()))
    dest = tmp_path / "must_not_exist"
    with pytest.raises(SystemExit) as exc:
        main(["generate", "--config", str(config), "--out", str(dest),
              "--condition", "--min-quality", "0.99"])
    assert exc.value.code == 2
    assert not dest.exists()


def test_generate_api_opt_in_and_boolean_validation():
    with pytest.raises(TypeError):
        generate(Config(resolution=8), condition="false")
    result = generate(Config(resolution=8), condition=True)
    assert "conditioning" in result["report"]
    assert _signed_volumes(result["points"], result["tetra"]).min() > 0


@pytest.mark.parametrize("scale", [1e-3, 1e3])
def test_coordinate_scale_does_not_change_conditioning_decision(scale):
    source = star_mesh()
    reference = condition_mesh(source, target_quality=.9)
    scaled = star_mesh()
    scaled["points"] *= scale
    result = condition_mesh(scaled, target_quality=.9)
    assert result["report"]["conditioning"]["accepted"]
    assert np.allclose(result["points"] / scale, reference["points"], atol=1e-8)


def test_browser_request_runs_conditioning_and_exports_report(tmp_path, monkeypatch):
    import tpmslab.web as web

    class Immediate:
        def submit(self, function, *args):
            function(*args)

    monkeypatch.setattr(web, "POOL", Immediate())
    monkeypatch.setattr(web, "RUNS", tmp_path)
    monkeypatch.setattr(web, "JOBS", {})
    monkeypatch.setattr(web, "BUSY", False)
    client = web.app.test_client()
    headers = {"X-TPMS-Token": web.TOKEN}
    invalid = client.post("/api/generate", json={"condition": "false"}, headers=headers)
    assert invalid.status_code == 400
    response = client.post("/api/generate", headers=headers,
                           json={**Config(resolution=8, domain_mode="solid_fluid").to_dict(),
                                 "condition": True})
    assert response.status_code == 200
    job = web.JOBS[response.json["id"]]
    assert job["status"] == "ready", job.get("error")
    assert job["report"]["conditioning"]["accepted"]
    assert (job["folder"] / "mesh.nas").is_file()
    stored = json.loads((job["folder"] / "report.json").read_text())
    assert stored["quality_audit"]["valid"]
