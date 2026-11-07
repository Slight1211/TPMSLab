"""Regression tests for the 0.3.9.dev2 review fixes."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tpmslab import Config, generate, save_model
from tpmslab import comsol
from tpmslab.comsol_dual import export_dual_java, spanning_fluid_domains
from tpmslab.volume import face_components, estimate_lattice_bytes


# --- connectivity -----------------------------------------------------------

def test_face_components_separate_vertex_and_edge_contact():
    # Two tetrahedra sharing one vertex (0) and two sharing an edge (0, 1).
    vertex_only = np.array([[0, 1, 2, 3], [0, 4, 5, 6]])
    edge_only = np.array([[0, 1, 2, 3], [0, 1, 4, 5]])
    shared_face = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
    assert face_components(vertex_only)[0] == 2
    assert face_components(edge_only)[0] == 2
    assert face_components(shared_face)[0] == 1


def test_report_counts_face_connectivity_and_pinches():
    report = generate(Config(resolution=8))["report"]
    assert report["connectivity"] == "shared faces"
    assert report["volume_components"] >= report["vertex_connected_components"]
    assert report["pinched_contacts"] == report["volume_components"] - report["vertex_connected_components"]


# --- memory pre-flight --------------------------------------------------------

def test_oversized_lattice_is_rejected_before_allocation():
    config = Config(cells=[100, 100, 100], resolution=128)  # accepted by validate()
    assert estimate_lattice_bytes(config) > 1e12
    with pytest.raises(ValueError, match="TPMSLAB_MAX_LATTICE_BYTES"):
        generate(config)


def test_lattice_limit_is_adjustable(monkeypatch):
    monkeypatch.setenv("TPMSLAB_MAX_LATTICE_BYTES", "1")
    with pytest.raises(ValueError, match="above the limit"):
        generate(Config(resolution=8))
    monkeypatch.setenv("TPMSLAB_MAX_LATTICE_BYTES", "oops")
    with pytest.raises(ValueError, match="number of bytes"):
        generate(Config(resolution=8))


# --- shared lattice / two-phase consistency ----------------------------------

@pytest.mark.parametrize("mode", ["sheet", "solid_above"])
def test_two_phase_reuses_one_lattice_and_matches_solid_only(mode):
    config = Config(family="Primitive Schwartz", mode=mode, resolution=10, snap_alpha=0.25)
    dual = generate(Config(**{**config.to_dict(), "domain_mode": "solid_fluid"}))
    solid = generate(config)
    assert dual["report"]["solid_volume_mm3"] == pytest.approx(solid["report"]["volume_mm3"], rel=1e-12)
    assert dual["report"]["vertex_snapping"] == solid["report"]["vertex_snapping"]
    assert dual["report"]["sampling_audit"]["performed"]
    assert len(dual["report"]["density_profile"]) == 10
    assert dual["report"]["density_profile_method"].startswith("centroid")
    assert all(0 <= row["sampled_density"] <= 1 for row in dual["report"]["density_profile"])


def test_two_phase_profile_reflects_gradient():
    report = generate(Config(resolution=10, domain_mode="solid_fluid",
                             density_start=.2, density_end=.6))["report"]
    values = [row["sampled_density"] for row in report["density_profile"]]
    assert np.mean(values[-3:]) > np.mean(values[:3]) + 0.2


# --- vectorised implicit roots -------------------------------------------------

def test_vectorised_implicit_root_matches_previous_accuracy():
    reference_density = 0.3749257028
    projected = generate(Config(family="Gyroid", domain_mode="solid_fluid", resolution=12,
                                edge_interpolation="implicit_root"))
    report = projected["report"]
    assert report["edge_root_unbracketed_cuts"] == 0
    assert report["edge_root_fully_projected_fraction"] > 0.95
    assert report["partition_volume_error_mm3"] < 1e-8
    # Every fully projected cut node lies on the implicit sheet surface.
    from tpmslab.model import field_and_threshold
    nodes = np.unique(projected["interface"])
    xyz = projected["points"][nodes]
    value, threshold = field_and_threshold(report_config := Config(**report["config"]), xyz[:, 0], xyz[:, 1], xyz[:, 2])
    residual = np.min(np.abs(np.stack((threshold - value, threshold + value))), axis=0)
    assert np.quantile(residual, 0.9) < 1e-9
    assert abs(report["actual_density"] - reference_density) < 0.01
    assert report_config.edge_interpolation == "implicit_root"


# --- sampling audit for built-in fields ----------------------------------------

def test_builtin_family_is_probed_and_warned_not_rejected():
    report = generate(Config(family="Gyroid", resolution=8))["report"]
    audit = report["sampling_audit"]
    assert audit["performed"] and audit["policy"] == "warn" and not audit["custom_field"]
    assert audit["voxels_checked"] == 8 ** 3


def test_custom_field_is_still_rejected():
    with pytest.raises(ValueError, match="under-resolved"):
        generate(Config(family="Custom", custom="sin(8*X)", gradient="uniform",
                        density_start=.25, phase_degrees=(0, 0, 0), resolution=8))


# --- COMSOL launcher --------------------------------------------------------------

@pytest.fixture
def exported(tmp_path, monkeypatch):
    model = generate(Config(family="Custom", custom="X", mode="solid_below",
                            gradient="uniform", density_start=.5,
                            phase_degrees=(0, 0, 0), resolution=8))
    folder = save_model(model, tmp_path / "model")
    monkeypatch.setattr(comsol, "detect_comsol", lambda: tmp_path)
    return folder, model["report"]


def fake_popen(calls, marker_line, version_line="***COMSOL 6.2.0.290 progress output file***\n"):
    def process(command, **kwargs):
        calls.append(command)
        working = Path(kwargs["cwd"])
        if "comsolcompile" in str(command[0]):
            stem = Path(command[-1]).stem
            (working / f"{stem}.class").write_bytes(b"mock class")
        else:
            (working / "comsol.log").write_text(version_line + marker_line, encoding="utf8")
        return SimpleNamespace(wait=lambda **k: 0)
    return process


def test_comsol_version_and_options_recorded_per_attempt(exported, monkeypatch):
    folder, report = exported
    calls = []
    monkeypatch.setattr(comsol.subprocess, "Popen", fake_popen(
        calls, 'TPMS_REOPEN_VERIFIED {"import_verified":true,"reopen_verified":true,"solved":false}'))
    result = comsol.build_mph(folder, report, element_order=2, processes=4)
    assert result["comsol_version"] == "6.2.0.290"
    state = json.loads((Path(result["output_directory"]) / "run.json").read_text())
    assert state["comsol_version"] == "6.2.0.290"
    assert state["element_order"] == 2 and state["processes"] == 4
    build = [c for c in calls if "comsolbatch" in str(c[0])][0]
    assert build[build.index("-np") + 1] == "4"
    java = (Path(result["output_directory"]) / "TPMSBuild.java").read_text()
    assert 'set("order_displacement",2)' in java
    assert "Math.abs(volume)*1e-10" in java  # relative reopen tolerance
    settings = json.loads((Path(result["output_directory"]) / "simulation_settings.json").read_text())
    assert settings["element_order"] == 2


@pytest.mark.parametrize("order", [0, 3, 1.0, True])
def test_invalid_element_order_rejected(exported, order):
    folder, report = exported
    with pytest.raises(ValueError, match="element_order"):
        comsol.build_mph(folder, report, element_order=order)


def test_remesh_uses_shared_launcher_with_reflection_and_preferences(tmp_path, monkeypatch):
    from tpmslab import comsol_remesh

    model = generate(Config(family="Custom", custom="X", mode="solid_below", gradient="uniform",
                            density_start=.5, phase_degrees=(0, 0, 0), resolution=8,
                            domain_mode="solid_fluid"))
    source = save_model(model, tmp_path / "source")
    prefs = tmp_path / "prefs"
    prefs.mkdir()
    (prefs / "comsol.prefs").write_text("security.external.enable=on\n")
    monkeypatch.setattr(comsol, "detect_comsol", lambda: tmp_path)
    calls = []
    audit = json.dumps({"tetrahedra": 1, "solid_tetrahedra": 1, "domain_volumes_mm3": [62.5, 62.5],
                        "total_volume_mm3": 125.0, "interface_area_mm2": 25.0,
                        "interface_selection_entities": 1, "cross_phase_faces": 1,
                        "interface_face_incidence_verified": True, "q01_solid": 0.9, "qmin_all": 0.5})
    monkeypatch.setattr(comsol.subprocess, "Popen", fake_popen(calls, "TPMS_QUALITY_REOPEN_VERIFIED " + audit))
    evidence = comsol_remesh.remesh_in_comsol(source, tmp_path / "remeshed", preferences_directory=prefs)
    java = (tmp_path / "remeshed" / "TPMSQualityRemesh.java").read_text()
    assert 'geometricModel("")' not in java and "independentMesh(m)" in java and "getMethod" in java
    for command in calls:
        assert command[command.index("-prefsdir") + 1] == str(tmp_path / "remeshed" / "preferences")
        assert "-recoverydir" in command and "-tmpdir" in command
    assert evidence["comsol_version"] == "6.2.0.290"
    assert evidence["verified"] and (tmp_path / "remeshed" / "runtime.json").is_file()


def test_audit_uses_shared_launcher(tmp_path, monkeypatch):
    from tpmslab.comsol_audit import audit_saved_flow

    monkeypatch.setattr(comsol, "detect_comsol", lambda: tmp_path)
    calls = []
    payload = json.dumps({"interface_area_comsol_mm2": 25.0, "interface_area_relative_error": 0.0,
                          "channels": [], "channel_sum_inlet_m3_s": 0, "channel_sum_outlet_m3_s": 0})
    monkeypatch.setattr(comsol.subprocess, "Popen", fake_popen(calls, "TPMS_AUDIT " + payload))
    (tmp_path / "model_solved.mph").write_bytes(b"mock")
    domain_map = [{"id": 1, "phase": "solid", "exterior_boundary_ids": [2, 3]},
                  {"id": 2, "phase": "fluid", "exterior_boundary_ids": [16, 17], "tetrahedra": 1, "volume_mm3": 1.0},
                  {"id": 3, "phase": "fluid", "exterior_boundary_ids": [16], "tetrahedra": 1, "volume_mm3": 1.0}]
    result = audit_saved_flow(tmp_path / "model_solved.mph", 25.0, domain_map, tmp_path / "audit")
    assert result["comsol_version"] == "6.2.0.290"
    java = (tmp_path / "audit" / "TPMSAudit.java").read_text()
    assert "new int[]{2}" in java  # the dead-end domain 3 is not audited as a channel
    assert any("-recoverydir" in c for c in calls)


# --- flow domains ----------------------------------------------------------------

def test_flow_export_excludes_dead_ends_instead_of_refusing(tmp_path):
    model = generate(Config(family="Custom", custom="X", mode="solid_below", gradient="uniform",
                            density_start=.5, phase_degrees=(0, 0, 0), resolution=8,
                            domain_mode="solid_fluid"))
    report = json.loads(json.dumps(model["report"]))
    spanning, excluded = spanning_fluid_domains(report["domain_map"])
    assert spanning and not excluded
    # Verified single-channel case: physics on fluid_domains, no restriction block.
    java = export_dual_java(tmp_path, report, solve=True).read_text()
    assert 'selection().named("fluid_domains")' in java and "flow_zmin" not in java
    assert 'excluded_fluid_domains\\":[]' in java
    # Add a synthetic closed pore: the model is still solvable on the spanning channel.
    report["domain_map"].append({"id": 9, "nastran_pid": 109, "phase": "fluid", "tetrahedra": 1,
                                 "volume_mm3": 0.1, "exterior_boundary_ids": []})
    report["fluid_components"] += 1
    java = export_dual_java(tmp_path, report, solve=True).read_text()
    assert 'selection().named("fluid_flow_domains")' in java
    assert '"flow_zmin"' in java and '"flow_zmax"' in java and 'excluded_fluid_domains\\":[9]' in java
    assert "@@" not in java
    settings = json.loads((tmp_path / "simulation_settings.json").read_text())
    assert settings["excluded_fluid_domains"] == [9]
    # No spanning channel at all is still rejected.
    report["domain_map"] = [d for d in report["domain_map"] if d["phase"] == "solid" or d["id"] == 9]
    with pytest.raises(ValueError, match="at least one fluid domain"):
        export_dual_java(tmp_path, report, solve=True)


def test_dual_element_order_sets_fluid_shape_order(tmp_path):
    model = generate(Config(family="Custom", custom="X", mode="solid_below", gradient="uniform",
                            density_start=.5, phase_degrees=(0, 0, 0), resolution=8,
                            domain_mode="solid_fluid"))
    java = export_dual_java(tmp_path, model["report"], solve=True, element_order=2).read_text()
    assert 'set("order_fluid",2)' in java
    java = export_dual_java(tmp_path, model["report"], solve=True, element_order=1).read_text()
    assert "order_fluid" not in java


# --- browser path keeps history --------------------------------------------------

def test_web_comsol_task_preserves_previous_verification(tmp_path, monkeypatch):
    from tpmslab import web

    model = generate(Config(family="Custom", custom="X", mode="solid_below", gradient="uniform",
                            density_start=.5, phase_degrees=(0, 0, 0), resolution=8))
    folder = save_model(model, tmp_path / "job")
    (folder / "comsol_verification.json").write_text('{"old": true}')
    seen = {}

    def fake_build(job_folder, report, solve, material, progress, **kwargs):
        seen["legacy_present"] = (job_folder / "comsol_verification.json").is_file()
        return {}

    monkeypatch.setattr(web, "build_mph", fake_build)
    job = {"folder": folder, "report": model["report"]}
    web.comsol_task(job, False, None)
    assert seen["legacy_present"], "web path must leave the historical record for build_mph to preserve"


def test_cli_generate_accepts_launch_options(tmp_path, monkeypatch):
    from tpmslab.cli import main

    captured = {}

    def fake_build(folder, report, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("tpmslab.comsol.build_mph", fake_build)
    config = tmp_path / "c.json"
    config.write_text(json.dumps({"resolution": 8}))
    assert main(["generate", "--config", str(config), "--out", str(tmp_path / "out"), "--mph",
                 "--timeout", "77", "--element-order", "2", "--np", "3",
                 "--preferences-directory", str(tmp_path)]) == 0
    assert captured["build_timeout_s"] == 77 and captured["element_order"] == 2
    assert captured["processes"] == 3 and captured["preferences_directory"] == tmp_path
