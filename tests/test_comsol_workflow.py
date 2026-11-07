import json
from pathlib import Path

import pytest

from tpmslab import Config, generate, run_comsol, save_model
from tpmslab import comsol
from tpmslab.io import file_sha256


@pytest.fixture
def exported(tmp_path, monkeypatch):
    model = generate(Config(family="Custom", custom="X", mode="solid_below",
                            gradient="uniform", density_start=.5,
                            phase_degrees=(0, 0, 0), resolution=8))
    folder = save_model(model, tmp_path / "model")
    monkeypatch.setattr(comsol, "detect_comsol", lambda: tmp_path)
    return folder, model["report"]


def test_success_then_failure_preserves_history_and_clears_current_success(exported, monkeypatch):
    folder, report = exported

    def success(attempt, report, solve, material, progress, **kwargs):
        assert file_sha256(attempt / "mesh.nas") == file_sha256(folder / "mesh.nas")
        (attempt / "model_solved.mph").write_bytes(b"mock MPH; not a solver validation")
        return {"import_verified": True, "reopen_verified": True, "solved": solve}

    monkeypatch.setattr(comsol, "_execute_mph", success)
    result = comsol.build_mph(folder, report, solve=True)
    first = Path(result["output_directory"])
    evidence = (first / "comsol_verification.json").read_bytes()
    assert json.loads((first / "run.json").read_text())["status"] == "succeeded"

    def failure(*args, **kwargs):
        raise RuntimeError("simulated compile failure")

    monkeypatch.setattr(comsol, "_execute_mph", failure)
    with pytest.raises(RuntimeError, match="simulated"):
        comsol.build_mph(folder, report, solve=True)
    current = json.loads((folder / "report.json").read_text())
    assert current["comsol_run"]["status"] == "failed"
    assert report["comsol_solved"] is False and current["comsol_verified"] is False
    assert "comsol" not in current
    assert not (folder / "comsol_verification.json").exists()
    assert (first / "comsol_verification.json").read_bytes() == evidence
    assert current["comsol_run"]["attempt_id"] != result["attempt_id"]


def test_repeated_tamper_rejection_does_not_rebaseline_inputs(exported, monkeypatch):
    folder, report = exported
    original = (folder / "input_manifest.json").read_bytes()
    with (folder / "mesh.nas").open("ab") as stream:
        stream.write(b"changed input")
    for _ in range(2):
        with pytest.raises(ValueError, match="mesh.nas.*manifest"):
            comsol.build_mph(folder, report)
        assert (folder / "input_manifest.json").read_bytes() == original
        assert report["comsol_run"]["status"] == "failed"


def test_invalid_material_marks_current_attempt_failed(exported):
    folder, report = exported
    report.update(comsol_solved=True, comsol_verified=True, comsol={"old": True})
    with pytest.raises(ValueError, match="Material"):
        comsol.build_mph(folder, report, solve=True, material={"E": -1, "nu": .3, "rho": 950})
    assert report["comsol_solved"] is False
    assert report["comsol_run"]["status"] == "failed"


def test_incomplete_evidence_is_not_success(exported, monkeypatch):
    folder, report = exported
    monkeypatch.setattr(comsol, "_execute_mph", lambda *a, **k: {"solved": True})
    with pytest.raises(RuntimeError, match="import and reopening"):
        comsol.build_mph(folder, report, solve=True)
    assert not report["comsol_solved"]


def test_zero_exit_from_failed_compiler_is_rejected(exported, monkeypatch):
    folder, report = exported
    calls = []

    def process(*args, **kwargs):
        calls.append(args[0])
        from types import SimpleNamespace
        return SimpleNamespace(wait=lambda **k: 0)

    monkeypatch.setattr(comsol.subprocess, "Popen", process)
    with pytest.raises(RuntimeError, match="produced no TPMSBuild.class"):
        comsol.build_mph(folder, report)
    assert len(calls) == 1
    assert not report["comsol_verified"]


def test_comsol_security_denial_is_explicit_and_not_success(exported, monkeypatch):
    folder, report = exported

    def process(command, **kwargs):
        from types import SimpleNamespace
        working = Path(kwargs["cwd"])
        if "comsolcompile" in str(command[0]):
            (working / "TPMSBuild.class").write_bytes(b"mock compiled class")
        else:
            (working / "comsol.log").write_text("java.security.AccessControlException: file read blocked")
        return SimpleNamespace(wait=lambda **k: 0)

    monkeypatch.setattr(comsol.subprocess, "Popen", process)
    with pytest.raises(RuntimeError, match="security denied file-system access"):
        comsol.build_mph(folder, report)
    assert report["comsol_run"]["status"] == "failed"
    assert not report["comsol_verified"]


def test_saved_model_public_api(exported, monkeypatch):
    folder, _ = exported
    monkeypatch.setattr(comsol, "_execute_mph", lambda *a, **k: {
        "solved": False, "import_verified": True, "reopen_verified": True})
    result = run_comsol(folder, solve=False)
    assert Path(result["output_directory"]).is_dir()
    assert result["solved"] is False


def test_explicit_preferences_are_copied_and_used_per_attempt(exported, monkeypatch, tmp_path):
    folder, report = exported
    requested = tmp_path / "trusted_preferences"
    requested.mkdir()
    content = b"security.external.enable=on\nsecurity.external.filepermission=limited\n"
    (requested / "comsol.prefs").write_bytes(content)
    calls = []

    def process(command, **kwargs):
        from types import SimpleNamespace
        calls.append(command)
        working = Path(kwargs["cwd"])
        if "comsolcompile" in str(command[0]):
            (working / "TPMSBuild.class").write_bytes(b"mock class")
        else:
            (working / "comsol.log").write_text('TPMS_REOPEN_VERIFIED {"import_verified":true,"reopen_verified":true,"solved":false}')
        return SimpleNamespace(wait=lambda **k: 0)

    monkeypatch.setattr(comsol.subprocess, "Popen", process)
    result = comsol.build_mph(folder, report, preferences_directory=requested)
    expected = Path(result["output_directory"]) / "preferences"
    for command in calls:
        assert command[command.index("-prefsdir") + 1] == str(expected)
        assert command[command.index("-migrateprefs") + 1] == "off"
        assert len(command[command.index("-recoverydir") + 1]) < len(str(expected / "recoveries"))
        assert "-tmpdir" in command and "-data" in command
    assert (requested / "comsol.prefs").read_bytes() == content
    assert (expected / "comsol.prefs").read_bytes() == content


def test_missing_preferences_file_is_rejected(exported, tmp_path):
    folder, report = exported
    with pytest.raises(ValueError, match="must contain comsol.prefs"):
        comsol.build_mph(folder, report, preferences_directory=tmp_path / "missing")
    assert report["comsol_run"]["status"] == "failed"


def test_recursive_manifest_and_streaming_hash(tmp_path):
    from tpmslab.io import refresh_manifest

    child = tmp_path / "run"
    child.mkdir()
    data = b"0123456789" * 200_000
    (child / "large.bin").write_bytes(data)
    refresh_manifest(tmp_path)
    record = json.loads((tmp_path / "manifest.json").read_text())["run/large.bin"]
    import hashlib
    assert record["sha256"] == hashlib.sha256(data).hexdigest()
