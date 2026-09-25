import hashlib
import json
import sys
import subprocess
from dataclasses import replace
import numpy as np
import pytest
from tpmslab import Config, generate, save_model


def test_quality_selection_preserves_boundary_and_volume():
    a = generate(Config(resolution=12, quality_strategy="pulling"))["report"]
    b = generate(Config(resolution=12))["report"]
    assert a["boundary_sha256"] == b["boundary_sha256"]
    assert b["volume_mm3"] == pytest.approx(a["volume_mm3"], rel=1e-12)
    assert b["quality_quantiles"]["min"] > a["quality_quantiles"]["min"]
    assert b["improved_cut_cells"] > 0
    assert b["tetrahedra"] > a["tetrahedra"]


@pytest.mark.parametrize("strategy", ["pulling", "quality_fan"])
def test_analytic_half_box(strategy):
    c = Config(
        family="Custom",
        custom="X",
        mode="solid_below",
        gradient="uniform",
        density_start=0.5,
        phase_degrees=(0, 0, 0),
        resolution=12,
        quality_strategy=strategy,
    )
    mesh = generate(c)
    assert mesh["report"]["volume_mm3"] == pytest.approx(62.5, abs=1e-10)
    assert mesh["points"][:, 0].max() == pytest.approx(2.5)
    assert mesh["report"]["volume_components"] == 1


def test_export_integrity_and_no_overwrite(tmp_path):
    c = Config(
        family="Custom",
        custom="X",
        mode="solid_below",
        gradient="uniform",
        density_start=0.5,
        phase_degrees=(0, 0, 0),
        resolution=12,
    )
    mesh = generate(c)
    folder = save_model(mesh, tmp_path / "model")
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, record in manifest.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == record["sha256"]
    with pytest.raises(FileExistsError):
        save_model(mesh, folder)
    assert generate(c)["report"]["mesh_sha256"] == mesh["report"]["mesh_sha256"]


def test_config_immutable_sequences_and_invalid_strategy():
    data = Config().to_dict()
    data["size"] = [5.0, 5.0, 5.0]
    c = Config.from_dict(data)
    data["size"][0] = 7
    assert c.size == (5.0, 5.0, 5.0)
    with pytest.raises(ValueError):
        replace(c, quality_strategy="unknown").validate()


def test_core_does_not_import_web_and_cli_works(tmp_path):
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tpmslab,sys; assert 'flask' not in sys.modules; print(tpmslab.__version__)",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    run = subprocess.run(
        [sys.executable, "-m", "tpmslab", "families"], cwd=tmp_path, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
    assert len(json.loads(run.stdout)) == 29


def test_phase_field_helper():
    from tpmslab.model import raw_field

    c = Config()
    a = raw_field(c, np.array([0.3]), np.array([0.8]), np.array([1.1]))
    b = raw_field(
        replace(c, phase_degrees=(0, 0, 0)), np.array([0.3]), np.array([0.8]), np.array([1.1])
    )
    assert not np.allclose(a, b)
