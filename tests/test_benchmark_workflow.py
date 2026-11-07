import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


def load_benchmark(name):
    path = Path(__file__).resolve().parents[1] / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gmsh_parameterization_failure_always_finalizes(monkeypatch):
    module = load_benchmark("transport_convergence")
    events = []

    def fail(*args):
        raise RuntimeError("parameterization failed")

    gmsh = SimpleNamespace(initialize=lambda: events.append("initialize"),
                           finalize=lambda: events.append("finalize"),
                           option=SimpleNamespace(setNumber=lambda *a: None), merge=lambda *a: None,
                           model=SimpleNamespace(mesh=SimpleNamespace(classifySurfaces=lambda *a: None,
                                                                      createGeometry=fail)))
    monkeypatch.setitem(sys.modules, "gmsh", gmsh)
    with pytest.raises(RuntimeError, match="parameterization"):
        module.gmsh_solid("unused.stl", .5)
    assert events == ["initialize", "finalize"]


def test_transport_checkpoint_survives_interruption_and_resume(tmp_path, monkeypatch):
    module = load_benchmark("transport_convergence")
    monkeypatch.setattr(module, "generate", lambda c: {
        "points": np.zeros((4, 3)), "tetra": np.zeros((2, 4), dtype=int), "phase_ids": np.array([1, 2])})
    calls = []

    def solve(*args):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return {"k_eff_over_bulk": .5, "cg_iterations": 1, "unknowns": 1}

    monkeypatch.setattr(module, "effective_coefficient", solve)
    options = ["--out", str(tmp_path / "study"), "--resolutions", "8", "--designs", "Gyroid_central"]
    with pytest.raises(KeyboardInterrupt):
        module.main(options)
    path = tmp_path / "study" / "transport_convergence.csv"
    with path.open(encoding="utf-8-sig") as stream:
        assert len(list(csv.DictReader(stream))) == 1
    module.main(options + ["--resume"])
    with path.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert len({(r["variant"], r["phase"]) for r in rows}) == 4


def test_comsol_elasticity_labels_actual_algorithm_and_resumes(tmp_path, monkeypatch):
    module = load_benchmark("comsol_convergence")
    monkeypatch.setattr(module, "detect_comsol", lambda: tmp_path)
    configs = []

    def generate(config):
        configs.append(config)
        return {"report": {"tetrahedra": 10, "vertices": 8, "actual_density": .4}}

    monkeypatch.setattr(module, "generate", generate)
    monkeypatch.setattr(module, "save_model", lambda model, folder: folder.mkdir())
    monkeypatch.setattr(module, "build_mph", lambda folder, *a, **k: {
        "output_directory": str(folder.resolve()), "attempt_id": "mock",
        "strain_energy_J": 1, "max_displacement_m": .001})
    options = ["--out", str(tmp_path / "study"), "--resolutions", "8", "--designs", "Gyroid_central", "--skip-flow"]
    assert module.main(options) == 0
    assert module.main(options + ["--resume"]) == 0
    assert len(configs) == 2
    with (tmp_path / "study" / "comsol_convergence.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["variant"] for r in rows] == ["baseline", "snapped"]
    assert all(r["edge_interpolation"] == "linear" for r in rows)
