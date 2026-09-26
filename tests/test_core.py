import numpy as np
import pytest
from tpmslab.model import Config, FAMILIES, evaluate
from tpmslab.volume import generate_volume, write_nastran


@pytest.mark.parametrize("name", list(FAMILIES))
def test_all_reference_families_closed_volume(name):
    m = generate_volume(Config(family=name, gradient="uniform", density_start=0.4, resolution=12))
    assert m["report"]["watertight"]
    assert 0 < m["report"]["actual_density"] < 1
    p = m["points"][m["tetra"]]
    vol = (
        np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6
    )
    assert np.all(vol > 0)
    assert np.isclose(vol.sum(), m["surface"].volume, rtol=1e-9)
    assert np.all(m["points"] >= -1e-10) and np.all(m["points"] <= 5 + 1e-10)


@pytest.mark.parametrize("mode", ["solid_above", "solid_below"])
@pytest.mark.parametrize("name", ["Gyroid", "Primitive Schwartz", "Diamond", "IWP"])
def test_solid_networks(name, mode):
    m = generate_volume(
        Config(family=name, mode=mode, resolution=12, density_start=0.3, density_end=0.55)
    )
    assert m["report"]["watertight"]


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('whoami')",
        "X.__class__",
        "[X for X in Y]",
        "X**100000",
        "open(X)",
        "exp(exp(exp(X)))",
    ],
)
def test_custom_expressions_reject_unsafe_or_nonfinite(expression):
    with pytest.raises((ValueError, SyntaxError)):
        evaluate(expression, np.array([100.0]), np.array([1.0]), np.array([1.0]))


@pytest.mark.parametrize(
    "change",
    [
        {"size": [0, 5, 5]},
        {"cells": [1, 1, 1.5]},
        {"resolution": 7},
        {"resolution": 8.5},
        {"resolution": True},
        {"cells": [0, 1, 1]},
        {"cells": [True, 1, 1]},
        {"density_start": float("nan")},
        {"family": "unknown"},
        {"gradient": "periodic", "density_start": 0.1},
    ],
)
def test_parameters_rejected(change):
    with pytest.raises(ValueError):
        Config.from_dict({**Config().to_dict(), **change})


def test_density_and_resolution_converge():
    coarse = generate_volume(Config(gradient="uniform", density_start=0.4, resolution=12))
    fine = generate_volume(Config(gradient="uniform", density_start=0.4, resolution=24))
    assert abs(fine["report"]["actual_density"] - 0.4) < abs(
        coarse["report"]["actual_density"] - 0.4
    )
    assert abs(fine["report"]["actual_density"] - 0.4) < 0.012


def test_density_gradient_has_measured_effect():
    m = generate_volume(Config(resolution=16, density_start=0.2, density_end=0.6))
    p = m["report"]["density_profile"]
    assert (
        np.mean([x["sampled_density"] for x in p[-3:]])
        > np.mean([x["sampled_density"] for x in p[:3]]) + 0.2
    )


def test_periodic_xyz():
    m = generate_volume(Config(resolution=12, gradient="periodic", density_start=0.4))
    assert m["report"]["volume_components"] == 1
    assert m["report"]["watertight"]


def test_nastran_has_tetrahedra_and_boundary_tags(tmp_path):
    m = generate_volume(Config(resolution=12))
    p = tmp_path / "mesh.nas"
    write_nastran(m, p)
    text = p.read_text()
    assert text.count("CTETRA,") == len(m["tetra"])
    assert text.count("CTRIA3,") == len(m["boundary"])
    assert set(m["boundary_ids"]) == set(range(2, 9))


def test_reference_aliases():
    for name, item in FAMILIES.items():
        if "alias_of" in item:
            x = np.linspace(0, 6, 19)
            assert np.allclose(
                evaluate(item["expression"], x, x * 0.3, x * 0.7),
                evaluate(FAMILIES[item["alias_of"]]["expression"], x, x * 0.3, x * 0.7),
            )


def test_http_rejects_cross_site_and_invalid_parameters():
    from tpmslab.web import app, TOKEN

    client = app.test_client()
    assert client.get("/api/info", headers={"Host": "evil.example"}).status_code == 403
    assert client.post("/api/generate", json={}).status_code == 403
    response = client.post(
        "/api/generate", json={"density_start": 2}, headers={"X-TPMS-Token": TOKEN}
    )
    assert response.status_code == 400
    assert "error" in response.json


@pytest.mark.parametrize("cells,resolution", [([1, 1, 1], 48), ([2, 2, 2], 32), ([6, 6, 6], 48), ([8, 9, 10], 64), ([100, 100, 100], 128)])
def test_large_grid_config_is_not_rejected(cells, resolution):
    config = Config.from_dict({**Config().to_dict(), "cells": cells, "resolution": resolution})
    assert config.resolution == resolution
    assert tuple(config.cells) == tuple(cells)


def test_mesh_inputs_have_no_fixed_upper_limit():
    from pathlib import Path
    from html.parser import HTMLParser
    import tpmslab

    class Inputs(HTMLParser):
        def __init__(self):
            super().__init__()
            self.inputs = {}

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "input" and attrs.get("id") in {"resolution", "nx", "ny", "nz"}:
                self.inputs[attrs["id"]] = attrs

    parser = Inputs()
    parser.feed((Path(tpmslab.__file__).parent / "static" / "index.html").read_text(encoding="utf8"))
    assert len(parser.inputs) == 4
    for attrs in parser.inputs.values():
        assert attrs["type"] == "number"
        assert attrs["step"] == "1"
        assert "max" not in attrs
