import json
import numpy as np
import pytest
from tpmslab.model import Config, calibration, evaluate
from tpmslab import generate, save_model

@pytest.mark.parametrize("value", [0, 7, -1, 56.5, True, "56", None])
def test_invalid_calibration_resolution(value):
    with pytest.raises(ValueError, match="m_cal"):
        Config.from_dict({"m_cal": value})

@pytest.mark.parametrize("mode", ["sheet", "solid_above", "solid_below"])
def test_calibration_samples_and_default_compatibility(mode):
    expression = Config().expression
    a = (np.arange(56) + 0.5) * (2*np.pi/56)
    values = evaluate(expression, a[:,None,None], a[None,:,None], a[None,None,:])
    if mode == "sheet": values = abs(values)
    elif mode == "solid_above": values = -values
    p, samples = calibration(expression, mode)
    np.testing.assert_array_equal(samples, np.sort(values.ravel()))
    np.testing.assert_array_equal(p, np.linspace(0,1,56**3))
    assert len(calibration(expression, mode, 16)[1]) == 16**3
    assert len(calibration(expression, mode, 24)[1]) == 24**3

@pytest.mark.parametrize("domain_mode", ["solid", "solid_fluid"])
def test_custom_calibration_reaches_generation_and_export(tmp_path, domain_mode):
    calibration.cache_clear()
    cfg = Config(resolution=8, m_cal=24, domain_mode=domain_mode)
    mesh = generate(cfg)
    assert mesh["report"]["config"]["m_cal"] == 24
    assert calibration.cache_info().currsize == 1
    before = calibration.cache_info().hits
    calibration(cfg.expression, cfg.mode, 24)
    assert calibration.cache_info().hits == before + 1
    folder = save_model(mesh, tmp_path/"model")
    records = list(folder.glob("*.json"))
    assert any('"m_cal": 24' in f.read_text(encoding="utf-8") for f in records)
    assert mesh["report"]["watertight"]

def test_legacy_config_and_unbounded_parameter():
    assert Config.from_dict({}).m_cal == 56
    cfg = Config.from_dict({"m_cal": 1024})
    assert Config.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
