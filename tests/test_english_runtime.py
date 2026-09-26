import json
import re

import pytest

from tpmslab import Config, generate, save_model
from tpmslab import comsol


def assert_english(value):
    assert not re.search(r"[\u4e00-\u9fff]", value), value


def test_progress_and_export_warnings_are_english(tmp_path):
    messages = []
    model = generate(Config(resolution=8, domain_mode="solid_fluid"), progress=messages.append)
    assert messages
    assert model["report"]["warnings"]
    for message in messages + model["report"]["warnings"]:
        assert_english(message)
    assert not any("static elasticity demo" in w for w in model["report"]["warnings"])
    folder = save_model(model, tmp_path / "model")
    assert_english(json.dumps(json.loads((folder / "report.json").read_text()), ensure_ascii=False))


def test_invalid_configuration_message_is_english():
    with pytest.raises(ValueError) as exc:
        Config.from_dict({"resolution": 1})
    assert "Resolution" in str(exc.value)
    assert_english(str(exc.value))


def test_missing_comsol_message_is_english(tmp_path, monkeypatch):
    monkeypatch.setattr(comsol, "detect_comsol", lambda: None)
    with pytest.raises(RuntimeError) as exc:
        comsol.build_mph(tmp_path, {})
    assert "COMSOL_BIN" in str(exc.value)
    assert_english(str(exc.value))
