"""The optional bridge must reject unusable input without touching outputs."""

import json

import pytest

from tpmslab import remesh_in_comsol


def test_requires_verified_two_phase_export(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("mesh.nas", "mesh.npz"):
        (source / name).write_bytes(b"placeholder")
    (source / "report.json").write_text(json.dumps({
        "config": {"domain_mode": "solid"},
        "interface_conforming": True,
        "domain_map": [{"phase": "solid"}],
    }), encoding="utf8")
    output = tmp_path / "new_model"
    with pytest.raises(ValueError, match="solid_fluid"):
        remesh_in_comsol(source, output)
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "important.txt").write_text("keep", encoding="utf8")
    with pytest.raises(FileExistsError):
        remesh_in_comsol(tmp_path / "missing", output)
    assert (output / "important.txt").read_text(encoding="utf8") == "keep"
