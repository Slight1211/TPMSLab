import json
from dataclasses import replace
import numpy as np
import pytest
from tpmslab import Config, generate, save_model
from tpmslab.volume import TET_FACES
from tpmslab.comsol import export_java


@pytest.mark.parametrize("family", ["Gyroid", "Primitive Schwartz", "Diamond"])
@pytest.mark.parametrize("mode", ["sheet", "solid_above", "solid_below"])
def test_complementary_partition(family, mode):
    c = Config(family=family, mode=mode, domain_mode="solid_fluid", resolution=12)
    m = generate(c)
    p = m["points"][m["tetra"]]
    v = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6
    assert np.all(v > 0)
    assert v.sum() == pytest.approx(np.prod(c.size), rel=1e-12)
    assert m["report"]["solid_volume_mm3"] + m["report"]["fluid_volume_mm3"] == pytest.approx(125)
    assert m["report"]["solid_volume_mm3"] == pytest.approx(
        generate(replace(c, domain_mode="solid"))["report"]["volume_mm3"]
    )
    assert set(m["phase_ids"]) == {1, 2}
    faces = m["tetra"][:, TET_FACES].reshape(-1, 3)
    face_owners = {}
    for i, face in enumerate(faces):
        face_owners.setdefault(tuple(sorted(face)), []).append(i // 4)
    expected_interface = set()
    for key, owners in face_owners.items():
        assert len(owners) <= 2
        if len(owners) == 2 and m["phase_ids"][owners[0]] != m["phase_ids"][owners[1]]:
            expected_interface.add(key)
    assert expected_interface == {tuple(sorted(t)) for t in m["interface"]}
    assert len(m["interface"]) == len(expected_interface)
    for d in m["report"]["domain_map"]:
        ids = m["domains"] == d["id"]
        assert len(set(m["phase_ids"][ids])) == 1
        assert d["volume_mm3"] == pytest.approx(v[ids].sum())
    assert m["surface"].is_watertight
    assert m["fluid_surface"].is_watertight
    assert m["surface"].volume == pytest.approx(m["report"]["solid_volume_mm3"])
    assert m["fluid_surface"].volume == pytest.approx(m["report"]["fluid_volume_mm3"])


@pytest.mark.parametrize("mode", ["sheet", "solid_below", "solid_above"])
def test_analytic_aligned_half_box(mode, tmp_path):
    m = generate(
        Config(
            family="Custom",
            custom="X",
            mode=mode,
            gradient="uniform",
            density_start=0.5,
            phase_degrees=(0, 0, 0),
            domain_mode="solid_fluid",
            resolution=12,
        )
    )
    assert m["report"]["solid_volume_mm3"] == pytest.approx(62.5)
    assert m["report"]["fluid_volume_mm3"] == pytest.approx(62.5)
    assert m["report"]["bounds_mm"] == [[0, 0, 0], [5, 5, 5]]
    p = m["points"][m["interface"]]
    area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum() / 2
    assert area == pytest.approx(25)
    folder = save_model(m, tmp_path / "dual")
    with np.load(folder / "mesh.npz") as a:
        assert np.array_equal(a["phase_ids"], m["phase_ids"])
        assert np.array_equal(a["interface"], m["interface"])
    assert (folder / "fluid_preview.stl").is_file()
    metadata = json.loads((folder / "domains.json").read_text())
    assert metadata["boundaries"]["8"] == "solid_fluid_interface"
    nas = (folder / "mesh.nas").read_text()
    for domain in metadata["domains"]:
        assert f"PSOLID,{domain['nastran_pid']},1" in nas
    java = export_java(folder, m["report"]).read_text()
    assert "@@" not in java
    assert "solid_domains" in java and "fluid_domains" in java


def test_shared_interface_both_strategies():
    reports = [
        generate(Config(domain_mode="solid_fluid", resolution=12, quality_strategy=s))["report"]
        for s in ("pulling", "quality_fan")
    ]
    assert reports[0]["boundary_sha256"] == reports[1]["boundary_sha256"]
    assert reports[0]["solid_volume_mm3"] == pytest.approx(reports[1]["solid_volume_mm3"])
    assert reports[0]["fluid_volume_mm3"] == pytest.approx(reports[1]["fluid_volume_mm3"])


def test_invalid_domain_mode():
    with pytest.raises(ValueError):
        Config(domain_mode="wrong").validate()
