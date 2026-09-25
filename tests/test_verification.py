import numpy as np
import pytest
from tpmslab import Config, generate
from tpmslab.verification import audit_interface


@pytest.mark.parametrize("family", ["Gyroid", "Primitive Schwartz", "Diamond"])
def test_coarse_dual_interface(family):
    m = generate(Config(family=family, domain_mode="solid_fluid", resolution=8))
    audit = audit_interface(m)
    assert audit["valid"]
    assert audit["invalid_interface_triangles"] == 0
    assert audit["interface_area_mm2"] == pytest.approx(m["report"]["interface_area_mm2"])
    assert m["report"]["total_mesh_volume_mm3"] == pytest.approx(125)


def test_audit_detects_corrupted_interfaces():
    m = generate(Config(domain_mode="solid_fluid", resolution=8))
    assert audit_interface(m)["valid"]
    missing = dict(m, interface=m["interface"][1:])
    assert audit_interface(missing)["missing_interface_triangles"] == 1
    duplicate = dict(m, interface=np.vstack([m["interface"], m["interface"][0]]))
    assert audit_interface(duplicate)["duplicate_interface_triangles"] == 1
    wrong_phase = dict(m, phase_ids=np.ones_like(m["phase_ids"]))
    assert audit_interface(wrong_phase)["invalid_interface_triangles"] == len(m["interface"])
    assert not audit_interface(wrong_phase)["valid"]


def test_resolution_below_eight_rejected():
    with pytest.raises(ValueError):
        Config(resolution=7).validate()


def test_analytic_channel_openings():
    from tpmslab.verification import audit_openings

    mesh = generate(
        Config(
            family="Custom",
            custom="X",
            gradient="uniform",
            density_start=0.5,
            phase_degrees=(0, 0, 0),
            domain_mode="solid_fluid",
            resolution=8,
        )
    )
    channels = audit_openings(mesh, (5, 5, 5))
    assert len(channels) == 1
    assert channels[0]["inlet_area_mesh_mm2"] == pytest.approx(12.5)
    assert channels[0]["outlet_area_mesh_mm2"] == pytest.approx(12.5)


@pytest.mark.parametrize("limit", [0, -1, 1000001, 1.5, True])
def test_solver_limit_validation(limit):
    from tpmslab.comsol import build_mph

    with pytest.raises(ValueError, match="max_solve_tetrahedra"):
        build_mph(".", {}, max_solve_tetrahedra=limit)


@pytest.mark.parametrize("seconds", [0, -1, 3601, 1.5, True])
def test_build_timeout_validation(seconds):
    from tpmslab.comsol import build_mph

    with pytest.raises(ValueError, match="build_timeout_s"):
        build_mph(".", {}, build_timeout_s=seconds)
