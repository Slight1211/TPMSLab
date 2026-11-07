"""Independent geometry reference for the optional exact-edge path."""

from dataclasses import replace

import pytest

from tpmslab import Config, generate


def test_inversion_safe_edge_projection_reduces_gyroid_volume_error():
    # Independent 512^3 midpoint integration of this fixed 5 mm design.
    reference_density = 0.3749257028
    baseline = Config(family="Gyroid", domain_mode="solid_fluid", resolution=16)
    linear = generate(baseline)
    projected = generate(replace(baseline, edge_interpolation="implicit_root"))
    assert projected["report"]["interface_conforming"]
    assert projected["report"]["partition_volume_error_mm3"] < 1e-8
    assert len(projected["tetra"]) == len(linear["tetra"])
    assert abs(projected["report"]["actual_density"] - reference_density) < 0.003
    assert abs(projected["report"]["actual_density"] - reference_density) < abs(
        linear["report"]["actual_density"] - reference_density
    )


def test_exact_edge_mode_requires_two_phase_export():
    with pytest.raises(ValueError, match="solid_fluid"):
        Config(edge_interpolation="implicit_root").validate()


def test_batched_quadratic_edge_correction_preserves_partition_and_density():
    reference_density = 0.3749257028
    baseline = Config(family="Gyroid", domain_mode="solid_fluid", resolution=16)
    corrected = generate(replace(baseline, edge_interpolation="quadratic_root"))
    report = corrected["report"]
    assert report["interface_conforming"]
    assert report["interface_invalid_triangles"] == 0
    assert report["partition_volume_error_mm3"] < 1e-8
    assert report["edge_root_fully_projected_fraction"] > 0.99
    assert report["domain_map"][0]["tetrahedra"] == 34831
    assert abs(report["actual_density"] - reference_density) < 0.0022


def test_batched_quadratic_mode_requires_two_phase_export():
    with pytest.raises(ValueError, match="solid_fluid"):
        Config(edge_interpolation="quadratic_root").validate()


@pytest.mark.parametrize("gradient", ["quadratic", "radial", "periodic"])
def test_batched_quadratic_mode_preserves_partition_for_other_gradients(gradient):
    config = Config(
        family="Gyroid",
        domain_mode="solid_fluid",
        resolution=8,
        gradient=gradient,
        edge_interpolation="quadratic_root",
    )
    report = generate(config)["report"]
    assert report["interface_conforming"]
    assert report["interface_invalid_triangles"] == 0
    assert abs(report["partition_volume_error_mm3"]) < 1e-8
