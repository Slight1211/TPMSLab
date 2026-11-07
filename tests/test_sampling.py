import pytest

from tpmslab import Config, generate


@pytest.mark.parametrize("domain", ["solid", "solid_fluid"])
def test_aliased_custom_field_is_rejected_before_export(domain):
    config = Config(family="Custom", custom="sin(8*X)", gradient="uniform",
                    density_start=.25, density_end=.25, phase_degrees=(0, 0, 0),
                    resolution=8, domain_mode=domain)
    with pytest.raises(ValueError, match="under-resolved.*off-grid"):
        generate(config)


def test_sampling_check_preserves_an_aligned_analytic_field():
    model = generate(Config(family="Custom", custom="X", gradient="uniform",
                            density_start=.5, phase_degrees=(0, 0, 0), resolution=8))
    assert model["report"]["actual_density"] == pytest.approx(.5)
    assert model["report"]["sampling_audit"]["performed"]
