import pytest
from tpmslab import Config, generate, __version__

@pytest.mark.parametrize("domain_mode", ["solid", "solid_fluid"])
def test_report_identifies_running_generator(domain_mode):
    mesh = generate(Config(resolution=8, domain_mode=domain_mode))
    assert mesh["report"]["generator_version"] == __version__
