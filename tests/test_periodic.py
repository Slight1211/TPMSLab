import pytest

from tpmslab import Config, generate, optimize_mesh
from tpmslab.periodic import check_periodic, periodic_axes


def _config(**extra):
    base = dict(family="Gyroid", resolution=8, lattice="bcc", snap_alpha=0.3, snap_alpha_boundary=0.35,
                domain_mode="solid_fluid", edge_interpolation="implicit_root", periodic_faces=True)
    base.update(extra)
    return Config(**base)


def test_axes_follow_the_gradient():
    assert periodic_axes(_config()) == [0, 1]              # linear grading along z
    assert periodic_axes(_config(gradient="uniform")) == [0, 1, 2]
    assert periodic_axes(_config(gradient="linear", axis="x")) == [1, 2]


@pytest.mark.parametrize("lattice", ["kuhn", "bcc"])
def test_generated_faces_match_exactly(lattice):
    model = generate(_config(lattice=lattice, gradient="uniform"))
    info = model["report"]["periodic"]
    assert info["matched"] and info["axes"] == ["x", "y", "z"]
    again = check_periodic(model, [0, 1, 2], model["report"]["config"]["size"], tol=1e-12)
    assert again["matched"]
    assert all(v["paired_nodes"] > 0 and v["face_triangles"] > 0 for v in again["per_axis"].values())


def test_optimisation_keeps_periodic_faces():
    model = optimize_mesh(generate(_config(family="Diamond")))
    assert model["report"]["periodic"]["matched"]
    moves = sum(h.get("periodic_pair_moves", 0) for h in model["report"]["optimization"]["history"])
    assert moves > 0


def test_non_periodic_field_is_rejected():
    with pytest.raises(ValueError, match="periodic_faces"):
        generate(Config(family="Custom", custom="X+Y+Z", mode="solid_below", resolution=8, periodic_faces=True))
