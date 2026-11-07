import numpy as np
import pytest

from tpmslab import Config, generate, optimize_mesh
from tpmslab.quality import signed_volumes, tetra_metrics, unique_rows
from tpmslab.volume import TET_FACES, bcc_background, sample_lattice


def test_bcc_background_tiles_the_box_conformingly():
    n, size = np.array([3, 2, 4]), np.array([3.0, 2.0, 4.0])  # cubic voxels
    axes = [np.linspace(0, L, k + 1) for L, k in zip(size, n)]
    corners = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    extra, tets, owner = bcc_background(n, size)
    xyz = np.concatenate([corners, extra])
    signed = signed_volumes(xyz[tets])
    tets[signed < 0] = tets[signed < 0][:, [1, 0, 2, 3]]
    volumes, quality = tetra_metrics(xyz[tets])
    assert np.all(signed_volumes(xyz[tets]) > 0)
    assert np.isclose(volumes.sum(), np.prod(size))
    boundary_faces = 2 * (n[0] * n[1] + n[1] * n[2] + n[0] * n[2])
    assert len(tets) == 12 * np.prod(n) + 2 * boundary_faces
    assert np.all(owner >= 0) and np.all(owner < np.prod(n))
    # every voxel owns half of its twelve-tetrahedron share
    assert np.allclose(np.bincount(owner, weights=volumes, minlength=np.prod(n)), np.prod(size / n))
    # interior tetrahedra are congruent (0.952), boundary half tetrahedra 0.8
    assert set(np.round(quality, 3)) == {0.952, 0.8}
    faces = np.sort(tets[:, TET_FACES].reshape(-1, 3), axis=1)
    keys, counts = unique_rows(faces, return_counts=True)
    assert counts.max() == 2
    exterior = keys[counts == 1]
    p = xyz[exterior]
    on_plane = ((np.abs(p) < 1e-12) | (np.abs(p - size) < 1e-12)).all(axis=1).any(axis=1)
    assert on_plane.all()


def test_bcc_rejects_unknown_lattice():
    with pytest.raises(ValueError, match="lattice"):
        Config(lattice="fcc").validate()


@pytest.mark.parametrize("domain_mode", ["solid", "solid_fluid"])
def test_bcc_generation_is_closed_and_partitions_the_box(domain_mode):
    extra = {"edge_interpolation": "implicit_root"} if domain_mode == "solid_fluid" else {}
    config = Config(family="Gyroid", resolution=8, lattice="bcc", snap_alpha=0.3, domain_mode=domain_mode, **extra)
    model = generate(config)
    report = model["report"]
    assert report["watertight"] and report["winding_consistent"]
    assert report["nonmanifold_edges"] == 0 and report["boundary_edges"] == 0
    volumes = np.abs(signed_volumes(model["points"][model["tetra"]]))
    if domain_mode == "solid_fluid":
        assert np.isclose(volumes.sum(), np.prod(config.size), rtol=1e-10)
    density = report["actual_density"]
    assert abs(density - 0.375) < 0.05
    profile = [row["sampled_density"] for row in report["density_profile"]]
    assert profile[-1] > profile[0]


def test_bcc_snapped_cut_beats_kuhn_on_worst_cells():
    common = dict(family="Gyroid", domain_mode="solid_fluid", edge_interpolation="implicit_root")
    kuhn = generate(Config(resolution=12, snap_alpha=0.25, **common))
    bcc = generate(Config(resolution=10, lattice="bcc", snap_alpha=0.3, **common))
    qk = tetra_metrics(kuhn["points"][kuhn["tetra"]])[1]
    qb = tetra_metrics(bcc["points"][bcc["tetra"]])[1]
    assert np.quantile(qb, .01) > np.quantile(qk, .01)
    assert np.median(qb) > np.median(qk)


def test_bcc_mesh_is_accepted_by_the_optimizer():
    config = Config(family="Primitive Schwartz", resolution=8, lattice="bcc", snap_alpha=0.3,
                    domain_mode="solid_fluid", edge_interpolation="implicit_root")
    model = generate(config)
    optimized = optimize_mesh(model)
    q0 = tetra_metrics(model["points"][model["tetra"]])[1]
    q1 = tetra_metrics(optimized["points"][optimized["tetra"]])[1]
    assert q1.min() >= q0.min()
    assert optimized["report"]["config"]["lattice"] == "bcc"


def test_bcc_lattice_shared_by_both_phases():
    config = Config(family="Diamond", resolution=8, lattice="bcc", snap_alpha=0.3, domain_mode="solid_fluid",
                    edge_interpolation="implicit_root")
    lattice = sample_lattice(config)
    assert len(lattice["owner_voxel"]) == len(lattice["background"]) == 12 * 8 ** 3 + 2 * 6 * 8 ** 2


def test_boundary_snap_alpha_only_affects_box_vertices():
    from tpmslab.snapping import warp_vertices
    common = dict(family="Gyroid", resolution=8, lattice="bcc", snap_alpha=0.3, domain_mode="solid_fluid",
                  edge_interpolation="implicit_root")
    base = sample_lattice(Config(**common))
    wide = sample_lattice(Config(**common, snap_alpha_boundary=0.35))
    same = sample_lattice(Config(**common, snap_alpha_boundary=0.3))
    assert np.array_equal(base["xyz"], same["xyz"])
    moved = np.any(base["xyz"] != wide["xyz"], axis=1)
    size = np.array(Config().size)
    on_box = ((np.abs(base["xyz"]) < 1e-9) | (np.abs(base["xyz"] - size) < 1e-9)).any(axis=1)
    assert moved.any() and np.all(on_box[moved])
    assert wide["snap_info"]["vertex_snap_alpha_boundary"] == 0.35
    with pytest.raises(ValueError, match="snap_alpha_boundary"):
        Config(snap_alpha_boundary=0.5).validate()
    assert warp_vertices is not None


def test_wall_resolution_reported_for_sheets():
    model = generate(Config(family="Diamond", density_start=.15, density_end=.3, resolution=8, lattice="bcc",
                            snap_alpha=0.3))
    info = model["report"]["wall_resolution"]
    assert 0.1 < info["thickness_mm_p05"] < info["thickness_mm_median"] < 1.0
    assert info["cells_across_p05"] < 1 and any("thinnest 5%" in w for w in model["report"]["warnings"])
    assert "wall_resolution" not in generate(Config(family="Gyroid", mode="solid_below", resolution=8))["report"]


def test_high_effort_targets_more_cells():
    from tpmslab.quality import tetra_metrics
    config = Config(family="Primitive Schwartz", resolution=10, lattice="bcc", snap_alpha=0.3, snap_alpha_boundary=0.35,
                    domain_mode="solid_fluid", edge_interpolation="implicit_root")
    model = generate(config)
    standard = optimize_mesh(model)
    high = optimize_mesh(model, effort="high")
    q_std = tetra_metrics(standard["points"][standard["tetra"]])[1]
    q_high = tetra_metrics(high["points"][high["tetra"]])[1]
    assert np.quantile(q_high, .01) > np.quantile(q_std, .01) + 0.03
    assert high["report"]["optimization"]["quality_threshold"] == 0.7
    with pytest.raises(ValueError, match="effort"):
        optimize_mesh(model, effort="max")


def test_surface_envelope_bounds_deviation_and_helps_quality():
    from tpmslab.quality import tetra_metrics
    from tpmslab.surface_smoothing import surface_distances
    config = Config(family="Gyroid", resolution=10, lattice="bcc", snap_alpha=0.3, snap_alpha_boundary=0.35,
                    domain_mode="solid_fluid", edge_interpolation="implicit_root")
    model = generate(config)
    exact = optimize_mesh(model, effort="high")
    loose = optimize_mesh(model, effort="high", surface_envelope=0.005)
    q_exact = tetra_metrics(exact["points"][exact["tetra"]])[1]
    q_loose = tetra_metrics(loose["points"][loose["tetra"]])[1]
    assert q_loose.min() > q_exact.min()
    info = loose["report"]["optimization"]
    assert 0 < info["surface_deviation_max_mm"] <= 0.005 * 1.01
    surface = np.unique(loose["interface"])
    assert surface_distances(loose["points"], surface, config).max() <= 0.005 * 1.01
    assert exact["report"]["optimization"]["surface_deviation_max_mm"] < 1e-6
    with pytest.raises(ValueError, match="surface_envelope"):
        optimize_mesh(model, surface_envelope=-1.0)
