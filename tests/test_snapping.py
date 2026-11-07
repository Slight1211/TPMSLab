import json

import numpy as np
import pytest
from tpmslab import Config, condition_mesh, generate, save_model
from tpmslab.snapping import warp_vertices
from tpmslab.verification import audit_interface
from tpmslab.volume import PATTERN


def lattice(n, size=(1.0, 1.0, 1.0)):
    n = np.asarray(n)
    shape = n + 1
    axes = [np.linspace(0, L, int(k) + 1) for L, k in zip(size, n)]
    xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    ijk = np.indices(tuple(n)).reshape(3, -1).T
    offsets = np.array([[j & 1, (j >> 1) & 1, (j >> 2) & 1] for j in range(8)])
    corners = ijk[:, None, :] + offsets
    ids = (corners[:, :, 0] * shape[1] + corners[:, :, 1]) * shape[2] + corners[:, :, 2]
    return xyz, ids[:, PATTERN].reshape(-1, 4)


def scalar_volumes(points, cells):
    return np.array([np.dot(points[b] - points[a],
                            np.cross(points[c] - points[a], points[d] - points[a])) / 6
                     for a, b, c, d in cells])


def test_alpha_zero_is_identity_and_default():
    assert Config().snap_alpha == 0
    xyz, background = lattice((2, 2, 2))
    fields = np.array([xyz[:, 2] - 0.49])
    out_xyz, out_fields, info = warp_vertices(xyz, fields, background, (1, 1, 1), 0.0)
    assert np.array_equal(out_xyz, xyz) and np.array_equal(out_fields, fields)
    assert info["snapped_vertices"] == 0


def test_invalid_alpha_rejected():
    with pytest.raises(ValueError):
        Config(snap_alpha=0.5).validate()
    with pytest.raises(ValueError):
        Config(snap_alpha=-0.1).validate()
    with pytest.raises(ValueError):
        Config(snap_alpha=0.25, snap_target="exact").validate()


def test_plane_close_to_lattice_layer_snaps_that_layer_only():
    # Plane z = 0.52 on a 2-cell lattice (spacing 0.5): the z=0.5 layer lies 0.04
    # from the cut, i.e. 0.08 of an edge, inside alpha=0.25 -> it moves to z=0.52.
    xyz, background = lattice((2, 2, 2))
    fields = np.array([0.52 - xyz[:, 2]])
    out_xyz, out_fields, info = warp_vertices(xyz, fields, background, (1, 1, 1), 0.25)
    layer = np.isclose(xyz[:, 2], 0.5)
    assert info["snapped_vertices"] == layer.sum()
    assert np.allclose(out_xyz[layer, 2], 0.52)
    assert np.all(out_fields[0, layer] == 0)
    assert np.array_equal(out_xyz[~layer], xyz[~layer])
    # x/y never change for a move along a lattice edge in z.
    assert np.array_equal(out_xyz[:, :2], xyz[:, :2])
    assert np.all(scalar_volumes(out_xyz, background) > 0)


def test_outer_box_nodes_never_leave_their_face():
    # A tilted plane produces cuts near box nodes along edges that leave the box
    # faces; those nodes must stay put, interior nodes may move.
    xyz, background = lattice((3, 3, 3))
    fields = np.array([0.41 - 0.3 * xyz[:, 0] - 0.2 * xyz[:, 1] - xyz[:, 2]])
    out_xyz, _, info = warp_vertices(xyz, fields, background, (1, 1, 1), 0.3)
    assert info["snapped_vertices"] > 0
    for axis in range(3):
        for value in (0.0, 1.0):
            on_face = np.isclose(xyz[:, axis], value)
            assert np.allclose(out_xyz[on_face, axis], value)
    assert np.all(scalar_volumes(out_xyz, background) > 0)


def test_two_sheet_fields_keep_sum_positive_at_snapped_nodes():
    xyz, background = lattice((3, 3, 3))
    f = np.sin(2 * np.pi * xyz[:, 0]) + np.cos(2 * np.pi * xyz[:, 2]) * 0.3
    q = 0.4
    fields = np.array([q - f, q + f])
    out_xyz, out_fields, info = warp_vertices(xyz, fields, background, (1, 1, 1), 0.25)
    assert info["snapped_vertices"] > 0
    moved = np.flatnonzero(np.any(out_xyz != xyz, axis=1))
    assert np.all(out_fields[:, moved].sum(axis=0) > 0)
    assert np.all(np.any(out_fields[:, moved] == 0, axis=0))


@pytest.fixture(scope="module")
def pair():
    config = Config(family="Diamond", density_start=.15, density_end=.3,
                    resolution=10, domain_mode="solid_fluid")
    base = generate(config)
    snapped = generate(Config(**{**config.to_dict(), "snap_alpha": 0.25}))
    return config, base, snapped


def test_dual_phase_snapping_is_conforming_and_volume_preserving(pair):
    config, base, snapped = pair
    assert snapped["report"]["vertex_snapping"]["snapped_vertices"] > 0
    assert audit_interface(snapped)["valid"]
    volumes = scalar_volumes(snapped["points"], snapped["tetra"])
    assert np.all(volumes > 0)
    assert volumes.sum() == pytest.approx(np.prod(config.size), rel=1e-9)
    # Independently computed boundary nodes still lie on their box faces.
    tol = max(config.size) * 1e-9
    for axis in range(3):
        for side, value in enumerate((0.0, config.size[axis])):
            nodes = np.unique(snapped["boundary"][snapped["boundary_ids"] == 2 + axis * 2 + side])
            assert np.all(np.abs(snapped["points"][nodes, axis] - value) < tol)
    assert np.all(snapped["points"] >= -tol) and np.all(snapped["points"] <= np.asarray(config.size) + tol)


def test_snapping_removes_sliver_tail_on_thin_diamond_sheet(pair):
    _, base, snapped = pair
    def quality(model):
        p = model["points"][model["tetra"]]
        v = scalar_volumes(model["points"], model["tetra"])
        lengths = sum(np.sum((p[:, i] - p[:, j]) ** 2, axis=1)
                      for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
        return 12 * (3 * v) ** (2 / 3) / lengths
    qb, qs = quality(base), quality(snapped)
    assert np.sum(qb < 0.01) > 0
    assert np.sum(qs < 0.01) < np.sum(qb < 0.01) / 4
    assert qs.min() > 5 * qb.min()
    assert np.quantile(qs, 0.01) > np.quantile(qb, 0.01)
    assert len(snapped["tetra"]) < len(base["tetra"])


def test_snapped_mesh_round_trips_and_conditions(pair, tmp_path):
    config, _, snapped = pair
    folder = save_model(snapped, tmp_path / "snapped")
    report = json.loads((folder / "report.json").read_text(encoding="utf8"))
    assert report["config"]["snap_alpha"] == 0.25
    assert report["vertex_snapping"]["snapped_vertices"] > 0
    data = np.load(folder / "mesh.npz")
    assert np.array_equal(data["tetrahedra"], snapped["tetra"])
    conditioned = condition_mesh(snapped, max_nodes=64, sweeps=1)
    assert audit_interface(conditioned)["valid"]
    assert np.array_equal(conditioned["tetra"], snapped["tetra"])


def test_solid_only_mode_accepts_snapping():
    model = generate(Config(resolution=8, snap_alpha=0.25))
    assert model["report"]["vertex_snapping"]["snapped_vertices"] > 0
    assert model["report"]["watertight"]
    assert np.all(scalar_volumes(model["points"], model["tetra"]) > 0)


def test_snapping_combined_with_implicit_root_projection(pair):
    config, base, snapped = pair
    from tpmslab.model import field_and_threshold
    combined = generate(Config(**{**config.to_dict(), "snap_alpha": 0.25,
                                  "edge_interpolation": "implicit_root"}))
    report = combined["report"]
    assert report["vertex_snapping"]["snapped_vertices"] > 0
    assert report["edge_root_unbracketed_cuts"] == 0
    assert audit_interface(combined)["valid"]
    assert np.all(scalar_volumes(combined["points"], combined["tetra"]) > 0)
    nodes = np.unique(combined["interface"])
    xyz = combined["points"][nodes]
    value, threshold = field_and_threshold(config, xyz[:, 0], xyz[:, 1], xyz[:, 2])
    residual = np.min(np.abs(np.stack((threshold - value, threshold + value))), axis=0)
    # Every interface node lies on an implicit sheet surface (cut roots and snapped roots).
    assert np.quantile(residual, 0.99) < 1e-8
    quality_base = quality_of(base)
    quality_combined = quality_of(combined)
    assert np.sum(quality_combined < 0.01) < np.sum(quality_base < 0.01) / 4


def quality_of(model):
    p = model["points"][model["tetra"]]
    v = scalar_volumes(model["points"], model["tetra"])
    lengths = sum(np.sum((p[:, i] - p[:, j]) ** 2, axis=1)
                  for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    return 12 * (3 * v) ** (2 / 3) / lengths


def test_browser_endpoint_accepts_and_validates_snap_alpha(tmp_path, monkeypatch):
    import time
    monkeypatch.setenv("TPMSLAB_OUTPUT_DIR", str(tmp_path))
    from tpmslab.web import app, TOKEN
    client = app.test_client()
    headers = {"X-TPMS-Token": TOKEN}
    bad = client.post("/api/generate", json={**Config().to_dict(), "snap_alpha": 0.9}, headers=headers)
    assert bad.status_code == 400 and "snap_alpha" in bad.json["error"]
    good = client.post("/api/generate", json={**Config(resolution=8).to_dict(), "snap_alpha": 0.25},
                       headers=headers)
    assert good.status_code == 200
    key = good.json["id"]
    for _ in range(180):
        job = client.get("/api/job/" + key, headers=headers).json
        if job["status"] != "working":
            break
        time.sleep(0.5)
    assert job["status"] == "ready", job
    assert job["report"]["config"]["snap_alpha"] == 0.25
    assert job["report"]["vertex_snapping"]["snapped_vertices"] > 0
