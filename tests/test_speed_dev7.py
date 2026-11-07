"""Numerical decisions of the fast fan screen retain the original selector."""

import numpy as np
import pytest

from tpmslab import Config, generate
from tpmslab import quality, volume


def _reference_fan(nodes, anchor, vertices, triangles, strategy):
    baseline = [[anchor, *tri] for tri in triangles if anchor not in tri]
    if strategy == "pulling" or not baseline:
        return baseline, None, None
    vb, qb = quality._fan_metrics(nodes, baseline)
    if np.any(vb <= 0) or qb.min() >= .3:
        return baseline, None, None
    point = np.mean([nodes[i] for i in vertices], axis=0)
    vc, qc = quality._fan_metrics(nodes, triangles, point.tolist())
    if np.any(vc <= 0) or qc.min() <= qb.min() * (1 + 1e-6):
        return baseline, None, None
    if not np.isclose(vc.sum(), vb.sum(), rtol=1e-10, atol=0):
        return baseline, None, None
    return [[len(nodes), *tri] for tri in triangles], point, float(qc.min() / qb.min())


def _same_fan(left, right):
    assert left[0] == right[0]
    assert (left[1] is None) == (right[1] is None)
    if left[1] is not None:
        assert np.array_equal(left[1], right[1])
    assert left[2] == right[2]


@pytest.mark.parametrize("options", [
    {"resolution": 8},
    {"resolution": 8, "combine": "blend"},
    {"resolution": 8, "domain_mode": "solid_fluid", "snap_alpha": .25,
     "edge_interpolation": "implicit_root"},
])
def test_generated_polytope_decisions_match_reference(monkeypatch, options):
    calls = []

    def checked(*args):
        result = quality.choose_fan(*args)
        _same_fan(result, _reference_fan(*args))
        calls.append(result[1] is not None)
        return result

    monkeypatch.setattr(volume, "choose_fan", checked)
    generate(Config.from_dict(options))
    assert calls


@pytest.mark.parametrize("scalar_dot", [True, False])
def test_near_threshold_and_numpy_fallback(monkeypatch, scalar_dot):
    if not scalar_dot:
        monkeypatch.setattr(quality, "_DOT", None)
    triangles = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    # Locate the baseline mean-ratio threshold, then perturb it by very small
    # amounts so the scalar screen must leave the final decision to numpy.
    low, high = 1e-5, .5
    for _ in range(55):
        height = (low + high) / 2
        points = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [.3, .3, height]]
        _, q = quality._fan_metrics(points, [[0, 1, 2, 3]])
        if q[0] < .3:
            low = height
        else:
            high = height
    for factor in (1 - 1e-11, 1 - 1e-14, 1., 1 + 1e-14, 1 + 1e-11):
        points[-1][-1] = height * factor
        args = (points, 0, [0, 1, 2, 3], triangles, "quality_fan")
        _same_fan(quality.choose_fan(*args), _reference_fan(*args))
