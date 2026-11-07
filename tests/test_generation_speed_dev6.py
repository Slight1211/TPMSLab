"""Preserve connectivity and shared-coordinate validation in generation fast paths."""

import numpy as np
import pytest

from tpmslab import Config, generate
from tpmslab import multidomain
from tpmslab.volume import face_components


@pytest.mark.parametrize("order", ([0, 1, 2, 3], [3, 1, 0, 2]))
def test_face_components_keeps_vertex_and_edge_contacts_separate(order):
    # The first pair shares a triangle. Cell 2 touches that pair at a vertex;
    # cell 3 touches cell 2 along an edge. Only the first pair is one domain.
    tetra = np.array([[0, 1, 2, 3], [0, 2, 1, 4], [4, 5, 6, 7], [6, 7, 8, 9]])
    count, labels = face_components(tetra[order], n_points=10)
    original_labels = labels[np.argsort(order)]
    assert count == 3
    assert original_labels[0] == original_labels[1]
    assert len(set(original_labels[[0, 2, 3]])) == 3


@pytest.fixture(scope="module")
def dual_reference():
    config = Config(resolution=8, domain_mode="solid_fluid")
    return config, generate(config)


@pytest.mark.parametrize("tolerance_factor", [0.5, 2.0, np.nan])
def test_dual_merge_checks_every_shared_coordinate(monkeypatch, dual_reference, tolerance_factor):
    config, expected = dual_reference
    original_generate = multidomain.generate_volume
    changed = []

    def with_perturbed_shared_vertex(*args, **kwargs):
        part = original_generate(*args, **kwargs)
        if kwargs.get("_fluid_side") == 0 and part is not None:
            # A side-0 cut vertex must also occur in the previously merged
            # solid. Perturb just the final coordinate of that shared node.
            cut = next(i for i, key in enumerate(part["vertex_keys"]) if key[0] == 1)
            part = dict(part, points=part["points"].copy())
            part["points"][cut, 2] += tolerance_factor * max(config.size) * 1e-12
            changed.append(cut)
        return part

    monkeypatch.setattr(multidomain, "generate_volume", with_perturbed_shared_vertex)
    if not np.isfinite(tolerance_factor) or tolerance_factor > 1:
        with pytest.raises(ValueError, match="Solid/fluid cut vertex coordinates disagree"):
            generate(config)
    else:
        actual = generate(config)
        # The original solid coordinate is retained when a duplicate is
        # within tolerance, including the shared interface and domain labels.
        for key in ("points", "tetra", "domains", "phase_ids", "boundary", "interface"):
            assert np.array_equal(actual[key], expected[key])
    assert len(changed) == 1
