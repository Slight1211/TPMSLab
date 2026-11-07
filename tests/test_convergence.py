import json

import numpy as np
import pytest

from tpmslab.convergence import richardson, summarize_transport


@pytest.mark.parametrize("hs", [1 / np.array([8, 12, 16, 24]), 1 / np.array([8, 16, 32, 64])])
@pytest.mark.parametrize("p", [0.5, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("coefficient", [-2.0, 2.0])
def test_known_power_law(hs, p, coefficient):
    result = richardson(hs[::-1], (1 + coefficient * hs**p)[::-1])
    assert result["order"] == pytest.approx(p, abs=1e-8)
    assert result["limit"] == pytest.approx(1, abs=1e-10)
    assert result["status"] == "fitted"


@pytest.mark.parametrize("hs,values", [([1, 2], [1, 2]), ([0, 1, 2], [1, 2, 3]),
                                       ([1, 1, 2], [1, 2, 3]), ([1, 2, 3], [1, np.nan, 3])])
def test_bad_extrapolation_inputs(hs, values):
    with pytest.raises(ValueError):
        richardson(hs, values)


@pytest.mark.parametrize("values", [[1, 2, 1], [1, 1, 2], [1, 1.5, 1.75]])
def test_unfittable_sequences_have_no_invented_estimate(values):
    result = richardson([1, 2, 4], values)
    assert result["order"] is None
    assert result["limit"] is None
    json.dumps(result, allow_nan=False)


def test_csv_summary_skips_failed_cases():
    rows = [{"design": "known", "variant": "baseline", "phase": "solid",
             "h_mm": str(h), "k_eff_over_bulk": str(1 + 2 * h**2)} for h in [1, .5, .25]]
    rows.append({**rows[0], "k_eff_over_bulk": "", "error": "solver failed"})
    assert summarize_transport(rows)["known/baseline/solid"]["order"] == pytest.approx(2)
