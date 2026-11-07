"""Three-grid power-law fits for arbitrary, distinct mesh spacings.

An extrapolation is a fit to f(h) = limit + C*h**p, not proof that a
sequence is asymptotic or that different meshes have the same limit.
"""

import numpy as np
from scipy.optimize import brentq


def richardson(hs, values):
    """Fit the three finest samples, returning JSON-safe diagnostics.

    Invalid input raises ValueError. Sequences without a positive fitted
    order return null estimates and a reason in ``status``.
    """
    hs, values = np.asarray(hs, dtype=float), np.asarray(values, dtype=float)
    if hs.ndim != 1 or values.shape != hs.shape or len(hs) < 3:
        raise ValueError("Provide at least three matching 1-D spacing/value pairs.")
    if not np.all(np.isfinite(hs)) or not np.all(hs > 0) or not np.all(np.isfinite(values)):
        raise ValueError("Spacings must be positive and all samples must be finite.")
    if len(np.unique(hs)) != len(hs):
        raise ValueError("Mesh spacings must be distinct.")
    order = np.argsort(hs)[:3]
    h1, h2, h3 = hs[order]
    f1, f2, f3 = values[order]
    result = {"order": None, "limit": None, "status": "nonmonotone_or_flat",
              "finest_spacings": [float(h1), float(h2), float(h3)],
              "assumption": "three-point fit: f(h)=limit+C*h**p; not a convergence proof"}
    d1, d2 = f2 - f1, f3 - f2
    if d1 == 0 and d2 == 0:
        return {**result, "limit": float(f1), "status": "constant_sequence"}
    if d1 == 0 or d2 == 0 or np.signbit(d1) != np.signbit(d2):
        return result
    a, b = np.log(h2) - np.log(h1), np.log(h3) - np.log(h2)
    target = np.log(abs(d2)) - np.log(abs(d1))

    def log_expm1(x):
        return x + np.log1p(-np.exp(-x)) if x > 50 else np.log(np.expm1(x))

    def residual(p):
        return log_expm1(p * b) - np.log(-np.expm1(-p * a)) - target

    if target <= np.log(b / a) or residual(64.0) < 0:
        return {**result, "status": "no_positive_order_in_0_to_64"}
    p = brentq(residual, 1e-10, 64.0, xtol=1e-12)
    limit = f1 - d1 / np.expm1(p * a)
    if not np.isfinite(limit):
        return {**result, "status": "nonfinite_extrapolation"}
    return {**result, "order": float(p), "limit": float(limit), "status": "fitted"}


def summarize_transport(rows):
    """Summarize successful transport rows, including CSV string values."""
    groups = {}
    for row in rows:
        if row.get("error") or row.get("k_eff_over_bulk") in (None, ""):
            continue
        key = "/".join(str(row[k]) for k in ("design", "variant", "phase"))
        groups.setdefault(key, []).append(row)
    return {
        key: richardson([float(r["h_mm"]) for r in group],
                        [float(r["k_eff_over_bulk"]) for r in group])
        for key, group in groups.items() if len(group) >= 3
    }
