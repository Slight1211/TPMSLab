# TPMS Lab 0.3.9.dev12

This pre-release publishes the development snapshot through dev12. Mesh
generation and NAS/NPZ exports use four-node tetrahedra. The optional ten-node
curved export has been removed. Experimental dev13 inversion pairing is excluded.

Compared with stable 0.3.4, this development line includes:

- Shared sampling for complementary domains, BCC background tetrahedra,
  vertex snapping, implicit/quadratic edge-root placement and periodic box faces.
- Cut-layer reconnection and smoothing, per-phase quality/volume budgets,
  high-effort optimization and an optional reported surface envelope.
- Spatial blending of TPMS families, connected-domain labels, sampling and
  wall-resolution diagnostics, and checkpointed convergence workflows.
- COMSOL attempt isolation, recorded launch settings, retained failure evidence
  and reusable import/solve preparation. COMSOL remains optional and separately licensed.

Install on Python 3.11 or later:

```sh
python -m pip install "tpmslab==0.3.9.dev12"
# Include the browser UI:
python -m pip install "tpmslab[web]==0.3.9.dev12"
```

The core `src/tpmslab` files are restored from the verified local dev12 snapshot.
Publication preparation changes release metadata, CI and benchmark/test lint only.
Historical benchmark and solver results retain their original scope and dates;
they are not new dev12 measurements. The publication workflow builds wheel/sdist,
checks metadata and tests the installed wheel before PyPI upload. It does not
run a newly licensed COMSOL solve. This release makes no universal speed,
mesh-quality or continuum-accuracy guarantee.
