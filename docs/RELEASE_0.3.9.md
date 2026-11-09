# TPMS Lab 0.3.9

Version 0.3.9 is the stable release of the 0.3.9.dev12 snapshot. Mesh generation
and NAS/NPZ exports use four-node tetrahedra.

Changes since 0.3.4 include:

- Shared sampling for complementary domains, BCC background tetrahedra,
  vertex snapping, implicit and quadratic edge-root placement, and periodic box faces.
- Cut-layer reconnection and smoothing, per-phase quality and volume budgets,
  high-effort optimization, and an optional surface envelope with reported deviation.
- Spatial blending of TPMS families, connected-domain labels, wall-resolution
  diagnostics, and checkpointed convergence workflows.
- Isolated COMSOL attempts with recorded settings, retained logs and failure
  evidence, and reusable import and solve preparation. COMSOL requires a
  separate compatible installation and licence.

Install with Python 3.11 or later.

```sh
python -m pip install "tpmslab==0.3.9"
# Include the browser UI
python -m pip install "tpmslab[web]==0.3.9"
```

The release workflow builds the wheel and source distribution, validates their
metadata, and runs the tests against the installed wheel before publishing.
Historical benchmark and COMSOL results retain the versions and conditions
recorded with each experiment.
