# Local validation record

Version: 0.2.0rc1. Date: 2026-09-24. Windows / Python 3.12.

- Automated suite: 63 passed, 77.07 s. JUnit: test-results.xml.
- Static checks: ruff passed for src, tests and benchmarks.
- Build: isolated PEP 517 source distribution and wheel built successfully.
- Metadata: twine check passed for both distributions.
- Fresh environment: wheel installed without system-site-packages; Python API
  and CLI generated 18,305 tetrahedra for the example outside the source tree.
  Flask was absent during the core check. pip check reported no broken requirements.
- Dependency environments: benchmark NumPy 2.3.5, SciPy 1.18.1, trimesh 5.1.0;
  independent wheel smoke used NumPy 2.5.3, SciPy 1.18.1, trimesh 5.1.0.
- Optional UI: installed Flask in the separate wheel environment; tested
  generation, strategy selection, stale-result notice, manifest download and
  390 px mobile layout. No JavaScript exceptions or horizontal overflow.
- Geometry: 36 benchmark runs / 18 pairs; all boundary fingerprints match;
  maximum paired volume difference 7.11e-15 mm^3. Minimum Q ratio 1.012–4.840;
  element overhead 3.56–25.96%. Median Q decreases in all pairs; counts below
  Q=0.01 increase in four pairs. See the adjacent benchmark folder for all data.
- Solver: COMSOL 6.3.0.290, six N=12 cases (three families by two strategies),
  import/solve/save/reopen successful. No FE convergence certification.

Only the listed local environments were executed. CI files do not certify
Linux, all supported Python releases or the entire dependency range. Timings
were collected with other validation tasks on the same machine and must not be
used to claim a speed advantage. Whole-file hashes and the baseline files make
results inspectable; numerical byte identity across environments is not promised.
