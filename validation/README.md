# Reproducibility records for TPMS Lab 0.3.0

These are measured records from the validated numerical implementation at
commit `686b783e0385ae40a8858f0aba15824c5d02ec92`. Version 0.3.0 retains the
same numerical Python modules; `provenance.json` records their hashes.
A version change does not turn the historical measurements into new runs.

- `summary.json`: all tables, nine-level refinement, quality and desktop timings.
- `density/density_results.json`: nine graded cases and independent field estimates.
- `existing_*_audit/audit.json`: imported interface/opening areas and individual channel fluxes.
- `reference_configs/`: exact configurations for the three saved reference specimens.
- `*_summary.csv`: plain tabular records for independent inspection.

## Reproduce

From a source checkout, install `python -m pip install ".[dev]"` then run:

```sh
python -m pytest -q
python benchmarks/density_study.py --out density_results
python benchmarks/revision_study.py --out refinement_results --solve
python benchmarks/revision_study.py --out pulling_results --resolutions 12 --repeats 1 --strategy pulling --solve
python benchmarks/revision_study.py --out extension_results --resolutions 24 28 32 36 40 --repeats 1 --solve --max-solve-tetrahedra 1000000
```

Check `python benchmarks/revision_study.py --help` for accepted argument syntax.
`--solve` requires a licensed COMSOL installation. Omit it for mesh-only runs.
Use new output directories. Timing and memory are host-dependent. Refinement
measures combined geometry/solver response stability, not rigorous FEM error.

Saved MPH files are not distributed in this small evidence set. The scripts
regenerate models and verify save/reopen behavior. The numerical records are
included in the source distribution, with a separate SHA-256 manifest. No
manuscript or submission instructions are stored in this software repository.
