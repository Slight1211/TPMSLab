# Changelog

## 0.3.0rc1 — solid and pore-fluid domains

- Add complementary solid/fluid meshes with shared interface vertices and faces.
- Preserve independently connected solid and fluid domains, phase arrays and named boundaries.
- Add fluid preview, domain metadata and optional COMSOL named selections.
- Verify three fixed-wall pore-flow examples in COMSOL; deformation-coupled FSI is not implemented.
- Add analytical partition and multi-family interface regression tests.

## 0.2.0rc1 — local release candidate

- Introduce src-layout packaging, public Python API and command-line commands.
- Keep Flask and the browser interface optional; COMSOL remains optional.
- Add conservative interior-fan selection with unchanged external boundary.
- Add quality-tail diagnostics, deterministic mesh/boundary fingerprints,
  environment records and file-integrity manifests.
- Normalize configuration lists to immutable tuples and apply phase controls
  consistently in the field helper.
- Include reproducible comparisons, automated tests and release documentation.

This is an unpublished release candidate, not a record of prior PyPI releases.
