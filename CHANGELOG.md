# Changelog

## 0.3.0

- Package the verified complementary solid/fluid workflow and independent audits.
- Include density, refinement, interface, channel and performance records with provenance.
- Retain the validated numerical implementation from commit 686b783e0385.
- Clarify geometry/solver response stability and local quality-strategy limitations.
- GitHub release publication and DOI registration are separate external actions.


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

No PyPI release has been made; earlier rc entries describe local release candidates.
