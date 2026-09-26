# TPMS Lab v0.3.0

A reproducible Python workflow from graded implicit TPMS designs to complementary
solid/pore-fluid volume meshes, connected domains and solver-preparation files.

- Generate graded sheet and solid-network geometries using the documented field registry.
- Preserve shared solid-fluid interface vertices/faces and connected-domain metadata.
- Export NASTRAN volume meshes, NPZ arrays and JSON labels; STL is a preview.
- Use an optional COMSOL bridge for import, named selections, separate fixed-wall
  pore-flow / solid-elasticity examples, saving and reopening checks.
- Include tests, reference configurations, measured validation data and provenance.

Install with Python 3.11+: `python -m pip install ./tpmslab-0.3.0-py3-none-any.whl`
(or install the source distribution). The Python core needs no COMSOL licence.
COMSOL model operations require a separately installed and licensed compatible solver.
The demonstrated version is COMSOL 6.3. No PyPI publication is claimed.

The exact build commit is recorded in RELEASE_PROVENANCE.json. Historical
measurements remain attributed to baseline `686b783e0385ae40a8858f0aba15824c5d02ec92`;
they are not relabelled as newly measured results. Fresh installation/reference
checks are supplied separately. SHA256SUMS.txt verifies the release attachments.

Limitations: mesh-defined geometry, approximate density calibration, no general
mesh-quality guarantee, no smooth CAD Boolean operations and no coupled FSI.
