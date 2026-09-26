# TPMS Lab 0.3.0

This version provides graded implicit TPMS generation, complementary solid and
pore-fluid tetrahedral meshes, shared interface connectivity, connected-domain
labels, NASTRAN/NPZ/JSON export and optional COMSOL preparation and verification.
It is a mesh-defined simulation-preparation workflow, not smooth CAD reconstruction.
Fixed-wall pore flow and solid elasticity are separate demonstrations; no FSI is claimed.

## Install an exact version

Use Python 3.11 or newer. Download the wheel from the published v0.3.0 release:

```sh
python -m pip install ./tpmslab-0.3.0-py3-none-any.whl
```

Alternatively, install the downloaded source archive:

```sh
python -m pip install ./tpmslab-0.3.0.tar.gz
```

The tag and release metadata identify the exact source commit. A source checkout
can also be installed with `python -m pip install .`. For the optional local web
interface, install `.[web]` from source. COMSOL is not a Python dependency:
creating, solving and reopening MPH files requires a compatible, separately
licensed COMSOL installation (the demonstrated solver is COMSOL 6.3).

## Reproducibility

Historical numerical records originate from commit
`686b783e0385ae40a8858f0aba15824c5d02ec92`, with the same numerical modules as the
0.3.0 preparation commit `70fa7207e56df929afd5ad489364b8cf35f2a4d9`.
`validation/provenance.json` preserves the module SHA-256 values. New packaging
checks and reference runs are additional checks, not replacements for old data.

From a source archive with the wheel installed:

```sh
python -m pytest tests --import-mode=importlib -q
python benchmarks/release_reference.py --source . --out fresh_gyroid
# Requires licensed COMSOL; use a different, nonexistent destination:
python benchmarks/release_reference.py --source . --out fresh_gyroid_comsol --solve
```

The release-reference script checks the installed package, module and data
provenance, the exact Gyroid reference configuration, mesh volumes, connected
domains, shared-interface audit and file export. With `--solve` it also solves,
reopens and independently audits the stored fixed-wall pore-flow solution.
The bundled measured records retain their original dates and source baseline.
A solved reference-model bundle can be attached to the release separately,
avoiding large solver models in Git history. Model bundles state whether a run
is historical or freshly generated, and include configuration, solver settings,
read-only reopening results and checksums. They require COMSOL to inspect.

## Release records

SHA256SUMS.txt verifies downloadable files. RELEASE_PROVENANCE.json identifies
the exact source commit used to build them. GitHub source archives and Python
source distributions have different contents by design: the latter includes the
package, tests, examples, documentation, benchmarks and compact validation data.

The release-preparation workflow builds from its exact commit, checks the wheel
in an isolated directory, runs tests and the reference mesh, and creates a draft
GitHub release. Publishing that draft is a separate step, allowing supplemental
MPH attachments and a Zenodo connection to be checked first. Never overwrite an
existing v0.3.0 tag or release with a different source commit.
