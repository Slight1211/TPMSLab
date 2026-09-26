# TPMS Lab

[English quick start](docs/QUICKSTART_EN.md) | [中文使用说明](docs/QUICKSTART_ZH.md)

An installable Python toolkit for graded TPMS-derived **solid and complementary pore-fluid volume meshes**.
Version 0.3.0. This repository is public. Source, tests, examples and versioned
reproducibility records are distributed together. Use an exact commit or the
`v0.3.0` tag, rather than a changing `main`, for reproducible work.

[Version and installation details](docs/VERSION_0.3.0.md) |
[GitHub releases](https://github.com/Slight1211/TPMSLab/releases)

A public source repository, a GitHub Release, a Zenodo archive and a PyPI upload
are separate records. Cite a DOI only after the archive is actually published.
No PyPI release has been made.

## Install

Use Python 3.11 or newer. From this source directory:

```sh
python -m pip install .
# Optional local browser interface
python -m pip install ".[web]"
```

Build a wheel locally with `python -m build` (requires the `build` package). The core does not require Flask, COMSOL, MATLAB,
OpenCascade, a CAD import licence or a graphics display. COMSOL model creation
and solving require a separate compatible COMSOL installation and licence.

## Python API

```python
from tpmslab import Config, generate, save_model

mesh = generate(Config(family="Gyroid", resolution=16,
                       density_start=0.25, density_end=0.50,
                       gradient="linear", quality_strategy="quality_fan"))
print(mesh["report"]["actual_density"])
save_model(mesh, "gyroid_result")  # destination must not exist
```

`generate` also accepts a dictionary. `list_families()` returns formula metadata.
Coordinates are millimetres; array connectivity is zero-based. The returned
mapping contains `points`, `tetra`, `boundary`, `boundary_ids`, `domains`,
`surface` (a trimesh object) and `report`. This API may change before
1.0. Configuration objects normalize sequence fields to immutable tuples.

## Command line

```sh
tpmslab init-config params.json
tpmslab generate --config params.json --out result
tpmslab families
tpmslab web --output local_models
# Requires licensed COMSOL on Windows:
tpmslab generate --config params.json --out solved_result --solve
```

Export includes `mesh.nas` (CTETRA volume elements and labelled boundary
triangles), `mesh.npz`, `preview.stl`, parameters, diagnostics, environment and a
SHA-256 manifest. Existing output directories are rejected. A failed export
can leave a partial directory; choose a new destination after addressing the
error. STL is only a boundary preview. The optional bridge imports the volume
mesh, creates an MPH and reopens it to check the embedded element count.
Set `COMSOL_BIN` to the folder containing `comsolbatch.exe` and
`comsolcompile.exe` if automatic Windows detection fails.

## Method and limits

The library samples trigonometric level-set approximations, calibrates density
against a 56 cubed periodic-cell sample and clips a six-tetrahedra-per-voxel
background mesh. Sheet structures use two separate interpolated inequalities;
they do not have constant physical wall thickness. Shared intersections and
canonical face diagonals form a conforming tetrahedral volume mesh. Material
volume, boundary closure, face incidence and orientation are checked before
export. The result is a piecewise-planar solid domain, **not editable smooth
STEP/NURBS CAD**. Mesh generation can reject difficult parameter combinations.

`quality_fan` compares the original pulling fan with an interior-centroid fan
for poor cut cells and accepts only a better worst mean-ratio quality while
preserving boundary triangles and volume. This increases element count and can
reduce median quality. It provides no angle bound, global optimization or
finite-element convergence guarantee. `pulling` retains the baseline method.

There are 29 reference labels (27 distinct expressions, with aliases), eight
density profiles, sheet and two network modes, phase controls, and a restricted
custom-formula parser. Built-in formulas are adapted from LattGen; see
`THIRD_PARTY_NOTICES.md`. They are not claimed as new TPMS families or exact
minimal surfaces. Density targets are periodic-cell calibrations, not guaranteed
finite-specimen volume fractions. Current bounds: box domains, 1–6 cells per
axis, 8–64 samples per cell and at most 200,000 background voxels. The built-in
static demo is limited to 180,000 tetrahedra and connected material. Top/bottom
contact, constraints and convergence must be checked for a research model.

## Development and reproduction

```sh
python -m pip install ".[dev]"
python -m pytest
python -m ruff check src tests benchmarks
python -m build
python -m twine check dist/*
python benchmarks/run_benchmarks.py --out benchmark_results
```

See `docs/METHOD.md`, `docs/SOLID_FLUID.md` and `docs/VALIDATION.md`.
The CI matrix is configured for Windows/Linux and Python 3.11/3.12; configuring
CI does not mean those jobs have run. The measured local environment is Windows
and Python 3.12. Optional COMSOL validation is separate from the open test suite.

## Licence and attribution

MIT. See `LICENSE`, `THIRD_PARTY_NOTICES.md` and `licenses/` for the
package licence and third-party formula and browser-library notices.

## Solid and pore-fluid domains (0.3)

```python
from tpmslab import Config, generate, save_model
from tpmslab.comsol import build_mph

mesh = generate(Config(domain_mode="solid_fluid", family="Gyroid", resolution=12))
folder = save_model(mesh, "solid_fluid_gyroid")
# Optional, licensed COMSOL: verifies selections, each domain volume and saved mesh.
build_mph(folder, mesh["report"], solve=False)
# solve=True instead runs the fixed-wall pore-flow demonstration described below.
```

In the browser, select **固体 + 孔隙流体（共享界面）** under **计算域**.
The phase selector switches the preview between solid and pore fluid.
The default remains solid-only for compatibility. The JSON/CLI parameter is
`"domain_mode": "solid_fluid"`.

Both phases partition the same bounding box. They use shared cut-edge vertices
and identical interface triangles. For sheet structures the two complementary
inequalities are meshed separately, allowing disconnected labyrinths to remain
separate domains. Connectivity is determined by shared tetrahedron faces.
The combined mesh is checked for positive volumes, opposite adjacent-face
orientation, complete interface matching and total volume conservation.

`phase_ids` is one value per tetrahedron (1=solid, 2=fluid); `domains` identifies
individual connected domains. `interface` is oriented out of the solid.
`boundary` includes the outer boundary AND the solid-fluid interface once.
Additional arrays are `solid_boundary` and `fluid_boundary`. `surface` and
`preview.stl` continue to show the solid; `fluid_surface` and `fluid_preview.stl`
show the pore fluid. The NPZ retains all phase and boundary arrays. `domains.json`
records domain-to-phase and boundary-name mappings. Report `volume_mm3` remains
solid volume; `total_mesh_volume_mm3` includes both phases.

NASTRAN PSOLID property IDs are `100 + domain_id`. Boundary PIDs 2–7 are solid
xmin/xmax/ymin/ymax/zmin/zmax; 12–17 are the corresponding fluid faces; PID 8
is the shared interface. PIDs are NOT COMSOL entity numbers. The bridge resolves
imported PID selections and creates `solid_domains`, `fluid_domains`, individual
connected-domain selections, `solid_fluid_interface`, and the nonempty exterior
face selections. It assigns demonstration solid/fluid materials separately.
An unsolved export requires no CFD physics interface and is a starting point
for the user's physics setup.

For dual-domain models, `solve=True` runs stationary **creeping flow only in the
fluid**, with fixed no-slip solid walls, pressure 0.01 Pa at fluid_zmin and 0 Pa
at fluid_zmax, and no-slip on other fluid exterior faces. Density is 1000 kg/m³
and viscosity 0.001 Pa·s. Every fluid component must reach both Z faces; otherwise
the automatic demo is refused while unsolved export remains available. This
requires the corresponding COMSOL physics licence. It reports inlet/outlet
flow and relative imbalance (the smoke-test threshold is 5%, not an accuracy
claim). It is **not deformation-coupled FSI**, not conjugate heat transfer, and
not a validated permeability prediction. Boundary layers, external reservoirs,
mesh convergence and realistic boundary conditions remain user responsibilities.
See `docs/SOLID_FLUID.md` for validation and reproducibility.

## Solid–fluid quick start

Run [the Python example](examples/solid_fluid_api.py). Installation, exported
files and optional COMSOL calculations are documented in the
[English quick start](docs/QUICKSTART_EN.md) and
[中文使用说明](docs/QUICKSTART_ZH.md).

## Independent verification

See [Verification and reproducible studies](docs/VERIFICATION.md) for density realization, independent interface incidence and area audits, channel-wise COMSOL flux checks, and refinement/performance scripts. `quality_fan` is optional local worst-element mitigation, not a general quality or solver-speed guarantee.

## Reproducibility records

See [validation/README.md](validation/README.md) for the numerical records,
configuration provenance and reproduction commands corresponding to the verified
workflow. The package provides mesh-defined simulation domains, not smooth CAD
reconstruction or coupled FSI.
