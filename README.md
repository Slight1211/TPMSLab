# TPMS Lab 0.3.9.dev12: Python to COMSOL workflow

Mesh export uses four-node tetrahedra.

This is the 0.3.9.dev12 pre-release. Install this exact version with
`python -m pip install "tpmslab==0.3.9.dev12"` (or use `[web]` for the local UI).
It includes the development improvements through dev12 and excludes dev13's
experimental phase-exchanging inversion pairing.
See [release notes](docs/RELEASE_0.3.9.dev12.md) for scope and validation.

New in dev11: optional `surface_envelope` (mm) in `optimize_mesh` / `--surface-envelope`:
interface nodes may leave the exact surface by a bounded, reported distance for
higher worst-cell quality. Next steps and project state: [docs/HANDOFF_ZH.md](docs/HANDOFF_ZH.md).


New in dev10: `optimize_mesh(model, effort="high")` / `--optimize-effort high`
targets all cells below quality 0.7 (BCC solid p01 about 0.57 median, up to 0.66;
higher than fTetWild in 12 of 16 benchmark cases at 2–5x less time).


New in dev9: `snap_alpha_boundary` (box-layer snapping, 0.35 with BCC),
`periodic_faces=True` (identical opposite faces for
periodic conditions) and a wall-resolution report. Recommended BCC preset:
`Config(lattice="bcc", resolution=10, snap_alpha=0.3, snap_alpha_boundary=0.35,
domain_mode="solid_fluid", edge_interpolation="implicit_root")` + `optimize_mesh` (add `effort="high"` for the best quality).
Side-by-side results against fTetWild, TetGen, Gmsh and microgen are in
[the comparison](docs/COMPARISON_ZH.md). COMSOL checks of the new outputs:
`Run-Dev9Validation.cmd`.


New in dev8: optional body-centred cubic background lattice,
`generate(Config(..., lattice="bcc", resolution=10, snap_alpha=0.3,
edge_interpolation="implicit_root"))` or `tpmslab generate --lattice bcc`.
All background tetrahedra are congruent (mean ratio 0.952, dihedral angles
60°/90°). Use about 5/6 of the Kuhn resolution for a similar element count.
Measured against Kuhn in [the comparison](docs/COMPARISON_ZH.md) (BCC section).
The default stays `lattice="kuhn"` and reproduces dev7 arrays exactly.

Start with [Python → COMSOL instructions](docs/PYTHON_COMSOL_ZH.md).
Generate once with `generate` / `save_model`, then use `run_comsol(saved_folder)`
or `python -m tpmslab comsol --source saved_folder`. Each attempt preserves its
own inputs, settings, logs and MPH files; the current status never inherits an
earlier successful solve. Mesh inputs are checked against an immutable manifest.
This build also fixes nonuniform-grid Richardson fits and empty-cut projection,
adds bounded off-grid checks for custom fields, and checkpoints convergence runs.
An explicit `preferences_directory` is copied per attempt; global COMSOL preferences
remain separate. Short native scratch/recovery paths avoid Windows path-length
failures. See the [tool comparison](docs/COMPARISON_ZH.md) for scope and limitations.

The sections below describe the previous development stages. Historical test
counts and numerical claims must be read with the corrected convergence notes.

## Previous 0.3.8.dev0: pre-cut vertex snapping

Local improvement build, 2026-10-29. [中文说明、实测结果与复现](docs/VERTEX_SNAPPING_ZH.md).
Opt in with `generate(Config(..., snap_alpha=0.25))` or `tpmslab generate --snap-alpha 0.25`.
Lattice vertices within `snap_alpha` of a cut point are moved onto the isosurface before
clipping, so sliver cut cells are not created; outer-box nodes stay on their faces, moves that
would invert, flatten or pinch are reverted, and the complementary phases keep an identical
interface. On the 12 study designs aggregate Q<0.01 cells went 2710 -> 1 and tetrahedra
489696 -> 211328, at the cost of a 10-25 % higher interface RMS distance to the implicit
surface and an unchanged-median but worse-worst-case solid-density error (3.66 -> 4.03 pp).
Combined with `edge_interpolation="implicit_root"` the interface nodes all lie on the implicit
surface and the median solid-density error halves (1.52 -> 0.69 pp) at the same quality tail.
Default `snap_alpha=0` reproduces the 0.3.7 arrays exactly. No solver claim is made.

## Previous 0.3.7.dev0 development: audited interior mesh conditioning

Local improvement build, 2026-10-28. [中文改进说明、实测结果与使用方法](docs/QUALITY_CONDITIONING_ZH.md).
Opt in with `generate(config, condition=True)`, `tpmslab generate --condition`,
or the browser's **优化内部网格** checkbox. Boundary/interface vertices, cells,
node count and labels remain fixed. Per-phase quality/volume gates retain only
qualified candidates; failed candidates are explicitly rolled back. No new runtime
dependency is required. This is local smoothing, not a new remeshing algorithm.

The 12 selected paired meshes reduced aggregate Q<0.01 counts from 2710 to 1476;
11 candidates passed and one was rolled back. All independent geometry audits and
the full-box scalar diffusion patch checks passed. Iterative solver counts changed
in both directions; no general speedup or actual pore-flow accuracy is established.
Raw case records and meshes are in `quality_validation_20261028/` in this delivery.
The original generator defaults are unchanged; conditioning is optional.

## Previous combined TPMS development

This local development version adds spatial transitions between two TPMS families. See [Combined TPMS](docs/COMBINED_TPMS.md) and [the Python example](examples/combined_tpms.py). Published 0.3.4 and frozen benchmark sources remain separate.

# TPMS Lab

[English quick start](docs/QUICKSTART_EN.md) | [中文使用说明](docs/QUICKSTART_ZH.md)

An installable Python toolkit for graded TPMS-derived **solid and complementary pore-fluid volume meshes**.
This directory contains the 0.3.9.dev12 pre-release. The previous stable
baseline is [TPMS Lab 0.3.4 on PyPI](https://pypi.org/project/tpmslab/0.3.4/)
and [source commit 1e1a560](https://github.com/Slight1211/TPMSLab/tree/1e1a56021277556593a6e4925b91b78b93c77fb2).
This version includes spatial TPMS blending,
implicit-edge projection during two-phase mesh generation, and optional COMSOL
quality remeshing after export. The default configuration keeps blending disabled.

[Quality-remeshing details](docs/QUALITY_REMESH.md) |
[GitHub releases](https://github.com/Slight1211/TPMSLab/releases)

A public source repository, a GitHub Release, a Zenodo archive and a PyPI upload
are separate records. Cite a DOI only after the archive is actually published.
See [0.3.4 patch notes](docs/RELEASE_NOTES_v0.3.4.md).

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

For an experimental geometry-accuracy option in two-phase mode, set
`edge_interpolation="implicit_root"` in `Config`. It moves shared cut vertices
to roots of the original implicit field, accepting only moves that keep all
incident tetrahedra positive. The default is `"linear"`. In the nine tested
`n=16` graded designs, the option reduced median absolute density error from
0.971 to 0.216 percentage points at unchanged tetrahedron counts, but it did
not repair the low-quality element tail. It is not yet validated at all sizes
or for every field and should be checked against an independent geometry
reference for the intended design.

The additional experimental `"quadratic_root"` option uses three batched
field evaluations per cut edge and a quadratic crossing instead of an
iterative scalar root solve; it applies the same positive-tetrahedron guard.
In nine `n=16` graded designs, median density error was 0.206 percentage
points versus 0.216 for `"implicit_root"` and 0.971 for `"linear"`, at
unchanged element counts. In three repeated central-gradient timing cases it
was 16–19% faster than iterative roots, but 5–6% slower than linear cuts.
These are local exploratory measurements. The quality tail remains and
global density error does not bound local surface or solver error.

### Cut-layer optimisation (`optimize_mesh`)

```python
from tpmslab import Config, generate

mesh = generate(Config(family="Gyroid", resolution=12, domain_mode="solid_fluid",
                       snap_alpha=0.25, edge_interpolation="implicit_root"), optimize=True)
print(mesh["report"]["optimization"]["p01_before"], mesh["report"]["optimization"]["p01_after"])
```

`optimize=True` (CLI `--optimize`, browser checkbox) runs three local steps on the
generated mesh: 2-3 / 3-2 / 4-4 reconnection of interior faces and edges of the
cut layer, maximin smoothing of interior nodes, and sliding of boundary nodes
within the box faces and along the *exact implicit level set* (each move is
re-projected onto the zero set). Box faces and the interface may be
retriangulated; interface cap flips can transfer cells between phases. The two
phases remain conforming, and the result is audited before it is returned.

Starting with dev6, surface moves check the sampled chord-error criterion
against the current neighbouring geometry, with `chord_tolerance=0.0` by
default. Each round must preserve each phase's minimum quality, p01 and
low-quality volume fraction; median quality may lose at most 15% relative to
the input. The cumulative phase volume shift is limited to 0.3 percentage
points of box volume. A rejected round returns the last accepted mesh, with
the reason recorded in `report["optimization"]`. These configurable budgets
are geometry checks, not a bound on finite-element error. For tighter control:

```python
from tpmslab import optimize_mesh

mesh = optimize_mesh(generate(config), max_phase_volume_shift_pp=0.05)
# To keep all boundary vertices and phase volumes fixed:
mesh = optimize_mesh(generate(config), surface=False)
```

Explicit `chord_tolerance=None` disables the hard chord gate, and
`max_phase_volume_shift_pp=None` disables the volume budget. Final density
profiles are recomputed from the exported cells; `slab_count` supplies the
weights for unequal-width bins. Optimisation clears input COMSOL solve
evidence because it applies to the previous mesh.

Dev7 also checks the worst incident quality separately for each phase before
accepting a mixed-phase vertex move. This prevents a local gain in one phase
from degrading the other phase and wasting a whole optimisation round. The
existing chord, volume and round-level quality budgets are unchanged. More
rounds can consequently be retained, so a better final quality may require
more optimisation time for some designs. Per-round
`surface_phase_guard_rejected_trials` records the local filtering.

Generation reuses exactly ordered shared-face SVD results and screens clearly
unhelpful fan candidates with scalar arithmetic; close decisions and accepted
fans retain the original NumPy scoring. Reconnection caches cell metrics only
while coordinates are fixed. Surface chord checks use actual incident faces
and skip candidates that cannot pass the quality tests.

`benchmarks/quality_vs_gmsh.py` compares the result with Gmsh on the same
boundary polyhedron (numbers in `docs/COMPARISON_ZH.md`).

### Optional quality remeshing in COMSOL

With a licensed COMSOL Windows installation (6.2.0.290 verified; the 6.3+ mesh API is selected by reflection), a verified `solid_fluid` export can
be passed to COMSOL's existing face and tetrahedral remeshing operations:

```python
from tpmslab import remesh_in_comsol

evidence = remesh_in_comsol(
    "gyroid_result", "gyroid_quality", volume_size=7,
    preferences_directory="evidence/authorized_preferences",  # optional, as for run_comsol
)
print(evidence["quality_p01_solid_before"], evidence["q01_solid"])
```

Scratch, recovery and workspace paths are short temporary directories recorded
in `runtime.json`; `temp_directory` is accepted for compatibility only. This
mode has not yet been exercised in the shipped evidence.
The COMSOL step can be applied to either interpolation mode. The new directory contains `quality_remeshed.mph`, logs, and
`quality_remesh_verification.json`. The function checks saved/reopened domain and
boundary selections, exact incidence of every cross-phase tetrahedral face
against the declared shared-interface triangles, the total box volume,
phase-domain volume drift, interface-area drift, and improvement in
first-percentile solid-tetrahedron quality.
`volume_size` and optional `face_size` are COMSOL's 1--9 size presets, not
physical millimetres. Choose settings for the application and inspect the
reported geometric changes; quality improvement does not imply improved
density fidelity or solver convergence. An audit failure raises and retains
its output for inspection. This is a COMSOL-dependent preparation step, not
a new meshing algorithm or part of the Python-only generation path.

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
tpmslab remesh-quality --source gyroid_result --out gyroid_quality --volume-size 7
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
finite-specimen volume fractions. Current bounds: box domains, positive integer cell counts per
axis and at least 8 samples per cell, with no fixed upper limits. There is no fixed background-voxel count cap; feasible grid sizes depend on available memory. The built-in
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

### Density calibration sampling

`Config(m_cal=80)` sets the number of midpoint samples per axis used to calibrate density to threshold (80 cubed samples). The default is 56; values must be integers of at least 8, with no fixed upper bound. This parameter is independent of mesh `resolution`. JSON configurations use the same `m_cal` key; older configurations default to 56. Exported configuration records include the selected value. Increasing it changes density calibration and can change the generated geometry. Historical validation used 56.
