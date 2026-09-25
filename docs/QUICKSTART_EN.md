# TPMS Lab: English quick start

TPMS Lab is a Python package for graded TPMS-derived solid and complementary pore-fluid volume meshes. Version 0.3.0rc1 is a release candidate, not a published PyPI release.

## 1. Install

Use Python 3.11 or later. Clone the repository using a GitHub account with access if the repository is private:

```sh
git clone https://github.com/Slight1211/TPMSLab.git
cd TPMSLab
python -m pip install .
```

The core requires NumPy, SciPy and trimesh. COMSOL is optional and separately licensed. Never put a GitHub access token directly in the installation command.

## 2. Run the example

```sh
python examples/solid_fluid_api.py
```

This creates a new timestamped folder under `results/`. The default example is a 5 mm Gyroid cube, with target density increasing linearly from 0.25 to 0.50 along z and resolution 12. Its verified reference configuration contains 38,292 tetrahedra, one connected solid domain and two disconnected pore-fluid domains.

## 3. Call the Python API

```python
from datetime import datetime
from pathlib import Path
from tpmslab import Config, generate, save_model

config = Config(
    family="Gyroid", mode="sheet", domain_mode="solid_fluid",
    size=(5.0, 5.0, 5.0), cells=(1, 1, 1),
    density_start=0.25, density_end=0.50,
    gradient="linear", axis="z", resolution=12,
    quality_strategy="quality_fan",
)
model = generate(config)
output = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
folder = save_model(model, output)
print(folder / "mesh.nas")
```

Coordinates and dimensions are in millimetres. Connectivity arrays use zero-based indices. `save_model` requires a new directory and will not overwrite an existing result. To inspect available families, run `tpmslab families` or call `tpmslab.list_families()`.

## 4. Create or solve a COMSOL model

On Windows, install COMSOL and the licences needed for the requested physics. The verified environment is COMSOL 6.3.0.290. If detection fails, set `COMSOL_BIN` to the directory containing `comsolbatch.exe` and `comsolcompile.exe`.

```python
from tpmslab.comsol import build_mph

# Create an unsolved MPH with phase materials and named selections:
build_mph(folder, model["report"], solve=False)

# Alternatively, create and solve the fixed-wall pore-flow demonstration:
build_mph(folder, model["report"], solve=True)
```

Equivalent commands, each generating its own new result folder:

```sh
python examples/solid_fluid_api.py --comsol create
python examples/solid_fluid_api.py --comsol solve
```

The dual-domain solver demonstration applies Creeping Flow to the fluid domains, with 0.01 Pa at the lower z openings, 0 Pa at the upper openings, and fixed no-slip walls. It does not solve solid deformation or coupled fluid-structure interaction. The automatic example requires every fluid component to reach both z faces. The bridge checks imported domains, saves the model and reopens it to verify stored results.

## 5. Output files

| File | Purpose |
|---|---|
| `mesh.nas` | Tetrahedral volume mesh and labelled boundary triangles for import |
| `mesh.npz` | NumPy mesh arrays, phase identifiers and shared interface |
| `preview.stl`, `fluid_preview.stl` | Solid and fluid surface previews |
| `config.json`, `report.json`, `domains.json` | Parameters, diagnostics and domain/boundary mapping |
| `environment.json`, `manifest.json` | Dependency versions and file checksums |
| `model_unsolved.mph` | Optional COMSOL model before solution |
| `model_solved.mph` | Optional COMSOL model containing the demonstration solution |

STL files are previews. The simulation domain is mesh-defined; the program does not export a smooth editable STEP/NURBS CAD solid.

## 6. Optional local web interface

```sh
python -m pip install ".[web]"
tpmslab web --output tpmslab-output
```

Open the local URL printed by the command. Mesh generation remains local.

## 7. Scope and validation

The included checks establish mesh partitioning, matching interfaces, domain-aware exports and the demonstrated COMSOL workflows. They do not establish mesh-converged permeability or stress, universal element quality, geometry-level Boolean operations, or deformation-coupled fluid-structure interaction. Increasing resolution increases memory and computational cost. See `docs/SOLID_FLUID.md`, `docs/VALIDATION.md` and `docs/METHOD.md` for validation records, methods and limitations.
