# Solid and pore fluid validation

Version 0.3.0rc1; local validation on 2026-09-24.

77 automated tests passed in 112.68 s. Ruff passed. Wheel and sdist built; twine check passed. Installed wheel exercised in a separate virtual environment inheriting the bundled NumPy runtime; SciPy, trimesh and Flask installed separately. Windows / Python 3.12 / NumPy 2.3.5 / SciPy 1.18.1 / trimesh 5.1.0. This is not a full cross-platform certification.

Browser verification: model restoration, solid/fluid preview switching, dual-domain generation, NASTRAN download, no JavaScript exceptions, and no horizontal overflow at 390 px.

## COMSOL 6.3.0.290 demonstrations

Each model: 5 mm cube, one cell per axis, 12 samples/cell, sheet structure, Z density gradient 0.25 to 0.50, default phases (0,0,90 degrees), quality_fan. Each contained one solid and two disconnected fluid domains. Three imports verified each connected-domain volume, solid/fluid selections, interface selection and tetrahedron count; three stationary pore-flow solves were saved and reopened successfully.

Fixed-wall creeping flow: inlet Z- pressure 0.01 Pa, outlet Z+ pressure 0 Pa; other fluid boundaries no slip. Fluid density 1000 kg/m^3, viscosity 0.001 Pa s. No solid deformation or FSI coupling.

| Family | Tetrahedra | Solid mm3 | Fluid mm3 | Interface triangles | Outlet m3/s | Relative flux imbalance |
|---|---:|---:|---:|---:|---:|---:|
| Gyroid | 38292 | 49.17571054 | 75.82428946 | 7392 | 8.89101254e-10 | 1.432e-08 |
| Primitive Schwartz | 31282 | 47.92241262 | 77.07758738 | 5608 | 4.45998409e-10 | 1.817e-09 |
| Diamond | 42280 | 50.20485764 | 74.79514236 | 9072 | 4.24304079e-10 | 4.699e-06 |

The flux check confirms consistency of these runs, not mesh convergence or permeability accuracy. No boundary-layer mesh, turbulence, deformation-coupled FSI or conjugate heat-transfer validation has been performed.

## Reproduce

```sh
python benchmarks/verify_solid_fluid.py --out new_dual_results
```

COMSOL and the relevant physics licence are required. The open tests do not require COMSOL. Export an unsolved MPH to assign custom physics and materials. The fluid phase is the pore volume within the same box; no outside fluid region or entrance/exit reservoir is included. Connected pores that do not reach both Z faces are retained in the mesh but cannot use the automatic flow demo.

Selection PIDs are documented in README.md. COMSOL entity numbers can differ from mesh domain labels; the bridge explicitly resolves imported property selections rather than assuming matching numbers.
