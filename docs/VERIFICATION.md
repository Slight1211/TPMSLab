# Verification and reproducible studies

The independent audits supplement the generator's own checks. They reconstruct
face incidence from tetrahedra rather than accepting report flags.

```python
from tpmslab import Config, generate, save_model
from tpmslab.verification import audit_interface, audit_openings

mesh = generate(Config(family="Gyroid", domain_mode="solid_fluid",
                       resolution=12, gradient="linear",
                       density_start=0.25, density_end=0.50))
assert audit_interface(mesh)["valid"]
print(audit_openings(mesh, mesh["report"]["config"]["size"]))
folder = save_model(mesh, "verified_gyroid")
```

With a licensed COMSOL installation, save/reopen and read-only audits are available:

```python
from tpmslab.comsol import build_mph
from tpmslab.comsol_audit import audit_saved_flow

build_mph(folder, mesh["report"], solve=True)
audit = audit_saved_flow(
    folder / "model_solved.mph",
    mesh["report"]["interface_area_mm2"],
    mesh["report"]["domain_map"],
    folder / "independent_audit",  # must be new
)
print(audit["channels"])
```

The audit verifies imported interface area, each channel's inlet/outlet signs and
flux balance, and opening areas against the mesh when `mesh.npz` is alongside
the MPH. It reads stored solutions and does not overwrite the source model.
A SHA-256 fingerprint identifies the source MPH. Mesh labels and COMSOL entity
numbers are distinct; both are reported.

## Reproduce numerical studies

Run from a source checkout after `python -m pip install ".[dev]"`:

```sh
python -m pytest -q
python benchmarks/density_study.py --out density_results
python benchmarks/revision_study.py --out refinement_results --solve
python benchmarks/revision_study.py --out pulling_results --resolutions 12 --repeats 1 --strategy pulling --solve
```

Omit `--solve` to run the generation/export benchmark without COMSOL. Default
resolutions are 8, 12, 16 and 20, with three fresh processes per complementary
case. Generation and export are timed separately. Peak process RSS includes
Python/imports and excludes COMSOL and the independent post-export audit.
Measurements depend on the host and should not be treated as cross-tool timings.

The density study generates Gyroid, Primitive Schwartz and Diamond at n=12,
with three linear profiles per family. It records exact tetrahedral phase
volumes, grid-aligned slab fractions and independent midpoint samples of the
unlinearized field at 128 cubed and 256 cubed. Target density is a periodic-cell
calibration, not an exact finite-specimen volume constraint.

Refinement uses both phase volumes, outlet flow and separate solid-only strain
energy, with successive relative change `abs(new-old)/abs(new)`. The geometry
and finite-element discretization change together. This is not an isolated
finite-element error estimate. Inspect every reported response before declaring
a stopping criterion met. Individual saved-model flags remain workflow checks,
not convergence certification.

To extend a research study explicitly beyond the default demo size limit:

```sh
python benchmarks/revision_study.py --out refinement_24 --resolutions 24 --repeats 1 --solve --max-solve-tetrahedra 1000000
```

The normal limit remains 180,000 tetrahedra. An explicit research override is
bounded at 1,000,000; sufficient memory and licensed solver access are required.
All output destinations must be new. A failed run may leave partial artifacts;
retain its logs and use a new destination when retrying.

## Measured evidence, 2026-09-25

The updated local suite passed 88 tests (Windows, Python 3.12.14). The n=12
reference cases have zero invalid, missing or duplicated interface triangles.
Imported interface-area relative differences are at most 1.02e-14; the six
individual pore-channel flux imbalances are at most 1.06e-5. These checks
establish preservation of the discretized domains, not exact smooth geometry.

For the reference nominal mean density 0.375, mesh fractions are 0.393406,
0.383379 and 0.401639 for Gyroid, Primitive Schwartz and Diamond respectively.
The relative target deviations are 4.908%, 2.234% and 7.104%; partition volume
remains conserved. The optional `quality_fan` strategy mitigates local worst
elements, but can increase element count and reduce median quality. Its n=12
flow test took 13 s / 5 outer iterations, versus 8 s / 4 for the baseline;
these are single-run stationary-solver log measurements, not performance guarantees.

中文：独立验证包含界面每个三角形两侧的相归属、导入前后的界面与孔口面积、
各独立孔道的通量，以及密度与加密研究。以上数值对应明确的测试设计，不能推广
为全部参数下的精度保证。论文、投稿材料和大体积求解文件独立保存，不放入软件仓库。
