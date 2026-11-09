# Optional COMSOL quality remeshing

This local development API takes an already verified `solid_fluid` export and
uses a licensed COMSOL on Windows (6.2/6.3 mesh API selected by reflection; this mode has not yet been exercised in the shipped evidence) with `RemeshFaces` and `FreeTet` to create another mesh-defined MPH. It launches through the same isolated-preference and short-scratch-path mechanism as the elasticity/flow bridge (`--preferences-directory`).
COMSOL and its licence are external, optional requirements. No TetGen package
or new tetrahedral algorithm is bundled.

The development source also offers a separate Python-side geometry option:
`Config(..., domain_mode="solid_fluid", edge_interpolation="implicit_root")`.
It moves the shared linear-cut vertices to implicit-field edge roots, with
local positive-volume checks. The default is `"linear"`. Either output can
then be exported and passed to the COMSOL remeshing step below. Edge projection
primarily improves geometric density in the tested cases; COMSOL remeshing
primarily improves the low-quality tail. Their effects must be evaluated
separately because the COMSOL operation can shift the projected interface.
An additional `"quadratic_root"` mode estimates the edge crossing from three
batched field evaluations instead of iterative roots. It uses the same
shared-vertex and positive-volume checks. In a 12-case graded-design pilot
it reduced density error relative to the linear default at unchanged
tetrahedron counts, with shorter generation time than iterative exact roots.
Both root-based modes are experimental; neither by itself fixes low-quality
cut tetrahedra.

```python
from tpmslab import remesh_in_comsol

evidence = remesh_in_comsol(
    "base_two_phase_export", "quality_model", volume_size=7,
    temp_directory="D:/TPMSLab_runs/comsol_tmp",
)
print(evidence["q01_solid"])
```

The same operation is available as
`tpmslab remesh-quality --source base_two_phase_export --out quality_model --volume-size 7`.

The function creates a new output directory, preserving the original mesh.
`quality_remeshed.mph` is saved and reopened. The JSON evidence contains
solid-tetrahedron count, first-percentile mean-ratio quality, each imported
phase-domain volume, shared-interface area, source hash, and drift from the
source. Every cross-phase tetrahedral face must match exactly one triangle in
the declared shared-interface selection, with no undeclared or duplicate
interface face. All imported domain and exterior-boundary selections must
remain nonempty; the whole-box tetrahedral volume must be conserved. The default
acceptance limits are 0.3 percentage points for each phase-domain volume and
2% for shared-interface area. A failed audit leaves logs and model files for
inspection and raises an exception.

The `volume_size` and `face_size` arguments use COMSOL's 1--9 size presets;
`face_size=0` leaves face sizing at COMSOL's default. `face_representation`
may be `curved` or `linear`; the tested mode is curved. Parameter choices in
the accompanying AES exploratory evidence were selected for three central
gradient `n=16` cases and then fixed for six other gradients. Their quality
improvement does not establish general behavior for other resolutions,
families, or geometries. Remeshing can change density realization, so check
the intended geometric and solver observables independently.

The optional remeshing mode was introduced in 0.3.5.dev0 and is available in
the 0.3.9 stable release. Cite the fixed 0.3.4 source when describing the
original head-to-head experiment.
