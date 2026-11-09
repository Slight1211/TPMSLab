# Combined TPMS: spatial transition between two families

Spatial blending was introduced in `0.3.6.dev0` and is available in the
`0.3.9` stable release. The manuscript benchmark used the frozen `0.3.4` source.

```python
from tpmslab import Config, generate, save_model

config = Config(
    family="Gyroid", combine="blend", family_secondary="Diamond",
    blend_axis="z", blend_start=0.25, blend_end=0.75,
    blend_profile="smoothstep", domain_mode="solid_fluid",
    density_start=0.25, density_end=0.50, gradient="linear",
    resolution=16, edge_interpolation="quadratic_root",
)
model = generate(config, progress=print)
save_model(model, "combined_result")
```

The same configuration is accepted by JSON/CLI and the local browser form:

```text
tpmslab generate --config examples/combined_gyroid_diamond.json --out combined_result
tpmslab web
```

## Geometry definition

The fields of two families share the configured cell counts, phase and structure
mode. At a physical point, the field is `f=(1-w)*f_A+w*f_B`. Before the transition,
`w=0`; after it, `w=1`. Bounds are fractions of the specimen dimension along
`blend_axis`, independent of the density-gradient axis. In a 5 mm cube with
`blend_start=.25` and `blend_end=.75`, transition occurs from 1.25 to 3.75 mm.
This is a morph of implicit fields, not a fraction of two materials or a blend of
signed distance functions. Both families form one solid material and its pore
complement; it does not introduce a second solid-material domain.

`smoothstep` uses the C2 quintic `10t^3-15t^4+6t^5`. `cosine` and `linear` are
also available. `family_secondary="Custom"` uses `custom_secondary` and the same
restricted mathematical-expression evaluator as the primary family. Exact
cancellation that makes a calibration sample constant is rejected explicitly.

## Density definition

Each fixed blend weight has its own periodic mixed-field calibration. The default
17 weight samples include both original families; quantiles are linearly
interpolated between weights and use `m_cal` samples per periodic axis. Increase
`blend_calibration_steps` to check this interpolation independently of mesh
resolution. At `w=0` or `w=1`, calibration is exactly the corresponding original
family's calibration. Blending two independently calibrated thresholds without
recalibrating their mixed field would define a different density convention.
The quintic weight is C2, but the tabulated threshold is piecewise linear in
weight and density; the complete level-set field is not claimed to be globally C2.
No amplitude normalization is applied to either family's expression. Changing a
custom expression's scale can change its blended geometry.

The target is a fixed-weight periodic solid fraction. It is not an exact physical
slice density in the finite transition region. Density law, finite specimen,
transition width, calibration sampling and tetrahedral discretization all affect
measured solid fraction. The exported report preserves both expressions, weight
profile, transition bounds and calibration settings.

## Export and numerical scope

Generation uses one combined field consistently for the solid, complementary pore,
root interpolation and density preview. NAS contains tetrahedral domains and
boundary labels; NPZ, solid/pore STL previews, JSON configuration and manifests
follow the existing export path. Java/MPH preparation uses the same mesh report.
STL remains a preview; this feature does not reconstruct smooth CAD.

Tests check exact end-family fields, identical-family regression, transition
continuity, mixed-field quantiles, validation, positive tetrahedra, shared-face
incidence, the 125 mm³ complementary partition and NAS/Java/config export. COMSOL
solving is external and licensed; a new combined specimen requires its own
constraint, response and resolution verification. Smooth weights do not guarantee
connected solids, open fluid channels, feature preservation or high-quality cells.

This release implements spatial blending; Boolean union/intersection/difference
are not aliases for this operation.
