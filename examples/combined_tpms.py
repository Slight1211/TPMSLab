"""Generate a graded Gyroid-to-Diamond solid/pore partition (no COMSOL required)."""
from pathlib import Path
from tpmslab import Config, generate, save_model

config = Config(
    family="Gyroid", combine="blend", family_secondary="Diamond",
    blend_axis="z", blend_start=.25, blend_end=.75,
    blend_profile="smoothstep", domain_mode="solid_fluid",
    density_start=.25, density_end=.50, gradient="linear", axis="z",
    resolution=16, edge_interpolation="quadratic_root",
)
model = generate(config, progress=print)
folder = save_model(model, Path("combined_tpms_result"))
print(f"Output directory: {folder}")
print(f"Measured solid fraction: {model['report']['actual_density']:.6f}")
