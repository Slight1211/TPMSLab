from tpmslab import Config, generate, save_model

mesh = generate(
    Config(
        family="Gyroid",
        resolution=12,
        density_start=0.25,
        density_end=0.5,
        quality_strategy="quality_fan",
    )
)
print(mesh["report"]["quality_quantiles"])
save_model(mesh, "graded_gyroid_output")
