# TPMS Lab 0.3.4

Per-cell resolution is an integer >= 8. Each axis cell count is a positive integer. Neither has a fixed upper bound in Python or the browser interface. The total-background-voxel cap remains absent. Actual mesh and solver sizes are constrained by available resources. This patch changes input validation and controls, not the numerical meshing or COMSOL templates. Historical validation provenance is preserved.

### Density calibration sampling

`Config(m_cal=80)` sets the number of midpoint samples per axis used to calibrate density to threshold (80 cubed samples). The default is 56; values must be integers of at least 8, with no fixed upper bound. This parameter is independent of mesh `resolution`. JSON configurations use the same `m_cal` key; older configurations default to 56. Exported configuration records include the selected value. Increasing it changes density calibration and can change the generated geometry. Historical validation used 56.
