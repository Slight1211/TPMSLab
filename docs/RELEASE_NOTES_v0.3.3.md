# TPMS Lab 0.3.3

The supported resolution is 8 through 48 subdivisions per cell edge in both Python and the browser interface. The fixed total-background-voxel cap remains removed. Cell counts remain 1 through 6 per axis. Larger multi-cell configurations still require sufficient memory.

This patch changes configuration validation and the resolution slider, not the numerical meshing algorithms. The n=48 flow and elasticity results were generated with the preceding implementation and passed saved-model reopening verification. Geometry experiments above n=48 are historical records outside this release range.

The generator version in mesh reports is now read from the package version. This corrects metadata without changing mesh coordinates, connectivity, phase labels or solver settings.
