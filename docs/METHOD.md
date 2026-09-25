# Method and interpretation

## Scalar fields and density

Physical coordinates in mm are transformed to angles using cell count, box
length and phase. The registry contains trigonometric approximations adapted
from LattGen, not exact minimal surfaces. The CDF of 56^3 midpoint samples of a
periodic cell maps prescribed density to a threshold q. The solid is defined
by q-f >= 0 for the below network, q+f >= 0 after calibrating -f for the above
network, and both q-f >= 0 and q+f >= 0 after calibrating |f| for a sheet.
Each inequality is linearly interpolated in each background tetrahedron. This
avoids applying |f| only at vertices, which could miss sheets crossing an edge.

Gradient profiles are uniform, linear, quadratic, cubic, cosine, exponential,
radial and three-axis periodic. Density is calibrated locally against a full
periodic cell, so finite specimens and spatial gradients do not necessarily
achieve either the nominal mean or a requested global volume fraction. Sheets
have constant level-set band width only when q is constant; physical thickness
also depends on the field gradient.

## Volume construction

A regular Cartesian grid is divided into six consistently oriented tetrahedra
per voxel. Material portions are convex clipped polytopes. Intersections on a
shared background edge reuse a global key. Polygon faces use a canonical fan
from their globally smallest vertex key, ensuring compatible face diagonals.
The two sheet inequalities cannot meet inside the domain when the interpolated
threshold is strictly positive. The implementation enumerates retained vertices
and valid background-edge intersections; difficult or degenerate cases can be
rejected by the subsequent checks rather than silently emitted.

The baseline triangulates each polytope from its smallest-key vertex. The
optional quality_fan strategy considers a centroid fan only when all baseline
tetrahedra have positive absolute volume and their minimum quality is below
0.3. Both candidates retain the same polygon-face triangulation. For a tet of
volume V and six edge lengths l_i the mean-ratio quality is

    Q = 12 (3 V)^(2/3) / sum_i(l_i^2)

The equilateral value is 1. A centroid candidate is accepted only if all
volumes are positive, its minimum Q exceeds the baseline by a relative 1e-6,
and its volume agrees with relative tolerance 1e-10. Only an interior vertex
is added. Final tetrahedra are reoriented to have positive signed volumes.
The strategy is a local heuristic and is not a new guaranteed-quality mesher.
It can increase element count and even increase the count of elements below a
fixed quality threshold, despite improving the worst quality in each selected
polytope. Inspect the median and tail, not just the minimum.

## Checks and output semantics

Export checks positive tetrahedral volumes, at-most-two face incidence, a
closed consistently oriented boundary with two incident triangles per boundary
edge, and agreement between summed tet volume and signed boundary volume.
Connectivity is computed through shared vertices; it is a diagnostic rather
than a proof of structural load-path adequacy. Self-intersection is not checked
by a general-purpose geometric collision algorithm. The construction and the
validation should not be described as a proof for every custom field.

CTETRA properties label material components. CTRIA3 property IDs 2/3, 4/5 and
6/7 identify min/max x, y and z faces; 8 identifies the remaining material
boundary. Array indices in Python and NPZ are zero-based; Nastran IDs are
one-based. Coordinates are mm. The bridge explicitly sets COMSOL length units.
Boundary hashes sort rounded coordinates to 11 decimal places; they are paired
regression fingerprints, not proofs of mathematical identity. File hashes
cover actual exported bytes. Exact output bytes across different numerical
library/OS versions are not promised.

## Validation scope

The accompanying benchmark covers three families, two density profiles, three
sampling resolutions and two strategies (36 runs, 18 pairs). An independent
affine half-box test checks exact volume 62.5 mm^3. The optional COMSOL benchmark
uses six single-cell models at resolution 12. It solves linear elasticity,
saves and reopens MPH files. It does not establish FE convergence, stress
accuracy, experimental agreement or superiority to a different mesher.

Material for the workflow demo: E=1.5 GPa, nu=0.3, mass density=950 kg/m^3.
A 5 mm box has its bottom fixed; top z displacement is -0.005 mm, with top x/y
unconstrained. Remaining boundaries are traction-free, geometric nonlinearity
is disabled, and displacement interpolation is first-order. These settings are
illustrative, not fitted to a particular printed polymer.
