# 0.3.9

- Promote the 0.3.9.dev12 snapshot to a stable release. See the
  [0.3.9 release notes](docs/RELEASE_0.3.9.md) for changes since 0.3.4.

# 0.3.9.dev12 — linear tetrahedron export

- Remove the optional 10-node curved export and its COMSOL import validation case.
  The NAS and NPZ exports contain four-node tetrahedra.

# 0.3.9.dev11 — optional surface envelope; hand-off (2026-11-05)

- `optimize_mesh(..., surface_envelope=mm)` (CLI `--surface-envelope`):
  interface nodes may leave the exact implicit surface by at most this
  first-order distance |g|/|grad g| when that improves the local worst quality
  (fTetWild-style envelope; default None keeps the exact surface). The maximum
  deviation is checked and reported (`optimization.surface_deviation_max_mm`).
  With effort="high" and 8.66 µm (fTetWild's default 1e-3 of the box diagonal)
  over the 16 bcc_crosscompare cases: median solid p01 0.62, median minimum
  0.48; versus fTetWild on the same boundary p01 higher in 15/16, minimum
  higher in 15/16, faster in 16/16 (1.4–4.2x), but solid volume error lower
  in only 4/16 (typically +0.01 to +0.1 pp higher; the envelope moves are not
  volume-neutral yet).
- docs/HANDOFF_ZH.md: state, architecture, reproduction commands and the open
  task list for the next maintainer.

# 0.3.9.dev10 — high-effort optimisation (2026-11-05)

- `optimize_mesh(..., effort="high")` (CLI `--optimize-effort high`, web
  "高质量优化"): every cell below quality 0.7 is targeted instead of 0.5. The
  standard threshold capped the BCC solid p01 near 0.52 at every resolution.
  16 bcc_crosscompare cases (BCC α 0.3 / boundary 0.35): median p01 0.51 →
  0.57 (up to 0.66), median end-to-end time 2.7 s → 4.1 s. Against fTetWild on
  the same boundary (single thread): p01 higher in 12 of 16 cases, solid volume
  error lower in 15 of 16, 2–5× less time; fTetWild keeps a higher minimum in
  8 of 16 (mostly Diamond and IWP box layers) and higher p01 on the coarsest
  Diamond/IWP levels.
- Edge-removal candidates are scored with the same scalar arithmetic but
  without per-call cache overhead (identical meshes; about 7 % less time).

# 0.3.9.dev9 — box-layer snapping, curved quadratic export, periodic faces (2026-11-05)

- `Config.snap_alpha_boundary` (CLI `--snap-alpha-boundary`): separate snap
  threshold for lattice vertices on the outer box. Diagnosis on the BCC meshes:
  80–100 % of the worst 1 % cells touch the box, mostly where the TPMS meets a
  box face at a shallow angle. With BCC α 0.3 / boundary 0.35 over the 16
  bcc_crosscompare cases: solid p01 up in 12 and down in 2 (at most −0.013),
  solid minimum up in 8 and down in 1 (−0.003), volume error unchanged within
  ±0.05 pp. 0.4 produced a degenerate case (Diamond r10) and is not recommended.
  Default None keeps dev8 arrays.
- `make_quadratic(model)` / `save_model(..., quadratic=True)` / CLI
  `--quadratic`: `mesh_quadratic.nas` with 10-node CTETRA and CTRIA6; mid-edge
  nodes of implicit-surface edges lie on the exact level set (box planes kept
  flat, both phases share them). Jacobian sampled at 19 points per element;
  curvature halved where det J / det J_linear < 0.2. Solid volume error versus
  the 256³ integral drops 5–64× (e.g. Gyroid r13 0.335 → 0.006 pp); solid +
  fluid still fill the box exactly. COMSOL import of the 10-node file is not
  yet verified (evidence/run_dev9_validation.py).
- `Config.periodic_faces` (CLI `--periodic-faces`): on every axis where the
  field repeats (checked numerically), opposite faces get identical node sets
  and triangulations: field samples copied, snaps paired, coordinates enforced
  and `report["periodic"]` verified after generation and optimisation. The
  optimiser moves paired face nodes together and skips face retriangulation
  there (p01 Gyroid 0.511 → 0.505, Diamond 0.504 → 0.492; about 2× slower).
- `report["wall_resolution"]` for sheets: 2q/|∇f| wall thickness against the
  lattice spacing, with a warning when the thinnest 5 % is under one cell.
- Optimiser incidence is built by array grouping: identical meshes, 14 % less
  optimisation time on the 12-case BCC timing set.
- Not adopted: thin-wall adaptive edge bisection (benchmarks/refine_experiment.py)
  did not beat uniform BCC at equal element count (Diamond_low and Gyroid
  0.08→0.5, P1 k_eff) and was 2–4× slower.
- Benchmarks: bcc_crosscompare.py, other_tools.py (TetGen, fTetWild, microgen),
  quadratic_volume.py, bcc_figures.py; validation launcher Run-Dev9Validation.cmd.

# 0.3.9.dev8 — optional body-centred cubic background lattice (2026-11-05)

- New `Config.lattice = "kuhn" | "bcc"` (CLI `--lattice`, web "背景格子").
  `bcc` adds voxel centres and box-face centres; every pair of face-adjacent
  voxels contributes four congruent tetrahedra (mean ratio 0.952, dihedral
  angles 60°/90°), and each box-face square is coned through its face centre
  (four half tetrahedra, mean ratio 0.800). The background is conforming to
  the box faces, so clipping, snapping, implicit-root cuts, two-phase
  generation and `optimize_mesh` are unchanged. Default stays `kuhn`.
- BCC has about twice the tetrahedra per voxel: use about 5/6 of the Kuhn
  resolution for a similar element count. Recommended with `snap_alpha=0.3`
  (web UI uses 0.3 for BCC, 0.25 for Kuhn).
- `benchmarks/bcc_vs_kuhn.py`: four quality_vs_gmsh designs, Kuhn r12/r16
  (α 0.25) vs BCC r10/r13 (α 0.3), optimised: solid p01 higher in 6 of 8
  cases (equal in one, −0.013 in one), minimum higher in 7 of 8, median
  +0.07 to +0.17 in all, volume error within ±0.07 pp of Kuhn, generate +
  optimise 1.1–3.8× faster. Larger α (0.4, or 0.25/0.40 by edge length) raises
  quality further on some designs but increases the volume error.
- Per-voxel density bookkeeping uses an explicit background-to-voxel owner
  map instead of `owner // 6`.
- `benchmarks/itt_*.py`: stage-0 prototype of template inheritance and
  iso-surface transport (not part of the package; see evidence/itt_stage0).

# 0.3.9.dev7 — less repeated work and phase-aware surface moves (2026-11-05)

- Generation reuses SVD polygon ordering only for shared faces with exactly
  matching vertex IDs and order. Scalar fan screening rejects clear failures;
  close decisions and accepted fans retain the original NumPy arithmetic.
- Cache ordered-tetrahedron metrics during reconnection, up to 65,536 entries;
  invalidate on every coordinate refresh. The cache preserves signed volume
  and does not change numerical decisions.
- Evaluate chord error on actual incident faces, without padded five-dimensional
  triangle copies; skip candidates already excluded by quality or orientation.
- Mixed-phase surface moves preserve each phase's local minimum quality.
  Round-level quality, median, chord and volume budgets remain unchanged.
  This can retain more rounds and improve final quality, with a design-dependent
  time cost. Record rejected phase trials in each round's history.
- Add numerical regressions for scalar screening, cache invalidation, packed
  chord errors and per-phase trial acceptance.

# 0.3.9.dev6 — generation work reuse and geometry budgets (2026-11-05)

- Generation reuses face incidence for connected components, precomputes face
  masks, and batches shared-vertex coordinate validation in two-phase meshes.
- Surface smoothing recomputes hard chord guards against the current geometry
  of each independent batch, with bounded chunk memory. `optimize_mesh` now
  defaults to `chord_tolerance=0.0`; explicit `None` retains soft ranking only.
- Each optimisation round checks per-phase minimum/p01/poor-volume quality,
  the existing median-loss budget, and cumulative phase-volume drift (default
  0.3 percentage points of box volume; configurable or `None`). A failed round
  restores the previous mesh and records its rejection reasons.
- Recompute density profiles, domain/phase volumes, counts and interface area
  from the final mesh. Profile `slab_count` records unequal-bin weights.
- Rebuild current boundaries before optional SLSQP conditioning. Clear input
  COMSOL solve evidence after optimisation.
- Add regressions for live-neighbour chord guards, exact rollback, geometry
  reports, SLSQP boundaries, shared-coordinate validation and face components.

# 0.3.9.dev5 — speed (2026-11-05)

- Generation: identical meshes (mesh_sha256 checked on eight configurations
  against 0.3.9.dev3), about 30% faster.
  - `choose_fan` scores its 3-10 cell fans with scalar volume / edge-length
    arithmetic that reproduces numpy's bit for bit (the 3-term `einsum` order
    is detected at import against the local numpy; `pow` stays in numpy);
    falls back to the numpy path if no order matches.
  - `quality.unique_rows` packs integer rows into one int64 key: identical
    output to `np.unique(axis=0)`, several times faster; used for face/edge
    incidence in generation, two-phase partition, conditioning and audits.
  - `boundary_digest` is vectorised (same stable sorts, same bytes).
  - Expressions are compiled once (whitelisted AST, empty builtins) instead of
    walked node by node on every evaluation; values are bitwise identical.
- `optimize_mesh`: about 2x faster than 0.3.9.dev4 at equal or better quality.
  - New `tpmslab.batch_smoothing`: interior and surface smoothing move
    independent vertex sets (stars sharing no cell) together; all trial
    positions of a batch are scored in one pass with the vertex-affine volume
    and vertex-quadratic edge-length forms, without materialising the moved
    tetrahedra.
  - Adjacency is built once per round and shared; boundary arrays for the
    surface step come from the mesh bookkeeping; oriented arrays are rebuilt
    once at the end.
  - The chord-error criterion uses one |grad g| per candidate vertex (six
    fewer field evaluations per triangle centroid); cap flips compare
    area x |g| directly.
- Mesh results of `optimize=True` change (Jacobi-style batches instead of
  vertex-by-vertex order); default meshes do not.

# 0.3.9.dev4 — faster, stronger cut-layer optimisation (2026-11-05)

- Local mesh operations score their 1-20 cell stars in pure Python
  (`quality.tet_scalar`/`star_scalar`) or one vectorised pass over all trial
  positions (`quality.star_trials`) instead of one numpy call per star;
  numpy call overhead was ~95% of the dev3 optimiser time.
- General Shewchuk-style edge removal (`_Mesh.remove_edge`): an interior edge
  with a ring of 3-7 tetrahedra is replaced by the maximin triangulation of
  its ring (O(n^3) dynamic programme); 3-2 and 4-4 flips are the n=3 and n=4
  cases. Candidate edges/faces that failed are not retried within a round.
- Interior and surface smoothing try two directions per vertex (neighbour
  centroid and the point that makes the worst incident cell regular) at four
  step lengths and keep the best admissible one.
- Each round works on the quality tail (cells below the 5% quantile, never
  above `quality_threshold`), `focus_quantile=0.05`.
- Surface moves are chosen with a chord-error criterion: among positions that
  reach at least `chord_gain_floor` (0.1) of the best quality gain, the one
  whose incident surface triangles depart least from the level set
  (area x |g|/|grad g| at the centroids) is taken. An optional hard gate is
  `chord_tolerance`. Without this, the stronger smoothing moved the phase
  volume away from the exact implicit volume on coarse meshes.
- The per-round SLSQP `condition_mesh` pass is no longer part of the default
  `smooth=True` path (it changed no p01/minimum value in the study);
  `smooth="slsqp"` keeps it.
- Tests: edge removal on a five-cell ring, vectorised star trials, and the
  optimise test now asserts that the solid volume does not move away from a
  128^3 midpoint integral of the exact field (was: shift < 0.1 pp).
- Box-face edge removal (`_Mesh.remove_boundary_edge`): an edge lying in a
  flat box face, whose two box-face triangles are coplanar, is removed with
  its open ring; the face quad takes the other diagonal. Region, box faces
  and phase volumes are exactly unchanged.
- Interface cap flips (`_Mesh.cap_flip`, two-phase, `surface=True` only): a
  cell with all four vertices on the interface and two interface faces is
  handed to the other phase (the interface takes the other triangulation of
  the same four surface points) and its now-interior edge is removed; kept
  only when quality improves and the new interface triangles are not further
  from the level set. Interface conformity, triangle counts per boundary id
  and connectivity are preserved; exterior/interface arrays are rebuilt.
- Default meshes (optimize=False) are unchanged.

# 0.3.9.dev3 — cut-layer mesh optimisation (2026-11-05)

- New `optimize_mesh` (`tpmslab.optimize`, `generate(..., optimize=True)`,
  `tpmslab generate --optimize`, browser checkbox): local reconnection of the
  cut layer (2-3, 3-2 and 4-4 flips restricted to interior faces/edges),
  maximin Laplacian interior smoothing with SLSQP refinement of the poorest
  stars, and `tpmslab.surface_smoothing`: boundary vertices slide within the
  box faces and *along the exact implicit level set* (vectorised Newton
  re-projection, residual < 1e-9 mm). Exterior/interface connectivity, phase
  and domain labels are unchanged; phase volumes shift by ~0.01 pp; the
  result is audited (`quality_audit`, interface, box planes, residuals).
- `benchmarks/quality_vs_gmsh.py`: like-for-like comparison with Gmsh on the
  same boundary polyhedron (fixed boundary, interior only) and with Gmsh
  re-meshing the surface as well; see docs/COMPARISON_ZH.md for numbers.
- `condition_mesh(max_median_loss=...)` exposes the median-loss budget of the
  non-regression gate (default unchanged, 5%).
- `quality.cross` / `quality.signed_volumes`: fast small-array helpers.
- Default meshes (optimize=False) are unchanged from 0.3.9.dev2.
- Not yet run in COMSOL: `evidence/run_dev2_validation.py` gained an
  `optimized_r12` case (flow and elasticity on optimised meshes).

# 0.3.9.dev2 — review fixes (2026-11-05)

Read-only review of 0.3.9.dev1 followed by these changes. No COMSOL run was
made for this version; see `evidence/README.md` for what the shipped
evidence does and does not cover.

- Evidence: rebuild `evidence/authorized_smoke_summary.json` from the attempt
  records (both smoke cases passed on their second attempt; the first-attempt
  summary is kept as `authorized_smoke_summary.first_attempt.json`); make
  `evidence/run_authorized_smoke.py` idempotent; add `evidence/README.md`.
- One COMSOL launcher (`tpmslab.comsol.run_comsol_class`) for the
  elasticity/flow bridge, `comsol_remesh` and `comsol_audit`: 6.2/6.3 mesh
  API reflection, isolated `-prefsdir`, short `-tmpdir/-recoverydir/-data`
  paths, log capture, marker parsing and the COMSOL version string recorded
  in `run.json`/`comsol_verification.json`. `remesh-quality` previously used a
  hard-coded `geometricModel("")` that does not compile on 6.2.
- `volume_components` now counts connectivity through shared faces in both
  single- and two-phase modes; vertex/edge-only contacts are reported as
  `pinched_contacts` with a warning instead of counting as one body.
- Reopen check compares the integrated volume with a relative tolerance
  (was an absolute 1e-15 m^3).
- Pre-flight estimate of the background-lattice memory; oversized
  `cells`×`resolution` fail with an explicit message (limit adjustable via
  `TPMSLAB_MAX_LATTICE_BYTES`).
- Two-phase generation samples, calibrates, audits and snaps the lattice
  once and shares it between the solid and complementary passes.
- `edge_interpolation="implicit_root"` uses vectorised bracketed bisection
  started from the linear cut instead of per-edge `brentq`; the
  unbracketed-cut fallback is unchanged. All other modes produce
  byte-identical meshes to 0.3.9.dev1 (checked by mesh_sha256 on seven
  configurations). `implicit_root` meshes differ only on edges where the
  field grazes zero several times within a few micrometres (12 of 5001
  nodes, at most 4.3 um, in the n=10 Gyroid check): both versions return
  a point on the implicit surface, but not always the same one.
- Two-phase `density_profile` is recomputed from the final (projected) solid
  cells; `density_profile_method` names the binning.
- Off-grid sampling probes run for built-in families as well; custom
  formulas are still rejected on a miss, built-in families get a warning.
- Flow demo restricts `spf` to fluid domains that connect Z- and Z+ and
  records `excluded_fluid_domains`; only a model without any spanning
  channel is rejected.
- `element_order` (1 or 2) and `processes` (`-np`) options on `build_mph`,
  `run_comsol`, the CLI and the convergence benchmark; recorded per attempt.
  Order 2 is not yet covered by any shipped evidence.
- `generate --mph/--solve` accepts the same launch options as `comsol`.
- The browser COMSOL path no longer deletes the previous top-level
  verification file; `build_mph` preserves it inside the new attempt.
- Documentation: version wording aligned with the 6.2.0.290 evidence.

# 0.3.9.dev1 — isolated COMSOL execution

- Explicit per-attempt preference files, CLI and benchmark options; global preferences unchanged.
- Short native temporary, recovery and workspace paths prevent Windows path-length failures.
- Actual COMSOL 6.2 solid and pore-flow smoke checks and documented comparison with related tools.

# 0.3.9.dev0 — Python to COMSOL workflow

- Correct nonuniform three-grid power-law fits and document the revised estimates.
- Preserve a separate, input-hashed directory for each COMSOL attempt; clear failed current state.
- Add run_comsol API and a comsol CLI command for saved models and material settings.
- Fix empty-cut projection and add bounded off-grid probes for custom formulas.
- Check compiler output as well as exit status; support 6.2/6.3 mesh API signatures.
- Checkpoint convergence rows, resume studies, fix Gmsh cleanup and label elasticity snapping accurately.
- Stream hashes; preserve historical delivery records under history/pre_0.3.9.
- Real COMSOL 6.2 Java compilation succeeded for both smoke cases; NAS import blocked by file-system security preferences. No successful solver validation is claimed.

# 0.3.8.dev0 (local improvement build; not published)

- Add opt-in pre-cut lattice-vertex snapping (`Config.snap_alpha`, `--snap-alpha`): a lattice vertex within `snap_alpha` of a cut point along one of its lattice edges is moved onto the isosurface before clipping, so the near-degenerate sliver that cut would have produced is never created. Default `snap_alpha=0` leaves every existing output byte-identical.
- Snap onto the exact implicit root by default (`snap_target="implicit"`); `"linear"` keeps the linear-cut point. Outer-box nodes may only move within their box face; moves that would invert a background tetrahedron, flatten a tetrahedron onto one isosurface, or pinch the material or its complement along a lattice edge are reverted and counted in `report["vertex_snapping"]`.
- Snapping is applied to the full inequality set before any fluid-side sign flip, so complementary solid/fluid passes share identical vertices and cut keys; the interface audit, box-face audit and scalar-diffusion patch check pass on all 12 study designs.
- Add `benchmarks/snapping_study.py` (12 paired designs, three variants, recomputed statistics, dense-sample reference density and implicit-surface fidelity) and `tests/test_snapping.py`.
- Allow `snap_alpha` together with `edge_interpolation="implicit_root"` / `"quadratic_root"`: root projection now uses the snapped lattice coordinates and keeps the linear cut when the true field does not bracket a root (`report["edge_root_unbracketed_cuts"]`). With both enabled every interface node lies on the implicit surface; the median solid-density error drops from 1.52 to 0.69 pp (worst 3.66 -> 2.03 pp) while the Q<0.01 count stays at 1 on the 12 designs.
- Add a browser checkbox (切割前顶点吸附) that sets `snap_alpha=0.25`, restore it from saved parameters, and show snapped/reverted vertex counts in the result notes.
- Measured on the 12 designs at `snap_alpha=0.25`: aggregate Q<0.01 cells 2710 -> 1 (Q<0.1: 60217 -> 250), tetrahedra 489696 -> 211328, per-phase minimum quality typically 0.001-0.01 -> 0.08-0.18. Costs: median absolute solid-density error unchanged (1.52 -> 1.51 pp) but the worst case rises (3.66 -> 4.03 pp); interface-node RMS distance to the implicit surface rises 10-25 % (10-40 um at 5 mm / resolution 12). No solver-accuracy or speedup claim is made.

# 0.3.7.dev0 (local improvement build; not published)

- Add bounded interior maximin smoothing with analytic quality derivatives and unchanged boundary/interface nodes, connectivity, domains and phase labels.
- Reconstruct input cell incidence; reject inverted cells, duplicates, nonmanifold faces and missing boundary/interface declarations.
- Audit candidates independently of optimizer success. Enforce phase minimum/p01/poor-count non-regression, a 5% relative median-loss budget and per-domain volume retention; roll back failed candidates with evidence.
- Expose `condition_mesh`, `quality_audit`, `generate(..., condition=True)`, CLI flags and a browser checkbox. Optional quality thresholds reject before export.
- Include 12 paired mesh records with independent geometry and exact scalar-diffusion patch checks. Do not infer FE/CFD accuracy or speedup from geometric quality alone.

# 0.3.6.dev0 (local development source; not a public release)

- Add `combine="blend"` for spatial transitions between two built-in or custom TPMS fields.
- Expose the secondary family, transition axis and bounds, weight profile, and calibration weight-sample count in Python, JSON, and the local browser interface.
- Calibrate the blended field using fixed-weight periodic quantiles; preserve exact original fields outside the transition and when both expressions are identical.
- Use one blended field for complementary solid/pore meshes and root interpolation, and export complete combined-design metadata.
- Add tests, a Python example, a JSON example, and combined-geometry documentation. The feature adds one composite solid phase and its complementary pore phase.
- Preserve frozen experiment sources and unchanged default mesh arrays; this feature has not established a mesh-quality or FEM accuracy advantage.

# 0.3.5.dev0 (local experimental source; not a public release)

- Add an optional COMSOL-native `RemeshFaces` plus `FreeTet` workflow for already verified two-phase exports.
- Require save/reopen, preserved domain and boundary selections, exact cross-phase face incidence, volume and interface-area tolerances, and improved solid quality before accepting the output.
- Add opt-in `edge_interpolation="implicit_root"` for two-phase meshes. It projects shared cut vertices to implicit-field edge roots subject to local positive-tetrahedron checks; the default remains linear interpolation.
- Add experimental `edge_interpolation="quadratic_root"`: batch-evaluate the continuous field at three positions on each cut edge, fit a quadratic crossing, and apply the same inversion guard to both phases. It offers a lower-cost approximation to iterative exact roots in the tested graded designs.
- Keep COMSOL external and optional. Preserve the 0.3.4 direct-mesh benchmark provenance separately from these exploratory modes.

# 0.3.4

- Add configurable density-calibration sampling via `m_cal` (default 56).

- Remove fixed upper limits on per-cell resolution and per-axis cell counts.
- Use an uncapped integer input in the browser interface.
- Preserve lower bounds, integer validation and numerical algorithms.

# 0.3.3

- Set the per-cell resolution maximum to 48 in Python and the browser slider.
- Retain removal of the total background-grid cap.
- Synchronize documentation and add boundary regression checks.
- Report the running generator version instead of a stale literal; geometry is unchanged.

# 0.3.2

- Remove the fixed 200,000-background-voxel limit from configuration validation.
- Retain per-axis cell-count and per-cell resolution validation.
- Add configuration regression coverage for grids above the former limit.

# 0.3.1

- Use English for Python progress callbacks, exceptions, API messages and exported warnings.
- Update generator metadata; preserve numerical algorithms and historical validation records.

# Changelog

## 0.3.0

- Package the verified complementary solid/fluid workflow and independent audits.
- Include density, refinement, interface, channel and performance records with provenance.
- Retain the validated numerical implementation from commit 686b783e0385.
- Clarify geometry/solver response stability and local quality-strategy limitations.
- Add isolated-wheel reference checks, exact-commit release preparation and public version documentation.
- Preserve historical validation provenance; GitHub publication and DOI registration remain separate records.


## 0.3.0rc1 — solid and pore-fluid domains

- Add complementary solid/fluid meshes with shared interface vertices and faces.
- Preserve independently connected solid and fluid domains, phase arrays and named boundaries.
- Add fluid preview, domain metadata and optional COMSOL named selections.
- Verify three fixed-wall pore-flow examples in COMSOL; deformation-coupled FSI is not implemented.
- Add analytical partition and multi-family interface regression tests.

## 0.2.0rc1 — local release candidate

- Introduce src-layout packaging, public Python API and command-line commands.
- Keep Flask and the browser interface optional; COMSOL remains optional.
- Add conservative interior-fan selection with unchanged external boundary.
- Add quality-tail diagnostics, deterministic mesh/boundary fingerprints,
  environment records and file-integrity manifests.
- Normalize configuration lists to immutable tuples and apply phase controls
  consistently in the field helper.
- Include reproducible comparisons, automated tests and release documentation.

Earlier rc entries describe local release candidates.
