# Changelog

[日本語](CHANGELOG.md) | English

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow [Semantic Versioning](https://semver.org/).

Because this is a `0.x` series, a minor release may contain backward-incompatible changes.
Versions 0.1.0 to 0.14.0 were **assigned retroactively from the development history** (organized as of 2026-08-07).

This is a translation of the Japanese `CHANGELOG.md`. If the two disagree, the Japanese file is authoritative.

## [0.60.1] — 2026-10-06

### Fixed

- Per-layer viscosity maps (`render_layer_map` / `render_layer_grid`): in a layer whose finite values were all 1e100 or more (or uniform), the top and bottom of the log scale
  were clipped to the same ceiling, the norm collapsed, and the layer was drawn in the lowest color. The `vmax + 1.0` safeguard in the grid version has no effect at the 1e100 magnitude.
  The bottom is now set one decade below the top, and the layer is drawn in the highest color. Leaving frozen cells (inf) blank is the 0.60.0 design and is unchanged
  (Codex P2 on mold-flow-fangate2#15; the same patch goes to fangate2).
- Tests: with every layer set to 1e200, both renderers build a LogNorm one decade wide. They fail before the fix.

## [0.60.0] — 2026-10-06

**The viscous heating of the multilayer model is now solved self-consistently with the viscosity after heating. Geometries whose fixed point fell into a period-2 oscillation under strong shear and did not converge now converge in a few iterations. The converged result is still the fixed point of the original equations. Also fixed Cross-WLF returning NaN or 0 in cold layers.**

### Fixed

- The rise from viscous heating (stage 1) `ΔT_k = η_k·γ̇_k²·min(t_arr, τ_thermal)/(ρ·cp)` is now solved with η_k read at the temperature after heating,
  `T_Neumann + ΔT`, instead of η_k from the previous iteration. Per element this is the one-variable equation `ΔT = k·η(T_Neumann + ΔT)`; the right-hand side
  decreases as ΔT grows, so it has exactly one root. Added `self_consistent_shear_heating` to `core/multilayer_thermal.py`. The root is searched on
  `log ΔT − log(k·η)` with the Illinois method and stops at a residual of 1e-9 K. `MultilayerHeleShawSolver._layer_temperatures` uses it both at the start of each iteration and in the re-read from #109.
- The old one-iteration lag iterated the map `ΔT ↦ k·η(T_c + ΔT)`. This map is decreasing, so the iteration jumps back and forth across the solution, and the error
  shrinks each step only by the factor of the slope `ΔT·|d ln η/dT|`. In cold wall layers under strong shear the slope approaches 1 (0.98 in the wall layer of 0.35 mm PP-T20 at 2 m/s).
  Combined with the τ update, the whole iteration oscillated, and the default gate of mold-flow-fangate2 (Br max 48–54) did not converge even with 80 iterations and damping 0.5.
- The convergence criterion `convergence_tol` is unchanged. Only the iteration path changed.
- `cross_wlf_viscosity`: from around `D2 − A2` downward, intermediate values overflowed float, and evaluating the formula as written gave NaN (inf/inf) or 0 (finite/inf).
  0 treats a frozen layer as the layer that flows most easily. Only the elements where `η0`, `η0·γ̇` (the product before dividing by τ*), or the ratio overflows are evaluated in log space,
  returning a huge finite value or `inf` (frozen) (the product condition is from the @claude review on PR #111: for ABS, PC, and PA66 with τ* ≥ 1.8e4 Pa, the product overflowed and gave 0 even when the ratio did not).
  Where nothing overflows, the formula is used as is, and results are bit-identical in the normal range for all resins (only the boundary band differs by 1e-13 of rounding).
  At the recommended mold temperatures this is reached by PA66 (`D2 − A2` = 164 °C, the whole range), PC (80–95 °C), and ABS (40–50 °C); the wall layers of the multilayer model cool down to the mold temperature
  (Codex P1 on PR #111).
- For frozen layers (`k·η(T_c)` = inf) and near-frozen layers (on the order of 1e60 K), the root search for viscous heating finds the upper bracket by doubling from 1 K. A viscosity function that returns NaN is rejected.
  Elements where the secant method did not close in within the iteration limit are finished by bisection; no value is returned unconverged.
- Per-layer viscosity maps: the top of the log color scale is capped at 1e100. Taking it up to the near-frozen 1e300 range overflows the scale in float and
  crashes matplotlib. Values above the top are drawn in the highest color; frozen cells (inf) are left blank.

### Verification

- Ran the mold-flow-fangate2 defaults (N=7, wall_refined, thermal coupling, viscous heating, 12 iterations, tol 1e-3) on a copy with the shared files replaced.
  Gate 1 converged in 3 main-analysis and 3 two-phase iterations, Gate 2 in 4 main-analysis and 3 two-phase iterations (before the fix, both were cut off at 12).
  T_fill is 0.26836 s for Gate 1 (inside the pre-fix oscillation, 0.2679–0.2684) and 0.30218 s for Gate 2 (slightly above the pre-fix 0.2902–0.3017).
- Gate 2 lands outside the oscillation range because this fixed point was unstable under the old iteration. Applying the old map once to the state solved again with tol 1e-8 (T_fill 0.302174 s)
  moves τ by only 2e-7, so it is a fixed point of the original equations. Iterating the old map from there, the deviation grows about 4× per step
  and moves into the oscillation. Gate 1 behaves the same; the first change is 1.7e-7. The difference in T_fill between the default tol 1e-3 result and the tol 1e-8 result is 5e-6 for Gate 2 and 1.3e-5 for Gate 1.
- On a small flat plate (20×8 mm, 0.35 mm, V 2 m/s, 0.2 cm³/s), the old code was cut off at 12 iterations in both the main analysis and two-phase; the new code converges in 4 and 2.
  Tightened to tol 1e-9, the old result after 3000 iterations, 0.4246046 s, and the new result after 10, 0.4246054 s, agree to 2e-6.
- Changes on the sim default screen (Film gate 1, multilayer N=7, viscous heating, two-phase ON): iterations are 2 main-analysis and 3 two-phase, the same as before the fix. T_fill changes by 1.9e-7 relative,
  the fill-time field by at most 4.9 µs (2e-4 of T_fill), and the normalized pressure by at most 7e-6. The two-phase pool and final shape are identical cell by cell,
  and the injection-phase times shift by at most 5 µs. The maximum viscous heating goes from 6.61 to 6.08 K, the mean from 0.207 to 0.206 K, and layer temperatures differ by at most 0.53 K.
  The old values were computed with the viscosity of the previous iteration.
- Small PA66 flat plate (V 200 mm/s, 2 cm³/s): before the fix, NaN appeared in the layer viscosity with viscous heating both OFF and ON (72 and 62 elements), and the iteration was cut off at 12.
  After the fix, OFF converges in 4 and ON in 3, with no NaN (with OFF, 60 frozen elements are inf). T_fill goes from 0.00107 to 0.00152 s with OFF
  and from 0.00092 to 0.00151 s with ON. PC, ABS, and PMMA under the same conditions with viscous heating ON were cut off at 12 before the fix and converge in 3–4 after it.
- Tests: on a strongly sheared flat plate, the main analysis and two-phase converge within 12 iterations; the reported ΔT_k/(η_k·γ̇_k²) is the same for every layer in a cell;
  the converged state is also a fixed point of the old map (τ changes by less than 1e-8); in the cold wall layers of PA66, layer viscosity and rise are finite and converge.
  Unit tests: the returned ΔT satisfies the stage-1 equation; it always lies between two consecutive iterates of the lagged map; a finite root is returned when starting from frozen and near-frozen states;
  a viscosity function that returns NaN is rejected; bisection reaches the root even when the secant method is cut off after one step; Cross-WLF returns neither NaN nor 0 in the frozen range or in the band where only the product overflows,
  and is bit-identical in the normal range; a viscosity map that contains frozen cells can be drawn.
  Mutations were applied to a copy: reverting the solver to the one-iteration lag, reverting Cross-WLF to the formula as written, removing the product condition, keeping the upper bracket of the root search at `k·η(T_c)`,
  removing the bisection finish, and removing the cap on the color scale.
  For each mutation, the test that guards that part fails.

## [0.59.1] — 2026-10-06

**When the injection phase of two-phase is solved with the multilayer model, the frozen cells of the center layer are now counted at the temperature of the τ that determined the pool. The main-analysis output is unchanged.**

### Fixed

- Each iteration of `MultilayerHeleShawSolver._fixed_point` builds arrival times and layer temperatures from the τ it starts with, and then updates τ.
  So the returned `short_shot_mask` (the center layer of the last `layer_T_K`) belongs to the τ one step before the returned τ. Two-phase
  `injection_center_solid_cells` counted the overlap of this mask with the pool Ω₁ determined from the returned τ. When converged, the difference stays within
  the tolerance of 1e-3, but when the iteration is cut off (an oscillating fixed point) the two disagree. `_fixed_point` now also returns `layer_T_K_end`, the layer temperatures re-read
  at the returned τ, and `short_shot_mask_end` of its center layer. The corrections for arrival time, Neumann, and viscous heating use the same expressions as the start of
  the next iteration. Two-phase counts with this mask. The main analysis's `short_shot_mask` and `layer_T_K` are unchanged
  (Issue #109; the same fix as the Codex P2 on mold-flow-fangate2#13).

### Verification

- The reference in the test is a 2-iteration run. The second iteration builds temperatures from the τ of the first, so its `layer_T_K` and `short_shot_mask` are
  bit-identical to `layer_T_K_end` and `short_shot_mask_end` of a 1-iteration run. The test runs with viscous heating both ON and OFF and compares temperatures
  (comparing the mask alone passes on a small geometry even if the viscous term is dropped from the re-read). The old expression counts 0 cells, the new one 4.
- Checked with mutations on a copy. Reverting two-phase to the old mask fails both ON and OFF; dropping the viscous term from the re-read fails ON.

## [0.59.0] — 2026-10-05

**The geometry solved with the current inputs can now be exported as IGES (3D). It is a solid built exactly, face by face, and before export it is checked against the solver's shape cell by cell.**

### Added

- `core/gate_iges.py`: `build_gate_solid(spec)` builds the resin side of the gate block as a single solid, and `iges_bytes` writes only its faces to
  IGES (trimmed surface, entity 144, mm). Instead of fitting surfaces to a raster, it turns the steps of `build_profile_gate_geometry`
  directly into solid operations (floor = union, cap = intersection, override = union of difference and intersection, steel = difference). All faces are exact:
  planes; the edge of a variable land length is a B-spline (exponents 1, 2, and 3 are polynomials and are represented exactly; other exponents are fitted within 1e-4 mm;
  edges with an exponent below 1 or a non-integer exponent are reparametrized to remove the infinite slope at the end); the ramp behind it is a sweep of that curve;
  the graded ramp angle is a ruled surface between two lines; a well is two cones and a trapezoidal prism; the corner R of the outer wall is a cylinder. Symmetric shapes are built as one half
  and mirrored about the valve axis; the runner and wells are added after mirroring. Coordinates are the same as in the received CAD (`Runner-block_3D_*.igs`):
  x = width from the valve axis, y = 20 − t, z = −depth (the PL is at z = 0). The part and the valve hole are not included.
- `read_iges` / `raster_depth` / `check_against_field`: read the exported IGES back, sew it, triangulate it, read it from above, and
  turn it back into a depth at each cell center. `export_gate_iges(spec, plate)` builds, writes, reads back, and checks against the depth field on the same 0.1 mm mesh as the drawing,
  and returns `GateIges.ok`. The conditions are that the read-back faces are sewn into a closed shell with nothing left open, with the same face count and volume as the written solid
  (volume to 1e-6); that the outline mismatch is 0 cells; and that the depth difference is within 2e-3 mm. What is checked is the exported file itself, so
  export stops if faces are dropped or changed while writing (Codex P1 on PR #108).
- UI: below the part design drawing, after the drawing, an expander "3D model (IGES)". The checkbox `gate_iges_on` (default OFF)
  builds and checks the solid and offers `{record_name}.igs` for download only on a match. On a mismatch, export is stopped and
  the number of mismatches is shown. Without OCP (Streamlit Cloud) only that fact is shown.
- Command line: `python -m core.gate_iges <spec.json | settings.json> <out.igs>` (a run's settings.json can also be read).
- Dependency: an optional `cad` extra (`cadquery-ocp-novtk>=7.9,<8`). It is not in `requirements.txt`. CI installs `.[dev,cad]` and runs the IGES tests as well.

### Fixed

- The coordinates of the drawing's depth field (`drawing_field`) now come from the builder's placement (y = −t) instead of `display_origin_mm` (the bottom edge of the first part row).
  When pad + t_max was not divisible by the mesh, y = 0 of the drawing was half a cell off the exit. The usual 0.1 mm drawings
  divide evenly and do not change. Only large blocks whose mesh became 0.2 mm or coarser had sections and dimensions off by up to 0.1 mm.

### Verification

- Checked the 31 specs on hand (the original drawings of Film gate 1–15, T-shape, fan, one-sided, candidates A–C, and others) and 22 combinations that no spec uses
  (coring welded up to the cap, a 90° cut-down step, exponents 0.5–4, variable land length / closure / thickening at both ends on one side, edge channels on both sides of a fan,
  wells with vertical walls, an offset valve axis) on a 0.1 mm mesh. In every case the outline mismatch is 0 cells, the depth difference is within 0.001 mm,
  and the volume difference is within 0.5 mm³.
- Read back, the exported IGES closes with no gaps at a sewing tolerance of 1e-6 mm, and its volume matches the original solid to 1e-6. The check including the read-back takes
  2–6 s per shape (0.1 mm mesh).
- Shapes that add the graded ramp angle or a variable land length to a flat main ramp (cap = land depth), and deep tilted coring inside a fan, can also be built
  (Codex P2 on PR #108).
- Added volume agreement on the 0.1 mm mesh (within 2e-3) to the check conditions. Checking at cell centers alone misses boundary shifts smaller than half a cell and
  features thinner than a cell. For large blocks with a coarse mesh, the mesh is stretched by 3% so that cell centers do not land on boundaries such as t = 1, 4, ….
  `available()` also checks that every OCP name used can be imported, and the UI catches every export exception and shows only a fixed message
  (@claude review on PR #108).
- The IGES of Film gate 15 (center 5) built by hand on 10/02 has a volume of 9,725 mm³; the one built now from the same spec has 9,731 mm³ (9,731 on the 0.1 mm mesh as well).
  Compared from above, the outlines match, and only the 10/02 version is up to 0.0027 mm shallower on the ramp (within |x| ≤ 133, with the maximum around mid-width, |x| ≈ 75).
  The 10/02 version drew the parabola at the end of the land on a face 1 mm wider than the exit, with a half-width of 150 (149 is correct).
  As a result, the end of the land sat up to 0.013 mm farther back around mid-width, and the land was longer by that amount.

## [0.58.0] — 2026-10-03

**Added dimension lines to the gate-block drawing. The plan view shows the gate exit width, closure width, start of the outer wall, block depth, and valve position; each section shows the land length, the t at which the cap depth is reached, the back of the pocket, the depth, and the ramp angle.**

### Added

- `core/gate_drawing.py`: `plan_dims` (the spec values as-is) and `section_dims` (read from the section shape and snapped to closed forms), `Dim`,
  and the closed forms `land_end_t` (variable land length) / `cap_reach` (reach line and cap depth for the graded ramp angle, offset for variable land length) / `wall_end_t` (outer-wall line).
  Section boundaries are read at the midpoint of two adjacent cells (error ≤ half the mesh); if a closed form, t_max, the end of a well, or the depth of a well or cut-down
  lies within 1 cell, that value is used, otherwise the read value is shown with "≈". Shapes without a closed form, such as coring, cut-downs, corner R, and fans,
  still get the dimensions of the shape as drawn. `GateDrawing.dims` returns all dimensions.
- Sections gained a dimension band (3 tiers) above the PL and depth dimensions on the left. Scale 5:1 is kept as long as two columns fit on A3.
- Notes are placed in two columns to the left of the title block, with "≈ marks a value read from the mesh" added.
- Known limitation: the land and ramp are read on the first span continuing from the part edge, so a ramp starting behind a steel break
  gets no "t at which the cap depth is reached" or "ramp angle" (the dimensions are simply missing; no wrong values are shown).

### Verification

- Drew all default shapes of Film gate 1–15 and inspected them visually (overlap of dimension lines and text, how "≈" appears).
- Variable land length (center 5 → edge 1): land length 5 and reach 16 at the center, land length 1 and reach 12 at the edge, and angle 11.06° all come out at the spec values.
  Graded ramp angle (with graded cap depth): reach t 2.5 and depth 3.5 at the edge; the cut-down behind the ramp shows "≈" at the edge.

## [0.57.0] — 2026-10-03

**Changed the default injection-rate input to "compute from screw diameter and injection speed" (Φ50 × 200 mm/s, single stage = 392.7 cm³/s).**

### Changed

- Changed the initial selection of the "Injection rate input" radio from direct input (589 cm³/s) to computing from screw diameter and injection speed
  (per user request, 2026-10-03). 589 is the maximum injection rate from the machine manual; with multilayer wall cooling the injection rate changes the cooling time,
  and the distribution shifts by a few mm (589 → 393 changes the center-minus-edge overshoot against the 9/24 photos from 12.9 → 9.9 mm).
- The fill time in the default analysis result becomes 589/392.7 ≈ 1.5 times longer. To reproduce results from v0.56.0 or earlier, choose direct input with 589.
## [0.56.0] — 2026-10-03

**The gate-block drawing can now be generated automatically as an A3 PDF from the geometry being solved with the current inputs. Turning on "Make drawing" under "Drawing (A3 PDF)" below the part design drawing produces a drawing with a 1:1 plan view, 4 sections, a dimension table, and a title block, and the PDF can be downloaded.**

### Added

- `core/gate_drawing.py`: `render_gate_drawing(spec, plate, title=...)` returns an A3 landscape drawing as PDF and PNG.
  The geometry is built with the same `build_profile_gate_geometry` as the solver at a 0.1 mm mesh, so what is drawn is exactly what is solved
  (a stale drawing cannot remain, by construction). Any `GateProfileSpec` (single pocket, fan, one-sided) can be drawn.
  - Plan view 1:1: outline, hatching of the land (depth = land depth), contour lines every 0.5 mm, valve gate, section positions.
  - 4 sections (center, about 1/3 and 2/3 of the exit half-width, edge): 5:1. When the block is too deep to fit in two columns, it is reduced to 4:1 or less.
  - Dimension table: all values of `spec.to_dict()` listed with Japanese names (derived values: the t at which the depth cap is reached, block depth,
    pocket volume, part dimensions and wall thickness). The spec `name` is not printed (it may contain part numbers or customer names).
  - Title block: name (shape name in the UI), scale, date, revision.
- UI: for every shape that uses a spec (Film gate 1–15, Profile gate), an expander "Drawing (A3 PDF)" below the part design drawing.
  Turning on the checkbox "Make drawing" (default OFF, since drawing takes a few seconds) shows a preview and a PDF download.
  The drawing is cached by the spec and part-dimension strings and redrawn when the inputs change. Not shown for Direct gate.
- `packages.txt` (for Streamlit Cloud): `fonts-ipaexfont-gothic`. Japanese PDFs need a TrueType Japanese font
  (embedding a CFF font such as Noto Sans CJK with `pdf.fonttype = 42` turns glyphs into different characters).
  `japanese_font()` prefers IPAex / Meiryo / BIZ UD and otherwise uses a CFF font with `pdf.fonttype = 3`.

### Verification

- Drew all default shapes of Film gate 1–15 and inspected them visually. Pocket volumes match the previously checked values (FG4 5,283, FG11 9,762,
  FG14 12,702, FG15 9,731 mm³).

## [0.55.0] — 2026-10-03

**Reorganized the injection-rate input into two options: "direct input" and "compute from screw diameter and injection speed". The compute option defaults to a single stage and takes only the screw diameter and injection speed. Shot-volume position, V/P switchover position, and per-stage speed switchover positions and speeds appear only when the number of injection stages is 2 or more.**

### Changed

- The options of the "Injection rate input" radio are now "Enter injection rate directly" (default, 589 cm³/s) and "Compute from screw diameter and injection speed"
  (formerly "Compute from machine conditions"), with direct input listed first.
- The default number of injection stages on the compute side changed from 3 → 1. With a single stage, `Q = πD²/4 × v` is passed to the solver as a constant injection rate
  (`injection_profile` is None). Φ50 × 200 mm/s gives 392.7 cm³/s. The caption shows the injection rate and the formula.
- With 2 or more stages, an `InjectionProfile` is built from shot-volume position, V/P switchover position, speed switchover positions, and per-stage speeds, as before.
  With 3 stages, the machine's switchover positions 30 → 28 → 22 → 18 are the initial values (other stage counts are divided equally).
  Per-stage speeds start from the speed entered for the single stage, and returning to a single stage carries over the first-stage speed.
- The warning shown when the constant-pressure clock is chosen for the skin layer is shown on the compute side regardless of the stage count (a single stage is also velocity-controlled).
- Added `machine` to `injection` in settings.json: single stage on the compute side is `{screw_diameter_mm, stages: 1, velocity_mms}`,
  2 or more stages is `{screw_diameter_mm, stages}` (positions and speeds are in `injection_profile`), direct input is null.

### Background

- If all stages run at the same speed, shot-volume position, V/P, and stage count have no effect on the result. The two-phase shot volume is a separate input (default: cavity volume),
  and the shot-volume position and V/P are used only for speed switchover points and the "theoretical shot volume is insufficient" warning. Asking for positions with a single stage
  only added input effort and a source of typos.
- The direct-input 589 cm³/s is the maximum injection rate from the machine manual. In the multilayer model the injection rate changes the cooling time and shifts the distribution by a few mm
  (589 → 393 changes the center-minus-edge overshoot against the 9/24 photos from 12.9 → 9.9 mm), so use the compute side to match the actual machine.
  Direct input was kept as the default for compatibility with existing results.

## [0.54.0] — 2026-10-02

**Added variable land length (a coat-hanger land) and added Film gate 15 (fan-shaped/variable land length), which uses it by default. The land depth stays the same; the land is longer at the center and shorter at the edges, so the pressure loss of running the transverse runner to the edges is balanced by the central land.**

### Added

- `LandProfileSpec(center_length, power=2.0)` and `LandSpec.profile` (JSON `land.profile`). Single-pocket shapes only.
  The land ends at `L(w) = length + (center_length − length)·(1 − w/w_edge)^power` (w=0 is the valve axis for symmetric shapes,
  the valve-side end for one-sided shapes; w_edge is the exit half-width / width), and the main ramp (including the graded ramp angle) starts from that end in the same shape
  = it is read shifted by L(w) − length in the t direction. The outline, closure (read with the original land), and overrides such as coring are
  not shifted. power 2 (parabola) is the shape that cancels the transverse-runner pressure loss ∝ (w_edge² − (w_edge − w)²)/2.
  Validation: single-pocket shapes only, `center_length > length`, `power > 0`. **Specifications whose finished geometry is identical to having no profile are
  rejected at assembly** (overrides such as coring cover the whole extended range, or the mesh is too coarse to have a cell center there).
- Under the land of every single-pocket Film gate, a block "Vary land length across the width" (center land-length slider,
  shape exponent, caption with the land length at w=0, 1/3, 2/3, and the edge). Default ON only for 15.
- Film gate 15 (fan-shaped/variable land length): proposal A3 of 2026/10/02 (drawing by Monday,
  `hamoko_gate_furiwake_landhanger_center5_20261002`, no 3D model). Takes the frame of Film gate 14 (kai02)
  (outer wall, well, valve, ramp to 2.5 at 11.06°), removes the grading, closure, and coring, and makes the land a parabola from center 5 → edge 1.

### Background

- Resistance concentrates where it is thinnest (∫dt/h³: 1 mm of 0.35 land ≈ 23, the whole 11° ramp ≈ 20, deeper than 0.8 ≈ 0).
  Neither a full-width coat hanger extending the deep ramp (A1) nor a full-width weir with a 0.5–1.0 gap (A2) removed the center lead of +20–45 mm.
- Multilayer without thermal coupling (the condition that best matched the 9/24 pre-compression short shots): flow-front waviness at end of injection 2 mm / center-minus-edge +0.4 mm /
  arrival spread along the side opposite the gate 6.0 % (the previous best, "grading 2.5 + coring + closure 10", was 10 / −10.0 / 10.5 %). With thermal coupling
  the optimum shifts toward center 6 (+12.9 at 5, +4.4 mm at 6). ±0.5 mm of center land length moves center-minus-edge by ±4–5 mm.

## [0.53.0] — 2026-10-01

**Extended the apex method for land closure to Film gate 9–13 as well. When the part-side width is changed, they all move the same trapezoid as FG14 around the apex t=21.224.**

### Changed

- From Film gate 9 on, the coring boundary is a single straight line (w=50 at t=0, 47.64 at t=1, 9.9 at t=17), which, extended, meets the axis at
  t=21.197 (reading of the 9/14 PDF) / 21.224 (CAD values from Film gate 12 on, 47.644148 → 9.95051). It is the same line differing only in the digits read,
  so 9–14 all have the CAD intersection t = 50/(50 − 47.644148) = 21.224 as their apex
  (`_KAI_CLOSURE_APEX_T`; 10–14 inherit `_FILM_GATE9_DEFAULTS.land_closed_apex_t`).
- The default geometry does not change (13 and 14, where closure is ON by default, are bit-identical with 100 → 95.288. 9–12 are OFF by default).
  What changes is moving the width: at 60, the old 9–13 had the coring (47.64) wider, giving a 30 → 30 rectangle,
  but now it is a 30 → 28.586489 trapezoid.
- Film gate 1–8 come from different drawings and stay as before (the land end follows the coring boundary if the coring is narrower, otherwise it is straight).

## [0.52.0] — 2026-10-01

**Made the land closure of Film gate 14 a single slider. The slanted side of the trapezoid passes through the apex, so moving the part-side width moves the land-end width and the angle of the slanted side together.**

### Changed

- Added an "apex" method to the land closure (a trapezoid seen from the PL; the part side t=0 is the long side, the land end the short side):
  for a Film gate with `_ProfileGateDefaults.land_closed_apex_t`, the slanted side is a straight line through (w=0, t=apex), and
  the land-end width = part-side width × (1 − land length / apex t) is derived (rounded to 1e-6 mm). The apex of Film gate 14 is
  the intersection t = 50/(50 − 47.644148) = 21.224 of the extended slanted side in the kai02 CAD (0.28 short of the valve center; the same point as the extended coring
  boundary). At the default 100 it is identical to the CAD (95.288, slanted side 67.0°); at 60 it is 57.173 and 54.7°. A caption below the block
  shows the land-end width and the angle of the slanted side.
- Removed the v0.51.0 checkbox "Specify the closure width at the land end" and the second slider (replaced by the apex method).
  In the old method the land-end width was independent of the part side, so on a 1.0 mm mesh, which sees the land only in the single row t=land length,
  moving the part-side width did not change the geometry at all.
- Renamed the slider to "Closure width (part side t=0, …)" (so it is clear which side it is; common to all single-pocket Film gates).
  For Film gates without an apex (1–13), the land-end side is derived from the coring boundary as before.

## [0.51.0] — 2026-10-01

**Added Film gate 14 (fan-shaped/graded ramp angle 3) to the UI. The end width of the land closure can now be specified independently of the coring.**

### Added

- `Film gate 14 (fan-shaped/graded ramp angle 3)` in the sidebar. Its inputs default to the 2026/09/30 CAD model `Runner-block_3D_kai02.igs`
  (`hamoko_gate_furiwake_rampends3_20260930`); it is the Film gate 13 pocket with the coring and the R10 corner of the outer wall
  removed. The center |w| ≤ 60 is a single ramp (11.06°) over t=1–12 with a floor of depth 2.5 behind it, and the edge walls meet the 3° outer wall with a sharp corner.
  The graded ramp angle at both ends (cap depth 2.5 → 3.5), outer-wall line, well, and land closure (exit w=50 → land end w=47.644) are
  face-for-face identical to kai01. Compared cell by cell at 0.5 / 0.25 mm against the depth field from vertical ray casting of the CAD: the outline matches exactly,
  the maximum depth difference is 0.019 mm (B-spline approximation), and the pocket volume is 12,704 mm³.
- In the land-closure block, a checkbox "Specify the closure width at the land end" and a slider "Closure width at the land end".
  OFF is as before (the end edge follows the exit-side boundary of the coring; straight if coring is OFF). kai02 has no coring yet
  keeps the chamfer to 47.644, so it could not be expressed by a shape derived from the coring (cutting the coring in FG13 makes the edge straight,
  5 cells short per side at 0.5 mm). `_ProfileGateDefaults.land_closed` also accepts a `(exit width, end width)` pair, which starts ON.
- `_ProfileGateDefaults.island_on`: initial state of the coring checkbox (OFF only for FG14).
- Tests: 6 in `test_film_gate14_ui.py` (defaults match the spec / default geometry is bit-identical to reading the spec directly / the center is a single ramp and
  the closure keeps the chamfer / relative to FG13 only the coring band and corner change / the end-width setting determines the chamfer / defaults do not leak into FG13).

## [0.50.0] — 2026-09-30

**Changed the initial conditions when the page opens. The injection rate starts at direct input 589 cm³/s and wall cooling starts at multilayer (N=7).**

### Changed

- The default of "Injection rate input" is now `direct` (589 cm³/s). In v0.42.0–v0.49.0 the machine conditions (screw diameter, positions, speeds) were the default.
  The initial values when switching to machine conditions (φ50, shot volume 30, V/P 18, 3 stages at 200 mm/s) are unchanged.
- The default of "Wall-cooling representation" is now "multilayer" (N=7 / wall_refined / 12 iterations remain the previous multilayer defaults). In v0.39.0–v0.49.0 it was
  "skin layer". Since v0.48.0 the multilayer model rides on the injection phase of the two-phase short shot, so the two-phase analysis (ON by default) still runs.
- Library and CLI defaults are unchanged (only the initial UI selection).
- Tests: `test_injection_ui.py` gained 1 test checking the state on page open (direct input 589); the rest now start by switching to machine conditions and the skin layer.
  Updated the default check in `test_two_phase_ui.py` to multilayer N=7.

## [0.49.0] — 2026-09-30

**Added Film gate 13 (fan-shaped/graded ramp angle 2). The 2026/09/30 3D model `Runner-block_3D_kai01.igs` can be built as-is.**

### Added

- `RampEndsSpec.depth_end` (graded cap depth): within the graded zone the cap depth also changes linearly from `cap_depth` (`w_from`) → `depth_end` (pocket edge).
  The ramp is a straight line from the land end to `(t_cap(w), D(w))`, and the floor (runner) behind the reach line also continues at `D(w)` up to the outer wall.
  This is exactly the CAD edge faces (bilinear patches (1, 60, 0.35)–(12, 60, 2.5) / (1, 149, 0.35)–(2.5, 149, 3.5)) and the back floor (a ruled surface at 2.5 at w=60
  and 3.5 at w=149). When omitted, the cap depth is constant as before. Values ≤ `land.depth` are rejected.
- `LandSpec.closed_line` (closure at the land center): within the land (`t ≤ land.length`), cells with `w < closed_line(t)` become steel.
  No resin enters the part from there. Center for symmetric shapes, valve-side end for one-sided shapes. In the CAD it slants from exit w=50 to w=47.644 at t=1
  (start of the coring). Single-pocket shapes only. Rejects lines that close the entire exit width, settings that close not a single cell, and rasters where the side opening is narrower than a cell
  so the part is cut off from the gate.
- `GateProfileSpec.outer_wall_corner_radius` (corner R of the outer wall): rounds the corner where the full gate-exit width turns into the outer-wall line with an arc tangent to both.
  Rejects R whose tangent length `R·tan(bend angle/2)` does not fit within the outer-wall start t and the outer-wall line length, shapes whose outer-wall line does not start at the exit width,
  and R that removes not a single cell. The CAD corner is R10 (the tangent point on the end wall, t=6.2466, matches the control point).
- UI: Film gate 13 (tag `f13`, defaults = CAD values). Every single-pocket Film gate gained a "Land closure" block (closure width;
  the land-end side follows the exit-side boundary of the coring), a "Cap depth at pocket edge" slider for the graded ramp angle, and "Outer-wall corner R" for the outer-wall line.
  Default ON / nonzero only for 13.
- Spec `hamoko_gate_furiwake_rampends2_20260930` (`local_specs`). Against the depth field from vertical ray casting of the CAD at 0.25 mm, the
  outline matches exactly cell by cell; the only depth difference is the 60° chamfer at the back end of the coring (vertical in the spec), 11 mm³ (0.1% of the CAD's 11,776 mm³).
- 20 tests: 12 in `test_geometry_profile_gate.py` (columns and back floor of the graded cap depth checked against the closed form / only deeper than the constant cap /
  bit-identical to the constant cap when equal to the cap / round-trip and validation / exact match and mirror symmetry of cells closed by the land closure / part before the closure fills
  by flowing around / round-trip and validation / rejection of 0 cells / corner-R removal within ±5% of the closed form `R²(tan(θ/2) − θ/2)`, only between the tangent points and outside the circle / round-trip and
  validation / rejection of 0 cells / rejection of rasters where the side opening of the closure is narrower than a cell and cuts off the part), 8 in `test_film_gate13_ui.py` (the cap-depth slider is not shown when the ramp cap is at the slider maximum).

## [0.48.0] — 2026-09-29

**The multilayer model now rides on the injection phase of the two-phase short shot. Choosing multilayer no longer skips the two-phase analysis.**

### Added

- `solve_two_phase_short_shot` accepts a `MultilayerHeleShawSolver`. The injection-phase τ is the multilayer fixed point
  (τ ↔ t_arr ↔ T_k ↔ η_k) solved on the mold-open gap, and the clock is the metered velocity control: the volume-CDF mapping uses the length `T_open_total` needed to fill the whole open cavity
  at the injection rate (through the profile if there is one), and does not inflate. The main analysis's ICM time shortening
  (`compression_fraction`) and constant-pressure inflation (`T_fill_inflation`) do not affect the injection phase. Same definition as the skin-layer injection phase.
- `MultilayerHeleShawSolver._fixed_point(h_open, dirichlet, T_fill_baseline, *, rate_controlled=False)`:
  an internal method carved out of the fixed-point loop of `solve()`. `rate_controlled=True` fixes the clock. The results of `solve()` are
  bit-identical to before the extraction (τ, fill time, and layer temperatures checked on `hamoko_gate_furiwake_rampends_20260928` with 7 layers and viscous heating ON).
- `wall_model` (`none` / `skin` / `multilayer`) in the two-phase metadata. For multilayer, also `num_layers`, `layer_distribution`,
  `thermal_coupling`, `shear_heating_enabled`, `multilayer_iterations`, `multilayer_converged`, and the diagnostic
  `injection_center_solid_cells` (number of cells in the pool at end of injection whose center layer fell below the solidification temperature). The settings record's
  `two_phase_short_shot` also has `wall_model`.
- UI: two-phase also runs with multilayer, and the result pane shows the layer count and fixed-point convergence in a caption. Warns if there are cells with a solidified center layer.
- 5 tests (4 in `test_two_phase.py`, 1 replacement in `test_two_phase_ui.py`): with 1 layer and no thermal coupling, the pool and final shape match the isothermal two-phase /
  the injection phase does not depend on `compression_fraction` and arrival times are within `T_inj` / with slow injection (2 s) the thick
  runner fills before the open thin plate, and with fast injection (0.2 s) the isothermal order is kept / volume contract (a partial shot is at or below the shot volume,
  the cavity volume fills completely) / in the UI, multilayer two-phase runs and is recorded.

### Changed

- Removed the `ValueError` "two-phase cannot be combined with multilayer" from `run_demo.py`.

### Limitations (intentional)

- The compression phase stays isothermal as before (the pool is an isobaric source advancing with a single representative viscosity). Layer temperatures do not affect advance during compression.
- The multilayer model has no freeze-off time, so nothing corresponding to the skin layer's `injection_sealed_mask` is output. Cells whose center layer fell below
  the solidification temperature are still treated as flow paths in the compression phase, and only their count is reported as a diagnostic.
- Melt does not reach cells beyond the pool, but they affect τ through the conductance. Their temperature is read at the arrival time "if injection had continued"
  (the multilayer temperature is a function of arrival time and has no exposure cutoff like the skin layer).

## [0.47.0] — 2026-09-29

**Made the horizontal section (weld dam) independent of the coring. With the coring kept as the base, only the range and width of the dam placed on top of it can be set.**

### Changed

- Carved the sidebar's horizontal section (weld dam) block out into `_weld_inputs` / `_weld_from_inputs` and gave it dimensions separate from the coring.
  In the old version the dam end was fixed to the coring end and its width to the coring boundary, so the only way to resize the dam was to
  move the coring itself. The new sliders are the range t (2 handles, land length to coring end), width (max = full coring width), and
  distance from the PL. The limits follow the coring dimensions, but moving the dam does not change the coring. When the upper end is the coring end and the width is the maximum,
  it is recorded as "follows the coring" (the Film gate 2 default is t=7 to coring end, full width, as before).
- When shrinking the coring cuts the dam, that is not treated as an edit (`{tag}_weld_mem` keeps only manually moved values).
  Restoring the coring restores the dam to its original dimensions. Streamlit rebuilds sliders whose bounds change and drops them to the default,
  so the remembered value is passed again as `value`.

### Added

- `WeldSpec.w_max` (optional, JSON key `w_max`): the range of the dam across the width, `w ≤ w_max` (w in the same coordinate as the coring boundary).
  The dam shape is always "coring ∩ t_range ∩ w ≤ w_max" and never goes outside the coring. `None` (omitted) is the full coring width = bit-identical to the old
  behavior. Validates `w_max > 0`.
- Specifications whose finished geometry is identical to having no dam are rejected at assembly (narrower than the mesh, range picks no cell center, fits under a well
  and is erased by the well's max, erased by edge deep-cut or cut-down). The check compares against the finished geometry rebuilt from the spec without the dam
  (`_reject_ineffective_weld`. Looking at the field right after would miss a well applied later; Codex P2 on PR #95).
  **Existing specs containing a "dam that does not appear in the geometry", which the old version silently accepted, will now raise `ValueError` at assembly (`st.error` in the UI).**
  The Film gate 2 default is unaffected.
- Tests: 5 in `test_geometry_profile_gate.py` (only the w_max box changes and the coring beside it is unchanged / w_max wider than the coring
  is bit-identical to full width / round-trip and omission / validation / rejection of a dam that picks no cell / rejection of a dam erased by a well) plus 1 for direct construction with non-finite values.
  Replaced the boundary test in `test_film_gate2_ui.py` and added 2 (moving the dam sliders leaves the coring record
  unchanged and only cells inside the box change / shrinking and restoring the coring restores the dam).

## [0.46.0] — 2026-09-28

**Added Film gate 12 (fan-shaped/graded ramp angle) to the UI. Added `ramp_ends`, which grades the ramp angle at both ends, to the spec.**

### Added

- `GateProfileSpec.ramp_ends` (`RampEndsSpec(w_from, t_end)`, single-pocket shapes only). For `w ≥ w_from`, the line where the ramp reaches the cap depth
  is bent straight from the main reach line `t = ramp_cap_t()` to `t_end` at the exit edge, and each w section becomes a straight slope from the land end
  to that line (a ruled surface). The ramp angle changes continuously with w and joins the main ramp without a step at `w_from`.
  It replaces the main ramp (it is not a floor); coring, edge deep-cut, cut-down, and land-end thickening are applied on top of it
  as before. Validation: `0 ≤ w_from <` exit half-width (exit width for one-sided), `t_end > land.length`, rejected for fan shapes.
  Specifications whose finished geometry is identical to having no grading (including when erased by land-end thickening applied later) are rejected at assembly.
- `Film gate 12 (fan-shaped/graded ramp angle)` in the sidebar. Its inputs default to the 2026/09/28 CAD model `Runner-block_3D_00.igs`
  (`hamoko_gate_furiwake_rampends_20260928`); at both ends of the Film gate 9 pocket (|w| ≥ 60) the
  reach line for depth 2.5 moves from t=12 → t=4 and the ramp angle changes from 11.06° → 35.6°. All other dimensions are also read from the CAD to full precision
  (less than 0.11 mm from the FG9 PDF reading). Compared cell by cell at 0.5 / 0.25 mm against vertical ray casting of the CAD, the outline
  matches exactly; the only depth difference is the 60° chamfer at the back end of the coring (vertical in the spec), 11 mm³ (0.1%).
- The block "Graded ramp angle (both ends)" appears below the main ramp of every single-pocket Film gate (default ON only for 12).
  The sliders are the width of the constant-angle center and the t at which the ramp reaches the cap depth at the pocket edge. The caption shows the center and edge ramp angles.
  Disabled when there is no ramp (cap = land depth).
- `_ProfileGateDefaults.ramp_angle_deg`: the sidebar's default ramp angle is held per drawing (11.06° only for FG12, the previous 10.95° for the others).
- 19 tests: 11 in `test_geometry_profile_gate.py` (columns match the ruled-surface closed form / center, land, and outline unchanged and mirror-symmetric /
  starts without a step and is monotonic toward the edge / volume within ±3% of the closed form / a replacement, not a floor / coring keeps its own depth / one-sided only at the far edge /
  round-trip / validation / rejection of no-change settings / rejection of settings erased by a later floor), 8 in `test_film_gate12_ui.py`.

### Known differences

- The CAD coring has a 60° chamfer at its back end (t=16.14–17), while the spec coring is cut vertically. As a flow path it is a 0.86 mm wide band in the deep
  region in front of the valve, with a volume of 11 mm³.
- The CAD has no valve, so Φ3 and t=21.5 are the Film gate 9 values.

## [0.45.0] — 2026-09-23

**Added Film gate 11 (fan-shaped/variable land thickness at both ends) to the UI. Added land-end thickening `land_ends` to the spec.**

### Added

- `GateProfileSpec.land_ends` (`LandEndsSpec(w_from, depth)`, single-pocket shapes only). Cuts the pocket at `w ≥ w_from` down
  to a plane at `depth` from the PL (`d = max(d, depth)`). The ramp face is untouched, so the flat extends to where the ramp reaches `depth`,
  `t = land.length + (depth − land.depth)/tan(ramp angle)`, and merges there without a step (the land becomes deeper and longer at the edges).
  The step at `|w| = w_from` is at most `depth − land.depth` and vanishes to 0 at that t. `w` is the same coordinate as elsewhere
  (both ends for symmetric shapes, only the far end for one-sided). Validation: `land.depth < depth ≤ main_ramp.cap_depth`,
  `0 ≤ w_from <` exit half-width (exit width for one-sided), rejected for fan shapes. Settings that deepen not a single cell (`w_from` beyond the last cell center) are
  rejected at assembly.
- `Film gate 11 (fan-shaped/variable land thickness at both ends)` in the sidebar. Its inputs default to candidate A "land-end thickening" of 2026/09/23
  (`hamoko_gate_furiwake_cand_A_landends05_20260923`); the Film gate 9 pocket with 49 at each end
  (|w| ≥ 100, the center 200 unchanged) cut to depth 0.5. The flat extends to t=1.78. Pocket volume on a 0.1 mm raster is
  9,762 mm³ (FG9 + 20, matching the closed form `2·49·(0.15·1 + 0.15·0.78/2)`).
- The block "Land-end thickening" appears below the main ramp of every single-pocket Film gate (default ON only for 11). The sliders are
  the unchanged center width ("unchanged width (from the valve-side end)" for one-sided) and the land depth after thickening. The caption shows the end t of the flat, the maximum step height,
  and the conductance ratio of the edge land `(depth/land)³`.
- 15 tests: 7 in `test_geometry_profile_gate.py` (only the ends, up to the end of the flat, mirror-symmetric / columns are `max(ramp, depth)` /
  volume within ±3% of the closed form / one-sided only at the far end / round-trip / validation / rejection of settings that deepen nothing),
  8 in `test_film_gate11_ui.py` (defaults match the spec, bit-identical, the flat exists, the difference from FG9 is only 98 cells in the land row,
  sliders follow, OFF gives FG9, no leakage into 9 / 10, disabled when ramp cap = land depth).
- When the ramp cap depth equals the land depth (a setting the slider allows), there is no depth to thicken to, so the checkbox is
  disabled with the reason shown. Previously the depth slider could only produce values above the cap and runs always failed validation (Codex P2 on PR #93).

### Changed

- The candidate A spec (`local_specs`) could not express `land_ends` before and approximated it with a 1.8 wide runner band.
  Rewrote it as `land_ends: {w_from: 100, depth: 0.5}`.

## [0.44.0] — 2026-09-23

**Added Film gate 10 (fan-shaped/thickness adjustment 0923) to the UI. Added the cut-down behind the ramp, `ramp_cut`, to the spec.**

### Added

- `GateProfileSpec.ramp_cut` (`RampCutSpec(line, depth, slope_angle_deg=90)`, single-pocket shapes only).
  Reads `line` as a function of w, `t_cut(w)`, and cuts the pocket behind the line flat with `d = max(d, depth)`.
  The step left on the line where the ramp has not yet reached `depth` is replaced by **a slope cut toward the part side**,
  `d = max(d, depth − (t_cut − t)·tan(slope))` (90° leaves the step as-is). Floor semantics, so the land and
  silhouette are unchanged, and laterally it covers exactly the line's w interval (inside the line's end w nothing is machined — the line stops without bending).
  Validation: t increasing, w strictly decreasing, `t1 > land.length`, positive depth, angle in (0, 90], **angles whose slope would be deeper than `land.depth`
  at the land end are rejected as "too shallow"** (boundary = `atan((depth − land.depth)/(t1 − land.length))`).
  Settings that deepen not a single cell are rejected at assembly (the same false green as edge deep-cut).
- `Film gate 10 (fan-shaped/thickness adjustment 0923)` in the sidebar. Its inputs default to the 2026/09/23 drawing "Runner block, thickness-adjusting gate (split)";
  the Film gate 9 pocket with the back of the ramp cut to depth 2.5 (the red-filled area in the drawing). The step line is parallel to the 3° outer wall
  with t=7.451 at the edge, and merges with the depth-2.5 reach line at w=60 (the center 120 is not machined). The default slope angle is 18.43°, at which the edge slope runs
  in one piece from the land end to the step line (the minimum angle with zero step and the land intact). On a 0.25 mm raster the gate-block volume is
  FG9 + 297 mm³ (+ 125 at 90°).
- The block "Cut-down behind the ramp (step line)" appears for every single-pocket Film gate, like edge deep-cut (default ON only for 10).
  The sliders are the line position t at the edge, the unmachined width, depth, and slope angle; the line's end point is fixed at `(t where the ramp reaches the cap, end of the unmachined width)`.
  The caption shows the line angle, the step height at the edge, and the "angle that leaves no step" for the current line.
- 15 tests: 8 in `test_geometry_profile_gate.py` (range and mirror symmetry of the step / stops at the line end / slope profile of the edge column at
  90, 45, 30° / monotonic volume / removed step volume within ±3% by double quadrature / round-trip / validation / rejection of settings that deepen nothing),
  7 in `test_film_gate10_ui.py` (defaults match the spec, bit-identical, the cut-down exists, the difference from FG9 is only deepening at w ≥ 60,
  the step at 90°, rejection of angles that cut the land, no leakage into 9).

## [0.43.0] — 2026-09-22

**Added Film gate 9 (fan-shaped/late-September prototype) to the UI.**

### Added

- `Film gate 9 (fan-shaped/late-September prototype)` in the sidebar. Its inputs default to the 2026/09/14 drawing "September 14 runner proposal";
  the 0807 modified pocket with **only the outer-wall line** moved. The current line (narrowing at 8° from the land side of the pocket edge
  toward the well) is lowered to a line starting at t=15.736 at the edge and running at 3° to the same end point (the "5°" in the drawing is the angle between the two lines).
  With this, the depth-2.5 reach line at t=12.11 no longer stops at the outer wall but reaches both ends of the pocket, and the back of the ramp becomes a nearly full-width
  band of depth 2.5 (3.6 mm long at the edges, about 11 mm at the center) — a transverse runner feeding the full film width rather than a fan.
  Land 1×0.35, ramp 10.95°, coring (exit width 100, 2.5°), and valve t=21.5 are the same as the 0807 modification; the well is the same as the 0703 drawing with
  a 60° wall angle (floor 18.1–24.9). The gate-block volume on a 0.25 mm raster is 9,741 mm³ (matches an independent integration).
- Just passes the defaults `_FILM_GATE9_DEFAULTS` to the existing `_profile_gate_sidebar`, so the slider layout is
  the same as Film gate 1 / 2 / 6 / 7. Widget keys are `f9_`. Edge deep-cut and the horizontal section are OFF by default.
- `tests/test_film_gate9_ui.py` (6 tests): default sliders match the drawing's spec (the outer-wall start 15.736 is not
  rounded to the slider's step 0.1) / outer wall is 3° / the default geometry is bit-identical in mask, wall thickness, and gate to reading the spec directly /
  **the back band exists in the thickness field** (at edge w=148.5, t=13–15 is 2.5 and t=16 is steel; at w=100.5, t=18 is 2.5 and t=19 is steel;
  the same cells are steel in the modified pocket) / **it only adds cells relative to the modified pocket** (existing cells keep their thickness, the added area is
  two triangles bounded by 2 straight lines, and the added depth is only the ramp or 2.5) / switching 9 → 6 does not leak the outer-wall start or band defaults.

### Reading the drawing (well)

- The well wall angle is 60° from the enlarged section view. The floor 18.1–24.9 matches an oblong of R4.5 centered at t=20 / 23 dug 4.5 deep at 60°
  (4.5 − 4.5/tan60° = 1.9). The R3 oblong in the plan detail (17 / 20 / 23, width 6) is read not as the floor but as the ridge where its wall meets the depth-2.5
  runner floor (4.5 − 2.5/tan60° = 3.06). The 71.6° of Film gate 2 / 6 is the value from reading the same oblong as the floor, and may be
  the same misreading (the 0807 drawing has no section, so this is unconfirmed and awaiting the designer's confirmation; not touched this time).

## [0.42.3] — 2026-09-15

**Fixed 4 Codex P2 findings (detected in the port PRs shostako/mold-flow-fangate#16 and shostako/mold-flow-fangate2#8).**

### Fixed

- **Inserting `injection_profile` in the middle of the dataclass shifted the positional arguments.** Up to 0.41 the
  7th positional argument was `compression_molding`; a call that passed a bool there now bound the bool to
  `injection_profile` and reached `bool.time_at_volume_mm3` inside `solve()`.
  Made it `field(kw_only=True)`, keeping its position in the source (next to the injection rate it replaces) while
  moving it to the end of `__init__`. Same for `MultilayerHeleShawSolver`.
- **The multilayer solver's metadata lacked `injection_Q_effective_cm3s`.** `HeleShawSolver` emitted it but the
  multilayer solver did not, so the `FlowResult` / `MultilayerFlowResult` contracts were asymmetric.
- **The two-phase extrapolation flag was raised whenever the open cavity exceeded the stroke, even if the shot volume was within it.**
  `T_open_total` also goes through the mapping, but only as the normalization factor of the injection phase's arrival-time field,
  where it cancels out — what comes out is each cell's mapped value at its own cumulative volume, and cells beyond the shot volume
  arrive after `T_inj`, so they fall out of both the pool and the skin clock. The extrapolated values never reach the report.
  The check now uses `V_shot` only, and the warning text was changed to "the shot volume exceeds the theoretical injection volume".
- **The extrapolation warning in the main result pane compared only against the final cavity volume.** With ICM ON,
  the volume CDF mapping reads the volume of the **open gap**, so even when the stroke covers the final geometry there is a band
  where the mapping reads beyond V/P. Moved the check to the solver side (`injection_extrapolated_past_vp` and
  `injection_swept_volume_cm3` in the `solve()` metadata); the UI reads that flag.

## [0.42.2] — 2026-09-15

**Made "velocity-controlled" the UI default for the skin-layer clock.**

### Changed (default behavior)

- The UI default for "Skin-layer clock" changed from `constant_pressure` to **`constant_rate`**. Now that the default injection input is
  the actual machine conditions (screw position and speed), leaving only the clock on the "machine cannot hold the speed" side
  means **a screen fed the machine's set injection time returns a fill time that does not match the setting**.
  Under velocity control `T_fill = V/Q` (`T_fill_inflation = 1.0`), and the added resistance shows up in skin thickness and pressure.
  Constant pressure remains as an option (for reproducing existing results). The stretching notice shown when constant pressure
  is chosen in machine-condition mode is still displayed as before.
- **The library default of `HeleShawSolver.skin_clock_mode` remains `constant_pressure`.** Only the UI's initial selection changed;
  CLI cases and existing calls do not move at all (the default arguments of `run_demo.py` are also unchanged).

## [0.42.1] — 2026-09-14

**Closed the path where the two-phase short shot silently extrapolated beyond V/P, and aligned the injection-condition error messages
with the sidebar labels.** (@claude review on PR #89)

### Fixed

- **Two-phase short-shot extrapolation had no warning.** The extrapolation warning in the main result pane only compared the theoretical
  injection volume with the **final** cavity volume, independent of both the two-phase shot volume and the mold-open gap volume.
  Tightening the V/P position or increasing the shot volume made `T_inj` / `T_open_total` extrapolated values at the last stage's
  injection rate, yet the two-phase panel showed nothing. The metadata of `solve_two_phase_short_shot` now carries
  `injection_extrapolated_past_vp` and `injection_profile_volume_cm3`, and the panel reads them to warn (held as a property of the
  run result, not of the current sidebar values).
  Also added tests, since the combination "two-phase ON and machine conditions ON" had never been exercised.
- **Injection-condition errors used internal names.** `stages[1].end_position_mm` is 0-based English, while the sidebar label is the
  1-based Japanese `第2段 速度切替位置`. It was impossible to tell which slider was wrong. `_injection_error_ja()` translates the stage
  number, field name and stock phrases into the sidebar's wording ("第2段の速度切替位置 (22.0) は 第1段の速度切替位置 (19.0) より
  小さくしてください"). The reproduction path named in the review (lower the switch-over position with 2 stages, then increase to
  3 stages) is pinned as a regression test.

## [0.42.0] — 2026-09-14

**The injection rate can now be derived from the molding machine's own settings (screw diameter, position, speed), and with
multi-stage injection the fill-time axis bends at each stage.** The injection speed slider only affected the representative shear
rate, i.e. viscosity, and the total fill time was set by the separate injection-rate input. Its default of 589 cm³/s is the
**maximum** injection rate from the manual, so every condition was pictured as if the machine ran wide open (Issue #88).

### Added

- `core/injection_profile.py`: `InjectionProfile` / `InjectionStage`. From the screw diameter, metering position, and each stage's
  speed switch-over position and speed, it produces the per-stage injection rate `Q_i = πD²/4·v_i`, time, injected volume, and a
  **piecewise-linear volume → time mapping**. Each stage's injected volume `πD²/4·L_i` depends only on the stroke, not on speed,
  so the breakpoint volumes are fixed and only the slope (injection rate) changes per stage. Volume beyond V/P is
  extrapolated at the last stage's injection rate (the fill-time field needs this when the cavity is larger than the theoretical injection volume).
- `HeleShawSolver.injection_profile` / `MultilayerHeleShawSolver.injection_profile`. When set, it **takes precedence** over
  `injection_volume_flow_cm3s`, and the volume CDF mapping goes through this mapping. Normalization to `T_fill` is unchanged,
  so composition with skin inflation and ICM shortening does not change — only the shape in between changes.
- An input-mode radio in the UI "Injection conditions". **The default is "Compute from machine conditions"**, with defaults taken
  from the actual machine settings (screw diameter 50 mm, metering position 30 mm, V/P 18 mm, 3 stages 30→28→22→18 all at 200 mm/s).
  Number of injection stages 1–5; at 2 or more, fields for each stage's speed switch-over position and speed open. The caption shows the
  average injection rate, injection time to V/P, and theoretical injection volume at V/P (with the per-stage breakdown for multi-stage).
- CLI `_solve_and_export(injection_profile=...)` and the reference case `FilmGate_PP_staged_injection`
  (50 mm screw at 30→28.25→18, 20 mm/s then 200 mm/s. The switch-over is placed at half the cavity volume so the
  breakpoint falls in the middle of the part).
- In the result pane, a caption for the injection conditions (average injection rate, time to V/P, theoretical injection volume) and
  **a warning when the theoretical injection volume is below the cavity volume**. Fill times beyond V/P are extrapolated at the last
  stage's injection rate, so this makes it explicit that they should not pass as predictions. It reads the run result's
  metadata, not the current sidebar values (they diverge when a cached result is drawn).
- `tests/test_injection_profile.py` (46 tests) and `tests/test_injection_ui.py` (16 tests).
  On a 1D strip of equal cells the volume CDF is the cell index itself, so arrival times can be checked cell by cell against the
  profile's own mapping without going through the solver.

### Changed

- "射出速度 [mm/s] (代表)" → "代表流動速度 [mm/s]" (representative flow speed). **It is a different quantity from screw speed**:
  the gap-averaged flow velocity in the cavity, which only enters the `6V/h` representative shear rate (an order of magnitude larger in thin plates).
  It remains an independent input — deriving it from Q would require deciding how to take the representative cross-section `W·h`,
  which is geometry-dependent and arbitrary.
- `injection_profile` (including the per-stage breakdown) and `injection_Q_effective_cm3s` in `metadata`.
  `rate_input_mode` and `injection_profile` under `injection` in `settings.json`.
- The two-phase short shot's `T_inj` and open-cavity fill time also go through this mapping (no longer fixed `V/Q`).
  The default shot volume **still follows the cavity volume as before** — the theoretical injection volume is display-only and not linked.
- `_arrival_time_field` changed from `@staticmethod` to an instance method (to read the profile).
  The path without a profile keeps the same formula, and existing results match bit for bit.

- A notice for machine conditions × the skin-layer "constant pressure" clock. Specifying position and speed is velocity control itself, so
  choosing constant pressure stretches the whole profile by the same factor, and the injection time to V/P disagrees with the set value.
  The default clock remains `constant_pressure` as before (so as not to move existing results).

### Known limitations

- First-order approximation of the theoretical injection volume. Acceleration/deceleration ramps, melt compression, check-valve leakage and machine response are not included.
- The compression phase (ICM) remains an isobaric-source approximation, and the injection profile only affects the injection phase.

## [0.41.1] — 2026-09-11

**Fixed a race where the shot volume's geometry tracking silently stopped under rapid clicking.** The check "untouched if the previous
automatic value equals the current value" misfired when geometry widgets were clicked rapidly and the next rerun started before the
previous rerun's new value reached the browser: the browser's stale value was written back into session_state, judged as "edited",
and tracking stopped from then on (found in fangate: the shot volume for wing t0.65 was kept while analyzing
t0.9, giving a 0.4% short shot, and the two-phase map title rounded to `100%` so it went unnoticed.
sim is where this logic originated and the same code was still there — a port of fangate PR #14).

### Fixed

- `app.py`: The untouched check now uses the flag `mfs_shot_volume_user_edited` set by the shot-volume field's `on_change`. However,
  Streamlit fires `on_change` for "widget values that differ from the previous state" before the script body, so the callback is called
  even for a stale echo (reproduced with a mutant using an unconditional flag, Codex P1) — if the arriving value matches any past automatic
  value (`mfs_shot_volume_auto_history`, last 32) it is treated as an echo and not as an edit. After editing, the button
  "Reset shot volume to cavity volume (resume geometry tracking)" restores tracking.
  When the shot volume is below the final cavity volume, the caption states the shortfall and "→ short shot". Toggling two-phase OFF → ON
  makes Streamlit drop the widget state, so the edited value is copied to `mfs_shot_volume_user_value` and restored if the widget is gone
  (before: only the flag survived and skipped initialization, so the field reappeared at the min of 0.01 cm³ — Codex P2 on PR #87)
- `core/visualizer.py`: The post-compression fill fraction in the two-phase map title is **floored** to one decimal when below 100% (`99.6%`;
  99.99% no longer turns into `100.0%`, Codex P2) — `_fraction_label`
- Tests: reproduce the browser's stale echo with `set_value(old automatic value)` through the same callback path as production, and check that
  tracking continues with both 1-rerun and 2-rerun delays, that edit → shortfall display → reset resumes tracking, and that the edited value
  survives OFF → ON (`tests/test_two_phase_ui.py` +3),
  `_fraction_label` (`tests/test_two_phase.py` +1)

## [0.41.0] — 2026-09-10

**Added coring (a triangular plateau) in front of the vertical runner to Film gate 8 (T-shape). Added a flat-bottom `floor_depth` to fan islands.**

### Added

- `SubIslandSpec.floor_depth` (optional): makes a fan island's band a **flat plateau** instead of a "shallow ramp growing by angle". Depth within the band is
  `min(ramp, floor_depth)` — where the ramp is still shallow it is left alone, so the plateau's upstream vertex naturally falls at "the t where the ramp reaches floor_depth".
  When specified, `angle_deg=0` is required and `0 < floor_depth ≤ main_ramp.cap_depth`. JSON key `floor_depth`; when unspecified it does not appear in to_dict, so existing specs are unchanged.
- A "Coring (triangular plateau in front of the vertical runner)" block for Film gate 8 (default ON). Sliders: base width (5) / top depth (upper vertex, default land depth) /
  foot depth (lower vertex, default horizontal runner depth). **A wedge that splits the melt coming from the vertical runner left and right**: the apex (narrow side) is placed on the
  centerline at the t where the ramp reaches the foot depth (vertical runner side = the side the melt comes from), the base at the t where the ramp reaches the top depth (land side),
  and the top is constant depth.
  Additional instruction from the chairman's sketch (2026/9/10): the 2 vertices on the centerline of the triangular pyramid are the ramp's head (land depth) and foot (vertical runner depth), the left and right vertices at the same height as the head, width 5.
  Taking the slope downward would create an undercut, so as a flow-path thickness it is a "plateau at the top depth". The first drawing, with the apex on the land side, was upside down (corrected after the user pointed it out).
- The island is given identically to **both fans**, the horizontal runner's and the vertical runner's — overlapping fans let the deeper one win, so with only one the other fan's ramp overwrites it and it disappears
  (pinned by `test_floor_depth_island_must_be_in_every_overlapping_fan`).
- `tests/test_geometry_twin_fan.py` +8, `tests/test_film_gate8_ui.py` 8 → 11 tests.

### Changed

- The "outer line > inner line" check for fan islands in `validate()` now uses the same **clamped evaluation** as the raster (before the first point, that point's w).
  It used to extrapolate, so lines starting deeper than the land (the apex of a coring with a top deeper than the land) were extrapolated to negative w and rejected for a false crossing.
  All existing specs have every line starting at the land length, so results are unchanged.
- The lower bound of Film gate 8's horizontal runner depth is now land depth + 0.3 (so the coring's two depth sliders keep `min < max`; a slider with `min == max` raises and the sidebar disappears).
- `ValueError` at assembly when a fan island picks up no cell centers (Codex P1 on PR #85). With a 1.0 mm mesh, the row centers of the T-shaped coring fall on integer t and
  the triangle comes out empty, so it was in the spec while the analysis ran without the coring. Same treatment as rejecting zero-cell edge channels.
- The island lines' ordering check is now done at both ends plus the breakpoints of both lines (Codex P2 on PR #85). After switching to clamped evaluation, shapes where two lines with different start t cross midway slipped past a check at the ends only.

## [0.40.0] — 2026-09-10

**Added Film gate 8 (T-gate) to the UI.**

### Added

- `Film gate 8 (T字/横ランナー+縦ランナー)` in the sidebar. Behind a full-width film land (1×0.35), a short ramp (1 mm, 58.78°) leads to
  a full-width horizontal runner of depth ② 2.0 (t=2–4, the crossbar of the T), and in the center a vertical runner of width ⑤ 5 and the same depth (the stem of the T) runs to the well.
  The well and Φ3 valve are identical to the split gate 0703 (t=21.5). The defaults are `hamoko_gate_T_20260910` (private), a drawing made from the
  2026/9/10 hand sketch "T字ゲート" (T-gate).
- The horizontal and vertical runners are built from 2 rectangles in `sub_gates` (no `runner`). With a capsule-shaped runner, either the root of the T gets rounded, or, if the start is pulled
  back far enough that the rounded end does not bite into the land, the root is missing. With rectangular fans the root is square and connects straight to the well.
- The ramp angle is not a slider but is derived as "horizontal runner depth − land depth" ÷ "ramp length". The original drawing dimensions the drop and the length, not the angle.
- The default mesh is 0.5 mm. With t_max=27.5 at 1.0 mm, cell centers fall on integer t, and boundary inclusion fattens the land and ramp by one row each,
  increasing volume by 20% (the other Film gates have the same quirk, but their ramps are gentler so it is less visible).
- `tests/test_film_gate8_ui.py` (8 tests).

### Known trade-offs

- The original trapezoidal cross-section (opening 2.5 on the PL side, floor 1.5, depth 2) was replaced with the near wall as the ramp and the far wall as a vertical wall at t=4.0 (volume about +150 mm³).
  In 2D Hele-Shaw, a thin wedge at the PL does not act as a flow path.

## [0.39.0] — 2026-09-10

**Changed the UI default wall-cooling model from "None" to "Skin layer". Corrected outdated statements in the equation notes.**

### Changed

- Changed the default of the "Wall cooling representation" radio to "Skin layer" (`index=1`). The reason for defaulting to "None" was
  that "the two-phase short shot works only with None", but since v0.37.0 the skin layer rides on the two-phase
  injection phase, so there was no longer any reason to keep it. With "None", η is a single constant representative value and
  `S ∝ h³`, so **neither material nor temperature affects the filling order** (it is determined by geometry and Q alone). With the skin layer,
  thermal diffusivity α acts through `s = c·√(αt)`. T_melt / T_mold still have no effect (the η factor cancels both in the
  τ ordering and in the T_fill inflation ratio: on a 0.4 mm plate a +40 K temperature difference gives 1e-13;
  PP_T20→PP (α 1.3e-7→9e-8) measured 1216→1538 freeze-off cells). Only the multilayer model makes temperature matter, and it
  cannot be combined with two-phase.
- Removed "multilayer recommended for ultra-thin plates" from the radio's help, and described the default and compatibility.
- Changed the skin layer's "fixed-point iteration limit" slider from 1–10 (default 5) to 1–40 (default 20 = the solver default).
  With the constant-pressure clock, freeze-off can avalanche, and stopping midway makes a half-frozen picture look converged
  (Codex P1). The "Skin layer / core layer" expander in the result pane shows the iteration count, convergence, T_fill inflation, and freeze-off / short-shot
  cell counts, and warns when not converged.

### Fixed

- In the "Equations used and scope" expander, the fill-time formula was still the linear mapping
  `t = τ/τ_max · T_fill` abolished in v0.25.0; updated to the general form of the volume CDF mapping
  `t = T_fill × V(τ ≤ τ(x)) / V_solved` (the denominator is the volume of the fillable region). Stated that `T_fill = V/Q` holds for
  constant-rate injection without compression (no wall cooling, or the skin layer's velocity-controlled clock), that ICM shortens it, and that
  the skin layer's constant-pressure clock and the multilayer thermal coupling rescale it by the volume-weighted τ ratio.
  Corrected `t_arr = (τ/τ_max)·T_fill` in the multilayer section the same way.
- The multilayer heading "recommended, default for ultra-thin" in the same expander disagreed with the actual default ("None" → now "Skin layer");
  changed it to "selectable; cannot be combined with the two-phase short shot".

## [0.38.1] — 2026-09-03

**Gate markers are now drawn one per injection point, as true-scale circles.**

### Changed

- Gate markers in the fill animation / frame PNGs / pressure plot / weld plot / skin-core layer plots changed from **one per cell** of
  `Geometry.gates` (the raster of the valve circle = a few to dozens of cells for Φ3) to one per **4-connected component** of gate cells
  (`gate_groups_mm`). Previously, 8 pt circles offset by the cell spacing overlapped many times and the outline looked multiplied. It uses the same
  4-connectivity as the reachability check, so split feeds (multiple disks along a runner) remain one per location.
- Markers are **data-coordinate `Circle`s**, not fixed-size symbols. The radius is area-equivalent from the cell count,
  `sqrt(n·dx²/π)`, which reproduces the valve diameter within one cell width for a rasterized circle (a 1-cell gate is
  roughly a 1-cell circle). On a 300 mm plate Φ3 appears small, but that is the true proportion.
- Consolidated the per-cell loops scattered across 11 result plots and the design preview in `app.py` into
  `draw_gate_markers(ax, geometry)` (the "gate" legend entry in the weld plot is attached to the first one only).
- **Recording the nominal valve** `Geometry.valve_marker_mm = (x, y, r)`: set by the 4 builders (film_gate / film_gate2 /
  direct_gate / profile_gate); markers prefer it over the raster. In one-sided shapes (Film gate 3 with the
  well OFF, etc.) the mask cuts the valve circle, so reconstructing from the remaining cells gives "a shifted small semicircle"
  (Codex P2). When the circle covers no cell center and the builder snapped to the nearest cell, it is not recorded, and the
  cells the solver actually used are drawn. `display_origin_mm` falls back to the marker's x when `valve_axis_x_mm` is missing,
  and `HeleShawSolver._restricted_to` carries both records over (@claude review).
- Side effect of true-scale drawing: a 1-cell gate (`build_demo_geometry`) or Φ3 on a 300 mm plate appears as
  a few px. A deliberate choice favoring true proportions; no minimum display diameter was added.

## [0.38.0] — 2026-09-02

**Two drawings of fan-shaped gates with edge deepening (edge channels) can now be selected as the slider inputs Film gate 6 / 7.**

### Added

- `Film gate 6 (扇状/縁部深彫り 0807)`: the 2026/08/07 revision drawing of Film gate 1. Both ends of the pocket extended from
  t=3 → 5, coring exit width 100, well wall angle 71.6°, and a band along the outer wall of width 2.0 and depth 2.5 (= the ramp's
  cap depth) (the drawing's "2" dimension = a depth-2.5 contour running parallel to the outer wall, visible from t≈3 at the end until it joins the cap line
  t=12.11). The t range is the outer-wall segment [5.0, 23.28] — the lower bound of `t_range` only trims the
  wall polyline and does not cut cells by t, so starting before the wall's start point raises a stub of the band along the end wall and sinks the land.
  The existing `hamoko_gate_furiwake_rework_20260807` lacks
  this band; it was extracted before the v0.36.0 edge-channel implementation.
- `Film gate 7 (扇状/縁部深彫り 0515)`: the 2026/05/15 proposal drawing "フィルムゲート(流動長150mm)" (film gate, flow length 150 mm).
  Both ends at t=4, a groove of constant depth 2.4 along the outer wall reaching the land (cross-section a trapezoid with floor width 3.0 / opening 4.0;
  the plan view is drawn with floor width 3.0, so width 3.0; t range is the outer-wall segment [4.0, 23.3]). Where the drawing has no dimensions (ramp, well,
  valve), Film gate 1's values are used.
- `_ProfileGateDefaults.edge_channel` (width, depth, t range): for inputs whose defining feature is the band itself, the
  edge-channel block starts default ON from those values. 1–5 remain default OFF as before.
  `_edge_channel_inputs(default=)`.
- `tests/test_film_gate6_ui.py` (6 tests) / `tests/test_film_gate7_ui.py` (5 tests): the default sliders match
  each spec (including the band), the default geometry matches a direct spec read bit for bit, the band actually appears in the thickness field
  (6: at the end t=4,5 are 2.5, the land end and t=2 remain ramp, and every cell the band changed was a cell where the ramp was below the cap /
  7: at the end t=2–4 are 2.4, the land row is 0.35, the band is 3 mm even midway along the wall, and every cell the band changed was below 2.4),
  the band drops when OFF,
  band defaults do not leak between 6 ↔ 1 / 7 ↔ 6, and the geometry builds even when the outer wall is shorter than the drawing's band start (clamping the drawing's t range to
  [0, outer-wall end] gives zero width — the default falls back to the live outer-wall segment, Codex P2).

## [0.37.0] — 2026-08-25

**The skin layer can now ride on the injection phase of the two-phase short shot. The skin-layer clock can be switched to
"velocity-controlled (fixed injection time)".**
The actual FG1 95% ICM short shot (injection 0.085 s, velocity-controlled) shows "the gate area fills first and the
part area forms a low dome", but the isothermal solution was "the part center runs to the top edge and both ends of the fan come last".
The cause was thermal, not the solution method: switching to time-stepped front tracking (moving-boundary) did not change the
order under isothermal conditions, while once the part area opened to 0.85 mm carries a 0.1 mm skin per side after 0.085 s of exposure,
`S` drops to 0.43×, making the thick gate area relatively cheap. The freeze-off age of the 0.35 land is
0.24 s, so it is in the band that does not close during injection.

### Added

- `HeleShawSolver.skin_clock_mode`: `"constant_pressure"` (default, the existing approximation where "under constant pressure the
  flow rate tapers and T_fill inflates by the volume-weighted τ ratio") / `"constant_rate"` (velocity control:
  `T_fill = V/Q` fixed, pressure rises). The default is unchanged, so existing results are bit-identical. `skin_clock_mode` in the metadata;
  in the UI, the "Skin-layer clock" radio under wall cooling "Skin layer".
- `HeleShawSolver._solve_domain(eta, T_fill_baseline_s=, clock_end_s=)`: overrides the clock length and
  the time at which exposure stops. Cells arriving after `clock_end_s` carry no skin, and the clock does not
  inflate.
- Two-phase short shot × skin layer: the injection phase solves `tau1` as the fixed point of exposure cut at `T_inj = V_shot/Q`
  (by the definition of metering V/Q, always velocity-controlled). `TwoPhaseShortShotResult` gains
  `injection_skin_thickness_mm` (the service-mean skin within Ω₁) and `injection_sealed_mask`
  (cells of the pool frozen off by T_inj). The metadata gains `skin_layer_enabled` /
  `skin_growth_constant` / `skin_clock_mode` / `skin_iterations` / `skin_converged` /
  `injection_skin_max_mm` / `injection_sealed_cells` / `injection_unfillable_cells` /
  `injection_domain_passes` / `injection_clock_end_s`.
- Two-phase map / animation: cells frozen off during injection are painted dark red, and `sealed during
  injection` is added to the legend (only when freeze-off occurs). With skin ON the title shows `c` and `T_inj`.
- UI: two-phase runs with wall cooling "None" or "Skin layer" ("Multilayer" is skipped with a constant
  warning as before). The result pane shows the maximum skin / freeze-off cell count, and warns if freeze-off occurs. settings.json gains
  `wall_cooling.skin_clock_mode` and `two_phase_short_shot.skin_layer`.
- CLI: `_solve_and_export(skin_clock_mode=)`, and combining `two_phase_shot_volume_cm3` with
  `skin_layer=True`. Reference case `FilmGate_PP_two_phase_skin`.

### Changed

- `solve_two_phase_short_shot` accepts a solver with the skin layer ON (before: `ValueError` at the
  entry. The actual part refuted the use-case definition that "metering-limited shots involve no freezing").

### Intended limitations

- The compression phase remains isothermal (the pool is an isobaric source, so internal skin does not affect advancement, and the model has no
  time scale for compression). However, cells frozen off during injection stay closed in the compression phase too —
  a solidified skin does not melt when the mold closes — and only the region connected to the pool through open cells is a candidate for
  advancement (`compression_unreachable_cells`).
- When freeze-off occurs in the injection phase, the cells cut off by it are removed from the candidates and the solve is redone over the reachable region only
  (removing the dead region's volume from the arrival-time = skin clock). There is no bisection — under velocity control the clock does not
  inflate, so the largest consistent prefix is the reachable region itself. Exposure is cut off at
  `T_inj` (even if the shot volume fills first, the walls keep aging during injection, `injection_clock_end_s`).
- There is no viscous heating in the gate area. If injection is slower than the freeze-off age (0.24 s for PP_T20 with a 0.35 land),
  the land freezes off and the part area turns into a short shot. The UI warns. Exemption / viscous heating is the next stage.
- `c_skin` becomes a calibration parameter. A single photo of an actual part is the first calibration point (with c=1.0 the flow reaches the fan tip
  at the end of injection, so 0.6–0.8 are candidates).

## [0.36.0] — 2026-08-25

**Edge deepening (edge channels): a band of constant width and depth can now be cut along the pocket wall.**
Parameterizes the practical technique of deepening the slanted edges at both ends of the fan into low-resistance leading flow paths
so the melt runs ahead to the ends of the long side. Since `S ∝ h³`, the band's effect rises with the cube of its depth.

### Added

- `EdgeChannelSpec(width, depth, t_range=None, side="outer")`: deepens pocket cells whose perpendicular distance to the wall line is
  within `width` by `d = max(d, depth)` (the same floor semantics as runners and wells). It does not change the silhouette, so it does not break
  connectivity. `t_range` limits the segment along the wall
  (full length if omitted).
- Two places to put it: `GateProfileSpec.edge_channels` (single-pocket shapes, along the outer wall; for symmetric
  shapes it stands on the slanted edges at both ends by mirroring) and `SubGateSpec.edge_channels` (per fan, `side` =
  `"outer"` / `"inner"`; both sides are 2 entries).
- Rasterized from a distance field to the effective wall polyline (including clamp points and steps). Specifications where the band picks up
  no cells are rejected at assembly (sealing off the false green of something recorded in the spec but absent from the geometry).
- UI: an "Edge deepening" block for all of Film gate 1–5 (checkbox + band width / band depth /
  a two-handle range slider for t; FG4/5 also have a radio for the target edge). Default OFF (keeps bit identity with the drawing-default
  specs). The lower bound of band width is tied to the mesh (preempting bands narrower than 1 cell).

### Changed

- An `edge_channels` key in the gate profile JSON (root / `sub_gates[i]`). Existing
  specs are bit-identical without it.

## [0.35.1] — 2026-08-24

**Unified the display coordinate system on the part. y = 0 is the part's bottom edge line (the gate-side edge), x = 0 is the valve axis.**
Previously the origin was the gate centroid, and with film gates the gate itself sat at y = 0, contradicting the caption
"the gate block / runner is on the y < 0 side". With the part's bottom edge as reference, a film gate's whole gate block is at y < 0
and a direct gate sits inside the part at y > 0, so both families read on the same part-based axes.

### Changed

- `Geometry.gate_origin_mm()` → `display_origin_mm()`: y0 is derived from the bottom edge of the lowest row of the part zone
  (`compression_mask`, which every builder sets on the part body).
  Old demo geometries without `compression_mask` fall back to the gate centroid y (behavior unchanged).
  x0 remains the gate centroid (valve axis). The design preview, all result maps and the 3D view
  share the same origin function, so they switch together.
- Updated the design preview caption to describe the new coordinate system.
- `Geometry.valve_axis_x_mm` (new field): x0 is not the centroid of the rasterized gate cells but
  **the nominal valve axis recorded by the builder**. In an asymmetric pocket (Film gate 3) the orifice is cut at the
  w = 0 end and the surviving cells lean to one side, so the centroid drifts from the axis depending on the mesh
  (Codex P2). Set by all parametric builders; geometries without it fall back to the centroid.

### Tests

- `test_geometry_film_gate.py` +2 (part bottom edge at y=0, gate at y<0, demo-geometry fallback),
  `test_geometry_direct_gate.py` +1 (gate inside the part at y ≈ +gate_offset),
  `test_geometry_profile_gate.py` +2 (x=0 is the nominal axis even for an asymmetric cut orifice;
  follows the valve.w offset for symmetric shapes).

## [0.35.0] — 2026-08-24

**Film gate 5 (split / L-shaped runner) is now in the UI. The same twin mini fans as Film gate 4, but
the runner extends straight sideways from the valve and connects vertically from below at the center of each fan's tip
(`hamoko_gate_furiwake_twin_mini_L_20260824`). A straight runner bites into the fan's inner wall side, biasing the
melt inlet toward the axis and making the filling center-heavy (actual behavior confirmed by flow analysis). Vertical connection at the tip center
makes the path lengths from the tip to the left and right corners of the fan base equal, balancing left and right.**

### Added

- UI `Film gate 5 (振り分け/L字ランナー)` (widget key `f5_`): the sidebar is shared with Film gate 4
  (`_twin_fan_sidebar`); the only difference is the derivation of the runner path chosen by `_TwinFanDefaults.runner_style="L"`.
  The path is a 3-point polyline: valve `(t_v, 0)` → straight sideways `(t_v, center half-width)` → fan tip
  `(t_tip, center half-width)`. The width of the vertical part is the runner width itself, flush with the fan tip width
  8 by default. **No schema change** — `RunnerSpec.path` has accepted multiple segments since v0.34.0.
- When the valve is at the same t as the fan tip, the corner of the L coincides with the end, producing a zero-length segment
  (a shape `validate()` rejects) — path assembly (`_twin_fan_runner_path`) drops consecutive duplicate points
  and collapses it to a 2-point path.
- The lower bound of Film gate 5's valve position follows the fan tip t — with the valve in front of the fan, the trunk becomes a deep band
  crossing the fan's interior, a different experiment from the "connect from below" design (Codex P2).
  Equality is allowed as the degenerate corner.
- `tests/test_film_gate5_ui.py` (7 tests): default sliders match the L-shaped spec / default geometry matches a direct spec
  build bit for bit / the L-shaped path follows the valve, fan tip and center half-width / with valve = fan tip t the
  corner collapses and the build completes / at the extremes of the center half-width the runner width upper bound holds on the L-shaped path /
  slider values do not leak FG5 → FG4 / the valve position's lower bound follows the fan tip t.

### Verification

- The builder's pocket volume matches an independent reference implementation of the drawing (0.1 mm raster): 5,675 mm³
  (+392 from the straight-runner design's 5,283 mm³, the runner extension).

## [0.34.0] — 2026-08-24

**Film gate 4 (split / twin mini fans) is now in the UI, and the gate spec JSON can represent multiple fans
(`sub_gates`) and a runner (`runner`). The design study where the center is fully cored with steel touching the PL
(a deformed rhombus) and a runner splits the flow from the valve well to the left and right mini fans
(`hamoko_gate_furiwake_twin_mini_20260824`) can be analyzed as is.**

### Added

- `GateProfileSpec.sub_gates`: a list of fan-shaped pockets. Each fan is bounded by `inner_wall_line` /
  `outer_wall_line` (straight lines in the (t, w) plane, clamped to the first point's w before that point) and `tip_t`;
  inside is the land + main ramp, overwritten by an optional `island` (a shallow band between `inner_line` / `outer_line`,
  `angle_deg` / `end_dist`). Everything outside the fans is steel (PL contact).
  `outer_wall_line` can be null; **`outer_wall_line` and `sub_gates` are mutually exclusive and exactly one is required**
  (`validate()`). The top-level `island` is for single-pocket shapes only (in fan shapes each fan has its own island).
- `GateProfileSpec.runner`: a band of width `width` along the (t, w) polyline `path`; within the band
  `d = max(d, depth)`. On steel it is a pocket in itself; inside a fan it is a floor (the deeper wins).
  It can also be specified for single-pocket shapes. A fan shape without a runner leaves the well as an isolated island, and the solver
  rejects it as "gate unreachable" — the runner is not decoration but solvability itself.
- `t_max()` / the new `w_max()` include the fan tips, island ends and runner reach (end of `path` + `width/2`),
  and the grid overflow check also looks at overhang across the runner.
- `data/gate_profiles/demo_twin_fan_gate.json`: a fan-shape demo with fictitious dimensions (for tests).
- UI `Film gate 4 (振り分け/ミニ扇×2)` (widget key `f4_`): from the fan tip t / the fan tip's center half-width /
  the fan tip width, it builds each fan's inner wall `(land length, 0) → (t_tip, center − width/2)` and outer wall `(land length, exit width/2) →
  (t_tip, center + width/2)`, and derives the runner path `(valve t, 0) → (t_tip, center half-width)`
  (not held as a dimension). The runner width's initial value is the fan tip width and its depth's initial value is the ramp cap (both are independent sliders and do not follow when the tip width changes). Each fan's coring has
  4 half-widths: exit side and end side of the inner line / outer line. The well, valve and part geometry use the same
  common blocks as the other Film gates (factored out into `_plate_shape_inputs` / `_well_inputs` / `_well_from_inputs`).
  The defaults match the twin-mini-fan study spec bit for bit (`tests/test_film_gate4_ui.py`).
- Made the elements of `_FILM_GATES` `_FilmGate(tag, record_name, sidebar, assemble)`, so the sidebar and assembly function can be swapped per Film gate
  (1–3 use the existing helpers, 4 uses
  `_twin_fan_sidebar` / `_twin_fan_from_inputs`).
- **Aligned the grid overflow check with "how the builder evaluates the lines"** (Codex P1×2).
  `w_max()` takes the reach width over the fan segment `[0, tip_t]` / the whole block length using the same evaluation as `_line_eval`
  (clamped before the first point, extrapolated beyond the second point), not the stored endpoints of the outer-wall line — if an outward-leaning wall
  ends before `tip_t`, reading endpoints underestimates the reach width, slips past the check, and
  the raster is silently truncated at the array edge (area, volume and conductance go wrong). The single-pocket
  `outer_wall_line` now uses the same evaluation too (the defect is a property of "how walls are evaluated", not specific to fans).
  The new `w_min()` returns the negative-side reach (`min(path.w) − width/2`) when, with `symmetric=False`, the runner passes near `w=0`,
  and the builder's one-sided check looks at it — previously only the positive side was checked,
  so a runner with insufficient margin was cut at the array edge and a flow path narrower than specified was solved.
- **Extended the fan width validation from the single tip point to the whole segment** (Codex P2). Both inner and outer walls are piecewise linear
  with a breakpoint at the clamp point, so the minimum gap occurs at a breakpoint or an endpoint — checking all of `{0, tip_t, each line's clamp point}`
  is exact, not sampling. Shapes that crossed on the land side and opened at the tip slipped past a tip-only check, and the analysis completed
  on a geometry where the raster had silently dropped the crossing.
- **Runners narrower than the mesh are rejected at assembly**. The band slips between cell centers and turns into a dotted line of
  islands, and the fans it fed get cut off from the valve (the continuous spec itself is sound, so it would
  proceed silently with a fragmented geometry). It is the same defect class as when `gate_exit_width` falls below the mesh,
  so it gets the same treatment — stop at assembly, naming the width and cell size. The check is **measured on the raster**
  (number of 4-connected components), not a formula on the ratio of width to cell spacing: where it becomes too thin depends on the angle between the path and the grid.
  The symmetric field is folded at `(t, |w|)`, so a band not touching the axis mapping to 2 pieces left and right is
  normal — what is counted is the half-plane before folding.
- **UI: solver `ValueError` goes to `st.error`**. Detection of unreachable cells (Issue #58) happens inside
  `solver.solve()`, which was not wrapped in try, so geometry combinations the builder cannot name
  appeared on screen as raw tracebacks.
- **UI: closed spots where the sidebar broke at slider bounds**. A slider with `min == max` is not
  disabled but raises `StreamlitAPIException`, and the widgets after it are not drawn.
  (1) Swinging the fan tip's center half-width to its extreme collapses the room for the fan tip width `2·min(axis, exit half-width − axis)` to
  0 → reserved a margin of the minimum tip width in the center half-width's bounds. (2) Minimizing the exit width
  makes the coring outer line's `min` (inner line + 0.5) catch up with its `max` (exit half-width + 0.5) →
  reserved a gap on the upper side of the inner line (widening the outer line's upper bound would offer a line outside the fan).
- **UI: tied the runner width's lower bound to the mesh** (roughly the cell diagonal `dx·√2`). Below 1 cell width
  the band breaks into a dotted line and assembly rejects it. The upper bound is also tied to the edge margin (`2·(pad + Wp/2 − center half-width)`). Swinging the center half-width to its extreme
  pushes the band's outside off the plate and assembly rejects it. The label stays a fixed string (the label is
  part of the widget key, so changing it with the upper bound would reset the value).
  Actually solved all slider corners of Film gate 4 (72 combinations of mesh × exit width × center half-width × tip width × runner width)
  and confirmed that all pass except 4 that hit the existing cell-count guard —
  keeping, like the other Film gates, "every value the sliders can express builds".
- Tests: `test_geometry_twin_fan.py` (41 tests: round-trip, exclusivity and various rejections, closed-form fan area,
  full land width and steel at the center, fan islands, runner capsule volume (on steel) and floor semantics (inside a fan),
  solvability with and without a runner, legacy specs bit-identical, **extrapolated reach / one-sided negative reach / whole-segment width validation / rejection of runners thinner than the mesh and no false positive from folding** (by mutation injection on a copy, confirmed that reverting each fix makes the corresponding test fail)), `test_film_gate4_ui.py` (10 tests, including the runner width bounds following and the sidebar not collapsing at both ends of the center half-width).

## [0.33.0] — 2026-08-24

**The skin-layer model's clock became an exposure clock, and thin chokes beside the gate now freeze off and
cut off what lies beyond (Issue #61). Short shots split into "frozen-off cells" and "cells never
reached", and frozen-off cells have a fill time.**

### Changed

- Replaced the skin growth clock from the arrival snapshot `s(t_arr)` with an **exposure clock**. The wall keeps
  aging from the moment the front passes, so the skin at time `t` is `s(t − t_arr)` (thickest at the gate, zero at the front — the old clock was exactly
  reversed). The conductance a single elliptic solve can hold is one value per cell, so the **time-averaged skin**
  over the service period `[t_arr, min(T_fill, t_close)]`, `(2/3)·s(a)` (the exact time average of the √ law, `SKIN_SERVICE_MEAN_FACTOR`), is applied.
- Freeze-off check: against the age at which the skins meet, `t_c = ((h − h_min)/(2c))²/α`, a cell whose service `T_fill − t_arr`
  reaches it **freezes off** at `t_close = t_arr + t_c`. Frozen-off cells closed after the front passed,
  so they have a fill time, and the meaning of `short_shot_mask` changed from "frozen and unfilled" to "frozen off (closed after
  filling)". It is **disjoint** from `unfillable_mask` (cells never reached)
  (old: `short_shot ⊆ unfillable`).
- The short-shot check became reachability with arrival times: a cell fills at its own arrival time when it connects to the gate by a
  4-neighbor path through cells that have already been reached and are open (not frozen off, or with `t_close` still ahead). Cells reached before the choke closes are
  filled through the choke; only cells reached after it closes are short shots (DP in arrival order, `_unfillable_cells(frozen, t_arr, t_close)`).
  The old implementation marked everything beyond a frozen cell as unfilled. `skin_thickness_mm` /
  `core_thickness_mm` of short-shot cells are NaN (0 would look like a closed core).
- The time-averaged skin does not exceed `(2/3)·(h − h_min)/2`, so `h_core ≥ h/3` is guaranteed, and the resistance increase of frozen-off
  cells stops at 27× at most. The runaway where, under the old clock, floor-hitting cells inflated T_fill by hundreds to tens of thousands of times
  structurally disappeared (LGP 0.4mm film gate, c_skin=0.5: old 414×, not converged →
  new 1.1×, converged, fill fraction 48%).
- When a cut-off occurs, the part that fills is found as **the largest prefix that does not cut itself off, by bisection on fill volume**
  (`_largest_consistent_prefix`, resolution `V/256`, the thinnest single cell as the lower bound, finished with a short monotone loop that drops the
  cut-off of the hi-side candidate and re-solves to tighten the boundary to the cell — this also catches slivers smaller than the resolution).
  Each candidate is solved independently from no skin, so the answer's clock carries no
  dead-region resistance. The old monotone shrinking of "cut and re-solve" cut too much in the first pass because of an inflated clock
  including the dead region, and the re-solve was too fast to freeze anything off — on LGP 0.4mm it produced a picture where "h_core on the gate side is
  full but the flow is cut off midway through the part". `short_shot_mask` is the cells closed in the final solution ∪
  the cells the smallest cut-off candidate saw as closed.
- Vectorized `_build_linear_system` per face (old: `lil_matrix` + Python double loop).
  Assembly for 10k cells 180 ms → 9 ms, 46 ms including `spsolve`. The matrix is identical (Dirichlet rows are unit rows,
  neighboring rows keep the gate columns). This absorbs solving 8–10 candidates in the bisection, and the LGP 0.4mm
  skin ON run went from 133 s → 14 s.
- `metadata`: `sealed_off_cells` became equal to `unfillable_cells` (every short shot lies beyond a freeze-off).
  Added `filled_volume_fraction` (filled volume / cavity volume) and `skin_clock="exposure"`.
- Changed the core layer map's legend and title from `short shot` to `sealed (skins met)`.

### Known constraints

- The positive feedback of the constant-pressure approximation can have two fixed points for the same cavity (stays open and fast / freezes off and
  slow). Candidates start from no skin, so the lower side "as if flowing from the start" is taken. At a fold where adding one drop
  tips the whole into the frozen-off side, lo is the answer and the freeze-off marks are the hi side's judgment.
- The bisection's resolution is 1/256 of the volume. Cells straddling the boundary's uncertainty band are tightened by the "hi − cut-off" finishing step, but
  in degenerate cases where the band is large relative to the live volume (a 0.04mm ring in a 2mm plate), the hi
  clock is contaminated and may drop extra cells within the band (4 of 8 ring cells).
- The reported skin of frozen-off cells is "the service mean until filling ends". With the lo clock ending just before the cut-off,
  frozen-off cells are still drawn open.

## [0.32.0] — 2026-08-22

**The two-phase short-shot animation can now be viewed with the same scrubber as the flow front (play / step /
seek / speed), and the default shot volume is now the cavity volume of the current
geometry.**

### Added

- `core.visualizer.export_two_phase_frames` / `two_phase_frame_labels`. With `frame_states`
  as the single source, the GIF, frame PNGs and the player readout all ride on the same frame sequence.
  The readout shows `t = … s` in the injection phase and `advance … %` in the compression phase (the model has no clock for compression).
- `build_fill_player_html(labels=...)`. Replaces the readout with one string per frame.
- `two_phase_player.html` in the result ZIP (standalone HTML, works fully offline).

### Changed

- The default shot volume `V_shot` changed from a fixed 5.0 cm³ to **the final cavity volume of the current geometry**
  (exactly a complete fill). It follows geometry changes, and once the user has touched the value it is
  left alone (the previous automatic value is kept in `mfs_shot_volume_auto` for comparison. The value is not rounded —
  a rounded default turns into a tiny short shot in the solver's raw volume comparison). For this reason
  `build_geometry()` is called inside the sidebar instead of the main column. The short-shot section reserves its
  position with an `st.container()` placeholder, and its contents are filled after the "Output" section and the version display
  — so that `st.stop()` on an inconsistency does not wipe out the version display.
- The shot-volume caption now shows the volume of the current geometry instead of "the geometry of the previous run".
## [0.31.1] — 2026-08-22

**Removed the "spec input" radio from Profile gate. Only one path remains: the local list + drop.**

### Removed

- The `デモプリセット` (demo preset) and `JSON貼り付け` (paste JSON) options, the `SpecMode` enum, `SpecOrigin.DEMO` /
  `SpecOrigin.PASTE`. Keeping specs drawn up from drawings as files had become the established practice, and
  there was never a case for loading them by pasting. The demo geometries are already covered by the default inputs of Film gate 1–3 / Direct gate,
  so there was no longer any point in letting Profile gate pick a bundled demo
  (`data/gate_profiles/demo_profile_gate.json` itself remains, since tests and the CLI use it).
  `choose_spec_origin` lost its `mode` argument and takes only `has_upload` / `has_local`.

## [0.31.0] — 2026-08-21

**Weld lines are now drawn by "the angle at which two flows meet", so the merge lines extending behind holes and
coring are now visible. Dark red = weld (opening angle 45° or more), light red = meld
(the mark of flows merging almost in parallel).**

### Changed

- Replaced the weld detector entirely. The old rule, "6 or more of the 8 neighbors filled earlier than this cell", was
  effectively local-maximum detection and structurally missed lines where the flow keeps going after merging (behind holes, downstream of coring)
  — the 3 downstream cells are always later, so at most 5 could ever line up. The new
  detector looks at the facing neighbor pairs of each cell (x / y / the 2 diagonals); if the flows on both sides **head toward that cell**
  and the cell **fills later** than both sides (a ridge in arrival time), it is a merge point, and the angle between the two flow directions
  (180° = head-on collision, 0° = parallel) is its strength. Weld and meld are split at the commercial CAE "merge angle 135°" boundary
  (opening angle 45°)
- Added `FlowResult.weld_angle_deg` (merge opening angle [deg], NaN where there is no merge). `weld_score`
  is this mapped through the default threshold. The drawing side re-thresholds the angle field, so changing
  the threshold does not require re-solving
- Added two levels, weld / meld, to the legends of the contour and weld plots. Melds are also drawn with at least 35% opacity
  (so even at small angles you can see "there is a line")
- Added a "minimum meld display angle [deg]" slider to the sidebar "Output" (0–40, default 0).
  Recorded in settings.json as `output.weld_min_angle_deg`. Moving the slider after an analysis
  redraws only weld.png from the cached result and also replaces the image in the ZIP and settings.json
  (Codex P2: moving it outside the run button did nothing until the next run)
- Three false-positive suppressions: (1) wall cells (with a wall among the 8 neighbors) are not used for the merge test, since their flow direction
  jitters in steps due to one-sided differences, (2) 2 cells around the gate are excluded, (3) cells where the arrival times on the two sides
  are skewed beyond 10:1 (flow turning around the corner of an obstacle that merely rode one side's isochrone) are not
  accepted as ridges. When the center 2 columns of a symmetric geometry are equal to machine precision, the equal side is read 1 cell further

### Known trade-offs

- For the merge line behind a hole, only the root (the first few mm it touches) is a weld of 45° or more; the rest
  continues to the top edge as a meld with an opening angle of a few degrees. This is exactly the behavior of the model, where the thin plate is fed as a line source
  from the thick ramp and the notch fills by cross flow, and it corresponds
  to the "tongue" in the center of the contours. The default minimum angle of 0 draws everything including melds; for drawings where the noise is bothersome, raise it with the
  slider

## [0.30.0] — 2026-08-21

**Film gate now comes in 3 variants, and the distance from the PL of the coring's horizontal section (welded dam) can be swept with a slider.
At 0 the steel touches the PL and it becomes a fully cored-out cavity (a hole).**

### Added
- `Film gate 2 (扇状/肉盗み2)` (fan-shaped / coring 2): a symmetric version defaulting to `hamoko_gate_furiwake_weld_20260818`.
  Downstream of the coring, t=7–17 is welded horizontal (`island.weld`, remaining flow thickness 0.1), outer wall starts at t=5,
  well wall angle 71.6° (floor range = `depth/tan(71.6°) = 1.5` inward on each side). The coring boundary line in the drawing
  starts at t=0, but the UI holds it at t = land length, so the default is set to 47.64, the same straight line read at t=1
  (results on a 1.0 mm grid are bit-identical to reading the spec directly)
- A "horizontal section (welded dam)" checkbox for every Film gate with coring enabled. Two sliders, start t and
  "distance from PL (remaining flow thickness)"; the end is fixed to the coring end. Default ON only for
  Film gate 2
- `GateProfileSpec.validate()` accepts `island.weld.depth = 0`, and the builder removes that band from the
  mask (leaving zero-thickness cells in the cavity makes `S = 0` and a singular system). Cells the well pierces
  remain in the cavity at the well's depth

### Changed
- Labels: `Film gate 1 (肉厚調整ゲート)` (thickness-adjusting gate) → `Film gate 1 (扇状/肉盗み1)` (fan-shaped / coring 1), the former
  `Film gate 2 (肉厚調整ゲート・片側)` (thickness-adjusting gate, one-sided) → `Film gate 3 (片側/二倍流動長)` (one-sided / double flow length) (widget key also
  `f2_` → `f3_`, record name `film_gate_3_parametric`). The UI default input remains the one-sided version
- Terminology: in the UI, unified "アイランド" (island) to "肉盗み" (coring), and "バルブオリフィス径" (valve orifice diameter) to "バルブゲート径" (valve gate diameter).
  JSON keys (`island` / `orifice_diameter`) and code identifiers are unchanged — existing specs
  load without modification
- The well wall angle moved from the `_WELL_WALL_ANGLE_DEG` constant to `_ProfileGateDefaults.well_wall_angle_deg`
  and is held per drawing (1 / 3 = 60°, 2 = 71.6°). The outer wall start t also moved to the defaults side

## [0.29.0] — 2026-08-21

**Film gate 2 is also now a parametric input for the thickness-adjusting gate (one-sided version), and the UI defaults switched
to match two-phase short-shot operation.**

### Changed

- **Replaced Film gate 2** — the old "variable gate position (right-trapezoid runner)" model was removed from the UI,
  and it became the **one-sided version** of the same depth-field model as Film gate 1 (`symmetric=False`, valve at the w=0 end, widths measured
  from the valve-side end). The radio label is `Film gate 2 (肉厚調整ゲート・片側)` (thickness-adjusting gate, one-sided).
  Defaults are `hamoko_gate_2bai_20260703` (exit width 299 / island boundary width 95.3→20.0 /
  outer wall (3,299)→(23.6,4.45) / valve t20.0 Φ3; land, ramp and well are the same as Film gate 1).
  The geometry run with the defaults matches the result of reading the spec with Profile gate in mask, wall thickness and gate cells
  (`tests/test_film_gate2_ui.py`)
- The ties between derived quantities are the same as Film gate 1 except for two points: the outer wall start width = **the exit width itself**
  (one-sided, so not the half width), and the default valve position = the drawing's value 20.0 (not the well center 21.5)
- Merged the sidebar and assembly of Film gate 1 / 2 into shared helpers (`_profile_gate_sidebar` /
  `_profile_gate_from_inputs` / `_build_film_gate`). Widget keys are separated by `f1_` / `f2_`
- **UI defaults**: wall cooling model "none" (was: multilayer), injection-compression molding ON, stroke 0.50 mm
  (was: OFF / 0.70), two-phase short shot ON (was: OFF). The initial state is now a combination in which the two-phase model actually
  runs
- The origin notation in the design drawing caption is now "valve gate position (red circle)" (in the one-sided version it is not the part center)
- The old `FilmGate2Config` / `build_film_gate2_geometry` and 33 tests are kept for CLI / library use
- All Film gate widgets have `f1_` / `f2_` prefixed keys (without keys, values of same-labelled widgets carry over when switching inputs).
  The post-build valve guard checks **the intersection of the orifice circle and the mask** instead of the center cell (in the one-sided version the center lies on the w=0
  boundary, and it was falsely rejected on fine meshes) (Codex P2 ×2)

## [0.28.0] — 2026-08-21

**Film gate 1 in the UI is now a parametric input that builds the production thickness-adjusting gate (land / main ramp / island / well / outer wall line)
with sliders.**

### Changed

- **Replaced Film gate 1** — the old "trapezoid + semicircular runner + ▽ coring" model was removed from the UI;
  the same depth-field model as Profile gate (`GateProfileSpec`) is now assembled in the sidebar and fed to
  `build_profile_gate_geometry`. Solver and visualization are unchanged. The radio label is
  `Film gate 1 (肉厚調整ゲート)` (thickness-adjusting gate)
- Defaults are the dimensions of `hamoko_gate_furiwake_20260703` (exit width 298 / land 0.35×1.0 /
  ramp 10.95°, cap 2.5 / island 2.5°, boundary half width 52.7→10.0, end 17 /
  outer wall (3,149)→(23.3,4.5) / well t15.5–27.5, half width 4.5, depth 4.5 / valve t21.5 Φ3).
  The geometry run with the defaults is bit-identical in mask and wall thickness to the result of reading that spec with Profile gate
  (`tests/test_film_gate1_ui.py`)
- Only the main dimensions are exposed as sliders. Derived quantities are fixed per the ties in the drawing: island boundary line
  t endpoints = land length / island end, outer wall start half width = exit width/2, well floor range =
  t_range moved inward by `depth/tan(60°)` on each side (wall angle 60° is a constant), default valve position =
  well center. Island and well can be turned OFF with checkboxes
- Constraints the UI enforces up front: well depth ≤ half width·tan(60°) (a depth the walls cannot reach is accepted by the spec
  but drawn shallower), valve position ∈ [radius, pocket end − radius] (out of range, the builder silently snaps to the nearest cell
  and the record and injection position diverge). Out-of-range in the width direction is detected and rejected after the build (Codex P1 ×2)
- Added a reachability check for well depth to `GateProfileSpec.validate()` (`depth ≤ half_width·tan(wall_angle)`;
  a wall angle of 90° is unlimited). It also applies to the JSON path — previously it was accepted and drawn shallower
- The old Film gate 1 `FilmGateConfig` / `build_film_gate_geometry`, the CLI's
  `FILM_GATE_CASES`, and 43 tests remain as they are (CLI only)

## [0.27.0] — 2026-08-20

**Two-phase short shot gained a history animation, and settings conflicts are now visible.**

### Added

- `render_two_phase_animation()` — a GIF of the two-phase history. In the injection phase the blue grows in real time (arrival time);
  in the compression phase the orange advances in normalized order (the model has no time scale for compression — the title states
  which clock is used). The frame sequence is supplied by the pure-data `frame_states()` (`core/two_phase.py`),
  which guarantees monotonic growth of the filled set and that the final frame = Ω₂. Included in the UI expander, the ZIP
  (`two_phase.gif`), and the CLI output
- UI: shows a guide to the cavity volume of the previous run's geometry (final / open gap) below the shot-volume input

### Fixed

- **Legend moved outside the plot (bottom center)** — on wide plates every corner inside the axes overlaps the part
  (in actual rendering the top-right corner was hidden). A figure-level legend does not collide regardless of aspect ratio.
  The layout is finalized after setting the title (doing tight_layout first cut the title off at the top edge)
- **Conflict with the wall cooling model is always visible** — the UI default wall cooling was "multilayer", so turning
  two-phase ON with defaults skipped it with only a single runtime warning, and it looked like "checking it does nothing".
  A persistent warning is now shown in the sidebar (where the setting is), and the skip reason is kept in session_state
  and also shown in the results pane

## [0.26.0] — 2026-08-19

**Added a two-phase model that predicts metered short shots using the actual machine parameters as they are.**

### Added

- `core/two_phase.py` — two-phase short-shot model `solve_two_phase_short_shot()`.
  (1) **Injection phase**: solve τ in the mold-open gap (h + stroke), cut the volume CDF at the shot volume
  V_shot to get the melt pool Ω₁ at the end of injection. (2) **Compression phase**: re-solve
  τ at the final wall thickness (all cells of Ω₁ as an equal-pressure source Dirichlet), and advance in τ order until volume is conserved to get
  the final shape Ω₂ after compression. Two linear solves, no time integration. Tie groups with equal τ are handled
  atomically (no half-filled cells are created). With ICM OFF / stroke 0, it degenerates to Ω₂ = Ω₁
  (a pure volume-limited short shot)
- `render_two_phase_map()` — categorical map (blue = filled by injection + injection isochrones,
  orange = advanced by compression, gray = unfilled)
- UI: sidebar "Short shot (metered)" expander (shot volume [cm³] input),
  a dedicated expander in the results pane (map + fill ratio metrics), the ZIP includes
  `two_phase_short_shot.png` / `two_phase_metadata.json`, recorded in settings.json
- CLI: `two_phase_shot_volume_cm3` kwarg in `run_demo.py` and the reference case
  `FilmGate_PP_two_phase_short` (stepped plate + stroke 0.70 + shot 4.5 cm³)
- 24 tests (analytic strip verification, volume conservation, tie atomicity, equal-pressure source contract,
  nesting monotonicity, 3 AppTest wiring tests). All 8 injected mutations detected on a cloned repo

### Intended constraints (not bugs)

- Freezing progress during compression is ignored (combining with the skin layer model is rejected at the entry). A metered
  short shot excluding freezing physics is the very definition of its purpose
- Overlapping injection/compression operation is outside its scope (strictly sequential two phases)
- The melt pool uses an equal-pressure source approximation (pressure loss within the pool ≪ resistance at the unfilled front; valid for thin plates)

---

## [0.25.0] — 2026-08-19

**Brought the fill-time mapping in line with the physics and changed the time scale to a normalization that a single cell cannot dominate (Issue #52).**

### Changed

- **Arrival-time mapping is now a volume CDF** (shared by `HeleShawSolver` / `MultilayerHeleShawSolver`):
  the fill time of a cell = the volume of all cells whose τ is at or below that cell's τ, divided by Q. Under constant-rate injection the flow
  front advances linearly in volume, so this is the physics itself. The old `(τ/τ_max)·T_fill` linear mapping reported the middle cell
  of even a healthy 1D strip as 0.75T (the correct value is 0.5T). A single outlier cell can shift the times of other cells only by its own
  volume, and adding one outlier leaves the absolute times of existing cells strictly unchanged.
- **T_fill inflation proxy is now the ratio of volume-weighted mean τ**: numerator and denominator are both evaluated over the same still-flowing
  set. The old implementation was the max excluding frozen cells ÷ the max over all cells, so the denominator was dominated by one pathological cell,
  and when that cell froze it dropped out of the numerator only, causing the ratio to avalanche. The measured values are reported in metadata as
  `tau_rep_flow` / `tau_rep_baseline`.
- **Skin-layer fixed point is re-solved per live domain** (`_solve_domain` + outer domain loop):
  the skin and fill time of the live domain no longer depend on the volume of dead domains the melt cannot reach. In a synthetic case the
  live result is bit-for-bit unchanged with respect to the amount of dead domain (20/100 cells).
- **`skin_max_iterations` default 5 → 20**: the positive feedback of the constant-pressure approximation (skin growth → higher resistance → larger T_fill →
  more growth) avalanches by design once past the critical point. The old default hit the cap mid-avalanche and returned a plausible
  intermediate state with zero freezing and 27× inflation, raising only the `skin_converged=False` flag.
- Known trade-off: **the skin-growth clock is still the arrival time** (arrival-snapshot semantics).
  As a result of correcting the arrival time, the thin choke next to the gate is snapshotted at "its own volume/Q ≈ 0 s"
  and almost no longer freezes. Physically, the region near the gate is exactly what keeps aging throughout filling
  (the exposure clock s(T_fill − t_arr) is the right direction). The old linear mapping happened to paper over this on the conservative side.
  Tracked in a separate issue.
- **Domain-loop safety valve made graceful** (Codex P2): frozen cells remaining when `MAX_DOMAIN_PASSES` (4→64; the domain
  strictly shrinks every pass, so this is a runaway stop that should not be needed) is hit are dropped to the unfillable side
  without re-solving, and `metadata["domain_converged"]=False` is set.
  In the old implementation, remaining frozen cells stayed live with finite fill times, and `short_shot_cells` could
  exceed `unfillable_cells`.

## [0.24.1] — 2026-08-19

**Cavity regions not connected to a gate are now rejected at the entrance instead of being filled with plausible garbage
(fixes Issue #58).**

### Fixed

- **`solve()` detects components unreachable from a gate and raises `ValueError`.** A connected component cut off
  from every gate is a pure-Neumann block with no Dirichlet point = a singular matrix, and
  `spsolve` returns garbage without warning. Depending on how it is cut off, the garbage becomes either an astronomical τ (noticeable
  on screen) or **a uniform fill time, as if the whole region filled at the same instant** (no clue
  on screen). The latter was actually happening with the default mesh of Profile gate.
  Detection is `check_gate_reachability()` (a module function in `core/solver.py`), which labels connected components with
  **4-neighbor** connectivity, the same as the 5-point stencil. With 8-neighbor connectivity, a "bridge" made only of diagonal contact would be mistaken
  for a connection and the singular block missed, so the definition of connectivity itself is pinned by a test
- **Same check in the multilayer solver.** `MultilayerHeleShawSolver.solve()` calls `_solve_tau_field` directly
  without going through the base `solve()`, so a check on the base side alone would let the
  multilayer path slip through
- **Profile gate is rejected ahead of time at assembly.** When `gate_exit_width` is below the mesh spacing
  and the gate bank fully closes the bottom row (e.g. exit width 0.5 mm / cell 1.0 mm),
  it is rejected before reaching the solver with a message naming the parameter: "widen the exit or refine the mesh."
  The solver-side check remains as a safety net for arbitrary masks
- **Why rejection was chosen over dropping into `unfillable_mask`**: a disconnection is a mistake in building the geometry,
  not "the melt physically cannot reach." Routing it into the short-shot mechanism would turn "the input is wrong" into
  "the model predicted a short shot"
- The case where gates exist but all lie outside the mask is also rejected with "no gate lies inside the cavity mask"
  (previously only empty gates were checked)

### Tests

- 422 → **427** (+5). Boundary tests that had been set up as strict xfail flipped to XPASS with the fix, so the
  markers were removed and they were rewritten to assert "is rejected" in the present tense.
  All 5 mutation injections (remove base check / remove multilayer check / remove bank check / switch to 8-neighbor / invert decision)
  were confirmed on a cloned repo to be caught one-to-one by the targeted test

---

## [0.24.0] — 2026-08-19

**Unified the terminology on "ショートショット" (short shot) and fixed outdated descriptions in `README.md` and an equation sign
that disagreed with the implementation.**

### Changed

- **"短ショット" → "ショートショット"** (20 places). What Japanese molding shops actually say is
  **ショートショット**, the phonetic rendering of "short shot"; "短ショット" is a half-translated coinage.
  Moreover `ショートショット` was already used in 8 places, so **both spellings were mixed**, and
  in `README.md`, within the same section, the skin layer was described as "ショートショット予測" while the multilayer model said
  "短ショット判定". Code identifiers (`short_shot_mask` etc.) are in English and unchanged.
  `logs/` are dated work records and were not touched
- **Matched the equation sign to the implementation.** `README.md` (2 places) and the `core/solver.py`
  docstring said `∇·(S∇τ) = 1`, but what is actually assembled is diagonal `+Σcoeff` /
  off-diagonal `−coeff` / right-hand side `+1`, i.e. **`−∇·(S∇τ) = 1`**.
  The resulting τ was correct, but readers tripped over it every time they followed the sign. `CLAUDE.md` noted
  this discrepancy, but fixing it is better than annotating it, so it was fixed
- **Brought the `README.md` roadmap up to date.** Items 1–3 were implemented (foundation /
  3 parametric gates + JSON spec / skin layer, multilayer, viscous heating stage 1) yet read as
  not started. Split into done and remaining, and added viscous heating stage 2 to the remaining items
- **The description of "local viscosity iteration" misattributed the multilayer solver** (Codex finding).
  "Evaluates Cross-WLF only once at a representative shear rate" applies to the no-wall-cooling mode;
  the multilayer mode re-solves the `(N, ny, nx)` viscosity field every iteration. What is missing is the
  origin of γ̇ — the velocity fed into the Poiseuille profile is fixed to a single representative injection speed `V`, and there is no loop
  that feeds the local flow velocity from the solved τ back in. Renamed the roadmap item and the feature-table row to
  "feedback from the flow field to velocity and shear rate"
- **Withdrew the description of `A` as "positive-definite form"** (Codex finding). With this sign convention
  the pre-constraint operator is symmetric positive semidefinite, but `_build_linear_system` applies Dirichlet **only to rows**
  — it collapses gate rows to identity rows while neighboring interior rows keep the `−coeff` in the gate columns.
  The assembled `A` is therefore non-symmetric and not SPD. Implementing the roadmap's CG / AMG with a symmetric-only solver
  on the basis of this description would lose the convergence guarantee.
  Corrected the 3 places in the `core/solver.py` docstring, `README.md`, and `CLAUDE.md`, and stated explicitly
  that eliminating the gate columns is a prerequisite (elimination is exact because `τ = 0` at the gate).
  Since these 2 claims are claims about the code, regression tests were added to `tests/test_solver_1d.py`
  (non-symmetric entries are confined to gate rows / elimination does not change the solution).
  Both still pass after a "symmetrize" fix is applied.
- **Also corrected the SPD guarantee in the customer-facing technical document** (Codex finding).
  `docs/流動解析の仕組み_想定問答_技術編.md` asserted to customers that "this tool's equations are symmetric positive definite."
  **The conclusion (existence and uniqueness of the solution) still holds** — the interior block with the unknowns fixed at the gate
  eliminated is measured to be symmetric with smallest eigenvalue 1.3e-07 > 0, and solving only the interior system
  agrees with the full-system solution to 5.7e-15. Only the wording of the justification was wrong, so instead of withdrawing the guarantee
  it was made precise as "the main block with fixed points eliminated is symmetric positive definite," with a note that the
  implementation uses the shortcut of writing only to rows.
  This property is a promise to customers, so a check was added to `tests/test_solver_1d.py`.
  **The cause of this oversight was skipping a sweep** — when correcting SPD, I listed file names as the target of `grep`
  instead of sweeping the whole repo, so all of `docs/` was out of view. The whole repo was swept again, and it was
  also confirmed that `docs/` has no leftover terminology (短ショット / short shot) or sign issues
- **Rewrote the "it's linear so it solves in one shot" section to match the default mode** (Codex finding).
  The technical document said "an exact answer comes out in one shot without iteration" and "uncertainty about whether it converges
  does not arise in the first place," but this applies to wall-cooling model "none." **The UI default is
  multilayer** (the `wall_model` radio in `app.py` has `index=2`), and both skin layer and multilayer iterate
  τ ↔ T_k ↔ η_k as a fixed point. It was telling customers the opposite of the default behavior.
  Rewrote it separately for the 3 modes, stating that for the 2 iterating modes convergence is not guaranteed in general,
  that in exchange the iteration count and convergence status are shown every time and damping is used for stabilization,
  and that the nonlinearity being recovered is only via temperature
- **Stated the gate-reachability condition in the SPD guarantee** (Codex finding). If there is a cavity component
  cut off from every gate, that block is a pure-Neumann Laplacian with a remaining zero eigenvalue, and
  the solution is not uniquely determined. Measured smallest eigenvalue 9.8e-20 / condition number 1.6e+16, and on top of that
  `solve()` runs to completion without raising and returns τ = 8.85e+20 (4.2e14 times the normal value).
  The document was fixed by stating the condition. **Hardening the implementation was split off into Issue #58** —
  the standard builders all produce connected masks, so it cannot be reached from the UI, but
  `Geometry` is a public API that accepts any mask, so it could be hit in the future.
  The boundary is held by `tests/test_solver_1d.py` in the form "either rejected or else singular,"
  so the tests pass whichever approach is used to fix Issue #58
- **Continuation of the above (4 Codex findings).** (1) The connectivity condition's "the standard builders all produce
  connected shapes, so normal use never violates it" was **wrong**. With Profile gate, if the gate exit width is
  narrower than the mesh spacing, the gate land wall at `core/profile_gate.py:757-763` closes the whole exit row,
  and the entire part is cut off from the gate. **It happens even at the default 1.0 mm/cell**
  (measured: with `gate_exit_width=0.5`, isolated cells 14700 = the whole plate). Moreover end-to-end
  no exception is raised, and the whole part becomes a **uniform value** of `0.5241 s` — unlike an absurd number, you cannot
  notice it on screen. Raised the severity of Issue #58 and added this to it. (2) "These 2 modes re-derive viscosity from each layer's temperature,
  apply damping, and display convergence" **does not apply to the skin layer**. The skin layer
  fixes `eta` once before the loop, and only `h_core` iterates. Damping appears in 11 places, only in
  `multilayer_solver.py`, and the convergence display is only inside `if multilayer_on` in `app.py`.
  Rewrote the 2 modes separately. (3) The regression test did not allow a fix via the `unfillable_mask` approach.
  Split the mathematical claim (the block is singular) and the requirement on the implementation (do not fabricate fill times)
  into 2 tests, and made the latter `xfail(strict=True)`. (4) Other customer-facing documents still asserted "evaluates viscosity once
  and solves once," contradicting the technical document this PR rewrote.
  Rewrote 2 places in `想定問答.md` and 3 places in `プログラム解説.md` to limit them to the "none" mode, and
  added a description of the iterating modes. Also updated level B in the technical document's level table from "reachable with effort" to
  "implemented, but accuracy cannot be guaranteed and viscosity iteration is via temperature only"
- **The end of `顧客向け解説.md` was also brought in line with reality, after the user's decision.** It said "even though options are listed,
  the computation behind them is not there," "it is not even useful as a reference,"
  and "modification of the program has stalled," but the multilayer solver is implemented with 42 tests
  and is the UI default mode, contradicting the facts. **The statement of limitations was kept**, and
  rephrased as "implemented and working, but accuracy cannot be guaranteed — the physical model is heavily simplified, and above all
  no calibration against the actual machine has been done." Made explicit that the option is positioned as
  "a way to see which direction the trend moves when solidification is considered," and added that this is the same character as the
  positioning of the whole tool (relative comparison, not absolute values). Also changed the ✅❌ table entry "local
  viscosity update during flow (evaluated once at a representative value)" to "feedback from flow velocity to viscosity"
- **Eliminated inconsistencies within the documents created by the rewrite above** (4 Codex findings). A textbook case of a fix
  creating new discrepancies; one of them was **recreating in the program explanation the very error just fixed in the
  technical document** (it wrote "the iteration count and convergence status are shown on screen" for both modes;
  the display is only in the multilayer pane of `app.py`). Instead of fixing things by name, I swept the whole of
  `docs/` for 3 classes — (A) assertions of the solve-once kind, (B) iteration/convergence display,
  (C) places that read as if only the skin layer handles cooling — before fixing. The 2 places of "solves in one shot" in
  `商用CAE_物理モデルと数式.md` **are still correct, because they mean there is no time stepping**, and were kept as distinct
  (the multilayer fixed point is not time stepping). Also specified `raises=AssertionError` on the `xfail`
  — without it, even an unrelated `RuntimeError` is swallowed as an "expected failure," and a
  regression turns into XFAIL and becomes invisible (confirmed by mutation injection)
- **Corrected the classification of the multilayer model as "3D"** (Codex finding). The level table had B,
  "simple 3D multilayer," as implemented, but the multilayer solver is a model that **integrates through the thickness to return to a
  2D fluidity**, and the in-plane unknown remains the 2D `τ(x,y)`. It also contradicted
  "this tool sticks to A" in the same document. Reorganized A as "2D + pseudo-3D in the thickness direction only" and
  B as "simple 3D including in-plane." Also corrected "the nonlinearity being recovered is only via
  temperature" — via temperature is the multilayer story; what the skin layer closes is a loop via
  narrowing of the flow channel
- **Matched the description of iteration to implementation details** (3 Codex findings). (1) What is passed to the skin layer's `s`
  is not the local flow velocity but the arrival time `t_arr` normalized from `τ` (`core/solver.py:405-425`).
  "The slower the flow, the thicker the frozen layer" is confusing next to the following "the loop from flow velocity is not closed,"
  and differs from the implementation. Changed to "the later a cell is reached, the longer it has to cool before then."
  (2) "Both repeat until balanced" reads as if it always converges. In reality, when
  `skin_max_iterations` / `max_iterations` is reached it stops unconverged, which also contradicted the
  technical document's "convergence is not guaranteed in general." Stated that it stops.
  (3) Where the multilayer iteration returns to. Re-deriving viscosity rebuilds the fluidity `S` and the matrix as well
  (`multilayer_solver.py:483-489`), so it goes back to 2 via 3–5, not "back and forth with 5 onward"
- **The zero-eigenvalue check depended on the sign** (Codex finding). `ev.min() / abs(ev).max() < 1e-12`
  **slips through when the interior block becomes negative definite** — even with no zero mode the ratio
  becomes negative, so it always passes, and it was not verifying "the solution is not unique" as the test name claims.
  The same pattern as the "vacuous guard" I separately found this session. Changed it to assert
  positive semidefiniteness (`ev.min()/scale > -1e-12`) and the existence of a zero mode (`abs(ev).min()/scale < 1e-12`)
  separately (confirmed detection with a negative-definite injection). Also added to the documents that the skin layer has a 3rd
  stopping condition (the iteration ends when all flow paths other than the gate are blocked, `core/solver.py:461-466`)
  — noting as well that this is a short-shot result, not a computation failure
- **Also unified the English notation `short shot` left in Japanese text** (Codex finding).
  3 places in `app.py` (help for the solidification threshold, the title of the skin-layer expander,
  the caption of the core layer), 1 place in `README.md`, and 3 places in `CLAUDE.md`.
  matplotlib in-figure labels, English docstrings / comments, and test descriptions are in an
  English context and were left as is. The heading `**short shot**` at `CLAUDE.md:97` was also fixed afterwards
  — the first sweep excluded **whole lines** containing identifiers, so lines where `short_shot_mask`
  and prose coexist dropped out entirely. After restructuring the sweep to blank out identifiers as strings first and then
  look at the rest, it was confirmed that no other prose was missed
  (the remaining English notations are only quotations of strings drawn in figures, explanations of etymology, and English contexts)

### Things checked in `README.md` and found fine

- `--cases PP_baseline PP_dual_gate` — both actually exist in `run_demo.py`
- The 8 materials, Python 3.11+, and the deployment-config caveat — match the current state

---

## [0.23.0] — 2026-08-19

**Removed image input (the path that builds geometry by binarizing a PNG/JPG).**

It was never used. It builds the outline from a binarization threshold and a 1px=1mm assumption, so there is no backing for its accuracy, and
now that the path of building a JSON spec from a drawing PDF and loading it (Profile gate) is established, it has no role left.
Rather than just hiding it from the input radio, **`core.geometry_from_image` was deleted outright** — a function unreachable from any entry point
rots into a claim of capability nobody verifies. If it is ever needed again, pick it up from git history.

### Removed

- `画像から生成 (PNG/JPG)` in the input radio, and its group of sidebar inputs
  (image upload / uniform wall thickness / px→mm conversion / invert / binarization threshold)
- The image branch of `build_geometry()`. The step that wrote the uploaded image to a `tempfile` went with it
- `core.geometry_from_image`, its export in `core/__init__.py`, and the `PIL` import in `core/geometry.py`

### Changed

- The end of `build_geometry()` raises `AssertionError`. Every branch returns, so
  this is reached only **when an option is added to the input radio and the builder is forgotten**
- Made `cfg` of `config_settings()` **required** (see below). Passing `cfg=None` raises `TypeError`
- Updated the scope caveat, the geometry input in `README.md`, and the relevant description in `CLAUDE.md`

The Pillow dependency **remains**. GIF export (`PillowWriter`) and getting the player's frame dimensions use it.

`assets/` (listed in `CLAUDE.md` as the place for image input) never existed in the first place, so
its description was deleted along with it.

### Tracking leftovers (3 rounds of Codex review)

Deletions fail on **leftovers** rather than on "what was deleted." The findings from the 3 rounds, and their causes.

| Round | What was found | Cause |
|----|--------------|------|
| 1 | The table in `README.md` / the rationale for the limit in `.streamlit/config.toml` / an incomplete task in `PROGRESS.md` | The sweep terms were only `画像から生成` and `geometry_from_image`, too narrow |
| 1 | `config_settings(cfg=None)` became unreachable | Missed that image input was its only user |
| 2 | `core/solver.py` / `tests/test_fill_render.py` / `core/visualizer.py` / a self-contradiction in this section | The sweep terms were widened, but **the 154-line result was cut with `head -25`** and judged "no remnants" |
| 3 | The result-ZIP contract in `CLAUDE.md` ("the input image is also recorded by name and hash only") | The sweep term was `画像入力`, and the actual wording was **"入力画像", with the word order reversed** |
| 3 | A duplicate block left in this section | When replacing the section, its end was searched for with `---`, but **the Markdown table separator row `\|------\|` contains `---`**, so it was cut in the middle of the first table |

What was learned:

- **A sweep cut with `head` is worse than no sweep.** Looking at 25 lines and acting as if you saw everything leaves false confidence.
  If you cut, state how many were dropped
- **Japanese compound words are not caught across word order.** Searched for "画像入力" and missed "入力画像"
- **Do not search for `---` as a section separator.** In documents with tables it is contained in separator rows. Anchor including the
  start and end of the line
- **A sweep only produces candidates; judge after reading the context.** "end up having to measure dimensions from the image and back-calculate"
  in `app.py` was a false positive; this image is the output figure that goes into the ZIP. Fixing it mechanically would have broken it
- **Do not patch documents piecemeal; rewrite the whole section.** This section's self-contradiction (directing the `cfg=None` contract
  opposite ways in the first and second halves) came from writing the before and after of the fix separately

`cfg` of `config_settings()` is now **required**. Passing `cfg=None` raises `TypeError`.
The shape of a record without a `config` key is no longer produced by anything, and therefore nothing verifies it.
The empty-source test was changed to pass a valid config — because with cfg required, as it was it would
**no longer be possible to tell whether it failed because of source or cfg**.

---

## [0.22.0] — 2026-08-19

**Gate specs kept locally can now be picked from a list instead of being uploaded every time.**

Spec JSON built from drawings cannot be placed in this repo (public), so until now every comparison meant
diving into a local folder in the file dialog and uploading again. This removes only that chore.
**Nothing more is stored** — it just points to one location, and makes neither copies nor caches.

### Added

- `core/spec_source.py` — resolves where specs are and the precedence of input sources. A set of pure functions with no IO
- Changed "JSONアップロード" in the "スペック入力" radio to **"ローカルから読込" (load from local)**.
  The drop zone stays as is, with the list dropdown placed above it
- When both the drop and the list are active, **the drop wins** (the most recent explicit action).
  The winning side is shown in the sidebar, and the losing list is `disabled`
- `tests/test_spec_source.py` — 44 tests

### Usage

Create `local_specs` at the repo root (gitignored). Once:

```
ln -s /path/to/specs local_specs
```

Without it the list does not appear. **That is the default state of the public instance.**

### Design rationale

- **The environment variable `MFS_SPEC_DIR` was not adopted.** Because it fails open. Streamlit Cloud has
  no UI for environment variables and secrets are the only channel, but `streamlit.runtime.secrets` promotes top-level
  `str`/`int`/`float` to `os.environ`. Just pasting the local `secrets.toml` into the Cloud
  settings field — the ordinary way to move secrets — would sprout a file-reading entry point on the public instance.
  A gitignored path **has no way to exist** on Cloud, which consists of checkout + secrets
- **Fixed the root to a single gitignored name.** With a configurable root, pointing it inside the repo and
  running `git add -A` becomes the shortest path to committing customer dimensions to a public repo, and neither an env gate nor
  path confinement helps against that. Fixing it makes this structurally impossible
- **The list defaults to "— 未選択 —" (not selected).** Geometry is built on every rerun, not behind the run button,
  so defaulting to a real file would **draw the customer's geometry the moment the page opens**. Also, with a default
  selection "both filled" becomes the norm, and the precedence fires every time
- **No text path field was made.** With no user-entered path, the problems of `..`, `~` expansion, prefix matching,
  extension restriction, and case **do not exist, rather than being forgotten**. A one-off file outside the root is loaded by drop

### Fixed (caught by my own tests during implementation)

- `Path.resolve()` throws **`RuntimeError`**, not `OSError`, on a symlink loop,
  and its message contains the absolute path. The default of `client.showErrorDetails` is `full`, so a missed catch
  shows the spec folder's path on screen
- `Path.glob()` **swallows permission errors and returns empty**. An unreadable folder was a false green
  indistinguishable from "0 specs." Changed to `os.scandir`
- The spec-loading path caught only `JSONDecodeError` and `ValueError`;
  `PermissionError` / `FileNotFoundError` went straight through. Added `OSError` and `UnicodeDecodeError`, and
  **the exception message is not displayed** (the absolute path contains customer and project names)

### Intended trade-off

The list **shows every file name in the folder on screen**. The exposure, which was 1 file with the upload approach, becomes
the number of files in the folder. File names are the project identifiers themselves, so exposure during screen sharing increases.
Accepted as the direct price of convenience (the expander is collapsed by default).

---

## [0.21.0] — 2026-08-19

### Rationale for the colormap guard (4 rounds of Codex review)

So that swapping the colormap does not break things, the tests judge not by "does the name match" but by **colorimetry of the ramp actually drawn**.
Review hardened it in 4 stages:

| Finding | Substance | Response |
|------|------|------|
| P1 stop parsing | Plotly returns `rgb()` form or hex depending on the colormap, and `mcolors.to_rgb` rejects the former. Cividis happens to be on the hex side | Support both hex and `rgb()`/`rgba()` |
| P2 gamma | The weighted sum was taken on gamma-encoded values. `rgb(205,78,46)→rgb(242,34,36)` decreases by weighted sum but increases in true luminance | Linearize sRGB, then WCAG relative luminance |
| P2 interpolation | With the gamma fix, the argument "monotonic at stops implies monotonic everywhere" no longer holds (Plotly interpolates linearly in encoded RGB; luminance is nonlinear). `bluered_r` actually reverses by 0.0023 between stops | Dense sampling between stops |
| P2 readability | Monotonic alone would also pass a near-black→black ramp, making the map pitch black and unreadable | **Contrast ratio of at least 3:1** between the thin end and the thick end (WCAG 1.4.11; `cividis_r` is 12.56:1) |

The HSV saturation guard on the thin end is kept. The yellow of `cividis_r` has only **1.25:1** against white in luminance,
and what separates it from the white background is **saturation**, not lightness — judging by lightness would reject
the adopted colormap itself. Saturation covers "can it be told apart from white," and the contrast ratio covers "does the ramp carry information."

**Fixed the wall-thickness map's light/dark direction being reversed — thin is now light and thick is dark.**

The direction was: the thicker, the brighter yellow; the thinner, the darker blue. This is the opposite of the universal reading
"darker ink = more quantity," and furthermore, in transparent resin parts **light is attenuated more the thicker the part, so it actually looks darker**,
so to eyes used to the real parts it was doubly upside down.

### Changed

- Reversed the wall-thickness map colormap `cividis` → **`cividis_r`**. Applied in all 3 places: the 2D design drawing ("成形品設計図" in `app.py`),
  the 3D view (`render_3d_thickness_map`), and the multilayer thickness panel (`render_layer_map` /
  `render_layer_grid`)
- Made `core.visualizer.THICKNESS_CMAP` the single source of the colormap. The 3D side has its own
  `core.visualizer_3d.THICKNESS_COLORSCALE` (`"Cividis_r"`) matching Plotly's naming, and consistency between the two is ensured by tests
- Fill time (`turbo`) and pressure (`magma` / `Turbo`) are **unchanged**. Wall thickness is the input geometry, not a solved result,
  so it stays a single-quantity lightness ramp to keep it visually in a different category from result figures

### Considered and not adopted

**Monochrome "darker = more" ramps (`Blues` / `bone_r` etc.)**. The purest as a metaphor, but a map whose low end
approaches white washes out the product plate. The plate is both the thinnest region and the only region people look at;
with `bone_r` the 0.35mm band became pure white and the outline itself disappeared, and in 3D the top surface melted into the light-gray PL floor and the white background.
The step contrast between adjacent zones (0.35 / 0.50) also drops. `cividis_r` avoids this because its thin end remains a saturated
yellow, and cividis's property of being designed for color vision diversity survives the reversal.

### Tests

15 added (358 → 373). The colormap is not matched by name; instead **the ramp actually drawn is measured** for 4 properties (the measured values below are from plotly 6.7.0. `plotly>=5.18` has no upper bound, so **the criteria are fixed and only the expected values follow**).
A name-only comparison would pass even if `Cividis_r` were not actually reversed, so it is not usable as a criterion.

| Property | Test | Failing example |
|------|------|---------|
| Monotonic light→dark | Densely sample the ramp including between stops; each adjacent sample must be darker | `Blues_r` (reversed), `jet_r` (light→dark at both ends, reverses by 0.719 in the middle), `bluered_r` (passes the stop test, reverses by 0.0023 within an interval) |
| Carries information | Contrast ratio between thin end and thick end ≥ 3:1 (WCAG 1.4.11) | near-black→black ramp (1.00:1), `jet_r` (1.44:1) |
| Distinguishable from white | HSV saturation of the thin end > 0.5 | `bone_r` (low end is pure white) |
| 2D and 3D match | `THICKNESS_COLORSCALE.lower() == THICKNESS_CMAP` | When only 3D uses a different colormap |

Measured for `cividis_r`: monotonic ✓ / 12.56:1 / saturation 0.78. **Separating from white by saturation is intentional**: the
thin end (yellow) of `cividis_r` has only 1.25:1 against white in luminance — what separates it from the white background is saturation, not lightness,
so judging by lightness would reject the adopted colormap itself.

Dense sampling is needed because Plotly interpolates linearly in encoded sRGB, while relative luminance applies a nonlinear transform
first. The transfer function is convex, so luminance within an interval can dip below the darker endpoint and come back — 
**this is a property of the color space, not of a particular colormap**. So the witness was pinned not with a built-in colormap but with
**a constructed counterexample** (green→magenta, endpoints 0.377→0.285 look monotonic but dip to 0.142 in between).
`bluered_r` currently shows the same behavior too (0.0023), but even if that changes, the need for dense sampling does not.

The discriminating power of the criteria themselves was pinned with an expected-value table for `Cividis_r` / `Blues` / `Blues_r` / `Jet_r` / `Bluered_r` / `Greys_r`.
With a mutation injection that swaps the constant, the name-match assert fails first and so does not test the criteria, so
the criteria are tested directly. Color computation is consolidated in `tests/colorimetry.py` — the reason for consolidating was that
**the luminance formula had been hand-copied into 2 files, and both had the same mistake**.

---

## [0.20.0] — 2026-08-19

**Molds in which part of an island is filled by welding to form a weir can now be solved as they are.**

A revised production drawing called for the modification "build up the 10mm downstream of the island by welding and make the channel thickness a constant 0.1mm."
The existing schema could not express this, and the closest approximation (making the island a constant depth)
squashed the center-to-edge fill-time difference to less than half and even moved the location of the last-filled point. 0.1 and 0.35
differ by 43× in conductance via h³, so the moment a weir is substituted with something that is not a weir, evaluating the modification
no longer holds.

### Added

- `island.weld` (optional): a subsection with `t_range` and `depth`. Overwrites with `depth` the depth of cells that are inside the island
  and within the `t_range` band. Weld metal fills the pocket,
  so `depth ≤ land.depth` is required, and `t_range` must lie inside
  `[land.length, island.end_dist]`. The steps at entry and exit are a sharp-cut approximation,
  like the island's own steep walls.
- Export `WeldSpec` from `core`.
- 6 tests in `tests/test_geometry_profile_gate.py` (constant depth within the band / unchanged outside the band and outside the island /
  volume reduction checked against quadrature / bit-identical to old behavior when unspecified / JSON round-trip /
  5 kinds of validation).

### Fixed

- **Numeric fields in spec JSON accepted NaN / ±Infinity** (Codex review finding,
  reported as the 2 weld fields, but the same hole exists in every existing numeric field).
  All range checks in `validate` are comparisons, so NaN **passes both the upper and lower bounds**.
  Downstream, if NaN enters the depth field, the volume and the solution all become NaN, and if it is used in a mask comparison
  all conditions become False and **the specified feature silently disappears** (the latter is worse — you cannot notice
  that the geometry is wrong).
  The defense was placed at **one point**, `validate()`. It recursively walks the dataclasses, collects every numeric leaf,
  and rejects non-finite values. Rejecting on the parser side (`_num` / `_pair` / `_line`) was tried first, but
  **the spec dataclasses are exported and can be constructed directly without going through JSON**, so the entrance cannot be fully sealed
  (Codex's follow-up finding). The walk follows `dataclasses.fields`, so **fields added later are
  automatically protected**.
  The walk's scalar test uses `numbers.Real`. With `(int, float)`, **NumPy scalars silently
  drop out** — `np.float32` / `np.int64` are not subclasses of the built-in types, so
  if a spec is assembled from values read from an optimization sweep or an array, the number of leaves in the walk shrinks and
  validation is bypassed (`np.float64` happens to be a subclass of `float` and is caught, but
  that is luck, not a guarantee. Codex's 4th finding).
  The tests do not name the reported locations; they check that (1) the walk reaches all 28 numeric leaves of the spec,
  (2) all 3 values NaN / +Inf / −Inf are rejected via JSON, (3) they are also rejected with direct construction bypassing the parser
  (top level / nested / doubly nested / tuple elements), and
  (4) NumPy scalars (float32 / float64 / int64) do not drop out of the walk, and are accepted if finite and rejected if NaN.

- **`[a, b]` / `[[t,w],[t,w]]` fields accepted anything with 2 elements**
  (Codex review finding). `a, b = val` takes any 2-element iterable, so
  `"t_range": "68"` was interpreted as `(6.0, 8.0)`, and **silently passed as a geometry whose weld band is in a completely different
  place**. Other broken values (`[true, false]` etc.) were rejected not by type checking but
  by happening to hit the later range check, and `"68"` is inside `[land.length, end_dist]`
  and increasing, so it hits nothing. Now requires an actual array whose elements are real numbers that are not bool.
  The test walks all 5 array fields and checks that 8 kinds of broken values are rejected.

### Changed

- None. The behavior of specs without `island.weld` is completely unchanged (ensured by tests).

---

## [0.19.0] — 2026-08-19

**Short shots no longer masquerade as "takes a long time."**

Cells whose flow channel the skin-layer model has closed are floored at `min_core_thickness_mm`
(default 0.01 mm) for numerical stability. The τ of closed cells is orders of magnitude larger than the rest, and it
became `tau_max` and **set the reference for absolute time**. As a result,

- The total fill time T_fill inflated to several times the actual value (measured: 0.115 s → correctly 0.031 s)
- Not a single cell exists in the upper 73% of the colorbar — all cells are squashed at the bottom end
- Frames divide time evenly, so 40+ of 60 frames are a dead match where nothing happens

Changed to **cells that do not fill have no fill time**.

### Changed
- **The time axis is taken from the τ of cells that fill** (`tau_max_flow`). Frozen cells and the regions sealed off
  behind them are not included in the reference. With the skin layer OFF it is completely identical to before
- **Frozen cells are treated as walls**, and regions that can no longer be reached from the gate are also
  included in "does not fill" (connected-component analysis with `scipy.ndimage.label`). What decides it is connectivity to the gate,
  not local wall thickness
- Added `FlowResult.unfillable_mask`. The `fill_time_s` / `pressure_norm` of those cells are
  **NaN** (not a large number. Putting in a number would mean "it will arrive eventually")
- In the fill animation, unfilled cells stay to the end in a dedicated color (`SHORT_SHOT_RGB`, the same red as the short-shot map).
  They are not exposed whatever is passed as `filled`
- Show `short shot: N cells` in the frame title. The fill ratio is displayed to 1 decimal place, so
  a short shot of a few cells is rounded into "filled = 100.0 %" and disappears
- `unfillable_cells` / `sealed_off_cells` / `tau_max_flow` in metadata
- `tests/test_short_shot_timeline.py` (30 tests)

### Implementation notes
- **The color field is extrapolated from "cells with finite values"** (not from the cavity mask).
  If NaN cells are filled with a substitute value and extended from the mask, that value bleeds into the neighboring live cells
- The decision to cover unfilled cells does not depend on `filled`. Right now `NaN <= t` is False, so
  they end up covered, but that is a property of NaN comparison, not a design decision
- **The multilayer solver's short-shot decision (based on center-layer temperature) is not included in this treatment**.
  It does not floor conductance, so it has no τ pathology, and it is only reported as an indicator

### Implementation notes (review findings)
- **Weld lines / air traps are computed only on cells that fill**. The τ of frozen cells is orders of magnitude
  larger than the rest, and that is exactly the local-maximum shape the air-trap detection looks for, so
  **defect marks appear where the melt never arrives**. The test first asserts the premise "with raw τ they actually appear"
  and then checks "after the fix they do not appear" (the first version, written without the premise, passed without observing anything,
  because the dead region's maximum was on the boundary and outside the detection range)
- **Contour lines are not drawn if there are fewer than 2 distinct finite fill times**. If frozen cells seal off everything
  except the gate, the only finite value is the gate's 0, and `contour` crashes with
  `ValueError: Contour levels must be increasing`. Losing a completed analysis
  for the sake of decoration is not worth it
- **The case where nothing flows is handled separately**. If frozen cells seal off everything except the gate,
  all that remains in "cells that fill" is the gate's τ=0. Falling back to the overall maximum here
  **brings back the τ of the dead cells that were supposed to be excluded** (measured: T_fill was 64 times the
  baseline, while the rendering side fell back to a fixed 1.0 s, so
  2 contradictory time axes came out for the same result). `_tau_reference` returns `None` if there is no
  usable reference, and solve keeps T_fill at the baseline
  (`no_flow` in metadata). The fallback of `fill_time_max` was also changed from 1.0 s to
  `total_fill_time_s`, so the axis and the headline agree
- **τ is re-solved after cutting off the unfilled region**. The first solve is over the whole cavity,
  so the sealed-off region keeps emitting unit sources, and the frozen band also
  lets flow through at the `min_core` floor conductance. **The volume of the dead region ends up passing through the live cells
  upstream, inflating their τ** (measured 3.3× on a strip blocked by a band). It is re-solved as a subproblem of
  only the live region, and all subsequent fill time, pressure,
  weld lines, and air traps are taken from that solution
- **The baseline time is also computed from the volume that fills**. Volume the melt does not reach takes no
  time. `_baseline_fill_time(geom)` was split out with a Geometry argument, and the effective compression
  factor is also evaluated on the live region
- **The default path that does not specify an injection rate also scales by volume**. When `injection_volume_flow_cm3s`
  is `None`, the default is "fill the cavity as drawn in 1.5 s" = **a rate**, not
  a duration. Read as a duration, a short shot with only 10% of the volume left would also get the same
  1.5 s, and live-volume scaling would stop working only on the default path.
  The rate is derived from `DEFAULT_FILL_TIME_S` and the original cavity volume and used
  (asking the restricted solver for the rate would divide back by its own volume and return to the default value)
- **The cavity and the skin-layer fixed point are determined together**. Skin thickness is determined by arrival time, and
  arrival time is determined by "which cavity was solved." If the fixed point is run only once over the whole cavity
  and that `h_core` is reused while re-solving only τ, **the live region's
  skin layer remains dependent on the dead volume**. Changing the dead region placed behind the same live geometry
  between 20 cells and 100 cells shifted the reported fill time by **3.9×** (0.31–8.28× over the
  whole sweep). Once the live region is cut out, **the skin-layer fixed point is re-solved there**,
  and if new cells froze, it is cut out again. The region only shrinks, so it always terminates
  (`MAX_DOMAIN_PASSES = 4` is a runaway stop; measured to converge in 1–2 passes). After the fix, the ratio is **1.0000** in all cases
  of the sweep above, and the live region's `h_core` and `fill_time` are bit-for-bit unchanged with respect to the
  amount of dead region
- `_restricted_to` returns a copy with the injection rate **pinned to its current value**. Restriction
  changes the cavity, not the molding machine (if the implicit rate were left, the shrunken volume would be divided by a rate derived from the
  same shrunken volume, and the restriction would cancel itself out)
- `domain_passes` in metadata (the number of cut-outs; 0 means no freezing)
- **Inflation compares 2 states of the same region**. The τ_max ratio of the live region "with skin" and
  "without skin." Comparing against a whole-cavity reference would mix geometry changes into the
  skin indicator
- **The time axis is decided in one place**. The color scale, the title, and the frame times each
  computed the maximum of `fill_time_s` on their own, and each fell back to 1.0 s.
  Fixing only one leaves the rest telling a different time (measured: headline 0.023 s while the
  animation ran over 1 s). `fill_frame_times` also goes through `fill_time_max`
- **The fixed-point loop exits as soon as everything except the gate has frozen**. The next iteration
  used `None` as a divisor and crashed with `TypeError` (the answer was already out, yet
  not only rendering but solve itself died). Reproduced with c_skin=3.0, thin wall 0.03mm, etc.
- **The pressure map explicitly paints unfilled cells**. NaN becomes the colormap's bad color
  (transparent), but the cavity's alpha is immediately fixed to 1, so it becomes **pitch black** and
  reads as "lowest pressure." Measured 8,257 pixels appearing as the bottom color of magma

### Known trade-offs
- **The displayed value of T_fill changes** (only when there are frozen cells). The old value was "the time it would take if the cells
  that freeze and do not fill were forced to fill," not the part's fill time.
  Showing the time over the range that fills + the number of unfilled cells is safer than disguising a fill failure as a duration

---

## [0.18.0] — 2026-08-18

**The settings that produced a result can now be recovered from the downloaded ZIP.**

The ZIP now includes `settings.json`. `metadata.json` held only the solved results (volume, τ, iteration count),
so there was no way to trace afterwards "which settings produced this result". In practice, identifying the
geometry of an old ZIP meant measuring dimensions from the images and brute-forcing configs until volume and
`tau_max` matched to every digit.

### Added
- **`settings.json`** (included in the ZIP) — every field of the geometry config, material, injection conditions,
  wall cooling model, compression molding, output settings, and the version and commit at analysis time. Key names are
  the dataclass / solver argument names, not the UI labels (unaffected by UI wording changes, and map directly to
  CLI arguments)
- `core/settings_record.py` — `config_settings()` / `file_fingerprint()` /
  `settings_json()`
- A "Settings that produced this result" expander in the results pane
- `tests/test_settings_record.py` (10 tests)

### Implementation notes
- **An uploaded spec JSON is recorded by name and SHA-256 only; its contents are not included.**
  Gate profiles often come from real drawings, and this ZIP is designed to be handed to other people
  (which is exactly why `player.html` is bundled). With a fingerprint, whoever holds the spec can confirm
  "which version was run", while the dimensions do not travel with the ZIP. For the same reason, image input
  is recorded by file name and hash only
- The geometry config is copied whole with `dataclasses.asdict` rather than a hand-written list of keys.
  The tests also compare against the dataclass itself, so **adding a new geometry parameter and forgetting
  to record it** cannot happen
- The leak test checks that "the actual numbers do not appear in the serialized output". A test that checks
  for the presence of keys would let a future change that embeds the contents slip through
- **The spec's own `name` field is also "contents"**, so it is not included. The first version recorded it
  as `spec_name` (removed after review). Having set a policy of emitting only the file name and hash,
  the next line was copying one field of the contents. A test now pins the keys under `spec` to
  `name` / `sha256` / `bytes` only

---

## [0.17.0] — 2026-08-18

**The filling animation can now be read the same way as in commercial CAE.**

Switched the colormap to a rainbow family (turbo), overlaid isochrones, and smoothly interpolated only the colors.
The outline and the flow front remain cell-exact.

### Added
- **Isochrone overlay** — contour lines connecting positions filled at the same time, 12 by default
  (exactly N when N is given). Lines that
  crowd = slow flow, lines that collide = weld, beyond where they break off = filled last. The quantitative reading
  of this figure, independent of color vision and surviving black-and-white printing. 0–24 lines via a UI slider
- **Colormap selector** — turbo (default) / jet / viridis / cividis. Accessibility for red-green color vision is
  covered by choosing cividis / viridis
- **Colorbar on frame PNGs** — the player and `player.html` have no surrounding explanation, so
  each frame needs to carry its own legend
- `tests/test_fill_render.py` (24 tests)

### Changed
- **Default colormap for the filling animation changed from `viridis` to `turbo`.** What a molder reads from this figure
  is not absolute values but the shape of the isochrones, and the hue contrast of a rainbow makes those bands stand out
  (viridis, with monotonic lightness, flattens them instead). It also matches the semantics of red = filled last =
  risk area. `turbo` rather than `jet` because jet jumps in lightness at cyan and yellow and draws
  **false bands that are not in the data**, which get confused with real isochrones
- **Only the color layer is now interpolated `bilinear`.** Values outside the cavity are first extended with the nearest
  interior value via `_nearest_extend`, colors are built from that, and unfilled cells are punched out on top with an
  opaque `nearest` overlay. **Fill time is a continuous field, so interpolating colors is not a lie, but the cavity
  outline and the flow front are correct only as the mask says**, so interpolating with the mask in the alpha
  channel (= the front bleeds by half a cell, narrow channels get thinner) was avoided
- Consolidated the RGBA construction and gate marker drawing duplicated in `render_fill_animation` and `export_frames`
  into shared helpers

### Implementation notes
- **Isochrones are drawn once over the whole cavity and hidden under the unfilled overlay.**
  Redrawing them per frame would need `ContourSet.remove()`, which exists only in Matplotlib 3.8
  and later (this package's floor is 3.7). Dropping the redraw removed the version dependency
  and reduced contour computation from once per frame to once
- Gate markers sit above the overlay (`_Z_GATE`). A marker is larger than one cell, so on
  the first frame, or for a gate on the cavity boundary, it gets eaten by the opaque unfilled fill.
  **This is the default of `_draw_gate_markers`**, so callers cannot forget to pass it
- Isochrones are not drawn for a cavity one cell wide (a 1-pixel-wide image upload).
  `contour` requires 2x2, but the analysis itself is valid, so discarding a completed
  result for the sake of decoration is not worth it
- **`export_frames` also builds the figure only once, like the GIF.** Rebuilding it per
  frame means running contour once per PNG (the UI default of 60 frames, measured at 41 ms per call on
  the 500k-cell cavity the UI allows = about 2.5 s). The output PNGs are
  identical either way, so **there is no way to detect this other than counting contour calls**
- Stacking order is set explicitly with `zorder`. matplotlib draws `imshow` at 0 and `contour` at 2,
  so merely calling the overlay later leaves the isochrones on top and the lines show through
  the unfilled area (this actually happened. A test that checked call order passed it, and it was
  found only by looking at the PNG)

### Known trade-offs
- **The flow front looks stair-stepped.** That is the cell resolution itself; smoothing it would draw
  the front at positions that do not exist. To make it smoother, use a finer mesh
- **This smoothing must not be carried over to the wall thickness map.** Thickness is inherently discontinuous
  (no intermediate thickness actually exists between t0.35 and t0.50), and interpolating would make
  nonexistent thicknesses appear. The same principle as the flat-top decision in 3D. Thickness coloring is to be considered separately
- Rainbow colormaps are unfavorable for red-green color vision deficiency (about 8% of men). This is handled by
  keeping rainbow as the default while leaving cividis / viridis as options

---

## [0.16.0] — 2026-08-18

**Someone handed the downloaded ZIP can now use the same frame stepping as on screen, with no extra software.**

The ZIP now includes a standalone HTML player, `player.html`. Double-clicking opens it in the default
browser, with the same seeking, frame stepping and speed switching as in the app. Frames are
embedded as data URIs, so it is fully offline, and it makes no external requests and writes no storage
(HTML only runs inside the browser sandbox and cannot touch the registry or file-extension
associations).

### Added
- `fill_player.wrap_standalone_html()` — wraps the embeddable fragment into a complete HTML
  document. The key point is placing `<meta charset="utf-8">` before the first non-ASCII byte;
  without it, a browser opening the file via `file://` falls back to the platform's default
  legacy encoding (CP932 on Japanese Windows) and the button
  labels become garbled
- `player.html` is included in the ZIP. The page heading shows the version and commit from `build_label()`,
  so the recipient can also tell which build produced the result
- 7 tests added to `tests/test_fill_player.py` (charset position, document completeness, fragment
  identity, offline self-containment, title/note escaping, leftover placeholders)

### Known trade-offs
- **The ZIP roughly doubles in size.** base64 inflates to 4/3 of the original, and PNGs are already compressed so
  deflate does not help; measured, a 24-frame case went from 310 KB to 628 KB.
  Equivalent to adding one more full set of frame PNGs
- Corporate mail gateways may quarantine HTML attachments. Even then,
  the numbered PNGs in `frames/` remain, so frames can still be stepped through with the arrow keys of a standard viewer

---

## [0.15.0] — 2026-08-18

**The filling animation can now be paused and viewed at any position.**

Replaced the auto-playing GIF with an HTML5 player that can be seeked by dragging.
Frame PNGs were already being exported (for the ZIP), so rendering cost did not increase.

### Added
- `core/fill_player.py` — `build_fill_player_html()`. A self-contained HTML player with frame PNGs
  embedded as data URIs. Play / pause, drag seeking, frame stepping,
  speed switching (0.25–2x), loop toggle, keyboard control (Space / ← / →). All controls
  run entirely in the browser and do not go back to the server while seeking
- `visualizer.fill_frame_times()` / `fill_frame_fractions()` — the single source of frame times shared by the GIF,
  the numbered PNGs and the player
- `tests/test_fill_player.py` (8 tests)

### Changed
- The UI's "Flow front animation" shows the player instead of the GIF. The GIF is
  still included in the ZIP as before (download use is unchanged)
- Consolidated the frame-time `linspace` duplicated in `render_fill_animation` and `export_frames`
  into `fill_frame_times()`. A cleanup so that a third consumer, the player, does not
  carry its own copy of the same formula

### Known trade-offs
- The player runs inside the iframe of `st.components.v1.html`, so it does not inherit Streamlit's
  theme colors. It is built with neutral colors readable in both light and dark
- All frames are embedded as data URIs, so the HTML is about 1.7 MB at the default 60 frames

---

## [0.14.0] — 2026-08-07

**Gate geometry can now be brought in from outside via a JSON spec.**

A fourth geometry family that takes a depth-field model extracted from a drawing as JSON and rasterizes it into
the cavity geometry. New gate geometries can be analyzed without writing another builder in code
as before.

### Added
- `GateProfileSpec` (spec definition with JSON I/O) and `build_profile_gate_geometry`
  — a depth field composed of land / main ramp (with bottoming out) / shallow island / outer wall line / obround well /
  valve orifice. Supports both symmetric and one-sided modes
- A "Profile gate (JSON spec)" family in the Streamlit UI (demo preset / upload / paste)
- A demo spec with fictitious dimensions, `data/gate_profiles/demo_profile_gate.json`
- 28 tests (including closed-form volume checks and a radial-integration check of the well volume)

### Fixed
- A `TypeError` slipped past the UI's error handling when a JSON section value was not an object
  (unified into a `ValueError` with a path)
- When the gate geometry extended beyond the raster grid it was silently clipped, giving wrong volume and
  filling results (now explicitly rejected by up-front validation)

### Changed
- Made `well.floor_t_range` optional and documented that it is metadata for referencing the drawing
  (the actual floor shape is derived from the depth and wall angle)
- CI: excluded `docs/` from ruff (newer ruff started formatting code snippets inside
  Markdown)

---

## [0.13.0] — 2026-06-09

**The 3D view became lighter, and steps in wall thickness now look correctly like steps.**

### Changed
- Replaced the top and floor rendering from a full-grid `Surface` with a sparse `Mesh3d` spanning only
  cavity cells — greatly reducing the load of rotating and dragging
- Changed the top surface to be drawn as a set of "flat blocks of constant thickness". Step walls are raised at
  boundaries where thickness changes between adjacent cells, faithfully reproducing designed steps as vertical steps
  (previously they were smoothed into a one-cell-wide slope by interpolation at cell centers)
- Relaxed the minimum mesh coarseness from 0.5 to 0.2 mm/cell. Added an upper-limit guard on cell count at the same time
  (prevents memory exhaustion in the cloud environment)

### Removed
- The display-only upsampling feature and the display-only analytic refinement feature
  — both withdrawn because their effect did not justify the rendering cost

### Known trade-offs
- Because flat blocks are used, intentionally continuous tapers (runner slopes, etc.) look like
  fine stairs. Distinguishing them from designed steps is inherently difficult, so showing steps on the
  part surface accurately takes priority

---

## [0.12.0] — 2026-06-07

**The UI stays easy to navigate even as parameters grow.**

### Changed
- Fully reorganized the input UI into a collapsible hierarchy of major, middle and minor sections. On startup
  all major sections are collapsed, and only the needed parts are opened
- Fixed so that turning taper staging OFF gives a single taper that is exactly identical across the full width

### Deferred
- Accent color customization — withdrawn because it cannot coexist with the Light/Dark theme switch

---

## [0.11.0] — 2026-06-06

**Thickness-adjusting gates (right trapezoid) are now supported.**

### Added
- A third geometry family, "Film gate 2 (variable gate position)"
  — right-trapezoid film gate, variable gate position, two-stage taper, deep runner, and
  step specification at the land boundary

---

## [0.10.0] — 2026-06-05

**Explanatory documents that can be shared outside the company are in place.**

### Added
- Integrated five customer-facing explanatory documents on flow analysis into the repository
  (how the analysis works, differences from commercial CAE, program walkthrough, and more)

### Changed
- Corrected the description of the governing equation to match the implementation (diffusion equation → Poisson equation)
- Unified where the documents draw the line on the treatment of mold cooling analysis

---

## [0.9.0] — 2026-05-15

**Viscous heating can now be taken into account, and the full set of results can be taken out together.**

### Added
- A correction model for shear heating (viscous dissipation). The Brinkman number is always computed, and
  the degree of heating influence is shown with a traffic light
- One-click ZIP download of the animated GIF and all frame PNGs

### Changed
- Unified the compression molding (ICM) UI into stroke specification
- Tidied up the names of the wall cooling models and adjusted defaults for ultra-thin plates

### Fixed
- In the cloud environment the material database cache was not refreshed,
  causing inconsistencies after schema changes

---

## [0.8.0] — 2026-05-13

**A multilayer solver that resolves temperature and viscosity distributions through the thickness arrived. A generational change of the physics engine.**

A new solver that discretizes the thickness direction, previously treated as a single layer, into N layers and solves the temperature,
viscosity and shear rate of each layer in a coupled way. Cooling near the wall and flow in the core can be evaluated separately.

### Added
- `MultilayerHeleShawSolver` — per-layer temperature evaluation (1D heat conduction solution), per-layer viscosity,
  coupling by iterative convergence. Matches the previous solver exactly at N=1
- A layer distribution refined near the walls, short-shot detection based on the center layer temperature,
  and adaptive damping of convergence
- Visualization of per-layer profiles (temperature and viscosity grid view, short-shot map)

### Added (compression molding)
- A mode that specifies the compression amount as a stroke (absolute amount)
  — a stepped plate can be compressed while keeping its steps

---

## [0.7.0] — 2026-05-08

**Direct gates are now supported, and the theoretical basis can be explained on screen.**

### Added
- A second geometry family, "Direct gate" — a form that places the gate directly inside the part.
  Also supports splitting the plate into two layers
- An on-screen explanation of the governing equations and scope of applicability (with rendered formulas)
- Added three talc-filled polypropylene grades (PP-T10 / T20 / T30) to the material database

### Changed
- Adjusted the default of the injection rate slider to match the actual machine
- Unified on-screen wording into a style and terminology fit for external explanation

---

## [0.6.0] — 2026-05-08

**Results can now be viewed in 3D.**

### Added
- 3D visualization with Plotly (wall thickness map / fill time / pressure)
  — extrusion relative to the parting line, colored side walls, true-scale display

---

## [0.5.0] — 2026-05-06

**Became a public app anyone can try from a browser.**

### Added
- Deployment settings and a public URL on Streamlit Community Cloud
- Download of analysis results (GIF, etc.)

### Changed
- Reset the defaults for injection rate and compression molding to the actual machine's operating conditions
- Moved the Run analysis button to the header row

---

## [0.4.0] — 2026-05-06

**Flow balance design studies became possible, and the frozen layer at the wall was brought into the physics model.**

### Added
- Flow balancer (coring that intentionally thins the center to distribute melt outward).
  Supports nesting up to 5 levels
- Wall frozen layer (skin layer) model — evaluates by iterative convergence how the frozen layer growing from the wall
  narrows the core channel. Locations where the channel closes off are shown as short shots
- Splitting the plate into two thickness bands, gate side / far side

---

## [0.3.0] — 2026-05-05

**Real geometries with gates can now be entered.**

### Added
- The first parametric geometry family, "Film gate"
  — generates the plan shape of a trapezoidal runner + semicircular section + valve gate from dimensions
- Wired into both the Streamlit UI and the CLI demo

---

## [0.2.0] — 2026-05-05

**The correctness of analysis results can now be verified automatically.**

### Added
- A verification test comparing the analytical solution of 1D strip flow with the numerical solution
- CI on GitHub Actions (lint, formatting and tests on Python 3.11 / 3.12)

### Changed
- Applied ruff code formatting throughout

---

## [0.1.0] — 2026-05-05

**The first working simulator.**

### Added
- A fill-time field solver using the Hele-Shaw approximation + Cross-WLF viscosity model + pseudo-conduction method
- Filling animation, pressure map, weld line / air trap estimation
- Streamlit UI and CLI batch execution
- Released under the MIT License, packaging based on pyproject.toml

[0.14.0]: https://github.com/shostako/mold-flow-sim/releases/tag/v0.14.0
[0.13.0]: https://github.com/shostako/mold-flow-sim/releases/tag/v0.13.0
[0.12.0]: https://github.com/shostako/mold-flow-sim/releases/tag/v0.12.0
[0.11.0]: https://github.com/shostako/mold-flow-sim/releases/tag/v0.11.0
[0.10.0]: https://github.com/shostako/mold-flow-sim/releases/tag/v0.10.0
[0.9.0]: https://github.com/shostako/mold-flow-sim/releases/tag/v0.9.0
[0.8.0]: https://github.com/shostako/mold-flow-sim/releases/tag/v0.8.0
[0.7.0]: https://github.com/shostako/mold-flow-sim/releases/tag/v0.7.0
[0.6.0]: https://github.com/shostako/mold-flow-sim/releases/tag/v0.6.0
[0.5.0]: https://github.com/shostako/mold-flow-sim/releases/tag/v0.5.0
[0.4.0]: https://github.com/shostako/mold-flow-sim/releases/tag/v0.4.0
[0.3.0]: https://github.com/shostako/mold-flow-sim/releases/tag/v0.3.0
[0.2.0]: https://github.com/shostako/mold-flow-sim/releases/tag/v0.2.0
[0.1.0]: https://github.com/shostako/mold-flow-sim/releases/tag/v0.1.0

