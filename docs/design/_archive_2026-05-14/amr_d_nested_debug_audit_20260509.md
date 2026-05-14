# AMR-D Nested Debug Audit - 2026-05-09

## Scope

This audit freezes the current AMR-D nested-channel debugging state before any
new algorithmic patch. The target remains voie D: static conservative-tree AMR,
KRK-reproducible cases, and macro-flow closure before using the GPU path for
production benchmarks.

Historical failing symptoms before the 2026-05-10 closure pass:

- nested Poiseuille/Couette profiles can remain far from the analytic profile;
- y-band / wall-yband / Couette nested4 profiles show step-like jumps near
  refined interfaces and walls;
- CPU and Metal do not give the same density cleanliness;
- long macro runs are too slow to be an efficient debug loop.

## Closure Update - 2026-05-10

The nested channel CPU/F64 dashboard is now closed for the four debug KRK cases:

```text
benchmarks/results/quicklook/amr_d_nested_closed_active_r2_tfinal_cpu_f64_12800_20260510
```

All four cases report `validation_status=validated`, `temporal_status=converged`,
and `tfinal_ok=1` at `12800` AMR steps (`204800` leaf-equivalent steps):

- `poiseuille_xband_nested4_debug`: active R2 `u` vs Cartesian `0.992761`,
  active R2 `u` vs analytic `0.992800`, active R2 `rho` vs both `0.999999997`.
- `poiseuille_yband_nested4_debug`: active R2 `u` vs Cartesian `0.999917`,
  active R2 `u` vs analytic `0.999916`, active R2 `rho` vs both `1.0`.
- `poiseuille_wall_ybands_nested4_debug`: active R2 `u` vs Cartesian
  `0.999959`, active R2 `u` vs analytic `0.999957`, active R2 `rho` vs both
  `1.0`.
- `couette_yband_nested4_debug`: active R2 `u` vs Cartesian `0.999999999991`,
  active R2 `u` vs analytic `0.999973`, active R2 `rho` vs both `1.0`.

The dashboard now exports both leaf-field R2 and active-cell R2. The validation
gate intentionally uses active-cell R2: the Cartesian and analytic references
are volume-averaged over each active AMR cell before comparison. This avoids
penalizing coarse cells as if they were resolved leaf cells while still keeping
the raw leaf-field R2 in `values.csv` for diagnostics.

The route configuration is case-specific:

- x-band uses `route_sampling=0` (`:leaf_equivalent`), which remains the best
  policy for vertical full-height refinement;
- y-band, wall-ybands, and Couette use `route_sampling=1`
  (`:level_native`) after the wall-corner F2C closure fix.

The lower-level wall-corner bug is closed by reflecting only the reflux
direction of level-native diagonal F2C packets that hit a physical y-wall at a
coarse/fine corner. The packet value still comes from the original outgoing
population. The canary that used to be broken now passes:

```text
test/test_conservative_tree_subcycling_2d.jl: 679 pass, 4 broken
```

## Frozen Evidence

Last AMR-D correctness commit used as baseline:

- `c9acb05 Fix AMR-D Float32 CPU diagnostics`

Correctness tests passed on this baseline:

- `test/test_conservative_tree_gpu_pack_2d.jl`: 4984 pass
- `test/test_amr_d_krk_validation_2d.jl`: 87 pass

Relevant result directories:

- CPU Aqua 12800-step sweep:
  `benchmarks/results/quicklook/amr_d_cpu_nested_channels_steady_sweep_12800_f32fix_aqua_20260509`
- CPU Aqua 25600-step single sweep:
  `benchmarks/results/quicklook/amr_d_cpu_nested_channels_steady_single_25600_f32fix_aqua_20260509`
- Metal 12800-step sweep:
  `benchmarks/results/quicklook/amr_d_metal_nested_channels_steady_sweep_12800_20260508`
- Metal 25600-step single sweep:
  `benchmarks/results/quicklook/amr_d_metal_nested_channels_steady_single_25600_20260508`

Observed high-level facts:

- x-band CPU and Metal are close in velocity error. The main profile error is
  therefore not only a Metal porting issue.
- y-band, wall-yband, and Couette nested cases still fail against the analytic
  profile on CPU. The nested profile bug is shared by CPU and GPU paths.
- CPU density is clean in y-band / wall-yband / Couette, with rho variations
  around `1e-6`. Metal has rho variations closer to `1e-4` and raw mass drift
  around `1e-5`. This is a separate GPU/mass-correction issue.
- CPU scalar AMR-D is too slow for long steady debug. The 12800/25600 step
  Aqua jobs took tens of minutes to more than one hour per case.

## Priority Findings

### F0 - Route 1 can hide field-level failure behind good profiles

Status: partially closed. The wall-corner rest canary is fixed and the final
12800 dashboard validates the selected nested channel cases. Route `1` is still
not a universal replacement for route `0`; x-band remains on route `0`.

The KRK route matrix added after this audit showed that row-mean profiles are
not sufficient. On `poiseuille_xband_nested4_debug.krk`, forcing
`route_sampling=1` keeps the profile close to the same-time Cartesian reference
after 64 coarse steps, but the leaf field is already corrupted:

- `rho_min ~= 0.987`, `rho_max ~= 1.027`;
- `ux_min ~= -0.04`, `ux_max ~= 0.04`;
- profile `linf` versus Cartesian remains only `O(1e-5)`.

This means `route_sampling=:level_native` can produce compensating positive and
negative field errors that vanish in row averages. Any production gate must
inspect field extrema and field errors, not only averaged profiles.

Current decision: field/profile gates must inspect active-cell and leaf-field
metrics separately. Route `1` is acceptable only where the active-cell R2
dashboard validates it; route `0` remains the x-band policy.

The failure was localized further: `:level_native` preserved a no-collision
rest state for an internal vertical x-band, but fails when the same vertical
coarse/fine interface touches the physical north/south walls. The defect is
therefore a vertical interface + wall-corner closure issue in diagonal D2Q9
populations. The corresponding canary is no longer broken.

### F1 - Current KRK comparisons mix route modes

Status: closed by explicit per-case route declarations and dashboard reporting.

The current nested KRK files are not apples-to-apples:

- `poiseuille_xband_nested4_debug.krk` sets `Define route_sampling = 0`, so it
  uses `:leaf_equivalent`.
- `poiseuille_yband_nested4_debug.krk` sets `Define route_sampling = 1`, so it
  uses `:level_native`.
- `poiseuille_wall_ybands_nested4_debug.krk` sets `Define route_sampling = 1`.
- `couette_yband_nested4_debug.krk` sets `Define route_sampling = 1`.

Consequence: the dashboard is no longer mixing implicit defaults. The route mode
is part of the explicit reproducibility contract in each KRK file.

### F2 - Likely mismatch: level-native routes with leaf-equivalent physics

The macro-flow runners map `route_sampling=:level_native` to
`interface_time_scaling=:level_native`, but the physical level setup still uses
leaf-equivalent omega and force scaling:

- route mode is resolved in
  `src/refinement/conservative_tree_krk_validation_2d.jl`;
- default predictor is `1.0` for `:level_native`, `0.5` otherwise;
- macro-flow runners use `interface_time_scaling=:level_native` when
  route sampling is level-native;
- Poiseuille/Couette still call
  `conservative_tree_leaf_equivalent_omega_2d` and
  `conservative_tree_leaf_equivalent_force_2d` for every level.

This may be conceptually inconsistent. Either level-native route sampling must
define a matching physical-time policy, or the production macro-flow path must
stay on the leaf-equivalent route policy until level-native is proved.

This is the top algorithmic suspect for nested4 y-band / wall-yband / Couette.

### F3 - Existing tests do not cover the failing regime

Current tests cover important pieces but not the actual failing state:

- max-level-2 non-equilibrium nested profile tests exist;
- max-level-4 rest / route conservation tests exist;
- macro-flow "level 1 to 4" tests check finite values, positive mass, and cell
  counts, but not analytic/profile accuracy;
- there is no max-level-3/4 non-equilibrium y-band or wall-yband analytic canary.

Consequence: the test suite can stay green while the nested macro-flow profiles
are physically wrong.

### F4 - The temporal convergence criterion is too weak

The temporal sweep can mark a case as converged when the solution stops moving,
even if the analytic error remains large. This is useful for detecting temporal
stagnation, but it is not a physical validation gate.

The convergence gate must include:

- error against analytic profile when available;
- error against same-time Cartesian reference;
- interface jump metric across each L/L+1 boundary;
- mass drift and rho range.

### F5 - Metal density drift is a second, separate bug

CPU y-band density is clean while Metal y-band density shows a larger vertical
rho structure. This points to a backend-specific issue in one of:

- GPU packed stream/collide;
- GPU route packet accumulation;
- GPU mass correction / reduction;
- Float32 reduction order and correction policy.

It should not be used as the primary explanation for the nested profile error,
because the profile error is already present on CPU.

### F6 - KRK ratio expansion changes the exact geometry

The `ratio = 16` helper expands into nested blocks with padding to maintain a
one-level-difference rule. The dashboard must therefore export the generated
level map and interface positions. A human-readable "y-band" in KRK is not
enough to know the exact active nested geometry.

## What Might Still Be Wrong

The route/physics mismatch is no longer the leading suspect for the validated
nested channel cases above. Remaining plausible issues are narrower:

- C2F predictor timing may be wrong for wall-normal gradients;
- wall BC application may be applied at the wrong level or wrong substep;
- profile extraction can still create visual stair steps if leaf-field values
  are interpreted as active-cell accuracy;
- level-native remains case-specific and must keep active-cell R2 gates before
  being promoted as a general route policy.

## Debug Ladder

This ladder is retained for future regressions. The 2026-05-10 closure pass
used it in the order wall-corner canary -> KRK canaries -> active-cell R2
dashboard.

### Step 1 - Route-mode matrix

Run the same small KRK/direct cases with both route policies:

- x-band: `route_sampling=0` and `route_sampling=1`
- y-band: `route_sampling=0` and `route_sampling=1`
- wall-ybands: `route_sampling=0` and `route_sampling=1`
- Couette y-band: `route_sampling=0` and `route_sampling=1`
- max levels: 2, 3, 4

Outputs:

- profile CSV;
- rho/ux/uy extrema;
- interface jump metric;
- same-time Cartesian error;
- analytic error when available.

Expected decision:

- if only `route_sampling=1` fails, level-native route policy is the target;
- if both fail only for wall-normal refinement, focus on y-interface / wall BC;
- if max-level-2 passes but max-level-3/4 fails, focus on recursive scheduling.

### Step 2 - Manufactured no-collision interface transport

Build a minimal two-level and four-level setup with manufactured populations:

- affine equilibrium in `y`;
- affine equilibrium in `x`;
- no collision;
- no forcing;
- one full coarse step.

Compare AMR-D against a dense leaf-equivalent oracle. This isolates route
transport, C2F/F2C ledgers, and recursive scheduling without BGK or BC noise.

### Step 3 - Force/tau consistency canary

Disable spatial transport or use a periodic uniform state. Apply one physical
coarse step with body force and level-dependent tau/force.

Compare each level's momentum increment against the dense leaf-equivalent
oracle. Run this for both route policies. This directly tests whether
`level_native` route timing and leaf-equivalent physical scaling are compatible.

### Step 4 - Wall BC canary

Use small Couette and Poiseuille wall cases:

- refined band away from walls;
- refined band touching one wall;
- two refined wall bands;
- max levels 2, 3, 4.

For each case, compare one-step and short-time results against a dense
Cartesian leaf-equivalent reference.

### Step 5 - Profile extraction canary

Export raw per-leaf values and reconstructed profile values for the same state.
The test must prove whether the stair-step near the walls is a plotting/rebinning
artifact or present in the leaf data itself.

### Step 6 - CPU vs Metal one-cycle equivalence

Only after CPU canaries pass:

- run CPU packed and Metal packed from the same initial `F`;
- disable mass correction first;
- compare one full subcycled coarse step;
- enable mass correction and repeat.

This isolates backend error from algorithmic error.

## Proposed Correction Strategy

Short-term safest production path:

1. Treat `route_sampling=:leaf_equivalent` as the reference policy for macro
   validations until `:level_native` passes the canaries above.
2. Keep `:level_native` available but experimental behind explicit KRK flags.
3. Add max-level-3/4 non-equilibrium canaries before touching macro dashboards.
4. Patch the scheduler/physics policy only after one surgical test fails in a
   reproducible way.
5. Re-run macro dashboards only after the surgical ladder is green.

If `:level_native` is kept as the final policy, it needs a documented invariant:

- exact physical time represented by every packet;
- matching force and omega scaling by level;
- predictor timing relative to parent post-collision state;
- conservative C2F/F2C closure for non-equilibrium populations, not only rest.

## Immediate Work Order

1. Add the route-mode matrix as surgical tests, not only benchmark scripts.
2. Add non-equilibrium max-level-3/4 y-band and wall-yband analytic canaries.
3. Decide from evidence whether the macro-flow default should be
   `:leaf_equivalent` or whether `:level_native` gets a full physical-time fix.
4. Fix CPU correctness first.
5. Re-run the KRK dashboards CPU-small.
6. Only then re-run Metal and debug the density/mass drift separately.

Implemented diagnostic tools:

- `test/test_conservative_tree_subcycling_2d.jl` now contains a short
  route-mode transient matrix against same-time Cartesian references.
- `benchmarks/amr_d_nested_route_matrix_2d.jl` writes per-route profile CSVs and
  a summary with profile, field, density, mass, and interface-jump metrics.

## 2026-05-10 Diagnostic Freeze

The smallest still-broken nested rest canary is the level-native vertical
interface touching the physical north/south wall:

```julia
_test_subcycled_rest_maxdiff_2d(wall_touch_x, :level_native)
```

The error is already present at max-level 1 and remains at max-level 4 with
the same leading magnitude, about `6.94e-3` on integrated diagonal
populations. The affected cells are the four coarse wall/interface corners and
their first refined neighbours. The drift swaps diagonal populations in pairs,
so this is not a collision, forcing, or macro-profile extraction issue.

Two hypotheses were rejected locally:

- dropping the C2F split when the same diagonal also touches the wall starves
  the refined corner cells;
- injecting C2F packets before child streaming makes the rest canary worse.

Keep `route_sampling=:level_native` experimental. The next correctness patch
should target the diagonal wall-corner reflux accounting itself, with a level-1
wall-touching x-band canary before any macro dashboard.

## 2026-05-10 R2/Tfinal Dashboard Freeze

The temporal dashboard runner now exports explicit `t_final_leaf_steps`,
AMR-vs-reference R2, and AMR-vs-analytic R2 for `ux` and `rho`. Local CPU
dashboards were written under:

```text
benchmarks/results/quicklook/amr_d_nested_r2_tfinal_cpu_f64_20260510
benchmarks/results/quicklook/amr_d_nested_r2_tfinal_cpu_f64_single12800_20260510
```

At 3200 coarse steps (`t_final_leaf_steps = 51200`), all four nested channel
dashboards have `tfinal_ok = 1` and density R2 near 1, but fail velocity R2:

- x-band Poiseuille is close but below the current 0.99 AMR-vs-Cartesian gate
  (`r2_ux_field_vs_reference = 0.973`).
- center y-band Poiseuille is under-driven (`r2_ux_field_vs_reference = -0.98`).
- wall-ybands Poiseuille is strongly wrong
  (`r2_ux_field_vs_reference = -3.97`).
- Couette writes at 3200 only with a loose diagnostic mass guard; its raw mass
  drift reaches `8.67e-6` and velocity R2 is poor.

At 12800 coarse steps (`t_final_leaf_steps = 204800`) with fixed-time R2
validation (`KRK_AMR_D_TEMP_REQUIRE_TEMPORAL=false`):

- x-band Poiseuille has good analytic shape but still fails AMR-vs-Cartesian
  R2 (`0.971`).
- center y-band Poiseuille remains stuck at about half the Cartesian/analytic
  velocity scale (`r2_ux_field_vs_reference = -1.25`).
- wall-ybands Poiseuille overshoots by about 5x
  (`ux_max = 1.38e-2` vs Cartesian `2.76e-3`,
  `r2_ux_field_vs_reference = -95.7`).
- Couette y-band does not complete at 12800 even with diagnostic mass guards up
  to `1e-3`; raw mass drift exceeds the guard before output.

A short y-band diagnostic shows `route_sampling=:level_native` improves the
center y-band transient relative to production `:leaf_equivalent`, but it does
not close the wall-touching cases because the level-native wall/interface
corner canary above remains broken. Do not claim nested channel validation
until a smaller wall/interface reflux canary explains these R2 failures.

## Stop Conditions

Do not proceed to BFS, square obstacle, cylinder, or long GPU benchmarks while:

- nested channel analytic error remains large;
- route modes are mixed across comparison cases;
- max-level-4 non-equilibrium canaries are absent;
- CPU and Metal one-cycle packed paths are not compared.
