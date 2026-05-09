# AMR-D Nested Debug Audit - 2026-05-09

## Scope

This audit freezes the current AMR-D nested-channel debugging state before any
new algorithmic patch. The target remains voie D: static conservative-tree AMR,
KRK-reproducible cases, and macro-flow closure before using the GPU path for
production benchmarks.

Current failing symptoms:

- nested Poiseuille/Couette profiles can remain far from the analytic profile;
- y-band / wall-yband / Couette nested4 profiles show step-like jumps near
  refined interfaces and walls;
- CPU and Metal do not give the same density cleanliness;
- long macro runs are too slow to be an efficient debug loop.

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

The KRK route matrix added after this audit shows that row-mean profiles are
not sufficient. On `poiseuille_xband_nested4_debug.krk`, forcing
`route_sampling=1` keeps the profile close to the same-time Cartesian reference
after 64 coarse steps, but the leaf field is already corrupted:

- `rho_min ~= 0.987`, `rho_max ~= 1.027`;
- `ux_min ~= -0.04`, `ux_max ~= 0.04`;
- profile `linf` versus Cartesian remains only `O(1e-5)`.

This means `route_sampling=:level_native` can produce compensating positive and
negative field errors that vanish in row averages. Any production gate must
inspect field extrema and field errors, not only averaged profiles.

Current decision: `route_sampling=:level_native` remains experimental. The
production nested-channel path should use `:leaf_equivalent` until the
field-level canary is fixed.

### F1 - Current KRK comparisons mix route modes

The current nested KRK files are not apples-to-apples:

- `poiseuille_xband_nested4_debug.krk` has no explicit `route_sampling`, so it
  uses the default `:leaf_equivalent`.
- `poiseuille_yband_nested4_debug.krk` sets `Define route_sampling = 1`, so it
  uses `:level_native`.
- `poiseuille_wall_ybands_nested4_debug.krk` sets `Define route_sampling = 1`.
- `couette_yband_nested4_debug.krk` sets `Define route_sampling = 1`.

Consequence: the observed x-band vs y-band difference may be a route-mode
difference, a direction bug, or both. This must be isolated before drawing
physical conclusions from the dashboards.

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

The route/physics mismatch is the leading suspect, but it is not proven yet.
Other plausible causes remain:

- C2F predictor timing may be wrong for wall-normal gradients;
- wall BC application may be applied at the wrong level or wrong substep;
- profile extraction may create visual stair steps near walls, although the CSV
  amplitude mismatch indicates a real physical error too;
- level-native corner closure may be conservative at rest but biased under
  forced non-equilibrium flow.

## Debug Ladder

No new long macro-flow campaign should run before these small tests isolate the
fault.

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

## Stop Conditions

Do not proceed to BFS, square obstacle, cylinder, or long GPU benchmarks while:

- nested channel analytic error remains large;
- route modes are mixed across comparison cases;
- max-level-4 non-equilibrium canaries are absent;
- CPU and Metal one-cycle packed paths are not compared.
