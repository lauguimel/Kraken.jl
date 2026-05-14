# AMR-D level-native symptom audit, 2026-05-13

Scope: diagnostic freeze for the level-native AMR-D channel symptoms. This is
not a fix proposal and not a validation claim.

## Protocol snapshot

Reference code point:

- diagnostic worktree based on `0762cf8` (`Compact AMR-D F2C hot path`);
- route mode: `level_native` for all cases;
- steps: `51603` top-level steps for every case;
- cases: uniform R0 plus yband, xband internal, xband full-height/wall, and
  explicit H for R2/R4/R8/R16.

Result folder:

```text
benchmarks/results/quicklook/amr_d_symptom_dashboard_small_pregate_0762cf8_N16x12_common_steps51603_all_native_ladder_r0_r16_20260513/
```

Important caveat: the dashboard profile column still compares against the
steady analytic profile, not a transient Cartesian field at the same physical
time. Therefore profile errors are symptoms, not proof of a steady-state AMR
operator error. The density field is more directly diagnostic here.

## Observed signatures

### Density residuals

`max|rho - 1|` from `summary.csv`:

```text
uniform R0:                2.2e-16
yband R2/R4/R8/R16:        2.5e-13, 5.9e-13, 2.0e-12, 2.6e-12
xband internal R2..R16:    2.7e-7,  1.7e-6,  6.5e-6,  2.0e-5
xband wall R2..R16:        1.1e-2,  1.4e-2,  2.2e-2,  2.9e-2
xband H R2..R16:           2.6e-2,  2.8e-2,  2.7e-2,  3.3e-2
```

Interpretation:

- R0 and yband are clean in `level_native`.
- xband internal has a small but growing density defect.
- xband wall and H have order-percent density defects from R2 onward.
- The bug is not a global mass drift: reported mass drift remains roundoff.
- The bug is not simply "any refinement": yband does not show it.

### Profile/velocity residuals

Profile amplitude ratio `max(u_AMR) / max(u_analytic_steady)`:

```text
uniform R0:                0.0047
yband R2/R4/R8/R16:        1.0006, 1.0001, 1.0000, 1.0000
xband internal R2..R16:    0.991,  0.933,  0.775,  0.558
xband wall R2..R16:        0.968,  0.872,  0.623,  0.460
xband H R2..R16:           1.008,  0.942,  0.824, 10.807
```

Interpretation:

- yband is the control case: density and velocity amplitude stay consistent.
- xband internal has little density oscillation, but its R16 velocity profile is
  too low.
- xband wall also becomes too low in velocity at high level.
- H R16 is qualitatively different: velocity blows up, not just too low.
- This strongly suggests at least two symptoms:
  - a density/interface-corner pressure defect;
  - a high-level velocity/force/timing/profile defect, especially visible at R16.

### Spatial location of density extrema

From `fields_long.csv`:

- xband wall R16 max density: near wall (`j=1`), level 1.
- xband H R16 strongest negative density: interior point (`i=97, j=145`),
  level 2; strongest positive: interior point (`i=161, j=137`), level 1.
- xband internal R16 extrema are small (`~2e-5`) and do not form the same
  order-percent pressure pattern.
- yband R16 extrema are roundoff.

Interpretation:

- The wall is an amplifier for xband wall, but not the only possible trigger.
- H shows a large pressure pattern at internal corners, so the defect is more
  likely tied to level-transition corners/faces seen by the streamwise flow.
- The relevant geometric signature is likely: x-normal AMR interfaces and
  corner closures under x-directed forcing.

## Consequences

The previous wall-phase hypothesis was too narrow. It can explain a subset of
wall-touching paths, but it does not explain the H interior-corner pressure
signature, and it risks adding route-specific machinery that hides the real
operator defect.

The next audit should not start from a macro fix. It should isolate the
smallest level-native operator defect for:

- x-normal L/L+1 interfaces under x-flow;
- corner transitions in the H topology;
- wall-touching xband only as an amplifier case, not as the root assumption.

## Recommended next checks

1. Rebuild the dashboard reference so the profile panel compares AMR-D against a
   transient Cartesian reference at the same declared physical time. Do not use
   the steady analytic curve as the only reference for profile conclusions.

2. Add a compact "corner/interface pressure packet" canary:

   - level-native route mode;
   - no macro Poiseuille dashboard;
   - one or a few forced BGK steps;
   - compare leaf oracle versus AMR for x-normal interface crossings and corner
     closures;
   - include variants: internal xband, wall xband, and H internal corner.

3. Audit these code zones before any correction:

   - native corner reflux target logic;
   - inactive parent route accumulation;
   - C2F/F2C timing under level-native;
   - any route-family skip/override that treats wall and non-wall corners
     differently.

4. Keep algorithms homogeneous in every dashboard:

   - all cases in `level_native`, or all cases in `leaf_equivalent`;
   - never use yband in one mode and xband in another mode for symptom
     comparison.

Working hypothesis after this sweep:

```text
The dominant density defect is an x-normal level-transition/corner defect in
level-native routing. Physical walls amplify one manifestation, but H shows that
internal corners can trigger the same class of pressure oscillation. The R16
velocity/profile failure may be a second defect or a consequence of the same
corner operator becoming nonlinear after many high-level substeps.
```

## Coefficient and Moment Canaries

Two small transport-only audits were added to avoid using the macro dashboard
as proof:

```text
tmp/audit_amr_d_level_native_corner_coefficients.jl
tmp/audit_amr_d_level_native_macro_moment_canary.jl
```

Result folders:

```text
benchmarks/results/quicklook/amr_d_level_native_corner_coefficients_20260513/
benchmarks/results/quicklook/amr_d_level_native_macro_moment_canary_20260513/
```

### Unit-packet coefficient audit

The unit-packet audit compares a leaf Cartesian oracle against the current
AMR-D `level_native` subcycled transport for each `(src_id, q)` packet.

Important caveat: this is stricter than the macro symptom. It intentionally
finds raw packet redistributions that cancel for uniform/equilibrium fields.
For example, `yband_r2` has unit-packet route differences, but the macro moment
canary below is exactly clean for rest and uniform `u_x`.

Most diagnostic rows:

```text
xband_h_r2:
  src L1 (9,8)  q6 -> oracle L1 (11,10) q6 = 1
                         AMR L0 (5,5)   q6 = 1 via COALESCE_CORNER

  src L1 (11,10) q8 -> oracle L1 (9,8) q8 = 1
                         AMR L0 (5,5)  q8 = 1 via COALESCE_CORNER
```

Those rows are internal H corners. No wall is involved.

For `xband_wall_r2`, the analogous worst rows are wall-amplified:

```text
src L1 (13,2) q8 -> oracle L0 (6,1) q6 = 1
                    AMR misses it; route signature COALESCE_CORNER:dst6
```

So the wall case is not the root mechanism; it is a boundary variant of the
same native corner/coalesce problem.

### Macro moment audit

The macro moment audit initializes a leaf-integrated equilibrium field, then
compares one level-native AMR step to the leaf oracle after restriction. Modes:
rest, uniform `u_x`, and weak Poiseuille `u_x(y)`.

Key `max_abs_drho` values:

```text
rest:
  uniform_r1:          0
  yband_r2:            0
  xband_internal_r2:   0
  xband_wall_r2:       0          with |uy| defect 1.3889e-2 at wall corners
  xband_h_r2:          1.3889e-2 at internal level corners
  xband_wall_r4:       6.9444e-3
  xband_h_r4:          1.3889e-2

ux_uniform:
  yband_r2:            0
  xband_internal_r2:   0
  xband_wall_r2:       1.6667e-4 plus |uy| 1.3889e-2
  xband_h_r2:          1.3889e-2

ux_poiseuille:
  yband_r2:            0 in rho, max |dux| 8.10e-6
  xband_internal_r2:   1.33e-5
  xband_wall_r2:       3.13e-5 plus |uy| 1.3889e-2
  xband_h_r2:          1.389e-2
  xband_h_r4:          1.389e-2
```

This matches the dashboard hierarchy better than the raw unit-packet audit:

- `yband` is a clean control for density.
- internal xband has only a small transport/moment defect.
- wall xband has a wall-corner momentum defect and a smaller density defect.
- H has an order-percent density defect at internal level corners even at rest.

## Revised Mechanism

The smallest failing path now appears to be:

```text
active fine/child cell
  -> diagonal step enters a refined inactive parent region
  -> current level-native route treats this as immediate COALESCE_CORNER/F2C
  -> leaf/native oracle would keep the packet at the child/native level for the
     remaining substep(s), often exiting into another active fine cell
```

In other words, level-native transport is prematurely coalescing packets that
should transit through inactive parent rows at the same native level before the
next parent sync. This produces:

- internal H corner density errors without walls;
- wall xband variants when the remaining transit also hits a physical wall;
- increasing high-level sensitivity because deeper nested regions create more
  native substeps and more inactive-parent transit opportunities.

The code zones matching this mechanism are:

```text
src/refinement/conservative_tree_routes_2d.jl
  _route_sample_level_for_active_cell_2d
  _level_native_diagonal_corner_closure_route_specs_2d!

src/refinement/conservative_tree_subcycling_2d.jl
  _conservative_tree_inactive_parent_coalesce_route_spec_2d
  conservative_tree_subcycle_accumulate_inactive_parent_routes_F_2d!
  _conservative_tree_level_native_corner_reflux_dst_2d
```

The next minimal implementation should not be a wall scatter. It should first
add a route/operator canary for the H internal transit:

```text
xband_h_r2:
  src L1 (9,8)   q6 -> L1 (11,10) q6 == 1
  src L1 (11,10) q8 -> L1 (9,8)   q8 == 1
  corresponding L0 (5,5) deposits == 0
```

Only after that canary is green should the same mechanism be checked on the
wall-amplified xband rows.

## Implementation Update

The first fix targets only the active-child inactive-parent transit path. It is
not a wall scatter and it does not change bulk C2F/F2C routing.

Code path:

```text
src/refinement/conservative_tree_subcycling_2d.jl
  _conservative_tree_subcycle_accumulate_fine_to_coarse_packet_unchecked_2d!
  _conservative_tree_level_native_corner_child_transit_state_2d
```

For `interface_time_scaling=:level_native` and `COALESCE_CORNER`, an active
child source is redirected only when all of the following are true:

- the first native child tick lands in the same fallback coarse active cell
  that the route table classified as the coalesced destination;
- that fallback cell is a true coarse leaf, not an inactive refined parent;
- at least one native child tick remains before parent sync;
- the remaining native ticks land in an active child cell under a refined
  parent at the same parent level.

In that case the packet is inserted into the existing coarse-to-fine ledger for
the final parent at the final substep of the parent interval. Otherwise the
existing native corner reflux path is left unchanged.

Validated canaries:

```text
level-native inactive-parent diagonal transit canary:
  L1 (9,8)   q6 -> L1 (11,10) q6 = 1, L0 (5,5) q6 = 0
  L1 (11,10) q8 -> L1 (9,8)   q8 = 1, L0 (5,5) q8 = 0

level-native pre-stream rest gates:
  mini noop/BGK, xband noop/BGK, bulk noop/BGK <= epsilon

julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_subcycling_2d.jl")'
  814 pass, 13 broken, 0 failed
```

This closes the smallest internal H corner packet defect and removes the
previous `xband/level_native` transient bounds from the broken-test list. The
remaining `@test_broken` rows are still separate known operator defects and
should not be treated as closed by this patch.

Post-fix in-memory rerun of the macro moment audit, without overwriting the
pre-fix CSV folders:

```text
rest / ux_uniform:
  xband_wall_r2, xband_h_r2, xband_wall_r4, xband_h_r4:
    max rho/ux/uy residual = 0 to roundoff

ux_poiseuille:
  xband_wall_r2: max rho 1.389e-5, ux 1.389e-5, uy 1.157e-5
  xband_h_r2:    max rho 2.899e-6, ux 2.896e-6, uy 1.160e-6
  xband_wall_r4: max rho 1.313e-5, ux 1.154e-5, uy 1.074e-5
  xband_h_r4:    max rho 2.899e-6, ux 2.896e-6, uy 1.160e-6
```

So the order-percent rest/equilibrium H and wall-corner signature is closed.
The weak Poiseuille residual is now five to six orders smaller than the
original H rest defect and should be audited separately before any claim about
long macro-flow convergence.

## Dashboard Rerun Attempt

A post-fix dashboard runner was added:

```text
tmp/run_amr_d_level_native_symptom_dashboard_postfix.jl
```

It reconstructs the exact pre-fix dashboard specs from:

```text
benchmarks/results/quicklook/amr_d_symptom_dashboard_small_pregate_0762cf8_N16x12_common_steps51603_all_native_ladder_r0_r16_20260513/fields_long.csv
```

and writes new `summary.csv`, `profiles_long.csv`, `fields_long.csv`, and a
combined `mesh | ux | rho | profile` dashboard with AMR-D, transient Cartesian,
and steady analytic profile curves.

Two blocking observations came out before the full 51603-step sweep:

```text
coarse_to_fine_predictor_weight = 1:
  xband_wall_r4 rest/noop on the reconstructed pre-fix spec loses mass
  maxdiff = 1.7361111111111119e-3
  drift   = -2.7777777777771462e-2

coarse_to_fine_predictor_weight = 0:
  rest/noop is conservative, but the macro path enters
  _stream_conservative_tree_level_native_wall_phase_transport_F_2d!
  and is too slow for local R16 dashboard sweeps.
```

A 4-step smoke completed and confirmed that reconstructed active counts match
the pre-fix dashboard. Attempts at 51603 steps and at 3200 steps were stopped
on local CPU because R16 remained in the wall-phase/phase-direct preparation
path for minutes without completing. The full dashboard should therefore not be
used as the next local gate. The next actionable gate is the predictor-1
rest/noop mass canary on the reconstructed `xband_wall_r4` spec, followed by
removing or bypassing the per-step wall-phase preparation before any long macro
sweep.

## Predictor-1 Pre-Stream Canary

The reconstructed `xband_wall_r4` spec is now frozen directly in
`test/test_conservative_tree_subcycling_2d.jl`:

```text
Nx=16, Ny=12
X1: 6:11, 1:12
X2: 13:20, 1:24, parent=X1
active cells = 984
```

The predictor-1 pre-stream rest gate remains broken and is intentionally kept
as `@test_broken`:

```text
noop, coarse_to_fine_predictor_weight=1:
  maxdiff = 1.7361111111111119e-3

BGK omega=1, coarse_to_fine_predictor_weight=1:
  maxdiff = 2.1010838694719425e-3
```

A sharper unit-packet canary identifies the local mechanism:

```text
source: L1 (11,1), q9
oracle leaf trace:
  -> L0 (5,1),  q7 = 0.75
  -> L1 (11,2), q7 = 0.25

current predictor-1/noop path:
  -> L0 (5,1),  q7 = 0.5
  -> L1 (11,2), q7 = 0.125
  active mass sum = 0.625
```

So the remaining predictor-1 defect is not a scalar mass correction and not a
dashboard artifact. It is a subcell phase-memory loss: after a diagonal packet
touches the physical wall, the level-native phase-direct path collapses the
subcell distribution back into the active L1 cell before the next native
advance. A one-line switch that disables phase-direct under `pre_stream_level!`
was tested and rejected: it falls back to the generic route path and makes the
rest residual `4x` larger (`6.9444444444444475e-3`) while also breaking macro
route-matrix bounds.

Current validation after freezing the canary:

```text
julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_subcycling_2d.jl")'
  820 pass, 18 broken, 0 failed
```

The next fix should therefore target phase memory or a source-keyed replacement
for these wall/interface diagonal packets. It should not be another rho-level
post-correction, and it should not be promoted to macro dashboards until this
unit packet canary is green.

## Proposed Fix Direction

The bug is narrower than the previous full WallPhaseScatter2D direction. The
current level-native phase path has two separate inconsistencies:

1. `route_sampling=:level_native` is only accepted with
   `coarse_to_fine_prolongation=:flat`, but both
   `conservative_tree_subcycle_sync_down_level_native_phase_routes_F_2d!` and
   `_stream_conservative_tree_level_native_phase_direct_level_routes_F_2d!`
   still call `_conservative_tree_limited_linear_child_packet_2d`
   unconditionally. This is the same wall-adjacent C2F dipole mechanism already
   tracked as a broken canary.
2. `_stream_conservative_tree_level_native_phase_direct_level_routes_F_2d!`
   traces `ratio` child ticks but only deposits packets whose final owner is
   the same active level. Packets that touch the wall and finish on a coarser
   owner are ignored by phase-direct and fall back to generic F2C/coalesce
   routes, which do not know the reflected subcell phase.

For the locked source:

```text
source L1 (11,1), q9, ratio=2

subcell (1,1): after tick 2 -> L0 (5,1), q7
subcell (2,1): after tick 2 -> L1 (11,1), q7; later -> L0
subcell (1,2): after tick 2 -> L1 (11,1), q7; later -> L0
subcell (2,2): after tick 2 -> L1 (12,1), q7; later -> L1 (11,2)
```

The old path collapses this evolving subcell state back into active L1/F2C
routes, so it loses the distinction between "already coarser after the native
interval" and "still same-level but reflected".

The appropriate fix is a bounded `PhaseNativeAdvance2D` replacement, not a
global wall scatter:

```text
At advance(level=L), when phase_direct_active:
  for each active source (src_id, q) at L:
    split into ratio^2 child subcells with flat weight F[src,q]/ratio^2
    advance exactly ratio child ticks with wall-y/periodic-x reflection
    if the path enters a refined child before the end:
      skip here; sync_down phase C2F owns it
    else if final owner level == L:
      Fscratch[dst_id, qcur] += packet
    else if final owner level < L:
      state_bank.levels[dst_level+1].reflux_to_coarse[dst_id, qcur] += packet
    else:
      fail/guard, because finer arrivals should have been sync_down-owned
```

At `sync_down(parent=L)`, keep the existing phase C2F idea but make its child
packet builder obey the actual `:flat` contract. That means replacing the
unconditional limited-linear packet by `F[src,q] / ratio^2` for level-native
flat mode. Do not enable limited-linear level-native until a separate canary
closes the wall-adjacent slope dipole.

During `phase_direct_active`, the generic active F2C accumulation must be
skipped for the same active sources handled by `PhaseNativeAdvance2D`; otherwise
coarser deposits are double-counted. Inactive-parent routes should remain on
their existing path unless a separate inactive-source canary fails.

This fix is source/route-local:

- no rho correction;
- no post-step mass redistribution;
- no full `1 << max_level` WallPhaseScatter table;
- no change to bulk interfaces that are already exact;
- GPU-compatible shape: the precomputed route pack can later store
  `(event_level, src_id, src_q, dst_kind, dst_id, dst_q, weight)` rows.

Minimum gates before macro:

```text
1. current @test_broken predictor-1/noop rest xband_wall_r4 -> @test
2. current @test_broken L1 (11,1), q9 packet:
   L0 (5,1), q7 and L1 (11,2), q7 plus sum == 1
3. wall-adjacent C2F dipole broken canary either stays unchanged and documented
   or is closed by the same flat-packet change
4. existing bulk interface packet canary remains max_err = 0
5. only then rerun small level-native dashboards
```

## Implemented Closure

The implemented closure is source-keyed, not a rho correction:

```text
stream_conservative_tree_subcycled_buffered_routes_F_2d!
  pre_stream_level! !== nothing
  route_sampling = :level_native
  coarse_to_fine_predictor_weight in {0, 1}
  wall_phase_transport_correction = true
```

For `predictor=0`, the existing wall-phase transport path is preserved whenever
there are wall-touching sources. For `predictor=1`, the path is now enabled only
when the precomputed scatter has a non-empty wall-then-interface event mask.
This prevents the yband and wall-yband macro cases from being routed through
the wall-phase replacement when their wall/interface cone is empty.

The predictor-1 gates promoted to `@test` are:

```text
xband_wall_r4 predictor=1 noop rest:
  maxdiff = 0

xband_wall_r4 predictor=1 BGK omega=1 rest:
  maxdiff <= epsilon

source L1 (11,1), q9:
  L0 (5,1),  q7 = 0.75
  L1 (11,2), q7 = 0.25
  active sum = 1
```

Validation:

```text
julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_subcycling_2d.jl")'
  825 pass, 13 broken, 0 failed

julia --project=. tmp/audit_amr_d_bulk_interface_packet_canary.jl
  one_level_bulk_vertical_interface diagonal packet max_err = 0
  two_level_bulk_vertical_interface diagonal packet max_err = 0
```

The wall-phase scatter rows are now cached on the subcycle route bank, so macro
loops that reuse `route_bank` no longer rebuild the dense source-keyed table at
every time step. This is still a CPU-side cached table, not the final GPU route
pack shape; before publication-scale GPU claims, these rows should be lowered
into the same compact packet family as the other precomputed ledgers.

The high-level macroflow default for `route_sampling=:level_native` remains
`coarse_to_fine_predictor_weight=1`, because the existing CPU macro gates rely
on the phase-direct predictor. The GPU scheduler still lacks the corresponding
phase-direct predictor path; the CPU-backend GPU pack test therefore remains
the next portability gap rather than a closed validation.

Smoke after adding the cache:

```text
AMRD_SYMPTOM_STEPS=64
AMRD_SYMPTOM_C2F_PREDICTOR_WEIGHT=1
cases: xband_wall R4 and R16

R4:  max_abs_rho_dev = 2.72e-8, ux_max = 9.26e-8
R16: max_abs_rho_dev = 1.69e-7, ux_max = 3.70e-7
```

The smoke is intentionally short and is not a steady-flow validation. Its only
purpose is to confirm that the cached route-bank path runs through the R16
macro plumbing without the previous per-step table-preparation stall.

## Post-Dashboard Correction Check

The cached wall-phase closure initially hid the density defect but also killed
the streamwise flux in full-height `xband_wall` runs. The failure was traced to
the `pre_stream_level!` early-return path:

```text
Fpost_run = apply pre_stream_level! over the level-native schedule
stream normal routes from Fin
scatter wall-phase rows from Fpost_run
```

That mixes two time states in the same transport. The minimal plumbing fix is
to stream the unmasked route families from `Fpost_run` as well, while keeping
`pre_stream_level! = nothing` inside the replacement call so the hook is not
applied twice.

Targeted dashboard after this one-line change:

```text
benchmarks/results/quicklook/amr_d_symptom_dashboard_postfix_fpost_routes_pred1_wallphase_on_steps2048_20260513/
```

`xband_wall`, level-native, predictor 1, wall-phase correction on:

```text
R2:  rho dev 1.88e-6, ux_max 3.61e-5, cart 4.31e-5
R4:  rho dev 6.78e-6, ux_max 8.77e-5, cart 1.72e-4
R8:  rho dev 2.80e-5, ux_max 2.21e-4, cart 6.53e-4
R16: rho dev 7.57e-5, ux_max 5.11e-4, cart 2.10e-3
```

This confirms that the previous `ux ~= 0` symptom was a real plumbing bug in
the wall-phase correction path. It is not the whole AMR-D bug: the high-level
`xband_wall` profile is still too weak by about a factor four at R16, and H
keeps a smaller residual:

```text
H R16: rho dev 3.54e-6, ux_max 2.07e-3, cart 2.10e-3
```

A new short forced-momentum canary freezes this distinction:

```text
tmp/audit_amr_d_level_native_forced_momentum_canary.jl
benchmarks/results/quicklook/amr_d_level_native_forced_momentum_canary_20260513/
```

It runs `xband_wall` and `H` in level-native predictor-1 mode, with wall-phase
correction both on and off, and writes:

```text
summary.csv
field_moment_diffs_top.csv
```

At 16 and 128 top-level steps up to R16, correction on/off are still
indistinguishable and classified `ok` with `ux_max / ux_cart ~= 1`. Therefore
the remaining R16 dashboard defect is a long-time/high-level accumulation, not
an instantaneous forced-BGK moment error. The next canary must target this
accumulation mechanism directly rather than adding another rho closure.

Validation after the plumbing change and forced-flux guard:

```text
julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_subcycling_2d.jl")'
  827 pass, 14 broken, 0 failed

julia --project=. tmp/audit_amr_d_bulk_interface_packet_canary.jl
  one_level_bulk_vertical_interface diagonal packet max_err = 0
  two_level_bulk_vertical_interface diagonal packet max_err = 0
```

## Versioned Dashboard Clarification

The quicklook dashboard runner now writes versioned folders and embeds the run
parameters in both the image header and `manifest.txt`.

Important correction: an intermediate dashboard accidentally used
`conservative_tree_wall_closed_xband_refine_blocks_2d` for both `xband_wall`
and `xband_h`. That means it generated two H/capped-xband variants, not the
simple full-height xband. The runner now uses the KRK lowering for simple
bands:

```text
yband_center:   Refine yband { region = [0, 3, 16, 9],  ratio = R }
xband_internal: Refine xband { region = [6, 3, 10, 9],  ratio = R }
xband_wall:     Refine xband { region = [6, 0, 10, 12], ratio = R }
xband_h:        conservative_tree_wall_closed_xband_refine_blocks_2d(...)
```

Corrected artefacts:

```text
v14: benchmarks/results/quicklook/amr_d_v14_symptom_pred1p0_wallphaseon_steps2048/
v16: benchmarks/results/quicklook/amr_d_v16_signed_residuals/
```

`v14` active counts confirm the geometry split:

```text
xband_wall R2/R4/R8/R16: 336, 984, 3432, 12936
xband_h    R2/R4/R8/R16: 648, 2088, 6936, 24360
```

`v14` reproduces the bad simple xband signature:

```text
xband_wall R16: rho dev 7.57e-5, ux_max 5.11e-4, cart 2.10e-3
xband_h    R16: rho dev 3.54e-6, ux_max 2.07e-3, cart 2.10e-3
```

The R16 Cartesian reference is also still transient in this run:
`cart_ux_max / analytic_ux_max = 0.761`. Therefore the analytic curve is a
stationarity indicator, not the primary bug reference for this dashboard. The
useful bug ratio is `ux_amr / ux_cart` at the same discrete physical time:

```text
xband_wall R2/R4/R8/R16: 0.836, 0.509, 0.338, 0.243
xband_h    R2/R4/R8/R16: 0.995, 0.988, 0.982, 0.984
```

This is the strongest symptom in `v14`: simple full-height `xband_wall` is
braked by the x-normal interfaces, and the braking grows monotonically with the
maximum refinement level. The H/capped geometry does not show the same velocity
loss in this corrected dashboard.

`v16` signed residual summary and visual inspection make the simple xband
density signature explicit. The physical xband pattern is:

```text
corner mur amont, interface est, and bulk est: drho < 0
corner mur aval, interface ouest, and bulk ouest: drho > 0
```

The scripted labels `upstream_west/downstream_east` are only coordinate bins;
they must not be read as the physical sign statement without checking the
dashboard. The robust observation is the antisymmetric xband pressure dipole
coupled to a streamwise velocity deficit. The velocity residual is not local:
`bulk_L0:d16:interior_y` still has `max |dux| = 1.66e-3`, which is visible
contamination of the non-refined bulk. This supports the current working
hypothesis: the remaining simple-xband defect is an x-normal interface transport
defect that accumulates with level depth, not a pure wall bounce-back correction.

Derived CSV:

```text
benchmarks/results/quicklook/amr_d_v16_signed_residuals/velocity_reference_ratios.csv
```

## Symptom Audit v17

Artifact:

```text
benchmarks/results/quicklook/amr_d_v17_symptom_audit/
```

This audit reads the corrected `v14` fields only; it does not rerun the
solver. It writes:

```text
velocity_reference_ratios.csv
physical_region_summary.csv
x_column_summary.csv
```

The `xband_wall` velocity loss is not a stationarity artifact. The Cartesian
R16 run is transient (`cart/analytic = 0.761`), but `xband_wall` is far below
the Cartesian reference at the same discrete time:

```text
xband_wall AMR/cart R2/R4/R8/R16: 0.836, 0.509, 0.338, 0.243
xband_h    AMR/cart R2/R4/R8/R16: 0.995, 0.988, 0.982, 0.984
```

So the dominant symptom is a streamwise momentum sink in the simple full-height
xband. It scales monotonically with refinement depth. The H/capped geometry is
a strong negative control: it has many interfaces but keeps `AMR/cart ~= 0.98`.
The distinguishing geometry is that `xband_h` refines the wall caps along the
whole periodic direction, while simple `xband_wall` has x-normal interfaces
that terminate directly at the physical north/south walls.

On R16, the density residual is antisymmetric and much smaller than the
velocity residual:

```text
bulk east far: mean drho ~= -7.80e-6, mean dux ~= -1.275e-3, uxratio ~= 0.221
bulk west far: mean drho ~= +7.81e-6, mean dux ~= -1.275e-3, uxratio ~= 0.221

xface east interior L2: mean drho ~= -2.94e-5, mean dux ~= -1.269e-3
xface west interior L2: mean drho ~= +2.94e-5, mean dux ~= -1.269e-3

xface wall-corner L2 west: mean drho ~= -6.21e-5, mean dux ~= -2.95e-4
xface wall-corner L2 east: mean drho ~= +6.25e-5, mean dux ~= -2.95e-4
```

The signs above are coordinate bins; the physical dashboard reading remains:

```text
corner mur amont, interface est, bulk est: drho < 0
corner mur aval, interface ouest, bulk ouest: drho > 0
```

The robust separation is:

- `rho` is a local signed dipole tied to x-normal interfaces and wall-corner
  closures.
- `ux` is a symmetric negative deficit that contaminates both east and west
  bulk regions.
- Mass correction is not hiding a mass leak here; the visible defect is a
  momentum/transport defect.

Relevant code path:

```text
src/refinement/conservative_tree_subcycling_2d.jl
  _stream_conservative_tree_level_native_wall_phase_transport_F_2d!
  _conservative_tree_apply_prestream_over_schedule_2d!
  conservative_tree_subcycle_sync_down_level_native_phase_routes_F_2d!
  _stream_conservative_tree_level_native_phase_direct_level_routes_F_2d!
```

In the macro path with `pre_stream_level! !== nothing` and predictor weight 1,
the current wall-phase branch computes a fully pre-streamed `Fpost_run` over
the schedule, then re-enters the transport path with
`coarse_to_fine_predictor_weight=0` and applies the wall-phase scatter from
`Fpost_run`. This was enough to preserve short forced flux and the H geometry,
but it is not a faithful event-local collision/transport sequence at the
simple xband wall-interface corners.

Working hypothesis after v17:

```text
The remaining bug is not a rho closure and not a generic bulk interface error.
It is a level-native event-timing/momentum defect at x-normal interfaces that
terminate on physical walls. The local wall-corner pressure dipole is the
visible source, and the repeated subcycled transport turns it into a global
streamwise momentum sink whose magnitude grows with max_level.
```

Next minimal canary should target momentum before any Poiseuille dashboard:

```text
uniform moving-equilibrium transport-only canary
  geometry: simple xband_wall R2/R4/R8, plus H as negative control
  init: rho = 1, uniform ux = small constant, uy = 0
  step: one level-native transport step, periodic_x_wall_y, no body force
  assert:
    total Jx_out / Jx_in == 1 within roundoff for H
    identify xband_wall loss by event family and by wall-corner source bins
```

If that canary loses `Jx`, the fix belongs in route/ledger transport at the
wall-terminating xface, not in macro forcing. If it preserves `Jx`, the next
canary is the same geometry with one BGK+force step and event-level momentum
budgeting (`force impulse`, `transport delta Jx`, `collision delta Jx`).

## Momentum Transport Canary v19

Artifact:

```text
benchmarks/results/quicklook/amr_d_v19_momentum_transport_canary/
```

Script:

```text
tmp/audit_amr_d_level_native_momentum_transport_canary.jl
```

This canary compares one level-native AMR transport step against a leaf
Cartesian oracle at the same physical time. It runs `rest`, `ux_uniform`, and
`ux_poiseuille` modes on uniform, simple `xband_wall`, and H/capped xband.

Result:

```text
ux_uniform:
  xband_wall R2/R4/R8: Jx_amr / Jx_oracle = 1.0 within roundoff
  xband_h    R2/R4/R8: Jx_amr / Jx_oracle = 1.0 within roundoff

ux_poiseuille:
  xband_wall R2/R4/R8: Jx_amr / Jx_oracle = 1.0 within roundoff
  xband_h    R2/R4/R8: Jx_amr / Jx_oracle = 1.0 within roundoff
```

The uniform-mode wall bounce reduces `Jx` relative to the initial state
(`Jx_oracle/Jx_in = 0.9444`) because the initial uniform flow is not a no-slip
channel equilibrium. AMR matches that leaf-oracle loss exactly. Therefore the
macro dashboard velocity collapse is not explained by a one-step pure transport
loss of total streamwise momentum.

There are local `ux` redistribution errors for nonuniform Poiseuille-like
initial data:

```text
xband_wall R8 ux_poiseuille: max |dux| ~= 2.17e-6
xband_h    R8 ux_poiseuille: max |dux| ~= 2.89e-7
```

So pure transport still has local interface/profile defects, stronger for
simple `xband_wall` than for H, but the signed total `Jx` budget closes. The
remaining macro failure must involve the collision/forcing/restriction timing
or repeated accumulation of these local redistribution errors, not a single
transport-step global momentum leak.

Updated next gate:

```text
one-step BGK+force momentum-budget canary
  same specs: xband_wall R2/R4/R8 and H negative control
  compare AMR to leaf oracle at same physical time
  export per-step budgets:
    Jx before collision
    Jx after collision/force
    Jx after transport
    Jx after restriction/sync
  classify residual by level/interface/wall-corner destination bins
```

## BGK+Force Budget Canary v22-v24

Artifact:

```text
benchmarks/results/quicklook/amr_d_v22_bgk_force_budget_canary/
benchmarks/results/quicklook/amr_d_v23_bgk_force_budget_canary/
benchmarks/results/quicklook/amr_d_v24_bgk_force_budget_canary/
```

Script:

```text
tmp/audit_amr_d_level_native_bgk_force_budget_canary.jl
```

This canary starts from rest, applies one Guo-forced BGK top-level AMR step,
and compares against a leaf Cartesian oracle with `1 << max_level` Guo+stream
ticks. It also writes `geom_summary.csv` and `population_summary.csv`.

Configuration v22 is the current wall-phase full-step correction path
(`wall_phase_transport_correction=true`). It removes the large local density
defect on `xband_wall`, but it loses streamwise momentum after the force has
been applied:

```text
xband_wall R2 : Jx_amr / Jx_oracle = 0.980769
xband_wall R4 : Jx_amr / Jx_oracle = 0.969377
xband_wall R8 : Jx_amr / Jx_oracle = 0.961870
xband_wall R16: Jx_amr / Jx_oracle = 0.956689

xband_h R2/R4/R8/R16: Jx_amr / Jx_oracle = 1.0 within roundoff
```

So the slowdown seen on the dashboard is already present after one forced step.
The negative control is important: the Guo collision and the generic AMR
interface machinery can close the global `Jx` budget on H, but not on the
simple full-height xband whose x-normal interfaces terminate at the physical
walls.

Configuration v23 disables the wall-phase correction. The global `Jx` loss is
smaller, but the local wall/interface defect returns:

```text
xband_wall R16 wall_phase=false:
  Jx_amr / Jx_oracle = 0.971698
  max |drho| ~= 2.41e-2
  max |dux|  ~= 4.58e-3
```

Therefore the current full-step correction is not the root bug. It masks the
large local wall-phase defect, but it also changes the forced momentum budget.

The population split in v22 shows the force-mode signature. For `xband_wall`
R16:

```text
q2 + q4 axial contribution      ~= +2.38e-6
q6 + q7 + q8 + q9 diagonal      ~= -1.55e-5
net Jx error                    ~= -1.31e-5
```

For H R16 the same axial/diagonal pattern cancels to roundoff:

```text
q2 + q4 axial contribution      ~= +1.70e-6
q6 + q7 + q8 + q9 diagonal      ~= -1.70e-6
net Jx error                    ~= 0
```

The bug is therefore population-level and force-mode specific. A pure
transport step of an equilibrium-like moving state conserves total `Jx`, but
transport of the Guo source populations does not balance on the full-height
xband.

Configuration v24 briefly enabled the existing WIP substep scatter table after
fixing two obvious hazards (no second `pre_stream_level!` read at sync-down when
the predictor is already postcollision, and masking `phase_direct` sources).
The result was invalid and has not been left active:

```text
xband_wall R4 : Jx_amr / Jx_oracle ~= 7.34e2
xband_wall R8 : Jx_amr / Jx_oracle ~= 3.57e2
xband_wall R16: Jx_amr / Jx_oracle ~= 1.86e2
xband_h remains exact
```

This falsifies the current substep table implementation, not the idea of
substep timing. The WIP table still over-injects xband wall-phase sources.

Current conclusion:

```text
Not a bulk interface bug.
Not a pure transport global momentum leak.
Not a raw Guo force omission.
Not fixed by the full-step wall-phase correction.

The unresolved defect is a forced diagonal population budget at x-normal
interfaces that terminate on physical y-walls. The full-step correction should
not be treated as the final macro fix because it transports force increments
with the wrong time support.
```

Next minimal canary:

```text
force-population packet canary
  geometry: xband_wall and H, R2/R4/R8/R16
  input: one D2Q9 Guo force source vector at rest, decomposed by q
  compare: AMR route transport vs leaf oracle, population by population
  assert:
    H axial/diagonal Jx cancellation closes to roundoff
    xband_wall reports the q6/q7/q8/q9 excess explicitly
```

Only after that packet canary should the substep table be re-enabled. The
required fix is an event-keyed population route correction whose rows conserve
each source/event before touching macro Poiseuille again.

## Force-Population Packet Canary v26

Artifact:

```text
benchmarks/results/quicklook/amr_d_v26_force_population_packet_canary/
```

Script:

```text
tmp/audit_amr_d_level_native_force_population_packet_canary.jl
```

This canary transports one static Guo source packet. It does not run collision
during the AMR step. The source packet is generated on the leaf grid at rest,
restricted to the AMR tree, transported with the level-native route machinery,
and compared with leaf transport of the same packet.

Result:

```text
xband_h R2/R4/R8/R16: exact to roundoff

xband_wall R2 : dJx = 0
xband_wall R4 : dJx = +8.33e-9
xband_wall R8 : dJx = +1.25e-8
xband_wall R16: dJx = +1.46e-8
```

The static force packet does not reproduce the v22 macro loss:

```text
v22 xband_wall R16 one forced step: dJx ~= -1.31e-5
v26 xband_wall R16 static packet : dJx ~= +1.46e-8
```

So the main slowdown is not a bad spatial coefficient for a single Guo source
packet. It is the time support of the force/collision source relative to
subcycled transport: the full-step wall-phase correction transports force
increments as if they existed for the whole top-level step, while the leaf
oracle creates and transports them tick by tick.

Revised next fix target:

```text
do not promote the full-step wall-phase correction as a macro fix
debug the event-keyed scatter on a conservation-only packet first
then feed it the Guo force packet at the event where it is created
```

The first implementation gate for that event-keyed path is not Poiseuille. It
is a table audit:

```text
for every (event_kind, event_tick, level, src_id, q):
  sum outgoing scatter weights == 1.0 for same-level rows
  sync_down + advance + reflux rows are mutually exclusive
  masked source families include phase_direct, C2F, F2C, inactive-parent
```

## Substep Table Audit v27-v33

Artifact:

```text
benchmarks/results/quicklook/amr_d_v32_wall_phase_substep_table_canary/
benchmarks/results/quicklook/amr_d_v33_bgk_force_budget_canary/
```

Script:

```text
tmp/audit_amr_d_wall_phase_substep_table_canary.jl
```

The first table audit showed why the initial substep activation v24 was
invalid: the event table was built with max-level leaf fractions while the
scheduler consumes local 2:1 event stencils. For example, an L0 source in an
R16 tree could emit `1/16` where the local `sync_down` stencil emits `1`.

The WIP table builder was changed to emit local 2:1 rows:

```text
level < max_level: 4 child samples, weight = 1/4
level == max_level: 1 sample, weight = 1
```

The corrected table closes the local stencil audit:

```text
v32 xband_wall R2/R4/R8/R16:
  max_abs_local_mismatch = 0
  local_mismatch_count   = 0
```

This is only a table-construction fix. Activating the corrected substep table
on the BGK+force canary gives v33:

```text
xband_wall R16:
  mass_amr - mass_oracle = -7.73e-12
  Jx_amr / Jx_oracle     = 0.971948
  max |drho|             = 2.42e-2
  max |dux|              = 3.17e-2

xband_h R16:
  Jx_amr / Jx_oracle     = 1.0 within roundoff
```

Compared to v23 (`wall_phase=false`), v33 fixes the global mass loss but keeps
the same order of streamwise momentum loss and reintroduces a large local
density/velocity defect. Compared to v22 (current full-step wall-phase
correction), v33 is worse locally. Therefore the corrected substep table must
remain disabled as the default path.

What v33 proves:

```text
The max-level weighting bug in the WIP substep table is real and fixed.
The remaining failure is not row-weight normalization.
The remaining failure is mask scope / replacement semantics:
  the substep table conserves each local event stencil,
  but replacing the native route families for those sources changes
  the macro state too broadly.
```

The active code path was restored to the v22 full-step correction after v33;
a short R4 check v34 reproduces v22:

```text
xband_wall R4: dJx = -2.291358e-6, Jx ratio = 0.969377
xband_h    R4: exact to roundoff
```

Next implementation gate:

```text
route-family replacement audit
  for each masked event source:
    compute native route-family output on a unit source
    compute substep scatter output on the same unit source
    compare only the families actually masked at that event

Expected:
  sync_down replacement matches C2F rows only
  advance replacement matches direct + F2C + inactive-parent rows only
  no source that is only partially wall-phase should lose unrelated native rows
```

## Route-Family Replacement Audit v38-v39

Artifact:

```text
benchmarks/results/quicklook/amr_d_v38_wall_phase_route_family_replacement/
benchmarks/results/quicklook/amr_d_v39_wall_phase_route_family_replacement/
```

Script:

```text
tmp/audit_amr_d_wall_phase_route_family_replacement.jl
```

After fixing an aliasing bug in the audit diff matrix, the replacement check no
longer reports the spurious `1/36` over-injection from v35-v37. The corrected
R4 run is:

```text
xband_wall_r4: rows = 16, bad_rows = 6, max_score = 1.441416e-2
xband_h_r4:    rows = 0,  bad_rows = 0
```

The family aggregation is decisive:

```text
xband_wall_r4:
  advance_f2c_reflux  sum_score ~= 3.18e-2, max_score ~= 1.44e-2
  sync_down_c2f        sum_score ~= 1.45e-4
  advance_direct       sum_score ~= 2.10e-4
```

The full ratio sweep confirms that the largest individual defect is already
present at shallow nesting and then persists while the number of bad rows grows:

```text
xband_wall R2/R4/R8/R16:
  bad_rows  = 1, 6, 18, 42
  max_score = 6.94e-3, 1.44e-2, 1.42e-2, 1.43e-2

xband_h R2/R4/R8/R16:
  wall-phase masked rows = 0
```

Top target rows are the wall/interface F2C corner populations, for example at
R16:

```text
advance_f2c_reflux tick 8 level 1:
  south_wall|west_iface|wall_level_corner q6  delta ~= +8.680568e-3
  north_wall|west_iface|wall_level_corner q9  delta ~= +8.680568e-3
  south_wall|east_iface|wall_level_corner q7  delta ~= +8.680543e-3
  north_wall|east_iface|wall_level_corner q8  delta ~= +8.680543e-3
```

Interpretation:

```text
The corrected substep wall-phase table has locally normalized weights, but its
current reflux application is not a valid replacement for native F2C/corner
routes. Directly adding coarser-owner packets into reflux_to_coarse bypasses the
level-native fine-to-coarse packet cache and corner redirection semantics. That
over-replaces the native F2C family exactly at the wall-terminating xface
corners.
```

Consequences:

```text
1. Keep the substep wall-phase scatter disabled in production.
2. Do not implement a direct-reflux wall scatter as a macro fix.
3. If the event-keyed wall path is resumed, F2C rows must be written through an
   F2C-equivalent packet/cache path, or compared against an explicitly leaf-
   owned oracle with exact target rows.
4. This audit does not explain H: the H rows are zero because the wall-phase
   mask is wall-touching by construction. H/internal-corner symptoms require a
   separate non-wall corner/restriction timing audit.
```

## Short Multi-Step Forced Momentum Canary v40

Artifact:

```text
benchmarks/results/quicklook/amr_d_v40_level_native_forced_momentum_canary/
```

Script:

```text
tmp/audit_amr_d_level_native_forced_momentum_canary.jl
```

The script was made parametric on `AMRD_FORCED_MOMENT_CORRECTIONS` so the active
correction path can be run without immediately aborting on the intentionally
unstable `correction=false` mass guard.

Configuration:

```text
topologies: xband_wall, xband_h
ratios: R2, R4
steps: 16, 128
mode: level_native, predictor=1, wall_phase_transport_correction=true
```

Summary:

```text
steps=128:
  xband_wall R2: ux_max / cart = 0.964896, max |rho-1| = 1.13e-6
  xband_wall R4: ux_max / cart = 0.920392, max |rho-1| = 3.57e-6
  xband_h    R2: ux_max / cart = 0.998373, max |rho-1| = 1.06e-7
  xband_h    R4: ux_max / cart = 0.998306, max |rho-1| = 1.40e-7
```

This short run agrees with the one-step budget canary: in the current code
state, the H/capped-xband geometry is a clean negative control for the
wall-phase/f2c replacement issue. The simple `xband_wall` already shows a
resolution-dependent velocity deficit over 128 steps, while H remains within
about two per mille of the Cartesian transient.

The top field differences for `xband_wall R4` are concentrated on the finest
wall-adjacent/interface rows, while H errors are an order of magnitude smaller.
This supports the v38-v39 interpretation: the immediate bug to fix is the
wall-terminating xface/F2C-corner path, not a global H/internal-corner rewrite.

## Current Coefficient Audit v41

Artifact:

```text
benchmarks/results/quicklook/amr_d_v41_level_native_corner_coefficients/
```

Script:

```text
tmp/audit_amr_d_level_native_corner_coefficients.jl
```

The coefficient audit output path is now controlled by
`AMRD_CORNER_COEFF_OUTDIR`, so current-code reruns do not overwrite the earlier
20260513 folder.

Current-code result:

```text
uniform R1:         exact
yband R2:           exact
xband_internal R2:  exact
xband_internal R4:  exact
xband_h R2:         exact
xband_wall R2:      bad=32,  max=0.0625
xband_wall R4:      bad=80,  max=0.0625
xband_h R4:         bad=128, max=0.75
```

This refines the interpretation from v40. The H/capped-xband moment budget can
look clean because the raw packet defects cancel for rest/short forced modes,
but the transport operator is not coefficient-exact at nested depth 2.

The smallest new H row is:

```text
xband_h_r4:
  src L2 (21,11) q6 -> oracle L2 (25,15) q6 = 1.0
                         AMR L2 (25,15) q6 = 0.25
                         AMR L2 (26,15) q6 = 0.25
                         AMR L2 (25,16) q6 = 0.25
                         AMR L2 (26,16) q6 = 0.25
```

The route signature for the first source step is direct, but the next native
substep hits a level-native corner/coalesce route and the packet loses its
subcell phase. It is later prolonged back as a flat `1/4` child block. This is
not a wall bounce bug; it is premature coalescence through an inactive parent
during same-level transit.

A broken unit canary was added:

```text
test/test_conservative_tree_subcycling_2d.jl
  "level-native nested H same-level transit canary"
```

Current test state:

```text
julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_subcycling_2d.jl")'
  833 pass, 18 broken, 0 failed
```

Recommended fix target:

```text
Do not revive wall scatter as the main fix.
Do not add a rho/profile correction.

Instead, preserve level-native phase through inactive same-level transit:
  active L+1 packet enters inactive L+1 cells under an active L owner
  it must remain a phase-carrying L+1 packet until the next active L+1 owner,
  not coalesce to L and then flat-prolong back to four children.
```

The likely implementation belongs around the level-native F2C/corner and
inactive-parent paths, not the macro runner:

```text
src/refinement/conservative_tree_routes_2d.jl
src/refinement/conservative_tree_subcycling_2d.jl
src/refinement/conservative_tree_subcycle_buffers_2d.jl
```

The minimal durable design is a phase-carry buffer for inactive level rows (or
equivalent precomputed phase-carry routes) that is consumed by later same-level
advances. A direct `reflux_to_coarse` scatter is explicitly ruled out by v38-v39
because it bypasses native F2C semantics and over-replaces wall xface corners.

## Delayed Same-Level Transit Canary v42

Artifact:

```text
benchmarks/results/quicklook/amr_d_v42_level_native_corner_coefficients/
```

Validation command:

```text
AMRD_CORNER_COEFF_OUTDIR=benchmarks/results/quicklook/amr_d_v42_level_native_corner_coefficients \
  julia --project=. tmp/audit_amr_d_level_native_corner_coefficients.jl
```

Implementation state:

```text
src/refinement/conservative_tree_subcycling_2d.jl
  level-native COALESCE_CORNER can push a delayed same-level packet when an
  active level-L packet first enters a coarser owner through inactive level-L
  cells, then re-enters an active level-L owner before the top-level step ends.
```

The delayed packet is applied at the future `advance(level)` event after the
normal direct stream. This closes the transport-only phase loss without using
the rejected wall/reflux scatter.

Current coefficient summary:

```text
uniform R1:         exact
yband R2:           exact
xband_internal R2:  exact
xband_internal R4:  exact
xband_h R2:         exact
xband_h R4:         exact
xband_wall R2:      bad=32, max=0.0625
xband_wall R4:      bad=80, max=0.0625
```

The locked unit canary is now green:

```text
test/test_conservative_tree_subcycling_2d.jl
  "level-native nested H same-level transit canary"
```

Full local AMR-D subcycling test:

```text
julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_subcycling_2d.jl")'
  837 pass, 14 broken, 0 failed
```

Interpretation:

```text
Closed:
  H/capped-xband internal same-level phase loss at nested depth 2.

Still open:
  wall-adjacent C2F dipole, max coefficient error 0.0625, independent of the
  same-level delayed-transit defect.
```

The delayed same-level correction is deliberately kept out of the macro BGK
path when the stream source is a post-collision collapsed wall-phase source.
Promoting it to collision-aware transport needs a separate canary that proves
the packet reads the right post-collision value at the delayed event, rather
than carrying an uncollided value through inactive cells.

Short forced-momentum anti-regression:

```text
benchmarks/results/quicklook/amr_d_v43_level_native_forced_momentum_canary/

steps=128:
  xband_wall R2: rho=1.128e-6, ux_ratio=0.965
  xband_wall R4: rho=3.573e-6, ux_ratio=0.920
  xband_h    R2: rho=1.060e-7, ux_ratio=0.998
  xband_h    R4: rho=1.404e-7, ux_ratio=0.998
```

This is intentionally unchanged from the v40 symptom split: H is clean on the
short forced-momentum budget, while wall xband still loses flux and keeps the
wall-adjacent density signature. The v42 fix closes one coefficient defect; it
does not claim to solve the remaining wall C2F dipole or the long-run profile
defect.

Rejected C2F shortcut:

```text
Attempt:
  zero one-sided limited-linear slopes at C2F edges so the wall-adjacent unit
  packet splits as [0.25, 0.25, 0.25, 0.25].

Result:
  the C2F packet canary passes, but the analytical affine level-native canary
  fails at max_refined_or_interface = 1.26953125e-4.

Decision:
  reject. The remaining wall C2F dipole is not safely fixed by disabling
  one-sided slopes; any C2F fix must preserve affine exactness.
```
