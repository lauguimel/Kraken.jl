# AMR-D leaf-equivalent subcycling audit

Date: 2026-05-11
Branch: `slbm-paper`
Scope: voie D conservative-tree streaming, 2D subcycled route scheduler.

## Question

The nested x-band density oscillation must be explained before another
production patch is added. The required operator contract is stricter than the
Poiseuille dashboard:

- one AMR coarse interval must reproduce the equivalent finest Cartesian
  streaming on active leaves;
- the check must hold on an affine integrated population field;
- the first failure must identify whether the route table, subcycle scheduler,
  or interface ledger is wrong;
- populations crossing a coarse/fine interface must be audited separately from
  same-level direct routes.

## Canary

Script:

```text
tmp/audit_amr_d_leaf_equiv_operator_canary.jl
```

Output:

```text
benchmarks/results/quicklook/amr_d_leaf_equiv_operator_canary_20260511/
```

The canary fills a single D2Q9 population with the cell-integral of:

```text
phi(x,y) = 1 + 0.2 x - 0.13 y
```

Then it compares:

1. one buffered AMR-D subcycled transport step;
2. `2^max_level` finest Cartesian `periodic_x_wall_y` transport steps;
3. the Cartesian result restricted back to each active AMR cell.

It runs two meshes:

- `one_level`: an L0 grid with one L1 refined block;
- `two_level`: the same style of L0/L1 grid plus an L2 nested block.

For each case it sweeps q = 2:9 and compares both route modes:

- `sampling=:leaf_equivalent`, `interface_time_scaling=:leaf_equivalent`;
- `sampling=:level_native`, `interface_time_scaling=:level_native`.

The diagnostic records the top cell errors and the route signature used by each
source cell.

The same analytical contract is now also present in the formal subcycling test
ladder:

```text
test/test_conservative_tree_subcycling_2d.jl
  "analytical affine leaf-equivalent operator canaries"
```

Those tests do not use a time-marched Cartesian reference. They evaluate the
expected post-streaming value directly from the affine population at the
departure leaf centers. Current status:

```text
julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_subcycling_2d.jl")'
756 pass, 18 broken
```

Resolved in the analytical ladder:

- `level_native` C2F/F2C populations `q=2:9` are affine-exact for the
  one-level interface canary;
- the nested L0/L1/L2 interface canary is affine-exact for `q=2:9`;
- the phase-resolved C2F and direct-route reconstruction is enabled for
  transport-only runs and for full-postcollision predictor runs
  (`coarse_to_fine_predictor_weight=1`), which is the default
  `route_sampling=:level_native` macro path.

The remaining broken entries are intentional gates for contracts that are not
closed yet:

- `leaf_equivalent` same-level L0 direct routes must become affine-exact under
  subcycled execution, or `leaf_equivalent` must be removed from subcycled
  production dispatch;
- macro-flow/collision propagation still needs a separate validation step
  for wall-touching refinement bands because the current phase path is still
  disabled when refined parents touch the south/north wall.

## Observation

The decisive failure is already present in the one-level case, away from any
interface.

For `one_level`, q=2, `leaf_equivalent`:

```text
max_abs_err      = 8.3333333333335258e-03
max_abs_bulk_err = 8.3333333333335258e-03
worst cell       = L0 (i=3,j=2), near_interface=0
route            = DIRECT:15:0.5|DIRECT:16:0.5
```

The top-error file confirms many L0 bulk direct cells have the same pattern:

```text
benchmarks/results/quicklook/amr_d_leaf_equiv_operator_canary_20260511/top_one_level_q2_leaf_equivalent.csv
```

For the same mesh and population with `level_native`, those L0 bulk direct rows
drop to roundoff:

```text
cell 15, L0 (i=3,j=2), near_interface=0
route = DIRECT:16:1
err   = 0
```

The remaining `level_native` errors are concentrated in L1/interface rows, for
example q=2:

```text
max_abs_err = 2.0572916666666829e-03
worst rows  = L1 cells and near-interface cells
```

The two-level mesh amplifies the same split:

- `leaf_equivalent` still has large L0 bulk direct errors;
- `level_native` removes the bulk direct error but still leaves C2F/F2C and
  corner/interface ledger errors.

## Root cause

Current `leaf_equivalent` route construction samples one finest-leaf
displacement, independently of the source level:

```text
sample_i = (src.i - 1) * scale + si + cx
sample_j = (src.j - 1) * scale + sj + cy
weight   = 1 / scale^2
```

For an L0 source in a max-level-1 tree, q=2 therefore maps the two fine samples
inside a coarse cell to:

```text
0.5 current coarse cell + 0.5 east coarse cell
```

That is the correct route for one finest tick, but the subcycled scheduler
executes the L0 direct route only once during the coarse-level advance. A
coarse-level native advance represents one L0 cell displacement over the coarse
interval. The exact same-level L0 route for q=2 during that event is therefore:

```text
1.0 east coarse cell
```

This explains why a uniform rest field can pass while an affine field fails:
splitting `0.5 current + 0.5 east` preserves constants but evaluates the affine
field at the wrong advected position. It also explains why wall/corner/interface
balances move the visible density oscillation instead of closing the bug.

The bug is therefore algorithmic:

```text
leaf-equivalent route tables are finest-tick routes, but the native subcycle
scheduler executes same-level direct routes as native-level events.
```

This is not an x-versus-y implementation mismatch and not a wall-only corner
bug. The x/y dashboards expose the same operator inconsistency through different
population directions and interface geometry.

## Secondary interface problem closed for transport-only

Switching same-level direct routes to the native displacement was necessary but
not sufficient. The canary also showed residual `level_native` errors in
diagonal L1/interface rows.

The first phase-resolved C2F patch fixed axis populations. The remaining
diagonal error was traced to the direct-route reconstruction time, not to the
F2C packet itself:

```text
sync_down:
  parent inactive rows are restricted from the initial fine state

sync_up:
  parent inactive rows are restricted again from the evolved fine state

parent advance:
  partial direct packets were reconstructed after sync_up
```

For diagonal populations, only part of a coarse cell stays on the parent level;
the rest enters the fine level and later refluxes. The direct part is therefore
sensitive to the parent-level slopes. Recomputing those slopes after `sync_up`
uses evolved inactive refined-parent rows and shifts the affine reconstruction
by one subcell phase.

The fix stores a parent-level source snapshot at `sync_down` for the
transport-only `level_native` path. During the later parent advance,
phase-resolved direct routes use that snapshot while F2C/reflux still uses the
current evolved fine state. This closes `q=6:9` on both one-level and nested
affine canaries.

The same snapshot path is enabled for macro-flow when a full postcollision
predictor is available at `sync_down`. It is intentionally disabled for partial
predictor blends because a blended source is not a well-defined streaming
operator state.

## Non-solutions

The following should not be promoted as the real fix:

- wall-corner balance after collection;
- wider rho/u tolerances in Poiseuille dashboards;
- limited-linear coarse-to-fine prediction as a substitute for exact transport;
- case-specific x-band/y-band correction;
- treating only visible positive/negative rho lobes at corners.

Those can reduce the dashboard symptom, but they do not satisfy the affine
operator canary.

## Implementation direction

The fix should be staged in this order:

1. Promote the affine single-population canary into the AMR-D test ladder. Keep
   `leaf_equivalent` marked failing for same-level native subcycling until the
   route contract is changed.
2. Change the subcycled direct-route policy so same-level direct routes use
   native-level displacement when they are executed by native-level advance
   events. Equivalently, the current `level_native` behavior is the correct
   base contract for direct rows.
3. Add a surgical one-interface C2F/F2C canary with a single active crossing
   population and explicit expected leaf packets. This should include diagonal
   and face crossings.
4. Replace inactive-parent aggregate coalescence by phase-resolved child packet
   accounting, or by an explicitly conservative reconstruction that is proven
   exact on constants and affine fields for the interface canary.
5. Disable/remove the wall-corner balance patch once the operator canary closes.
6. Only after these tests are green, rerun the Poiseuille x-band/y-band
   dashboards and validate `rho` and `u` against Cartesian and analytic output
   at the same final physical time.

## Current status

The analytical transport bug is closed for `sampling=:level_native` in the
transport-only subcycled operator:

```text
benchmarks/results/quicklook/amr_d_leaf_equiv_operator_canary_20260511/summary.csv
one_level q=2:9 level_native max_abs_err <= 4.5e-16
two_level q=2:9 level_native max_abs_err <= 4.5e-16
```

This is not yet a macro-flow closure. The next step is to propagate the same
phase contract through wall-boundary ledger handling.

The first macro smoke after propagation to the full-postcollision predictor is:

```text
benchmarks/results/quicklook/amr_d_level_native_phase_fix_poiseuille_20260511/
```

Cases:

- `amr_d_poiseuille_xband_internal_nested4_debug`, `steps=200`,
  `t_final_leaf_steps=3200`, validation `validated`;
- `amr_d_poiseuille_yband_nested4_debug`, `steps=200`,
  `t_final_leaf_steps=3200`, validation `validated`.

Key active-field gates from `validation.csv`:

```text
xband internal:
  r2_ux_active_vs_reference = 0.9974870997457517
  r2_rho_active_vs_reference = 0.9999999999999866
  r2_ux_active_vs_analytic = 0.9974973816168377
  r2_rho_active_vs_analytic = 0.9999999999999866

yband:
  r2_ux_active_vs_reference = 0.9971209068010744
  r2_rho_active_vs_reference = 1.0
  r2_ux_active_vs_analytic = 0.9971341989109154
  r2_rho_active_vs_analytic = 1.0
```

Attempting to remove the wall-touch guard makes the existing wall x-band rest
canary lose mass (`191.97222222222354` vs `192.00000000000136`), so the band
that reaches the wall remains a separate unresolved boundary-phase task.

## Full-height x-band check

The requested top-to-bottom x-band case was added as:

```text
benchmarks/krk/amr_d_convergence_2d/poiseuille_xband_fullheight_level_native_nested4_debug.krk
```

It uses:

```text
route_sampling = 1
wall_xband_closure = 0
Refine xband { region = [6, 0, 10, 12], ratio = 16 }
```

The strict dashboard run does not reach plotting. It fails the mass guard:

```text
AMR-D mass residual 0.006756464642130292 exceeds roundoff guard 1e-6
```

Relaxing the mass guard for diagnostics is not useful: the run diverges to a
mass residual of `249964.33333287257`. The status is recorded in:

```text
benchmarks/results/quicklook/amr_d_level_native_phase_fix_poiseuille_fullheight_20260511/summary.csv
```

The deeper audit is:

```text
tmp/audit_amr_d_fullheight_wall_phase.jl
benchmarks/results/quicklook/amr_d_fullheight_wall_phase_audit_20260511/summary.csv
```

This separates the failure into three operators:

```text
transport rest:
  fullheight level_native mass_drift ~= 0
  fullheight level_native max local population diff = 9.114583333333332e-03
  worst = L0 (i=5,j=1), q7

macro, Fx=0:
  stable through 40 steps
  mass drift ~= roundoff

macro, Fx=1e-7:
  first negative population by 30 steps
  steps=30 minF = -8.947254232238196e-03 at L0 (i=14,j=1), q6
  steps=40 minF = -8.289839276484317e+01 at L0 (i=3,j=1), q7
```

The key route signature at the south wall/interface corner is:

```text
L0 (i=5,j=1), q7 deficit:
  q9 from same L0 wall/interface corner is split as
  SPLIT_CORNER weight 0.5 + ROUTE_BOUNDARY weight 0.5
  then the fine child refluxes through COALESCE_CORNER weight 1.0
```

For a uniform rest state this keeps total mass but does not preserve the local
D2Q9 rest population. The forced macro run then amplifies that local diagonal
non-equilibrium; the problem is visible first in reflected wall diagonals
`q6/q7` on the bottom L0 row.

So the conceptual bug is now narrower:

```text
The level-native wall/interface diagonal closure is not an exact wall
reflection operator on the subcycled leaf timeline. It mixes a same-event
coarse bounceback with a fine half-step path and only enforces global mass.
```

This is why an internal x-band is fixed by the phase-resolved C2F/direct
snapshot, while a top-to-bottom x-band is not. A correct solution needs a
wall-phase ledger for diagonal wall/interface packets, not another global mass
balance or a macro-flow tolerance.

For comparison, the historical full-height x-band case still validates because
it uses `route_sampling=0` plus the explicit wall x-band closure:

```text
benchmarks/results/quicklook/amr_d_xband_fullheight_legacy_closure_20260511/
r2_ux_active_vs_reference = 0.9930766773746872
r2_rho_active_vs_reference = 0.9999999999999698
r2_ux_active_vs_analytic = 0.9930271244866871
r2_rho_active_vs_analytic = 0.9999999999999698
```

This confirms the current state precisely: internal `level_native` x-band is
closed by the phase fix; top-to-bottom x-band without the old closure still
needs the wall-boundary phase ledger.
