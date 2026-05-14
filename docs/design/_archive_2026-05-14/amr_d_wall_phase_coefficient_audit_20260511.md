# AMR-D wall-phase coefficient audit

Date: 2026-05-11
Branch: `slbm-paper`

This audit refines `amr_d_wall_phase_simple_solution_audit_20260511.md` with
per-population coefficient evidence from a new canary and the result of a
line-by-line read of `_stream_conservative_tree_level_native_phase_direct_*`,
`conservative_tree_subcycle_sync_down_level_native_phase_routes_*`,
`_stream_conservative_tree_compact_boundary_*` and
`_conservative_tree_limited_linear_child_packet_2d`.

## What the new canary does

Script: `tmp/audit_amr_d_wall_phase_coefficient_canary.jl`
Output:  `benchmarks/results/quicklook/amr_d_wall_phase_coefficient_canary_20260511/`

For each active spec cell and each non-rest population `q`, the canary:

1. fires a unit integrated packet at `(src_id, q)` through the buffered
   `:level_native` subcycle on three specs:
   - `wall_touch_max_level_1` (passes rest, fails per-population);
   - `wall_touch_max_level_2` (the smallest failing rest case);
   - `bulk_touch_max_level_2` (no wall touch, two refinement levels).
2. fires the same packet through the leaf oracle (project to the finest
   Cartesian, run `1 << max_level` steps of `stream_periodic_x_wall_y_F_2d!`,
   restrict to the tree, keep every leaf `q'`).
3. classifies every nonzero (src,q) -> (dst,q') deposit by tracing the leaf
   trajectory: `direct`, `wall_only`, `c2f`, `f2c`, `wall_then_interface`,
   `interface_then_wall`, `amr_only` (only AMR has nonzero), etc.
4. writes a per-row CSV with `oracle_w`, `amr_w`, `diff`, `classification` and
   the level-native route signature for the source.

## Top-line evidence

`summary.csv`:

```text
wall_touch_max_level_1   max_err = 6.25e-2
wall_touch_max_level_2   max_err = 1.00
bulk_touch_max_level_2   max_err = 6.25e-2
```

`by_classification.csv` (sum of |diff| per class):

```text
wall_touch_max_level_1   wall_only:2.5    c2f:2.0    wall_then_interface:0.25  interface_then_wall:0.25
wall_touch_max_level_2   wall_only:2.5    c2f:13.7   wall_then_interface:2.75  interface_then_wall:19.03   amr_only:11.0
bulk_touch_max_level_2   wall_only:4.0    c2f:6.5
```

The audit doc's "rest passes at max_level=1" remains true. Per-population it
does not. This is the central new finding: the operator is not bit-exact
per-population on any wall-touching configuration, including the bulk_touch
case which has no rest defect. The bug is a superposition of independent
defects which only cancel on a strictly uniform input (rest).

## Defects exposed by the canary

### Defect 1: limited-linear slope is applied even with `:flat`

`_conservative_tree_limited_linear_child_packet_2d` is called unconditionally
from both `_stream_conservative_tree_level_native_phase_direct_level_routes_F_2d!`
(line 1727) and `conservative_tree_subcycle_sync_down_level_native_phase_routes_F_2d!`
(line 1243). The user-facing flag `coarse_to_fine_prolongation=:flat` does not
disable the slope correction in the phase paths.

At a wall row the slope is one-sided: `has_south=false` returns
`north_value - center = -F[src,q]` for any south-wall source, breaking the
symmetry between subcell (1, *) and (1, * + 1). On a unit packet this becomes a
1/16 dipole between dst at (i, j) and dst at (i ± 1, j), with sign flipped per
yoff.

Top entries with classification `wall_only`:

```text
src L0 (1,1) q8  ->  L0 (2,1) q6   oracle 0.25  amr 0.3125  diff +0.0625
src L0 (1,1) q8  ->  L0 (8,1) q6   oracle 0.25  amr 0.1875  diff -0.0625
src L0 (2,1) q8  ->  L0 (1,1) q6   oracle 0.25  amr 0.1875  diff -0.0625
src L0 (2,1) q8  ->  L0 (3,1) q6   oracle 0.25  amr 0.3125  diff +0.0625
```

This dipole cancels at rest because every source contributes both sides of the
dipole symmetrically, but on a non-uniform input (Poiseuille, perturbation,
collision step) it leaks into the macro field. It is therefore a real bug
even though the audit doc's rest residual missed it.

### Defect 2: boundary routes silently skipped under phase_resolved_level_native

`_stream_conservative_tree_direct_level_routes_F_2d!` returns early when
`phase_resolved_level_native && is_solid === nothing` (line 1866), before the
boundary routes are processed. Similarly,
`conservative_tree_subcycle_sync_down_routes_F_2d!` (line 1296) short-circuits
to the phase path and never reads `split_route_ranges_by_parent_level`.

Concretely the wall populations at `level < max_level` rely on the phase trace
to consume them: the route table still contains a ROUTE_BOUNDARY entry, but
the operator never visits it. Phase trace is only correct if it can complete
the leaf trajectory within `ratio` substeps and inside the same level family.
That is not always true (see Defect 3).

### Defect 3: phase_direct drops mass when the leaf trajectory crosses two levels

`_stream_conservative_tree_level_native_phase_direct_level_routes_F_2d!`
deposits only when
`spec.cells[dst_id].level == level || continue` (line 1726). For level-1 or
deeper sources, any subcell trajectory that ends inside a `level - 1` cell
silently disappears.

The worst diff in the canary is exactly that:

```text
src L2 (15,4) q8  ->  L0 (3,1) q6   oracle 1.0   amr 0.0   diff -1.0
                                                          classification = interface_then_wall
                                                          route_sig: COALESCE_CORNER:dst35:w1
```

L2 (15,4) is a south-wall L2 cell whose q8 packet, after 4 leaf substeps with
one bounce, lands in L0 (3,1). In the AMR scheduler the packet is:

1. taken from L2 via the COALESCE_CORNER F2C ledger at L2 substep 1
   (correct: leaf trajectory passes L1 (8,2) -> L1 (8,1) -> L1 (7,1) phase
   destination);
2. applied at L1 (7,1) by `apply_sync_up_F_2d!` (correct so far);
3. advanced through L1 phase_direct at L1 substep 1, where three of the four
   L2 subcells either enter L2 fine again *after sync_down(L1->L2) has already
   fired*, or land at L0 (3,1) and are dropped by the same-level check.

Net effect: the entire 1.0 packet is lost between L1 substep 1 and the end of
the L0 step.

The same mechanism explains the `interface_then_wall` family in
`wall_touch_max_level_2`: it accounts for 19 out of 35 units of total |diff|.

### Defect 4: late C2F deposits never reach the corresponding L2 step

In `_stream_conservative_tree_level_native_phase_direct_level_routes_F_2d!`,
when a subcell trajectory enters a refined parent at substep > 1 the function
just `break`s. The phase sync_down deposit for the *next* L1 substep would be
the only way to feed the L2 ghost on time, but sync_down(L1->L2) at the next
substep reads the L1 owned *after* the current advance, by which point the
mass has already been zeroed.

This is what produces the `amr_only` class with weight 0.5 deposits at L1
ledger slots that the oracle never visits.

```text
src L2 (16,3) q8  ->  L1 (7,1) q6   oracle 0   amr 0.5   diff +0.5    amr_only
src L2 (17,3) q9  ->  L1 (10,1) q7  oracle 0   amr 0.5   diff +0.5    amr_only
```

## Why rest still passes at `max_level = 1`

At rest, `F[src, q] = f_eq[q] * volume(src)` is uniform per level after the
inactive-parent restriction. Slopes are zero, so Defect 1 vanishes. The leaf
trajectory of an L0 wall packet completes within `ratio = 2` L1 substeps and
never lands in an L0 cell from an L1 source, so Defect 3 is dormant. Defect 2
is invisible because phase trace handles the wall bounce inline. Defect 4
requires at least two levels of L > 0 sources, so it cannot fire at
`max_level = 1`.

At `max_level = 2` rest only exposes the residual `1.7361e-3` because Defects 3
and 4 do fire for L1 wall sources around the L2 block. That residual is the
*difference of contributions that do not cancel even under uniform input*,
because the L1 source loses mass into L2 paths it never closes.

## Re-evaluation of the three simple options

### Option A: precomputed wall-phase correction route table

Original recommendation. Adds `weight_delta = oracle_w - current_w` entries on
top of the existing operator.

Coefficient evidence shows two reasons to reject Option A as the primary fix:

1. The dominant error class is mass loss, not redistribution: Defects 3 and 4
   together account for ~30 of the 48 units of `|diff|` in the
   `wall_touch_max_level_2` case and the worst single row is a full 1.0 loss.
   A delta table cannot create mass that is not deposited by any route in the
   first place. It would have to *also* introduce new routes for every
   level-jump trajectory, at which point it is no longer "small".
2. The wall-only dipole from Defect 1 is structurally a *slope correction
   bug*, not a routing bug. Encoding it as deltas in a wall-phase table will
   only mask it for the canary geometry, and any small change to the
   refinement topology will regenerate the same pattern.

Option A remains useful as a *diagnostic*: the same machinery that builds the
correction table can validate any new operator.

### Option B: replace wall-touch streaming by a local leaf-phase scatter

Original wording: scatter at child-level granularity for parent/child pairs
touching south/north walls.

The canary makes the constraint sharper. The smallest defect-free operator
for a wall-touching cell must:

- trace each leaf sub-cell at the *finest* level used by oracle, not at
  `src.level + 1`;
- terminate at the active leaf covering the final leaf position regardless of
  whether that leaf sits one level above, at, or one level below the source;
- not double-count with the existing direct/C2F/F2C/boundary table entries,
  so the wall-phase region must opt out of those entries entirely.

This rules out the "level pair" scoping suggested by the audit doc: the
worst case is a packet that traverses three levels (L2 -> L1 -> L0) within
one L0 step. The scatter has to operate at leaf granularity for every cell
in the wall-phase causal cone, not only the immediate parent/child pair.

A clean version is therefore:

- Build a precomputed `WallPhaseScatter2D` table at setup time that lists,
  per `(src_id, q)`, the *exact* oracle deposits `(dst_id, dst_q, weight)`
  produced by the leaf trace, restricted to cells whose causal cone touches
  a physical y-wall.
- During the buffered subcycle timeline, replace the entire chain
  (direct + C2F + F2C + boundary) by these scatter rows for the listed
  sources.
- Leave bulk cells handled by the existing operator, since the canary
  confirms bulk interfaces are exact.

This is structurally identical to a "diagonal corner closure" generalised to
the *full* causal cone of any wall-touching cell.

### Option C: full phase packet operator for all level-native pairs

The canary does not change the case for Option C: it would still subsume A,
B and the bulk operator. Cost remains the same: a rewrite of
`_stream_conservative_tree_direct_level_routes_F_2d!` and the sync_down /
sync_up paths into a single multi-substep phase operator that consults the
full leaf trajectory. The risk of regressing already-green bulk interfaces is
real and the canary doesn't reduce it.

## Ranking

| Option | Canary exactness | Double-counting risk | CPU complexity | GPU compatibility | Risk to bulk |
|--------|------------------|----------------------|----------------|-------------------|--------------|
| A (delta table) | partial: misses mass-loss class | medium (closure already at the same site) | small | precomputed, compact | low |
| B (leaf-scatter for wall-cone) | exact by construction | low if the source-side opt-out is enforced cell-by-cell | moderate (leaf trace + compact pack) | precomputed, compact | low |
| C (full phase operator) | exact, but largest change | high during refactor | large | requires kernel rewrite | medium-high |

Option B as refined above is the smallest change that can be exact, since the
operator already builds the four primitives (direct, split, coalesce,
boundary) and the leaf-scatter is a precomputed override on a small subset of
cells.

## Recommendation

Implement Option B with leaf-granularity scope, not the child-level scope
described in the previous audit:

1. Identify wall-phase cells: any active cell whose leaf bounding box has at
   least one leaf row at `j_leaf = 1` or `j_leaf = ny_leaf`, plus its two
   neighbour columns and the leaves it can reach within `1 << max_level`
   substeps after a wall bounce. The closure is small (it scales with the
   number of refined parents touching walls, not the whole domain).
2. Precompute the scatter: for each wall-phase `(src_id, q)`, run the same
   leaf trace the canary uses and store the resulting
   `(dst_id, dst_q, weight)` rows.
3. In the buffered subcycle, override:
   - the direct route group for `(src_id, q)`;
   - the boundary route group for `(src_id, q)`;
   - the C2F deposits whose source is `(src_id, q)`;
   - the F2C accumulate whose source is `(src_id, q)`.
   The override must be substep-keyed and inserted into the same
   collision/transport timeline as the routes it replaces. A single
   post-step scatter at `advance(L0)` is valid only for the transport-only
   canary; it is not a macro-flow algorithm because it would collapse the
   separate L2/L1/L0 collision cadence.
4. Keep the user-facing `coarse_to_fine_prolongation = :flat` semantics:
   in the leaf trace, every leaf sub-cell carries `F[src,q] / scale^2`
   without slope correction. This kills Defect 1 by construction.
5. The phase_resolved branches in the buffered subcycle must skip wall-phase
   sources entirely so they no longer call `_conservative_tree_limited_linear_*`
   for those cells.

This converts the audit doc's "Option A correction table" into a fully
non-redundant Option B by deleting the original routes for the same sources
instead of adding deltas. Side-by-side comparison with the transport-only
canary's oracle weights must be bit-exact before any collision-aware claim.

## Next minimal canary

Before changing any operator code:

1. Promote the existing leaf-oracle row of the canary into a unit test inside
   `test/test_conservative_tree_subcycling_2d.jl`, using the
   `wall_touch_max_level_2` spec and the `interface_then_wall` row
   `src L2 (15,4) q8 -> L0 (3,1) q6`. The expected post-step value is `1.0`,
   currently `0.0`. Assert coefficient values, not the aggregate
   classification label. Until the fix is implemented this should be recorded
   as `@test_broken`; the fix converts it to a normal `@test`.
2. Add a sibling case `wall_touch_max_level_2_north` (use `j_range` at the
   north wall) so the fix is forced to handle both walls symmetrically.
3. Run `tmp/audit_amr_d_bulk_interface_packet_canary.jl` after the fix to
   confirm bulk interfaces still report `max_err = 0`.

Do not move on to Poiseuille macro-flow or rho dashboards until both 1) and 3)
are green at `1e-14`. The next gate after transport is collision-aware:
initialize equilibrium plus an odd diagonal perturbation, run one
level-native BGK collide+stream step, and compare against the finest Cartesian
oracle at the same physical time. The current Poiseuille rho oscillation is a
symptom of Defects 1-4 acting together; fixing the transport canary above is
only the first closure gate.

## Transport-only closure update

The first implementation pass is intentionally transport-only. In
`stream_conservative_tree_subcycled_buffered_routes_F_2d!`, when all of the
following hold:

```text
boundary = :periodic_x_wall_y
table.sampling = :level_native
interface_time_scaling = :level_native
coarse_to_fine_prolongation = :flat
coarse_to_fine_predictor_weight = 0
alpha_c2f = alpha_f2c = 1
pre_stream_level! = nothing
is_solid = nothing
```

the operator masks source populations whose finest-grid trajectory touches a
physical y wall, runs the existing subcycle operator on the masked field, then
adds the exact finest-grid scatter for the masked sources. This is a
source-side replacement for the transport canary, not the final collision-aware
route family.

Validation after this pass:

```text
test/test_conservative_tree_subcycling_2d.jl
  pass = 771
  broken = 21
  fail = 0

tmp/audit_amr_d_bulk_interface_packet_canary.jl
  one_level_bulk_vertical_interface  max_err = 0
  two_level_bulk_vertical_interface  max_err = 0

tmp/audit_amr_d_wall_phase_coefficients.jl
  wall_touch_xband_max2 rest maxdiff = 4.163336342344337e-17
  worst remaining coefficient class = C2F
  worst remaining coefficient diff = 0.0625

tmp/audit_amr_d_wall_phase_coefficient_canary.jl
  wall_touch_max_level_1  max_err = 6.25e-2
  wall_touch_max_level_2  max_err = 1.25e-1
  bulk_touch_max_level_2  max_err = 6.25e-2
```

The wall-phase mass-loss/rest bug is therefore closed for transport-only
canaries. The remaining per-population coefficient rows are C2F/flat-vs-slope
defects away from the wall-phase causal cone and must not be conflated with
the rho oscillation closure. The next implementation step is still a
collision-aware, substep-keyed `WallPhaseScatter2D`; the current masked
transport replacement is the oracle gate it must preserve.

## Collision-aware gate update

Additional guards now separate the remaining C2F defect from the closed
wall-phase transport defect:

```text
test/test_conservative_tree_subcycling_2d.jl
  level-native wall-only packet guard:
    L0 (1,1) q8 -> L0 (1,1) q6 = 0.5
    L0 (1,1) q8 -> L0 (2,1) q6 = 0.25
    L0 (1,1) q8 -> L0 (8,1) q6 = 0.25
    plus symmetric q9 -> q7

  level-native wall-adjacent C2F dipole canary:
    L1 (7,1) q2 -> L2 (17:18,1:2) q2 should be 0.25 each
    current max deviation remains broken
```

Full test file status after adding these guards:

```text
test/test_conservative_tree_subcycling_2d.jl
  pass = 790
  broken = 24
  fail = 0
```

The collision-aware canary is now expanded and frozen at:

```text
tmp/audit_amr_d_wall_phase_collision_canary.jl
benchmarks/results/quicklook/amr_d_wall_phase_collision_canary_20260511/

mini_wall_touch_max_level_2_noop_rest       max_absdiff = 0
mini_wall_touch_max_level_2_bgk_omega1_rest max_absdiff = 6.944444444444437e-3
wall_touch_xband_max2_noop_rest             max_absdiff = 0
wall_touch_xband_max2_bgk_omega1_rest       max_absdiff = 6.944444444444437e-3
bulk_touch_max_level_2_noop_rest            max_absdiff = 0
bulk_touch_max_level_2_bgk_omega1_rest      max_absdiff = 1.665334536937735e-16
mini_equilibrium_bgk_stream                 max_absdiff = 6.944444444444437e-3
mini_odd_diagonal_bgk_stream                max_absdiff = 6.944444444444437e-3
mini_odd_diagonal_delta                     max_absdiff = 1.682222555698942e-10
```

Residual-contribution ventilation was added to the same canary:

```text
residual_contributions.csv
residual_level_summary.csv
```

After the gate-1 replacement, the no-op residual rows disappear. The remaining
non-roundoff residual is fully inside the wall causal cone and belongs to the
BGK/pre-stream gate:

```text
mini bgk omega=1 rest:
  level 0 in_wall_cone=false max_absdiff=1.665334536937735e-16
  level 0 in_wall_cone=true  max_absdiff=6.944444444444455e-3
  level 1 in_wall_cone=true  max_absdiff=1.540530692729766e-3
  level 2 in_wall_cone=true  max_absdiff=4.340277777777784e-4

xband bgk omega=1 rest:
  level 0 in_wall_cone=false max_absdiff=1.665334536937735e-16
  level 0 in_wall_cone=true  max_absdiff=6.944444444444455e-3
  level 1 in_wall_cone=true  max_absdiff=1.540530692729765e-3
  level 2 in_wall_cone=true  max_absdiff=4.340277777777784e-4

bulk bgk omega=1 rest:
  in_wall_cone=false max_absdiff=1.665334536937735e-16
  in_wall_cone=true  max_absdiff=1.665334536937735e-16
```

Gate 1 is now closed by a source-side replacement table that is enabled under
`pre_stream_level!` only when a structural perturbation probe proves the hook is
a true no-op on wall-phase source rows. This preserves the BGK gate as a real
red gate: `omega=1` rest is still red on wall-touch specs and green on the
bulk-touch spec.

The collision canary now also exports the leaf-oracle timing target for the
next gate:

```text
oracle_tick_consumption.csv
oracle_tick_source_summary.csv
```

The detail table is aggregated by initial `(src_id, src_q)`, `leaf_tick`,
current owner, destination owner, and `owner_advance_tick`. Columns
`source_weight`, `post_collision_F_value`, and `oracle_route_consumption` make
explicit both the normalized path weight and which post-collision population
value is consumed at each tick. The normalized conservation checks are:

```text
all per-tick source weights sum to 1
all final-tick source weights sum to 1
max |post_collision_F_value - oracle_route_consumption| = 0
```

The population-valued columns are not normalized weights: summing
`post_collision_F_value` over ticks intentionally depends on `w_q`, cell volume,
and the number of ticks. For the locked south canary source:

```text
src L2 (15,4) q8:
  tick 1 owner L2 (15,4) -> L1 (7,2), q8
  tick 2 owner L1 (7,2)  -> L1 (7,1), q8
  tick 3 owner L1 (7,1)  -> L0 (3,1), q8
  tick 4 owner L0 (3,1)  -> L0 (3,1), q6, wall_hit=true
```

This is the oracle to match before any Gate 2 implementation. Gate 2 must move
the scatter read to the appropriate post-`pre_stream_level!` event without
weakening the structural no-op guard.

Two Gate-3 range files are also written:

```text
oracle_tick_gate3_prediction.csv
oracle_tick_gate3_prediction_by_dst.csv
```

The column is named `gate3_residual_upper_bound`: it sums absolute
source-range magnitudes and does not encode cancellation signs. On the current
mini odd-diagonal dump the maximum destination bound is:

```text
dst L2 (15,1) q3  upper_bound = 4.7362963290914495e-10
observed mini_odd_diagonal_delta max_absdiff = 1.6822225556989423e-10
```

So the bound is conservative and same-order, but not a signed prediction. It is
still useful as a falsification guard: a tick-decomposed scatter should drive
the observed delta below this envelope and ultimately to roundoff.

The timing dump also shows that an `advance(level)`-only scatter is not a
complete event model. Aggregated by current owner level and next destination
level on the mini dump:

```text
0->0 6768
0->1 144
1->0 144
1->1 864
1->2 144
2->1 144
2->2 1008
```

The `0->1` and `1->2` rows are finer-owner arrivals and must be represented by
the `sync_down` / child-injection side of the schedule, not by a late parent
`advance` deposit. Therefore the next implementation should not be an
`advance`-only patch even for Gate 2; it needs the event table split across
`sync_down`, `advance`, and coarser reflux paths.

A naive substep scatter insertion into the existing `:advance` loop was tested
locally and not kept: with a no-op `pre_stream_level!`, rest still failed at
`5.208333333333332e-3`, showing that arrival timing through `owned`,
`ghost_from_coarse`, and reflux buffers must be designed explicitly.

Updated recommendation:

1. Keep the transport-only closure as the validated analytical baseline.
2. Do not promote to Poiseuille/xband macro yet.
3. Implement `WallPhaseScatter2D` as a precomputed source-side replacement
   family with explicit substep timing.
4. First green gate is closed: no-op `pre_stream_level!` rest is at roundoff
   on mini, wall-touch xband max2, and bulk-touch max_level=2.
5. Next gate: BGK `omega=1` rest at roundoff on the same three specs.
6. Then: mini equilibrium BGK `omega<1` collide+stream against
   the finest Cartesian oracle.
7. Then: mini odd-diagonal delta against equilibrium.
8. Only then rerun the macro rho oscillation case.

## Files referenced

- `tmp/audit_amr_d_wall_phase_coefficient_canary.jl`
- `tmp/audit_amr_d_bulk_interface_packet_canary.jl`
- `tmp/audit_amr_d_wall_phase_collision_canary.jl`
- `benchmarks/results/quicklook/amr_d_wall_phase_coefficient_canary_20260511/`
- `benchmarks/results/quicklook/amr_d_wall_phase_collision_canary_20260511/`
- `src/refinement/conservative_tree_subcycling_2d.jl`
  (functions cited: `_stream_conservative_tree_level_native_phase_direct_level_routes_F_2d!`,
  `conservative_tree_subcycle_sync_down_level_native_phase_routes_F_2d!`,
  `_stream_conservative_tree_direct_level_routes_F_2d!`,
  `_stream_conservative_tree_compact_boundary_level_routes_F_2d!`,
  `_conservative_tree_limited_linear_child_packet_2d`)
- `src/refinement/conservative_tree_routes_2d.jl`
  (`_level_native_diagonal_corner_closure_route_specs_2d!`,
  `_apply_level_native_route_closure_2d!`)
