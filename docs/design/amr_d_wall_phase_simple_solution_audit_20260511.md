# AMR-D wall-phase simple solution audit

Date: 2026-05-11
Branch: `slbm-paper`

## Diagnostic freeze

The visible `rho` oscillation must not be treated as a wall-only scalar defect.
The new bulk-interface packet canary shows that a vertical coarse/fine interface
away from physical walls is exact for diagonal populations:

```text
benchmarks/results/quicklook/amr_d_bulk_interface_packet_canary_20260511/summary.csv
one_level_bulk_vertical_interface  max_err = 0
two_level_bulk_vertical_interface  max_err = 0
```

The smallest failing case is therefore not a bulk L/L+1 interface by itself.
It is a wall-touching x-band with at least two refinement levels:

```text
max_level=1  wall-touch x-band level_native rest maxdiff = 0
max_level=2  wall-touch x-band level_native rest maxdiff = 1.7361111111111119e-3
```

The first failing target is the south wall/interface corner:

```text
target L0 (i=4,j=1), q6/q7
```

Packet coefficient comparison against the finest Cartesian oracle shows a
redistribution error, not a single missing mass:

```text
target L0 (i=4,j=1), q6
  src L0 (i=3,j=1), q8   oracle 0.25  amr 0.3125  diff +0.0625
  src L1 (i=9,j=2), q8   oracle 0.75  amr 1.0     diff +0.25
  src L1 (i=10,j=2), q8  oracle 0.25  amr 0       diff -0.25

target L0 (i=4,j=1), q7
  src L0 (i=3,j=1), q9   oracle 0.25  amr 0.1875  diff -0.0625
  src L1 (i=9,j=1), q9   oracle 0.75  amr 0.5     diff -0.25
  src L1 (i=10,j=1), q9  oracle 0.25  amr 0.3125  diff +0.0625
```

At the next level the same problem is larger because L2 packets are
mis-attributed around the wall phase:

```text
target L1 (i=9,j=1), q6
  src L2 (i=21,j=4), q8  oracle 1.0    amr 0.1875  diff -0.8125
  neighboring L2 sources are over-attributed by 0.1875 each
```

So the defect is a phase-timeline closure bug:

```text
diagonal packet touches a physical y-wall, changes q, then crosses an L/L+1
interface before the next native-level synchronization.
```

## Rejected simple fixes

### Scalar rho correction

Rejected. The coefficient errors are per-population and per-source. A scalar
mass or density correction can only move the dipole.

### Local corner balance

Rejected as a durable fix. It can hide the first corner, but the residual moves
to the next row or next level because the operator coefficients are still wrong.

### Disable the phase-resolved path near walls

Rejected by canary. Falling back to the legacy level-native route near the wall
increases the rest error to:

```text
maxdiff = 6.9444444444444475e-3
```

### Let boundary routes handle wall outgoing packets separately

Rejected by canary. Skipping outgoing wall populations in the phase-direct
scatter and leaving compact `ROUTE_BOUNDARY` to reflect them also returns the
same larger legacy error:

```text
maxdiff = 6.9444444444444475e-3
```

### Add one missing reflected parent packet

Rejected. A naive added parent deposit reduces one deficit but overfills the
coarse wall cell and leaves/refactors the L1 deficit. The oracle shows several
source weights change together, so a one-route additive patch is not exact.

## Viable simple fixes

### Option A: Precomputed wall-phase correction route table

Build a compact table only for the wall-phase causal cone. At setup time,
enumerate the ratio-2 subcell paths for cells whose diagonal population can
touch a physical y-wall and cross an adjacent L/L+1 interface. Store correction
routes:

```text
(src_id, src_q, dst_id, dst_q, weight_delta, event_level, substep)
```

where `weight_delta = oracle_weight - current_weight`.

Apply these deltas during the level-native subcycle, before `sync_up` deposits
the parent cache.

Pros:

- Smallest CPU patch.
- Exact on the frozen packet canary by construction.
- Does not affect bulk interfaces, which already pass.
- GPU-compatible later because the table is compact and precomputed.

Cons:

- It is still a correction table around the existing operator, not a cleaner
  rewrite of phase streaming.
- It must be tested carefully so it does not encode one topology shape.

Required canaries before macro:

```text
bulk interface packet canary remains max_err = 0
wall-touch xband max_level=1..4 rest maxdiff <= 1e-14
single-population wall-phase coefficient canary max_err <= 1e-14
```

### Option B: Replace wall-touch level-native streaming by a local leaf-phase scatter

For parent/child pairs touching south/north walls, follow each source cell's
ratio-2 subcells through the exact child-level phase timeline and scatter to:

```text
same-level Fout
coarse-to-fine ghost
fine-to-coarse parent cache
boundary-reflected q
```

This is the cleanest algorithmic fix while staying local to the affected level
pairs.

Pros:

- Directly implements the oracle mechanics.
- Avoids post-hoc deltas.
- Should generalize to arbitrary wall-touch x-band shapes.

Cons:

- More code than Option A.
- Must replace existing direct/C2F/F2C paths for affected packets to avoid
  double counting.

### Option C: Full phase packet operator for all level-native pairs

Replace split/direct/coalesce handling for `:level_native` with one phase packet
operator for every L/L+1 pair.

Pros:

- Conceptually cleanest long-term model.
- One operator handles bulk, walls, and corners.

Cons:

- Too large for the current bug closure.
- Higher CPU cost unless compacted aggressively.
- More risk to already-passing internal interface canaries.

## Recommendation

Implement Option A first as a constrained, test-driven fix:

1. Generate correction routes from the same packet oracle used by the audit,
   limited to wall-phase/interface cells.
2. Apply only population deltas, never scalar rho correction.
3. Keep the correction table generic over level and topology.
4. Promote the wall-touch coefficient canary into
   `test/test_conservative_tree_subcycling_2d.jl`.
5. If Option A becomes too broad or fragile, stop and implement Option B.

Do not rerun Poiseuille dashboards as proof until the wall-touch rest and
single-population coefficient canaries are exact.

## Coefficient audit update

The focused coefficient audit is now reproducible with:

```text
tmp/audit_amr_d_wall_phase_coefficients.jl
benchmarks/results/quicklook/amr_d_wall_phase_coefficients_20260511/
```

It uses the smallest wall-touch x-band with `max_level=2`, compares the
finest Cartesian wall oracle against the current subcycled AMR operator, and
records source-to-destination population coefficients.

Summary:

```text
active_cells = 1344
candidate_packets = 2552
coefficient_rows = 320
rest_maxdiff = 1.7361111111111119e-3
worst rest row = L0 (i=4,j=1), q7
```

Classified coefficient rows:

```text
C2F                  rows 100  max_absdiff 1.5625e-1
F2C                  rows  40  max_absdiff 5.0000e-1
boundary             rows 104  max_absdiff 8.1250e-1
wall-then-interface  rows  76  max_absdiff 6.8750e-1
```

The largest pure wall-then-interface row is:

```text
src L2 (id=579, i=21,j=2), q8
dst L2 (id=626, i=22,j=3), q6
oracle_weight = 1.0
amr_weight    = 0.3125
diff          = -0.6875
oracle event  = iface@t1:L2>L1 + wall@t2 + iface@t3:L1>L2
current route = COALESCE_CORNER dst L1 id=194 weight 1.0
```

The smallest visible south-wall target remains:

```text
dst L0 (id=4, i=4,j=1), q7
src L1 (id=193, i=9,j=1), q9
oracle_weight = 0.75
amr_weight    = 0.5
diff          = -0.25
oracle event  = wall then L1>L0 before native sync closure
current route = ROUTE_BOUNDARY weight 1.0
```

This confirms that the defect is not a scalar wall-rho error. The current
operator sends some wall-phase packets through same-event boundary/F2C routes
and recreates other packets as AMR-only spillover rows. Any fix must therefore
act at population coefficient level and must cover both wall-then-interface
and interface-then-wall rows in the same local causal cone.

## Option ranking after coefficient audit

| option | canary exactness | double-count risk | CPU complexity | GPU compatibility | bulk-interface risk |
| --- | --- | --- | --- | --- | --- |
| A: precomputed `weight_delta` table | High for the frozen coefficient/rest canary if generated from the oracle; needs a nonuniform packet canary before promotion | Medium-high, because deltas must subtract current AMR-only spillover as well as add deficits | Low-medium | Good if packed as compact route deltas | Low if strictly gated to wall/interface causal cones |
| B: local exact scatter phase for wall-touch level pairs | High for the causal mechanism, including wall-then-interface and interface-then-wall | Low-medium if it replaces existing local direct/C2F/F2C packets rather than overlaying them | Medium | Good if prepacked as source-subcell scatter records | Low if enabled only for wall-touch level pairs |
| C: full `level_native` phase operator | Highest long-term | Lowest conceptually | High | Good only after aggressive prepacking | Medium-high, because it rewrites already green bulk interfaces |

Updated recommendation: implement the next canary before any fix. Then prefer
Option B unless Option A can be proven as a narrow replacement-free delta table
with no double counting on both the L0 visible row and the L2 maximum row.
Option C remains a later cleanup path, not the bug-closing step.

Next minimal canary:

```text
case: wall_touch_xband_max2
source: L1 id=193 (i=9,j=1), q9
expected oracle coefficient:
  dst L0 id=4 (i=4,j=1), q7 = 0.75
current AMR coefficient:
  dst L0 id=4 (i=4,j=1), q7 = 0.5
```

The same test file should also include the L2 maximum row as a second assertion
before implementing the fix:

```text
source: L2 id=579 (i=21,j=2), q8
expected oracle coefficient:
  dst L2 id=626 (i=22,j=3), q6 = 1.0
current AMR coefficient:
  dst L2 id=626 (i=22,j=3), q6 = 0.3125
```

## Transport-only closure update

The first implementation pass uses a source-side wall-phase replacement only
for transport-only `:level_native` calls. Wall-touch source populations are
masked out of the existing route operator and reinserted through the exact
finest-grid wall scatter. Collision hooks and macro-flow paths are not routed
through this shortcut.

Validated status:

```text
test/test_conservative_tree_subcycling_2d.jl:
  pass = 771, broken = 21, fail = 0

tmp/audit_amr_d_bulk_interface_packet_canary.jl:
  one_level_bulk_vertical_interface max_err = 0
  two_level_bulk_vertical_interface max_err = 0

tmp/audit_amr_d_wall_phase_coefficients.jl:
  wall_touch_xband_max2 rest maxdiff = 4.163336342344337e-17
```

So the wall-phase transport/rest gate is closed. The remaining coefficient
audit rows are now C2F flat-vs-slope discrepancies away from the wall-phase
causal cone; they are a separate per-population exactness issue. The next gate
before any macro claim is the collision-aware odd-diagonal perturbation canary.

## Collision-aware gate update

The next gate has been added as:

```text
tmp/audit_amr_d_wall_phase_collision_canary.jl
benchmarks/results/quicklook/amr_d_wall_phase_collision_canary_20260511/
```

It compares one level-native BGK collide+stream step against a finest-grid
Cartesian oracle on the wall-touch max_level=2 mini spec. It reports both the
raw field error and the perturbation-only delta after subtracting the
equilibrium run.

Current result:

```text
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

The same canary now writes `residual_contributions.csv` and
`residual_level_summary.csv`. The no-op rows have disappeared after the
gate-1 source-side replacement. Remaining non-epsilon rows are entirely within
`in_wall_cone=true` on the mini and xband BGK `omega=1` specs; the bulk-touch
spec only reports roundoff-level rows (`1.665334536937735e-16`).

This still fails before any macro-flow claim because the BGK/pre-stream path is
red on wall-touch and green on bulk. The gate-1 implementation is deliberately
limited to structural no-op hooks: it probes a copied wall-phase source state
with a local perturbation and only removes the hook when the hook is truly
inactive on those rows.

For the next gate, the script also writes the leaf-oracle timing dump:

```text
oracle_tick_consumption.csv
oracle_tick_source_summary.csv
```

The dump records `(src_id, src_q, leaf_tick)` plus current owner, destination
owner, `owner_advance_tick`, `post_collision_F_value`, and
`oracle_route_consumption`. It also includes a normalized `source_weight`; the
per-tick and final-tick source weights both sum to `1.0`, while
`post_collision_F_value` and `oracle_route_consumption` match exactly. It is
the target for the BGK `omega=1` timing fix, not a macro-flow proof.

The companion `oracle_tick_gate3_prediction*.csv` files expose
`gate3_residual_upper_bound`. It is a conservative magnitude bound, not a
signed prediction: the current max destination bound is `4.7362963290914495e-10`
while the observed perturbation delta is `1.6822225556989423e-10`.

The dump contains finer-owner transitions (`0->1` and `1->2`) as well as
coarser reflux transitions. A Gate 2 implementation must therefore split the
scatter event table across `sync_down`, `advance`, and reflux application;
an `advance(level)`-only patch is not a complete timing model.

The next implementation should therefore not be a post-step macro override.
It must be a real `WallPhaseScatter2D` route family with explicit substep
arrival timing. The strict gate order is:

```text
1. closed: no-op pre_stream rest on mini + wall-touch xband + bulk-touch
2. next: BGK omega=1 rest on the same three specs
3. mini equilibrium BGK omega<1 collide+stream vs finest Cartesian oracle
4. mini odd-diagonal delta vs equilibrium
5. macro rho oscillation check
```
