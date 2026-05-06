# AMR D Multilevel Implementation Log

Date: 2026-05-05

## ML0 Static Tree

Implemented the first multilevel D layer as a static, state-free tree builder.
This is intentionally not a runtime dispatch path yet.

New public APIs:

- `ConservativeTreeRefineBlock2D`;
- `ConservativeTreeSpec2D`;
- `create_conservative_tree_spec_2d`;
- `conservative_tree_refine_blocks_from_krk_2d`;
- `create_conservative_tree_spec_from_krk_2d`;
- `conservative_tree_cell_id_2d`;
- `conservative_tree_children_2d`;
- `conservative_tree_is_active_leaf_2d`.

The builder supports:

- named root refine blocks;
- nested child refine blocks with `parent = <name>`;
- the `base` parent alias for explicit root-level `.krk` blocks;
- ratio-2 programmatic blocks;
- power-of-two `.krk` ratios expanded into nested ratio-2 blocks;
- active leaf extraction;
- parent/child tables;
- active volume checks;
- 2:1 balance validation across D2Q9 neighbor directions.

Runtime status:

- existing 0/1-level D runners are unchanged;
- nested `.krk` files can now build a static `ConservativeTreeSpec2D`;
- nested runtime streaming remains disabled;
- `create_conservative_tree_patch_set_from_krk_2d` still rejects nested
  `Refine parent` blocks, as intended.

## KRK Auto-Nesting Helper

The static spec helper now accepts `.krk` `Refine` ratios greater than 2 when
the ratio is a power of two.

Example:

```krk
Refine cylinder_roi { region = [3, 1.5, 4, 2.5], ratio = 8 }
```

This expands to a nested chain of ratio-2 blocks:

- `cylinder_roi_L1`;
- `cylinder_roi_L2`;
- `cylinder_roi`.

The final block keeps the user-facing name. Intermediate blocks are generated
with `_Lk` suffixes. The target fine region is quantized at the requested
parent level, then every coarser level is reconstructed by padding the child
range by one child-level cell before coarsening. This gives a simple 2:1
nesting buffer while keeping the `.krk` file compact.

Scope:

- enabled for `conservative_tree_refine_blocks_from_krk_2d` and
  `create_conservative_tree_spec_from_krk_2d`;
- still state-free and runtime-disabled;
- `create_conservative_tree_patch_set_from_krk_2d` remains ratio-2 only.

## Surgical Tests

Added `test/test_conservative_tree_spec_2d.jl`.

Coverage:

- programmatic two-level nesting;
- active volume conservation;
- parent/child lookup;
- active leaf lookup;
- four-level static nesting canary;
- same-level overlap rejection;
- missing parent rejection;
- child-outside-parent rejection;
- 2:1 balance rejection;
- `.krk` nested refine conversion;
- `.krk` ratio 4 and 8 auto-nesting expansion;
- non-power-of-two ratio rejection;
- existing `cylinder_nested4_probe.krk` static build.

Commands run:

```bash
julia --project=. -e 'using Test; include("test/test_conservative_tree_spec_2d.jl")'
julia --project=. -e 'using Test; include("test/test_conservative_tree_multipatch_2d.jl")'
julia --project=. -e 'using Test; include("test/test_conservative_tree_topology_2d.jl")'
```

Observed results:

- `test_conservative_tree_spec_2d.jl`: 39 pass;
- `test_conservative_tree_multipatch_2d.jl`: 23 pass;
- `test_conservative_tree_topology_2d.jl`: 13794 pass.

## Next Patch

ML1 is now implemented as recursive conservative projection on the static tree.

New public APIs:

- `allocate_conservative_tree_F_2d`;
- `active_population_sums_F_2d`;
- `level_population_sums_F_2d`;
- `coalesce_conservative_tree_ledgers_F_2d!`;
- `explode_conservative_tree_ledgers_F_2d!`.

The storage contract is deliberately minimal:

- one dense CPU matrix row per tree cell;
- nine D2Q9 integrated populations per row;
- active leaves and inactive parent ledgers share the same matrix;
- no runtime stream/collide state is implied yet.

Additional tests:

- random active-leaf populations coalesce bottom-up to level-0 ledgers;
- every inactive parent ledger equals the sum of its four children;
- random level-0 populations explode top-down to active leaves;
- explode then coalesce recovers level-0 populations to roundoff;
- incompatible matrix dimensions are rejected.

Updated observed result:

- `test_conservative_tree_spec_2d.jl`: 1772 pass.

## ML2 Static Route Table

Implemented static multilevel route-table construction without applying
streaming.

New public APIs:

- `ConservativeTreeRouteTable2D`;
- `create_conservative_tree_route_table_2d`.

The route builder:

- loops over active leaves and D2Q9 populations;
- samples each source cell at one leaf-equivalent level below the source;
- maps samples back to the active leaf that owns them;
- accumulates duplicate destination/kind pairs;
- classifies routes as `DIRECT`, `SPLIT_FACE`, `SPLIT_CORNER`,
  `COALESCE_FACE`, `COALESCE_CORNER`, or `ROUTE_BOUNDARY`;
- rejects route construction if a route crosses more than one AMR level.

Additional tests:

- every `(src, q)` route group has total weight 1;
- direct same-level packet route;
- boundary packet route;
- coarse-to-fine face split;
- fine-to-coarse face coalesce;
- no non-boundary route crosses more than one level;
- route table builds for the existing four-level cylinder `.krk` canary.

Updated observed result:

- `test_conservative_tree_spec_2d.jl`: 2868 pass.

## ML3 CPU Route Scatter

Implemented a CPU reference scatter over the static route table.

New public API:

- `stream_conservative_tree_routes_F_2d!`.

Boundary policy is explicit:

- `:skip` drops boundary routes for packet-level route tests;
- `:bounceback` reflects boundary routes into `opposite(q)` on the source cell.

Additional tests:

- direct packet scatter;
- coarse-to-fine split packet scatter;
- fine-to-coarse packet scatter;
- invalid output/input matrix rejection;
- invalid boundary policy rejection;
- closed nested mass conservation with test-only bounceback.

Important limitation found while testing:

- local rest-state equality on a nested tree is not green yet. Packet weights
  and global closed mass are conserved, but interface cells do not remain
  exactly at rest after one scatter. This must be solved before collision or
  macro-flow runners are enabled on the multilevel path.

Updated observed result:

- `test_conservative_tree_spec_2d.jl`: 2876 pass.

## Next Patch

ML3b diagnostic was started on route-interface rest state.

Finding:

- A single 0/1-level tree can keep local rest with the current leaf-equivalent
  route table.
- A nested tree with deeper levels conserves closed mass, but local rest is not
  exact at coarse/fine interfaces.
- A simple local halo rule for choosing leaf-equivalent routes is not sufficient:
  it fixes one interface and breaks another. The remaining fix is not a
  one-line routing predicate.

Test status:

- The known local rest-state gap is now captured as `@test_broken` in
  `test/test_conservative_tree_spec_2d.jl`.
- This keeps CI honest: global mass conservation remains tested, and the next
  milestone is explicitly blocked until `maximum(abs.(Fout - Fin)) <= 1e-14`
  becomes true on a nested rest state.

## Next Patch

Next target is ML3b completion: debug route-interface rest state before any
collide step.

Required before collision:

- isolate the smallest one-interface rest-state failure;
- compare against the existing 0/1-level route-native table;
- decide whether the multilevel route table needs source reconstruction,
  destination accumulation rescaling, or a two-stage ledger synchronization;
- add a rest-state surgical test for direct, face interface, and corner
  interface separately;
- keep macro-flow runners disabled until rest-state tests are green.

## ML3c Route Rest-State Diagnostic

Debugged the nested rest-state failure down to route-table construction.

Finding:

- the 0/1-level route table preserves closed rest state exactly;
- the first multilevel route table sampled every intermediate active source at
  `level + 1`, even when that source was leaving toward a coarser level;
- that underfilled coarse destinations at the outer boundary of an
  intermediate patch;
- switching intermediate sources to direct/coalesced routes away from the next
  finer halo reduces the nested rest-state residual, but does not close it.

Implemented:

- level-0 sources keep the historical leaf-equivalent route convention;
- intermediate-level sources use leaf-equivalent sampling only when their
  same-level Moore neighborhood touches an inactive parent refined to the next
  level;
- intermediate-level sources away from a finer halo route directly to same
  level or coalesce to the parent level.

Current surgical status:

- single-level closed rest-state route streaming is now explicitly tested green;
- nested closed mass remains conserved to roundoff;
- local nested rest-state equality remains `@test_broken`.

Remaining interpretation:

- the unresolved residual is now concentrated around corners/faces of the next
  finer level;
- this is consistent with the existing subcycling ledger tests: face balance
  needs two fine substeps per coarse step, and corner balance needs the reflux /
  time-integration closure rather than a pure one-pass scatter table.

## ML4 Level-Agnostic Subcycling Architecture

Added a recursive subcycling schedule layer. This is the architectural split
needed to avoid hard-coding `L0/L1`, `L1/L2`, or any specific number of levels.

New concepts:

- `ConservativeTreeSubcycleEvent2D`;
- `ConservativeTreeSubcycleSchedule2D`;
- `ConservativeTreeSubcycleLedgerBank2D`;
- `create_conservative_tree_subcycle_schedule_2d`;
- `create_conservative_tree_subcycle_ledger_bank_2d`;
- `conservative_tree_subcycle_events_at_tick_2d`;
- `conservative_tree_subcycle_advance_counts_2d`;
- `conservative_tree_subcycle_sync_counts_2d`.

Contract:

- one level-0 step is represented in finest-level integer ticks;
- with ratio 2 and `Lmax = N`, level `l` advances every
  `2^(N-l)` finest ticks;
- each parent interval is:
  1. `sync_down parent -> child`;
  2. recursively execute child sub-intervals;
  3. `sync_up child -> parent`;
  4. `advance parent`.

Example for `Lmax = 3`:

- level 0 advances once;
- level 1 advances twice;
- level 2 advances four times;
- level 3 advances eight times;
- sync events exist only between adjacent levels.

This schedule owns no populations and performs no physics. It is the future
dispatch spine for route streaming, subcycling ledgers, and reflux. The next
implementation patch should bind the existing face/corner ledgers to this
schedule for a minimal `L/L+1` interface, then recurse over all adjacent pairs.

Ledger binding:

- one runtime ledger buffer is allocated per adjacent pair `L/L+1`;
- `sync_down L -> L+1` deposits coarse-to-fine face/corner packets into that
  pair ledger;
- every `advance L+1` event maps to a local substep of the parent interval and
  accumulates fine-to-coarse face/corner packets into the same pair ledger;
- `sync_up L+1 -> L` exposes the completed pair ledger for the future reflux /
  parent update;
- the same helpers work for a single `L/L+1` pair and recursively for all
  adjacent pairs in `L0..Lmax`.

This is still an accounting layer. It does not yet apply the completed ledger
to population arrays.

## ML4b Spatial Ledger Binding

Added the missing spatial ownership layer between the recursive schedule and
real population matrices.

New internal APIs:

- `ConservativeTreeSubcycleSpatialLedgerBank2D`;
- `create_conservative_tree_subcycle_spatial_ledger_bank_2d`;
- `conservative_tree_subcycle_spatial_pair_ledgers_2d`;
- `conservative_tree_subcycle_spatial_ledger_2d`;
- `reset_conservative_tree_subcycle_spatial_bank_2d!`;
- `reset_conservative_tree_subcycle_spatial_pair_2d!`;
- `conservative_tree_subcycle_sync_down_routes_F_2d!`;
- `conservative_tree_subcycle_accumulate_advance_routes_F_2d!`;
- `conservative_tree_subcycle_apply_child_advance_injection_F_2d!`;
- `conservative_tree_subcycle_apply_sync_up_F_2d!`.

Contract:

- one ledger is allocated per refined parent cell, grouped by adjacent level
  pair `L/L+1`;
- `sync_down` consumes split routes from the static route table and deposits the
  routed coarse packet fraction into the child slot;
- child `advance` events accumulate coalesce routes into the owning parent-cell
  ledger for the correct local substep;
- child advance injection applies the coarse-to-fine substep contribution to
  `F[child_cell, q]`;
- `sync_up` applies the accumulated fine-to-coarse packets to the parent-level
  destination cell derived from `(parent.i + c_qx, parent.j + c_qy)`.

Important design point:

- the spatial ledger follows the route table topology, but the subcycled
  transport applies time weights: coarse-to-fine split packets are distributed
  over the fine substeps, and fine-to-coarse packets are accumulated with
  `dt_f / dt_c = 1 / ratio` per fine advance. This preserves active mass in the
  transport skeleton and reduces the local rest residual, while leaving the
  remaining corner closure visible as a broken canary.

Validated by:

- one `L/L+1` interface deposits coarse-to-fine packets over two child
  substeps, then accumulates time-weighted fine-to-coarse packets back to the
  coarse row;
- a two-level nested tree recursively allocates and uses ledgers for every
  adjacent pair;
- existing nested route/spec/topology tests remain green.

Commands run:

```bash
julia --project=. -e 'using Test; include("test/test_conservative_tree_subcycling_2d.jl")'
julia --project=. -e 'using Test; include("test/test_conservative_tree_spec_2d.jl")'
julia --project=. -e 'using Test; include("test/test_conservative_tree_topology_2d.jl")'
```

Observed results:

- `test_conservative_tree_subcycling_2d.jl`: 180 pass;
- `test_conservative_tree_spec_2d.jl`: 2906 pass, 1 broken known nested
  one-pass rest-state gap;
- `test_conservative_tree_topology_2d.jl`: 13794 pass.

Next implementation patch:

- add a reference subcycled transport step that separates same-level route
  scatter from interface route ledgers;
- keep collision disabled until the transport-only rest-state canary is green;
- then add BGK/Guo active-leaf collision and channel patch tests.

## ML4c Reference Subcycled Transport Skeleton

Added a CPU matrix reference transport driven by the recursive schedule.

New internal API:

- `stream_conservative_tree_subcycled_routes_F_2d!`.

Contract:

- `max_level = 0` delegates to the existing one-shot route scatter;
- each `:advance` event scatters only direct/boundary routes whose source cell
  belongs to the event level;
- split routes are collected on `:sync_down` into the spatial ledgers;
- child advance events inject the relevant coarse-to-fine substep into the
  child-level output rows;
- child advance events also accumulate coalesce routes into the owning
  parent-cell ledger;
- `:sync_up` applies fine-to-coarse packets into a pending parent-level output
  buffer so they are not streamed a second time by the parent advance.

Validated by:

- no-refinement subcycled transport is exactly equal to
  `stream_conservative_tree_routes_F_2d!`;
- one-level closed rest state conserves active mass to roundoff.

Known open canary:

- closed one-level local active-leaf rest equality is now green after preserving
  route-level fine-to-coarse destinations. Nested `Lmax >= 2` transport still
  needs the same closure audit across recursive parent ledgers before collision
  is enabled.

Observed result:

- `test_conservative_tree_subcycling_2d.jl`: 183 pass.

## ML4d Fine-To-Coarse Spatial Destinations

Closed the one-level subcycled rest-state canary.

Finding:

- the spatial ledger kept fine-to-coarse orientation totals per refined parent,
  but applying those totals by `(parent.i + c_qx, parent.j + c_qy)` is not
  sufficient for diagonal corner routes;
- several children of the same refined parent can have the same diagonal `q`
  while routing to different coarse cells;
- aggregating only by `(parent, q)` preserves mass but redistributes diagonal
  populations locally.

Implemented:

- `ConservativeTreeSubcycleSpatialLedgerBank2D` now carries
  `fine_to_coarse_route_packets`, grouped by adjacent pair and keyed by the
  actual route destination `(dst_cell, q)`;
- route accumulation still updates the orientation ledger for diagnostics, but
  `sync_up` applies the route-spatial packets when present;
- reset helpers clear both orientation ledgers and route-spatial packet maps.

Gate now green:

- no-refinement subcycled transport equals the existing route scatter exactly;
- one-level closed rest state preserves active mass and every active population
  to roundoff.

Remaining blocker:

- nested rest-state (`Lmax >= 2`) still has a mass/local residual. The next
  patch should repeat this route-spatial audit recursively: likely the
  intermediate inactive parent rows need their own pending destination maps
  before the next `sync_down`.

## ML4e Nested Rest-State Diagnostic

Started the recursive inactive-parent path needed by nested transport.

Implemented:

- inactive parent rows at level `L` can now generate coalesce routes toward
  level `L-1` during an `advance L` event;
- pending packets received from level `L+1` are made visible to level `L`
  before `advance L`, except for level 0 where applying them before streaming
  would double-stream the reflux;
- nested rest-state is captured as an explicit broken canary in
  `test_conservative_tree_subcycling_2d.jl`.

Current nested status:

- one-level subcycled transport remains green to roundoff;
- two-level nested rest still has a mass/local residual. The remaining gap is
  now narrower and points to recursive `sync_down`: packets injected from
  `L-1` into inactive `L` parent rows must be made available to `L -> L+1`
  before the child schedule starts, without double-applying them to active
  level rows.

Next patch:

- split pending buffers by direction (`down` vs `up`) instead of using one
  generic pending matrix;
- apply `down` pending immediately before recursive child `sync_down`;
- keep `up` pending delayed until the receiver level output buffer, as in the
  one-level fix.

## ML4f Rest Diagnostic Helper

Added a reusable rest-state diagnostic for the subcycled transport skeleton.

New internal API:

- `diagnose_conservative_tree_subcycled_rest_2d`.

It returns:

- active initial/final mass;
- active mass drift;
- maximum active-population residual;
- per-level mass drift;
- per-orientation D2Q9 drift.

Validated:

- the diagnostic reports roundoff drift for the green one-level subcycled
  rest canary;
- the nested canary now records the same failure through both direct
  `Fout - Fin` checks and diagnostic fields.

Negative experiments kept out of source:

- applying `L2 -> L1` pending packets as a source before `advance L1`
  reduced one part of the residual but worsened the final nested mass;
- coalescing inactive parent rows before `sync_down` and splitting from those
  rows amplified the nested residual. The route-spatial source of truth must
  remain active-route based until a stricter recursive closure is derived.

## ML4g Algorithm Step-Back And Buffer Contract

The nested subcycling blocker was reclassified as an algorithmic state
separation problem, not as a local route-weight tuning problem.

New reference document:

- `docs/design/amr_d_subcycling_algorithm.md`.

New internal APIs:

- `ConservativeTreeSubcycleLevelBuffers2D`;
- `ConservativeTreeSubcycleBufferBank2D`;
- `create_conservative_tree_subcycle_buffer_bank_2d`;
- `reset_conservative_tree_subcycle_level_buffers_2d!`;
- `reset_conservative_tree_subcycle_buffer_bank_2d!`;
- `conservative_tree_subcycle_store_owned_level_2d!`;
- `conservative_tree_subcycle_store_active_owned_2d!`;
- `conservative_tree_subcycle_restore_owned_level_2d!`;
- `conservative_tree_subcycle_apply_reflux_to_owned_level_2d!`;
- `conservative_tree_subcycle_restrict_level_2d!`;
- `conservative_tree_subcycle_restrict_all_levels_2d!`;
- `conservative_tree_subcycle_prolong_F_to_child_ghost_2d!`.

Contract:

- `owned` is the committed level state;
- `ghost_from_coarse` is the parent-to-child reconstruction buffer;
- `reflux_to_coarse` is child-to-parent correction waiting for synchronization;
- `restrict_to_parent` is the conservative child sum used for bottom-up
  synchronization.

Important decision:

- the existing transport skeleton is not changed by this patch;
- the nested rest-state canary remains broken on purpose;
- the next patch must route the scheduler through these buffers instead of
  mutating one shared `Fstate` during every phase.

New surgical tests:

- buffer roles are disjoint;
- reflux is explicit and clearable;
- bottom-up restriction sums active descendants through nested inactive parents;
- coarse-to-fine ghost prolongation is conservative and does not touch owned
  child rows.

## ML4h Buffered Transport Reference

Added a separate transport-only reference path that drives the existing
recursive schedule through the explicit buffer contract.

New internal API:

- `stream_conservative_tree_subcycled_buffered_routes_F_2d!`.

Implemented timing:

- `sync_down` reads the committed parent `owned` buffer;
- child advance writes coarse-to-fine injection through `ghost_from_coarse`;
- `sync_up` writes fine-to-coarse packets into `reflux_to_coarse`;
- child completion also refreshes `restrict_to_parent` for covered inactive
  parent rows;
- parent advance streams from committed `owned` rows and applies reflux as an
  output correction for that parent interval.

Validated:

- no-refinement buffered transport delegates to the existing route scatter;
- single-level buffered subcycling preserves closed rest state to roundoff;
- the legacy one-level path remains green.

Current nested status:

- buffered nested rest is still broken, with the same local population residual
  scale as the legacy path;
- this confirms that separating buffers is necessary but not sufficient;
- the next fix must add an explicit coarse/fine reconstruction rule at the
  interface, likely `f_eq + alpha f_neq`, rather than tuning route weights.

## ML4i Global-Finest Route Sampling

Root cause found for the nested rest residual:

- the route table was generated by sampling at most one level below each source
  cell;
- this is exact for `Lmax = 1`, but not well-balanced when an intermediate
  level also owns a finer island;
- in that case the active leaves are only an exact partition on the finest
  level of the tree.

Patch:

- `create_conservative_tree_route_table_2d` now samples every active source on
  `spec.max_level` and maps each sample back to the owning active leaf;
- this preserves the existing one-level behavior, because `spec.max_level == 1`
  gives the old sample level;
- it is a CPU reference correctness path. A production path can later compress
  these routes analytically, but must preserve the same finest-level partition
  semantics.

New status:

- static closed nested rest is locally exact to roundoff;
- buffered recursive subcycling is locally exact on the nested rest canary;
- the old shared-`Fstate` subcycling path remains broken and should not be used
  as the production algorithm.

Next gate:

- add the physical interface reconstruction layer
  `f = f_eq(rho, u) + alpha f_neq`;
- first surgical tests: rest, uniform velocity, and shear crossing a nested
  interface;
- only then resume Couette/Poiseuille and obstacle macro-flow ramps.

## ML4j Integrated Eq/Neq Reconstruction Primitive

Added the scalar reconstruction primitive required before any physical
coarse/fine interface closure is attempted.

New API:

- `macrostate_integrated_D2Q9(Fcell, volume)`;
- `reconstruct_integrated_D2Q9_eq_neq!(Fdst, dst_volume, Fsrc, src_volume;
  alpha=1)`.

Contract:

- rows store integrated populations `F_q = f_q * cell_volume`;
- the source macrostate is recovered from integrated moments;
- destination populations are reconstructed as
  `Fdst_q = Vdst * (feq_q(rho, u) + alpha * (Fsrc_q/Vsrc - feq_q))`;
- mass and momentum are preserved for any `alpha` because the non-equilibrium
  part has zero zeroth and first moments;
- `alpha = 1` makes a parent-to-four-children split coalesce exactly back to
  the parent row.

Validated:

- mass and momentum preservation;
- non-equilibrium shear stress scales with `alpha`;
- `alpha = 1` child split roundtrips through coalescence;
- `Float32` smoke test compiles and preserves equilibrium.

Not wired yet:

- `sync_down` still injects route packets directly;
- the next patch must use this primitive at the interface and add uniform
  velocity/shear canaries before macro-flow ramps.
