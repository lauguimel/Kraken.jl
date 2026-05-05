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
- `create_conservative_tree_subcycle_schedule_2d`;
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
