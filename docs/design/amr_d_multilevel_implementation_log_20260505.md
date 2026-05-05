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
- ratio-2 only;
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

## Next Patch

Next target is ML3: stream-only rest-state canary on the route table.

Required before collision:

- add a CPU `stream_conservative_tree_routes_F_2d!` over the route table;
- keep boundary routes explicit and initially skipped or bounced by a test-only
  policy;
- verify periodic closed rest state on 1, 2, 3, and 4 levels;
- verify one-packet scatter for direct, split, coalesce, and boundary routes;
- keep macro-flow runners disabled until stream-only tests are green.
