# Multi-block structured meshes (v0.3)

Kraken's v0.3 milestone is a **multi-block structured LBM solver**: a
collection of logically-rectangular blocks stitched together at
interfaces, each block running its own curvilinear LBM step on the
GPU. The design target is to handle CFD-grade complex geometries —
body-fitted O-grids around bluff bodies, C-grids around airfoils —
that cannot be mapped onto a single transfinite patch.

This page describes the Phase A MVP (halo-strict exchange, 2-block
validation). Phase B will add the ghost-layer refinement needed for
bit-exact single-block equivalence at interior-boundary cells, and
Phase C will wire up LI-BB per block for curved-body closures on
multi-block meshes.

## Data model

Three types live in [src/multiblock/topology.jl](../../../src/multiblock/topology.jl):

- **`Block`** — an `id::Symbol`, a self-contained `CurvilinearMesh`,
  and a `NamedTuple` of four edge tags `(west, east, south, north)`.
  Each tag is either a user-chosen physical name (`:inlet`, `:cylinder`,
  `:wall`, …) or the reserved `INTERFACE_TAG = :interface` meaning
  "this edge is stitched to another block".

- **`Interface`** — two 2-tuples `(block_id, edge_symbol)` declaring
  that edge X of block A talks to edge Y of block B.

- **`MultiBlockMesh2D`** — a flat vector of blocks + a flat vector of
  interfaces, plus a precomputed `block_by_id::Dict` for O(1) name
  lookup. Construction does not run any validation; call
  `sanity_check_multiblock` explicitly.

The design is deliberately **imperative and modular**: no macros, no
hidden dispatch, everything is a plain struct + function a user can
trace in a debugger. The fused-kernel DSL lives at the per-block
inner-loop level ([src/kernels/dsl/](../../../src/kernels/dsl/)); this
layer is purely for assembling blocks.

```julia
mesh_A = cartesian_mesh(; x_min=0.0, x_max=0.5, y_min=0.0, y_max=1.0, Nx=11, Ny=11)
mesh_B = cartesian_mesh(; x_min=0.5, x_max=1.0, y_min=0.0, y_max=1.0, Nx=11, Ny=11)

blk_A = Block(:left,  mesh_A; west=:inlet,     east=:interface,
                               south=:wall,    north=:wall)
blk_B = Block(:right, mesh_B; west=:interface, east=:outlet,
                               south=:wall,    north=:wall)

mbm = MultiBlockMesh2D([blk_A, blk_B];
                        interfaces=[Interface(; from=(:left, :east),
                                                 to=(:right, :west))])

issues = sanity_check_multiblock(mbm)
any(iss -> iss.severity === :error, issues) && error(...)
```

## Sanity invariants

[src/multiblock/sanity.jl](../../../src/multiblock/sanity.jl) implements
nine invariant families. They fire as `MultiBlockSanityIssue` records
with a severity (`:error` / `:warning`), a short `code` symbol, a
locating `(block_id, edge)` pair, and a human-readable message:

| Code | Catches |
|---|---|
| `NoDuplicateBlockIDs` | two blocks with the same `id` symbol |
| `AllEdgeTagsKnown` | a non-`Symbol` edge tag |
| `UniformElementType` | mixed-precision meshes (FP32 + FP64) |
| `InterfaceRefsExist` | an `Interface` points at an unknown block or edge |
| `InterfaceBothEdgesMarked` | one of the interface edges is tagged as a physical BC instead of `:interface` |
| `InterfaceEdgesSameLength` | `Nξ`/`Nη` mismatch on the two edges |
| `InterfaceEdgesColocated` | physical coordinates of the two edges differ by more than `tol` |
| `InterfaceOrientationTrivial` | edges run in opposite directions (flip not yet supported) |
| `InterfaceEveryMarkedEdgeUsed` | an edge tagged `:interface` is not referenced by any `Interface` |

The sanity check is called explicitly (not at construction) so the
user can inspect the full diagnostic list before deciding to fix or
proceed.

## Halo-strict exchange (Phase A MVP)

Each block runs its standard single-block LBM step on its own `f`
array of size `(Nξ × Nη × 9)`. At the interface edges, the step's
pull for populations that would read *outside* the block falls back
to halfway-BB (the usual boundary treatment in
[src/kernels/dsl/bricks.jl](../../../src/kernels/dsl/bricks.jl)). Those
values are wrong by design; the exchange kernel immediately overwrites
them with the valid post-step populations from the neighbour block,
where those same directions streamed from interior cells.

For a west-east interface (block A on the left with its `:east` edge
at `i = Nξ_A`, block B on the right with its `:west` edge at `i = 1`),
the split by velocity-x sign is:

| populations | sign of `c_qx` | source of valid data | exchange direction |
|---|---|---|---|
| `q ∈ {2, 6, 9}` | `c_qx > 0` | A's interior at `i = Nξ_A − c_qx` | A → B |
| `q ∈ {4, 7, 8}` | `c_qx < 0` | B's interior at `i = 1 − c_qx` | B → A |
| `q ∈ {1, 3, 5}` | `c_qx = 0` | either side (identical by construction) | A → B (deterministic) |

Analogous split for south-north interfaces with `c_qy`.

The exchange is implemented with `view(...) .= view(...)` broadcasts
so it works transparently on CPU, CUDA, and Metal backends without a
dedicated KernelAbstractions kernel.

## Known limitation of the halo-strict MVP

The halo-strict approach achieves **bit-exact single-block
equivalence only when the halfway-BB fallback at interior-boundary
edges gives the same result as a valid interior pull** — which is the
case for a rest initial condition (`u = 0`) where the equilibrium is
symmetric in ±x and ±y, but **not** for a generic flow.

For `u ≠ 0`, each block's step kernel uses wrong populations for `f_q`
with `c_q · n_boundary > 0` (populations that would have streamed
from outside the block). These wrong populations contaminate the
`(ρ, u_x, u_y)` moments at the interface cell, and the BGK collision
uses those wrong moments to produce a wrong post-collision `f_out`.

The exchange then syncs interface cells between A and B — but since
**both** blocks' steps produced wrong values, neither has the right
answer. The error remains at the interface and propagates one cell
inward per step.

**Fix (Phase A.5b, not yet implemented)**: add a ghost row/column
beyond each interface edge, pre-fill it from the neighbour's interior
*before* the step, and the step kernel's pull then reads valid data
instead of triggering halfway-BB. This is the same pattern Kraken
already uses for grid refinement
([src/refinement/refinement.jl](../../../src/refinement/refinement.jl),
`RefinementPatch` with `n_ghost = 2`); the multi-block code will reuse
it as-is.

## MVP validation

[test/test_multiblock_topology.jl](../../../test/test_multiblock_topology.jl) (31 tests)
covers the data-model invariants and the sanity-check happy/failure
paths.

[test/test_multiblock_exchange.jl](../../../test/test_multiblock_exchange.jl) (44 tests)
exercises the exchange kernel in isolation: population splits by
`c_qx`/`c_qy` sign for both W-E and S-N interfaces, away-from-edge
interior untouched, swapped `from ↔ to`, idempotence, error paths.

[test/test_multiblock_canal.jl](../../../test/test_multiblock_canal.jl) (6 tests)
runs a 2-block BGK canal and verifies the three coarse invariants
appropriate for the halo-strict MVP:

1. `u = 0` rest → multi-block ≡ single-block bit-exact over 100 steps.
2. `u = 0.05` → interface columns of A and B agree bit-exact after
   every exchange; total mass drifts less than 1 % over 50 steps
   (drift is dominated by halfway-BB at physical walls, not the
   interface).
3. A 1000-step run produces finite populations (no NaN/Inf blow-up).
