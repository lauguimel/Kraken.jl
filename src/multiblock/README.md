# multiblock/

Multi-block structured coupling for Kraken — Gmsh import, shared-node
block exchange, mesh extension, BFS reorientation, topology checks.
Allows decomposing complex geometries into Cartesian or curvilinear blocks
that exchange populations at interfaces every timestep.

## Key entry points

| File | Symbol | Purpose |
|---|---|---|
| `multiblock.jl` | `MultiBlockTopology`, `BlockState2D` | Top-level types + assembly |
| `topology.jl` | `build_topology` | Block connectivity graph |
| `exchange.jl` | `exchange!` kernels | Shared-node halo population exchange at interfaces, including Δf rescaling for FH compatibility |
| `mesh_extend.jl` | mesh extension to interfaces | Adds ghost rings around each block |
| `reorient.jl` | BFS reorientation | Ensures consistent block-local axes after Gmsh import |
| `mesh_gmsh_multiblock.jl` | Gmsh multi-block loader | Imports a Gmsh geometry as N coupled blocks |
| `state.jl` | `BlockState2D` mutable struct | Holds per-block `f`, `fo` arrays + metadata |
| `sanity.jl` | sanity checks | Pre-run validation: orientation, connectivity, dx consistency |

## Critical invariants

- **Non-overlap blocks**: shared nodes are stripped at Gmsh load to avoid
  drag double-counting and profile-offset bugs
  (`feedback_nonoverlap_preferred`).
- **BlockState2D swap**: `f`/`fo` swap MUST update the struct field
  (`bst.f = ...`), not rebind a local — otherwise the next kernel reads
  stale data (`feedback_blockstate_swap`).
- **Corner ownership**: halfway-BB at S/N walls must include `i=1, i=Nx`
  in the corner indices when the corner is an interface-wall (not just
  ZouHe-owned); otherwise 1.4e-3 bit-exact per-step error
  (`project_multiblock_corner_bc_bug`).
- **Diagonal ghost corners must be filled** by exchange (otherwise L2
  4-block diverges around step 948, `project_multiblock_corner_ghost`).
- **Per-block `cx_lu` MUST be sliced from a global `q_wall`** — never
  recomputed from `(cx_phys − x0_block) / dx + 1` (1e-14 FP drift flips
  213 q_wall entries and biases Cd, `feedback_multiblock_geometry`).
- **FH Δf rescaling required at interfaces** under stretched (anisotropic
  dx) Gmsh layouts (`project_bodyfit_square_stretching`).

## Cross-module dependencies

Reads from: `lattice`, `kernels` (BCSpec applied per block face),
`curvilinear` (when blocks carry a non-Cartesian mesh).
Provides to: `drivers` for multi-block flow configurations.

## Status / scope notes

- v0.3 active development. L0–L2 validated below 0.01%; L4 O-grid
  in progress (`project_multiblock_session_*`).
- 9-block configurations validated; SLBM oblique 2-block 0° machine
  precision, 10° at 4.6% (`project_multiblock_oblique_session`).
