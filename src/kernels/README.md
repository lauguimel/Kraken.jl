# kernels/

GPU-portable LBM kernels (collide, stream, boundary conditions, equilibrium)
and the runtime kernel DSL (PullSLBM bricks, BCSpec). Every Kraken
simulation flows through this module on every timestep. All kernels are
`KernelAbstractions`-portable (CPU / CUDA / Metal).

## Key entry points

| File | Symbol | Purpose |
|---|---|---|
| `dsl/lbm_spec.jl`, `dsl/bricks.jl`, `dsl/lbm_builder.jl` | `PullSLBM`, `BCSpec`, `build_lbm` | Runtime kernel fusion DSL — assemble a complete timestep from bricks |
| `collide_stream_2d.jl` | `collide_stream_2d!` | Fused collide+stream kernel (BGK), default 2D path |
| `collide_guo_2d.jl` | `collide_guo_2d!` | BGK with Guo half-step forcing (Convention I) |
| `collide_mrt_2d.jl` | `collide_mrt_2d!` | Multi-Relaxation-Time variant |
| `fused_trt_2d.jl`, `fused_bgk_2d.jl` | Fused TRT/BGK | Single-pass kernels (post-DSL) |
| `boundary_2d.jl`, `boundary_rebuild.jl` | `BCSpec` brick set | Modular BC dispatch (ZouHe, BB, periodic, outflow, LI-BB) |
| `li_bb_2d_v2.jl` | LI-BB curved boundary | Lallemand-Luo interpolated bounce-back — used for body-fitted Cd computation |
| `equilibrium_helpers.jl` | `feq_2d`, `feq_3d` | Inlined equilibrium population computation |
| `aa_bgk_2d.jl`, `persistent_bgk_2d.jl` | A-A pattern / persistent kernels | GPU-optimised variants — 4.3× baseline on H100 |
| `drag_gpu.jl` | `compute_drag_gpu!` | GPU-native drag reduction (no host-side per-step transfer) |
| `enzyme_rules.jl` | Enzyme custom rules | Analytic geometry derivatives for AD shape optimisation |

## Critical invariants

- **AoS layout**: populations stored as `f[i, j, q]` (or `[i, j, k, q]` in 3D).
  The SoA layout `f[q, i, j]` was abandoned in 2024 — see project memory
  `feedback_soa_layout`. Code touching `f` MUST follow the AoS convention.
- **Bit-exact reproducibility** on identical seeds + same backend.
- **Periodic BCs preserve mass exactly**; halfway-BB at solid walls
  preserves mass when corners are owned correctly (see project memory
  `project_multiblock_corner_bc_bug`).
- **Equilibrium helpers are `@inline`** for GPU compatibility — never call
  through a function-pointer-like construct in inner loops.

## Cross-module dependencies

Reads from: `lattice` (weights, c_q vectors). Provides to: every other
module that does any physics (`drivers`, `refinement`, `multiblock`,
`curvilinear`, `rheology`).

## Status / scope notes

- DSL is the canonical way to assemble new flow configurations; legacy
  monolithic kernels (e.g. `collide_stream_2d!`) remain for back-compat.
- Multiphase / VOF / phasefield / pressure-VOF kernels are present but
  ship outside v0.1.0 scope.
