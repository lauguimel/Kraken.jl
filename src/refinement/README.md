# refinement/

Adaptive Mesh Refinement (AMR-D voie) via patch-based nested conservative
tree with Filippova–Hänel ω rescaling. This is the largest module in
Kraken — about 20 files — and the most physics-sensitive: see project
memory `project_voie_d_*` for the open research questions.

## Key entry points

| File | Symbol | Purpose |
|---|---|---|
| `refinement.jl` | `RefinementPatch`, `apply_refinement!` | Top-level patch struct + driver call |
| `refinement_3d.jl` | `RefinementPatch3D`, 3D analogue | 3D variant |
| `conservative_tree_streaming_2d.jl` | `run_conservative_tree_*_2d` | Production AMR-D streamer (called from `benchmarks/`) |
| `conservative_tree_macroflows_subcycled_2d.jl` | macroflow loop | Sub-cycled time stepping with multiple `route_sampling` modes |
| `conservative_tree_gpu_pack_2d.jl` | c2f deposit / injection kernels | **Fires in Metal mode** despite the "gpu" name |
| `conservative_tree_routes_2d.jl` | route table builder | Static c2f / f2c route weights |
| `thermal_refinement.jl` | thermal AMR variant | DDF-coupled AMR |

## Critical invariants

- **ν preservation across c/f**: `τ_f = ratio · (τ_c − 0.5) + 0.5`
  (Filippova–Hänel); ν_phys must be invariant under the rescaling.
- **Mass drift `O(h^d)`** on Form A (this module's class): bit-id
  Cartesian on uniform / nested-1 channel flows
  (mass drift ≈ 2.5e-13 at `2d27bf68`), but `Σ_global · ρu ≤ ε_machine`
  is **algebraically unattainable** for cell-centered restriction-overwrite
  AMR-LBM (see project memory `project_amr_strategy_pivot`).
- **Mode discrimination**: 3 `route_sampling` modes
  (`:leaf_equivalent` production = 0, `:level_native` debug = 1,
  `:subcycled_hybrid` experimental = 2); 2 c2f modes (`:flat`, `:limited_linear`).
  `_apply_level_native_route_closure_2d!` is dead code under the
  default `:leaf_equivalent`.

## Cross-module dependencies

Reads from: `lattice` (weights for FH rescaling), `kernels`
(collide / stream / BCSpec), `io` (`.krk` `Refine{}` block parsing).
Provides to: `drivers` (via `run_conservative_tree_*_2d`), benchmarks.

## Status / scope notes

- Production code path = `:leaf_equivalent` + `c2f_prolongation=:flat`.
- Fixtures with `_debug.krk` suffix typically opt into `:level_native`
  for diagnostic purposes — see `kraken-codebase-map` skill for full
  discriminator tables.
- Voie D scope-strict ship in v0.2: AMR-D channel flows nested-1 ✓,
  uniform cylinder Re=20 ✓, **NO wall in fine zone** (nested-4 wall
  yband leak 22-45%); AMR-D cylinder Cd quant deferred to v0.3 or
  pivot to OpenLB Lagrava port (Kraken-AMR.jl spinoff).
