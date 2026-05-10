# AMR-D Performance Audit - 2026-05-09

## Scope

This note records the first AMR-D runtime pass after the nested-channel debug
audit. The target is still voie D: static conservative-tree AMR, conservative
subcycling, KRK reproducibility, and correctness before production GPU claims.

The performance work here does not change the AMR-D algorithm. It removes
avoidable scalar overhead from the current CPU scheduler so long nested macro
flows are usable as a debug loop.

## Main Fixes

1. Fast-path route packets when `alpha == 1`.
   In this production mode, a route packet is exactly `weight * F[src, q]`.
   The old path rebuilt `rho`, `u`, and `feq` for every packet through
   `reconstructed_integrated_D2Q9_packet`, which dominated nested runs.

2. Reuse the route-packet cache.
   KRK macro-flow runners prepare the cache before stepping. The stream kernel
   now rebuilds it only if the table size changes.

3. Iterate over level row IDs instead of scanning all cells.
   Level copies and inactive parent restriction now use the precomputed
   `active_ids_by_level` / level ID lists from the subcycle state bank.

4. Use scalar row collision kernels in macro flows.
   The hot BGK/Guo collision loops avoid per-row views and generic reductions.

5. Split no-solid streaming from solid-aware streaming.
   Channel flows avoid repeated solid-mask checks in direct and boundary route
   loops.

6. Cache inactive-parent coalescing metadata.
   The inactive refined-parent routes now store `dst` and `kind` alongside the
   packet slot during topology preparation. The hot loop no longer calls the
   route-spec helper or cell-ID lookup for those packets.

7. Compile direct same-level routes into level arrays.
   Channel flows with flat prolongation and no solid mask use compact
   `src/dst/q/weight` arrays instead of traversing route objects in the direct
   streaming hot loop.

8. Compile F2C coalescing routes for the production path.
   For `alpha=1` and `interface_time_scaling=:leaf_equivalent`, the scheduler
   now accumulates precomputed `src/q/weight/parent_slot/packet_slot` arrays.
   Experimental `:level_native` corner logic keeps the checked path.

9. Add guarded CPU threading.
   Collision loops thread over independent active cells, and direct streaming
   threads only on levels where `(dst, q)` is unique. This is a CPU debug-loop
   accelerator; GPU production still needs route-native kernels.

## Local Measurements

All measurements were run locally on the same branch with `Float32` where noted.
They are intended as relative debug-loop numbers, not final publication
benchmarks.

### Central Nested4 Poiseuille

Command class:

```julia
spec = Kraken.create_conservative_tree_nested_channel_spec_2d(4)
Kraken.run_conservative_tree_poiseuille_subcycled_2d(
    max_level=4, spec=spec, steps=steps, Fx=1e-7, omega=1.0,
    route_sampling=:leaf_equivalent, enforce_mass=false)
```

Before this pass:

- active cells: 960
- leaf-equivalent cells: 49152
- 20 steps: `1.0066 s`
- active MLUPS: `0.019`
- leaf-equivalent MLUPS: `0.98`

After this pass:

- active cells: 960
- leaf-equivalent cells: 49152
- 200 steps: `0.2127 s`
- active MLUPS: `0.903`
- leaf-equivalent MLUPS: `46.2`

The gain here is about `35x` on active-cell MLUPS and about `35x` on
leaf-equivalent MLUPS. The dominant fix is the `alpha == 1` packet fast path.

### KRK X-Band Nested4 Debug Case

Command class:

```julia
setup = load_kraken(
    "benchmarks/krk/amr_d_convergence_2d/poiseuille_xband_nested4_debug.krk")
setup.user_vars[:route_sampling] = 0.0
run_conservative_tree_amr_d_case_from_krk_2d(setup; steps_override=200,
                                            T=Float32)
```

Before this pass:

- active cells: 12936
- leaf-equivalent cells: 49152
- 200 steps: `5.549 s`
- active MLUPS: `0.466`
- leaf-equivalent MLUPS: `1.77`

After this pass:

- active cells: 12936
- leaf-equivalent cells: 49152
- 200 steps, 1 Julia thread: `3.115 s`
- active MLUPS: `0.831`
- leaf-equivalent MLUPS: `3.16`
- 200 steps, 8 Julia threads: `2.645 s`
- active MLUPS: `0.978`
- leaf-equivalent MLUPS: `3.72`

The larger KRK case improves by about `1.78x` in one-thread mode and about
`2.10x` with the guarded 8-thread CPU path. This is still far below a dense LBM
kernel; the remaining overhead is the conservative subcycling scheduler itself,
especially direct streaming, C2F application, collision, and buffer copies.

## Validation

Post-patch checks:

```bash
julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_subcycling_2d.jl")'
julia --project=. -e 'using Test; using Kraken; include("test/test_amr_d_krk_validation_2d.jl")'
julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_gpu_pack_2d.jl")'
```

Results:

- subcycling 2D: `669 pass`, `14 broken`
- AMR-D KRK validation 2D: `87 pass`
- GPU route pack 2D: `4984 pass`

## Remaining Bottlenecks

The next performance pass should focus on precomputed route execution arrays:

- boundary route arrays for channel BCs, keeping solid-aware paths separate;
- GPU route-native kernels consuming the same compact arrays.

This keeps the CPU and GPU execution models aligned: build topology once, then
run stable array kernels over precomputed routes.
