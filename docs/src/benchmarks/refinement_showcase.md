# Refinement showcase — honest status (v0.1.0)

This page reports where patch-based grid refinement stands in
Kraken.jl v0.1.0. **Refinement is validated for stability**: the
Filippova–Hänel (FH) rescaling, temporal sub-cycling and restriction all
work across CPU, CUDA and Metal backends, and the 2D tests pass. A
**cost-vs-accuracy win on integrated quantities** (Nu for natural
convection, Cd for cylinder flow) is **not yet demonstrated** and is
tracked as a v0.2.0 item.

## What works in v0.1.0

- `Refine { region = [...], ratio = r }` blocks in `.krk` files
  automatically route to the patch runner; no custom driver is needed.
- **Two symmetric wall patches** are instantiated in the natural
  convection driver (`run_natural_convection_refined_2d`): one at the
  hot wall and one at the cold wall, each with physical width
  `wall_fraction · L` and the full vertical extent of the cavity.
- Patch ghost cells use FH f_neq rescaling for flow and
  bilinear/temporal interpolation for temperature; restriction is a
  block-average of ratio² fine cells back to each coarse cell.
- 2D cavity flow with central patch, 2D natural convection with wall
  patches and 3D refined conduction all run stably on CPU and GPU.

## What does not yet showcase a win

Results on Metal (Apple M3 Max, Float32) at Ra = 10³ with
wall_fraction = 0.20, ratio = 2, after the GPU-sync fix (see below).

| Mode | N_base | Nu | error | wall-time | steps/s |
|---|---:|---:|---:|---:|---:|
| uniform | 64  | 1.1419 | 2.13 % | 6.0 s   | 10 182 |
| uniform | 128 | 1.1285 | 0.94 % | 25.6 s  |  9 601 |
| refined | 64  | 1.1880 | 6.26 % | 506 s   |    121 |
| refined | 96  | 1.2161 | 8.77 % | 1 202 s |    115 |
| refined | 128 | 1.2086 | 8.10 % | 2 375 s |    103 |

Two independent problems stand out:

1. **Throughput gap**: even with the sync fix, refined runs at
   ~110 steps/s while uniform runs at ~10 000 steps/s — a 90 × gap due
   to many small kernel launches per coarse step (ghost fills, stream,
   BC, collide, macro, restrict, copyto!s) versus one fused kernel for
   the uniform case.
2. **Accuracy plateau with a positive bias**: the refined error does
   not improve as `N_base` grows; it actually gets worse (6.26 → 8.77 %)
   before stabilising near 8 %. This is a zeroth-order bias at the
   patch interface, not a convergence problem.

## Perf bug fixed (2026-04-14)

Until commit `<sync-fix>` every kernel wrapper ended with
`KernelAbstractions.synchronize(backend)`. That is fine when the CPU
driver launches one large kernel per time step (the uniform case: one
`fused_natconv_step!`), but it is disastrous inside the sub-cycled
refined step where ~50 wrappers are called per coarse step — each
blocking the CPU on GPU completion.

The fix removed the unconditional `KernelAbstractions.synchronize`
calls from 134 kernel wrappers in `src/kernels/` and `src/refinement/`,
and flipped the default of the opt-in `sync` keyword on
`stream_2d!`, `collide_2d!` and `compute_macroscopic_2d!` to `false`.
KernelAbstractions already serialises kernels submitted on the same
stream, so intermediate syncs are redundant for GPU-only pipelines;
the only natural sync point is the final `Array(...)` copy back to
host in the top-level driver.

Measured impact on Metal Float32:

| Mode | N_base | Before | After | Speedup |
|---|---:|---:|---:|---:|
| uniform | 64  |   ? | 10 182 sps | — |
| uniform | 128 | 1 984 sps | 9 601 sps | 4.8 × |
| refined | 64  |    40 sps |   121 sps | 3.0 × |

The uniform case ran 128 ²=  16 384 cells in one fused kernel so
each step was already launch-bound; removing the per-step sync
converts wall-time almost entirely to compute. The refined case
still has many small launches, so the speedup is smaller but real.

## Root causes (accuracy)

1. **Thermal ghost fill uses no FH rescaling.** The thermal patch
   interpolates coarse temperature populations bilinearly in space and
   linearly in time, without non-equilibrium rescaling (see
   `src/refinement/thermal_refinement.jl`). This breaks strict
   conservation of the heat flux across the patch interface.
2. **Boussinesq feedback at the patch boundary.** The buoyancy force
   depends on the local temperature, so any interpolation error in the
   ghost cells feeds back into the flow, which in turn biases the
   temperature gradient at the hot wall — i.e. exactly the quantity
   that sets Nu.
3. **Bulk runs coarse.** With `wall_fraction = 0.2` and `ratio = 2`,
   only 20 % of the domain near each wall is refined; the central
   recirculation is at the base resolution. For Ra = 10⁵ the core
   contributes as much to the Nu budget as the boundary layer, so
   refining only the walls cannot reach the accuracy of a uniform
   fine grid.
4. **Per-step cost is dominated by the patch.** Each coarse step
   triggers `ratio` fine sub-steps on each patch. At `ratio = 2` and
   two wall patches covering 20 % of the domain, the sub-cycled patch
   work is `2 · 0.2 · L² · ratio = 0.8 · L²` per coarse step, on top
   of `L²` for the bulk. The patch contribution dominates the 10 ×
   slowdown observed.

## Design constraint (v0.1.0)

Refinement is **generic**: it must work for any `.krk` setup, not be
hand-coded per benchmark. The user's BCs reach the patch sub-step as
`bc_*_patch_fns` closures, which cannot be marshalled into a GPU kernel.
That is why the patch sub-step currently issues many small kernels
(~8–10 per sub-step) instead of a single fused kernel: the closure
pattern preserves genericity at the cost of launch overhead.

A case-specific fused kernel (e.g. `fused_natconv_patch_step!`) was
prototyped and measured on Metal M3 Max: it recovered ~1.2× over the
unfused path for Ra = 10³ natconv, but locked the refinement code to a
specific BC template. The prototype was rejected for v0.1.0 and the
generic closure-based path kept.

## Where refinement should win (planned for v0.2.0)

1. **Compile-time BC tags** instead of closures, so a single templated
   kernel can fuse stream + BC + macro + collide while remaining generic.
2. **FH-rescaled thermal populations.** The FH factor that works for
   flow (α = ratio) *overshoots* for the passive scalar on De Vahl Davis
   (measured Nu = 0.6 vs ref 1.118). The correct scaling for thermal
   passive scalar remains open.
3. **Cylinder Cd with FH-rescaled MEA.** Momentum-exchange drag on a
   refined patch around a cylinder requires rescaling the non-equilibrium
   populations before accumulating the boundary-link sum (Lagrava et
   al. 2012). Scaffolding is in
   [`benchmarks/convergence_cylinder_refinement.jl`](https://github.com/lauguimel/Kraken.jl/blob/main/benchmarks/convergence_cylinder_refinement.jl),
   with diagnostic hooks `patch_diag_fns` and `coarse_diag_fn` already
   added to `advance_refined_step!`.
4. **Localised-feature benchmarks** where the bulk is trivial:
   impinging jet on a heated plate, shear layer, contraction.
5. **Adaptive refinement** (wavelet error estimator) rather than
   user-specified patches — the Basilisk-style approach that gives the
   large speed-ups observed in the AMR literature.

## How to reproduce

```bash
# Run the sweep (CPU by default; GPU via backend kwarg).
julia --project benchmarks/convergence_natconv_refinement.jl
```

The script writes a CSV to `benchmarks/results/` and prints a table
with Nu, error, wall-time and `cell·steps` for each configuration. The
cylinder scaffolding lives in
[`convergence_cylinder_refinement.jl`](https://github.com/lauguimel/Kraken.jl/blob/main/benchmarks/convergence_cylinder_refinement.jl)
and documents the FH-force-rescaling TODO inline.

## Cross-reference

- Theory: [Grid refinement](../theory/18_grid_refinement.md)
- Driver: `run_natural_convection_refined_2d`
  ([`drivers/thermal.jl`](https://github.com/lauguimel/Kraken.jl/blob/main/src/drivers/thermal.jl))
- Sanity check: the `_check_thermal!` routine warns when
  `N_eff < 3·Ra^(1/4)` (see [Sanity](../krk/sanity.md)).
- Capabilities status: [Capabilities §4](../capabilities.md).
