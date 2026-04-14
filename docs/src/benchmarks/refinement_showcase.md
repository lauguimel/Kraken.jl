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

We ran the De Vahl Davis (1983) natural-convection cavity with and
without wall refinement on Metal (Apple M3 Max, Float32). Reference
values are `Nu = 1.118` at `Ra = 10³` and `Nu = 4.519` at `Ra = 10⁵`.

| Ra | Mode | N_base | ratio | wall_fraction | Nu | error | wall-time |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10³ | uniform | 128 | — | — | 1.129 | 0.94 % | 124 s |
| 10³ | refined | 64  | 2  | 0.20 | 1.188 | 6.26 % | 1595 s |
| 10⁵ | uniform | 192 | — | — | 4.646 | 2.81 % | 297 s |
| 10⁵ | refined | 96  | 2  | 0.15 | 5.460 | 20.82 % | 3695 s |

At both Ra, refinement produces a **larger** Nu error than a uniform
grid at the same *near-wall* resolution, and takes **10–12 × more
wall-time**. This is not just failure to converge further: the refined
runs show a **systematic over-prediction** of Nu that does not
disappear at longer integration times.

## Root causes

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

## Where refinement should win (planned for v0.2.0)

1. **Cylinder Cd with FH-rescaled MEA.** Momentum-exchange drag on a
   refined patch around a cylinder requires rescaling the non-equilibrium
   populations before accumulating the boundary-link sum (Lagrava et
   al. 2012). Scaffolding is in
   [`benchmarks/convergence_cylinder_refinement.jl`](https://github.com/lauguimel/Kraken.jl/blob/main/benchmarks/convergence_cylinder_refinement.jl),
   with diagnostic hooks `patch_diag_fns` and `coarse_diag_fn` already
   added to `advance_refined_step!`.
2. **FH-rescaled thermal populations.** Apply the same rescaling to
   thermal f_neq as is used for flow f_neq; this should remove the
   systematic Nu bias.
3. **Localised-feature benchmarks** where the bulk is trivial:
   impinging jet on a heated plate, shear layer, contraction.
4. **Adaptive refinement** (wavelet error estimator) rather than
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
