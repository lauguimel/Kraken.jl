# Viscoelastic Cylinder Tutorial

This tutorial covers Kraken's **viscoelastic** validation case: Oldroyd-B flow past
a confined cylinder, the standard benchmark for non-Newtonian LBM solvers. It
reports the validated drag results honestly, including where the solver is proven
and where it is not.

The log-conformation finite-volume viscoelastic solver (`_logfv`) **runs on this
branch**: the driver `run_viscoelastic_logfv_cylinder_coupled_2d` is shipped and
dispatched through the `viscoelastic` module, and a coarse smoke `.krk` is included
so you can run the case end-to-end (see [Reproduce](@ref) below). The production
`R = 50` tables were re-validated against rheoTool to **≤ 1 % on `C_d`** (Wi ≤ 1).

!!! note "Coarse smoke vs the production sweep"
    The shipped `.krk` is a *quick smoke* (240×32, R = 8, 20 steps) confirming the
    solver dispatches and returns a `C_d`. It is **not** the converged `R = 50`
    table value; reproducing the validated drag requires the A100-class production
    sweep described under [Reproduce](@ref).

```@contents
Pages = ["users/tutorials/viscoelastic-cylinder.md"]
Depth = 2
```

## The problem

A cylinder held on the axis of a planar channel, with a viscoelastic
**Oldroyd-B** fluid flowing past it. Oldroyd-B is the simplest viscoelastic model:
a Newtonian solvent (viscosity `η_s`) plus a polymer contribution (viscosity `η_p`,
relaxation time `λ`) whose elastic stress is carried by a conformation tensor. The
total viscosity is constant — Oldroyd-B is *not* shear-thinning — so the rich
behaviour comes purely from elasticity.

The dimensionless drivers are:

- the **Weissenberg number** `Wi = λ U / R` (elasticity strength), and
- the **viscosity ratio** `β = η_s / (η_s + η_p)` (solvent fraction).

The reported quantity is the **drag coefficient** `C_d` on the cylinder, compared to
fine-mesh **rheoTool** (OpenFOAM-based, Oldroyd-B log-conformation) reference runs.

## Validated results

### Drag at `Re = 1`, the production resolution `R = 50`

This is the benchmark gate. At cylinder resolution `R = 50` (the production mesh),
Kraken's drag matches the rheoTool fine-mesh references to **≤ 1 %** across the
validated Weissenberg range, with the default half-way bounce-back wall treatment,
in Float64:

| Wi  | rheoTool fine-mesh `C_d` | Kraken `C_d` vs reference |
|-----|--------------------------|---------------------------|
| 0.1 | 130.43                   | ≤ 1 %                     |
| 0.5 | 119.71                   | ≤ 1 %                     |
| 1.0 | 120.40                   | ≤ 1 %                     |

The drag-vs-`Wi` curve shows the expected non-monotone shape — a minimum near
`Wi ≈ 0.5` followed by an upturn — reproducing the qualitative behaviour known for
this benchmark.

### Viscosity-ratio (`β`) cross-validation

Sweeping the solvent fraction `β`, Kraken and rheoTool agree to **≤ 1 % at
β = 0.3** (and at `β = 0.59`). Both codes diverge from each other at `β ≤ 0.1`,
which is therefore a *cross-code-validated* hard regime — a shared wall, not a
Kraken-specific defect.

### Stability is not the same as convergence

Kraken's half-way bounce-back cylinder is **NaN-free up to `Wi = 10`** at both
`R = 30` and `R = 50` — well past the `Wi ≈ 1` ceiling typical of the
LBM-viscoelastic literature, and without the High-Weissenberg-Number-Problem
blow-up that limits many solvers.

!!! warning "High-Wi drag is not converged"
    Stability does not imply accuracy. **Beyond `Wi ≈ 1–2` the high-Wi `C_d` is
    resolution-dependent and is not a validated datapoint.** The relaxation time
    `λ = Wi · R / U` grows very large at high `Wi`, so a fixed 300 000-step run is
    far from steady, and the `R = 30` and `R = 50` resolutions disagree
    increasingly as `Wi` rises. The minimum (`Wi ≈ 0.5`) and the upturn
    (`Wi ≈ 1.5–3`) are qualitatively correct, but the absolute high-Wi drag would
    need a finer mesh and longer runs. **The clean, validated result is `Wi ≤ 1`
    at `R = 50` (≤ 1 %).**

### First normal-stress difference `N1` — an open gap

Kraken logs the maximum first normal-stress difference `N1_max` along the wake (for
example `N1_max = 8.4e-4` at `R = 50`, `Wi = 1`), and the `N1` trend is concordant
with rheoTool (it grows with `Wi` and as `β` decreases in both codes). **However, no
rheoTool `N1` reference value was extracted**, so a quantitative `N1` comparison is
not available. The `N1` leg of the validation is therefore **open**: trends agree,
but the absolute `N1` match is unverified and is not claimed as a pass.

## Reproduce

The case is `benchmarks/krk/viscoelastic/cylinder_oldroyd_b.krk` (a copy also ships
under `benchmarks/results/rheotool_compare/viscoelastic/`). It combines a cylinder
`Obstacle` with an `oldroyd_b` `Rheology` block; the polymer parameters follow the
diffusive scaling `ν_s = β·ν_total`, `ν_p = (1−β)·ν_total`, `λ = Wi · R / U`. See the
[KRK reference `Rheology` block](../krk-reference.md) for the grammar:

```
Rheology oldroyd_b { nu_s = ...  lambda = Wi*R/u_mean }
```

Run the shipped smoke directly on this branch:

```julia
using Kraken
result = run_simulation("benchmarks/results/rheotool_compare/viscoelastic/cylinder_oldroyd_b.krk")
@show result.Cd   # coarse smoke (240×32, R = 8, 20 steps) — NOT the R = 50 table value
```

This is a quick dispatch check (it returns a non-converged `C_d`). The validated
`R = 50` tables (300 000 steps, CUDA Float64) were produced on **Aqua (A100)** via
`bench/viscoelastic_logfv/run_ve_revalidate_r50_halfwaybb.pbs` and are **not**
CI-reproducible — they require an A100-class run.

## Summary

- **Quantitative gate: passed.** `C_d` matches fine-mesh rheoTool to ≤ 1 % at
  `Wi ∈ {0.1, 0.5, 1.0}`, `R = 50`, `Re = 1`, half-way bounce-back, Float64.
- **`β` cross-validated** to ≤ 1 % at `β = 0.3`; shared divergence at `β ≤ 0.1`.
- **Stable (NaN-free) to `Wi = 10`**, but high-Wi `C_d` is resolution-divergent and
  unconverged — validated only for `Wi ≤ 1` (the qualitative min+upturn is
  reproduced).
- **`N1` gap:** Kraken logs `N1_max` and the trend agrees with rheoTool, but no
  rheoTool `N1` reference was extracted, so the absolute `N1` match is unverified.
- **Solver runs on this branch.** The `_logfv` cut-link driver is shipped and
  dispatched via the `viscoelastic` module; the included `.krk` runs as a coarse
  smoke. The ≤ 1 % `C_d` tables are the A100 production sweep, not the local smoke.

## Where to go next

- The [KRK reference `Rheology` block](../krk-reference.md) — the
  viscoelastic and non-Newtonian constitutive grammar.
- The Newtonian [lid-driven cavity example](../../examples/04_cavity_2d.md) and
  [Cartesian cavity benchmark](../benchmarks/cartesian-cavity.md) — the validated
  baseline the viscoelastic path builds on.
