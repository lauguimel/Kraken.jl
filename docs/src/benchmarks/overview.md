# Benchmarks

Kraken.jl is validated against a suite of classical CFD benchmarks with known analytical or reference solutions. Each benchmark tests specific operators and solver components.

## Summary

| Benchmark | Physics tested | Reference solution | Grid sizes | Status |
|-----------|---------------|-------------------|------------|--------|
| [Heat Diffusion](heat-diffusion.md) | Diffusion (`laplacian!`) | Analytical | 32, 64, 128 | Validated |
| [Lid-Driven Cavity](lid-cavity.md) | Full NS (projection) | Ghia et al. 1982 | 64 | Validated |
| [Taylor-Green Vortex](taylor-green.md) | NS periodic (advection + diffusion + pressure) | Analytical | 32, 64, 128 | Validated |
| [Poiseuille Flow](poiseuille.md) | Diffusion + body force | Analytical (parabolic) | 64 | Validated |
| [Couette Flow](couette.md) | Diffusion + wall BCs | Analytical (linear) | 64 | Validated |
| [Rotation Advection](rotation-advection.md) | Pure advection (`advect!`) | IC recovery after 1 rotation | 128 | Validated |
| [Advection-Diffusion](advection-diffusion.md) | Advection + diffusion | Analytical (Gaussian) | 128 | Validated |

## Running the benchmarks

All benchmarks can be run from the `benchmarks/` directory:

```julia
include("benchmarks/run_all.jl")
```

Each benchmark function accepts `save_figures=true` to generate validation plots in `docs/src/assets/figures/`.

## What is validated

- **Spatial operators**: [`laplacian!`](@ref), [`gradient!`](@ref), [`divergence!`](@ref), [`advect!`](@ref)
- **Solvers**: [`solve_poisson_fft!`](@ref), [`solve_poisson_cg!`](@ref), [`projection_step!`](@ref)
- **Convergence rates**: O(h^2) for diffusion, O(h) for upwind advection
- **Steady-state accuracy**: Poiseuille and Couette flows match exact profiles
- **GPU acceleration**: Metal backend timing included where applicable
