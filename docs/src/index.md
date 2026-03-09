# Kraken.jl

**GPU-native multi-physics CFD framework in Julia.**

Kraken.jl provides composable operators for computational fluid dynamics simulations with automatic GPU acceleration via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

## Key Features

- **Multi-backend GPU**: write once, run on CUDA, Metal (Apple Silicon), AMD ROCm, and CPU — powered by KernelAbstractions.jl
- **Composable operators**: `laplacian!`, `gradient!`, `divergence!`, `advect!` — mix and match to build your PDE solver
- **Projection method**: fractional-step incompressible Navier-Stokes solver
- **FFT & CG solvers**: Poisson pressure solve via FFT (periodic/Dirichlet) or conjugate gradient (general BCs)
- **VTK output**: write fields to `.vti` / `.pvd` files for visualization in ParaView
- **YAML configuration**: define simulations in a single YAML file

## Quick Start

```julia
using Kraken

# Run lid-driven cavity benchmark (64×64, Re=100)
run_cavity(; N=64, Re=100.0, dt=0.001, nsteps=5000)
```

## Documentation

- **[Installation](@ref)** — how to install Kraken.jl and set up GPU backends
- **[Theory](@ref "Governing Equations")** — mathematical formulation and numerical methods
- **[Benchmarks](@ref "Benchmarks")** — validation against reference solutions
- **[Tutorials](@ref "Getting Started")** — step-by-step guides
- **[API Reference](@ref)** — complete function reference
