# Kraken.jl

**GPU-native Lattice Boltzmann Method (LBM) framework in Julia.**

Kraken.jl provides a composable, high-performance LBM solver for incompressible
flows with automatic GPU acceleration via
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

## Key Features

- **Multi-backend GPU**: write once, run on CUDA, Metal (Apple Silicon), AMD ROCm, and CPU
- **D2Q9 & D3Q19 lattices**: standard lattice Boltzmann velocity sets
- **BGK collision**: single-relaxation-time with Guo forcing scheme
- **Boundary conditions**: bounce-back, Zou-He velocity/pressure, periodic
- **Thermal LBM**: double distribution function with Boussinesq coupling
- **Axisymmetric LBM**: cylindrical coordinates via Li et al. (2010) scheme
- **Momentum exchange**: drag/lift computation for immersed bodies
- **VTK output**: write fields to `.vti` / `.pvd` for ParaView visualization

## Quick Start

```julia
using Kraken

# Lid-driven cavity at Re = 100 on a 128×128 grid
N = 128
ν = 0.1 * N / 100  # ν = u_lid · N / Re
config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=ν, u_lid=0.1, max_steps=30000)
result = run_cavity_2d(config)
```

## Documentation

- **[Installation](@ref)** — set up Kraken.jl and GPU backends
- **[Theory](@ref "LBM Fundamentals")** — from kinetic theory to lattice Boltzmann (10 progressive chapters)
- **[Examples](@ref "Poiseuille Flow (2D)")** — validated simulations with plots and convergence studies
- **[Benchmarks](@ref "Performance: MLUPs")** — performance and accuracy measurements
- **[API Reference](@ref)** — complete function reference

## Physics Capabilities

| Capability | Lattice | Driver |
|:-----------|:--------|:-------|
| Lid-driven cavity | D2Q9, D3Q19 | `run_cavity_2d`, `run_cavity_3d` |
| Channel flow (Poiseuille) | D2Q9 | `run_poiseuille_2d` |
| Couette flow | D2Q9 | `run_couette_2d` |
| Taylor-Green vortex | D2Q9 | `run_taylor_green_2d` |
| Cylinder drag | D2Q9 | `run_cylinder_2d` |
| Thermal convection | D2Q9 | `run_rayleigh_benard_2d` |
| Axisymmetric pipe flow | D2Q9 | `run_hagen_poiseuille_2d` |
