# Getting Started

This tutorial walks you through installing Kraken.jl and running your first
fluid dynamics simulation in under 5 minutes.

## Install Julia

The recommended way to install Julia is via [juliaup](https://github.com/JuliaLang/juliaup):

```bash
curl -fsSL https://install.julialang.org | sh
```

Verify the installation:

```bash
julia --version
```

## Install Kraken.jl

From the Julia REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/lauguimel/Kraken.jl.git")
```

## Run your first simulation

Let's solve the classic **lid-driven cavity** at Re = 100 on a 64×64 grid:

```julia
using Kraken

u, v, p, converged = run_cavity(N=64, Re=100.0, verbose=true)
println("Converged: $converged")
```

The function returns the velocity components `u`, `v`, the pressure field `p`,
and a boolean indicating whether the solver reached the prescribed tolerance.

## Visualize with VTK

Kraken.jl can write [VTK](https://vtk.org/) files that you can open in
[ParaView](https://www.paraview.org/):

```julia
dx = 1.0 / 63
write_vtk("cavity_result", 64, 64, dx, Dict(
    "velocity_x" => u,
    "velocity_y" => v,
    "pressure"   => p,
))
# Open cavity_result.vtr in ParaView
```

## GPU acceleration

If you have an Apple Silicon Mac with [Metal.jl](https://github.com/JuliaGPU/Metal.jl),
you can run the same simulation on the GPU:

```julia
using Metal, KernelAbstractions

u, v, p, converged = run_cavity(
    N=64, Re=100.0,
    backend=MetalBackend(),
    float_type=Float32,
    verbose=true,
)
```

On NVIDIA hardware, use `CUDA.jl` and `CUDABackend()` instead.

## Check available backends

To see which compute backends are available on your system:

```julia
println(available_backends())
```

This returns a list of backends (CPU is always available; GPU backends appear
when the corresponding packages are loaded).

## Next steps

- [Running from YAML Configuration](@ref) — set up simulations via config files
- [Composing Your Own Solver](@ref) — use Kraken's operators as building blocks
- [API Reference](@ref) — full function documentation
