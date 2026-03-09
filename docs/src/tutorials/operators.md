# Composing Your Own Solver

Kraken.jl is built around **composable operators** — small, GPU-ready functions
that you can combine to create custom solvers. This tutorial shows how to use
them directly.

## The operators

Kraken exports four core spatial operators. They all work on 2D arrays and take
a grid spacing `dx`:

```julia
using Kraken

N = 64
dx = 1.0 / (N - 1)
f = zeros(N, N)

# Laplacian: ∇²f
out = zeros(N, N)
laplacian!(out, f, dx)

# Gradient: ∇p → (∂p/∂x, ∂p/∂y)
p = zeros(N, N)
gx, gy = zeros(N, N), zeros(N, N)
gradient!(gx, gy, p, dx)

# Divergence: ∇·u
u, v = zeros(N, N), zeros(N, N)
d = zeros(N, N)
divergence!(d, u, v, dx)

# Advection: (u·∇)φ
phi = zeros(N, N)
advect!(out, u, v, phi, dx)
```

Each operator writes its result into the first argument (in-place), so there
are no allocations during a time loop.

## Building a heat equation solver

Let's solve the 2D heat equation ``\frac{\partial T}{\partial t} = \kappa \nabla^2 T``
using just the [`laplacian!`](@ref) operator:

```julia
using Kraken

N = 64
dx = 1.0 / (N - 1)
xs = range(0, 1, length=N)
ys = range(0, 1, length=N)

# Initial condition: a sine bump
T = [sin(π * x) * sin(π * y) for x in xs, y in ys]
lap = similar(T)

κ = 0.01
dt = 0.0001

for step in 1:1000
    laplacian!(lap, T, dx)
    T[2:end-1, 2:end-1] .+= dt * κ .* lap[2:end-1, 2:end-1]
    # Dirichlet BCs (T=0 on all walls)
    T[:, 1] .= 0; T[:, end] .= 0
    T[1, :] .= 0; T[end, :] .= 0
end
```

The temperature decays exponentially toward zero — exactly matching the
analytical solution ``T(t) = e^{-2\pi^2 \kappa t} \sin(\pi x)\sin(\pi y)``.

## Building a custom Navier-Stokes solver

Now let's compose all operators into a minimal incompressible NS solver.
We'll use Taylor-Green vortex initial conditions on a periodic domain:

```julia
using Kraken

N = 64
dx = 1.0 / (N - 1)
dt = 0.001
ν = 0.01  # kinematic viscosity

# Taylor-Green initial condition
xs = range(0, 1, length=N)
ys = range(0, 1, length=N)
u = [sin(2π * x) * cos(2π * y) for x in xs, y in ys]
v = [-cos(2π * x) * sin(2π * y) for x in xs, y in ys]
p = zeros(N, N)

# Scratch arrays
adv_u, adv_v = similar(u), similar(v)
lap_u, lap_v = similar(u), similar(v)
gx, gy = similar(u), similar(v)
d, rhs = similar(u), similar(v)

for step in 1:500
    # 1. Advection + diffusion
    advect!(adv_u, u, v, u, dx)
    advect!(adv_v, u, v, v, dx)
    laplacian!(lap_u, u, dx)
    laplacian!(lap_v, v, dx)
    u .+= dt .* (-adv_u .+ ν .* lap_u)
    v .+= dt .* (-adv_v .+ ν .* lap_v)

    # 2. Pressure projection
    divergence!(d, u, v, dx)
    rhs .= d ./ dt
    solve_poisson_fft!(p, rhs, dx)
    gradient!(gx, gy, p, dx)
    u .-= dt .* gx
    v .-= dt .* gy
end
```

This is essentially the same projection method that [`run_cavity`](@ref) uses
internally, but here you have full control over every step.

## GPU: same code, different arrays

The beauty of KernelAbstractions.jl is that **the same operators work on GPU
arrays** with zero code changes. Just allocate on the right backend:

```julia
using Metal, KernelAbstractions

backend = MetalBackend()
N = 64
dx = Float32(1.0 / (N - 1))

T_gpu = KernelAbstractions.zeros(backend, Float32, N, N)
lap_gpu = KernelAbstractions.zeros(backend, Float32, N, N)

# Same operator, now running on the GPU
laplacian!(lap_gpu, T_gpu, dx)
```

Replace `Metal`/`MetalBackend()` with `CUDA`/`CUDABackend()` on NVIDIA
hardware. The operator code is identical.

## Next steps

- [Getting Started](@ref) — install and run your first simulation
- [Projection Method](@ref) — theory behind the pressure-velocity coupling
- [API Reference](@ref) — full function documentation
