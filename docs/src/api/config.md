# Configuration and initialisation

`LBMConfig` is the immutable configuration struct threaded through
every driver — it holds the grid size, relaxation rate, body force,
and output cadence. The `omega` and `reynolds` helpers convert
between lattice-unit viscosity and Reynolds number, and
`initialize_2d` / `initialize_3d` allocate the distribution and
macroscopic arrays for a given config.


## Quick reference

| Symbol | Purpose |
|--------|---------|
| `LBMConfig` | Immutable run configuration (Nx, Ny, τ, force, …) |
| `omega` | Relaxation rate ω from kinematic viscosity |
| `reynolds` | Reynolds number from characteristic scales |
| `initialize_2d` | Allocate and initialise 2D distribution + macros |
| `initialize_3d` | Allocate and initialise 3D distribution + macros |

## Details

### `LBMConfig`

**Source:** `src/drivers/basic.jl`

```julia
"""
    LBMConfig{L <: AbstractLattice}

Configuration for an LBM simulation.
"""
struct LBMConfig{L <: AbstractLattice}
    lattice::L
    Nx::Int
    Ny::Int
    Nz::Int          # 1 for 2D
    ν::Float64       # kinematic viscosity (lattice units)
    u_lid::Float64   # lid velocity (cavity)
    max_steps::Int
    output_interval::Int
end
```


### `omega`

**Source:** `src/drivers/basic.jl`

```julia
"""
    omega(config::LBMConfig) -> Float64

Compute BGK relaxation parameter from viscosity: ω = 1 / (3ν + 0.5).
"""
omega(config::LBMConfig) = 1.0 / (3.0 * config.ν + 0.5)
```


### `reynolds`

**Source:** `src/drivers/basic.jl`

```julia
"""
    reynolds(config::LBMConfig) -> Float64

Effective Reynolds number: Re = u_lid · N / ν.
Uses Ny for 2D, Nz for 3D (cavity height).
"""
function reynolds(config::LBMConfig)
    L = lattice_dim(config.lattice) == 2 ? config.Ny : config.Nz
    return config.u_lid * L / config.ν
end
```


### `initialize_2d`

**Source:** `src/drivers/basic.jl`

```julia
"""
    initialize_2d(config::LBMConfig{D2Q9}, T=Float64; backend=CPU())

Create initial LBM state for 2D simulation. Populations set to equilibrium
with ρ=1, u=0.
"""
function initialize_2d(config::LBMConfig{D2Q9}, ::Type{T}=Float64;
                        backend=KernelAbstractions.CPU()) where T
    Nx, Ny = config.Nx, config.Ny

    # Allocate on the desired backend
    f_in  = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    f_out = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    ρ     = KernelAbstractions.ones(backend, T, Nx, Ny)
    ux    = KernelAbstractions.zeros(backend, T, Nx, Ny)
    uy    = KernelAbstractions.zeros(backend, T, Nx, Ny)
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny)

    # Initialize to equilibrium (ρ=1, u=0 → f_eq = w_q)
    w = weights(D2Q9())
    f_cpu = zeros(T, Nx, Ny, 9)
    for q in 1:9
        f_cpu[:, :, q] .= T(w[q])
    end
    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)

    return (f_in=f_in, f_out=f_out, ρ=ρ, ux=ux, uy=uy, is_solid=is_solid)
end
```


### `initialize_3d`

**Source:** `src/drivers/basic.jl`

```julia
"""
    initialize_3d(config::LBMConfig{D3Q19}, T=Float64; backend=CPU())

Create initial LBM state for 3D simulation.
"""
function initialize_3d(config::LBMConfig{D3Q19}, ::Type{T}=Float64;
                        backend=KernelAbstractions.CPU()) where T
    Nx, Ny, Nz = config.Nx, config.Ny, config.Nz

    f_in  = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz, 19)
    f_out = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz, 19)
    ρ     = KernelAbstractions.ones(backend, T, Nx, Ny, Nz)
    ux    = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz)
    uy    = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz)
    uz    = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz)
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny, Nz)

    w = weights(D3Q19())
    f_cpu = zeros(T, Nx, Ny, Nz, 19)
    for q in 1:19
        f_cpu[:, :, :, q] .= T(w[q])
    end
    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)

    return (f_in=f_in, f_out=f_out, ρ=ρ, ux=ux, uy=uy, uz=uz, is_solid=is_solid)
end
```


