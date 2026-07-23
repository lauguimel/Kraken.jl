# Macroscopic fields

These helpers reconstruct the hydrodynamic moments ρ and u (and
pressure p = cs² ρ) from the distribution `f`. The `_forced`
variants include the half-step body-force correction required by
Guo's scheme, and the `_pressure` helper materialises the pressure
field for VTK output. They are the last step of every time loop,
just before I/O.


## Quick reference

| Symbol | Purpose |
|--------|---------|
| `compute_macroscopic_2d!` | ρ, u from f — 2D |
| `compute_macroscopic_3d!` | ρ, u from f — 3D |
| `compute_macroscopic_forced_2d!` | ρ, u from f with body-force correction — 2D |
| `compute_macroscopic_forced_3d!` | ρ, u from f with body-force correction — 3D |
| `compute_macroscopic_pressure_2d!` | Pressure from ρ (cs² ρ) — 2D |

## Details

### `compute_macroscopic_2d!`

**Source:** `src/kernels/macroscopic.jl`

```julia
function compute_macroscopic_2d!(ρ, ux, uy, f; sync=true)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(ρ)
    kernel! = compute_macroscopic_2d_kernel!(backend)
    kernel!(ρ, ux, uy, f; ndrange=(Nx, Ny))
    sync && KernelAbstractions.synchronize(backend)
end
```


### `compute_macroscopic_3d!`

**Source:** `src/kernels/macroscopic.jl`

```julia
function compute_macroscopic_3d!(ρ, ux, uy, uz, f)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny, Nz = size(ρ)
    kernel! = compute_macroscopic_3d_kernel!(backend)
    kernel!(ρ, ux, uy, uz, f; ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end
```


### `compute_macroscopic_forced_2d!`

**Source:** `src/kernels/macroscopic.jl`

```julia
function compute_macroscopic_forced_2d!(ρ, ux, uy, f, Fx, Fy)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(ρ)
    kernel! = compute_macroscopic_forced_2d_kernel!(backend)
    kernel!(ρ, ux, uy, f, Fx, Fy; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
```


### `compute_macroscopic_pressure_2d!`

**Source:** `src/kernels/macroscopic.jl`

```julia
"""
    compute_macroscopic_pressure_2d!(p, ux, uy, f, C, Fx, Fy; ρ_l=1.0, ρ_g=0.001)

Pressure-based macroscopic computation for two-phase flows (He-Chen-Zhang model).

- `p`:  pressure field (p = cs²·Σf)
- `ux, uy`: velocity (momentum / ρ(C), with half-force correction)
- `C`:  VOF field (liquid fraction)
- `Fx, Fy`: total body force (surface tension + axisym correction + gravity)
- `ρ_l, ρ_g`: physical densities of liquid and gas
"""
function compute_macroscopic_pressure_2d!(p, ux, uy, f, C, Fx, Fy;
                                          ρ_l=1.0, ρ_g=0.001)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(p)
    T = eltype(f)
    kernel! = compute_macroscopic_pressure_2d_kernel!(backend)
    kernel!(p, ux, uy, f, C, Fx, Fy, T(ρ_l), T(ρ_g); ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
```


