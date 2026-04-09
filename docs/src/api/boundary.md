# Boundary conditions

Boundary conditions come in three families: **Zou-He** (velocity or
pressure Dirichlet), **bounce-back** (no-slip walls), and
**thermal** Dirichlet for the advection-diffusion DDF. Each wall
has its own kernel so the driver code stays branch-free on the
interior. The `*_spatial_2d!` variants accept per-cell profile
fields for inlet conditions parsed from a `.krk` expression.


## Quick reference

| Symbol | Purpose |
|--------|---------|
| `apply_zou_he_north_2d!` | Zou-He velocity BC on one wall — 2D |
| `apply_zou_he_south_2d!` | Zou-He velocity BC on one wall — 2D |
| `apply_zou_he_west_2d!` | Zou-He velocity BC on one wall — 2D |
| `apply_zou_he_pressure_east_2d!` | Zou-He pressure outlet (east) — 2D |
| `apply_extrapolate_east_2d!` | Zeroth-order extrapolation outlet (east) — 2D |
| `apply_zou_he_top_3d!` | Zou-He velocity BC on one wall — 3D |
| `apply_zou_he_bottom_3d!` | Zou-He velocity BC on one wall — 3D |
| `apply_zou_he_west_3d!` | Zou-He velocity BC on one wall — 3D |
| `apply_zou_he_east_3d!` | Zou-He velocity BC on one wall — 3D |
| `apply_zou_he_south_3d!` | Zou-He velocity BC on one wall — 3D |
| `apply_zou_he_north_3d!` | Zou-He velocity BC on one wall — 3D |
| `apply_zou_he_pressure_east_3d!` | Zou-He pressure outlet (east) — 3D |
| `apply_zou_he_pressure_top_3d!` | Zou-He pressure outlet (top) — 3D |
| `apply_bounce_back_walls_3d!` | No-slip bounce-back on all walls — 3D |
| `apply_bounce_back_walls_2d!` | No-slip bounce-back on all walls — 2D |
| `apply_bounce_back_wall_2d!` | No-slip bounce-back on a single wall — 2D |
| `apply_fixed_temp_south_2d!` | Fixed-temperature (Dirichlet) on south wall — thermal |
| `apply_fixed_temp_north_2d!` | Fixed-temperature (Dirichlet) on north wall — thermal |
| `apply_fixed_temp_west_2d!` | Fixed-temperature (Dirichlet) on west wall — thermal |
| `apply_fixed_temp_east_2d!` | Fixed-temperature (Dirichlet) on east wall — thermal |
| `apply_zou_he_north_spatial_2d!` | Zou-He velocity BC with spatial profile (north) |
| `apply_zou_he_south_spatial_2d!` | Zou-He velocity BC with spatial profile (south) |
| `apply_zou_he_west_spatial_2d!` | Zou-He velocity BC with spatial profile (west) |
| `apply_zou_he_pressure_east_spatial_2d!` | Zou-He pressure BC with spatial profile (east) |
| `apply_zou_he_pressure_inlet_west_2d!` | Zou-He pressure inlet (west) — 2D |

## Details

### `apply_zou_he_north_2d!`

**Source:** `src/kernels/boundary_2d.jl`

```julia
function apply_zou_he_north_2d!(f, u_wall_x, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_velocity_north_2d_kernel!(backend)
    kernel!(f, eltype(f)(u_wall_x), Ny; ndrange=(Nx,))
    KernelAbstractions.synchronize(backend)
end
```


### `apply_bounce_back_walls_2d!`

**Source:** `src/kernels/boundary_2d.jl`

```julia
"""
    apply_bounce_back_walls_2d!(f, Nx, Ny)

Apply simple bounce-back on south, west, and east walls of a 2D cavity.
North wall is handled by Zou-He (lid velocity).
"""
function apply_bounce_back_walls_2d!(f, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = bounce_back_walls_2d_kernel!(backend)

    # South wall (j=1)
    kernel!(f, Nx, Ny, Int32(1); ndrange=(Nx,))
    # West wall (i=1)
    kernel!(f, Nx, Ny, Int32(2); ndrange=(Ny,))
    # East wall (i=Nx)
    kernel!(f, Nx, Ny, Int32(3); ndrange=(Ny,))

    KernelAbstractions.synchronize(backend)
end
```


### `apply_zou_he_west_3d!`

**Source:** `src/kernels/boundary_3d.jl`

```julia
"""
    apply_zou_he_west_3d!(f, ux_w, uy_w, uz_w, Ny, Nz)

Apply Zou-He velocity BC on the west face (i = 1).
Unknown populations (cx=+1): f2(+x), f8(+x,+y), f10(+x,-y), f12(+x,+z), f14(+x,-z).
"""
function apply_zou_he_west_3d!(f, ux_w, uy_w, uz_w, Ny, Nz)
    # West: normal=x, sign=-1, u_normal=ux_w, tang1=y => uy_w, tang2=z => uz_w
    _apply_zou_he_velocity_3d!(f, ZH_WEST, 1, ux_w, uy_w, uz_w, Ny, Nz)
end
```


### `apply_fixed_temp_north_2d!`

**Source:** `src/kernels/thermal_2d.jl`

```julia
function apply_fixed_temp_north_2d!(g, T_wall, Nx, Ny)
    backend = KernelAbstractions.get_backend(g)
    kernel! = apply_fixed_temp_north_2d_kernel!(backend)
    kernel!(g, eltype(g)(T_wall), Ny; ndrange=(Nx,))
    KernelAbstractions.synchronize(backend)
end
```


### `apply_zou_he_north_spatial_2d!`

**Source:** `src/kernels/boundary_spatial_2d.jl`

```julia
"""
    apply_zou_he_north_spatial_2d!(f, ux_arr, uy_arr, Nx, Ny)

Zou-He velocity BC on north wall with per-node velocity arrays.
"""
function apply_zou_he_north_spatial_2d!(f, ux_arr, uy_arr, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_velocity_north_spatial_2d_kernel!(backend)
    kernel!(f, ux_arr, uy_arr, Ny; ndrange=(Nx,))
    KernelAbstractions.synchronize(backend)
end
```


