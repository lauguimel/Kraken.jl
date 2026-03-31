using KernelAbstractions

# ===========================================================================
# Prescribed-velocity advection helpers for VOF/CLSVOF
#
# Used for pure advection tests (Zalesak disk, reversed vortex, shear)
# where the velocity field is analytical — no LBM solve needed.
#
# Reuses existing advect_vof_2d! and advect_ls_2d! kernels.
# ===========================================================================

# --- GPU kernel: clamp field values ---

@kernel function clamp_field_2d_kernel!(C, lo, hi)
    i, j = @index(Global, NTuple)
    @inbounds begin
        C[i, j] = clamp(C[i, j], lo, hi)
    end
end

"""
    clamp_field_2d!(C, lo, hi)

Clamp all values of 2D array `C` to `[lo, hi]` on GPU.
"""
function clamp_field_2d!(C, lo, hi)
    backend = KernelAbstractions.get_backend(C)
    Nx, Ny = size(C)
    T = eltype(C)
    kernel! = clamp_field_2d_kernel!(backend)
    kernel!(C, T(lo), T(hi); ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# --- Advect VOF one step + clamp ---

"""
    advect_vof_step!(C, C_new, ux, uy, Nx, Ny)

Advect volume fraction `C` one step using MUSCL-Superbee TVD scheme,
then clamp to [0, 1]. Uses `advect_vof_2d!` (directional splitting).
"""
function advect_vof_step!(C, C_new, ux, uy, Nx, Ny)
    advect_vof_2d!(C_new, C, ux, uy, Nx, Ny)
    clamp_field_2d!(C_new, zero(eltype(C_new)), one(eltype(C_new)))
end

"""
    advect_vof_plic_step!(C, C_new, nx_n, ny_n, cc_field, ux, uy, Nx, Ny)

Advect volume fraction `C` one step using geometric PLIC reconstruction,
then clamp to [0, 1]. Requires pre-allocated normal arrays `(nx_n, ny_n)`
and Weymouth-Yue work array `cc_field`.
Normals are computed internally before each sweep.
"""
function advect_vof_plic_step!(C, C_new, nx_n, ny_n, cc_field, ux, uy, Nx, Ny; step::Int=1)
    compute_vof_normal_2d!(nx_n, ny_n, C, Nx, Ny)
    advect_vof_plic_2d!(C_new, C, nx_n, ny_n, cc_field, ux, uy, Nx, Ny; step=step)
    clamp_field_2d!(C_new, zero(eltype(C_new)), one(eltype(C_new)))
end

# --- Fill velocity arrays from analytical function ---

"""
    fill_velocity_field!(ux, uy, velocity_fn, dx, t, backend, T)

Fill GPU arrays `(ux, uy)` from an analytical velocity function.

`velocity_fn(x, y, t) -> (vx, vy)` is evaluated at each cell center
(physical coordinates: x = (i-0.5)*dx, y = (j-0.5)*dx).

This is CPU-side: builds arrays then copies to backend. Suitable for
prescribed-velocity tests on moderate grids where the velocity field
changes rarely (e.g., at reversal).
"""
function fill_velocity_field!(ux, uy, velocity_fn, dx, t,
                              backend, ::Type{T}) where T
    Nx, Ny = size(ux)
    ux_cpu = zeros(T, Nx, Ny)
    uy_cpu = zeros(T, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        x = (i - T(0.5)) * dx
        y = (j - T(0.5)) * dx
        vx, vy = velocity_fn(x, y, t)
        ux_cpu[i, j] = T(vx)
        uy_cpu[i, j] = T(vy)
    end
    copyto!(ux, ux_cpu)
    copyto!(uy, uy_cpu)
end

# --- Initialize VOF field from analytical function ---

"""
    init_vof_field!(C, init_fn, dx, backend, T)

Initialize volume fraction `C` from `init_fn(x, y) -> C₀ ∈ [0,1]`.
"""
function init_vof_field!(C, init_fn, dx, ::Type{T}) where T
    Nx, Ny = size(C)
    C_cpu = zeros(T, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        x = (i - T(0.5)) * dx
        y = (j - T(0.5)) * dx
        C_cpu[i, j] = clamp(T(init_fn(x, y)), zero(T), one(T))
    end
    copyto!(C, C_cpu)
end
