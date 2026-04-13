using KernelAbstractions

# ===========================================================================
# Dual-grid operators for VOF-PLIC coupled with LBM
#
# Fine grid (Nx_f × Ny_f): VOF advection, interface geometry, surface tension
# Coarse grid (Nx_c × Ny_c): LBM fluid solver (stream, collide)
# Refinement ratio: r (Nx_f = r·Nx_c, Ny_f = r·Ny_c)
# Fine grid spacing: dx_f = 1/r (in coarse lattice units)
#
# References:
# - Dual-resolution VOF: Chen & Zhang (2016) doi:10.1016/j.jcp.2016.02.003
# - Height-function curvature: Cummins et al. (2005) doi:10.1016/j.compfluid.2004.03.005
# ===========================================================================

# --- Prolongation: bilinear interpolation coarse → fine ---

@kernel function prolongate_bilinear_2d_kernel!(q_fine, @Const(q_coarse),
                                                  r, Nx_c, Ny_c, scale)
    i_f, j_f = @index(Global, NTuple)

    @inbounds begin
        T = eltype(q_fine)
        # Fine cell center in coarse continuous-index space
        # Coarse cell ic has center at position ic; fine cell maps as:
        xc = (T(i_f) - T(0.5)) / T(r) + T(0.5)
        yc = (T(j_f) - T(0.5)) / T(r) + T(0.5)

        # Bracketing coarse indices (trunc = floor for positive values)
        i0_raw = unsafe_trunc(Int, xc)
        j0_raw = unsafe_trunc(Int, yc)

        # Bilinear weights (before clamping)
        tx = xc - T(i0_raw)
        ty = yc - T(j0_raw)

        # Clamp indices to valid range (handles boundary fine cells)
        i0 = clamp(i0_raw, 1, Nx_c)
        i1 = clamp(i0_raw + 1, 1, Nx_c)
        j0 = clamp(j0_raw, 1, Ny_c)
        j1 = clamp(j0_raw + 1, 1, Ny_c)
        tx = clamp(tx, zero(T), one(T))
        ty = clamp(ty, zero(T), one(T))

        q_fine[i_f, j_f] = scale * (
            (one(T) - tx) * (one(T) - ty) * q_coarse[i0, j0] +
            tx             * (one(T) - ty) * q_coarse[i1, j0] +
            (one(T) - tx) * ty             * q_coarse[i0, j1] +
            tx             * ty             * q_coarse[i1, j1]
        )
    end
end

"""
    prolongate_bilinear_2d!(q_fine, q_coarse, r; scale=1)

Bilinear interpolation from coarse grid to fine grid (ratio `r`).
Optional `scale` multiplies the result (use `r` for velocity CFL scaling).
"""
function prolongate_bilinear_2d!(q_fine, q_coarse, r; scale=one(eltype(q_fine)))
    backend = KernelAbstractions.get_backend(q_fine)
    Nx_f, Ny_f = size(q_fine)
    Nx_c, Ny_c = size(q_coarse)
    T = eltype(q_fine)
    kernel! = prolongate_bilinear_2d_kernel!(backend)
    kernel!(q_fine, q_coarse, r, Nx_c, Ny_c, T(scale); ndrange=(Nx_f, Ny_f))
    KernelAbstractions.synchronize(backend)
end

# --- Restriction: block average fine → coarse ---

@kernel function restrict_average_2d_kernel!(q_coarse, @Const(q_fine), r)
    ic, jc = @index(Global, NTuple)

    @inbounds begin
        T = eltype(q_coarse)
        inv_r2 = one(T) / T(r * r)
        acc = zero(T)
        for dj in 0:(r - 1)
            jf = (jc - 1) * r + 1 + dj
            for di in 0:(r - 1)
                i_f = (ic - 1) * r + 1 + di
                acc += q_fine[i_f, jf]
            end
        end
        q_coarse[ic, jc] = acc * inv_r2
    end
end

"""
    restrict_average_2d!(q_coarse, q_fine, r)

Block-average from fine grid to coarse grid (ratio `r`).
Each coarse cell = mean of its r×r fine children.
"""
function restrict_average_2d!(q_coarse, q_fine, r)
    backend = KernelAbstractions.get_backend(q_coarse)
    Nx_c, Ny_c = size(q_coarse)
    kernel! = restrict_average_2d_kernel!(backend)
    kernel!(q_coarse, q_fine, r; ndrange=(Nx_c, Ny_c))
    KernelAbstractions.synchronize(backend)
end

# --- Height function curvature with explicit grid spacing ---
#
# Same algorithm as compute_hf_curvature_2d but all finite differences
# use physical spacing dx instead of assuming dx=1.
# κ = -h''/(1+h'²)^{3/2}  where h = Σ C·dx, h' = dh/dx, h'' = d²h/dx²

@kernel function compute_hf_curvature_dx_2d_kernel!(κ, @Const(C), @Const(nx_n), @Const(ny_n),
                                                       Nx, Ny, dx, hw)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(C)
        c = C[i, j]

        if c > T(0.01) && c < T(0.99)
            if abs(ny_n[i, j]) >= abs(nx_n[i, j])
                # Interface more horizontal → vertical columns h(x)
                im = ifelse(i > 1, i - 1, Nx)
                ip = ifelse(i < Nx, i + 1, 1)

                h_m = zero(T); h_0 = zero(T); h_p = zero(T)
                for dj in -hw:hw
                    jj = j + dj
                    jj = ifelse(jj < 1, jj + Ny, ifelse(jj > Ny, jj - Ny, jj))
                    h_m += C[im, jj]
                    h_0 += C[i,  jj]
                    h_p += C[ip, jj]
                end
                # Physical height: h = Σ C · dx
                h_m *= dx; h_0 *= dx; h_p *= dx

                hp  = (h_p - h_m) / (T(2) * dx)
                hpp = (h_p - T(2) * h_0 + h_m) / (dx * dx)
                κ[i, j] = -hpp / (one(T) + hp^2)^T(1.5)
            else
                # Interface more vertical → horizontal rows h(y)
                jm = ifelse(j > 1, j - 1, Ny)
                jp = ifelse(j < Ny, j + 1, 1)

                h_m = zero(T); h_0 = zero(T); h_p = zero(T)
                for di in -hw:hw
                    ii = i + di
                    ii = ifelse(ii < 1, ii + Nx, ifelse(ii > Nx, ii - Nx, ii))
                    h_m += C[ii, jm]
                    h_0 += C[ii, j]
                    h_p += C[ii, jp]
                end
                h_m *= dx; h_0 *= dx; h_p *= dx

                hp  = (h_p - h_m) / (T(2) * dx)
                hpp = (h_p - T(2) * h_0 + h_m) / (dx * dx)
                κ[i, j] = -hpp / (one(T) + hp^2)^T(1.5)
            end
        else
            κ[i, j] = zero(T)
        end
    end
end

"""
    compute_hf_curvature_dx_2d!(κ, C, nx, ny, Nx, Ny, dx; hw=2)

Height-function curvature with explicit grid spacing `dx` and stencil
half-width `hw` (default 2). For dual-grid with refinement `r`, use
`hw = 2*r` to match the physical stencil width of the coarse grid.
"""
function compute_hf_curvature_dx_2d!(κ, C, nx, ny, Nx, Ny, dx; hw=2)
    backend = KernelAbstractions.get_backend(C)
    T = eltype(C)
    kernel! = compute_hf_curvature_dx_2d_kernel!(backend)
    kernel!(κ, C, nx, ny, Nx, Ny, T(dx), Int(hw); ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# --- CSF surface tension with explicit grid spacing ---

@kernel function compute_surface_tension_dx_2d_kernel!(Fx, Fy, @Const(κ), @Const(C),
                                                          σ, Nx, Ny, dx)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(C)
        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)
        jp = ifelse(j < Ny, j + 1, 1)
        jm = ifelse(j > 1,  j - 1, Ny)

        # ∇C with physical spacing
        dCdx = (C[ip, j] - C[im, j]) / (T(2) * dx)
        dCdy = (C[i, jp] - C[i, jm]) / (T(2) * dx)

        # CSF: F = σ·κ·∇C (force/volume in physical units)
        Fx[i, j] = σ * κ[i, j] * dCdx
        Fy[i, j] = σ * κ[i, j] * dCdy
    end
end

"""
    compute_surface_tension_dx_2d!(Fx, Fy, κ, C, σ, Nx, Ny, dx)

CSF surface tension force with explicit grid spacing `dx`.
F = σ·κ·∇C in physical units (force/volume).
"""
function compute_surface_tension_dx_2d!(Fx, Fy, κ, C, σ, Nx, Ny, dx)
    backend = KernelAbstractions.get_backend(C)
    T = eltype(C)
    kernel! = compute_surface_tension_dx_2d_kernel!(backend)
    kernel!(Fx, Fy, κ, C, T(σ), Nx, Ny, T(dx); ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
