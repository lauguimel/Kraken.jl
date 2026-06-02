# Ghost fluid velocity extrapolation for two-phase LBM.
#
# In the ghost fluid approach, the LBM runs at ρ_lbm=1 everywhere (no density
# discontinuity in distributions). After computing macroscopic velocity, gas cell
# velocities are overwritten with values extrapolated from the nearest liquid cells.
#
# This eliminates the streaming blow-up at high density ratios (ρ_l/ρ_g >> 1)
# while preserving sharp PLIC interface tracking and exact mass conservation.
#
# The gas phase has no independent dynamics — its velocity is slaved to the liquid.
# Valid for flows where gas inertia is negligible (e.g. CIJ at ρ_ratio=1000).

using KernelAbstractions

# --- Valid mask initialization ---

@kernel function initialize_valid_mask_2d_kernel!(is_valid, @Const(C), C_threshold)
    i, j = @index(Global, NTuple)
    @inbounds is_valid[i,j] = C[i,j] >= C_threshold
end

# --- Single-pass velocity extrapolation ---

@kernel function extrapolate_velocity_step_2d_kernel!(
        ux_out, uy_out, is_valid_new,
        @Const(ux_in), @Const(uy_in), @Const(is_valid),
        Nx, Ny)
    i, j = @index(Global, NTuple)
    T = eltype(ux_in)
    @inbounds begin
        if is_valid[i,j]
            # Liquid/interface cell: copy through
            ux_out[i,j] = ux_in[i,j]
            uy_out[i,j] = uy_in[i,j]
            is_valid_new[i,j] = true
        else
            # Gas cell: average valid neighbors in 8-stencil
            sum_ux = zero(T)
            sum_uy = zero(T)
            count = Int32(0)

            # Periodic in x (index 1), clamped in y (index 2)
            im = ifelse(i > 1, i - 1, Nx)
            ip = ifelse(i < Nx, i + 1, 1)
            jm = max(j - 1, Int32(1))
            jp = min(j + 1, Ny)

            # Check all 8 neighbors
            if is_valid[im,jm]; sum_ux += ux_in[im,jm]; sum_uy += uy_in[im,jm]; count += Int32(1); end
            if is_valid[i, jm]; sum_ux += ux_in[i, jm]; sum_uy += uy_in[i, jm]; count += Int32(1); end
            if is_valid[ip,jm]; sum_ux += ux_in[ip,jm]; sum_uy += uy_in[ip,jm]; count += Int32(1); end
            if is_valid[im,j ]; sum_ux += ux_in[im,j ]; sum_uy += uy_in[im,j ]; count += Int32(1); end
            if is_valid[ip,j ]; sum_ux += ux_in[ip,j ]; sum_uy += uy_in[ip,j ]; count += Int32(1); end
            if is_valid[im,jp]; sum_ux += ux_in[im,jp]; sum_uy += uy_in[im,jp]; count += Int32(1); end
            if is_valid[i, jp]; sum_ux += ux_in[i, jp]; sum_uy += uy_in[i, jp]; count += Int32(1); end
            if is_valid[ip,jp]; sum_ux += ux_in[ip,jp]; sum_uy += uy_in[ip,jp]; count += Int32(1); end

            if count > Int32(0)
                inv_count = one(T) / T(count)
                ux_out[i,j] = sum_ux * inv_count
                uy_out[i,j] = sum_uy * inv_count
                is_valid_new[i,j] = true
            else
                # No valid neighbor yet — keep current value, mark invalid
                ux_out[i,j] = ux_in[i,j]
                uy_out[i,j] = uy_in[i,j]
                is_valid_new[i,j] = false
            end
        end
    end
end

"""
    extrapolate_velocity_ghost_2d!(ux, uy, C, ux_tmp, uy_tmp, is_valid, is_valid_new;
                                    C_threshold=0.5, n_layers=3)

Extrapolate liquid velocity into gas cells using iterative nearest-neighbor averaging.

For each gas cell (C < C_threshold), the velocity is replaced by the average of its
valid (liquid or previously extrapolated) neighbors. The process is iterated n_layers
times to propagate velocity outward from the interface.

# Arguments
- `ux, uy`: velocity fields (modified in-place)
- `C`: VOF field (read-only, used for initial valid mask)
- `ux_tmp, uy_tmp`: workspace arrays (same size as ux)
- `is_valid, is_valid_new`: Bool workspace arrays (same size as ux)
- `C_threshold`: VOF threshold for liquid/gas classification (default 0.5)
- `n_layers`: number of extrapolation layers (default 3)
"""
function extrapolate_velocity_ghost_2d!(ux, uy, C, ux_tmp, uy_tmp,
                                         is_valid, is_valid_new;
                                         C_threshold=0.5, n_layers=3)
    backend = KernelAbstractions.get_backend(ux)
    Nx, Ny = size(ux)

    # Initialize valid mask from VOF
    mask_kernel! = initialize_valid_mask_2d_kernel!(backend)
    mask_kernel!(is_valid, C, eltype(C)(C_threshold); ndrange=(Nx, Ny))

    step_kernel! = extrapolate_velocity_step_2d_kernel!(backend)

    for layer in 1:n_layers
        if layer % 2 == 1
            # ux,uy → ux_tmp,uy_tmp
            step_kernel!(ux_tmp, uy_tmp, is_valid_new,
                         ux, uy, is_valid,
                         Int32(Nx), Int32(Ny); ndrange=(Nx, Ny))
            KernelAbstractions.synchronize(backend)
        else
            # ux_tmp,uy_tmp → ux,uy
            step_kernel!(ux, uy, is_valid,
                         ux_tmp, uy_tmp, is_valid_new,
                         Int32(Nx), Int32(Ny); ndrange=(Nx, Ny))
            KernelAbstractions.synchronize(backend)
        end
    end

    # Ensure result is in ux, uy
    if n_layers % 2 == 1
        copyto!(ux, ux_tmp)
        copyto!(uy, uy_tmp)
        copyto!(is_valid, is_valid_new)
    end
end

# --- Reset gas distributions to equilibrium with ghost velocity ---

@kernel function reset_feq_ghost_2d_kernel!(f, @Const(ux), @Const(uy), @Const(C), C_threshold)
    i, j = @index(Global, NTuple)
    T = eltype(f)
    @inbounds begin
        if C[i,j] < C_threshold
            # D2Q9 weights
            w1 = T(4)/T(9)
            w2 = T(1)/T(9)
            w6 = T(1)/T(36)

            u = ux[i,j]
            v = uy[i,j]
            usq = u*u + v*v

            # f_eq at ρ_lbm = 1
            f[i,j,1] = w1 * (one(T)                                  - T(1.5)*usq)
            f[i,j,2] = w2 * (one(T) + T(3)*u     + T(4.5)*u*u       - T(1.5)*usq)
            f[i,j,3] = w2 * (one(T) + T(3)*v     + T(4.5)*v*v       - T(1.5)*usq)
            f[i,j,4] = w2 * (one(T) - T(3)*u     + T(4.5)*u*u       - T(1.5)*usq)
            f[i,j,5] = w2 * (one(T) - T(3)*v     + T(4.5)*v*v       - T(1.5)*usq)
            f[i,j,6] = w6 * (one(T) + T(3)*(u+v) + T(4.5)*(u+v)^2   - T(1.5)*usq)
            f[i,j,7] = w6 * (one(T) + T(3)*(-u+v)+ T(4.5)*(-u+v)^2  - T(1.5)*usq)
            f[i,j,8] = w6 * (one(T) + T(3)*(-u-v)+ T(4.5)*(-u-v)^2  - T(1.5)*usq)
            f[i,j,9] = w6 * (one(T) + T(3)*(u-v) + T(4.5)*(u-v)^2   - T(1.5)*usq)
        end
    end
end

"""
    reset_feq_ghost_2d!(f, ux, uy, C; C_threshold=0.5)

Reset gas cell distributions to equilibrium at ρ_lbm=1 with the ghost velocity.
Prevents stale non-equilibrium momentum from streaming back into liquid cells.
"""
function reset_feq_ghost_2d!(f, ux, uy, C; C_threshold=0.5)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = reset_feq_ghost_2d_kernel!(backend)
    kernel!(f, ux, uy, C, eltype(C)(C_threshold); ndrange=(Nx, Ny))
end
