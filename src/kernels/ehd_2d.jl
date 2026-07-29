using KernelAbstractions

# D2Q9 EHD scalar kernels. Direction order matches Kraken.D2Q9 and Jiachen's
# MATLAB Lattice.m: rest, E, N, W, S, NE, NW, SW, SE.

@inline ehd_w(::Val{1}, T) = T(4) / T(9)
@inline ehd_w(::Val{2}, T) = T(1) / T(9)
@inline ehd_w(::Val{3}, T) = T(1) / T(9)
@inline ehd_w(::Val{4}, T) = T(1) / T(9)
@inline ehd_w(::Val{5}, T) = T(1) / T(9)
@inline ehd_w(::Val{6}, T) = T(1) / T(36)
@inline ehd_w(::Val{7}, T) = T(1) / T(36)
@inline ehd_w(::Val{8}, T) = T(1) / T(36)
@inline ehd_w(::Val{9}, T) = T(1) / T(36)

@inline ehd_cx(::Val{1}, T) = zero(T)
@inline ehd_cx(::Val{2}, T) = one(T)
@inline ehd_cx(::Val{3}, T) = zero(T)
@inline ehd_cx(::Val{4}, T) = -one(T)
@inline ehd_cx(::Val{5}, T) = zero(T)
@inline ehd_cx(::Val{6}, T) = one(T)
@inline ehd_cx(::Val{7}, T) = -one(T)
@inline ehd_cx(::Val{8}, T) = -one(T)
@inline ehd_cx(::Val{9}, T) = one(T)

@inline ehd_cy(::Val{1}, T) = zero(T)
@inline ehd_cy(::Val{2}, T) = zero(T)
@inline ehd_cy(::Val{3}, T) = one(T)
@inline ehd_cy(::Val{4}, T) = zero(T)
@inline ehd_cy(::Val{5}, T) = -one(T)
@inline ehd_cy(::Val{6}, T) = one(T)
@inline ehd_cy(::Val{7}, T) = one(T)
@inline ehd_cy(::Val{8}, T) = -one(T)
@inline ehd_cy(::Val{9}, T) = -one(T)

@inline function ehd_charge_feq(::Val{Q}, q, ux, uy) where {Q}
    T = typeof(q)
    cx = ehd_cx(Val(Q), T)
    cy = ehd_cy(Val(Q), T)
    cu3 = T(3) * (cx * ux + cy * uy)
    usq = ux * ux + uy * uy
    return q * ehd_w(Val(Q), T) * (one(T) + cu3 + T(0.5) * cu3 * cu3 - T(1.5) * usq)
end

@inline function ehd_phi_feq(::Val{Q}, phi) where {Q}
    return ehd_w(Val(Q), typeof(phi)) * phi
end

@kernel function compute_ehd_scalar_2d_kernel!(field, @Const(f))
    i, j = @index(Global, NTuple)
    @inbounds field[i, j] = f[i,j,1] + f[i,j,2] + f[i,j,3] + f[i,j,4] +
                            f[i,j,5] + f[i,j,6] + f[i,j,7] + f[i,j,8] + f[i,j,9]
end

function compute_ehd_scalar_2d!(field, f)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(field)
    kernel! = compute_ehd_scalar_2d_kernel!(backend)
    kernel!(field, f; ndrange=(Nx, Ny))
end

@kernel function collide_electric_potential_2d_kernel!(f, @Const(q), eps, ω_U, nu_U)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f)
        phi = f[i,j,1] + f[i,j,2] + f[i,j,3] + f[i,j,4] +
              f[i,j,5] + f[i,j,6] + f[i,j,7] + f[i,j,8] + f[i,j,9]
        source_scale = nu_U * q[i, j] / eps
        f[i,j,1] -= ω_U * (f[i,j,1] - ehd_phi_feq(Val(1), phi)) - ehd_w(Val(1), T) * source_scale
        f[i,j,2] -= ω_U * (f[i,j,2] - ehd_phi_feq(Val(2), phi)) - ehd_w(Val(2), T) * source_scale
        f[i,j,3] -= ω_U * (f[i,j,3] - ehd_phi_feq(Val(3), phi)) - ehd_w(Val(3), T) * source_scale
        f[i,j,4] -= ω_U * (f[i,j,4] - ehd_phi_feq(Val(4), phi)) - ehd_w(Val(4), T) * source_scale
        f[i,j,5] -= ω_U * (f[i,j,5] - ehd_phi_feq(Val(5), phi)) - ehd_w(Val(5), T) * source_scale
        f[i,j,6] -= ω_U * (f[i,j,6] - ehd_phi_feq(Val(6), phi)) - ehd_w(Val(6), T) * source_scale
        f[i,j,7] -= ω_U * (f[i,j,7] - ehd_phi_feq(Val(7), phi)) - ehd_w(Val(7), T) * source_scale
        f[i,j,8] -= ω_U * (f[i,j,8] - ehd_phi_feq(Val(8), phi)) - ehd_w(Val(8), T) * source_scale
        f[i,j,9] -= ω_U * (f[i,j,9] - ehd_phi_feq(Val(9), phi)) - ehd_w(Val(9), T) * source_scale
    end
end

function collide_electric_potential_2d!(f, q, eps, ω_U, nu_U)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = collide_electric_potential_2d_kernel!(backend)
    kernel!(f, q, eltype(f)(eps), eltype(f)(ω_U), eltype(f)(nu_U); ndrange=(Nx, Ny))
end

@kernel function compute_electric_field_2d_kernel!(Ex, Ey, @Const(f), tau_U)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f)
        inv = one(T) / (tau_U * (one(T) / T(3)))
        Ex[i, j] = (f[i,j,2] - f[i,j,4] + f[i,j,6] - f[i,j,7] - f[i,j,8] + f[i,j,9]) * inv
        Ey[i, j] = (f[i,j,3] - f[i,j,5] + f[i,j,6] + f[i,j,7] - f[i,j,8] - f[i,j,9]) * inv
    end
end

function compute_electric_field_2d!(Ex, Ey, f, tau_U)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(Ex)
    kernel! = compute_electric_field_2d_kernel!(backend)
    kernel!(Ex, Ey, f, eltype(f)(tau_U); ndrange=(Nx, Ny))
end

@kernel function collide_electric_charge_srt_2d_kernel!(f, @Const(Ex), @Const(Ey), tau_q, K)
    i, j = @index(Global, NTuple)
    @inbounds begin
        q = f[i,j,1] + f[i,j,2] + f[i,j,3] + f[i,j,4] +
            f[i,j,5] + f[i,j,6] + f[i,j,7] + f[i,j,8] + f[i,j,9]
        ux = K * Ex[i, j]
        uy = K * Ey[i, j]
        ω = one(eltype(f)) / tau_q
        f[i,j,1] -= ω * (f[i,j,1] - ehd_charge_feq(Val(1), q, ux, uy))
        f[i,j,2] -= ω * (f[i,j,2] - ehd_charge_feq(Val(2), q, ux, uy))
        f[i,j,3] -= ω * (f[i,j,3] - ehd_charge_feq(Val(3), q, ux, uy))
        f[i,j,4] -= ω * (f[i,j,4] - ehd_charge_feq(Val(4), q, ux, uy))
        f[i,j,5] -= ω * (f[i,j,5] - ehd_charge_feq(Val(5), q, ux, uy))
        f[i,j,6] -= ω * (f[i,j,6] - ehd_charge_feq(Val(6), q, ux, uy))
        f[i,j,7] -= ω * (f[i,j,7] - ehd_charge_feq(Val(7), q, ux, uy))
        f[i,j,8] -= ω * (f[i,j,8] - ehd_charge_feq(Val(8), q, ux, uy))
        f[i,j,9] -= ω * (f[i,j,9] - ehd_charge_feq(Val(9), q, ux, uy))
    end
end

function collide_electric_charge_srt_2d!(f, Ex, Ey, tau_q, K)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = collide_electric_charge_srt_2d_kernel!(backend)
    kernel!(f, Ex, Ey, eltype(f)(tau_q), eltype(f)(K); ndrange=(Nx, Ny))
end

@kernel function collide_electric_charge_regularized_2d_kernel!(f, @Const(Ex), @Const(Ey), tau_q, K)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f)
        q = f[i,j,1] + f[i,j,2] + f[i,j,3] + f[i,j,4] +
            f[i,j,5] + f[i,j,6] + f[i,j,7] + f[i,j,8] + f[i,j,9]
        ux = K * Ex[i, j]
        uy = K * Ey[i, j]
        feq1 = ehd_charge_feq(Val(1), q, ux, uy)
        feq2 = ehd_charge_feq(Val(2), q, ux, uy)
        feq3 = ehd_charge_feq(Val(3), q, ux, uy)
        feq4 = ehd_charge_feq(Val(4), q, ux, uy)
        feq5 = ehd_charge_feq(Val(5), q, ux, uy)
        feq6 = ehd_charge_feq(Val(6), q, ux, uy)
        feq7 = ehd_charge_feq(Val(7), q, ux, uy)
        feq8 = ehd_charge_feq(Val(8), q, ux, uy)
        feq9 = ehd_charge_feq(Val(9), q, ux, uy)
        jx = (f[i,j,2] - feq2) - (f[i,j,4] - feq4) +
             (f[i,j,6] - feq6) - (f[i,j,7] - feq7) -
             (f[i,j,8] - feq8) + (f[i,j,9] - feq9)
        jy = (f[i,j,3] - feq3) - (f[i,j,5] - feq5) +
             (f[i,j,6] - feq6) + (f[i,j,7] - feq7) -
             (f[i,j,8] - feq8) - (f[i,j,9] - feq9)
        pref = one(T) - one(T) / tau_q
        f[i,j,1] = feq1
        f[i,j,2] = feq2 + pref * ehd_w(Val(2), T) * T(3) * jx
        f[i,j,3] = feq3 + pref * ehd_w(Val(3), T) * T(3) * jy
        f[i,j,4] = feq4 - pref * ehd_w(Val(4), T) * T(3) * jx
        f[i,j,5] = feq5 - pref * ehd_w(Val(5), T) * T(3) * jy
        f[i,j,6] = feq6 + pref * ehd_w(Val(6), T) * T(3) * (jx + jy)
        f[i,j,7] = feq7 + pref * ehd_w(Val(7), T) * T(3) * (-jx + jy)
        f[i,j,8] = feq8 + pref * ehd_w(Val(8), T) * T(3) * (-jx - jy)
        f[i,j,9] = feq9 + pref * ehd_w(Val(9), T) * T(3) * (jx - jy)
    end
end

function collide_electric_charge_regularized_2d!(f, Ex, Ey, tau_q, K)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = collide_electric_charge_regularized_2d_kernel!(backend)
    kernel!(f, Ex, Ey, eltype(f)(tau_q), eltype(f)(K); ndrange=(Nx, Ny))
end

@kernel function apply_phi_nee_walls_2d_kernel!(f, @Const(phi), phi_bottom, phi_top, Ny)
    i = @index(Global)
    @inbounds begin
        T = eltype(f)
        for qdir in 1:9
            wq = ifelse(qdir == 1, T(4)/T(9), ifelse(qdir <= 5, T(1)/T(9), T(1)/T(36)))
            f[i, 1, qdir] = f[i, 2, qdir] + wq * (phi_bottom - phi[i, 2])
            f[i, Ny, qdir] = f[i, Ny - 1, qdir] + wq * (phi_top - phi[i, Ny - 1])
        end
    end
end

function apply_phi_nee_walls_2d!(f, phi, phi_bottom, phi_top, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = apply_phi_nee_walls_2d_kernel!(backend)
    kernel!(f, phi, eltype(f)(phi_bottom), eltype(f)(phi_top), Ny; ndrange=(Nx,))
end

@kernel function apply_charge_nee_walls_2d_kernel!(f, @Const(qfield), @Const(Ex), @Const(Ey),
                                                   q_bottom, grad_top, K, Ny)
    i = @index(Global)
    @inbounds begin
        qb = q_bottom
        qt = qfield[i, Ny - 1] - grad_top
        ux1 = K * Ex[i, 1]
        uy1 = K * Ey[i, 1]
        ux2 = K * Ex[i, 2]
        uy2 = K * Ey[i, 2]
        uxt = K * Ex[i, Ny]
        uyt = K * Ey[i, Ny]
        uxn = K * Ex[i, Ny - 1]
        uyn = K * Ey[i, Ny - 1]
        f[i, 1, 1] = f[i, 2, 1] + ehd_charge_feq(Val(1), qb, ux1, uy1) - ehd_charge_feq(Val(1), qfield[i, 2], ux2, uy2)
        f[i, 1, 2] = f[i, 2, 2] + ehd_charge_feq(Val(2), qb, ux1, uy1) - ehd_charge_feq(Val(2), qfield[i, 2], ux2, uy2)
        f[i, 1, 3] = f[i, 2, 3] + ehd_charge_feq(Val(3), qb, ux1, uy1) - ehd_charge_feq(Val(3), qfield[i, 2], ux2, uy2)
        f[i, 1, 4] = f[i, 2, 4] + ehd_charge_feq(Val(4), qb, ux1, uy1) - ehd_charge_feq(Val(4), qfield[i, 2], ux2, uy2)
        f[i, 1, 5] = f[i, 2, 5] + ehd_charge_feq(Val(5), qb, ux1, uy1) - ehd_charge_feq(Val(5), qfield[i, 2], ux2, uy2)
        f[i, 1, 6] = f[i, 2, 6] + ehd_charge_feq(Val(6), qb, ux1, uy1) - ehd_charge_feq(Val(6), qfield[i, 2], ux2, uy2)
        f[i, 1, 7] = f[i, 2, 7] + ehd_charge_feq(Val(7), qb, ux1, uy1) - ehd_charge_feq(Val(7), qfield[i, 2], ux2, uy2)
        f[i, 1, 8] = f[i, 2, 8] + ehd_charge_feq(Val(8), qb, ux1, uy1) - ehd_charge_feq(Val(8), qfield[i, 2], ux2, uy2)
        f[i, 1, 9] = f[i, 2, 9] + ehd_charge_feq(Val(9), qb, ux1, uy1) - ehd_charge_feq(Val(9), qfield[i, 2], ux2, uy2)
        f[i, Ny, 1] = f[i, Ny - 1, 1] + ehd_charge_feq(Val(1), qt, uxt, uyt) - ehd_charge_feq(Val(1), qfield[i, Ny - 1], uxn, uyn)
        f[i, Ny, 2] = f[i, Ny - 1, 2] + ehd_charge_feq(Val(2), qt, uxt, uyt) - ehd_charge_feq(Val(2), qfield[i, Ny - 1], uxn, uyn)
        f[i, Ny, 3] = f[i, Ny - 1, 3] + ehd_charge_feq(Val(3), qt, uxt, uyt) - ehd_charge_feq(Val(3), qfield[i, Ny - 1], uxn, uyn)
        f[i, Ny, 4] = f[i, Ny - 1, 4] + ehd_charge_feq(Val(4), qt, uxt, uyt) - ehd_charge_feq(Val(4), qfield[i, Ny - 1], uxn, uyn)
        f[i, Ny, 5] = f[i, Ny - 1, 5] + ehd_charge_feq(Val(5), qt, uxt, uyt) - ehd_charge_feq(Val(5), qfield[i, Ny - 1], uxn, uyn)
        f[i, Ny, 6] = f[i, Ny - 1, 6] + ehd_charge_feq(Val(6), qt, uxt, uyt) - ehd_charge_feq(Val(6), qfield[i, Ny - 1], uxn, uyn)
        f[i, Ny, 7] = f[i, Ny - 1, 7] + ehd_charge_feq(Val(7), qt, uxt, uyt) - ehd_charge_feq(Val(7), qfield[i, Ny - 1], uxn, uyn)
        f[i, Ny, 8] = f[i, Ny - 1, 8] + ehd_charge_feq(Val(8), qt, uxt, uyt) - ehd_charge_feq(Val(8), qfield[i, Ny - 1], uxn, uyn)
        f[i, Ny, 9] = f[i, Ny - 1, 9] + ehd_charge_feq(Val(9), qt, uxt, uyt) - ehd_charge_feq(Val(9), qfield[i, Ny - 1], uxn, uyn)
    end
end

function apply_charge_nee_walls_2d!(f, qfield, Ex, Ey, q_bottom, grad_top, K, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = apply_charge_nee_walls_2d_kernel!(backend)
    kernel!(f, qfield, Ex, Ey, eltype(f)(q_bottom), eltype(f)(grad_top), eltype(f)(K), Ny; ndrange=(Nx,))
end
