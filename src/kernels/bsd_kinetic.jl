using KernelAbstractions

# Kinetic-moment route to the BSD body force. Exposes Pi^{neq} extraction
# from the LBM distribution and assembles a body force that is, on smooth
# interior cells, numerically equivalent to the FD-laplacian BSD path
# (Chapman-Enskog identity, validated 5.85e-16 on the N=32 t=2 smoke).
# Kept as infrastructure for future rheology diagnostics (effective
# viscosity recovery, Reynolds stress) and as the substrate on which a
# fully lattice-stencil-aware BSD (different from FD-central) can be
# built once that path is justified by data.

# Pi_neq_ab[i,j] = sum_q c_qa c_qb (f[i,j,q] - f_eq[i,j,q]).
@kernel function compute_pi_neq_2d_kernel!(
    Pi_xx, Pi_xy, Pi_yy,
    @Const(f), @Const(rho), @Const(ux), @Const(uy), @Const(is_solid),
    Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            T = eltype(Pi_xx)
            if is_solid[i, j]
                Pi_xx[i, j] = zero(T)
                Pi_xy[i, j] = zero(T)
                Pi_yy[i, j] = zero(T)
            else
                rho_l = rho[i, j]
                ux_l = ux[i, j]
                uy_l = uy[i, j]
                usq = ux_l * ux_l + uy_l * uy_l

                f1 = f[i, j, 1]; feq1 = feq_2d(Val(1), rho_l, ux_l, uy_l, usq)
                f2 = f[i, j, 2]; feq2 = feq_2d(Val(2), rho_l, ux_l, uy_l, usq)
                f3 = f[i, j, 3]; feq3 = feq_2d(Val(3), rho_l, ux_l, uy_l, usq)
                f4 = f[i, j, 4]; feq4 = feq_2d(Val(4), rho_l, ux_l, uy_l, usq)
                f5 = f[i, j, 5]; feq5 = feq_2d(Val(5), rho_l, ux_l, uy_l, usq)
                f6 = f[i, j, 6]; feq6 = feq_2d(Val(6), rho_l, ux_l, uy_l, usq)
                f7 = f[i, j, 7]; feq7 = feq_2d(Val(7), rho_l, ux_l, uy_l, usq)
                f8 = f[i, j, 8]; feq8 = feq_2d(Val(8), rho_l, ux_l, uy_l, usq)
                f9 = f[i, j, 9]; feq9 = feq_2d(Val(9), rho_l, ux_l, uy_l, usq)

                d1 = f1 - feq1; d2 = f2 - feq2; d3 = f3 - feq3
                d4 = f4 - feq4; d5 = f5 - feq5; d6 = f6 - feq6
                d7 = f7 - feq7; d8 = f8 - feq8; d9 = f9 - feq9

                Pi_xx[i, j] = d2 + d4 + d6 + d7 + d8 + d9
                Pi_yy[i, j] = d3 + d5 + d6 + d7 + d8 + d9
                Pi_xy[i, j] = d6 - d7 + d8 - d9
            end
        end
    end
end

# Solid-aware cell-centred divergence of Pi_neq_raw, assembled as FD-BSD force.
@kernel function bsd_kinetic_assemble_2d_kernel!(
    fx_out, fy_out,
    @Const(fx_poly), @Const(fy_poly),
    @Const(Pi_xx), @Const(Pi_xy), @Const(Pi_yy),
    @Const(rho), @Const(is_solid),
    coeff, inv_2dx, inv_2dy, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            T = eltype(fx_out)
            if is_solid[i, j]
                fx_out[i, j] = zero(T)
                fy_out[i, j] = zero(T)
            else
                inv_dx = T(2) * inv_2dx
                inv_dy = T(2) * inv_2dy
                dxx_x = _logfv_solid_aware_derivative_x_2d(Pi_xx, is_solid, i, j, Nx, inv_dx, inv_2dx)
                dxy_y = _logfv_solid_aware_derivative_y_2d(Pi_xy, is_solid, i, j, Ny, inv_dy, inv_2dy)
                dxy_x = _logfv_solid_aware_derivative_x_2d(Pi_xy, is_solid, i, j, Nx, inv_dx, inv_2dx)
                dyy_y = _logfv_solid_aware_derivative_y_2d(Pi_yy, is_solid, i, j, Ny, inv_dy, inv_2dy)
                div_x = dxx_x + dxy_y
                div_y = dxy_x + dyy_y
                inv_rho = one(T) / rho[i, j]

                # Pi_raw has the Chapman-Enskog sign, so +div(Pi_raw) is the FD-BSD sign.
                fx_out[i, j] = fx_poly[i, j] + coeff * inv_rho * div_x
                fy_out[i, j] = fy_poly[i, j] + coeff * inv_rho * div_y
            end
        end
    end
end

# Builds the BSD-corrected body force from the LBM non-equilibrium momentum tensor.
function compute_bsd_force_kinetic_2d!(
    fx_out, fy_out, fx_poly, fy_poly,
    f, rho, ux, uy, is_solid,
    zeta, nu_p, s_plus, dx, dy;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(fx_out)
    T = eltype(fx_out)
    Nx, Ny = size(fx_out)

    nu_eff = (one(T) / T(3)) * (one(T) / T(s_plus) - one(T) / T(2))
    guo_pref = one(T) - T(s_plus) / T(2)
    coeff = T(zeta) * T(nu_p) / nu_eff * guo_pref

    Pi_xx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    Pi_xy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    Pi_yy = KernelAbstractions.zeros(backend, T, Nx, Ny)

    pi_kernel! = compute_pi_neq_2d_kernel!(backend)
    pi_kernel!(Pi_xx, Pi_xy, Pi_yy, f, rho, ux, uy, is_solid, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)

    assemble! = bsd_kinetic_assemble_2d_kernel!(backend)
    assemble!(
        fx_out, fy_out, fx_poly, fy_poly,
        Pi_xx, Pi_xy, Pi_yy, rho, is_solid,
        coeff, inv(T(2) * T(dx)), inv(T(2) * T(dy)), Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end
