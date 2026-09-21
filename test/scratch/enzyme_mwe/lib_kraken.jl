# Kraken-level helpers for the issue #41 probe: the FAST case of
# test/ad/test_ad_ve_sensitivity.jl, the suite's forward JVP, and a local copy
# of ad_ve_coupled_step! (src/ad/ad_ve_step.jl:120) whose advection operator
# and solid mask are selected by a Val argument so one file can test variants
# without touching src/. Not part of the test suite.
using Kraken, Enzyme, LinearAlgebra
include(joinpath(@__DIR__, "lib_standalone.jl"))
const K = Kraken

const FAST = (; Nx=24, Ny=24, cx=12.35, cy=11.65, R=5.13,
              Wi=0.5, beta=0.5, nu_p=0.02, nu_s=0.08, Fx_body=2e-4, samples=16)
const SMALL = (; Nx=12, Ny=12, cx=6.35, cy=5.65, R=2.13,
               Wi=0.5, beta=0.5, nu_p=0.02, nu_s=0.08, Fx_body=2e-4, samples=16)

function build_case(c)
    lambda = c.Wi
    pref = c.nu_p / lambda
    s_plus, s_minus = K.ad_ve_trt_rates(c.nu_s)
    p = K.ADVECoupledParams(c.Nx, c.Ny, lambda, 0.05, 4, pref, c.nu_s,
                            c.Fx_body, s_plus, s_minus)
    geom = K.ad_ve_build_geom(c.Nx, c.Ny, c.cx, c.cy, c.R;
                              samples=c.samples, u_mean=c.Fx_body)
    return p, geom
end

# converge=true: the suite's converged state (fwd_tol=1e-13, ~3 s).
# converge=false: warm start plus 20 plain steps (crash probes only).
function base_state(c, p, geom; converge::Bool)
    w0 = K.ad_ve_initial_state(geom.g, c.Nx, c.Ny, 0.05)
    if converge
        fwd = K.ad_ve_forward_solve(w0, geom, p; fwd_tol=1e-13)
        println("forward solve: n_iter=$(fwd.n_iter) reached_tol=$(fwd.reached_tol)")
        return fwd.w_star
    end
    w_in = copy(w0); w_out = zeros(length(w0))
    for _ in 1:20
        K.ad_ve_coupled_step!(w_out, w_in, geom.g, geom.q_wall, p, geom.u_profile, 1.0, nothing)
        w_in, w_out = w_out, w_in
    end
    return w_in
end

function seeds(len)
    u = [sin(0.137 * idx + 0.3) for idx in 1:len]
    v = [cos(0.211 * idx + 0.7) for idx in 1:len]
    return u ./ norm(u), v ./ norm(v)
end

# test-side forward JVP, verbatim from test/ad/test_ad_ve_sensitivity.jl:64
function suite_jvp(w_star, u, geom, p)
    out_len = length(w_star)
    out = zeros(Float64, out_len)
    dout = zeros(Float64, out_len)
    Enzyme.autodiff(RT_FWD, K.ad_ve_coupled_step!,
                    Enzyme.Duplicated(out, dout),
                    Enzyme.Duplicated(copy(w_star), copy(u)),
                    Enzyme.Const(geom.g), Enzyme.Const(geom.q_wall),
                    Enzyme.Const(p), Enzyme.Const(geom.u_profile),
                    Enzyme.Const(1.0), Enzyme.Const(nothing))
    return dout
end

# ----------------------------------------------------------------------------
# Local copy of ad_ve_coupled_step! with two switches:
#   V (advection): :kraken   -> K.ad_ve_advect_prodbc (verbatim path)
#                  :noclosure, :pass1, :prealloc  -> lib_standalone variants
#   is_solid is an explicit argument (BitMatrix from g, or Matrix{Bool}).
#   E (east BC vector): :slice -> psixx_in[Nx, :] (verbatim); :loop -> explicit copy.
@inline _advect(::Val{:kraken}, phi, ux_face, uy_face, is_solid, Nx, Ny, e) =
    K.ad_ve_advect_prodbc(phi, ux_face, uy_face, is_solid, Nx, Ny, e)
@inline _advect(::Val{:noclosure}, phi, ux_face, uy_face, is_solid, Nx, Ny, e) =
    advect_noclosure(phi, ux_face, uy_face, is_solid, Nx, Ny, e)
@inline _advect(::Val{:pass1}, phi, ux_face, uy_face, is_solid, Nx, Ny, e) =
    advect_pass1(phi, ux_face, uy_face, is_solid, Nx, Ny, e)
@inline function _advect(::Val{:prealloc}, phi, ux_face, uy_face, is_solid, Nx, Ny, e)
    adv = zeros(Nx, Ny)
    advect_prealloc!(adv, phi, ux_face, uy_face, is_solid, Nx, Ny, e)
    return adv
end

@inline _east(::Val{:slice}, psi, Nx, Ny) = psi[Nx, :]
@inline function _east(::Val{:loop}, psi, Nx, Ny)
    e = zeros(Ny)
    @inbounds for j in 1:Ny
        e[j] = psi[Nx, j]
    end
    return e
end

function step_local!(w_out, w_in, g::K.ADVEEmbeddedGeom, is_solid, q_wall::Array{Float64,3},
                     p::K.ADVECoupledParams, u_profile::Vector{Float64}, V::Val, E::Val)
    Nx, Ny = p.Nx, p.Ny
    n = Nx * Ny
    inv_dx = 1.0; inv_dy = 1.0
    inv_2dx = 0.5; inv_2dy = 0.5
    wb, eb, sb, nb = K.AD_VE_WB, K.AD_VE_EB, K.AD_VE_SB, K.AD_VE_NB
    sp = p.s_plus; sm = p.s_minus
    foff = 0; poff = 9n
    rho_out = 1.0

    psixx_in = zeros(Float64, Nx, Ny)
    psixy_in = zeros(Float64, Nx, Ny)
    psiyy_in = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        k = K.ad_ve_lin(i, j, Nx)
        psixx_in[i, j] = w_in[poff + k]
        psixy_in[i, j] = w_in[poff + n + k]
        psiyy_in[i, j] = w_in[poff + 2n + k]
    end

    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        f1 = w_in[foff + K.ad_ve_fpop(i, j, 1, Nx, Ny)]
        f2 = w_in[foff + K.ad_ve_fpop(i, j, 2, Nx, Ny)]
        f3 = w_in[foff + K.ad_ve_fpop(i, j, 3, Nx, Ny)]
        f4 = w_in[foff + K.ad_ve_fpop(i, j, 4, Nx, Ny)]
        f5 = w_in[foff + K.ad_ve_fpop(i, j, 5, Nx, Ny)]
        f6 = w_in[foff + K.ad_ve_fpop(i, j, 6, Nx, Ny)]
        f7 = w_in[foff + K.ad_ve_fpop(i, j, 7, Nx, Ny)]
        f8 = w_in[foff + K.ad_ve_fpop(i, j, 8, Nx, Ny)]
        f9 = w_in[foff + K.ad_ve_fpop(i, j, 9, Nx, Ny)]
        inv_rho = 1.0 / (f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9)
        ux[i, j] = (f2 - f4 + f6 - f7 - f8 + f9) * inv_rho
        uy[i, j] = (f3 - f5 + f6 + f7 - f8 - f9) * inv_rho
    end

    ux_face = zeros(Float64, Nx + 1, Ny)
    uy_face = zeros(Float64, Nx, Ny + 1)
    K.ad_ve_cell_velocity_to_faces_westprofile!(ux_face, uy_face, ux, uy, g, Nx, Ny, u_profile)

    east_xx = _east(E, psixx_in, Nx, Ny)
    east_xy = _east(E, psixy_in, Nx, Ny)
    east_yy = _east(E, psiyy_in, Nx, Ny)
    psixx_adv = _advect(V, psixx_in, ux_face, uy_face, is_solid, Nx, Ny, east_xx)
    psixy_adv = _advect(V, psixy_in, ux_face, uy_face, is_solid, Nx, Ny, east_xy)
    psiyy_adv = _advect(V, psiyy_in, ux_face, uy_face, is_solid, Nx, Ny, east_yy)

    dudx = zeros(Float64, Nx, Ny); dudy = zeros(Float64, Nx, Ny)
    dvdx = zeros(Float64, Nx, Ny); dvdy = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        gxx = K.ad_ve_deriv_x_2d(ux, is_solid, i, j, Nx, inv_dx, inv_2dx, wb, eb)
        gxy = K.ad_ve_deriv_y_2d(ux, is_solid, i, j, Ny, inv_dy, inv_2dy, sb, nb)
        gyx = K.ad_ve_deriv_x_2d(uy, is_solid, i, j, Nx, inv_dx, inv_2dx, wb, eb)
        gyy = K.ad_ve_deriv_y_2d(uy, is_solid, i, j, Ny, inv_dy, inv_2dy, sb, nb)
        gxx, gxy = K.ad_ve_apply_embedded_wall_gradient(gxx, gxy, ux, g.wall_nx, g.wall_ny,
                                                        g.wall_inv_distance_to_center, i, j)
        gyx, gyy = K.ad_ve_apply_embedded_wall_gradient(gyx, gyy, uy, g.wall_nx, g.wall_ny,
                                                        g.wall_inv_distance_to_center, i, j)
        dudx[i, j] = gxx; dudy[i, j] = gxy; dvdx[i, j] = gyx; dvdy[i, j] = gyy
    end

    psixx_p = zeros(Float64, Nx, Ny); psixy_p = zeros(Float64, Nx, Ny)
    psiyy_p = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        px = psixx_adv[i, j]; py = psixy_adv[i, j]; pyy = psiyy_adv[i, j]
        for _ in 1:p.n_substeps
            px, py, pyy = K.ad_ve_constitutive_step_log_2d(
                px, py, pyy, dudx[i, j], dudy[i, j], dvdx[i, j], dvdy[i, j],
                p.lambda, p.dt)
        end
        psixx_p[i, j] = px; psixy_p[i, j] = py; psiyy_p[i, j] = pyy
    end

    tauxx = zeros(Float64, Nx, Ny); tauxy = zeros(Float64, Nx, Ny)
    tauyy = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        txx, txy, tyy = K.ad_ve_stress_from_log_2d(psixx_p[i, j], psixy_p[i, j],
                                                   psiyy_p[i, j], p.prefactor)
        tauxx[i, j] = txx; tauxy[i, j] = txy; tauyy[i, j] = tyy
    end

    fx_poly = zeros(Float64, Nx, Ny); fy_poly = zeros(Float64, Nx, Ny)
    K.ad_ve_tensor_divergence_embedded!(fx_poly, fy_poly, tauxx, tauxy, tauyy, g, Nx, Ny,
                                        inv_dx, inv_dy)
    @inbounds for j in 1:Ny, i in 1:Nx
        if !is_solid[i, j]
            c = g.cell_fraction[i, j]
            fx_poly[i, j] *= c
            fy_poly[i, j] *= c
        end
    end

    fx_total = zeros(Float64, Nx, Ny); fy_total = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        if !is_solid[i, j]
            fx_total[i, j] = fx_poly[i, j] + p.Fx_body
            fy_total[i, j] = fy_poly[i, j]
        end
    end

    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            w_out[foff + K.ad_ve_fpop(i, j, 1, Nx, Ny)] = 4.0 / 9.0
            w_out[foff + K.ad_ve_fpop(i, j, 2, Nx, Ny)] = 1.0 / 9.0
            w_out[foff + K.ad_ve_fpop(i, j, 3, Nx, Ny)] = 1.0 / 9.0
            w_out[foff + K.ad_ve_fpop(i, j, 4, Nx, Ny)] = 1.0 / 9.0
            w_out[foff + K.ad_ve_fpop(i, j, 5, Nx, Ny)] = 1.0 / 9.0
            w_out[foff + K.ad_ve_fpop(i, j, 6, Nx, Ny)] = 1.0 / 36.0
            w_out[foff + K.ad_ve_fpop(i, j, 7, Nx, Ny)] = 1.0 / 36.0
            w_out[foff + K.ad_ve_fpop(i, j, 8, Nx, Ny)] = 1.0 / 36.0
            w_out[foff + K.ad_ve_fpop(i, j, 9, Nx, Ny)] = 1.0 / 36.0
            continue
        end

        fp1 = w_in[foff + K.ad_ve_fpop(i, j, 1, Nx, Ny)]
        fp2 = i > 1  ? w_in[foff + K.ad_ve_fpop(i-1, j, 2, Nx, Ny)] : w_in[foff + K.ad_ve_fpop(i, j, 4, Nx, Ny)]
        fp3 = j > 1  ? w_in[foff + K.ad_ve_fpop(i, j-1, 3, Nx, Ny)] : w_in[foff + K.ad_ve_fpop(i, j, 5, Nx, Ny)]
        fp4 = i < Nx ? w_in[foff + K.ad_ve_fpop(i+1, j, 4, Nx, Ny)] : w_in[foff + K.ad_ve_fpop(i, j, 2, Nx, Ny)]
        fp5 = j < Ny ? w_in[foff + K.ad_ve_fpop(i, j+1, 5, Nx, Ny)] : w_in[foff + K.ad_ve_fpop(i, j, 3, Nx, Ny)]
        fp6 = (i > 1 && j > 1)   ? w_in[foff + K.ad_ve_fpop(i-1, j-1, 6, Nx, Ny)] : w_in[foff + K.ad_ve_fpop(i, j, 8, Nx, Ny)]
        fp7 = (i < Nx && j > 1)  ? w_in[foff + K.ad_ve_fpop(i+1, j-1, 7, Nx, Ny)] : w_in[foff + K.ad_ve_fpop(i, j, 9, Nx, Ny)]
        fp8 = (i < Nx && j < Ny) ? w_in[foff + K.ad_ve_fpop(i+1, j+1, 8, Nx, Ny)] : w_in[foff + K.ad_ve_fpop(i, j, 6, Nx, Ny)]
        fp9 = (i > 1 && j < Ny)  ? w_in[foff + K.ad_ve_fpop(i-1, j+1, 9, Nx, Ny)] : w_in[foff + K.ad_ve_fpop(i, j, 7, Nx, Ny)]

        h1 = w_in[foff + K.ad_ve_fpop(i, j, 1, Nx, Ny)]
        h2 = w_in[foff + K.ad_ve_fpop(i, j, 2, Nx, Ny)]
        h3 = w_in[foff + K.ad_ve_fpop(i, j, 3, Nx, Ny)]
        h4 = w_in[foff + K.ad_ve_fpop(i, j, 4, Nx, Ny)]
        h5 = w_in[foff + K.ad_ve_fpop(i, j, 5, Nx, Ny)]
        h6 = w_in[foff + K.ad_ve_fpop(i, j, 6, Nx, Ny)]
        h7 = w_in[foff + K.ad_ve_fpop(i, j, 7, Nx, Ny)]
        h8 = w_in[foff + K.ad_ve_fpop(i, j, 8, Nx, Ny)]
        h9 = w_in[foff + K.ad_ve_fpop(i, j, 9, Nx, Ny)]
        qw2 = q_wall[i, j, 2]; (qw2 > 0.0) && (fp4 = K.ad_ve_libb_branch(qw2, h2, fp2, h4))
        qw4 = q_wall[i, j, 4]; (qw4 > 0.0) && (fp2 = K.ad_ve_libb_branch(qw4, h4, fp4, h2))
        qw3 = q_wall[i, j, 3]; (qw3 > 0.0) && (fp5 = K.ad_ve_libb_branch(qw3, h3, fp3, h5))
        qw5 = q_wall[i, j, 5]; (qw5 > 0.0) && (fp3 = K.ad_ve_libb_branch(qw5, h5, fp5, h3))
        qw6 = q_wall[i, j, 6]; (qw6 > 0.0) && (fp8 = K.ad_ve_libb_branch(qw6, h6, fp6, h8))
        qw8 = q_wall[i, j, 8]; (qw8 > 0.0) && (fp6 = K.ad_ve_libb_branch(qw8, h8, fp8, h6))
        qw7 = q_wall[i, j, 7]; (qw7 > 0.0) && (fp9 = K.ad_ve_libb_branch(qw7, h7, fp7, h9))
        qw9 = q_wall[i, j, 9]; (qw9 > 0.0) && (fp7 = K.ad_ve_libb_branch(qw9, h9, fp9, h7))

        rho = fp1 + fp2 + fp3 + fp4 + fp5 + fp6 + fp7 + fp8 + fp9
        inv_rho = 1.0 / rho
        ux_raw = (fp2 - fp4 + fp6 - fp7 - fp8 + fp9) * inv_rho
        uy_raw = (fp3 - fp5 + fp6 + fp7 - fp8 - fp9) * inv_rho

        fx = fx_total[i, j]; fy = fy_total[i, j]
        ux_c = ux_raw; uy_c = uy_raw
        if fx != 0.0 || fy != 0.0
            ux_c = (rho * ux_raw + fx / 2.0) * inv_rho
            uy_c = (rho * uy_raw + fy / 2.0) * inv_rho
        end
        usq = ux_c * ux_c + uy_c * uy_c
        feq1 = K.ad_ve_feq(1, rho, ux_c, uy_c, usq)
        feq2 = K.ad_ve_feq(2, rho, ux_c, uy_c, usq)
        feq3 = K.ad_ve_feq(3, rho, ux_c, uy_c, usq)
        feq4 = K.ad_ve_feq(4, rho, ux_c, uy_c, usq)
        feq5 = K.ad_ve_feq(5, rho, ux_c, uy_c, usq)
        feq6 = K.ad_ve_feq(6, rho, ux_c, uy_c, usq)
        feq7 = K.ad_ve_feq(7, rho, ux_c, uy_c, usq)
        feq8 = K.ad_ve_feq(8, rho, ux_c, uy_c, usq)
        feq9 = K.ad_ve_feq(9, rho, ux_c, uy_c, usq)
        a = 0.5 * (sp + sm)
        b = 0.5 * (sp - sm)
        guo_pref = 1.0 - sp / 2.0

        Sq1 = (4.0/9.0)  * ((-ux_c)*fx + (-uy_c)*fy) * 3.0
        Sq2 = (1.0/9.0)  * ((1.0-ux_c)*fx + (-uy_c)*fy) * 3.0 + (1.0/9.0)*ux_c*fx*9.0
        Sq3 = (1.0/9.0)  * ((-ux_c)*fx + (1.0-uy_c)*fy) * 3.0 + (1.0/9.0)*uy_c*fy*9.0
        Sq4 = (1.0/9.0)  * ((-1.0-ux_c)*fx + (-uy_c)*fy) * 3.0 + (1.0/9.0)*ux_c*fx*9.0
        Sq5 = (1.0/9.0)  * ((-ux_c)*fx + (-1.0-uy_c)*fy) * 3.0 + (1.0/9.0)*uy_c*fy*9.0
        Sq6 = (1.0/36.0) * ((1.0-ux_c)*fx + (1.0-uy_c)*fy) * 3.0 + (1.0/36.0)*(ux_c+uy_c)*(fx+fy)*9.0
        Sq7 = (1.0/36.0) * ((-1.0-ux_c)*fx + (1.0-uy_c)*fy) * 3.0 + (1.0/36.0)*(-ux_c+uy_c)*(-fx+fy)*9.0
        Sq8 = (1.0/36.0) * ((-1.0-ux_c)*fx + (-1.0-uy_c)*fy) * 3.0 + (1.0/36.0)*(-ux_c-uy_c)*(-fx-fy)*9.0
        Sq9 = (1.0/36.0) * ((1.0-ux_c)*fx + (-1.0-uy_c)*fy) * 3.0 + (1.0/36.0)*(ux_c-uy_c)*(fx-fy)*9.0

        w_out[foff + K.ad_ve_fpop(i, j, 1, Nx, Ny)] = fp1 - sp*(fp1-feq1) + guo_pref*Sq1
        w_out[foff + K.ad_ve_fpop(i, j, 2, Nx, Ny)] = fp2 - a*(fp2-feq2) - b*(fp4-feq4) + guo_pref*Sq2
        w_out[foff + K.ad_ve_fpop(i, j, 4, Nx, Ny)] = fp4 - a*(fp4-feq4) - b*(fp2-feq2) + guo_pref*Sq4
        w_out[foff + K.ad_ve_fpop(i, j, 3, Nx, Ny)] = fp3 - a*(fp3-feq3) - b*(fp5-feq5) + guo_pref*Sq3
        w_out[foff + K.ad_ve_fpop(i, j, 5, Nx, Ny)] = fp5 - a*(fp5-feq5) - b*(fp3-feq3) + guo_pref*Sq5
        w_out[foff + K.ad_ve_fpop(i, j, 6, Nx, Ny)] = fp6 - a*(fp6-feq6) - b*(fp8-feq8) + guo_pref*Sq6
        w_out[foff + K.ad_ve_fpop(i, j, 8, Nx, Ny)] = fp8 - a*(fp8-feq8) - b*(fp6-feq6) + guo_pref*Sq8
        w_out[foff + K.ad_ve_fpop(i, j, 7, Nx, Ny)] = fp7 - a*(fp7-feq7) - b*(fp9-feq9) + guo_pref*Sq7
        w_out[foff + K.ad_ve_fpop(i, j, 9, Nx, Ny)] = fp9 - a*(fp9-feq9) - b*(fp7-feq7) + guo_pref*Sq9
    end

    @inbounds for j in 2:(Ny-1)
        fp1 = w_in[foff + K.ad_ve_fpop(1, j,   1, Nx, Ny)]
        fp3 = w_in[foff + K.ad_ve_fpop(1, j-1, 3, Nx, Ny)]
        fp4 = w_in[foff + K.ad_ve_fpop(2, j,   4, Nx, Ny)]
        fp5 = w_in[foff + K.ad_ve_fpop(1, j+1, 5, Nx, Ny)]
        fp7 = w_in[foff + K.ad_ve_fpop(2, j-1, 7, Nx, Ny)]
        fp8 = w_in[foff + K.ad_ve_fpop(2, j+1, 8, Nx, Ny)]
        u_in = u_profile[j]
        rho_w = (fp1 + fp3 + fp5 + 2.0*(fp4 + fp7 + fp8)) / (1.0 - u_in)
        fp2 = fp4 + (2.0/3.0)*rho_w*u_in
        fp6 = fp8 - 0.5*(fp3 - fp5) + (1.0/6.0)*rho_w*u_in
        fp9 = fp7 + 0.5*(fp3 - fp5) + (1.0/6.0)*rho_w*u_in
        F = K.ad_ve_trt_collide_local(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, sp, sm)
        w_out[foff + K.ad_ve_fpop(1, j, 1, Nx, Ny)] = F[1]
        w_out[foff + K.ad_ve_fpop(1, j, 2, Nx, Ny)] = F[2]
        w_out[foff + K.ad_ve_fpop(1, j, 3, Nx, Ny)] = F[3]
        w_out[foff + K.ad_ve_fpop(1, j, 4, Nx, Ny)] = F[4]
        w_out[foff + K.ad_ve_fpop(1, j, 5, Nx, Ny)] = F[5]
        w_out[foff + K.ad_ve_fpop(1, j, 6, Nx, Ny)] = F[6]
        w_out[foff + K.ad_ve_fpop(1, j, 7, Nx, Ny)] = F[7]
        w_out[foff + K.ad_ve_fpop(1, j, 8, Nx, Ny)] = F[8]
        w_out[foff + K.ad_ve_fpop(1, j, 9, Nx, Ny)] = F[9]
        gp1 = w_in[foff + K.ad_ve_fpop(Nx,   j,   1, Nx, Ny)]
        gp2 = w_in[foff + K.ad_ve_fpop(Nx-1, j,   2, Nx, Ny)]
        gp3 = w_in[foff + K.ad_ve_fpop(Nx,   j-1, 3, Nx, Ny)]
        gp5 = w_in[foff + K.ad_ve_fpop(Nx,   j+1, 5, Nx, Ny)]
        gp6 = w_in[foff + K.ad_ve_fpop(Nx-1, j-1, 6, Nx, Ny)]
        gp9 = w_in[foff + K.ad_ve_fpop(Nx-1, j+1, 9, Nx, Ny)]
        u_x = -1.0 + (gp1 + gp3 + gp5 + 2.0*(gp2 + gp6 + gp9)) / rho_out
        gp4 = gp2 - (2.0/3.0)*rho_out*u_x
        gp7 = gp9 - 0.5*(gp3 - gp5) - (1.0/6.0)*rho_out*u_x
        gp8 = gp6 + 0.5*(gp3 - gp5) - (1.0/6.0)*rho_out*u_x
        G = K.ad_ve_trt_collide_local(gp1, gp2, gp3, gp4, gp5, gp6, gp7, gp8, gp9, sp, sm)
        w_out[foff + K.ad_ve_fpop(Nx, j, 1, Nx, Ny)] = G[1]
        w_out[foff + K.ad_ve_fpop(Nx, j, 2, Nx, Ny)] = G[2]
        w_out[foff + K.ad_ve_fpop(Nx, j, 3, Nx, Ny)] = G[3]
        w_out[foff + K.ad_ve_fpop(Nx, j, 4, Nx, Ny)] = G[4]
        w_out[foff + K.ad_ve_fpop(Nx, j, 5, Nx, Ny)] = G[5]
        w_out[foff + K.ad_ve_fpop(Nx, j, 6, Nx, Ny)] = G[6]
        w_out[foff + K.ad_ve_fpop(Nx, j, 7, Nx, Ny)] = G[7]
        w_out[foff + K.ad_ve_fpop(Nx, j, 8, Nx, Ny)] = G[8]
        w_out[foff + K.ad_ve_fpop(Nx, j, 9, Nx, Ny)] = G[9]
    end

    @inbounds for j in 1:Ny, i in 1:Nx
        k = K.ad_ve_lin(i, j, Nx)
        w_out[poff + k]      = psixx_p[i, j]
        w_out[poff + n + k]  = psixy_p[i, j]
        w_out[poff + 2n + k] = psiyy_p[i, j]
    end
    return nothing
end

# Reverse then forward over step_local! with the given switches; prints the
# transpose identity. `mask` converts g.is_solid (BitMatrix) or leaves it.
function run_local_step_case(name, c, V::Val, E::Val; mask=identity, converge=false,
                             forward_first=false)
    p, geom = build_case(c)
    w_star = base_state(c, p, geom; converge)
    is_solid = mask(geom.g.is_solid)
    println("[$name] mask=$(typeof(is_solid)) V=$V E=$E grid=$(c.Nx)x$(c.Ny)")
    len = length(w_star)
    # primal fidelity vs Kraken's own operator
    ref = zeros(len); loc = zeros(len)
    K.ad_ve_coupled_step!(ref, w_star, geom.g, geom.q_wall, p, geom.u_profile, 1.0, nothing)
    step_local!(loc, w_star, geom.g, is_solid, geom.q_wall, p, geom.u_profile, V, E)
    println("[$name] primal max|local-kraken| = $(maximum(abs.(loc .- ref)))")
    u, v = seeds(len)
    function reverse()
        println("[$name] reverse: compile+run"); flush(stdout)
        out = zeros(len); dout = copy(v); dw = zeros(len)
        Enzyme.autodiff(RT_REV, step_local!,
                        Enzyme.Duplicated(out, dout), Enzyme.Duplicated(copy(w_star), dw),
                        Enzyme.Const(geom.g), Enzyme.Const(is_solid), Enzyme.Const(geom.q_wall),
                        Enzyme.Const(p), Enzyme.Const(geom.u_profile),
                        Enzyme.Const(V), Enzyme.Const(E))
        println("[$name] REVERSE_OK"); flush(stdout)
        return dot(dw, u)
    end
    function forward()
        println("[$name] forward: compile+run"); flush(stdout)
        out = zeros(len); dout = zeros(len)
        Enzyme.autodiff(RT_FWD, step_local!,
                        Enzyme.Duplicated(out, dout), Enzyme.Duplicated(copy(w_star), copy(u)),
                        Enzyme.Const(geom.g), Enzyme.Const(is_solid), Enzyme.Const(geom.q_wall),
                        Enzyme.Const(p), Enzyme.Const(geom.u_profile),
                        Enzyme.Const(V), Enzyme.Const(E))
        println("[$name] FORWARD_OK"); flush(stdout)
        return dot(v, dout)
    end
    if forward_first
        vJu = forward(); Jtvu = reverse()
    else
        Jtvu = reverse(); vJu = forward()
    end
    rel = abs(vJu - Jtvu) / max(abs(vJu), eps(Float64))
    println("[$name] transpose_rel = $rel ", rel < 1e-10 ? "IDENTITY_OK" : "IDENTITY_FAIL")
    flush(stdout)
    return rel
end

# ---- round 2 switches ---------------------------------------------------------
@inline _advect(::Val{:split}, phi, ux_face, uy_face, is_solid, Nx, Ny, e) =
    advect_split(phi, ux_face, uy_face, is_solid, Nx, Ny, e)
# :none -> advection removed (psi_adv = psi_in); bisection only, changes numerics
@inline function _advect(::Val{:none}, phi, ux_face, uy_face, is_solid, Nx, Ny, e)
    adv = zeros(Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        adv[i, j] = phi[i, j]
    end
    return adv
end
