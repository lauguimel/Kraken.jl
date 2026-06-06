module KrakenADExt

using Kraken
using Enzyme

import Kraken: _ad_dJdf, _ad_vjp_GtT, _ad_dqwall_terms
import Kraken: _ad_dNudw, _ad_thermal_vjp_GtT, _ad_thermal_dqwall_terms
import Kraken: _ad_ve_dJdw, _ad_ve_vjp_GtT, _ad_ve_dGdR_jvp

_short_error(err) = first(sprint(showerror, err), min(360, lastindex(sprint(showerror, err))))
_short_string(msg) = first(msg, min(360, lastindex(msg)))

function _ad_dJdf(f, q_wall, u_ref, D, Nx::Int, Ny::Int)
    djdf = zeros(Float64, size(f))
    Enzyme.autodiff(Enzyme.Reverse, Kraken.cd_pure, Enzyme.Active,
                    Enzyme.Duplicated(copy(f), djdf),
                    Enzyme.Const(q_wall), Enzyme.Const(u_ref),
                    Enzyme.Const(D), Enzyme.Const(Nx), Enzyme.Const(Ny))
    return djdf
end

function _ad_vjp_GtT(f_star, v, q_wall, is_solid, u_profile, rho_out,
                     s_plus, s_minus, Nx::Int, Ny::Int)
    out = zeros(Float64, size(f_star))
    dout = copy(v)
    df = zeros(Float64, size(f_star))
    Enzyme.autodiff(Enzyme.Reverse, Kraken.ad_step!,
                    Enzyme.Duplicated(out, dout),
                    Enzyme.Duplicated(copy(f_star), df),
                    Enzyme.Const(q_wall), Enzyme.Const(is_solid),
                    Enzyme.Const(u_profile), Enzyme.Const(rho_out),
                    Enzyme.Const(s_plus), Enzyme.Const(s_minus),
                    Enzyme.Const(Nx), Enzyme.Const(Ny))
    return df
end

function _dCd_dqwall_taped(f, q_wall, u_ref, D, Nx::Int, Ny::Int)
    dqw = zeros(Float64, size(q_wall))
    Enzyme.autodiff(Enzyme.Reverse, Kraken.cd_pure, Enzyme.Active,
                    Enzyme.Const(f), Enzyme.Duplicated(copy(q_wall), dqw),
                    Enzyme.Const(u_ref), Enzyme.Const(D),
                    Enzyme.Const(Nx), Enzyme.Const(Ny))
    return dqw
end

function _lambda_dot_dG_dqwall_with_mode(mode, f_star, lambda, q_wall,
                                         is_solid, u_profile, rho_out,
                                         s_plus, s_minus, Nx::Int, Ny::Int)
    out = zeros(Float64, size(f_star))
    dout = copy(lambda)
    dqw = zeros(Float64, size(q_wall))
    Enzyme.autodiff(mode, Kraken.ad_step!,
                    Enzyme.Duplicated(out, dout),
                    Enzyme.Const(f_star),
                    Enzyme.Duplicated(copy(q_wall), dqw),
                    Enzyme.Const(is_solid), Enzyme.Const(u_profile),
                    Enzyme.Const(rho_out), Enzyme.Const(s_plus),
                    Enzyme.Const(s_minus), Enzyme.Const(Nx), Enzyme.Const(Ny))
    return dqw
end

function _dCd_dqwall(f, q_wall, u_ref, D, Nx::Int, Ny::Int)
    try
        dqw = _dCd_dqwall_taped(f, q_wall, u_ref, D, Nx, Ny)
        if all(isfinite, dqw) && Kraken.cut_cotangent_norm(dqw, q_wall) > 0.0
            return (; dqw=dqw, path="taped", error="")
        end
        return (; dqw=Kraken.fd_dCd_dqwall(f, q_wall, u_ref, D, Nx, Ny),
                path="Fallback A (FD-on-q_wall)",
                error="taped explicit Cd/q_wall returned nonfinite or zero cut-link cotangent")
    catch err
        return (; dqw=Kraken.fd_dCd_dqwall(f, q_wall, u_ref, D, Nx, Ny),
                path="Fallback A (FD-on-q_wall)", error=_short_error(err))
    end
end

function _lambda_dot_dG_dqwall(f_star, lambda, q_wall, is_solid, u_profile,
                               rho_out, s_plus, s_minus, Nx::Int, Ny::Int)
    plain_err = ""
    runtime_err = ""
    try
        dqw = _lambda_dot_dG_dqwall_with_mode(Enzyme.Reverse, f_star, lambda,
                                              q_wall, is_solid, u_profile,
                                              rho_out, s_plus, s_minus, Nx, Ny)
        if all(isfinite, dqw) && Kraken.cut_cotangent_norm(dqw, q_wall) > 0.0
            return (; dqw=dqw, path="taped", error="")
        end
        plain_err = "plain Reverse returned nonfinite or zero cut-link cotangent"
    catch err
        plain_err = sprint(showerror, err)
    end

    try
        if !isdefined(Enzyme, :set_runtime_activity)
            error("Enzyme.set_runtime_activity is unavailable")
        end
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
        dqw = _lambda_dot_dG_dqwall_with_mode(mode, f_star, lambda,
                                              q_wall, is_solid, u_profile,
                                              rho_out, s_plus, s_minus, Nx, Ny)
        if all(isfinite, dqw) && Kraken.cut_cotangent_norm(dqw, q_wall) > 0.0
            return (; dqw=dqw, path="runtime-activity", error=plain_err)
        end
        runtime_err = "runtime-activity returned nonfinite or zero cut-link cotangent"
    catch err
        runtime_err = sprint(showerror, err)
    end

    dqw = Kraken.fd_lambda_dot_dG_dqwall(f_star, lambda, q_wall, is_solid,
                                         u_profile, rho_out, s_plus, s_minus,
                                         Nx, Ny)
    msg = "plain Reverse: " * _short_string(plain_err) *
          " | runtime-activity: " * _short_string(runtime_err)
    return (; dqw=dqw, path="Fallback A (FD-on-q_wall)", error=msg)
end

function _ad_dqwall_terms(f_star, lambda, q_wall, is_solid, u_profile,
                          rho_out, s_plus, s_minus, Nx::Int, Ny::Int,
                          u_ref, D, dq_dR)
    explicit = _dCd_dqwall(f_star, q_wall, u_ref, D, Nx, Ny)
    implicit = _lambda_dot_dG_dqwall(f_star, lambda, q_wall, is_solid,
                                     u_profile, rho_out, s_plus, s_minus,
                                     Nx, Ny)

    explicit_q_R = Kraken.ad_dot_arrays(explicit.dqw, dq_dR)
    explicit_dirfd = Kraken.directional_fd_Cd_qwall(f_star, q_wall, dq_dR,
                                                    u_ref, D, Nx, Ny)
    explicit_rel = Kraken.ad_rel_delta(explicit_q_R, explicit_dirfd)
    if explicit_rel > 1e-5
        explicit = (; dqw=Kraken.fd_dCd_dqwall(f_star, q_wall, u_ref, D, Nx, Ny),
                    path="Fallback A (FD-on-q_wall after directional mismatch)",
                    error="directional contraction mismatch",
                    directional_fd=explicit_dirfd,
                    directional_rel=explicit_rel)
    else
        explicit = (; explicit..., directional_fd=explicit_dirfd,
                    directional_rel=explicit_rel)
    end

    implicit_R = Kraken.ad_dot_arrays(implicit.dqw, dq_dR)
    implicit_dirfd = Kraken.directional_fd_lambdaG_qwall(f_star, lambda,
                                                         q_wall, dq_dR,
                                                         is_solid, u_profile,
                                                         rho_out, s_plus,
                                                         s_minus, Nx, Ny)
    implicit_rel = Kraken.ad_rel_delta(implicit_R, implicit_dirfd)
    if implicit_rel > 1e-5
        implicit = (; dqw=Kraken.fd_lambda_dot_dG_dqwall(f_star, lambda,
                                                         q_wall, is_solid,
                                                         u_profile, rho_out,
                                                         s_plus, s_minus,
                                                         Nx, Ny),
                    path="Fallback A (FD-on-q_wall after directional mismatch)",
                    error="directional contraction mismatch",
                    directional_fd=implicit_dirfd,
                    directional_rel=implicit_rel)
    else
        implicit = (; implicit..., directional_fd=implicit_dirfd,
                    directional_rel=implicit_rel)
    end

    return (; explicit=explicit, implicit=implicit)
end

function _ad_dNudw(w_star, q_wall, p)
    dwd = zeros(Float64, length(w_star))
    Enzyme.autodiff(Enzyme.Reverse, Kraken.nu_pure, Enzyme.Active,
                    Enzyme.Duplicated(copy(w_star), dwd),
                    Enzyme.Const(q_wall),
                    Enzyme.Const(p))
    return dwd
end

function _ad_thermal_vjp_GtT(w_star, v, q_flow, q_therm, p)
    out = zeros(Float64, length(w_star))
    dout = copy(v)
    dw = zeros(Float64, length(w_star))
    Enzyme.autodiff(Enzyme.Reverse, Kraken.ad_thermal_cut_step!,
                    Enzyme.Duplicated(out, dout),
                    Enzyme.Duplicated(copy(w_star), dw),
                    Enzyme.Const(q_flow),
                    Enzyme.Const(q_therm),
                    Enzyme.Const(p))
    return dw
end

function _ad_dNu_dqwall(w_star, q_wall, p)
    dqw = zeros(Float64, size(q_wall))
    Enzyme.autodiff(Enzyme.Reverse, Kraken.nu_pure, Enzyme.Active,
                    Enzyme.Const(w_star),
                    Enzyme.Duplicated(copy(q_wall), dqw),
                    Enzyme.Const(p))
    return (; dqw=dqw, path="taped", error="")
end

function _lambda_dot_thermal_dG_dqwalls(w_star, lambda, q_flow, q_therm, p)
    out = zeros(Float64, length(w_star))
    dout = copy(lambda)
    dqw_flow = zeros(Float64, size(q_flow))
    dqw_therm = zeros(Float64, size(q_therm))
    Enzyme.autodiff(Enzyme.Reverse, Kraken.ad_thermal_cut_step!,
                    Enzyme.Duplicated(out, dout),
                    Enzyme.Const(w_star),
                    Enzyme.Duplicated(copy(q_flow), dqw_flow),
                    Enzyme.Duplicated(copy(q_therm), dqw_therm),
                    Enzyme.Const(p))
    return (; flow=dqw_flow, thermal=dqw_therm, path="taped", error="")
end

function _ad_thermal_dqwall_terms(w_star, lambda, q_flow, q_therm, p, dq_dL)
    explicit = _ad_dNu_dqwall(w_star, q_therm, p)
    implicit = _lambda_dot_thermal_dG_dqwalls(w_star, lambda, q_flow, q_therm, p)
    implicit_L = Kraken.ad_dot_arrays(implicit.flow, dq_dL) +
                 Kraken.ad_dot_arrays(implicit.thermal, dq_dL)
    dirfd = Kraken.directional_fd_thermal_lambdaG_qwalls(w_star, lambda,
                                                         q_flow, q_therm,
                                                         dq_dL, p)
    rel = Kraken.ad_rel_delta(implicit_L, dirfd)
    return (; explicit=explicit,
            implicit=(; implicit..., directional_fd=dirfd,
                      directional_rel=rel))
end

# ============================================================================
# Viscoelastic (Cd_polymer) shape-adjoint seams. Mirror of the validated scratch
# chain (ve_ad_c1.jl dJ_dw_enzyme, ve_ad_c3_matched.jl c3m_vjp, ve_ad_c3_analytic.jl
# dGdR_analytic_jvp). The tapeable plain-Julia operators live in src/ad/ad_ve_*.jl;
# `using Kraken` stays Enzyme-free. `ADVECoupledParams`, `ADVEEmbeddedGeom`,
# `ADVEWallPoint` are Const (the EmbeddedGeom struct of arrays / params never carry
# a state cotangent here). The geometry forward-JVP seeds the EmbeddedGeom shadow
# (`Duplicated(g, dg)`) + the q_wall shadow (`Duplicated(q_wall, dq_wall)`).
# ============================================================================

# Reverse of the QoI ad_ve_J_fx wrt the stacked state w -> dJ/dw (psi-only; the
# f-block cotangent is structurally zero since J reads only the psi block).
function _ad_ve_dJdw(w_star, pts, g, p)
    djdw = zeros(Float64, length(w_star))
    Enzyme.autodiff(Enzyme.Reverse, Kraken.ad_ve_J_fx, Enzyme.Active,
                    Enzyme.Duplicated(copy(w_star), djdw),
                    Enzyme.Const(pts), Enzyme.Const(g), Enzyme.Const(p))
    return djdw
end

# dG^T . v : reverse over the coupled VE one-step operator ad_ve_coupled_step!.
# set_runtime_activity(Reverse) (the scratch c3m_vjp default) handles the
# is_solid-masked branches / mixed-activity geometry struct.
function _ad_ve_vjp_GtT(w_star, v, g, q_wall, u_profile, p)
    out = zeros(Float64, length(w_star))
    dout = copy(v)
    dw = zeros(Float64, length(w_star))
    mode = isdefined(Enzyme, :set_runtime_activity) ?
           Enzyme.set_runtime_activity(Enzyme.Reverse) : Enzyme.Reverse
    Enzyme.autodiff(mode, Kraken.ad_ve_coupled_step!,
                    Enzyme.Duplicated(out, dout),
                    Enzyme.Duplicated(copy(w_star), dw),
                    Enzyme.Const(g), Enzyme.Const(q_wall),
                    Enzyme.Const(p), Enzyme.Const(u_profile),
                    Enzyme.Const(1.0), Enzyme.Const(nothing))
    return dw
end

# dG/dR : forward-JVP of ad_ve_coupled_step! with the state seed = 0 and the
# geometry/q_wall shadows seeded by the analytic d(geom)/dR. One forward pass.
function _ad_ve_dGdR_jvp(w_star, g, dg, q_wall, dq_wall, u_profile, p)
    out = zeros(Float64, length(w_star))
    dout = zeros(Float64, length(w_star))
    dw = zeros(Float64, length(w_star))          # geometry-only JVP: no state seed
    mode = isdefined(Enzyme, :set_runtime_activity) ?
           Enzyme.set_runtime_activity(Enzyme.Forward) : Enzyme.Forward
    Enzyme.autodiff(mode, Kraken.ad_ve_coupled_step!,
                    Enzyme.Duplicated(out, dout),
                    Enzyme.Duplicated(copy(w_star), dw),
                    Enzyme.Duplicated(g, dg),
                    Enzyme.Duplicated(copy(q_wall), copy(dq_wall)),
                    Enzyme.Const(p), Enzyme.Const(u_profile),
                    Enzyme.Const(1.0), Enzyme.Const(nothing))
    return dout
end

end # module KrakenADExt
