const _AD_ENZYME_LOAD_ERROR = "Load Enzyme to enable AD: `using " * "Enzyme`"

function _ad_dJdf(args...)
    error(_AD_ENZYME_LOAD_ERROR)
end

function _ad_vjp_GtT(args...)
    error(_AD_ENZYME_LOAD_ERROR)
end

function _ad_dqwall_terms(args...)
    error(_AD_ENZYME_LOAD_ERROR)
end

function _ad_dNudw(args...)
    error(_AD_ENZYME_LOAD_ERROR)
end

function _ad_thermal_vjp_GtT(args...)
    error(_AD_ENZYME_LOAD_ERROR)
end

function _ad_thermal_dqwall_terms(args...)
    error(_AD_ENZYME_LOAD_ERROR)
end

"""
    steady_shape_sensitivity(; qoi=:drag, ...)

Compute a steady shape sensitivity on the validated CPU Float64 AD paths.
Load Enzyme before calling this function.

For `qoi=:drag`, pass `Nx, Ny, radius, u_in, ν` (or `nu`), with
`wrt=:radius` and optional `cx, cy, ρ_out, tol, max_steps, inlet,
gmres_tol, adjoint_tol, fd_check, fd_h`.

For `qoi=:nusselt`, pass `N, Ra, Pr`, with `wrt=:wall_position` and
optional `L` (or `wall_param`/`wall_position`), `q_hot, q_cold, T_hot,
T_cold, tol, max_steps, gmres_tol, adjoint_tol, fd_check, fd_h`.
"""
function steady_shape_sensitivity(; qoi::Symbol=:drag,
                                    wrt::Symbol=(qoi === :nusselt ?
                                                 :wall_position : :radius),
                                    kwargs...)
    if qoi === :drag
        return _steady_drag_sensitivity(; wrt=wrt, kwargs...)
    elseif qoi === :nusselt
        return _steady_nusselt_sensitivity(; wrt=wrt, kwargs...)
    end
    throw(ArgumentError("unsupported qoi=$(qoi); supported qoi values are :drag and :nusselt"))
end

function _ad_pick_viscosity(ν, nu)
    if ν !== nothing && nu !== nothing
        throw(ArgumentError("pass only one of `ν` or `nu`"))
    end
    value = ν === nothing ? nu : ν
    value === nothing && throw(ArgumentError("missing required viscosity keyword `ν` or `nu`"))
    return Float64(value)
end

function _steady_drag_sensitivity(; Nx::Int, Ny::Int, radius::Real,
                                    u_in::Real,
                                    ν=nothing,
                                    nu=nothing,
                                    ρ_out::Real=1.0,
                                    cx::Union{Nothing,Real}=nothing,
                                    cy::Union{Nothing,Real}=nothing,
                                    wrt::Symbol=:radius,
                                    tol::Real=1e-12,
                                    fd_check::Bool=false,
                                    fd_h::Real=0.05,
                                    max_steps::Int=120_000,
                                    inlet::Symbol=:parabolic,
                                    gmres_tol::Real=1e-11,
                                    adjoint_tol::Real=AD_LINEAR_RES_TOL,
                                    kwargs...)
    isempty(kwargs) ||
        throw(ArgumentError("unsupported keyword(s): $(collect(keys(kwargs)))"))
    wrt === :radius ||
        throw(ArgumentError("unsupported wrt=$(wrt); supported wrt is :radius"))
    nu_f = _ad_pick_viscosity(ν, nu)

    forward = ad_forward_solve(; Nx=Nx, Ny=Ny, cx=cx, cy=cy, radius=radius,
                               u_in=u_in, nu=nu_f, inlet=inlet, rho_out=ρ_out,
                               tol=tol, max_steps=max_steps)
    forward.converged ||
        error("steady AD forward did not converge: residual=$(forward.residual), n_iter=$(forward.n_iter)")

    qoi_value = forward.Cd
    rhs = _ad_dJdf(forward.f_star, forward.q_wall, forward.u_ref,
                  forward.D, forward.Nx, forward.Ny)
    apply_GtT = v -> _ad_vjp_GtT(forward.f_star, v, forward.q_wall,
                                 forward.is_solid, forward.u_profile,
                                 forward.rho_out, forward.s_plus,
                                 forward.s_minus, forward.Nx, forward.Ny)
    adj = gmres_adjoint(apply_GtT, rhs; tol=gmres_tol, linear_tol=adjoint_tol)
    adj.converged ||
        error("steady AD adjoint did not converge: solver=$(adj.solver), linres=$(adj.linres)")

    dq_dR = dq_wall_dR_cylinder(forward.Nx, forward.Ny, forward.cx, forward.cy,
                                forward.radius; FT=Float64)
    qwall_terms = _ad_dqwall_terms(forward.f_star, adj.lambda, forward.q_wall,
                                  forward.is_solid, forward.u_profile,
                                  forward.rho_out, forward.s_plus,
                                  forward.s_minus, forward.Nx, forward.Ny,
                                  forward.u_ref, forward.D, dq_dR)
    terms = ad_assemble_radius_terms(qoi_value, forward.radius, dq_dR,
                                     qwall_terms.explicit, qwall_terms.implicit)

    fd = if fd_check
        fd_res = ad_fd_dCd_dR(forward; h=fd_h, tol=tol, max_steps=max_steps)
        (; fd_res..., relerr=abs(terms.gradient - fd_res.value) /
                          max(abs(fd_res.value), eps(Float64)))
    else
        nothing
    end

    solver = (;
        method=adj.solver,
        n_iter=adj.n_iter,
        rhohat=adj.rhohat,
        converged=adj.converged,
        linres=adj.linres,
        note=adj.note,
    )
    forward_info = (;
        n_iter=forward.n_iter,
        residual=forward.residual,
        converged=forward.converged,
        cut_links=ad_cut_link_count(forward.q_wall),
        solid_nodes=ad_solid_count(forward.is_solid),
    )

    return (;
        value=qoi_value,
        gradient=terms.gradient,
        qoi_value=qoi_value,
        solver=solver,
        terms=terms,
        n_iter=forward.n_iter,
        residual=forward.residual,
        forward=forward_info,
        fd_check=fd,
    )
end

function _ad_pick_wall_position(N::Int, L, wall_param, wall_position,
                                q_hot::Real, q_cold::Real)
    supplied = 0
    value = nothing
    for candidate in (L, wall_param, wall_position)
        if candidate !== nothing
            supplied += 1
            value = candidate
        end
    end
    supplied > 1 &&
        throw(ArgumentError("pass only one of `L`, `wall_param`, or `wall_position`"))
    return Float64(value === nothing ?
                   Float64(N - 1) + Float64(q_hot) + Float64(q_cold) :
                   value)
end

function _steady_nusselt_sensitivity(; N::Int,
                                       Ra::Real,
                                       Pr::Real,
                                       L=nothing,
                                       wall_param=nothing,
                                       wall_position=nothing,
                                       q_hot::Real=0.5,
                                       q_cold::Real=0.7,
                                       T_hot::Real=1.0,
                                       T_cold::Real=0.0,
                                       wrt::Symbol=:wall_position,
                                       tol::Real=1e-11,
                                       fd_check::Bool=false,
                                       fd_h::Real=0.01,
                                       max_steps::Int=450_000,
                                       gmres_tol::Real=1e-10,
                                       adjoint_tol::Real=AD_LINEAR_RES_TOL,
                                       gmres_restart::Int=640,
                                       gmres_max_restarts::Int=16,
                                       kwargs...)
    isempty(kwargs) ||
        throw(ArgumentError("unsupported keyword(s): $(collect(keys(kwargs)))"))
    wrt === :wall_position ||
        throw(ArgumentError("unsupported wrt=$(wrt); supported wrt is :wall_position"))
    L_f = _ad_pick_wall_position(N, L, wall_param, wall_position, q_hot, q_cold)

    forward = ad_thermal_forward_solve(; N=N, Ra=Ra, Pr=Pr, L=L_f,
                                       q_hot=q_hot, q_cold=q_cold,
                                       T_hot=T_hot, T_cold=T_cold,
                                       tol=tol, max_steps=max_steps)
    forward.converged ||
        error("steady thermal AD forward did not converge: residual=$(forward.residual), n_iter=$(forward.n_iter)")

    qoi_value = forward.Nu
    rhs = _ad_dNudw(forward.w_star, forward.q_wall, forward.params)
    apply_GtT = v -> _ad_thermal_vjp_GtT(forward.w_star, v,
                                         forward.q_wall, forward.q_wall,
                                         forward.params)
    rhohat = ad_richardson_rhohat(apply_GtT, rhs; n_iter=100)
    mass = ad_thermal_mass_gradient(forward.params)
    adj = ad_gauge_augmented_adjoint(apply_GtT, rhs, mass;
                                     tol=gmres_tol,
                                     restart=min(gmres_restart, length(rhs) + 1),
                                     max_restarts=gmres_max_restarts,
                                     linear_tol=adjoint_tol,
                                     rhohat=rhohat)
    adj.converged ||
        error("steady thermal AD adjoint did not converge: solver=$(adj.solver), linres=$(adj.linres), original_linres=$(adj.original_linres)")

    qwall_terms = _ad_thermal_dqwall_terms(forward.w_star, adj.lambda,
                                           forward.q_wall, forward.q_wall,
                                           forward.params, forward.dq_dL)
    terms = ad_assemble_wall_position_terms(qwall_terms.explicit,
                                            qwall_terms.implicit,
                                            forward.dq_dL)

    fd = if fd_check
        fd_res = ad_fd_dNu_dL(forward; h=fd_h, tol=tol, max_steps=max_steps)
        (; fd_res..., relerr=abs(terms.gradient - fd_res.value) /
                          max(abs(fd_res.value), eps(Float64)))
    else
        nothing
    end

    solver = (;
        method=adj.solver,
        n_iter=adj.n_iter,
        rhohat=adj.rhohat,
        converged=adj.converged,
        linres=adj.linres,
        original_linres=adj.original_linres,
        gauge=adj.gauge,
        eta=adj.eta,
        note=adj.note,
    )
    forward_info = (;
        n_iter=forward.n_iter,
        residual=forward.residual,
        converged=forward.converged,
        cut_links=ad_cut_link_count(forward.q_wall),
        q_hot=forward.q_hot,
        q_cold=forward.q_cold,
        L=forward.L,
    )

    return (;
        value=qoi_value,
        gradient=terms.gradient,
        qoi_value=qoi_value,
        solver=solver,
        terms=terms,
        n_iter=forward.n_iter,
        residual=forward.residual,
        forward=forward_info,
        fd_check=fd,
    )
end
