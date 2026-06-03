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

"""
    steady_shape_sensitivity(; Nx, Ny, radius, u_in, nu/ν,
                              qoi=:drag, wrt=:radius, tol=1e-12)

Compute the steady cylinder drag sensitivity `dCd/dR` on the validated
CPU Float64 AD path. Load Enzyme before calling this function.
"""
function steady_shape_sensitivity(; Nx::Int, Ny::Int, radius::Real,
                                    u_in::Real,
                                    ν::Real,
                                    ρ_out::Real=1.0,
                                    cx::Union{Nothing,Real}=nothing,
                                    cy::Union{Nothing,Real}=nothing,
                                    qoi::Symbol=:drag,
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
    qoi === :drag ||
        throw(ArgumentError("unsupported qoi=$(qoi); supported qoi is :drag"))
    wrt === :radius ||
        throw(ArgumentError("unsupported wrt=$(wrt); supported wrt is :radius"))

    forward = ad_forward_solve(; Nx=Nx, Ny=Ny, cx=cx, cy=cy, radius=radius,
                               u_in=u_in, nu=ν, inlet=inlet, rho_out=ρ_out,
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

