function ad_dot_arrays(a, b)
    s = 0.0
    @inbounds for idx in eachindex(a, b)
        s += a[idx] * b[idx]
    end
    return s
end

ad_rel_delta(a, b) = abs(a - b) / max(abs(b), eps(Float64))
ad_cut_link_count(q_wall) = count(x -> x > 0.0, q_wall)
ad_solid_count(is_solid) = count(identity, is_solid)

function ad_assemble_radius_terms(qoi_value::Real, radius::Real, dq_dR,
                                  explicit, implicit)
    explicit_qwall = ad_dot_arrays(explicit.dqw, dq_dR)
    direct_D = -Float64(qoi_value) / Float64(radius)
    implicit_qwall = ad_dot_arrays(implicit.dqw, dq_dR)
    gradient = explicit_qwall + direct_D + implicit_qwall
    return (;
        explicit_qwall=explicit_qwall,
        direct_D=direct_D,
        implicit_qwall=implicit_qwall,
        gradient=gradient,
        explicit_path=explicit.path,
        implicit_path=implicit.path,
        explicit_directional_fd=hasproperty(explicit, :directional_fd) ? explicit.directional_fd : NaN,
        explicit_directional_rel=hasproperty(explicit, :directional_rel) ? explicit.directional_rel : NaN,
        implicit_directional_fd=hasproperty(implicit, :directional_fd) ? implicit.directional_fd : NaN,
        implicit_directional_rel=hasproperty(implicit, :directional_rel) ? implicit.directional_rel : NaN,
    )
end

function ad_fd_dCd_dR(base; h::Real=0.05, tol::Real=1e-12, max_steps::Int=120_000)
    hp = Float64(h)
    plus = ad_forward_solve(; Nx=base.Nx, Ny=base.Ny, cx=base.cx, cy=base.cy,
                            radius=base.radius + hp, u_in=base.u_in,
                            nu=base.nu, inlet=base.inlet, rho_out=base.rho_out,
                            tol=tol, max_steps=max_steps, f_init=base.f_star)
    minus = ad_forward_solve(; Nx=base.Nx, Ny=base.Ny, cx=base.cx, cy=base.cy,
                             radius=base.radius - hp, u_in=base.u_in,
                             nu=base.nu, inlet=base.inlet, rho_out=base.rho_out,
                             tol=tol, max_steps=max_steps, f_init=base.f_star)
    value = (plus.Cd - minus.Cd) / (2.0 * hp)
    return (;
        value=value,
        h=hp,
        cd_plus=plus.Cd,
        cd_minus=minus.Cd,
        plus_converged=plus.converged,
        minus_converged=minus.converged,
        plus_residual=plus.residual,
        minus_residual=minus.residual,
        cut_plus=ad_cut_link_count(plus.q_wall),
        cut_minus=ad_cut_link_count(minus.q_wall),
        solid_plus=ad_solid_count(plus.is_solid),
        solid_minus=ad_solid_count(minus.is_solid),
    )
end
