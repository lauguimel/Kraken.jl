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

function ad_cavity_wall_geometry(Nx::Int, Ny::Int, L::Real; q_hot::Real=0.5)
    qh = Float64(q_hot)
    qc = Float64(L) - (Float64(Nx - 1) + qh)
    if !(0.0 < qh <= 1.0 && 0.0 < qc <= 1.0)
        throw(ArgumentError("L=$(Float64(L)) gives q_hot=$(qh), q_cold=$(qc); expected both in (0, 1]"))
    end

    q_wall = zeros(Float64, Nx, Ny, 9)
    dq_dL = zeros(Float64, Nx, Ny, 9)
    @inbounds for j in 1:Ny
        q_wall[1, j, 4] = qh
        q_wall[1, j, 7] = qh
        q_wall[1, j, 8] = qh

        q_wall[Nx, j, 2] = qc
        q_wall[Nx, j, 6] = qc
        q_wall[Nx, j, 9] = qc

        dq_dL[Nx, j, 2] = 1.0
        dq_dL[Nx, j, 6] = 1.0
        dq_dL[Nx, j, 9] = 1.0
    end
    return (; q_wall=q_wall, dq_dL=dq_dL, q_hot=qh, q_cold=qc,
            x_hot=0.0, x_cold=Float64(L))
end

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

function ad_assemble_wall_position_terms(explicit, implicit, dq_dL)
    explicit_qwall = ad_dot_arrays(explicit.dqw, dq_dL)
    flow_qwall = ad_dot_arrays(implicit.flow, dq_dL)
    thermal_qwall = ad_dot_arrays(implicit.thermal, dq_dL)
    gradient = explicit_qwall + flow_qwall + thermal_qwall
    return (;
        explicit_qwall=explicit_qwall,
        flow_qwall=flow_qwall,
        thermal_qwall=thermal_qwall,
        implicit_qwall=flow_qwall + thermal_qwall,
        gradient=gradient,
        explicit_path=explicit.path,
        implicit_path=implicit.path,
        flow_moving_norm=moving_cut_cotangent_norm(implicit.flow, dq_dL),
        thermal_moving_norm=moving_cut_cotangent_norm(implicit.thermal, dq_dL),
    )
end

function ad_fd_dNu_dL(base; h::Real=0.01, tol::Real=1e-11,
                      max_steps::Int=450_000)
    hp = Float64(h)
    plus = ad_thermal_forward_solve(; N=base.N, Ra=base.Ra, Pr=base.Pr,
                                    L=base.L + hp, q_hot=base.q_hot,
                                    T_hot=base.params.T_hot,
                                    T_cold=base.params.T_cold,
                                    tol=tol, max_steps=max_steps,
                                    w_init=base.w_star)
    minus = ad_thermal_forward_solve(; N=base.N, Ra=base.Ra, Pr=base.Pr,
                                     L=base.L - hp, q_hot=base.q_hot,
                                     T_hot=base.params.T_hot,
                                     T_cold=base.params.T_cold,
                                     tol=tol, max_steps=max_steps,
                                     w_init=base.w_star)
    value = (plus.Nu - minus.Nu) / (2.0 * hp)
    return (;
        value=value,
        h=hp,
        Nu_plus=plus.Nu,
        Nu_minus=minus.Nu,
        plus_converged=plus.converged,
        minus_converged=minus.converged,
        plus_residual=plus.residual,
        minus_residual=minus.residual,
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
