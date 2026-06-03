# BIT-MIRROR of production compute_drag_libb_mei_2d for stationary walls.

function cd_pure(f, q_wall, u_ref, D, Nx::Int, Ny::Int)
    Fx = 0.0
    @inbounds for j in 1:Ny, i in 1:Nx
        for q in 2:9
            q_w = q_wall[i, j, q]
            q_w > 0.0 || continue
            qbar = AD_OPP[q]
            im = i - Int(AD_CXV[q])
            jm = j - Int(AD_CYV[q])
            fp_q_back = (1 <= im <= Nx && 1 <= jm <= Ny) ? f[im, jm, q] : f[i, j, qbar]
            fq_here = f[i, j, q]
            fqbar_here = f[i, j, qbar]
            arriving = if q_w <= 0.5
                2.0 * q_w * fq_here + (1.0 - 2.0 * q_w) * fp_q_back
            else
                inv2q = 1.0 / (2.0 * q_w)
                inv2q * fq_here + (1.0 - inv2q) * fqbar_here
            end
            Fx += AD_CXV[q] * (fq_here + arriving)
        end
    end
    return 2.0 * Fx / (u_ref * u_ref * D)
end

function cd_production(f, q_wall, u_ref, D, Nx::Int, Ny::Int)
    uw_x = zeros(Float64, size(q_wall))
    uw_y = zeros(Float64, size(q_wall))
    drag = compute_drag_libb_mei_2d(f, q_wall, uw_x, uw_y, Nx, Ny)
    return 2.0 * Float64(drag.Fx) / (u_ref * u_ref * D)
end

function fd_dCd_dqwall(f, q_wall, u_ref, D, Nx::Int, Ny::Int)
    dqw = zeros(Float64, size(q_wall))
    eps_q = 1e-6
    @inbounds for idx in CartesianIndices(q_wall)
        q_wall[idx] > 0.0 || continue
        qp = copy(q_wall)
        qm = copy(q_wall)
        qp[idx] += eps_q
        qm[idx] -= eps_q
        vp = cd_pure(f, qp, u_ref, D, Nx, Ny)
        vm = cd_pure(f, qm, u_ref, D, Nx, Ny)
        dqw[idx] = (vp - vm) / (2.0 * eps_q)
    end
    return dqw
end

function ad_lambda_dot_G_value(f_star, lambda, q_wall, is_solid, u_profile,
                               rho_out, s_plus, s_minus, Nx::Int, Ny::Int)
    out = zeros(Float64, size(f_star))
    ad_step!(out, f_star, q_wall, is_solid, u_profile, rho_out,
             s_plus, s_minus, Nx, Ny)
    return ad_dot_arrays(lambda, out)
end

function fd_lambda_dot_dG_dqwall(f_star, lambda, q_wall, is_solid, u_profile,
                                 rho_out, s_plus, s_minus, Nx::Int, Ny::Int)
    dqw = zeros(Float64, size(q_wall))
    eps_q = 1e-6
    @inbounds for idx in CartesianIndices(q_wall)
        q_wall[idx] > 0.0 || continue
        qp = copy(q_wall)
        qm = copy(q_wall)
        qp[idx] += eps_q
        qm[idx] -= eps_q
        vp = ad_lambda_dot_G_value(f_star, lambda, qp, is_solid, u_profile,
                                   rho_out, s_plus, s_minus, Nx, Ny)
        vm = ad_lambda_dot_G_value(f_star, lambda, qm, is_solid, u_profile,
                                   rho_out, s_plus, s_minus, Nx, Ny)
        dqw[idx] = (vp - vm) / (2.0 * eps_q)
    end
    return dqw
end

function directional_fd_Cd_qwall(f, q_wall, dq_dR, u_ref, D, Nx::Int, Ny::Int)
    eps_R = 1e-5
    qp = copy(q_wall)
    qm = copy(q_wall)
    @inbounds for idx in eachindex(q_wall, dq_dR)
        qp[idx] += eps_R * dq_dR[idx]
        qm[idx] -= eps_R * dq_dR[idx]
    end
    return (cd_pure(f, qp, u_ref, D, Nx, Ny) -
            cd_pure(f, qm, u_ref, D, Nx, Ny)) / (2.0 * eps_R)
end

function directional_fd_lambdaG_qwall(f_star, lambda, q_wall, dq_dR,
                                      is_solid, u_profile, rho_out,
                                      s_plus, s_minus, Nx::Int, Ny::Int)
    eps_R = 1e-5
    qp = copy(q_wall)
    qm = copy(q_wall)
    @inbounds for idx in eachindex(q_wall, dq_dR)
        qp[idx] += eps_R * dq_dR[idx]
        qm[idx] -= eps_R * dq_dR[idx]
    end
    vp = ad_lambda_dot_G_value(f_star, lambda, qp, is_solid, u_profile,
                               rho_out, s_plus, s_minus, Nx, Ny)
    vm = ad_lambda_dot_G_value(f_star, lambda, qm, is_solid, u_profile,
                               rho_out, s_plus, s_minus, Nx, Ny)
    return (vp - vm) / (2.0 * eps_R)
end

function cut_cotangent_norm(dqw, q_wall)
    s = 0.0
    @inbounds for idx in eachindex(dqw, q_wall)
        if q_wall[idx] > 0.0
            s += dqw[idx] * dqw[idx]
        end
    end
    return sqrt(s)
end

