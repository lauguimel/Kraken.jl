function _ehd_ec_lattice_params(Ny, C, M, T_ehd, Ma_E, alpha, delta_U, gamma; FT)
    spec = Units.EHDSpec{FT}(FT(T_ehd), FT(C), FT(M), FT(alpha), FT(Ma_E))
    return Units.ehd_ec_lattice_params(spec, Ny, delta_U, gamma; FT=FT)
end

function _fill_charge_populations_ec!(f_cpu, q_init, Ey_profile, K, Nx, Ny, FT)
    for j in 1:Ny, i in 1:Nx, qdir in 1:9
        f_cpu[i, j, qdir] = _charge_feq_host(q_init[i, j], zero(FT), K * Ey_profile[j], qdir, FT)
    end
    return f_cpu
end

function _project_coulomb_force_rows!(Fx, Fy, is_solid, mode)
    mode === :none && return nothing
    mode in (:xy, :y) || throw(ArgumentError("force_projection must be :none, :xy, or :y."))
    mode_code = mode === :xy ? 1 : 2
    Nx, Ny = size(Fx)
    project_coulomb_force_rows_2d!(Fx, Fy, is_solid, mode_code, Nx, Ny)
    return nothing
end

"""
    run_electroconvection_2d(; Nx, Ny, C, M, T, Ma_E, alpha, max_cycles, ...)

Run a CPU-oriented coupled EHD electroconvection canary. The electric potential
uses the pseudo-time DDF Poisson solve, charge uses drift equilibrium
`u + K*E`, and Navier-Stokes uses BGK + Guo forcing with force density `q*E`.
Sidewalls are EHD-local zero-gradient scalar NEE and post-stream free-slip flow
mirroring ported from Jiachen's MATLAB driver.
"""
function run_electroconvection_2d(; Nx=60, Ny=96, C=10.0, M=10.0, T=175.0,
                                    Ma_E=1e-2, alpha=1e-4, delta_U=1.0,
                                    gamma=0.3, max_cycles=2000,
                                    target_t_star=nothing,
                                    phi_tol=1e-4, phi_max_iter=10000,
                                    phi_substeps=nothing,
                                    phi_scheme=:lbm,
                                    charge_scheme=:regularized,
                                    ns_scheme=:bgk,
                                    perturb_amplitude=1e-4,
                                    perturb_mode=1,
                                    force_projection=:none,
                                    velocity_stop=0.2,
                                    history_interval=1,
                                    backend=KernelAbstractions.CPU(),
                                    FT=Float64)
    # Run control lives here; everything else is the state contract
    # (src/drivers/ehd_ec_state.jl): init_state -> advance! -> solution.
    s = init_state(ECState; Nx=Nx, Ny=Ny, C=C, M=M, T=T, Ma_E=Ma_E, alpha=alpha,
                   delta_U=delta_U, gamma=gamma, phi_tol=phi_tol,
                   phi_max_iter=phi_max_iter, phi_substeps=phi_substeps,
                   phi_scheme=phi_scheme, charge_scheme=charge_scheme,
                   ns_scheme=ns_scheme, perturb_amplitude=perturb_amplitude,
                   perturb_mode=perturb_mode, force_projection=force_projection,
                   velocity_stop=velocity_stop, history_interval=history_interval,
                   backend=backend, FT=FT)
    p = s.p
    if target_t_star !== nothing
        target_cycles = Int(ceil(FT(target_t_star) / p.dt_star))
        max_cycles = min(Int(max_cycles), target_cycles)
    else
        max_cycles = Int(max_cycles)
    end
    advance!(s, max_cycles; sample_final=true)
    return solution(s).result
end
