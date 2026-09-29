# ============================================================================
# src/drivers/ehd_ec_state.jl
# Electroconvection (EHD, 2D) as a client of the platform state contract
# (`src/platform/state.jl`): `init_state` + `advance!` + `solution`.
#
# `run_electroconvection_2d` (src/drivers/ehd_ec.jl) is the one-shot wrapper over
# these three verbs. The cycle body below is the body of the former monolithic
# loop, in the same order of operations; `test/analytical/ehd_ec_split_parity_2d.jl`
# compares it bit for bit against a frozen copy of that function.
#
# Export / restore to disk and parameter updates are not defined here yet.
# ============================================================================

"""
    ECState <: AbstractSimulationState

Everything the 2D electroconvection solver carries from one cycle to the next.

- `config`: the configuration keywords exactly as the caller passed them
  (`history_interval` converted to `Int`); `p`: the derived lattice parameters.
- Dynamic state: the three population pairs (`phi_f_*`, `q_f_*`, `f_*`), the force
  history `Fx_prev` / `Fy_prev`, `qfield`, and `phi` (under `phi_scheme = :direct`
  it lags the charge by one cycle and cannot be rebuilt from populations).
- Derived buffers, overwritten before being read in every cycle: `Ex`, `Ey`, `rho`,
  `ux`, `uy`, `Fx`, `Fy`, `phi_prev`, `q_prev`, `diag`, `diag_host`.
- Carried scalars: `phi_iters_last`, `phi_rel_last`, `q_rel_last`.
- `cycle`: global cycle counter (cycles completed since `init_state`).
- `umax_history` / `cycle_history`: histories sampled on the global counter.
- `loop_ns`: wall time accumulated inside `advance!`.
- `at_boundary`: `false` while a cycle is in progress (see [`at_boundary`](@ref)).

The struct is fully parametric so that `advance!` specializes on the array types:
the buffer swaps are swaps of two fields of the same concrete type.
"""
mutable struct ECState{FT,A3,A2,A1,AB,P,PS,B,CFG} <: AbstractSimulationState
    config::CFG
    p::P
    backend::B
    A::FT
    phi_f_in::A3
    phi_f_out::A3
    q_f_in::A3
    q_f_out::A3
    f_in::A3
    f_out::A3
    phi::A2
    qfield::A2
    Ex::A2
    Ey::A2
    rho::A2
    ux::A2
    uy::A2
    Fx::A2
    Fy::A2
    Fx_prev::A2
    Fy_prev::A2
    phi_prev::A2
    q_prev::A2
    diag::A1
    diag_host::Vector{FT}
    is_solid::AB
    poisson_setup::PS
    phi_iters_last::Int
    phi_rel_last::FT
    q_rel_last::FT
    cycle::Int
    umax_history::Vector{FT}
    cycle_history::Vector{Int}
    loop_ns::UInt64
    at_boundary::Bool
end

"""
    ECSolution{R} <: AbstractSolution

Nominal wrapper around the `NamedTuple` that `run_electroconvection_2d` returns
(same pattern as [`LBMSolution`](@ref)): `solution(state).result`.
"""
struct ECSolution{R} <: AbstractSolution
    result::R
end

at_boundary(s::ECState) = s.at_boundary

"""
    init_state(ECState; Nx, Ny, C, M, T, Ma_E, alpha, ...) -> ECState

Validate the configuration, allocate on `backend` and set the initial conditions of
the electroconvection solver, at cycle 0. Same keywords and defaults as
`run_electroconvection_2d`, minus the run control (`max_cycles`, `target_t_star`).

`sidewall_bc` selects `:free_slip` (default) or stationary `:no_slip` flow walls.
It is fixed simulation identity, not run control or a continuation parameter.
The future EC checkpoint client must store/compare this configuration key on
restore; EC snapshot/restore is not implemented by this sidewall capability.
"""
function init_state(::Type{ECState}; Nx=60, Ny=96, C=10.0, M=10.0, T=175.0,
                    Ma_E=1e-2, alpha=1e-4, delta_U=1.0,
                    gamma=0.3,
                    phi_tol=1e-4, phi_max_iter=10000,
                    phi_substeps=nothing,
                    phi_scheme=:lbm,
                    charge_scheme=:regularized,
                    ns_scheme=:bgk,
                    sidewall_bc=:free_slip,
                    perturb_amplitude=1e-4,
                    perturb_mode=1,
                    force_projection=:none,
                    velocity_stop=0.2,
                    history_interval=1,
                    backend=KernelAbstractions.CPU(),
                    FT=Float64)
    Nx < 4 && throw(ArgumentError("Nx must be at least 4."))
    Ny < 8 && throw(ArgumentError("Ny must be at least 8."))
    charge_scheme in (:srt, :regularized) ||
        throw(ArgumentError("charge_scheme must be :srt or :regularized."))
    ns_scheme in (:bgk, :mrt) ||
        throw(ArgumentError("ns_scheme must be :bgk or :mrt."))
    sidewall_bc in (:free_slip, :no_slip) ||
        throw(ArgumentError("sidewall_bc must be :free_slip or :no_slip."))
    force_projection in (:none, :xy, :y) ||
        throw(ArgumentError("force_projection must be :none, :xy, or :y."))
    phi_scheme in (:lbm, :direct) ||
        throw(ArgumentError("phi_scheme must be :lbm or :direct."))
    history_interval = Int(history_interval)
    history_interval > 0 ||
        throw(ArgumentError("history_interval must be positive."))

    p = _ehd_ec_lattice_params(Ny, C, M, T, Ma_E, alpha, delta_U, gamma; FT=FT)
    p.tau <= FT(0.5) && error("NS relaxation time must be greater than 0.5.")
    p.tau_q <= FT(0.5) && error("Charge relaxation time must be greater than 0.5.")

    analytic = ehd_hydrostatic_profiles(C, Ny; FT=FT)
    q_profile = FT(p.q_inj) .* analytic.q_star
    Ey_profile = FT(delta_U) .* analytic.E_star ./ FT(p.H)
    A = FT(Nx - 1) / FT(Ny - 1)

    phi_f_in = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    phi_f_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    q_f_in = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    q_f_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    f_in = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    f_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)

    phi = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    qfield = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Ex = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Ey = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    rho = KernelAbstractions.ones(backend, FT, Nx, Ny)
    ux = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    uy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fx_prev = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fy_prev = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    phi_prev = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    q_prev = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    diag = KernelAbstractions.zeros(backend, FT, 2)
    diag_host = Vector{FT}(undef, 2)
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny)

    q_init = zeros(FT, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        xstar = A * FT(i - 1) / FT(Nx - 1)
        ystar = FT(j - 1) / FT(Ny - 1)
        q_init[i, j] = q_profile[j] +
                       FT(perturb_amplitude) * FT(p.q_inj) *
                       sin(FT(pi) * ystar) *
                       cos(FT(pi) * FT(perturb_mode) * xstar / A)
    end
    q_init[:, 1] .= FT(p.q_inj)

    phi_init = _fill_phi_populations!(zeros(FT, Nx, Ny, 9), analytic.phi, Nx, Ny, FT)
    q_pop_init = _fill_charge_populations_ec!(zeros(FT, Nx, Ny, 9), q_init, Ey_profile, FT(p.K), Nx, Ny, FT)
    ns_init = zeros(FT, Nx, Ny, 9)
    for qdir in 1:9
        ns_init[:, :, qdir] .= ehd_w(Val(qdir), FT)
    end
    # Matrix{Bool}, not falses(): copyto!(CuArray{Bool}, BitMatrix) falls back to scalar indexing
    solid_cpu = zeros(Bool, Nx, Ny)
    solid_cpu[:, 1] .= true
    solid_cpu[:, Ny] .= true

    copyto!(phi_f_in, phi_init)
    copyto!(phi_f_out, phi_init)
    copyto!(q_f_in, q_pop_init)
    copyto!(q_f_out, q_pop_init)
    copyto!(f_in, ns_init)
    copyto!(f_out, ns_init)
    copyto!(is_solid, solid_cpu)

    compute_ehd_scalar_2d!(phi, phi_f_in)
    compute_ehd_scalar_2d!(qfield, q_f_in)
    poisson_setup = phi_scheme === :direct ? ehd_poisson_setup(Nx, Ny, p.eps; xbc=:neumann, backend=backend) : nothing
    if phi_scheme === :direct
        ehd_poisson_solve!(phi, poisson_setup, qfield)
        compute_electric_field_fd_2d!(Ex, Ey, phi, :neumann, Nx, Ny)
    else
        compute_electric_field_2d!(Ex, Ey, phi_f_in, p.tau_U)
    end

    config = (; Nx, Ny, C, M, T, Ma_E, alpha, delta_U, gamma, phi_tol, phi_max_iter,
              phi_substeps, phi_scheme, charge_scheme, ns_scheme, sidewall_bc, perturb_amplitude,
              perturb_mode, force_projection, velocity_stop, history_interval)
    return ECState(config, p, backend, A,
                   phi_f_in, phi_f_out, q_f_in, q_f_out, f_in, f_out,
                   phi, qfield, Ex, Ey, rho, ux, uy, Fx, Fy, Fx_prev, Fy_prev,
                   phi_prev, q_prev, diag, diag_host, is_solid, poisson_setup,
                   0, FT(Inf), FT(Inf), 0, FT[], Int[], UInt64(0), true)
end

"""
    advance!(s::ECState, n; sample_final=false) -> s

Run `n` coupled cycles (potential, charge, Navier-Stokes) in place. A cycle is
sampled (history entry, relative changes, finiteness and stability checks) when its
global number is a multiple of `history_interval`, or when it is the last cycle of
the call and `sample_final` is set. Sampling never feeds back into the dynamics, so
any split of `n` leaves the same populations and force history.

An exception raised in the middle of a cycle leaves `at_boundary(s) == false`; every
later `advance!` (even with `n == 0`) or `solution` on that state throws an
`ArgumentError` without touching it ([`require_boundary`](@ref)).
"""
function advance!(s::ECState{FT}, n::Integer; sample_final::Bool=false) where {FT}
    require_boundary(s, "advance!")
    n < 0 && throw(ArgumentError("advance!: the number of cycles must be non-negative, got $n."))
    # Bindings that never change during a run. The population pairs and the carried
    # scalars are read and written through `s` because they do change.
    (; Nx, Ny, phi_tol, phi_max_iter, phi_substeps, phi_scheme, charge_scheme,
       ns_scheme, sidewall_bc, force_projection, velocity_stop, history_interval) = s.config
    p = s.p
    poisson_setup = s.poisson_setup
    (; phi, qfield, Ex, Ey, rho, ux, uy, Fx, Fy, Fx_prev, Fy_prev, phi_prev, q_prev,
       diag, diag_host, is_solid) = s
    phi_check_every = 8

    t0 = time_ns()
    for k in 1:n
        s.at_boundary = false
        cycle = s.cycle + 1
        sample_cycle = (cycle % history_interval == 0) || (sample_final && k == n)

        if phi_scheme === :direct
            ehd_poisson_solve!(phi, poisson_setup, qfield)
            compute_electric_field_fd_2d!(Ex, Ey, phi, :neumann, Nx, Ny)
            s.phi_iters_last = 1
            s.phi_rel_last = zero(FT)
        elseif phi_substeps === nothing
            for iter in 1:phi_max_iter
                copyto!(phi_prev, phi)
                collide_electric_potential_2d!(s.phi_f_in, qfield, p.eps, p.omega_U, p.nu_U)
                stream_wall_x_wall_y_2d!(s.phi_f_out, s.phi_f_in, Nx, Ny)
                compute_ehd_scalar_2d!(phi, s.phi_f_out)
                apply_phi_nee_box_2d!(s.phi_f_out, phi, one(FT), zero(FT), Nx, Ny)
                compute_ehd_scalar_2d!(phi, s.phi_f_out)
                s.phi_f_in, s.phi_f_out = s.phi_f_out, s.phi_f_in
                s.phi_iters_last = iter
                if iter % phi_check_every == 0 || iter == phi_max_iter
                    ehd_rel_change_2d!(diag, phi, phi_prev, Nx, Ny)
                    copyto!(diag_host, diag)
                    s.phi_rel_last = diag_host[1]
                    s.phi_rel_last <= phi_tol && break
                    iter == phi_max_iter &&
                        error("Electric potential solve did not converge within $(phi_max_iter) iterations. Last relative change: $(s.phi_rel_last).")
                end
            end
            compute_electric_field_2d!(Ex, Ey, s.phi_f_in, p.tau_U)
        else
            sample_cycle && copyto!(phi_prev, phi)
            for _ in 1:Int(phi_substeps)
                collide_electric_potential_2d!(s.phi_f_in, qfield, p.eps, p.omega_U, p.nu_U)
                stream_wall_x_wall_y_2d!(s.phi_f_out, s.phi_f_in, Nx, Ny)
                compute_ehd_scalar_2d!(phi, s.phi_f_out)
                apply_phi_nee_box_2d!(s.phi_f_out, phi, one(FT), zero(FT), Nx, Ny)
                compute_ehd_scalar_2d!(phi, s.phi_f_out)
                s.phi_f_in, s.phi_f_out = s.phi_f_out, s.phi_f_in
            end
            if sample_cycle
                ehd_rel_change_2d!(diag, phi, phi_prev, Nx, Ny)
                copyto!(diag_host, diag)
                s.phi_rel_last = diag_host[1]
            end
            s.phi_iters_last = Int(phi_substeps)
            compute_electric_field_2d!(Ex, Ey, s.phi_f_in, p.tau_U)
        end

        compute_macroscopic_guo_field_2d!(rho, ux, uy, s.f_in, Fx_prev, Fy_prev, Nx, Ny)
        sidewall_bc === :free_slip && enforce_free_side_macros_2d!(ux, uy, Nx, Ny)

        sample_cycle && copyto!(q_prev, qfield)
        if charge_scheme == :srt
            collide_electric_charge_srt_2d!(s.q_f_in, ux, uy, Ex, Ey, p.tau_q, p.K)
        else
            collide_electric_charge_regularized_2d!(s.q_f_in, ux, uy, Ex, Ey, p.tau_q, p.K)
        end
        stream_wall_x_wall_y_2d!(s.q_f_out, s.q_f_in, Nx, Ny)
        compute_ehd_scalar_2d!(qfield, s.q_f_out)
        apply_charge_nee_box_2d!(s.q_f_out, qfield, ux, uy, Ex, Ey, p.q_inj, zero(FT), p.K, Nx, Ny)
        compute_ehd_scalar_2d!(qfield, s.q_f_out)
        if sample_cycle
            ehd_rel_change_2d!(diag, qfield, q_prev, Nx, Ny)
            copyto!(diag_host, diag)
            s.q_rel_last = diag_host[1]
            diag_host[2] == zero(FT) && error("Charge field became non-finite at cycle $(cycle).")
        end
        s.q_f_in, s.q_f_out = s.q_f_out, s.q_f_in

        compute_coulomb_force_2d!(Fx, Fy, qfield, Ex, Ey, Nx, Ny)
        _project_coulomb_force_rows!(Fx, Fy, is_solid, force_projection)
        if ns_scheme == :bgk
            collide_guo_field_2d!(s.f_in, is_solid, Fx, Fy, p.omega)
        else
            ehd_collide_mrt_2d!(s.f_in, Fx, Fy, is_solid, p.nu)
        end
        stream_wall_x_wall_y_2d!(s.f_out, s.f_in, Nx, Ny)
        if sidewall_bc === :free_slip
            apply_free_slip_sidewalls_2d!(s.f_out, Nx, Ny)
        else
            apply_no_slip_sidewalls_2d!(s.f_out, Fx, Fy, Nx, Ny)
        end
        s.f_in, s.f_out = s.f_out, s.f_in

        if sample_cycle
            compute_macroscopic_guo_field_2d!(rho, ux, uy, s.f_in, Fx, Fy, Nx, Ny)
            sidewall_bc === :free_slip && enforce_free_side_macros_2d!(ux, uy, Nx, Ny)
            ehd_maxspeed_2d!(diag, ux, uy, Nx, Ny)
            copyto!(diag_host, diag)
            umax = diag_host[1]
            diag_host[2] == zero(FT) && error("Flow velocity became non-finite at cycle $(cycle).")
            umax > FT(velocity_stop) &&
                error("Flow field became unstable at cycle $(cycle): max(|u|) = $(umax).")
            push!(s.umax_history, umax)
            push!(s.cycle_history, cycle)
        end
        copyto!(Fx_prev, Fx)
        copyto!(Fy_prev, Fy)
        s.cycle = cycle
        s.at_boundary = true
    end
    s.loop_ns += time_ns() - t0
    return s
end

"""
    solution(s::ECState) -> ECSolution

Host-side result at the current cycle: the `NamedTuple` `run_electroconvection_2d`
returns, wrapped in an [`ECSolution`](@ref).

For legacy output parity, `result.sidewall_bc` reports `:free_slip_ported` for
the input `:free_slip` (and `:no_slip` unchanged); checkpoint identity must use
the canonical `config.sidewall_bc`, not this legacy result label.

The derived buffers `qfield`, `phi` (`:lbm` only), `Ex`, `Ey`, `rho`, `ux`, `uy` are
recomputed in place from the populations, as the one-shot driver did after its
loop. `qfield`, `phi`, `Ex`, `Ey` get the values they already hold; `rho`, `ux`,
`uy` are recomputed by the next cycle before anything reads them. A later
`advance!` is therefore unaffected. Refused (`ArgumentError`) on a state left in the
middle of a cycle.
"""
function solution(s::ECState)
    require_boundary(s, "solution")
    c = s.config
    p = s.p
    (; phi, qfield, Ex, Ey, rho, ux, uy, Fx, Fy) = s

    c.phi_scheme === :lbm && compute_ehd_scalar_2d!(phi, s.phi_f_in)
    compute_ehd_scalar_2d!(qfield, s.q_f_in)
    if c.phi_scheme === :direct
        compute_electric_field_fd_2d!(Ex, Ey, phi, :neumann, c.Nx, c.Ny)
    else
        compute_electric_field_2d!(Ex, Ey, s.phi_f_in, p.tau_U)
    end
    compute_macroscopic_guo_field_2d!(rho, ux, uy, s.f_in, Fx, Fy, c.Nx, c.Ny)
    c.sidewall_bc === :free_slip && enforce_free_side_macros_2d!(ux, uy, c.Nx, c.Ny)

    steps_done = s.cycle
    result = (ux=Array(ux), uy=Array(uy), rho=Array(rho), q=Array(qfield),
              phi=Array(phi), Ex=Array(Ex), Ey=Array(Ey), Fx=Array(Fx), Fy=Array(Fy),
              umax_history=copy(s.umax_history),
              cycle_history=copy(s.cycle_history),
              steps=steps_done, Nx=c.Nx, Ny=c.Ny, A=s.A, C=c.C, M=c.M, T=c.T,
              Ma_E=c.Ma_E, alpha=c.alpha, perturb_amplitude=c.perturb_amplitude,
              perturb_mode=c.perturb_mode, force_projection=c.force_projection,
              charge_scheme=c.charge_scheme, ns_scheme=c.ns_scheme,
              phi_scheme=c.phi_scheme, phi_substeps=c.phi_substeps,
              phi_iters_last=s.phi_iters_last, phi_rel_last=s.phi_rel_last,
              q_rel_change=s.q_rel_last, params=p,
              ns_collision=(c.ns_scheme == :bgk ? :bgk_guo : :mrt_guo_moment),
              sidewall_bc=(c.sidewall_bc === :free_slip ? :free_slip_ported : :no_slip),
              loop_ms_per_step=steps_done > 0 ? s.loop_ns / 1e6 / steps_done : 0.0)
    return ECSolution(result)
end
