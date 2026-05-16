using KernelAbstractions

"""
    run_viscoelastic_logfv_cavity_coupled_2d(; kwargs...)

Run a closed lid-driven cavity with the coupled LBM-solvent + log-FV
polymer pipeline. The square domain has all four sides as walls
(`logfv_wallxwally_bcspec_2d`); the top wall is a moving lid imposed
through Zou-He velocity with a smooth ramp matching rheoTool's
Cavity/Oldroyd-BLog tutorial:

```text
U_lid(x, t) = 8 * u_max * (1 + tanh(ramp_steepness * (t - ramp_start)))
              * x^2 * (1 - x)^2
```

`x` is the physical coordinate in `[0, 1]` (cell-centered) and `t` the
physical time `(step * dt_phys)`.

The default parameters are calibrated for the rheoTool tutorial
reference (Re=0.01, De=1, beta=0.5, mesh 127x127, endTime=8). Set `N`
to match rheoTool resolution and tune `u_max` to balance Mach number
against wall-clock cost.

Returns a NamedTuple with all final fields on CPU plus saved-time
snapshots of `ux`, `uy`, `psi*`, `tau*` at the requested `sample_times`
(physical time units).
"""
function run_viscoelastic_logfv_cavity_coupled_2d(;
    N::Integer=64,
    nu_s::Real=0.1,
    nu_p::Real=0.1,
    lambda_phys::Real=1.0,
    bsd_fraction::Real=1.0,
    u_max::Real=0.01,
    L_max::Real=10.0,
    polymer_model=:oldroydb,
    polymer_substeps=:auto,
    max_polymer_substeps::Integer=64,
    max_memory_deformation_increment::Real=0.07,
    subcycle_relative_tolerance::Real=0.01,
    end_time::Real=8.0,
    sample_times::AbstractVector{<:Real}=Float64[1.0, 2.0, 4.0, 6.0, 8.0],
    ramp_start::Real=0.5,
    ramp_steepness::Real=8.0,
    skip_top_corners::Bool=false,
    bsd_kind::Symbol = :fd,
    diagnose_bsd_dual::Bool = false,
    polymer_wall_extrap::Symbol = :quadratic,
    diagnostic_stride::Integer=0,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)
    N >= 8 || throw(ArgumentError("N must be >= 8"))
    nu_s > 0 || throw(ArgumentError("nu_s must be positive"))
    nu_p >= 0 || throw(ArgumentError("nu_p must be non-negative"))
    0 <= bsd_fraction <= 1 || throw(ArgumentError("bsd_fraction must be in [0, 1]"))
    u_max > 0 || throw(ArgumentError("u_max must be positive"))
    lambda_phys > 0 || throw(ArgumentError("lambda_phys must be positive"))
    end_time > 0 || throw(ArgumentError("end_time must be positive"))
    diagnostic_stride >= 0 || throw(ArgumentError("diagnostic_stride must be non-negative"))
    bsd_kind in (:fd, :kinetic) || throw(ArgumentError("bsd_kind must be :fd or :kinetic"))
    polymer_wall_extrap in (:quadratic, :linear) ||
        throw(ArgumentError("polymer_wall_extrap must be :quadratic or :linear"))

    Nx = Int(N)
    Ny = Int(N)
    dt_phys = u_max / Nx
    max_steps = max(1, ceil(Int, end_time / dt_phys))
    lambda_lu = T(lambda_phys) / T(dt_phys)
    nu_s_t = T(nu_s)
    nu_p_t = T(nu_p)
    nu_total_t = nu_s_t + nu_p_t
    bsd_t = T(bsd_fraction)
    nu_lbm_t = nu_s_t + bsd_t * nu_p_t
    nu_lbm_t > zero(T) || throw(ArgumentError("nu_s + bsd_fraction * nu_p must be positive"))
    model_cfg = _logfv_polymer_model_config(polymer_model, L_max, T)
    model_code = model_cfg.model_code
    L2_t = model_cfg.L2
    prefactor_t = nu_p_t / lambda_lu
    dx = one(T)
    dy = one(T)

    # Geometry: pure square cavity, no interior solid cells.
    # Use a concrete Matrix{Bool} (not BitMatrix) so MtlArray copyto!
    # works without scalar indexing.
    is_solid_h = zeros(Bool, Nx, Ny)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    copyto!(is_solid, is_solid_h)
    # q_wall placeholder (no curved geometry); shaped for fused kernel compatibility
    q_wall = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)

    # Lid profile (Nx values on the north face, device-resident)
    u_lid_profile = KernelAbstractions.zeros(backend, T, Nx)
    _logfv_cavity_update_lid_profile!(
        u_lid_profile, 0.0, u_max, ramp_start, ramp_steepness,
    )

    # LBM populations at rest (rho=1, u=0)
    f_in = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    f_out = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    f_in_h = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_in_h[i, j, q] = equilibrium(D2Q9(), one(T), zero(T), zero(T), q)
    end
    copyto!(f_in, f_in_h)
    fill!(f_out, zero(T))

    rho = KernelAbstractions.allocate(backend, T, Nx, Ny); fill!(rho, one(T))
    ux = KernelAbstractions.zeros(backend, T, Nx, Ny)
    uy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    uwx = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    uwy = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)

    # Log-conformation initial state: C = I -> Psi = log(I) = 0
    psixx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixx_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixx_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)

    tauxx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dudx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dudy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdy = KernelAbstractions.zeros(backend, T, Nx, Ny)

    fx_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_total = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_total = KernelAbstractions.zeros(backend, T, Nx, Ny)
    if diagnose_bsd_dual
        fx_alt = KernelAbstractions.zeros(backend, T, Nx, Ny)
        fy_alt = KernelAbstractions.zeros(backend, T, Nx, Ny)
    end

    ux_face = KernelAbstractions.zeros(backend, T, Nx + 1, Ny)
    uy_face = KernelAbstractions.zeros(backend, T, Nx, Ny + 1)

    # Dummy face BC arrays — ignored by the wall dispatch but required by the
    # BC-aware kernels' signatures.
    dummy_y = KernelAbstractions.zeros(backend, T, Ny)
    dummy_x = KernelAbstractions.zeros(backend, T, Nx)

    logfv_bc = logfv_wallxwally_bcspec_2d()

    # LBM BC spec: HW-BB everywhere except north which is Zou-He with the
    # moving lid profile. The profile object is updated in place each step.
    bcspec = BCSpec2D(north=ZouHeVelocity(u_lid_profile))

    # Polymer substep choice. In a lid-driven cavity the dominant shear is
    # at the top wall, ~ U_lid_peak / (dy_LU / 2) = 2 * u_max in lattice
    # units (lid peak is u_max LU, evaluated over a half-cell to the wall).
    # The channel-style bulk-shear estimate `4*u_max/N` underestimates this
    # by ~N/8 and produces too few substeps for cavity. Use the lid shear.
    lid_shear_estimate = T(2) * T(u_max)
    subcycle_estimate = logfv_oldroydb_subcycle_estimate(
        Float64(lid_shear_estimate), Float64(lambda_lu), 1.0;
        max_memory_deformation_increment=Float64(max_memory_deformation_increment),
        min_substeps=1,
        max_substeps=max_polymer_substeps,
    )
    selected_polymer_substeps = if polymer_substeps === :auto
        subcycle_estimate.recommended
    elseif polymer_substeps isa Integer
        polymer_substeps >= 1 || throw(ArgumentError("polymer_substeps must be >= 1"))
        polymer_substeps
    else
        throw(ArgumentError("polymer_substeps must be an Integer or :auto"))
    end
    dt_poly = one(T) / T(selected_polymer_substeps)

    # Sample-time scheduling
    sample_times_sorted = sort(unique(Float64.(sample_times)))
    sample_step_targets = Int[max(1, ceil(Int, ts / dt_phys)) for ts in sample_times_sorted]
    snapshots = Dict{Float64,NamedTuple}()
    kinetic_energy_history = Tuple{Float64,Float64,Float64}[]
    bsd_dual_path_diagnostic = Tuple{Int,Float64,Float64}[]

    first_nonfinite_step = 0
    first_nonfinite_field = :none
    first_nonfinite_i = 0
    first_nonfinite_j = 0
    completed_steps = 0

    logfv_compute_macroscopic_forced_field_2d!(rho, ux, uy, f_in, fx_total, fy_total; sync=false)

    for step in 1:max_steps
        completed_steps = step
        t_phys = step * dt_phys
        _logfv_cavity_update_lid_profile!(
            u_lid_profile, t_phys, u_max, ramp_start, ramp_steepness; sync=false,
        )

        # 1. Cell -> face velocity (wall BC -> face velocity = 0 at all 4 sides)
        logfv_cell_velocity_to_faces_bc_aware_2d!(
            ux_face, uy_face, ux, uy, is_solid,
            dummy_y, dummy_y, dummy_x, dummy_x, logfv_bc; sync=false,
        )

        # 2. Polymer (log-conformation) advection
        logfv_advect_upwind_bc_aware_2d!(
            psixx_adv, psixy_adv, psiyy_adv,
            psixx, psixy, psiyy,
            dummy_y, dummy_y, dummy_y, dummy_y, dummy_y, dummy_y,
            dummy_x, dummy_x, dummy_x, dummy_x, dummy_x, dummy_x,
            ux_face, uy_face, is_solid, dx, dy, logfv_bc, one(T); sync=false,
        )

        # 3. Velocity gradient (standard solid-aware) + moving-wall correction
        fvfd_velocity_gradient_2d!(
            dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, logfv_bc; sync=false,
        )
        _logfv_cavity_apply_wall_gradient_correction!(
            dudx, dudy, dvdx, dvdy, ux, uy, u_lid_profile, dx, dy;
            skip_top_corners=skip_top_corners, sync=false,
        )

        # 4. Polymer source (substepped)
        psixx_work, psixy_work, psiyy_work = psixx_adv, psixy_adv, psiyy_adv
        for _ in 1:selected_polymer_substeps
            logfv_step_constitutive_log_2d!(
                psixx_next, psixy_next, psiyy_next,
                psixx_work, psixy_work, psiyy_work,
                dudx, dudy, dvdx, dvdy,
                lambda_lu, dt_poly, model_code, L2_t; sync=false,
            )
            psixx_work, psixx_next = psixx_next, psixx_work
            psixy_work, psixy_next = psixy_next, psixy_work
            psiyy_work, psiyy_next = psiyy_next, psiyy_work
        end
        psixx, psixx_adv = psixx_work, psixx
        psixy, psixy_adv = psixy_work, psixy
        psiyy, psiyy_adv = psiyy_work, psiyy

        # 5. Stress from log-conformation
        logfv_stress_from_log_2d!(
            tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor_t;
            model_code, L2=L2_t, sync=false,
        )

        # 6. Polymer body force = div(tau)
        logfv_polymer_force_bc_aware_2d!(
            fx_poly, fy_poly, tauxx, tauxy, tauyy, is_solid, dx, dy, logfv_bc; sync=false,
            polymer_wall_extrap=polymer_wall_extrap,
        )

        # 7. BSD correction
        if bsd_kind === :kinetic
            s_plus_t = T(trt_rates(nu_lbm_t)[1])
            compute_bsd_force_kinetic_2d!(
                fx_total, fy_total, fx_poly, fy_poly,
                f_in, rho, ux, uy, is_solid,
                bsd_t, nu_p_t, s_plus_t, dx, dy; sync=false,
            )
        else
            logfv_bsd_correct_force_bc_aware_2d!(
                fx_total, fy_total, fx_poly, fy_poly, ux, uy, is_solid, bsd_t, nu_p_t, dx, dy,
                logfv_bc; sync=false,
            )
        end

        if diagnose_bsd_dual
            if bsd_kind === :fd
                s_plus_t = T(trt_rates(nu_lbm_t)[1])
                compute_bsd_force_kinetic_2d!(
                    fx_alt, fy_alt, fx_poly, fy_poly,
                    f_in, rho, ux, uy, is_solid,
                    bsd_t, nu_p_t, s_plus_t, dx, dy; sync=false,
                )
            else
                logfv_bsd_correct_force_bc_aware_2d!(
                    fx_alt, fy_alt, fx_poly, fy_poly, ux, uy, is_solid, bsd_t, nu_p_t, dx, dy,
                    logfv_bc; sync=false,
                )
            end
            rel_l2 = _logfv_bsd_dual_path_relative_l2_2d(
                fx_total, fy_total, fx_alt, fy_alt, is_solid_h, backend,
            )
            push!(bsd_dual_path_diagnostic, (step, Float64(t_phys), rel_l2))
        end

        # 8. LBM solvent step with Guo body force
        fused_trt_libb_v2_guo_field_step!(
            f_out, f_in, rho, ux, uy, is_solid, q_wall, uwx, uwy, fx_total, fy_total,
            Nx, Ny, nu_lbm_t,
        )

        # 9. BC rebuild: Zou-He moving lid at north + HW-BB elsewhere
        apply_bc_rebuild_2d!(f_out, f_in, bcspec, nu_lbm_t, Nx, Ny)

        # 10. Update macroscopic fields
        logfv_compute_macroscopic_forced_field_2d!(rho, ux, uy, f_out, fx_total, fy_total; sync=false)

        # 11. Diagnostics + snapshots
        if diagnostic_stride > 0 &&
           (step == 1 || step % diagnostic_stride == 0 || step == max_steps)
            KernelAbstractions.synchronize(backend)
            finite_diag = _logfv_first_nonfinite_field_2d(
                is_solid_h,
                :rho => rho,
                :ux => ux,
                :uy => uy,
                :psixx => psixx,
                :psixy => psixy,
                :psiyy => psiyy,
                :tauxx => tauxx,
                :tauxy => tauxy,
                :tauyy => tauyy,
                :fx_poly => fx_poly,
                :fy_poly => fy_poly,
            )
            if !finite_diag.finite
                first_nonfinite_step = step
                first_nonfinite_field = finite_diag.field
                first_nonfinite_i = finite_diag.i
                first_nonfinite_j = finite_diag.j
                break
            end
        end

        # Kinetic + elastic "energy" (matches rheoTool kineticEnergy functionObject)
        if step == max_steps || (step % max(1, max_steps ÷ 200) == 0)
            KernelAbstractions.synchronize(backend)
            ux_cpu_tmp = Array(ux)
            uy_cpu_tmp = Array(uy)
            tauxx_cpu_tmp = Array(tauxx)
            tauyy_cpu_tmp = Array(tauyy)
            n_cells = Nx * Ny
            kin = 0.5 / n_cells * sum(@. ux_cpu_tmp^2 + uy_cpu_tmp^2)
            tr_tau = sum(@. tauxx_cpu_tmp + tauyy_cpu_tmp)
            elastic = 0.5 / n_cells * (lambda_lu / nu_p_t) * tr_tau
            push!(kinetic_energy_history,
                  (Float64(t_phys), Float64(kin), Float64(elastic)))
        end

        # Snapshot capture at requested physical times
        sample_idx = findfirst(s -> s == step, sample_step_targets)
        if !isnothing(sample_idx)
            KernelAbstractions.synchronize(backend)
            ts = sample_times_sorted[sample_idx]
            snapshots[ts] = (
                step=step,
                t_phys=Float64(t_phys),
                ux=copy(Array(ux)),
                uy=copy(Array(uy)),
                psixx=copy(Array(psixx)),
                psixy=copy(Array(psixy)),
                psiyy=copy(Array(psiyy)),
                tauxx=copy(Array(tauxx)),
                tauxy=copy(Array(tauxy)),
                tauyy=copy(Array(tauyy)),
            )
        end

        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    return (
        N=Nx,
        nu_s=Float64(nu_s_t),
        nu_p=Float64(nu_p_t),
        bsd_fraction=Float64(bsd_t),
        lambda_phys=Float64(lambda_phys),
        lambda_lu=Float64(lambda_lu),
        u_max=Float64(u_max),
        dt_phys=Float64(dt_phys),
        end_time_target=Float64(end_time),
        max_steps=max_steps,
        completed_steps=completed_steps,
        selected_polymer_substeps=selected_polymer_substeps,
        polymer_model=model_cfg.polymer_model,
        L_max=Float64(model_cfg.L_max),
        ramp_start=Float64(ramp_start),
        ramp_steepness=Float64(ramp_steepness),
        first_nonfinite_step=first_nonfinite_step,
        first_nonfinite_field=first_nonfinite_field,
        first_nonfinite_i=first_nonfinite_i,
        first_nonfinite_j=first_nonfinite_j,
        rho=Array(rho),
        ux=Array(ux),
        uy=Array(uy),
        psixx=Array(psixx),
        psixy=Array(psixy),
        psiyy=Array(psiyy),
        tauxx=Array(tauxx),
        tauxy=Array(tauxy),
        tauyy=Array(tauyy),
        snapshots=snapshots,
        kinetic_energy_history=kinetic_energy_history,
        bsd_dual_path_diagnostic=bsd_dual_path_diagnostic,
    )
end
