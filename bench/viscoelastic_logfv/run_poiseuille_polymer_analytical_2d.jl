#!/usr/bin/env julia

using KernelAbstractions
using Printf

using Kraken

function parse_args(args)
    if length(args) > 1
        error("unrecognised arguments: " * join(args, " "))
    end
    if length(args) == 1 && args[1] != "--self-test"
        error("unrecognised argument: $(args[1])")
    end
    return nothing
end

function default_config(::Type{T}) where {T}
    Nx = 32
    Ny = 32
    lambda_lu = T(32.0)
    nu_p = T(0.1)
    n_substeps = 8
    return (;
        Nx,
        Ny,
        width=T(1.0),
        height=T(1.0),
        U_max_lu=T(0.05),
        lambda_lu,
        nu_p,
        prefactor=nu_p / lambda_lu,
        n_substeps,
        dt_poly=one(T) / T(n_substeps),
        nsteps=200,
        bar=T(1.0e-3),
    )
end

function poiseuille_reference_fields(cfg, ::Type{T}) where {T}
    Nx = cfg.Nx
    Ny = cfg.Ny
    dx = cfg.width / T(Nx)
    dy = cfg.height / T(Ny)
    half_height = cfg.height / T(2)

    ux = zeros(T, Nx, Ny)
    uy = zeros(T, Nx, Ny)
    psixx = zeros(T, Nx, Ny)
    psixy = zeros(T, Nx, Ny)
    psiyy = zeros(T, Nx, Ny)
    tauxx = zeros(T, Nx, Ny)
    tauxy = zeros(T, Nx, Ny)
    tauyy = zeros(T, Nx, Ny)

    for j in 1:Ny
        y = (T(j) - T(0.5)) * dy
        gamma = -T(8) * cfg.U_max_lu * (y - half_height) / (cfg.height * cfg.height)
        ux_j = cfg.U_max_lu * (one(T) - (T(2) * (y - half_height) / cfg.height)^2)
        cxx = one(T) + T(2) * (cfg.lambda_lu * gamma)^2
        cxy = cfg.lambda_lu * gamma
        cyy = one(T)
        psi_xx, psi_xy, psi_yy = Kraken.logfv_log_spd_sym2_2d(cxx, cxy, cyy)
        tau_xx = cfg.prefactor * (cxx - one(T))
        tau_xy = cfg.prefactor * cxy
        tau_yy = cfg.prefactor * (cyy - one(T))

        for i in 1:Nx
            ux[i, j] = ux_j
            psixx[i, j] = psi_xx
            psixy[i, j] = psi_xy
            psiyy[i, j] = psi_yy
            tauxx[i, j] = tau_xx
            tauxy[i, j] = tau_xy
            tauyy[i, j] = tau_yy
        end
    end

    return (; dx, dy, ux, uy, psixx, psixy, psiyy, tauxx, tauxy, tauyy)
end

function run_polymer_pipeline(ref, cfg, backend, ::Type{T}) where {T}
    Nx = cfg.Nx
    Ny = cfg.Ny
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    fill!(is_solid, false)

    ux = KernelAbstractions.allocate(backend, T, Nx, Ny)
    uy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    psixx = KernelAbstractions.allocate(backend, T, Nx, Ny)
    psixy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    psiyy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    copyto!(ux, ref.ux)
    copyto!(uy, ref.uy)
    copyto!(psixx, ref.psixx)
    copyto!(psixy, ref.psixy)
    copyto!(psiyy, ref.psiyy)

    psixx_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixx_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    ux_face = KernelAbstractions.zeros(backend, T, Nx + 1, Ny)
    uy_face = KernelAbstractions.zeros(backend, T, Nx, Ny + 1)
    dudx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dudy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dummy_y = KernelAbstractions.zeros(backend, T, Ny)
    dummy_x = KernelAbstractions.zeros(backend, T, Nx)
    logfv_bc = Kraken.logfv_periodicx_wally_bcspec_2d()

    for _ in 1:cfg.nsteps
        Kraken.logfv_cell_velocity_to_faces_bc_aware_2d!(
            ux_face, uy_face, ux, uy, is_solid,
            dummy_y, dummy_y, dummy_x, dummy_x, logfv_bc; sync=false,
        )
        Kraken.logfv_advect_upwind_bc_aware_2d!(
            psixx_adv, psixy_adv, psiyy_adv,
            psixx, psixy, psiyy,
            dummy_y, dummy_y, dummy_y, dummy_y, dummy_y, dummy_y,
            dummy_x, dummy_x, dummy_x, dummy_x, dummy_x, dummy_x,
            ux_face, uy_face, is_solid, ref.dx, ref.dy, logfv_bc, one(T); sync=false,
        )
        Kraken.fvfd_velocity_gradient_2d!(
            dudx, dudy, dvdx, dvdy, ux, uy, is_solid, ref.dx, ref.dy, logfv_bc; sync=false,
        )
        psixx_work, psixy_work, psiyy_work = psixx_adv, psixy_adv, psiyy_adv
        for _ in 1:cfg.n_substeps
            Kraken.logfv_step_constitutive_log_2d!(
                psixx_next, psixy_next, psiyy_next,
                psixx_work, psixy_work, psiyy_work,
                dudx, dudy, dvdx, dvdy,
                cfg.lambda_lu, cfg.dt_poly, Kraken.LOGFV_MODEL_OLDROYDB, T(0.0); sync=false,
            )
            psixx_work, psixx_next = psixx_next, psixx_work
            psixy_work, psixy_next = psixy_next, psixy_work
            psiyy_work, psiyy_next = psiyy_next, psiyy_work
        end
        psixx, psixx_adv = psixx_work, psixx
        psixy, psixy_adv = psixy_work, psixy
        psiyy, psiyy_adv = psiyy_work, psiyy
        Kraken.logfv_stress_from_log_2d!(
            tauxx, tauxy, tauyy, psixx, psixy, psiyy, cfg.prefactor;
            model_code=Kraken.LOGFV_MODEL_OLDROYDB, L2=T(0.0), sync=false,
        )
    end
    KernelAbstractions.synchronize(backend)

    return (;
        psixx=Array(psixx),
        psixy=Array(psixy),
        psiyy=Array(psiyy),
        dudx=Array(dudx),
        dudy=Array(dudy),
        dvdx=Array(dvdx),
        dvdy=Array(dvdy),
        tauxx=Array(tauxx),
        tauxy=Array(tauxy),
        tauyy=Array(tauyy),
    )
end

function y_profile_average(field::AbstractMatrix{T}) where {T}
    Nx, Ny = size(field)
    profile = zeros(T, Ny)
    for j in 1:Ny
        total = zero(T)
        for i in 1:Nx
            total += field[i, j]
        end
        profile[j] = total / T(Nx)
    end
    return profile
end

function interior_rel_l2(profile, reference, jlo::Integer, jhi::Integer)
    num = zero(eltype(profile))
    den = zero(eltype(profile))
    for j in jlo:jhi
        num += abs2(profile[j] - reference[j])
        den += abs2(reference[j])
    end
    return den == 0 ? oftype(num, Inf) : sqrt(num / den)
end

function component_profile_errors(result, ref)
    tauxy_profile = y_profile_average(result.tauxy)
    tauxy_ref_profile = y_profile_average(ref.tauxy)
    n1_profile = y_profile_average(result.tauxx .- result.tauyy)
    n1_ref_profile = y_profile_average(ref.tauxx .- ref.tauyy)
    Ny = length(tauxy_profile)
    return (;
        tauxy=interior_rel_l2(tauxy_profile, tauxy_ref_profile, 2, Ny - 1),
        n1=interior_rel_l2(n1_profile, n1_ref_profile, 2, Ny - 1),
    )
end

function all_finite(values...)
    for value in values
        if value isa AbstractArray
            all(isfinite, value) || return false
        else
            isfinite(value) || return false
        end
    end
    return true
end

function print_m8_report(errors, finite::Bool, bar)
    tauxy_pass = finite && errors.tauxy < bar
    n1_pass = finite && errors.n1 < bar
    println(@sprintf(
        "M8 τ_xy interior rel L2 = %.16e  bar 1e-3  %s",
        errors.tauxy,
        tauxy_pass ? "PASS" : "FAIL",
    ))
    println(@sprintf(
        "M8 N1   interior rel L2 = %.16e  bar 1e-3  %s",
        errors.n1,
        n1_pass ? "PASS" : "FAIL",
    ))
    println("M8 overall: " * (tauxy_pass && n1_pass ? "PASS" : "FAIL"))
    return tauxy_pass && n1_pass
end

function main(args=ARGS)
    parse_args(args)
    T = Float64
    exit_code = mktempdir() do _
        cfg = default_config(T)
        backend = KernelAbstractions.CPU()
        ref = poiseuille_reference_fields(cfg, T)
        result = run_polymer_pipeline(ref, cfg, backend, T)
        errors = component_profile_errors(result, ref)
        finite = all_finite(
            ref.ux, ref.uy, ref.psixx, ref.psixy, ref.psiyy, ref.tauxx, ref.tauxy, ref.tauyy,
            result.psixx, result.psixy, result.psiyy,
            result.dudx, result.dudy, result.dvdx, result.dvdy,
            result.tauxx, result.tauxy, result.tauyy,
            errors.tauxy, errors.n1,
        )
        print_m8_report(errors, finite, cfg.bar)
        return finite ? 0 : 1
    end
    exit(exit_code)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
