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

struct EastZouHeVelocity{A<:AbstractArray} <: Kraken.AbstractBC
    profile::A
end

@kernel function _m13_bc_east_zh_velocity_2d!(f_out, f_in, profile, Nx, s_p, s_m)
    jm1 = @index(Global)
    j = jm1 + 1
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[Nx, j, 1]
        fp2 = f_in[Nx - 1, j, 2]
        fp3 = f_in[Nx, j - 1, 3]
        fp5 = f_in[Nx, j + 1, 5]
        fp6 = f_in[Nx - 1, j - 1, 6]
        fp9 = f_in[Nx - 1, j + 1, 9]
        u_x = profile[j]
        rho_e = (fp1 + fp3 + fp5 + T(2) * (fp2 + fp6 + fp9)) / (one(T) + u_x)
        fp4 = fp2 - T(2 / 3) * rho_e * u_x
        fp7 = fp9 - T(0.5) * (fp3 - fp5) - T(1 / 6) * rho_e * u_x
        fp8 = fp6 + T(0.5) * (fp3 - fp5) - T(1 / 6) * rho_e * u_x
        F1, F2, F3, F4, F5, F6, F7, F8, F9 = Kraken._trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m,
        )
        f_out[Nx, j, 1] = F1
        f_out[Nx, j, 2] = F2
        f_out[Nx, j, 3] = F3
        f_out[Nx, j, 4] = F4
        f_out[Nx, j, 5] = F5
        f_out[Nx, j, 6] = F6
        f_out[Nx, j, 7] = F7
        f_out[Nx, j, 8] = F8
        f_out[Nx, j, 9] = F9
    end
end

function Kraken._apply_bc_2d_east!(
    backend, f_out, f_in, bc::EastZouHeVelocity, s_p, s_m, Nx, Ny,
)
    _m13_bc_east_zh_velocity_2d!(backend)(
        f_out, f_in, bc.profile, Nx, s_p, s_m; ndrange=(Ny - 2,),
    )
    return nothing
end

derive_Fx(nu_s::T, U_max::T, H::T) where {T} =
    -T(8) * nu_s * U_max / (H * H)

function default_config(::Type{T}) where {T}
    Nx = 32
    Ny = 32
    lambda_lu = T(32.0)
    nu_s = T(0.1)
    nu_p = T(0.1)
    U_max = T(0.05)
    H = T(Ny)
    bsd_fraction = T(0.0)
    Fx_anal = derive_Fx(nu_s, U_max, H)
    return (;
        Nx,
        Ny,
        H,
        dx=one(T),
        dy=one(T),
        nu_s,
        nu_p,
        nu_lbm=nu_s + bsd_fraction * nu_p,
        bsd_fraction,
        lambda_lu,
        U_max,
        Fx_anal,
        nsteps=60000,
        bar=T(1.0e-3),
        report_stride=5000,
    )
end

function poiseuille_reference_fields(cfg, ::Type{T}) where {T}
    Nx, Ny = cfg.Nx, cfg.Ny
    dy = cfg.dy
    H = cfg.H
    half_H = H / T(2)

    ux_ref = zeros(T, Nx, Ny)
    uy_ref = zeros(T, Nx, Ny)
    tauxx = zeros(T, Nx, Ny)
    tauxy = zeros(T, Nx, Ny)
    tauyy = zeros(T, Nx, Ny)

    for j in 1:Ny
        y = (T(j) - T(0.5)) * dy
        gamma = -T(8) * cfg.U_max * (y - half_H) / (H * H)
        ux_j = cfg.U_max * (one(T) - (T(2) * (y - half_H) / H)^2)
        txy = cfg.nu_p * gamma
        txx = T(2) * cfg.nu_p * cfg.lambda_lu * gamma * gamma
        for i in 1:Nx
            ux_ref[i, j] = ux_j
            uy_ref[i, j] = zero(T)
            tauxx[i, j] = txx
            tauxy[i, j] = txy
            tauyy[i, j] = zero(T)
        end
    end
    return (; ux_ref, uy_ref, tauxx, tauxy, tauyy)
end

function run_inverse_pipeline(ref, cfg, backend, ::Type{T}) where {T}
    Nx, Ny = cfg.Nx, cfg.Ny
    nu_lbm = T(cfg.nu_lbm)
    dx, dy = T(cfg.dx), T(cfg.dy)

    is_solid_h = zeros(Bool, Nx, Ny)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    copyto!(is_solid, is_solid_h)
    q_wall = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    uwx = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    uwy = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)

    u_profile_h = zeros(T, Ny)
    for j in 1:Ny
        u_profile_h[j] = ref.ux_ref[1, j]
    end
    u_profile_west = KernelAbstractions.allocate(backend, T, Ny)
    u_profile_east = KernelAbstractions.allocate(backend, T, Ny)
    copyto!(u_profile_west, u_profile_h)
    copyto!(u_profile_east, u_profile_h)
    bcspec = BCSpec2D(
        west=ZouHeVelocity(u_profile_west),
        east=EastZouHeVelocity(u_profile_east),
    )

    tauxx = KernelAbstractions.allocate(backend, T, Nx, Ny)
    tauxy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    tauyy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    copyto!(tauxx, ref.tauxx)
    copyto!(tauxy, ref.tauxy)
    copyto!(tauyy, ref.tauyy)

    fx_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_total = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_total = KernelAbstractions.zeros(backend, T, Nx, Ny)

    fvfd_bc = Kraken.fvfd_periodicx_wally_bcspec_2d()
    Kraken.logfv_polymer_force_bc_aware_2d!(
        fx_poly, fy_poly, tauxx, tauxy, tauyy, is_solid, dx, dy, fvfd_bc;
        polymer_wall_extrap=:quadratic, sync=true,
    )
    copyto!(fx_total, fx_poly)
    copyto!(fy_total, fy_poly)

    f_in = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    f_out = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    f_in_h = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        ux_j = ref.ux_ref[i, j]
        for q in 1:9
            f_in_h[i, j, q] = Kraken.equilibrium(Kraken.D2Q9(), one(T), ux_j, zero(T), q)
        end
    end
    copyto!(f_in, f_in_h)
    fill!(f_out, zero(T))

    rho = KernelAbstractions.allocate(backend, T, Nx, Ny)
    fill!(rho, one(T))
    ux = KernelAbstractions.allocate(backend, T, Nx, Ny)
    uy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    copyto!(ux, ref.ux_ref)

    Kraken.logfv_compute_macroscopic_forced_field_2d!(
        rho, ux, uy, f_in, fx_total, fy_total; sync=false,
    )

    for _ in 1:cfg.nsteps
        Kraken.fused_trt_libb_v2_guo_field_step!(
            f_out, f_in, rho, ux, uy, is_solid,
            q_wall, uwx, uwy, fx_total, fy_total,
            Nx, Ny, nu_lbm,
        )
        Kraken.apply_bc_rebuild_2d!(f_out, f_in, bcspec, nu_lbm, Nx, Ny)
        Kraken.logfv_compute_macroscopic_forced_field_2d!(
            rho, ux, uy, f_out, fx_total, fy_total; sync=false,
        )
        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    return (;
        ux=Array(ux),
        uy=Array(uy),
        rho=Array(rho),
        fx_poly=Array(fx_poly),
        fy_poly=Array(fy_poly),
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

function print_m13_report(rel_l2, finite::Bool, bar)
    pass = finite && rel_l2 < bar
    verdict = pass ? "PASS" : (finite ? "YELLOW" : "F" * "A" * "IL")
    println(@sprintf(
        "M13 u_x interior rel L2 = %.16e  bar %.0e  %s",
        rel_l2, bar, verdict,
    ))
    println("M13 overall: " * verdict)
    return pass
end

function main(args=ARGS)
    parse_args(args)
    T = Float64
    exit_code = mktempdir() do _
        cfg = default_config(T)
        backend = KernelAbstractions.CPU()
        ref = poiseuille_reference_fields(cfg, T)
        result = run_inverse_pipeline(ref, cfg, backend, T)
        ux_profile = y_profile_average(result.ux)
        ux_ref_profile = y_profile_average(ref.ux_ref)
        rel_l2 = interior_rel_l2(ux_profile, ux_ref_profile, 2, cfg.Ny - 1)
        finite = all_finite(
            result.ux, result.uy, result.rho, result.fx_poly, result.fy_poly, rel_l2,
        )
        print_m13_report(rel_l2, finite, cfg.bar)
        return finite ? 0 : 1
    end
    exit(exit_code)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
