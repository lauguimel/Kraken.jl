#!/usr/bin/env julia

using Printf
using KernelAbstractions
using Kraken

include("L3_recompute_cylinder.jl")

const L3G_CSV = joinpath(M57_DIR, "L3_guo_trajectory.csv")
const QX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
const QY = (0, 0, 1, 0, -1, 1, 1, -1, -1)
const QW = (4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)

function d2q9_eq(::Type{T}, rho, ux, uy, q) where {T}
    cu = T(3) * (T(QX[q]) * ux + T(QY[q]) * uy)
    u2 = ux * ux + uy * uy
    return T(QW[q]) * rho * (one(T) + cu + cu * cu / T(2) - T(1.5) * u2)
end

function run_l3_guo()
    backend, FT, backend_name = backend_choice()
    warmup_steps = parse(Int, get(ENV, "M57_L3_WARMUP_STEPS", "10000"))
    steps = parse(Int, get(ENV, "M57_L3_GUO_STEPS", "5000"))
    log_every = parse(Int, get(ENV, "M57_L3_LOG_EVERY", "100"))
    warm, warm_s = warmup_newtonian(backend, FT, warmup_steps)
    c = m57_l3_context(backend, FT, warm)
    Nx, Ny = size(warm.ux)

    Umean = FT(0.005)
    beta = FT(0.59)
    nu_total = Umean * FT(10)
    nu_s = beta * nu_total
    nu_p = (one(FT) - beta) * nu_total
    bsd = one(FT)
    lambda = FT(2000)
    prefactor = nu_p / lambda
    nu_lbm = nu_s + bsd * nu_p

    m57_compute_gradient!(c; sync=true)
    max_grad0 = m57_max_grad(c)
    substeps = m57_substeps(max_grad0, lambda)
    dt_poly = one(FT) / FT(substeps)

    step_geom = Kraken.transfer_step_geometry_2d(warm.geometry, backend)
    q_wall = step_geom.q_wall
    is_solid = step_geom.is_solid
    u_profile_h = Kraken.parabolic_face_profile_2d(
        warm.geometry; face=:west, mean_velocity=Umean, FT=FT,
    )
    u_profile = KernelAbstractions.allocate(backend, FT, Ny)
    copyto!(u_profile, u_profile_h)
    bcspec = Kraken.default_step_bcspec_2d(step_geom, u_profile, one(FT))

    f_in = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    f_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    f_h = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_h[i, j, q] = warm.is_solid[i, j] ? FT(QW[q]) :
                       d2q9_eq(FT, one(FT), FT(warm.ux[i, j]), FT(warm.uy[i, j]), q)
    end
    copyto!(f_in, f_h)
    rho = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    uwx = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    uwy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    fx_total = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    fy_total = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ux_frozen = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    uy_frozen = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    copyto!(ux_frozen, FT.(warm.ux))
    copyto!(uy_frozen, FT.(warm.uy))

    mkpath(M57_DIR)
    final_trace = NaN
    nan_step = 0
    open(L3G_CSV, "w") do io
        println(io, "step,trace_C_max_at_k,F_poly_max_at_k,F_poly_i,F_poly_j,first_nan_i,first_nan_j")
        for step in 1:steps
            m57_compute_gradient!(c; sync=false)
            m57_polymer_step!(c, lambda, dt_poly, substeps, prefactor)
            Kraken.logfv_bsd_correct_force_bc_aware_2d!(
                fx_total, fy_total, c.fx, c.fy, c.ux, c.uy, is_solid,
                bsd, nu_p, one(FT), one(FT), c.bc; sync=false,
            )
            Kraken.logfv_add_constant_force_fluid_2d!(
                fx_total, fy_total, is_solid, zero(FT), zero(FT); sync=false,
            )
            Kraken.fused_trt_libb_v2_guo_field_step!(
                f_out, f_in, rho, c.ux, c.uy, is_solid, q_wall, uwx, uwy,
                fx_total, fy_total, Nx, Ny, nu_lbm; wall_bc=:halfwayBB,
            )
            Kraken.apply_bc_rebuild_2d!(f_out, f_in, bcspec, nu_lbm, Nx, Ny)
            Kraken.logfv_compute_macroscopic_forced_field_2d!(
                rho, c.ux, c.uy, f_out, fx_total, fy_total; sync=false,
            )
            copyto!(c.ux, ux_frozen)
            copyto!(c.uy, uy_frozen)
            f_in, f_out = f_out, f_in
            if step == 1 || step % log_every == 0 || step == steps
                KernelAbstractions.synchronize(backend)
                scan = first_scan(c.psixx, c.psixy, c.psiyy, c.fx, c.fy, c.is_solid_h)
                final_trace = scan.max_trace
                bad_i = scan.first_bad === nothing ? 0 : scan.first_bad.i
                bad_j = scan.first_bad === nothing ? 0 : scan.first_bad.j
                @printf(io, "%d,%.17g,%.17g,%d,%d,%d,%d\n",
                        step, scan.max_trace, scan.max_force, scan.force_loc.i,
                        scan.force_loc.j, bad_i, bad_j)
                flush(io)
                if scan.first_bad !== nothing
                    nan_step = step
                    break
                end
            end
        end
    end
    verdict = nan_step > 0 ? "RED" : final_trace < 200 ? "GREEN" :
              final_trace < 1000 ? "YELLOW" : "RED"
    measured = nan_step > 0 ? @sprintf("nan_step=%d", nan_step) :
               @sprintf("trace_C_max=%.17g", final_trace)
    @printf("M57_L3_guo backend=%s warmup_s=%.3f substeps=%d verdict=%s %s csv=%s\n",
            backend_name, warm_s, substeps, verdict, measured, L3G_CSV)
    return verdict == "RED" ? 1 : 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(run_l3_guo())
end
