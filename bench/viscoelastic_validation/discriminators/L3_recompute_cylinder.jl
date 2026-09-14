#!/usr/bin/env julia

using Printf
using KernelAbstractions
using Kraken

include("L3_frozen_u_cylinder.jl")

const M57_DIR = joinpath("scratch", "M57")
const L3R_CSV = joinpath(M57_DIR, "L3_recompute_trajectory.csv")

mutable struct M57L3Ctx
    backend; FT; is_solid_h; geom; bc; ux; uy
    ux_west; ux_east; uy_south; uy_north
    psixx; psixy; psiyy; psixx_adv; psixy_adv; psiyy_adv
    psixx_next; psixy_next; psiyy_next
    zero_w; east_xx; east_xy; east_yy; zero_s
    ux_face; uy_face; dudx; dudy; dvdx; dvdy; tauxx; tauxy; tauyy; fx; fy
end

function m57_l3_context(backend, FT, warm)
    R = 10
    Nx, Ny = size(warm.ux)
    cx = 15.0 * R
    cy = (Ny - 1) / 2
    q_wall, is_solid_h = Kraken.precompute_q_wall_cylinder(Nx, Ny, cx, cy, R; FT=FT)
    embedded_h = Kraken.fvfd_embedded_boundary_from_qwall_2d(q_wall; FT=FT)
    bc = Kraken.logfv_openx_wally_bcspec_2d()
    geom_h = Kraken.FVFDGeometry2D(is_solid_h, embedded_h, Kraken.FVFDPatch2D(one(FT), one(FT)), bc)
    geom = Kraken.fvfd_transfer_geometry_2d(geom_h, backend, FT)
    ux = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    uy = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    copyto!(ux, FT.(warm.ux)); copyto!(uy, FT.(warm.uy))
    ux_west = KernelAbstractions.allocate(backend, FT, Ny)
    ux_east = KernelAbstractions.allocate(backend, FT, Ny)
    uy_south = KernelAbstractions.allocate(backend, FT, Nx)
    uy_north = KernelAbstractions.allocate(backend, FT, Nx)
    copyto!(ux_west, FT.(warm.ux[1, :]))
    copyto!(ux_east, FT.(warm.ux[end, :]))
    copyto!(uy_south, FT.(warm.uy[:, 1]))
    copyto!(uy_north, FT.(warm.uy[:, end]))
    zny() = KernelAbstractions.zeros(backend, FT, Ny)
    znx() = KernelAbstractions.zeros(backend, FT, Nx)
    z2() = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    return M57L3Ctx(
        backend, FT, is_solid_h, geom, bc, ux, uy,
        ux_west, ux_east, uy_south, uy_north,
        z2(), z2(), z2(), z2(), z2(), z2(), z2(), z2(), z2(),
        zny(), zny(), zny(), zny(), znx(),
        KernelAbstractions.zeros(backend, FT, Nx + 1, Ny),
        KernelAbstractions.zeros(backend, FT, Nx, Ny + 1),
        z2(), z2(), z2(), z2(), z2(), z2(), z2(), z2(), z2(),
    )
end

function m57_compute_gradient!(c::M57L3Ctx; sync=true)
    Kraken.fvfd_velocity_gradient_embedded_2d!(
        c.dudx, c.dudy, c.dvdx, c.dvdy, c.ux, c.uy, c.geom; sync,
    )
end

function m57_max_grad(c::M57L3Ctx)
    KernelAbstractions.synchronize(c.backend)
    vals = (Array(c.dudx), Array(c.dudy), Array(c.dvdx), Array(c.dvdy))
    return maximum(maximum(abs, a) for a in vals)
end

function m57_grad_snapshot(c::M57L3Ctx)
    KernelAbstractions.synchronize(c.backend)
    return (Array(c.dudx), Array(c.dudy), Array(c.dvdx), Array(c.dvdy))
end

function m57_grad_drift(c::M57L3Ctx, g0)
    g = m57_grad_snapshot(c)
    return maximum(maximum(abs.(g[k] .- g0[k])) for k in 1:4)
end

function m57_substeps(max_grad, lambda)
    est = Kraken.logfv_oldroydb_subcycle_estimate(
        Float64(max_grad), Float64(lambda), 1.0;
        relative_tolerance=0.01, max_deformation_increment=0.05,
        max_memory_deformation_increment=0.07, min_substeps=1, max_substeps=64,
    )
    return est.recommended
end

function m57_polymer_step!(c::M57L3Ctx, lambda, dt_poly, substeps, prefactor)
    FT = c.FT
    Nx, _ = size(c.psixx)
    Kraken.logfv_copy_column_profile_2d!(c.east_xx, c.psixx, Nx; sync=false)
    Kraken.logfv_copy_column_profile_2d!(c.east_xy, c.psixy, Nx; sync=false)
    Kraken.logfv_copy_column_profile_2d!(c.east_yy, c.psiyy, Nx; sync=false)
    Kraken.logfv_cell_velocity_to_faces_bc_aware_2d!(
        c.ux_face, c.uy_face, c.ux, c.uy, c.geom.is_solid,
        c.ux_west, c.ux_east, c.uy_south, c.uy_north, c.bc; sync=false,
    )
    Kraken.logfv_advect_upwind_bc_aware_2d!(
        c.psixx_adv, c.psixy_adv, c.psiyy_adv, c.psixx, c.psixy, c.psiyy,
        c.zero_w, c.zero_w, c.zero_w, c.east_xx, c.east_xy, c.east_yy,
        c.zero_s, c.zero_s, c.zero_s, c.zero_s, c.zero_s, c.zero_s,
        c.ux_face, c.uy_face, c.geom.is_solid, one(FT), one(FT), c.bc, one(FT);
        sync=false, advection_scheme=:muscl_superbee,
    )
    px, py, pz = c.psixx_adv, c.psixy_adv, c.psiyy_adv
    for _ in 1:substeps
        Kraken.logfv_step_constitutive_log_2d!(
            c.psixx_next, c.psixy_next, c.psiyy_next, px, py, pz,
            c.dudx, c.dudy, c.dvdx, c.dvdy, lambda, dt_poly,
            Kraken.LOGFV_MODEL_OLDROYDB, zero(FT); sync=false,
        )
        px, c.psixx_next = c.psixx_next, px
        py, c.psixy_next = c.psixy_next, py
        pz, c.psiyy_next = c.psiyy_next, pz
    end
    c.psixx, c.psixx_adv = px, c.psixx
    c.psixy, c.psixy_adv = py, c.psixy
    c.psiyy, c.psiyy_adv = pz, c.psiyy
    Kraken.logfv_stress_from_log_2d!(
        c.tauxx, c.tauxy, c.tauyy, c.psixx, c.psixy, c.psiyy, prefactor; sync=false,
    )
    Kraken.fvfd_tensor_divergence_2d!(
        c.fx, c.fy, c.tauxx, c.tauxy, c.tauyy, c.geom.is_solid, one(FT), one(FT), c.bc; sync=false,
    )
end

function run_l3_recompute()
    backend, FT, backend_name = backend_choice()
    warmup_steps = parse(Int, get(ENV, "M57_L3_WARMUP_STEPS", "10000"))
    steps = parse(Int, get(ENV, "M57_L3_RECOMPUTE_STEPS", "5000"))
    log_every = parse(Int, get(ENV, "M57_L3_LOG_EVERY", "100"))
    warm, warm_s = warmup_newtonian(backend, FT, warmup_steps)
    c = m57_l3_context(backend, FT, warm)
    lambda = FT(2000)
    prefactor = FT((1.0 - 0.59) * 0.05 / 2000.0)
    m57_compute_gradient!(c; sync=true)
    g0 = m57_grad_snapshot(c)
    max_grad0 = maximum(maximum(abs, a) for a in g0)
    substeps = m57_substeps(max_grad0, lambda)
    dt_poly = one(FT) / FT(substeps)

    final_ratio = NaN
    mkpath(M57_DIR)
    open(L3R_CSV, "w") do io
        println(io, "step,max_grad_at_k,grad_drift_at_k,trace_C_max_at_k,F_poly_max_at_k,first_nan_i,first_nan_j")
        for step in 1:steps
            m57_compute_gradient!(c; sync=false)
            drift = (step == 1 || step % log_every == 0 || step == steps) ?
                    m57_grad_drift(c, g0) : NaN
            m57_polymer_step!(c, lambda, dt_poly, substeps, prefactor)
            if step == 1 || step % log_every == 0 || step == steps
                KernelAbstractions.synchronize(backend)
                scan = first_scan(c.psixx, c.psixy, c.psiyy, c.fx, c.fy, c.is_solid_h)
                bad_i = scan.first_bad === nothing ? 0 : scan.first_bad.i
                bad_j = scan.first_bad === nothing ? 0 : scan.first_bad.j
                maxg = m57_max_grad(c)
                final_ratio = drift / max(max_grad0, eps(Float64))
                @printf(io, "%d,%.17g,%.17g,%.17g,%.17g,%d,%d\n",
                        step, maxg, drift, scan.max_trace, scan.max_force, bad_i, bad_j)
                flush(io)
                scan.first_bad === nothing || break
            end
        end
    end
    verdict = final_ratio < 0.05 ? "GREEN" : final_ratio < 0.5 ? "YELLOW" : "RED"
    @printf("M57_L3_recompute backend=%s warmup_s=%.3f substeps=%d verdict=%s drift_ratio=%.17g max_grad0=%.17g csv=%s\n",
            backend_name, warm_s, substeps, verdict, final_ratio, max_grad0, L3R_CSV)
    return verdict == "RED" ? 1 : 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(run_l3_recompute())
end
