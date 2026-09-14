#!/usr/bin/env julia

using Printf
using Statistics
using KernelAbstractions
using Kraken

include("L3_frozen_u_cylinder.jl")

const M57_OUTDIR = joinpath("scratch", "M57")
const M57_CSV = joinpath(M57_OUTDIR, "L3b_per_qw_bin.csv")
const D2Q9_CX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
const D2Q9_CY = (0, 0, 1, 0, -1, 1, 1, -1, -1)

function link_normal(q)
    cx = Float64(D2Q9_CX[q])
    cy = Float64(D2Q9_CY[q])
    h = hypot(cx, cy)
    return -cx / h, -cy / h
end

function normal_grad_mag(dudx, dudy, dvdx, dvdy, i, j, q)
    nx, ny = link_normal(q)
    du_dn = nx * Float64(dudx[i, j]) + ny * Float64(dudy[i, j])
    dv_dn = nx * Float64(dvdx[i, j]) + ny * Float64(dvdy[i, j])
    return hypot(du_dn, dv_dn)
end

function bin_index(qw)
    return clamp(fld(Int(floor(10 * (Float64(qw) - 0.1))), 1) + 1, 1, 8)
end

function run_l3b()
    backend, FT, backend_name = backend_choice()
    warmup_steps = parse(Int, get(ENV, "M57_L3_WARMUP_STEPS", "10000"))
    warm, warm_s = warmup_newtonian(backend, FT, warmup_steps)
    Nx, Ny = size(warm.ux)
    R = 10
    cx = 15.0 * R
    cy = (Ny - 1) / 2
    q_wall, is_solid_h = Kraken.precompute_q_wall_cylinder(Nx, Ny, cx, cy, R; FT=FT)
    embedded_h = Kraken.fvfd_embedded_boundary_from_qwall_2d(q_wall; FT=FT)
    bc = Kraken.logfv_openx_wally_bcspec_2d()
    geom_h = Kraken.FVFDGeometry2D(
        is_solid_h, embedded_h, Kraken.FVFDPatch2D(one(FT), one(FT)), bc,
    )
    geom = Kraken.fvfd_transfer_geometry_2d(geom_h, backend, FT)

    ux = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    uy = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    copyto!(ux, FT.(warm.ux))
    copyto!(uy, FT.(warm.uy))

    dudx_m = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    dudy_m = similar(dudx_m)
    dvdx_m = similar(dudx_m)
    dvdy_m = similar(dudx_m)
    dudx_f = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    dudy_f = similar(dudx_f)
    dvdx_f = similar(dudx_f)
    dvdy_f = similar(dudx_f)

    Kraken.fvfd_velocity_gradient_embedded_2d!(
        dudx_m, dudy_m, dvdx_m, dvdy_m, ux, uy, geom; sync=true,
    )
    Kraken.fvfd_velocity_gradient_2d!(
        dudx_f, dudy_f, dvdx_f, dvdy_f, ux, uy, geom; sync=true,
    )

    dm = Array(dudx_m); dmy = Array(dudy_m); vm = Array(dvdx_m); vmy = Array(dvdy_m)
    df = Array(dudx_f); dfy = Array(dudy_f); vf = Array(dvdx_f); vfy = Array(dvdy_f)

    n = zeros(Int, 8)
    sum_delta = zeros(Float64, 8)
    max_delta = zeros(Float64, 8)
    sum_fo = zeros(Float64, 8)
    max_fo = zeros(Float64, 8)
    worst = fill("", 8)

    for j in 1:Ny, i in 1:Nx, q in 2:9
        qw = Float64(q_wall[i, j, q])
        0.1 <= qw <= 0.9 || continue
        b = bin_index(qw)
        m55 = normal_grad_mag(dm, dmy, vm, vmy, i, j, q)
        fo = normal_grad_mag(df, dfy, vf, vfy, i, j, q)
        delta = abs(m55 - fo)
        n[b] += 1
        sum_delta[b] += delta
        sum_fo[b] += fo
        max_fo[b] = max(max_fo[b], fo)
        if delta > max_delta[b]
            max_delta[b] = delta
            worst[b] = @sprintf("(%d,%d;q=%d;qw=%.6g)", i, j, q, qw)
        end
    end

    mkpath(M57_OUTDIR)
    open(M57_CSV, "w") do io
        println(io, "qw_low,qw_high,n_cells,mean_delta,max_abs_delta,mean_FO,max_abs_FO,ratio_M55_vs_FO,abs_max_cell")
        for b in 1:8
            low = 0.1 + 0.1 * (b - 1)
            high = low + 0.1
            md = n[b] == 0 ? NaN : sum_delta[b] / n[b]
            mf = n[b] == 0 ? NaN : sum_fo[b] / n[b]
            ratio = max_delta[b] / max(max_fo[b], eps(Float64))
            @printf(io, "%.2f,%.2f,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%s\n",
                    low, high, n[b], md, max_delta[b], mf, max_fo[b], ratio, worst[b])
        end
    end

    global_max_delta = maximum(max_delta)
    global_max_fo = maximum(max_fo)
    ratio = global_max_delta / max(global_max_fo, eps(Float64))
    verdict = ratio <= 0.5 ? "GREEN" : ratio < 2.0 ? "YELLOW" : "RED"
    worst_bin = argmax(max_delta)
    @printf("M57_L3b backend=%s warmup_s=%.3f verdict=%s max_delta=%.17g max_FO=%.17g ratio=%.17g worst_bin=%.2f-%.2f csv=%s\n",
            backend_name, warm_s, verdict, global_max_delta, global_max_fo, ratio,
            0.1 + 0.1 * (worst_bin - 1), 0.2 + 0.1 * (worst_bin - 1), M57_CSV)
    return verdict == "RED" ? 1 : 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(run_l3b())
end
