#!/usr/bin/env julia

using Printf
using Serialization
using KernelAbstractions
using Kraken

try
    @eval using Metal
catch
end

const OUTDIR = joinpath("scratch", "M57")
const CSV = joinpath(OUTDIR, "M48_forensic_R30_first_nan_cell.csv")
const PRE_NAN = joinpath(OUTDIR, "M48_forensic_R30_pre_nan_snapshot.jls")
const QX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
const QY = (0, 0, 1, 0, -1, 1, 1, -1, -1)

function backend_choice()
    requested = lowercase(get(ENV, "KRAKEN_BACKEND", "metal"))
    if requested == "metal" && isdefined(Main, :Metal)
        try
            metal = getfield(Main, :Metal)
            @eval KernelAbstractions.allocate(::Metal.MetalBackend, ::Type{T}, dims::Tuple;
                                              unified=nothing) where {T} =
                Metal.MtlArray{T}(undef, dims)
            @eval KernelAbstractions.zeros(::Metal.MetalBackend, ::Type{T}, dims::Tuple;
                                           unified=nothing) where {T} =
                Metal.zeros(T, dims)
            return metal.MetalBackend(), Float32, "metal"
        catch err
            @printf("M57_M48 backend_warning=metal_unavailable error=%s\n", sprint(showerror, err))
        end
    end
    return KernelAbstractions.CPU(), Float32, "cpu"
end

function snapshot(step, s)
    return (; step, ux=Array(s.ux), uy=Array(s.uy),
            tauxx=Array(s.tauxx), tauxy=Array(s.tauxy), tauyy=Array(s.tauyy),
            fx=Array(s.fx_poly), fy=Array(s.fy_poly))
end

function link_normal(q)
    h = hypot(Float64(QX[q]), Float64(QY[q]))
    return -Float64(QX[q]) / h, -Float64(QY[q]) / h
end

function theta_deg(i, j, q, qw, cx, cy)
    x = Float64(i - 1)
    y = Float64(j - 1)
    if q > 0 && qw > 0
        x += qw * Float64(QX[q])
        y += qw * Float64(QY[q])
    end
    return atan(y - cy, x - cx) * 180 / pi
end

function u2_sample(phi, is_solid, embedded, i, j, q)
    T = eltype(phi)
    nx = T(embedded.wall_nx[i, j])
    ny = T(embedded.wall_ny[i, j])
    nrm = hypot(nx, ny)
    nrm > sqrt(eps(T)) || return false, zero(T)
    nx /= nrm
    ny /= nrm
    H = hypot(T(QX[q]), T(QY[q]))
    x2 = T(i) + H * nx
    y2 = T(j) + H * ny
    return Kraken._fvfd_qw_bilinear_sample_2d(phi, is_solid, x2, y2, size(phi, 1), size(phi, 2))
end

function corrected_fo(seedx, seedy, phi, embedded, i, j)
    return Kraken._fvfd_apply_embedded_wall_gradient_2d(
        seedx[i, j], seedy[i, j], phi, embedded.wall_nx, embedded.wall_ny,
        embedded.wall_inv_distance_to_center, i, j,
    )
end

function analyze_snapshot(snap, first_i, first_j, q_wall, is_solid, cx, cy, R, FT)
    Nx, Ny = size(snap.ux)
    embedded = Kraken.fvfd_embedded_boundary_from_qwall_2d(q_wall; FT=FT)
    bc = Kraken.logfv_openx_wally_bcspec_2d()
    geom = Kraken.FVFDGeometry2D(is_solid, embedded, Kraken.FVFDPatch2D(one(FT), one(FT)), bc)
    z() = zeros(FT, Nx, Ny)
    dudx_m, dudy_m, dvdx_m, dvdy_m = z(), z(), z(), z()
    dudx_s, dudy_s, dvdx_s, dvdy_s = z(), z(), z(), z()
    Kraken.fvfd_velocity_gradient_embedded_2d!(
        dudx_m, dudy_m, dvdx_m, dvdy_m, FT.(snap.ux), FT.(snap.uy), geom,
    )
    Kraken.fvfd_velocity_gradient_2d!(
        dudx_s, dudy_s, dvdx_s, dvdy_s, FT.(snap.ux), FT.(snap.uy), is_solid,
        one(FT), one(FT), bc,
    )
    rows = NamedTuple[]
    for dj in -1:1, di in -1:1
        i = first_i + di
        j = first_j + dj
        1 <= i <= Nx && 1 <= j <= Ny || continue
        tau_mag = hypot(Float64(snap.tauxx[i, j]), Float64(snap.tauxy[i, j]), Float64(snap.tauyy[i, j]))
        fmag = hypot(Float64(snap.fx[i, j]), Float64(snap.fy[i, j]))
        push!(rows, (; kind="cell", di, dj, i, j, q=0, qw=0.0,
                     theta=theta_deg(i, j, 0, 0.0, cx, cy),
                     nx=Float64(embedded.wall_nx[i, j]), ny=Float64(embedded.wall_ny[i, j]),
                     ux=Float64(snap.ux[i, j]), uy=Float64(snap.uy[i, j]),
                     u2x=NaN, u2y=NaN, du_m55=NaN, du_fo=NaN, delta=NaN,
                     tauxx=Float64(snap.tauxx[i, j]), tauxy=Float64(snap.tauxy[i, j]),
                     tauyy=Float64(snap.tauyy[i, j]), fx=Float64(snap.fx[i, j]),
                     fy=Float64(snap.fy[i, j]), tau_mag, fmag))
        for q in 2:9
            qw = Float64(q_wall[i, j, q])
            nx, ny = link_normal(q)
            fx1, fy1 = corrected_fo(dudx_s, dudy_s, FT.(snap.ux), embedded, i, j)
            gx1, gy1 = corrected_fo(dvdx_s, dvdy_s, FT.(snap.uy), embedded, i, j)
            du_fo = hypot(nx * Float64(fx1) + ny * Float64(fy1),
                          nx * Float64(gx1) + ny * Float64(gy1))
            du_m55 = hypot(nx * Float64(dudx_m[i, j]) + ny * Float64(dudy_m[i, j]),
                           nx * Float64(dvdx_m[i, j]) + ny * Float64(dvdy_m[i, j]))
            okx, u2x = u2_sample(FT.(snap.ux), is_solid, embedded, i, j, q)
            oky, u2y = u2_sample(FT.(snap.uy), is_solid, embedded, i, j, q)
            push!(rows, (; kind="dir", di, dj, i, j, q, qw,
                         theta=theta_deg(i, j, q, qw, cx, cy), nx, ny,
                         ux=Float64(snap.ux[i, j]), uy=Float64(snap.uy[i, j]),
                         u2x=okx ? Float64(u2x) : NaN, u2y=oky ? Float64(u2y) : NaN,
                         du_m55, du_fo, delta=du_m55 - du_fo,
                         tauxx=Float64(snap.tauxx[i, j]), tauxy=Float64(snap.tauxy[i, j]),
                         tauyy=Float64(snap.tauyy[i, j]), fx=Float64(snap.fx[i, j]),
                         fy=Float64(snap.fy[i, j]), tau_mag, fmag))
        end
    end
    return rows
end

function write_rows(rows)
    mkpath(OUTDIR)
    open(CSV, "w") do io
        println(io, "kind,di,dj,i,j,q,q_w,theta_deg,wall_nx,wall_ny,ux,uy,u2x,u2y,du_dn_M55,du_dn_FO,delta,tauxx,tauxy,tauyy,fx,fy,tau_mag,F_poly_mag")
        for r in rows
            @printf(io, "%s,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    r.kind, r.di, r.dj, r.i, r.j, r.q, r.qw, r.theta, r.nx, r.ny,
                    r.ux, r.uy, r.u2x, r.u2y, r.du_m55, r.du_fo, r.delta,
                    r.tauxx, r.tauxy, r.tauyy, r.fx, r.fy, r.tau_mag, r.fmag)
        end
    end
end

function run_m48_forensic()
    backend, FT, backend_name = backend_choice()
    R = 30
    Umean = FT(0.005)
    beta = FT(0.59)
    BSD = FT(1)
    max_steps = parse(Int, get(ENV, "M57_M48_STEPS", "3000"))
    L_up = 15.0
    L_down = 15.0
    H = 4 * R
    Nx = ceil(Int, (L_up + L_down) * R)
    Ny = H
    cx = L_up * R
    cy = (Ny - 1) / 2
    nu_total = Umean * FT(R)
    nu_s = beta * nu_total
    nu_p = (one(FT) - beta) * nu_total
    lambda = FT(R) / Umean
    q_wall, is_solid = Kraken.precompute_q_wall_cylinder(Nx, Ny, cx, cy, R; FT=FT)
    latest = Ref{Any}(nothing)
    mkpath(OUTDIR)
    cb = function (step, s)
        latest[] = snapshot(step, s)
        if step == 1 || step % 100 == 0
            serialize(joinpath(OUTDIR, @sprintf("M48_forensic_R30_step_%05d.jls", step)), latest[])
        end
        return nothing
    end
    result = Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(;
        radius=R, H, L_up, L_down, nu_s=FT(nu_s), nu_p=FT(nu_p), lambda=FT(lambda),
        polymer_model=:oldroydb, L_max=FT(10), u_mean=FT(Umean), Fx_body=FT(0),
        bsd_fraction=FT(BSD), polymer_substeps=:auto, subcycle_relative_tolerance=FT(0.01),
        max_deformation_increment=FT(0.05), max_memory_deformation_increment=FT(0.07),
        max_polymer_substeps=64, max_steps, avg_window=max_steps, drag_stride=500,
        diagnostic_stride=1, embedded_geometry=:qwall, embedded_gradient=true,
        embedded_advection=false, embedded_force=false, embedded_drag=false,
        advection_scheme=:muscl_superbee, wall_bc=:halfwayBB,
        embedded_circle_samples=32, force_boundary_fill=:bc_aware,
        step_callback=cb, backend, T=FT,
    )
    snap = latest[]
    if snap !== nothing
        serialize(PRE_NAN, snap)
    end
    nan_step = result.first_nonfinite_step
    first_i = result.first_nonfinite_i
    first_j = result.first_nonfinite_j
    if nan_step == 0 && snap !== nothing
        f = snap.fx .^ 2 .+ snap.fy .^ 2
        idx = argmax(f)
        first_i, first_j = Tuple(idx)
    end
    rows = snap === nothing ? NamedTuple[] :
           analyze_snapshot(snap, first_i, first_j, q_wall, is_solid, cx, cy, R, FT)
    write_rows(rows)
    dir_rows = [r for r in rows if r.kind == "dir"]
    worst = isempty(dir_rows) ? nothing : dir_rows[argmax(abs.(getfield.(dir_rows, :delta)))]
    worst_txt = worst === nothing ? "none" :
                @sprintf("cell=(%d,%d) q=%d qw=%.6g delta=%.6g",
                         worst.i, worst.j, worst.q, worst.qw, worst.delta)
    @printf("M57_M48_forensic backend=%s completed=%d first_nonfinite_step=%d field=%s cell=(%d,%d) pre_nan_step=%s worst_delta=%s csv=%s\n",
            backend_name, result.completed_steps, nan_step, string(result.first_nonfinite_field),
            first_i, first_j, snap === nothing ? "none" : string(snap.step), worst_txt, CSV)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(run_m48_forensic())
end
