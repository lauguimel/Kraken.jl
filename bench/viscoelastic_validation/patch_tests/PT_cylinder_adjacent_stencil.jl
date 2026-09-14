#!/usr/bin/env julia

using Printf
using Statistics
using Kraken

const OUTDIR = joinpath(@__DIR__, "..", "..", "..", "scratch", "M52b_canary")
const CSV = joinpath(OUTDIR, "cyl_adj_canary.csv")

const D2Q9_CX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
const D2Q9_CY = (0, 0, 1, 0, -1, 1, 1, -1, -1)

function c1_velocity!(ux, uy, is_solid, cx, cy, R)
    R2 = R * R
    @inbounds for j in axes(ux, 2), i in axes(ux, 1)
        uy[i, j] = 0.0
        if !is_solid[i, j]
            x = i - 1.0
            y = j - 1.0
            ux[i, j] = ((x - cx)^2 + (y - cy)^2 - R2) / R2
        end
    end
    return nothing
end

function cutlink_rows(path, q_wall, dudx, dudy, cx, cy, R)
    rows = NamedTuple[]
    analytic = 2.0 / R
    @inbounds for j in axes(q_wall, 2), i in axes(q_wall, 1), q in 2:9
        qw = q_wall[i, j, q]
        0.1 <= qw <= 0.9 || continue
        xw = (i - 1.0) + qw * D2Q9_CX[q]
        yw = (j - 1.0) + qw * D2Q9_CY[q]
        rx = xw - cx
        ry = yw - cy
        r = hypot(rx, ry)
        nx = rx / r
        ny = ry / r
        du_dn = nx * dudx[i, j] + ny * dudy[i, j]
        err = abs(du_dn - analytic)
        theta = atan(ry, rx) * 180.0 / pi
        push!(rows, (case="C1", path=path, cell_i=i, cell_j=j, q_w=qw, theta_deg=theta,
                    nx=nx, ny=ny, dudx_comp=dudx[i, j], dudy_comp=dudy[i, j],
                    du_dn_comp=du_dn, du_dn_analytic=analytic, abs_err=err,
                    rel_err=err / abs(analytic)))
    end
    return rows
end

function write_csv(rows)
    mkpath(OUTDIR)
    open(CSV, "w") do io
        println(io, "case,path,cell_i,cell_j,q_w,theta_deg,nx,ny,dudx_comp,dudy_comp,du_dn_comp,du_dn_analytic,abs_err,rel_err")
        for r in rows
            @printf(io, "%s,%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    r.case, string(r.path), r.cell_i, r.cell_j, r.q_w, r.theta_deg, r.nx, r.ny,
                    r.dudx_comp, r.dudy_comp, r.du_dn_comp, r.du_dn_analytic,
                    r.abs_err, r.rel_err)
        end
    end
end

corr(xs, ys) = length(xs) < 2 ? NaN :
    sum((xs .- mean(xs)) .* (ys .- mean(ys))) /
    sqrt(sum(abs2, xs .- mean(xs)) * sum(abs2, ys .- mean(ys)))

function path_stats(rows, path)
    rs = [r for r in rows if r.path == path]
    errs = [r.abs_err for r in rs]
    qws = [r.q_w for r in rs]
    return (n=length(rs), mean=mean(errs), median=median(errs), max=maximum(errs),
            corr=corr(qws, errs))
end

function main()
    t0 = time()
    Nx = Ny = 32
    cx = cy = 16.5
    R = 4.0
    dx = dy = 1.0

    q_wall, is_solid = Kraken.precompute_q_wall_cylinder(Nx, Ny, cx, cy, R; FT=Float64)
    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    c1_velocity!(ux, uy, is_solid, cx, cy, R)

    dudx_d = zeros(Float64, Nx, Ny)
    dudy_d = similar(dudx_d)
    dvdx_d = similar(dudx_d)
    dvdy_d = similar(dudx_d)
    bc = Kraken.fvfd_periodicx_wally_bcspec_2d()
    Kraken.fvfd_velocity_gradient_2d!(
        dudx_d, dudy_d, dvdx_d, dvdy_d, ux, uy, is_solid, dx, dy, bc,
    )

    dudx_e = zeros(Float64, Nx, Ny)
    dudy_e = similar(dudx_e)
    dvdx_e = similar(dudx_e)
    dvdy_e = similar(dudx_e)
    embedded_h = Kraken.fvfd_embedded_boundary_from_qwall_2d(q_wall; FT=Float64)
    Kraken.fvfd_velocity_gradient_embedded_2d!(
        dudx_e, dudy_e, dvdx_e, dvdy_e, ux, uy, is_solid, dx, dy, bc, embedded_h,
    )

    rows = vcat(
        cutlink_rows(:default, q_wall, dudx_d, dudy_d, cx, cy, R),
        cutlink_rows(:embedded, q_wall, dudx_e, dudy_e, cx, cy, R),
    )
    write_csv(rows)
    default = path_stats(rows, :default)
    embedded = path_stats(rows, :embedded)
    improvement = default.mean / embedded.mean
    verdict = embedded.mean < 1e-2 && embedded.max < 5e-2 ? "GREEN" :
              improvement >= 10 ? "YELLOW" : "RED"
    @printf("[M52b-CYL-ADJ-CANARY] csv at %s\n", CSV)
    @printf("- Default mean abs_err: %.17g; max: %.17g; corr(q_w, abs_err): %.17g\n",
            default.mean, default.max, default.corr)
    @printf("- Embedded mean abs_err: %.17g; max: %.17g; corr(q_w, abs_err): %.17g\n",
            embedded.mean, embedded.max, embedded.corr)
    @printf("- Improvement factor: %.17g\n- Verdict: %s\n- Wall time: %.6f s\n",
            improvement, verdict, time() - t0)
    return verdict == "GREEN" ? 0 : 1
end

exit(main())
