# Generate the validation data (CSV) for the incompressible steady Navier-Stokes
# (FVFD/SIMPLE) documentation page: docs/src/users/incompressible-navier-stokes.md.
#
# Runs the standalone solvers on the CPU backend and writes the centreline /
# channel profiles next to the analytic / Ghia (1982) references into
# docs/incns_figdata/*.csv. The dark figures are then rendered by
# docs/plot_incns_validation.py (krakendark, conda env kraken-v0-3-figures).
#
# Usage (root env is the only env that resolves the solver includes):
#   julia --project=. -t auto docs/generate_incns_figures.jl            # all cases
#   julia --project=. -t auto docs/generate_incns_figures.jl poiseuille re100
#   julia --project=. -t auto docs/generate_incns_figures.jl re1000     # long (~5-15 min CPU)
#
# Cases (parameters mirror the validation tests / the A100 bench driver):
#   poiseuille  solve_incns_simple, body-force plane Poiseuille, 8x64
#               (test/analytical/incns_poiseuille.jl)
#   re100       solve_incns_cavity_mg, lid-driven cavity Re=100, 128^2
#               (test/analytical/incns_cavity_mg_ghia.jl)
#   re1000      solve_incns_cavity_mg, lid-driven cavity Re=1000, 256^2
#               (benchmarks/krk/inc_ns/cavity_gpu_bench.jl settings; the 512^2
#               case of the A100 bench is a multi-hour CPU run locally)

using Printf
using KernelAbstractions

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTDIR = joinpath(@__DIR__, "incns_figdata")
mkpath(OUTDIR)

include(joinpath(ROOT, "src", "methods", "inc_ns", "simple.jl"))
include(joinpath(ROOT, "src", "methods", "inc_ns", "cavity_mg.jl"))

# ---------------------------------------------------------------------------
# Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, 387-411 — Tables I and II.
# Same arrays as test/analytical/incns_cavity_mg_ghia.jl (the validation gate).
# ---------------------------------------------------------------------------
const GHIA_Y = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
                0.9531, 0.9609, 0.9688, 0.9766, 1.0000]
const GHIA_X = [0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
                0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
                0.9453, 0.9531, 0.9609, 0.9688, 1.0000]
const GHIA_RE100_U = [0.0000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                      -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                      0.68717, 0.73722, 0.78871, 0.84123, 1.00000]
const GHIA_RE100_V = [0.0000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
                      0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
                      -0.10313, -0.08864, -0.07391, -0.05906, 0.00000]
const GHIA_RE1000_U = [0.0000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
                       -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304,
                       0.46604, 0.51117, 0.57492, 0.65928, 1.00000]
const GHIA_RE1000_V = [0.0000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095,
                       0.33075, 0.32235, 0.02526, -0.31966, -0.42665, -0.51550,
                       -0.39188, -0.33714, -0.27669, -0.21388, 0.00000]

function write_csv(path, header, cols...)
    n = length(cols[1])
    open(path, "w") do io
        println(io, header)
        for k in 1:n
            println(io, join((@sprintf("%.10g", c[k]) for c in cols), ","))
        end
    end
    println("wrote $path  ($n rows)")
end

# Piecewise-linear interpolation of a cell-centred profile, with the wall
# Dirichlet values used outside the first/last centre (same convention as
# test/analytical/incns_cavity_mg_ghia.jl).
function interp_with_walls(vals, centers, q, wall_lo, wall_hi)
    if q <= centers[1]
        return wall_lo + (vals[1] - wall_lo) * (q - 0.0) / (centers[1] - 0.0)
    elseif q >= centers[end]
        return vals[end] + (wall_hi - vals[end]) * (q - centers[end]) / (1.0 - centers[end])
    else
        jhi = findfirst(c -> c >= q, centers)
        jlo = jhi - 1
        t = (q - centers[jlo]) / (centers[jhi] - centers[jlo])
        return (1 - t) * vals[jlo] + t * vals[jhi]
    end
end

# ---------------------------------------------------------------------------
# Case 1 — plane Poiseuille (solve_incns_simple), parameters of the analytic
# validation test.
# ---------------------------------------------------------------------------
function run_poiseuille()
    nx, ny = 8, 64
    H = 1.0; mu = 1.0; G = 1.0
    t = @elapsed res = solve_incns_simple(; nx, ny, H, mu, G,
                                          relax=(u=0.7, p=0.3),
                                          tol=1e-10, maxiter=300, backend=CPU())
    uprof = vec(sum(res.u; dims=1)) ./ nx                  # x-averaged profile
    uan = [(G / (2mu)) * y * (H - y) for y in res.ycenters]
    l2_rel = sqrt(sum(abs2, uprof .- uan) / ny) / sqrt(sum(abs2, uan) / ny)
    @printf("poiseuille: %dx%d, iters=%d, converged=%s, L2_rel=%.4f%%, %.2fs\n",
            nx, ny, res.iters, res.converged, 100l2_rel, t)
    write_csv(joinpath(OUTDIR, "poiseuille_profile.csv"),
              "y,u_kraken,u_analytic", res.ycenters, uprof, uan)
    return l2_rel
end

# ---------------------------------------------------------------------------
# Cases 2/3 — lid-driven cavity (solve_incns_cavity_mg) vs Ghia (1982).
# ---------------------------------------------------------------------------
function run_cavity(tag, N, Re, relax, tol, vel_tol, maxiter, ghia_u, ghia_v)
    U_lid = 1.0
    t = @elapsed res = solve_incns_cavity_mg(; nx=N, ny=N, U_lid, Re,
                                             relax, tol, vel_tol, maxiter,
                                             backend_ka=KernelAbstractions.CPU(),
                                             atype=Array{Float64})
    ic = N ÷ 2
    u_col = (res.u[ic, :] .+ res.u[ic + 1, :]) ./ 2 ./ U_lid   # u(y) at x=0.5
    jc = N ÷ 2
    v_row = (res.v[:, jc] .+ res.v[:, jc + 1]) ./ 2 ./ U_lid   # v(x) at y=0.5

    u_num = [interp_with_walls(u_col, res.ycenters, y, 0.0, 1.0) for y in GHIA_Y]
    v_num = [interp_with_walls(v_row, res.xcenters, x, 0.0, 0.0) for x in GHIA_X]
    max_err_u = maximum(abs.(u_num .- ghia_u))
    max_err_v = maximum(abs.(v_num .- ghia_v))
    @printf("cavity %s: %d^2, iters=%d, converged=%s, max|u-Ghia|=%.3f%%, max|v-Ghia|=%.3f%%, %.1fs\n",
            tag, N, res.iters, res.converged, 100max_err_u, 100max_err_v, t)

    # Full-resolution centreline profiles (with the wall points appended so the
    # plotted lines reach the boundaries).
    y_full = vcat(0.0, res.ycenters, 1.0)
    u_full = vcat(0.0, u_col, 1.0)
    x_full = vcat(0.0, res.xcenters, 1.0)
    v_full = vcat(0.0, v_row, 0.0)
    write_csv(joinpath(OUTDIR, "cavity_$(tag)_u_centerline.csv"),
              "y,u_kraken", y_full, u_full)
    write_csv(joinpath(OUTDIR, "cavity_$(tag)_v_centerline.csv"),
              "x,v_kraken", x_full, v_full)
    write_csv(joinpath(OUTDIR, "cavity_$(tag)_ghia_u.csv"),
              "y,u_ghia", GHIA_Y, ghia_u)
    write_csv(joinpath(OUTDIR, "cavity_$(tag)_ghia_v.csv"),
              "x,v_ghia", GHIA_X, ghia_v)
    return max(max_err_u, max_err_v)
end

const CASES = isempty(ARGS) ? ["poiseuille", "re100", "re1000"] : ARGS

"poiseuille" in CASES && run_poiseuille()
# Test-gate parameters (test/analytical/incns_cavity_mg_ghia.jl).
"re100" in CASES && run_cavity("re100", 128, 100.0, (u=0.7, p=0.3),
                               1e-7, 1e-6, 8000, GHIA_RE100_U, GHIA_RE100_V)
# A100 bench physical case (benchmarks/krk/inc_ns/cavity_gpu_bench.jl) at 256^2:
# 512^2 (the bench grid) is a multi-hour CPU run locally — the figure caption
# states the grid actually used.
"re1000" in CASES && run_cavity("re1000", 256, 1000.0, (u=0.5, p=0.2),
                                1e-7, 1e-6, 60000, GHIA_RE1000_U, GHIA_RE1000_V)
