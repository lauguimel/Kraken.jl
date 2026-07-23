# Analytical validation: steady SIMPLE 2D lid-driven cavity at Re=100 against
# the benchmark data of Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, 387-411.
#
# Unit square [0,1]^2; top wall moves at u = U_lid = 1, v = 0; the other three
# walls are no-slip. Re = U_lid * L / nu = 100 (rho = 1 => mu = nu = 0.01).
#
# Asserts the converged centreline profiles match Ghia's tabulated values to
# <= 5% LOCAL at every sample point, measured as the max |u_num - u_ghia| over
# the vertical-centreline u(y) at x=0.5 and the horizontal-centreline v(x) at
# y=0.5 (both normalised by U_lid = 1; "5% local" = 0.05 in these units). Also
# checks convergence and a checkerboard-free pressure field (Rhie-Chow working).

using Test
using KernelAbstractions   # CPU() — no longer inherited once the include below auto-skips

if !isdefined(@__MODULE__, :solve_incns_cavity)
    include(joinpath(@__DIR__, "..", "..", "src", "methods", "inc_ns", "cavity.jl"))
end

# Ghia et al. (1982), Re=100. Vertical centreline (x=0.5): u/U_lid vs y/L.
const GHIA_RE100_Y = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                      0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
                      0.9531, 0.9609, 0.9688, 0.9766, 1.0000]
const GHIA_RE100_U = [0.0000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                      -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                      0.68717, 0.73722, 0.78871, 0.84123, 1.00000]
# Horizontal centreline (y=0.5): v/U_lid vs x/L.
const GHIA_RE100_X = [0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
                      0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
                      0.9453, 0.9531, 0.9609, 0.9688, 1.0000]
const GHIA_RE100_V = [0.0000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
                      0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
                      -0.10313, -0.08864, -0.07391, -0.05906, 0.00000]

const INCNS_CAVITY_RESULTS = Dict{Symbol,Any}()

# Linear interpolation of a cell-centred line profile `vals` (at `centers`) onto
# query `q in [0,1]`, with prescribed wall values `wall_lo` at 0 and `wall_hi`
# at 1 (Dirichlet velocity walls).
function _cavity_interp(vals, centers, q, wall_lo, wall_hi)
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

function incns_cavity_ghia_case(; nx::Integer=128, ny::Integer=128,
                                U_lid::Real=1.0, Re::Real=100.0,
                                tol::Real=1e-7, maxiter::Integer=6000,
                                backend=CPU())
    res = solve_incns_cavity(; nx, ny, U_lid, Re,
                             relax=(u=0.7, p=0.3), tol, maxiter, backend)

    # u(y) at x=0.5: average the two columns straddling the x=0.5 cell face.
    ic = nx ÷ 2
    u_col = (res.u[ic, :] .+ res.u[ic + 1, :]) ./ 2 ./ U_lid
    # v(x) at y=0.5: average the two rows straddling the y=0.5 cell face.
    jc = ny ÷ 2
    v_row = (res.v[:, jc] .+ res.v[:, jc + 1]) ./ 2 ./ U_lid

    # u walls: bottom (y=0) u=0, top (y=1, lid) u=1. v walls: both 0.
    u_at(q) = _cavity_interp(u_col, res.ycenters, q, 0.0, 1.0)
    v_at(q) = _cavity_interp(v_row, res.xcenters, q, 0.0, 0.0)

    u_num = [u_at(y) for y in GHIA_RE100_Y]
    v_num = [v_at(x) for x in GHIA_RE100_X]
    err_u = abs.(u_num .- GHIA_RE100_U)
    err_v = abs.(v_num .- GHIA_RE100_V)
    max_err_u = maximum(err_u)
    max_err_v = maximum(err_v)

    return (; res, u_num, v_num, err_u, err_v, max_err_u, max_err_v,
            max_err=max(max_err_u, max_err_v))
end

@testset "IncNS steady SIMPLE lid-driven cavity vs Ghia (1982) Re=100" begin
    backend = CPU()
    c = incns_cavity_ghia_case(; nx=128, ny=128, backend)
    INCNS_CAVITY_RESULTS[:re100] = c

    # Converged with a multi-order residual drop.
    @test c.res.converged
    rh = c.res.residual_history
    @test rh[end] / rh[1] < 1e-4          # >= 4 orders of magnitude drop

    # Centreline profiles match Ghia to <= 5% LOCAL at every sample point.
    @test c.max_err_u <= 0.05
    @test c.max_err_v <= 0.05

    # Pressure smooth (no checkerboard): the Rhie-Chow coupling keeps the
    # high-frequency Laplacian energy of p small relative to its variance.
    @test c.res.checkerboard < 0.5

    # Sanity on the recirculation: the primary vortex drives u<0 in the lower
    # half of the centreline and the lid drives u>0 near the top.
    @test minimum(c.res.u) < -0.1        # return flow exists
    @test c.res.u[64, end] > 0.5         # cell below the lid is dragged along
end
