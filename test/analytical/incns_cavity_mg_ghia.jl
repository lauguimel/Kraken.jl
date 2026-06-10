# Analytical validation: backend-parametric SIMPLE 2D lid-driven cavity using the
# matrix-free multigrid for BOTH the pressure-correction and the viscous/momentum
# solves, against Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, 387-411.
#
# Re=100 (128²) and Re=1000 (256²). Unit square [0,1]²; top wall u=U_lid=1, v=0;
# other three walls no-slip. Re = U_lid·L/ν, ρ=1 ⇒ μ=ν. Runs on the CPU backend
# (KernelAbstractions CPU + Array); the SAME source runs on CUDA by passing
# backend_ka=CUDABackend(), atype=CuArray{Float64}.
#
# Asserts the converged centreline profiles (vertical u(y) at x=0.5, horizontal
# v(x) at y=0.5, both normalised by U_lid) match Ghia to <= 5% local at Re=100
# (parity with the assembled solver) and <= 10% local at Re=1000 (harder; the
# first-order upwind advection damps the steep-gradient peaks). Also checks
# convergence and a checkerboard-free pressure field.

using Test
using KernelAbstractions

if !isdefined(@__MODULE__, :solve_incns_cavity_mg)
    include(joinpath(@__DIR__, "..", "..", "src", "methods", "inc_ns", "cavity_mg.jl"))
end

# Ghia (1982) Re=100.
const GHIA_MG_Y = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                   0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
                   0.9531, 0.9609, 0.9688, 0.9766, 1.0000]
const GHIA_MG_X = [0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
                   0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
                   0.9453, 0.9531, 0.9609, 0.9688, 1.0000]
const GHIA_MG_RE100_U = [0.0000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                         -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                         0.68717, 0.73722, 0.78871, 0.84123, 1.00000]
const GHIA_MG_RE100_V = [0.0000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
                         0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
                         -0.10313, -0.08864, -0.07391, -0.05906, 0.00000]
# Ghia (1982) Re=1000.
const GHIA_MG_RE1000_U = [0.0000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
                          -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304,
                          0.46604, 0.51117, 0.57492, 0.65928, 1.00000]
const GHIA_MG_RE1000_V = [0.0000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095,
                          0.33075, 0.32235, 0.02526, -0.31966, -0.42665, -0.51550,
                          -0.39188, -0.33714, -0.27669, -0.21388, 0.00000]

const INCNS_CAVITY_MG_RESULTS = Dict{Symbol,Any}()

function _cavity_mg_interp(vals, centers, q, wall_lo, wall_hi)
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

function incns_cavity_mg_case(; N::Integer, Re::Real, ghia_u, ghia_v,
                              relax, tol::Real, vel_tol::Real, maxiter::Integer,
                              U_lid::Real=1.0,
                              backend_ka=KernelAbstractions.CPU(),
                              atype::Type{<:AbstractArray}=Array{Float64})
    res = solve_incns_cavity_mg(; nx=N, ny=N, U_lid, Re, relax, tol, vel_tol,
                                maxiter, backend_ka, atype,
                                mg_tol=1e-3, mg_maxcycles=25,
                                mom_mg_tol=1e-3, mom_mg_maxcycles=25)

    ic = N ÷ 2
    u_col = (res.u[ic, :] .+ res.u[ic + 1, :]) ./ 2 ./ U_lid
    jc = N ÷ 2
    v_row = (res.v[:, jc] .+ res.v[:, jc + 1]) ./ 2 ./ U_lid
    u_at(q) = _cavity_mg_interp(u_col, res.ycenters, q, 0.0, 1.0)
    v_at(q) = _cavity_mg_interp(v_row, res.xcenters, q, 0.0, 0.0)

    u_num = [u_at(y) for y in GHIA_MG_Y]
    v_num = [v_at(x) for x in GHIA_MG_X]
    err_u = abs.(u_num .- ghia_u)
    err_v = abs.(v_num .- ghia_v)
    max_err_u = maximum(err_u)
    max_err_v = maximum(err_v)
    return (; res, u_num, v_num, err_u, err_v, max_err_u, max_err_v,
            max_err=max(max_err_u, max_err_v))
end

@testset "IncNS backend-parametric MG lid-driven cavity vs Ghia (1982)" begin
    @testset "Re=100 parity (128²) <= 5% local" begin
        c = incns_cavity_mg_case(; N=128, Re=100.0,
                                 ghia_u=GHIA_MG_RE100_U, ghia_v=GHIA_MG_RE100_V,
                                 relax=(u=0.7, p=0.3), tol=1e-7, vel_tol=1e-6,
                                 maxiter=8000)
        INCNS_CAVITY_MG_RESULTS[:re100] = c
        @test c.res.converged
        rh = c.res.residual_history
        @test rh[end] / rh[1] < 1e-4
        @test c.max_err_u <= 0.05
        @test c.max_err_v <= 0.05
        @test c.res.checkerboard < 0.5
        @test minimum(c.res.u) < -0.1
        @test c.res.u[64, end] > 0.5
    end

    @testset "Re=1000 (256²) <= 10% local" begin
        c = incns_cavity_mg_case(; N=256, Re=1000.0,
                                 ghia_u=GHIA_MG_RE1000_U, ghia_v=GHIA_MG_RE1000_V,
                                 relax=(u=0.5, p=0.2), tol=1e-6, vel_tol=1e-6,
                                 maxiter=40000)
        INCNS_CAVITY_MG_RESULTS[:re1000] = c
        @test c.res.converged
        rh = c.res.residual_history
        @test rh[end] / rh[1] < 1e-4
        @test c.max_err_u <= 0.10
        @test c.max_err_v <= 0.10
        @test c.res.checkerboard < 0.5
        # Re=1000 primary vortex: stronger return flow than Re=100.
        @test minimum(c.res.u) < -0.3
    end
end
