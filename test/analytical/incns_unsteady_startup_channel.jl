# Analytical validation: unsteady projection (fractional-step) solver on the
# impulsively-started plane Poiseuille flow (startup channel).
#
# Body-force-driven channel: height H in y (no-slip walls y = 0, H), periodic
# in x, constant streamwise body force G = -dP/dx, u(y, 0) = 0. Transient
# analytic solution (Fourier sine series, rho = 1, nu = mu):
#   u(y,t) = (G/(2 nu)) y (H - y)
#            - (4 G H^2 / (nu pi^3)) sum_{n odd} sin(n pi y/H)/n^3
#                                               * exp(-(n pi/H)^2 nu t)
# (same steady parabola as the analytic helper in incns_poiseuille.jl, plus
# the decaying transient; inlined here because that file asserts on the steady
# SIMPLE solver when included).
#
# Asserts the x-averaged transient profile against the series at several times
# t < t_steady, convergence to the steady parabola at large t, zero cross-flow
# and post-projection divergence at solver tolerance.

using Test
using KernelAbstractions   # CPU() — no longer inherited once the include below auto-skips

if !isdefined(@__MODULE__, :solve_incns_projection)
    include(joinpath(@__DIR__, "..", "..", "src", "methods", "inc_ns", "projection.jl"))
end

const INCNS_STARTUP_RESULTS = Dict{Symbol,Any}()

# Transient analytic profile. Terms decay like 1/n^3 * exp(-(n pi/H)^2 nu t);
# nmax = 399 odd terms is far beyond machine precision for the times used here.
function startup_channel_u(y::Real, t::Real; G::Real, nu::Real, H::Real,
                           nmax::Integer = 399)
    us = (G / (2.0 * nu)) * y * (H - y)
    t <= 0.0 && return 0.0
    s = 0.0
    n = 1
    while n <= nmax
        s += sin(n * pi * y / H) / n^3 * exp(-(n * pi / H)^2 * nu * t)
        n += 2
    end
    return us - (4.0 * G * H^2 / (nu * pi^3)) * s
end

# Relative L2 error of the x-averaged numerical profile vs an analytic profile.
function _startup_profile_err(uprof, yc, t; G, nu, H)
    uan = [startup_channel_u(y, t; G, nu, H) for y in yc]
    return sqrt(sum(abs2, uprof .- uan) / length(yc)) /
           sqrt(sum(abs2, uan) / length(yc))
end

function startup_channel_case(; nx::Integer = 8, ny::Integer = 64,
                              H::Real = 1.0, Lx::Real = 1.0,
                              nu::Real = 0.1, G::Real = 1.0,
                              dt::Real = 2.5e-3, t_final::Real = 6.0,
                              snap_times = (0.1, 0.3, 1.0),
                              scheme::Symbol = :cn, backend = CPU())
    nsteps = round(Int, t_final / dt)
    # Snapshot steps; snap times must land on a step boundary.
    snap_steps = Dict(round(Int, t / dt) => t for t in snap_times)
    for (s, t) in snap_steps
        @assert isapprox(s * dt, t; rtol = 1e-12) "snap time $t is not a multiple of dt"
    end
    snaps = Dict{Float64,Vector{Float64}}()   # t => x-averaged u profile
    cb = (step, _t, u, _v, _p) -> begin
        if haskey(snap_steps, step)
            snaps[snap_steps[step]] = vec(sum(u; dims = 1)) ./ size(u, 1)
        end
        nothing
    end

    res = solve_incns_projection(; nx, ny, Lx, Ly = H, nu, dt, nsteps,
                                 bc_x = :periodic, bc_y = :wall,
                                 fx = G, scheme, backend, callback = cb)

    uprof_final = vec(sum(res.u; dims = 1)) ./ nx
    errs_transient = Dict(t => _startup_profile_err(prof, res.ycenters, t;
                                                    G, nu, H)
                          for (t, prof) in snaps)

    # Large-t checks vs the STEADY parabola (transient < 0.3% of steady at
    # t = 6 with nu = 0.1, H = 1: exp(-pi^2*0.6) ~ 2.7e-3 on the n=1 mode).
    uan_steady = [(G / (2.0 * nu)) * y * (H - y) for y in res.ycenters]
    err_steady = sqrt(sum(abs2, uprof_final .- uan_steady) / ny) /
                 sqrt(sum(abs2, uan_steady) / ny)
    err_final_transient = _startup_profile_err(uprof_final, res.ycenters,
                                               res.t_final; G, nu, H)

    return (; res, snaps, errs_transient, err_steady, err_final_transient,
            uprof_final, uan_steady, maxv = maximum(abs, res.v))
end

@testset "IncNS unsteady projection startup channel (plane Poiseuille)" begin
    backend = CPU()
    G = 1.0; nu = 0.1; H = 1.0

    c = startup_channel_case(; G, nu, H, backend)
    INCNS_STARTUP_RESULTS[:case] = c

    # Transient profiles match the Fourier-series solution (t < t_steady ~ 5;
    # at t = 0.1 the flow is at ~9% of the steady centreline velocity).
    # Measured: 0.11% / 0.05% / 0.03% relative L2 at t = 0.1 / 0.3 / 1.0.
    for t in (0.1, 0.3, 1.0)
        @test haskey(c.errs_transient, t)
        @test c.errs_transient[t] < 0.005   # <= 0.5% relative L2 per snapshot
    end

    # Converged to the steady analytic parabola at t = 6 (>= 5 viscous time
    # constants H^2/(pi^2 nu) ~ 1). Measured: 0.24% (wall-cell O(h^2) bias).
    @test c.err_steady < 0.005
    @test c.err_final_transient < 0.005

    # Centreline magnitude sanity: u_max -> G H^2/(8 nu).
    umax_an = G * H^2 / (8.0 * nu)
    @test abs(maximum(c.uprof_final) - umax_an) / umax_an < 0.01

    # No cross-flow, post-projection divergence at solver tolerance.
    @test c.maxv < 1e-10
    @test c.res.max_div_inf < 1e-11

    # Body-force driving needs no pressure: p stays at roundoff (also a
    # checkerboard guard — wall rows inject nothing spurious into p).
    @test maximum(abs, c.res.p) < 1e-10

    # Factorize-once receipts.
    @test c.res.nfactorizations == 2
    @test c.res.nlinsolves == 3 * c.res.nsteps
end
