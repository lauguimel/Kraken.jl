# Analytical validation: unsteady projection (fractional-step) solver on the
# 2D Taylor-Green vortex, [0,2pi]^2 fully periodic.
#
# Exact solution (rho = 1):
#   u(x,y,t) =  sin(x) cos(y) exp(-2 nu t)
#   v(x,y,t) = -cos(x) sin(y) exp(-2 nu t)
#   p(x,y,t) =  1/4 (cos 2x + cos 2y) exp(-4 nu t)
# Kinetic energy decays as E(t) = E(0) exp(-4 nu t).
#
# Asserts:
#   (a) spatial order ~2 under grid refinement at fixed small dt (error vs the
#       analytic field at t = T),
#   (b) temporal order under dt refinement at a fixed grid, measured by
#       self-convergence against a small-dt reference on the SAME grid (this
#       isolates the time error from the fixed O(h^2) spatial error);
#       Crank-Nicolson + AB2 + incremental projection with the :increment
#       momentum interpolation => order ~2 (>= 1.8). The :be fallback is also
#       measured (expect ~1, >= 0.9). NOTE: the :full d=dt Rhie-Chow
#       interpolation degrades this measurement to ~1.1-1.6 via its O(dt*h^2)
#       advecting-flux deviation — documented in projection.jl, not asserted,
#   (c) post-projection face divergence at solver tolerance (direct solve),
#   (d) kinetic-energy decay matches exp(-4 nu t),
#   plus the factorize-once receipts (2 factorizations for 3*nsteps solves).

using Test
using KernelAbstractions   # CPU() — no longer inherited once the include below auto-skips

if !isdefined(@__MODULE__, :solve_incns_projection)
    include(joinpath(@__DIR__, "..", "..", "src", "methods", "inc_ns", "projection.jl"))
end

const INCNS_TG_RESULTS = Dict{Symbol,Any}()

tg_u(x, y, t, nu) = sin(x) * cos(y) * exp(-2.0 * nu * t)
tg_v(x, y, t, nu) = -cos(x) * sin(y) * exp(-2.0 * nu * t)
tg_p(x, y, t, nu) = 0.25 * (cos(2x) + cos(2y)) * exp(-4.0 * nu * t)

# Relative L2 error of (u, v) vs the analytic field at time t.
function _tg_l2_error(res, t, nu)
    err2 = 0.0
    ref2 = 0.0
    for j in 1:res.ny, i in 1:res.nx
        x = res.xcenters[i]; y = res.ycenters[j]
        ua = tg_u(x, y, t, nu); va = tg_v(x, y, t, nu)
        err2 += (res.u[i, j] - ua)^2 + (res.v[i, j] - va)^2
        ref2 += ua^2 + va^2
    end
    return sqrt(err2 / ref2)
end

function tg_case(; n::Integer, dt::Real, T::Real, nu::Real = 0.1,
                 scheme::Symbol = :cn, backend = CPU())
    L = 2pi
    nsteps = round(Int, T / dt)
    @assert isapprox(nsteps * dt, T; rtol = 1e-12) "T must be a multiple of dt"
    res = solve_incns_projection(; nx = n, ny = n, Lx = L, Ly = L, nu, dt, nsteps,
                                 bc_x = :periodic, bc_y = :periodic,
                                 u0 = (x, y) -> tg_u(x, y, 0.0, nu),
                                 v0 = (x, y) -> tg_v(x, y, 0.0, nu),
                                 p0 = (x, y) -> tg_p(x, y, 0.0, nu),
                                 scheme, backend)
    return (; res, t = res.t_final, l2_rel = _tg_l2_error(res, res.t_final, nu), nu)
end

@testset "IncNS unsteady projection Taylor-Green vortex" begin
    backend = CPU()
    nu = 0.1

    # ---- (a) spatial order: fixed small dt, grid refinement ----
    dt_s = 2e-3
    T_s = 0.2
    grids = (16, 32, 64)
    cases = [tg_case(; n, dt = dt_s, T = T_s, nu, backend) for n in grids]
    errs = [c.l2_rel for c in cases]
    sp_orders = [log2(errs[k] / errs[k + 1]) for k in 1:length(errs) - 1]
    INCNS_TG_RESULTS[:spatial] = (; grids, errs, orders = sp_orders)

    @test all(diff(errs) .< 0.0)              # monotone decrease
    @test all(o -> o >= 1.7, sp_orders)
    @test 1.8 <= sp_orders[end] <= 2.3        # asymptotic order ~2
    @test errs[end] < 1e-4                    # finest grid resolves the vortex

    # ---- (b) temporal order: fixed grid, dt refinement, self-convergence ----
    n_t = 32
    T_t = 0.32
    dts = (0.02, 0.01, 0.005)
    dt_ref = 0.000625                          # 8x below the smallest dt
    ref = tg_case(; n = n_t, dt = dt_ref, T = T_t, nu, backend)
    runs = [tg_case(; n = n_t, dt, T = T_t, nu, backend) for dt in dts]
    unorm = sqrt(sum(abs2, ref.res.u) + sum(abs2, ref.res.v))
    terrs = [sqrt(sum(abs2, r.res.u .- ref.res.u) +
                  sum(abs2, r.res.v .- ref.res.v)) / unorm for r in runs]
    t_orders = [log2(terrs[k] / terrs[k + 1]) for k in 1:length(terrs) - 1]
    INCNS_TG_RESULTS[:temporal] = (; dts, errs = terrs, orders = t_orders,
                                   scheme = :cn)

    @test all(diff(terrs) .< 0.0)
    @test all(o -> 1.8 <= o <= 2.4, t_orders) # CN + AB2 => 2nd order in time

    # Backward-Euler fallback: same protocol, expect ~1st order (>= 0.9).
    ref_be = tg_case(; n = n_t, dt = dt_ref, T = T_t, nu, scheme = :be, backend)
    runs_be = [tg_case(; n = n_t, dt, T = T_t, nu, scheme = :be, backend)
               for dt in dts]
    unorm_be = sqrt(sum(abs2, ref_be.res.u) + sum(abs2, ref_be.res.v))
    terrs_be = [sqrt(sum(abs2, r.res.u .- ref_be.res.u) +
                     sum(abs2, r.res.v .- ref_be.res.v)) / unorm_be
                for r in runs_be]
    t_orders_be = [log2(terrs_be[k] / terrs_be[k + 1])
                   for k in 1:length(terrs_be) - 1]
    INCNS_TG_RESULTS[:temporal_be] = (; dts, errs = terrs_be,
                                      orders = t_orders_be, scheme = :be)

    @test all(diff(terrs_be) .< 0.0)
    @test all(o -> o >= 0.9, t_orders_be)     # BE diffusion => 1st order

    # ---- (c) divergence-free to solver tolerance (direct Poisson solve) ----
    fine = cases[end].res
    INCNS_TG_RESULTS[:div] = (; max_div_inf = fine.max_div_inf,
                              div_inf_final = fine.div_inf_final)
    @test fine.max_div_inf < 1e-11
    @test fine.div_inf_final < 1e-11

    # ---- (d) kinetic-energy decay E(T)/E(0) = exp(-4 nu T) ----
    ke_ratio = fine.ke_history[end] / fine.ke_history[1]
    ke_exact = exp(-4.0 * nu * T_s)
    ke_err = abs(ke_ratio - ke_exact) / ke_exact
    INCNS_TG_RESULTS[:ke] = (; ke_ratio, ke_exact, ke_err)
    @test ke_err < 1e-3

    # ---- factorize-once receipts: 2 factorizations, 3 solves per step ----
    @test fine.nfactorizations == 2
    @test fine.nlinsolves == 3 * fine.nsteps
end
