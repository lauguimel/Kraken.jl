# J2 feasibility gate (plan §4.1 J2, brief j2_brief.md §4): steady GNF adjoint
# on the GPU for the periodic obstacle channel.
#
#   julia --project=. -t 4 benchmarks/inverse_rheology/j2_gate.jl out=<dir> [key=value ...]
#
# Keys (lists are '+'-separated; a case is run:m:nu0, run N = Newtonian):
#   gate=4:64:0.34          case of the pass criterion
#   cases=N:32:0.17+1:32:0.17+4:32:0.17+N:64:0.34+1:64:0.34   other measured cases
#   timing=4:72:0.34        timing-only case (value + gradient at tol 1e-6)
#   restart_case=1:32:0.17  restart study: 240 / 1000 / none (memory-limited)
#   restart=1000 orth=mgs2  GMRES of the measurement series (orth also cgs2)
#   alt_orth=cgs2           second orthogonalisation measured on the gate case ("" = none)
#   max_solve=900           cap per GMRES solve, seconds
#   max_solve_base=         cap of the base solve of each case (default max_solve)
#   budget=6300             wall budget of the script, seconds (items beyond it are skipped)
#   fallbacks=auto          auto = only if the gate fails; force = always; none
#   deflate=32 series_N=4000  parameters of fallbacks (ii) and (iii)
#   fb_extra=v+vi           Boss-approved fallbacks (2026-10-20), measured before (ii)-(iv):
#                           (v) fixed-point adjoint iteration, two-step stop, mean of the
#                           last two iterates (fp_check=1000); (vi) GMRES on the K-step
#                           system (I - M^K) l = sum_{j<K} M^j b' (kstep_K=1001 odd,
#                           kstep_restart=50, orth cgs2), with T4c-K / T4d-K on the GPU
#   case_solver=gmres       solver of the base solve of every case (gate, cases, timing):
#                           gmres = GMRES(restart, orth) to 1e-8 with snapshots; kstep =
#                           fallback (vi) (kstep_K, kstep_restart, cgs2), one separate solve
#                           per tolerance (1e-4, 1e-6, 1e-8; timing case 1e-6), each converged
#                           on the explicit residual, plus T4c-K / T4d-K per case
#   warmup=1                JIT warm-up on warmup_case=4:8:0.1 before the first case
#   norestart=3500          one GMRES(cgs2) without restart at this basis size, only if
#                           the GPU has >= 70 GiB (else skipped)
#   only=                   comma-free subset of items: gate+cases+restart+timing
#   restart_none_max=20000  cap of the "no restart" GMRES (also limited by memory)
#   backend=cuda            cpu = smoke run of the script on the KA CPU backend
#                           (small cases only; no GPU measurement)
#   t4d_only=0              1 = per case (gate, then cases, as selected by only=): forward,
#                           linearisation, T4c, T4d (with its Richardson extrapolation)
#                           and power iteration only; no adjoint solve
#
# Every case: forward from rest to dist_est ≤ 1e-10 (two-step estimator of
# run_gnf_periodic_steady, fused path, check_every 1000), observation = J1
# observable (reference positions on the frozen mask, bilinear interpolation of
# the node velocity, u* = u_lat/u_ref; J = ½ Σ (u*_K − u*_Z)², i.e. half the
# numerator of the J1 RRMSE), linearisation (CPU blocks) + copy, T4c/T4d on
# the GPU, power iteration (50 it.), one adjoint solve to tol 1e-8 with
# snapshots at 1e-4 and 1e-6 (iterations and time to each tolerance, gradient
# at each), central finite differences of J on ln k (Newtonian case: ln η∞,
# since ν does not depend on k there) with forwards warm-started from the
# base state to dist ≤ 1e-10. The gate case adds finite differences on the
# four parameters, harmonic Ritz values, the alternative orthogonalisation and
# the fallbacks (i)-(iv) of the brief, applied cumulatively.
#
# Outputs in <out>: manifest.toml, results.toml (rewritten after every item),
# convergence_<item>.csv (relative residual per iteration), summary.txt.

include(joinpath(@__DIR__, "obstacle_channel_case.jl"))
using CUDA
using LinearAlgebra
using Random

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const T_START = time()

argd = Dict{String,String}()
for a in ARGS
    k, v = split(a, "="; limit = 2)
    argd[k] = v
end
getarg(k, d) = get(argd, k, d)

const OUT = getarg("out", joinpath(REPO, "benchmarks", "inverse_rheology", "out",
                                   "gate_" * readchomp(`date +%Y%m%d_%H%M%S`)))
mkpath(OUT)
const BUDGET = parse(Float64, getarg("budget", "6300"))
const MAX_SOLVE = parse(Float64, getarg("max_solve", "900"))
const MAX_SOLVE_BASE = parse(Float64, getarg("max_solve_base", string(MAX_SOLVE)))
const RESTART = parse(Int, getarg("restart", "1000"))
const ORTH = Symbol(getarg("orth", "mgs2"))
const ALT_ORTH = getarg("alt_orth", "cgs2")
const DEFLATE = parse(Int, getarg("deflate", "32"))
const SERIES_N = parse(Int, getarg("series_N", "4000"))
const FALLBACKS = getarg("fallbacks", "auto")
const ONLY = split(getarg("only", "gate+cases+restart+timing"), '+')
const PASS_S = 120.0
const PASS_REL = 0.01
const BK = getarg("backend", "cuda") == "cpu" ? KernelAbstractions.CPU() : CUDABackend()
const ON_GPU = BK isa CUDABackend
const RESTART_NONE_MAX = parse(Int, getarg("restart_none_max", "20000"))
const FB_EXTRA = filter(!isempty, split(getarg("fb_extra", ""), '+'))
const FP_CHECK = parse(Int, getarg("fp_check", "1000"))
const KSTEP_K = parse(Int, getarg("kstep_K", "1001"))
const KSTEP_RESTART = parse(Int, getarg("kstep_restart", "50"))
const NORESTART = parse(Int, getarg("norestart", "0"))
const CASE_SOLVER = Symbol(getarg("case_solver", "gmres"))
const T4D_ONLY = getarg("t4d_only", "0") == "1"
CASE_SOLVER in (:gmres, :kstep) || error("case_solver must be gmres or kstep")
dev(x) = ON_GPU ? CuArray(x) : copy(x)
sync() = KernelAbstractions.synchronize(BK)
reclaim() = (GC.gc(); ON_GPU && CUDA.reclaim(); nothing)
free_mem() = ON_GPU ? CUDA.free_memory() : Sys.free_memory()
devname() = ON_GPU ? CUDA.name(CUDA.device()) : "CPU (KernelAbstractions, smoke)"

left() = BUDGET - (time() - T_START)
tkey(x) = @sprintf("%.0e", x)          # dictionary key of a tolerance or step
solve_cap(cap = MAX_SOLVE) = max(30.0, min(cap, left() - 120.0))
tomlsafe(v::AbstractFloat) = isfinite(v) ? Float64(v) : string(v)
tomlsafe(v::Bool) = v
tomlsafe(v::Integer) = Int(v)
tomlsafe(v::AbstractVector) = [tomlsafe(x) for x in v]
tomlsafe(v::AbstractDict) = Dict(string(k) => tomlsafe(x) for (k, x) in v)
tomlsafe(v::Symbol) = string(v)
tomlsafe(v::Complex) = [tomlsafe(real(v)), tomlsafe(imag(v))]
tomlsafe(v) = v
gpu_mem_gib() = ON_GPU ? (try (CUDA.total_memory() - CUDA.free_memory()) / 2^30 catch; NaN end) : NaN
gpu_pool_gib() = ON_GPU ? (try CUDA.used_memory() / 2^30 catch; NaN end) : NaN

const RESULTS = Dict{String,Any}()
const SUMMARY = String[]
function save_results()
    open(joinpath(OUT, "results.toml"), "w") do io
        TOML.print(io, tomlsafe(RESULTS))
    end
end
function say(s)
    println(s); flush(stdout)
    push!(SUMMARY, s)
    open(joinpath(OUT, "summary.txt"), "w") do io
        foreach(l -> println(io, l), SUMMARY)
    end
end

# ---------------------------------------------------------------- cases ---
function parse_case(s)
    r, m, nu = split(s, ':')
    return (run = r == "N" ? 0 : parse(Int, r), m = parse(Int, m), nu0 = parse(Float64, nu))
end

# Bilinear weights of `interp_node` (obstacle_channel_case.jl): up to 4 (linear node, weight).
function interp_weights(m, Nx, Ny, x, y)
    ξ = x * m + 0.5; η = y * m + 0.5
    i0 = floor(Int, ξ); tx = ξ - i0
    ia = mod1(i0, Nx); ib = mod1(i0 + 1, Nx)
    L(i, j) = i + Nx * (j - 1)
    if η <= 1
        w = clamp((η - 0.5) / 0.5, 0.0, 1.0)
        return (L(ia, 1), L(ib, 1), 0, 0), ((1 - tx) * w, tx * w, 0.0, 0.0)
    elseif η >= Ny
        w = clamp((Ny + 0.5 - η) / 0.5, 0.0, 1.0)
        return (L(ia, Ny), L(ib, Ny), 0, 0), ((1 - tx) * w, tx * w, 0.0, 0.0)
    else
        j0 = floor(Int, η); ty = η - j0
        return (L(ia, j0), L(ia, j0 + 1), L(ib, j0), L(ib, j0 + 1)),
               ((1 - tx) * (1 - ty), (1 - tx) * ty, tx * (1 - ty), tx * ty)
    end
end

function case_obs(U::CaseUnits, Nx, Ny, run)
    zx, zy = load_reference_fields(run)
    mask = read_npy(REF_MASK_FROZEN) .!= 0
    idx = findall(mask)
    M = 2 * length(idx)
    node = zeros(Int, 4, M); w = zeros(4, M); comp = zeros(Int, M); tgt = zeros(M)
    k = 0
    for c in idx
        i, j = Tuple(c)
        for (cmp, pos, z) in ((1, ux_pos(i, j), zx), (2, uy_pos(i, j), zy))
            k += 1
            nd, ww = interp_weights(U.m, Nx, Ny, pos...)
            node[:, k] .= nd; w[:, k] .= ww; comp[k] = cmp; tgt[k] = z[i, j]
        end
    end
    return GNFVelocityObs(node, w, comp, tgt, 1 / u_ref_lat(U)), zx, zy, mask
end

function make_case(spec)
    run, m, nu0 = spec.run, spec.m, spec.nu0
    U = CaseUnits(m, nu0)
    Nx, Ny = grid_size(U)
    solid = case_solid_mask(U)
    G = g_lat(U)
    if run == 0
        θ0 = [0.0, log(5.0), 0.7, 2.0]     # η∞ = η0: ν ≡ ν0 for any (k, n, a)
        obs_run = 1
    else
        p = CASE_RUNS[run]
        θ0 = [log(p.eta_inf), log(p.lam), p.n, 2.0]
        obs_run = run
    end
    closure_of = θ -> CarreauYasuda(nu0, exp(θ[1]) * nu0, exp(θ[2]) * m^2 / nu0, θ[4], θ[3])
    if run != 0
        c1 = closure_of(θ0); c2 = case_closure(U, run)
        all(isapprox(getfield(c1, f), getfield(c2, f); rtol = 1e-14)
            for f in fieldnames(typeof(c1)) if getfield(c1, f) isa Number) ||
            error("closure_of(θ0) differs from case_closure for run $run")
    end
    obs, zx, zy, mask = case_obs(U, Nx, Ny, obs_run)
    tag = (run == 0 ? "newt" : "run$run") * "_m$m"
    dfp = run == 0 ? 1 : 2
    return (; tag, run, m, nu0, U, Nx, Ny, solid, G, θ0, closure_of, obs, obs_run, zx, zy, mask, dfp)
end

const PNAMES = ("ln_eta_inf", "ln_k", "n", "a")

function forward(c, θ; f_init = nothing, tau_init = nothing, max_wall = 900.0)
    t = @elapsed r = run_gnf_periodic_steady(; is_solid = c.solid, closure = c.closure_of(θ),
                                             Fx = c.G, Fy = 0.0, backend = BK, T = Float64,
                                             tol = 1e-10, stop = :distance, max_steps = 3_000_000,
                                             check_every = 1000, fused = true, max_wall,
                                             f_init, tau_init)
    return r, t
end

J_of(c, r) = gnf_obs_objective(c.obs, r.ux, r.uy)

# Central finite differences of J on parameter p for relative steps hs
# (δ = h max(|θ_p|, 1)), forwards warm-started from the base state.
function fd_param(c, base, p, hs)
    out = Dict{String,Any}()
    for h in hs
        left() < 300 && break
        δ = h * max(abs(c.θ0[p]), 1.0)
        θp = copy(c.θ0); θp[p] += δ
        θm = copy(c.θ0); θm[p] -= δ
        rp, tp = forward(c, θp; f_init = base.f, tau_init = base.tau)
        rm, tm = forward(c, θm; f_init = base.f, tau_init = base.tau)
        df = (J_of(c, rp) - J_of(c, rm)) / (2δ)
        out[tkey(h)] = Dict("h" => h, "delta" => δ, "df" => df, "dist_p" => rp.dist_est,
                              "dist_m" => rm.dist_est, "steps" => rp.steps + rm.steps,
                              "wall_s" => tp + tm, "converged" => rp.converged && rm.converged)
    end
    return out
end

# Best-h relative error of gradient component g against the DF sweep d.
function df_err(g, d)
    isempty(d) && return (NaN, NaN)
    errs = [(abs(g - v["df"]) / abs(v["df"]), v["h"]) for v in values(d)]
    e, h = minimum(errs)
    return e, h
end

function write_conv(item, info)
    open(joinpath(OUT, "convergence_$(item).csv"), "w") do io
        println(io, "iter,relres_implicit,t_s")
        for (k, rr) in enumerate(info.relres_iter)
            (k <= 100 || k % 10 == 0 || k == length(info.relres_iter)) &&
                println(io, k, ",", rr, ",", info.t_iter[k])
        end
    end
end

function write_fp_conv(item, info)
    open(joinpath(OUT, "convergence_$(item).csv"), "w") do io
        println(io, "iter,r_1step,r_2step,dist_est,t_s")
        for (k, h) in enumerate(info.history)
            println(io, h[1], ",", h[2], ",", h[5], ",", info.relres_iter[k], ",", info.t_iter[k])
        end
    end
end

snap_of(info, tol) = (i = findfirst(s -> s.tol == tol, info.snaps); i === nothing ? nothing : info.snaps[i])

# One adjoint solve to tol 1e-8 with snapshots; gradients at 1e-4, 1e-6, 1e-8.
function solve_record(c, lin, op, b; restart = RESTART, orth = ORTH, deflate = 0, λ0 = nothing,
                      tol = 1e-8, snaps = (1e-4, 1e-6), ritz = false, item = "", maxiter = 10^8,
                      solver = :gmres, K = KSTEP_K, cap = MAX_SOLVE)
    reclaim()
    m0 = gpu_mem_gib()
    λ, η, info = if solver === :gmres
        gnf_adjoint_solve(op, b; tol, restart, maxiter, orth, deflate, λ0,
                          max_wall = solve_cap(cap), ritz, snapshots = snaps)
    elseif solver === :fixed_point
        gnf_adjoint_fixed_point(op, b; tol, check_every = FP_CHECK, λ0, max_wall = solve_cap(cap),
                                snapshots = snaps)
    else
        gnf_adjoint_kstep_solve(op, b; K, tol, restart, orth, λ0, max_wall = solve_cap(cap),
                                snapshots = snaps)
    end
    sync()
    mem = gpu_mem_gib(); pool = gpu_pool_gib()
    rec = Dict{String,Any}("restart" => restart, "orth" => string(orth), "deflate" => deflate,
                           "converged" => info.converged, "iters" => info.iters, "cycles" => info.cycles,
                           "relres" => info.relres, "t_total" => info.t_total, "t_op" => info.t_op,
                           "t_orth" => info.t_orth, "t_diag" => info.t_diag, "n_apply" => info.n_apply,
                           "t_per_apply" => info.t_op / max(info.n_apply, 1),
                           "relres_cycles" => info.relres_cycles, "eta" => η,
                           "gpu_mem_gib_before" => m0, "gpu_mem_gib_after" => mem,
                           "gpu_pool_gib_after" => pool, "tol" => tol, "solver" => string(solver))
    for key in (:n_vjp, :t_rhs, :relres_kstep, :K, :rounds, :relres_true_rounds, :contraction, :dist_est, :check_every)
        hasproperty(info, key) && (rec[string(key)] = getproperty(info, key))
    end
    hasproperty(info, :n_vjp) && (rec["t_per_vjp"] = (info.t_op + info.t_rhs) / max(info.n_vjp, 1))
    grads = Dict{String,Any}()
    for s in info.snaps
        tg = @elapsed g, _ = gnf_parameter_gradient(lin, s.x[1:op.n], c.θ0, c.closure_of)
        grads[tkey(s.tol)] = Dict("iters" => s.iters, "t" => s.t, "grad" => g, "t_grad" => tg)
    end
    if info.converged
        tg = @elapsed g, _ = gnf_parameter_gradient(lin, λ, c.θ0, c.closure_of)
        grads[tkey(tol)] = Dict("iters" => info.iters, "t" => info.t_total, "grad" => g, "t_grad" => tg)
    end
    rec["grads"] = grads
    if ritz && !isempty(info.ritz)
        rv = sort(info.ritz; by = abs)
        rec["ritz_smallest_abs"] = [abs(z) for z in rv[1:min(10, end)]]
        rec["ritz_smallest"] = [[real(z), imag(z)] for z in rv[1:min(10, end)]]
    end
    isempty(item) || (solver === :fixed_point ? write_fp_conv(item, info) : write_conv(item, info))
    return rec, λ, info
end

# Fallback (vi) to each tolerance by a separate solve from zero, each converged
# on the explicit residual ‖b' − (I − M)λ‖/‖b'‖ (the snapshots of a single
# K-step solve are taken on the K-step residual, 10–60× below the explicit one).
# The merged "grads" table has the layout of a GMRES record with snapshots.
function solve_kstep_tols(c, lin, op, b, tols; item = "", cap = MAX_SOLVE)
    rec = Dict{String,Any}("solver" => "kstep", "K" => KSTEP_K, "restart" => KSTEP_RESTART,
                           "orth" => "cgs2", "per_tol" => Dict{String,Any}())
    grads = Dict{String,Any}(); infos = Dict{Float64,Any}(); λ = nothing
    for tol in tols
        left() < 200 && break
        sr, λ, info = solve_record(c, lin, op, b; solver = :kstep, K = KSTEP_K, restart = KSTEP_RESTART,
                                   orth = :cgs2, tol, snaps = (), cap,
                                   item = isempty(item) ? "" : item * "_kstep_" * tkey(tol))
        rec["per_tol"][tkey(tol)] = sr
        haskey(sr["grads"], tkey(tol)) && (grads[tkey(tol)] = sr["grads"][tkey(tol)])
        infos[tol] = info
        say(@sprintf("  (vi) K-step GMRES(%d, cgs2), K = %d, tol %.0e: %s, %d it (%d VJP), %.1f s (rhs %.1f, op %.1f, orth %.1f), relres %.1e (K-step %.1e, %d refine rounds), %.3f ms/VJP, GPU mem %.1f GiB",
                     KSTEP_RESTART, KSTEP_K, tol, info.converged ? "converged" : "NOT converged",
                     info.iters, info.n_vjp, info.t_total, info.t_rhs, info.t_op, info.t_orth, info.relres,
                     info.relres_kstep, info.rounds, 1e3 * sr["t_per_vjp"], sr["gpu_mem_gib_after"]))
        if haskey(grads, tkey(tol))
            say(@sprintf("    tol %.0e: grad = [%s]", tol,
                         join([@sprintf("%.6e", x) for x in grads[tkey(tol)]["grad"]], ", ")))
        end
        save_results()
    end
    rec["grads"] = grads
    return rec, λ, infos
end

# Warm forward (deliverable cost, not a criterion): from the converged state,
# ln k perturbed by 1 % (delta = 0.01 max(|ln k|, 1), as the DF).
function warm_forward!(c, rec, r, t_fwd)
    θw = copy(c.θ0); θw[2] += 0.01 * max(abs(c.θ0[2]), 1.0)
    rw, tw = forward(c, θw; f_init = r.f, tau_init = r.tau)
    rec["warm_forward"] = Dict("delta_ln_k" => θw[2] - c.θ0[2], "steps" => rw.steps,
                               "wall_s" => tw, "dist_est" => rw.dist_est, "converged" => rw.converged)
    say(@sprintf("  warm forward (ln k + %.4f): %d steps, %.1f s, dist_est %.1e ; cold forward %d steps, %.1f s",
                 θw[2] - c.θ0[2], rw.steps, tw, rw.dist_est, rec["forward"]["steps"], t_fwd))
    save_results()
end

# ------------------------------------------------------- one measured case ---
function measure_case(spec; gate = false, timing_only = false, t4d_only = false)
    c = make_case(spec)
    rec = Dict{String,Any}("run" => c.run, "m" => c.m, "nu0_lat" => c.nu0, "grid" => [c.Nx, c.Ny],
                           "G_lat" => c.G, "theta0" => c.θ0, "u_ref_lat" => u_ref_lat(c.U))
    RESULTS[c.tag] = rec
    say("=== case $(c.tag) (run $(c.run), m $(c.m), nu0 $(c.nu0)) start, budget left $(round(Int, left())) s")

    # forward from rest
    r, t_fwd = forward(c, c.θ0)
    mach = sqrt(3) * r.u_max
    rec["forward"] = Dict("steps" => r.steps, "wall_s" => t_fwd, "converged" => r.converged,
                          "dist_est" => r.dist_est, "contraction" => r.contraction, "r_u2" => r.r_u2,
                          "u_lat_max" => r.u_max, "mach" => mach, "tau_min" => r.tau_min,
                          "tau_max" => r.tau_max)
    say(@sprintf("  forward: %d steps, %.1f s, dist_est %.1e, rho %.7f, Ma %.3f, tau [%.3f, %.3f]",
                 r.steps, t_fwd, r.dist_est, r.contraction, mach, r.tau_min, r.tau_max))

    # linearisation (CPU blocks) + device copy
    t_lin = @elapsed lin = gnf_linearise(r.f, r.tau, c.solid, c.closure_of(c.θ0), c.G)
    t_copy = @elapsed begin
        op = GNFStepOperator(lin; backend = BK)
        J, bh = gnf_obs_rhs(c.obs, lin)
        b = dev(bh)
        sync()
    end
    kx, ky = kraken_at_reference(r.ux ./ u_ref_lat(c.U), r.uy ./ u_ref_lat(c.U), c.m)
    num = sum(((kx .- c.zx) .^ 2)[c.mask]) + sum(((ky .- c.zy) .^ 2)[c.mask])
    fl = .!c.solid
    rec["linearisation"] = Dict("t_lin" => t_lin, "t_copy" => t_copy, "n_active" => lin.n_active,
                                "n_fluid" => lin.n_fluid, "n_small_I1" => lin.n_small_I1,
                                "n_small_I1_star_1e-12" => count(fl .& (lin.I1 .* c.m^4 ./ c.nu0^2 .< 1e-12)),
                                "I1_min_lat" => lin.I1_min, "nu_min_over_nu0" => minimum(lin.nu[fl]) / c.nu0,
                                "nu_max_over_nu0" => maximum(lin.nu[fl]) / c.nu0, "J" => J,
                                "J_vs_half_rrmse_num" => abs(J - num / 2) / (num / 2),
                                "n_obs" => length(c.obs.target), "gpu_mem_gib" => gpu_mem_gib())
    say(@sprintf("  linearise %.1f s + copy %.1f s ; n_active %d ; small I1 %d ; nu/nu0 [%.3f, %.3f] ; J %.6e (|J - num/2|/(num/2) = %.1e)",
                 t_lin, t_copy, lin.n_active, lin.n_small_I1, rec["linearisation"]["nu_min_over_nu0"],
                 rec["linearisation"]["nu_max_over_nu0"], J, abs(J - num / 2) / (num / 2)))

    n = op.n
    act = op.active
    rng = MersenneTwister(20261020)
    # VJP / JVP time
    x = dev(randn(rng, n)) .* act; y = similar(x)
    gnf_vjp!(y, op, x); gnf_jvp!(y, op, x); sync()
    t_vjp = @elapsed (for _ in 1:200; gnf_vjp!(y, op, x); end; sync())
    t_jvp = @elapsed (for _ in 1:200; gnf_jvp!(y, op, x); end; sync())
    rec["t_vjp"] = t_vjp / 200; rec["t_jvp"] = t_jvp / 200
    if !timing_only
        # T4c on the GPU
        worst = 0.0
        for _ in 1:5
            λr = dev(randn(rng, n)) .* act; v = dev(randn(rng, n)) .* act
            ATl = similar(λr); Av = similar(v)
            gnf_vjp!(ATl, op, λr); gnf_jvp!(Av, op, v)
            a1 = dot(ATl, v); b1 = dot(λr, Av)
            worst = max(worst, abs(a1 - b1) / max(abs(a1), abs(b1)))
        end
        rec["T4c_worst"] = worst
        # T4d on the GPU
        w0 = gnf_pack_state(r.f, r.tau)
        solid_d = dev(c.solid); Fx_d = dev(fill(c.G, c.Nx, c.Ny)); Fy_d = dev(zeros(c.Nx, c.Ny))
        cl = c.closure_of(c.θ0)
        # brief sweep h = 1e-3..1e-6 plus h = 1e-7, reported (strongly
        # shear-thinning cases have a large h² constant: FD truncation, error
        # ratio 100 per decade down to h = 1e-7); verdict on the Richardson
        # extrapolation below
        hs = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
        # Richardson extrapolation of the central difference (Boss, 2026-10-20):
        # D(h) = Av + c h² + O(h⁴), so R(h) = (4 D(h) - D(2h))/3
        # = D(h) + (D(h) - D(2h))/3 cancels the h² term. Verdict on h = 1e-7
        # (with D(2e-7)), threshold 1e-6 per direction (Boss decision,
        # 2026-10-20). h = 1e-6 is information only: its O(h⁴) remainder is
        # ~1e4 times the h = 1e-7 one and exceeds 1e-6 on the shear-thinning
        # cases (job 27007329: up to 1.5e-5 on run4_m64, vs <= 1.7e-9 at 1e-7).
        hr = (1e-7, 1e-6)     # (verdict, information only)
        cdiff = (v, h) -> (gnf_step_state(w0 .+ h .* v, solid_d, cl, Fx_d, Fy_d) .-
                           gnf_step_state(w0 .- h .* v, solid_d, cl, Fx_d, Fy_d)) ./ (2h) .* act
        best = 0.0; best7 = 0.0; slopes_ok = true; t4d = Vector{Vector{Float64}}()
        rich = Vector{Vector{Float64}}()
        for _ in 1:5
            v = dev(randn(rng, n)) .* act .* abs.(w0)
            Av = similar(v); gnf_jvp!(Av, op, v)
            nAv = norm(Av)
            errs = Float64[]; Dh = Dict{Float64,typeof(Av)}()
            for h in hs
                fd = cdiff(v, h)
                h in hr && (Dh[h] = fd)
                push!(errs, norm(fd .- Av) / nAv)
            end
            push!(t4d, errs)
            push!(rich, [norm((4 .* Dh[h] .- cdiff(v, 2h)) ./ 3 .- Av) / nAv for h in hr])
            best = max(best, minimum(errs[1:4])); best7 = max(best7, minimum(errs))
            # h²: some decade of the brief sweep above round-off with ratio 30..300
            slopes_ok &= any(k -> errs[k+1] > 1e-12 && 30 <= errs[k] / errs[k+1] <= 300, 1:3)
        end
        rec["T4d_h"] = collect(hs)
        rec["T4d_errs"] = t4d; rec["T4d_worst_best"] = best; rec["T4d_worst_best_with_1e-7"] = best7
        rec["T4d_h2_ok"] = slopes_ok
        rec["T4d_richardson_h"] = hr[1]; rec["T4d_richardson_errs"] = first.(rich)
        rec["T4d_richardson_worst"] = maximum(first.(rich))
        rec["T4d_richardson_pass"] = [e[1] <= 1e-6 for e in rich]
        rec["T4d_richardson_ok"] = all(rec["T4d_richardson_pass"])
        rec["T4d_richardson_info_only_h"] = hr[2]; rec["T4d_richardson_info_only_errs"] = last.(rich)
        say(@sprintf("  VJP %.3f ms, JVP %.3f ms ; T4c worst %.1e (<= 1e-12: %s) ; T4d worst best-h (1e-3..1e-6) %.1e (<= 1e-6: %s, h^2: %s), with h=1e-7 %.1e",
                     1e3 * rec["t_vjp"], 1e3 * rec["t_jvp"], worst, worst <= 1e-12, best, best <= 1e-6, slopes_ok, best7))
        say("    T4d errors per direction (h = 1e-3..1e-7): " *
            join([join([@sprintf("%.1e", e) for e in es], " ") for es in t4d], " | "))
        say(@sprintf("    T4d Richardson (4 D(h) - D(2h))/3, h = 1e-7 (verdict <= 1e-6) / h = 1e-6 (information only): worst %.1e (pass: %s) ; ",
                     rec["T4d_richardson_worst"], rec["T4d_richardson_ok"]) *
            join([@sprintf("dir %d %.1e %s / %.1e", k, e[1], e[1] <= 1e-6 ? "pass" : "broken (measured $(@sprintf("%.1e", e[1])))", e[2])
                  for (k, e) in enumerate(rich)], " | "))
        # power iteration (50 it.)
        pw = gnf_power_iteration(op; iters = 50)
        rec["power_mu"] = pw.mu; rec["power_ratio"] = pw.ratio
        say(@sprintf("  power iteration (50 it): Rayleigh mu %.6f, |y|/|x| %.6f", pw.mu, pw.ratio))
    end
    save_results()
    t4d_only && return c, rec

    if timing_only
        if CASE_SOLVER === :kstep
            sr, λ, infos = solve_kstep_tols(c, lin, op, b, (1e-6,); item = c.tag)
            info = get(infos, 1e-6, nothing)
        else
            sr, λ, info = solve_record(c, lin, op, b; tol = 1e-6, snaps = (1e-4,), item = c.tag)
        end
        rec["solve"] = sr
        g6 = get(sr["grads"], tkey(1e-6), nothing)
        tg = g6 === nothing ? NaN : g6["t_grad"]
        adj = info === nothing ? NaN : t_lin + t_copy + info.t_total + tg
        rec["value_gradient_tol1e-6"] = Dict("adjoint_s" => adj, "with_forward_s" => adj + t_fwd,
                                             "solver" => string(CASE_SOLVER))
        info === nothing || say(@sprintf("  timing (%s): tol 1e-6 %s in %d it, solve %.1f s (op %.1f, orth %.1f) ; adjoint total %.1f s ; with forward %.1f s",
                     CASE_SOLVER, info.converged ? "reached" : "NOT reached", info.iters, info.t_total, info.t_op,
                     info.t_orth, adj, adj + t_fwd))
        save_results()
        CASE_SOLVER === :kstep && left() > 200 && warm_forward!(c, rec, r, t_fwd)
        return c, rec
    end

    if CASE_SOLVER === :kstep
        # fallback (vi) as the base solver: T4c-K / T4d-K, then one solve per tolerance
        t4_kstep!(rec, c, (; op, r), KSTEP_K)
        sr, λ, infos = solve_kstep_tols(c, lin, op, b, (1e-4, 1e-6, 1e-8); item = c.tag, cap = MAX_SOLVE_BASE)
        info = get(infos, 1e-6, nothing)
        rec["solve"] = sr
    else
    # adjoint solve to 1e-8, snapshots 1e-4, 1e-6
    sr, λ, info = solve_record(c, lin, op, b; item = c.tag, cap = MAX_SOLVE_BASE)
    rec["solve"] = sr
    say(@sprintf("  GMRES(%d, %s): %s, %d it, %d cycles, relres %.1e, %.1f s (op %.1f, orth %.1f), %.2f ms/apply, GPU mem %.1f GiB",
                 RESTART, ORTH, info.converged ? "converged" : "NOT converged", info.iters, info.cycles,
                 info.relres, info.t_total, info.t_op, info.t_orth, 1e3 * sr["t_per_apply"],
                 sr["gpu_mem_gib_after"]))
    for tol in (1e-4, 1e-6, 1e-8)
        g = get(sr["grads"], tkey(tol), nothing)
        g === nothing && (say(@sprintf("    tol %.0e: not reached", tol)); continue)
        say(@sprintf("    tol %.0e: %6d it, %7.1f s ; grad = [%s]", tol, g["iters"], g["t"],
                     join([@sprintf("%.6e", x) for x in g["grad"]], ", ")))
    end
    end
    save_results()

    # finite differences
    dfs = Dict{String,Any}()
    params = gate ? (1:4) : (c.dfp:c.dfp)
    for p in params
        hs = p == c.dfp ? (1e-2, 1e-3, 1e-4) : (1e-3, 1e-4)
        dfs[PNAMES[p]] = fd_param(c, r, p, hs)
        save_results()
    end
    rec["fd"] = dfs
    errtab = Dict{String,Any}()
    for (tk, g) in sr["grads"], p in params
        d = dfs[PNAMES[p]]
        e, h = df_err(g["grad"][p], d)
        errtab["$(tk)_$(PNAMES[p])"] = Dict("rel_err_best_h" => e, "best_h" => h,
                                           "rel_err_all_h" => Dict(k => abs(g["grad"][p] - v["df"]) / abs(v["df"]) for (k, v) in d))
    end
    rec["grad_vs_fd"] = errtab
    for p in params
        d = dfs[PNAMES[p]]
        isempty(d) && continue
        line = @sprintf("  DF %s:", PNAMES[p])
        for k in sort(collect(keys(d)); by = x -> -parse(Float64, x))
            line *= @sprintf(" h=%s %.6e (%.0f s, dist %.0e/%.0e)", k, d[k]["df"], d[k]["wall_s"],
                             d[k]["dist_p"], d[k]["dist_m"])
        end
        say(line)
        for tol in (1e-4, 1e-6, 1e-8)
            e = get(errtab, "$(tkey(tol))_$(PNAMES[p])", nothing)
            e === nothing && continue
            say(@sprintf("    tol %.0e: |adj - DF|/|DF| best h %.2e (h=%.0e)", tol, e["rel_err_best_h"], e["best_h"]))
        end
    end
    save_results()
    return c, rec, (; r, t_fwd, lin, op, b, t_lin, t_copy, info, sr)
end

# Value + gradient time at a tolerance from a solve record (snapshot or final).
function vg_time(st, sr, tolkey, extra = 0.0)
    g = get(sr["grads"], tolkey, nothing)
    g === nothing && return (NaN, NaN, nothing)
    adj = st.t_lin + st.t_copy + g["t"] + g["t_grad"] + extra
    return adj, adj + st.t_fwd, g["grad"]
end

function gate_verdict!(c, rec, st, label, sr, tolkey; extra = 0.0, all_dirs = false)
    adj, tot, g = vg_time(st, sr, tolkey, extra)
    dfs = rec["fd"]
    errs = Dict{String,Float64}()
    if g !== nothing
        for p in 1:4
            haskey(dfs, PNAMES[p]) || continue
            errs[PNAMES[p]] = df_err(g[p], dfs[PNAMES[p]])[1]
        end
    end
    elnk = get(errs, "ln_k", NaN)
    ok_grad = all_dirs ? (!isempty(errs) && all(e -> e <= PASS_REL, values(errs))) : elnk <= PASS_REL
    pass = isfinite(tot) && tot < PASS_S && ok_grad
    v = Dict("label" => label, "tol" => tolkey, "adjoint_s" => adj, "with_forward_s" => tot,
             "rel_err" => errs, "pass" => pass)
    get!(rec, "verdicts", Dict{String,Any}())[label] = v
    say(@sprintf("  VERDICT %-28s tol %-7s value+gradient: adjoint %.1f s, with forward %.1f s ; grad/DF ln k %.2e%s -> %s",
                 label, tolkey, adj, tot, elnk,
                 all_dirs ? " (all: " * join([@sprintf("%s %.1e", k, e) for (k, e) in errs], ", ") * ")" : "",
                 pass ? "PASS" : "fail"))
    save_results()
    return pass
end

# T4c-K / T4d-K of the K-step operator on the GPU (fallback vi): transposition
# <M^K l, v> = <l, (M^T)^K v> (M = Q P A^T P) and (P A P)^K v against central
# differences of K real steps (reported, no threshold: the one-step T4d already
# shows the large h^2 constant of this case).
function t4_kstep!(rec, c, st, K)
    op = st.op; n = op.n; act = op.active
    rng = MersenneTwister(20261021)
    worst = 0.0
    tc = @elapsed for _ in 1:3
        l = dev(randn(rng, n)) .* act; v = dev(randn(rng, n)) .* act
        Ml = similar(l); Mv = similar(v)
        gnf_adjoint_power!(Ml, op, l, K); gnf_tangent_power!(Mv, op, v, K)
        worst = max(worst, abs(dot(Ml, v) - dot(l, Mv)) / (norm(Ml) * norm(v)))
    end
    w0 = gnf_pack_state(st.r.f, st.r.tau)
    solid_d = dev(c.solid); Fx_d = dev(fill(c.G, c.Nx, c.Ny)); Fy_d = dev(zeros(c.Nx, c.Ny))
    cl = c.closure_of(c.θ0)
    stepK(w) = (for _ in 1:K; w = gnf_step_state(w, solid_d, cl, Fx_d, Fy_d); end; w)
    hs = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
    errs_all = Vector{Vector{Float64}}()
    td = @elapsed for _ in 1:2
        v = dev(randn(rng, n)) .* act .* abs.(w0)
        u = copy(v); t = similar(v)
        for _ in 1:K
            gnf_jvp!(t, op, u); u, t = t, u
        end
        errs = Float64[]
        for h in hs
            fd = (stepK(w0 .+ h .* v) .- stepK(w0 .- h .* v)) ./ (2h) .* act
            push!(errs, norm(fd .- u) / norm(u))
        end
        push!(errs_all, errs)
    end
    rec["T4cK"] = Dict("K" => K, "worst" => worst, "t_s" => tc)
    rec["T4dK"] = Dict("K" => K, "h" => collect(hs), "errs" => errs_all, "t_s" => td)
    say(@sprintf("  T4c-K (K = %d): worst |<M^K l, v> - <l, (M^T)^K v>|/(|M^K l||v|) %.1e (%.1f s)", K, worst, tc))
    say(@sprintf("  T4d-K (K = %d), rel err at h = 1e-3..1e-7: ", K) *
        join([join([@sprintf("%.1e", e) for e in es], " ") for es in errs_all], " | ") * @sprintf(" (%.1f s)", td))
    save_results()
end

# JIT warm-up on a small grid (same kernel types as the measured cases), so
# that the cold forward and the solver times of the first case exclude the
# compilation (submission 1 measured the gate case first, compilation included).
function jit_warmup()
    t = @elapsed begin
        c = make_case(parse_case(getarg("warmup_case", "4:8:0.1")))
        r = run_gnf_periodic_steady(; is_solid = c.solid, closure = c.closure_of(c.θ0), Fx = c.G,
                                    Fy = 0.0, backend = BK, T = Float64, tol = 1e-10,
                                    stop = :distance, max_steps = 3000, check_every = 1000,
                                    fused = true)
        lin = gnf_linearise(r.f, r.tau, c.solid, c.closure_of(c.θ0), c.G)
        op = GNFStepOperator(lin; backend = BK)
        _, bh = gnf_obs_rhs(c.obs, lin)
        b = dev(bh)
        gnf_adjoint_solve(op, b; tol = 1e-3, restart = 20, maxiter = 20, orth = ORTH)
        gnf_adjoint_solve(op, b; tol = 1e-3, restart = 20, maxiter = 40, orth = :cgs2, deflate = 4)
        gnf_adjoint_fixed_point(op, b; tol = 1e-3, check_every = 100, maxiter = 300)
        gnf_adjoint_kstep_solve(op, b; K = 11, tol = 1e-3, restart = 5, maxiter = 3)
        gnf_parameter_gradient(lin, b, c.θ0, c.closure_of)
        sync()
    end
    say(@sprintf("JIT warm-up on %s: %.1f s (not part of any timing)", getarg("warmup_case", "4:8:0.1"), t))
end

# ----------------------------------------------------------------- main ---
function main()
    sha, dirty = git_state(REPO)
    dname = devname()
    manifest = Dict{String,Any}(
        "case" => "periodic obstacle channel, steady GNF adjoint gate (J2)",
        "gate" => getarg("gate", "4:64:0.34"),
        "cases" => getarg("cases", "N:32:0.17+1:32:0.17+4:32:0.17+N:64:0.34+1:64:0.34"),
        "timing" => getarg("timing", "4:72:0.34"), "restart_case" => getarg("restart_case", "1:32:0.17"),
        "G_star" => CASE_G, "G_lat" => "5 nu0^2/m^3 (per case in results.toml)",
        "forward" => "run_gnf_periodic_steady, fused, stop=distance, tol 1e-10, check_every 1000, from rest; FD forwards warm-started from the base state",
        "observable" => "J = 1/2 sum over the frozen mask (ux, uy at the reference staggered positions, bilinear interp_node, u* = u_lat/u_ref) of (u*_K - u*_Z)^2",
        "adjoint_tols" => [1e-4, 1e-6, 1e-8], "restart" => RESTART, "orth" => string(ORTH),
        "alt_orth" => ALT_ORTH, "deflate" => DEFLATE, "series_N" => SERIES_N, "fallbacks" => FALLBACKS,
        "case_solver" => string(CASE_SOLVER), "kstep_K" => KSTEP_K, "kstep_restart" => KSTEP_RESTART,
        "t4d_only" => T4D_ONLY,
        "pass" => "value+gradient at tol 1e-6 < 120 s on the gate case, |grad - DF|/|DF| <= 1 % on ln k (best h)",
        "fd_h_rel" => "ln k (Newtonian: ln eta_inf): 1e-2, 1e-3, 1e-4; others on the gate case: 1e-3, 1e-4; delta = h max(|theta|, 1)",
        "seed" => 20261020, "dtype" => "Float64 (Float32 VJP only in fallback iv)",
        "device" => dname, "backend" => string(BK), "commit" => sha, "worktree_dirty" => dirty, "julia" => string(VERSION),
        "threads" => Threads.nthreads(), "budget_s" => BUDGET, "max_solve_s" => MAX_SOLVE,
        "date" => readchomp(`date -Iseconds`))
    open(joinpath(OUT, "manifest.toml"), "w") do io; TOML.print(io, manifest); end
    say("J2 gate on $dname, commit $sha (dirty $dirty), out $OUT")
    getarg("warmup", "1") == "1" && jit_warmup()

    # ---- T4c / T4d (+ Richardson) only, per case, no adjoint solve ----------
    if T4D_ONLY
        specs = String[]
        "gate" in ONLY && push!(specs, getarg("gate", "4:64:0.34"))
        "cases" in ONLY && append!(specs, split(getarg("cases", "N:32:0.17+1:32:0.17+4:32:0.17+N:64:0.34+1:64:0.34"), '+'))
        for s in specs
            left() < 120 && (say("--- skipped case $s (budget)"); continue)
            try
                measure_case(parse_case(s); t4d_only = true)
            catch e
                say("!!! case $s failed: " * sprint(showerror, e, catch_backtrace())[1:min(end, 3000)])
            end
            reclaim()
        end
        RESULTS["wall_s"] = time() - T_START
        save_results()
        say(@sprintf("J2 gate (t4d_only) done in %.0f s", time() - T_START))
        return
    end

    # ---- gate case + fallbacks ------------------------------------------
    if "gate" in ONLY
        try
            c, rec, st = measure_case(parse_case(getarg("gate", "4:64:0.34")); gate = true)
            pass = gate_verdict!(c, rec, st, CASE_SOLVER === :kstep ? "base_kstep$(KSTEP_K)" : "base",
                                 st.sr, tkey(1e-6))
            if CASE_SOLVER === :gmres && !isempty(ALT_ORTH) && left() > 600
                sr2, _, info2 = solve_record(c, st.lin, st.op, st.b; orth = Symbol(ALT_ORTH),
                                             item = c.tag * "_" * ALT_ORTH)
                rec["solve_" * ALT_ORTH] = sr2
                say(@sprintf("  GMRES(%d, %s): %d it, %.1f s (op %.1f, orth %.1f)", RESTART, ALT_ORTH,
                             info2.iters, info2.t_total, info2.t_op, info2.t_orth))
                gate_verdict!(c, rec, st, "base_" * ALT_ORTH, sr2, tkey(1e-6))
            end
            if CASE_SOLVER === :gmres && left() > 900
                # harmonic Ritz values of one full cycle (separate solve: not in the timings)
                srr, _, _ = solve_record(c, st.lin, st.op, st.b; tol = 1e-14, snaps = (), ritz = true,
                                         item = "", maxiter = RESTART)
                rec["ritz_cycle"] = Dict(k => srr[k] for k in ("ritz_smallest_abs", "ritz_smallest") if haskey(srr, k))
                haskey(srr, "ritz_smallest_abs") &&
                    say("  harmonic Ritz values of smallest |theta| (first full cycle): " *
                        join([@sprintf("%.2e", x) for x in srr["ritz_smallest_abs"]], " "))
            end
            if FALLBACKS == "force" || (FALLBACKS == "auto" && !pass)
                # (i) tol 1e-4, accepted only if every direction is within 1 %
                tol_fb = 1e-6
                okI = gate_verdict!(c, rec, st, "fallback_i_tol1e-4", st.sr, tkey(1e-4); all_dirs = true)
                g4 = get(st.sr["grads"], tkey(1e-4), nothing)
                if g4 !== nothing && all(p -> df_err(g4["grad"][p], rec["fd"][PNAMES[p]])[1] <= PASS_REL, 1:4)
                    tol_fb = 1e-4
                end
                rec["fallback_tol"] = tol_fb
                done = okI
                # force: every fallback is measured, even after one passes
                more() = FALLBACKS == "force" || !done
                fb_snaps = tol_fb < 1e-4 ? (1e-4,) : ()
                # (v) fixed-point adjoint iteration (Boss, 2026-10-20), measured first
                if "v" in FB_EXTRA && more() && left() > 300
                    sr, _, inf5 = solve_record(c, st.lin, st.op, st.b; solver = :fixed_point, tol = tol_fb,
                                               snaps = fb_snaps, item = c.tag * "_fb5")
                    rec["fallback_v"] = sr
                    say(@sprintf("  (v) fixed point: %s, %d it, %.1f s, %.3f ms/it (VJP %.3f ms, forward step %.3f ms), contraction %.7f, dist_est %.1e, relres %.1e",
                                 inf5.converged ? "converged" : "NOT converged", inf5.iters, inf5.t_total,
                                 1e3 * inf5.t_total / max(inf5.iters, 1), 1e3 * rec["t_vjp"],
                                 1e3 * st.t_fwd / max(rec["forward"]["steps"], 1), inf5.contraction,
                                 inf5.dist_est, inf5.relres))
                    done |= gate_verdict!(c, rec, st, "fallback_v_fixedpoint", sr, tkey(tol_fb))
                end
                # (vi) GMRES on the K-step system (Boss, 2026-10-20)
                if "vi" in FB_EXTRA && more() && left() > 300
                    t4_kstep!(rec, c, st, KSTEP_K)
                    sr, _, inf6 = solve_record(c, st.lin, st.op, st.b; solver = :kstep, K = KSTEP_K,
                                               restart = KSTEP_RESTART, orth = :cgs2, tol = tol_fb,
                                               snaps = fb_snaps, item = c.tag * "_fb6")
                    rec["fallback_vi"] = sr
                    say(@sprintf("  (vi) K-step GMRES(%d, cgs2), K = %d: %s, %d it (%d VJP), %.1f s (rhs %.1f, op %.1f, orth %.1f), relres %.1e (K-step %.1e, %d refine rounds), %.3f ms/VJP",
                                 KSTEP_RESTART, KSTEP_K, inf6.converged ? "converged" : "NOT converged",
                                 inf6.iters, inf6.n_vjp, inf6.t_total, inf6.t_rhs, inf6.t_op, inf6.t_orth,
                                 inf6.relres, inf6.relres_kstep, inf6.rounds, 1e3 * sr["t_per_vjp"]))
                    done |= gate_verdict!(c, rec, st, "fallback_vi_kstep$(KSTEP_K)", sr, tkey(tol_fb))
                end
                # (ii) deflated restarting
                if more() && left() > 400
                    sr, _, _ = solve_record(c, st.lin, st.op, st.b; deflate = DEFLATE, tol = tol_fb,
                                            snaps = (), item = c.tag * "_fb2")
                    rec["fallback_ii"] = sr
                    done |= gate_verdict!(c, rec, st, "fallback_ii_deflate$(DEFLATE)", sr, tkey(tol_fb))
                end
                # (iii) truncated series as the starting point (+ deflation)
                if more() && left() > 400
                    ts = @elapsed λs = gnf_adjoint_series(st.op, st.b, (SERIES_N,))[SERIES_N]
                    sr, _, _ = solve_record(c, st.lin, st.op, st.b; deflate = DEFLATE, tol = tol_fb,
                                            λ0 = λs, snaps = (), item = c.tag * "_fb3")
                    sr["t_series"] = ts
                    rec["fallback_iii"] = sr
                    done |= gate_verdict!(c, rec, st, "fallback_iii_series$(SERIES_N)", sr, tkey(tol_fb); extra = ts)
                end
                # (iv) Float32 VJP, Float64 vectors and orthogonalisation (+ ii, iii)
                if more() && left() > 400
                    t32 = @elapsed (op32 = GNFStepOperator(st.lin; backend = BK, T = Float32); sync())
                    ts = @elapsed λs = gnf_adjoint_series(op32, st.b, (SERIES_N,))[SERIES_N]
                    sr, _, _ = solve_record(c, st.lin, op32, st.b; deflate = DEFLATE, tol = tol_fb,
                                            λ0 = λs, snaps = (), item = c.tag * "_fb4")
                    sr["t_series"] = ts; sr["t_copy_f32"] = t32
                    rec["fallback_iv"] = sr
                    done |= gate_verdict!(c, rec, st, "fallback_iv_f32", sr, tkey(tol_fb); extra = ts + t32)
                end
                say("  fallbacks: " * (done ? "a fallback passes (see VERDICT lines)" : "no pass after the fallbacks run"))
            end
            # GMRES without restart (only on a >= 70 GiB GPU; Boss, 2026-10-20)
            if NORESTART > 0
                totgib = ON_GPU ? CUDA.total_memory() / 2^30 : 0.0
                if totgib >= 70 && left() > 400
                    sr, _, infn = solve_record(c, st.lin, st.op, st.b; restart = NORESTART, orth = :cgs2,
                                               maxiter = NORESTART, tol = 1e-6, snaps = (1e-4,),
                                               item = c.tag * "_norestart")
                    rec["gmres_norestart"] = sr
                    say(@sprintf("  GMRES no restart (basis %d, cgs2): %s, %d it, relres %.1e, %.1f s (orth %.1f), GPU mem %.1f GiB",
                                 NORESTART, infn.converged ? "converged" : "NOT converged", infn.iters,
                                 infn.relres, infn.t_total, infn.t_orth, sr["gpu_mem_gib_after"]))
                    gate_verdict!(c, rec, st, "gmres_norestart$(NORESTART)", sr, tkey(1e-6))
                else
                    say(@sprintf("  GMRES no restart skipped: GPU total memory %.1f GiB (< 70) or budget", totgib))
                end
            end
            # warm forward (deliverable cost, not a criterion)
            left() > 200 && warm_forward!(c, rec, st.r, st.t_fwd)
        catch e
            say("!!! gate case failed: " * sprint(showerror, e, catch_backtrace())[1:min(end, 3000)])
        end
        reclaim()
    end

    # ---- other cases ----------------------------------------------------
    if "cases" in ONLY
        for s in split(getarg("cases", "N:32:0.17+1:32:0.17+4:32:0.17+N:64:0.34+1:64:0.34"), '+')
            if left() < 600
                say("--- skipped case $s (budget)"); continue
            end
            try
                measure_case(parse_case(s))
            catch e
                say("!!! case $s failed: " * sprint(showerror, e, catch_backtrace())[1:min(end, 3000)])
            end
            reclaim()
        end
    end

    # ---- restart study (same operator) ------------------------------------
    if "restart" in ONLY && left() > 600
        try
            spec = parse_case(getarg("restart_case", "1:32:0.17"))
            c = make_case(spec)
            r, _ = forward(c, c.θ0)
            lin = gnf_linearise(r.f, r.tau, c.solid, c.closure_of(c.θ0), c.G)
            op = GNFStepOperator(lin; backend = BK)
            _, bh = gnf_obs_rhs(c.obs, lin); b = dev(bh)
            reclaim()
            free = free_mem()
            rnone = min(RESTART_NONE_MAX, floor(Int, 0.6 * free / (8 * (op.n + 1))) - 2)
            study = Dict{String,Any}("case" => c.tag, "restart_none_cap" => rnone)
            for (lab, R, o) in (("240", 240, ORTH), ("1000", 1000, ORTH), ("none", rnone, :cgs2),
                                ("240_" * ALT_ORTH, 240, Symbol(isempty(ALT_ORTH) ? "mgs2" : ALT_ORTH)))
                left() < 300 && (say("--- restart $lab skipped (budget)"); continue)
                sr, _, info = solve_record(c, lin, op, b; restart = R, orth = o, item = "restart_$(lab)")
                study[lab] = sr
                say(@sprintf("  restart study %s (%s, m=%d): %s %d it, %d cycles, %.1f s (orth %.1f) ; to 1e-4/1e-6: %s",
                             c.tag, lab, R, info.converged ? "converged" : "NOT converged", info.iters,
                             info.cycles, info.t_total, info.t_orth,
                             join([haskey(sr["grads"], k) ? @sprintf("%d it %.1f s", sr["grads"][k]["iters"], sr["grads"][k]["t"]) : "-"
                                   for k in (tkey(1e-4), tkey(1e-6))], " / ")))
                RESULTS["restart_study"] = study; save_results()
                reclaim()
            end
        catch e
            say("!!! restart study failed: " * sprint(showerror, e, catch_backtrace())[1:min(end, 3000)])
        end
    end

    # ---- timing-only case ----------------------------------------------
    if "timing" in ONLY && left() > 600
        try
            measure_case(parse_case(getarg("timing", "4:72:0.34")); timing_only = true)
        catch e
            say("!!! timing case failed: " * sprint(showerror, e, catch_backtrace())[1:min(end, 3000)])
        end
    end
    RESULTS["wall_s"] = time() - T_START
    save_results()
    say(@sprintf("J2 gate done in %.0f s", time() - T_START))
end

main()
