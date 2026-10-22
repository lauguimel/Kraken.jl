# J1 GPU measurements for the periodic obstacle channel (plan §4.1 J1):
# steady CY forward, convergence, Mach, τ visited, Re at the throat, T2b
# stability, comparison with the reference field on the frozen mask, H14 size,
# split vs fused timing.
#
#   julia --project=. benchmarks/inverse_rheology/run_j1_forward.jl run=1 backend=cuda \
#         out=$HOME/inverse_rheology_runs/j1 [m=64 nu0=0.34 tol=1e-10 stop=distance
#         max_steps=3000000 check_every=1000 max_wall=2400 t2b_units=2.5 timing_steps=2000]
#
# m and nu0 default to the grid rule of `case_grid(run)` (obstacle_channel_case.jl):
# (64; 0.34), or (72; 0.34) for a fluid whose Mach exceeds 0.1 (run 4).
# stop=distance (default) stops on the estimated distance to the fixed point,
# stop=step_change on the one-step change r_u (J1 criterion); see
# `run_gnf_periodic_steady`.
#
# Outputs (never in the repo): <out>/run<r>_m<m>_nu<ν0>/{manifest.toml, results.toml,
# history.csv, throat.csv, t2b.csv, *.npy}.

include(joinpath(@__DIR__, "obstacle_channel_case.jl"))
using CUDA

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

argd = Dict{String,String}()
for a in ARGS
    k, v = split(a, "="; limit = 2)
    argd[k] = v
end
getarg(k, d) = get(argd, k, d)

function shear_rate_local(fp, Fx, Fy, τ; corrected::Bool)
    ρ, ux, uy = gnf_moments_cell(fp, Fx, Fy)
    fe = gnf_feq_cell(ρ, ux, uy)
    n = fp .- fe
    Pxx = n[2] + n[4] + n[6] + n[7] + n[8] + n[9]
    Pyy = n[3] + n[5] + n[6] + n[7] + n[8] + n[9]
    Pxy = n[6] - n[7] + n[8] - n[9]
    if corrected
        Pxx += Fx * ux; Pyy += Fy * uy; Pxy += (Fx * uy + Fy * ux) / 2
    end
    c = -3 / (2 * ρ * τ)
    return sqrt(2 * ((c * Pxx)^2 + (c * Pyy)^2 + 2 * (c * Pxy)^2))
end

# Least-squares slope of log(y) against x.
function log_slope(x, y)
    lx = Float64.(x); ly = log.(Float64.(y))
    mx = sum(lx) / length(lx); my = sum(ly) / length(ly)
    return sum((lx .- mx) .* (ly .- my)) / sum((lx .- mx) .^ 2)
end

tomlsafe(v::AbstractFloat) = isfinite(v) ? Float64(v) : string(v)
tomlsafe(v) = v

function main()
    run = parse(Int, getarg("run", "1"))
    grid = case_grid(run)
    m = parse(Int, getarg("m", string(grid.m)))
    nu0 = parse(Float64, getarg("nu0", string(grid.nu0)))
    bk = getarg("backend", "cuda")
    tol = parse(Float64, getarg("tol", "1e-10"))
    stop = Symbol(getarg("stop", "distance"))
    max_steps = parse(Int, getarg("max_steps", "3000000"))
    check_every = parse(Int, getarg("check_every", "1000"))
    max_wall = parse(Float64, getarg("max_wall", "2400"))
    t2b_units = parse(Float64, getarg("t2b_units", "2.5"))
    timing_steps = parse(Int, getarg("timing_steps", "2000"))
    outroot = getarg("out", joinpath(homedir(), "inverse_rheology_runs", "j1"))

    if bk == "cuda"
        CUDA.functional() || error("CUDA not functional on this node")
        backend = CUDABackend()
        device = CUDA.name(CUDA.device())
    else
        backend = KernelAbstractions.CPU()
        device = "cpu"
    end

    U = CaseUnits(m, nu0)
    Nx, Ny = grid_size(U)
    solid = case_solid_mask(U)
    cl = case_closure(U, run)
    G = g_lat(U)
    uref = u_ref_lat(U)
    tag = @sprintf("run%d_m%d_nu%.3f", run, m, nu0)
    out = joinpath(outroot, tag)
    mkpath(out)
    sha, dirty = git_state(REPO)
    p = CASE_RUNS[run]

    manifest = Dict{String,Any}(
        "case" => "periodic obstacle channel, CY truth run $run",
        "reference_folder" => p.folder, "mask" => REF_MASK_FROZEN,
        "grid" => [Nx, Ny], "m" => m, "nu0_lat" => nu0, "G_star" => CASE_G, "G_lat" => G,
        "u_ref_lat" => uref, "dt_star" => dt_star(U),
        "cy_star" => Dict("eta0" => 1.0, "eta_inf" => p.eta_inf, "lambda" => p.lam, "n" => p.n, "a" => 2.0),
        "cy_lat" => Dict("eta_0" => cl.eta_0, "eta_inf" => cl.eta_inf, "lambda" => cl.lambda, "n" => cl.n, "a" => cl.a),
        "tau0_nominal" => 3 * nu0 + 0.5, "tau_inf_nominal" => 3 * cl.eta_inf + 0.5,
        "solid_nodes" => count(solid), "tol" => tol, "stop" => string(stop),
        "grid_rule" => "(m; nu0) = $(Tuple(grid)) for run $run (case_grid)",
        "residual" => "r_u = ||u^k - u^(k-1)||_2/||u^k||_2, r_u2 = ||u^k - u^(k-2)||_2/||u^k||_2 over fluid nodes; dist_est = r_u2 rho^2/(1-rho^2), rho fitted on log r_u2 over the last 20 checks",
        "path_closure_input" => "I1 = S:S (gnf_viscosity(closure, I1))",
        "check_every" => check_every, "max_steps" => max_steps, "max_wall_s" => max_wall,
        "dtype" => "Float64", "seed" => "none (deterministic)", "path" => "split (strain, closure_eval!, collide)",
        "backend" => bk, "device" => device, "commit" => sha, "worktree_dirty" => dirty,
        "julia" => string(VERSION), "date" => readchomp(`date -Iseconds`))
    open(joinpath(out, "manifest.toml"), "w") do io; TOML.print(io, manifest); end
    @info "J1 forward" tag Nx Ny G uref device

    # ---- steady solve ------------------------------------------------------
    res = run_gnf_periodic_steady(; is_solid = solid, closure = cl, Fx = G, Fy = 0.0,
                                  backend, T = Float64, tol, stop, max_steps, check_every,
                                  max_wall, verbose = true)
    s_per_step = res.wall_time / res.steps
    h = res.history
    open(joinpath(out, "history.csv"), "w") do io
        println(io, "step,r_u,r_f,r_tau,r_u2")
        for r in h; @printf(io, "%d,%.6e,%.6e,%.6e,%.6e\n", r[1], r[2], r[3], r[4], r[5]); end
    end
    rho1 = res.contraction; dist_rel = res.dist_est     # computed by the driver
    @info "steady solve" res.steps res.converged res.r_u res.r_u2 res.r_f res.r_tau res.wall_time s_per_step rho1 dist_rel

    # ---- fields, Mach, τ, Re at the throat --------------------------------
    uxs = res.ux ./ uref; uys = res.uy ./ uref
    mach = res.u_max * sqrt(3)
    kx, ky = kraken_at_reference(uxs, uys, m)
    zx, zy = load_reference_fields(run)
    mask = read_npy(REF_MASK_FROZEN)
    size(mask) == (REF_NX, REF_NY) || error("mask shape $(size(mask))")
    cmp = compare_fields(kx, ky, zx, zy, mask)
    thr = throat_profile(kx, zx)
    u_thr_K = maximum(r[3] for r in thr); u_thr_Z = maximum(r[2] for r in thr)
    nu_min_star = (res.tau_min - 0.5) / 3 / nu0
    re_throat = u_thr_K * 1.0 / nu_min_star
    open(joinpath(out, "throat.csv"), "w") do io
        println(io, "y,ux_reference,ux_kraken")
        for r in thr; @printf(io, "%.6f,%.6e,%.6e\n", r[1], r[2], r[3]); end
    end
    write_npy(joinpath(out, "ux_star_nodes.npy"), uxs)
    write_npy(joinpath(out, "uy_star_nodes.npy"), uys)
    write_npy(joinpath(out, "ux_kraken_at_reference.npy"), kx)
    write_npy(joinpath(out, "uy_kraken_at_reference.npy"), ky)
    write_npy(joinpath(out, "err_ux.npy"), cmp.ex)
    write_npy(joinpath(out, "err_uy.npy"), cmp.ey)
    write_npy(joinpath(out, "walldist_cells_ux.npy"), cmp.dx_cells)
    write_npy(joinpath(out, "walldist_cells_uy.npy"), cmp.dy_cells)
    write_npy(joinpath(out, "f_state.npy"), Array(res.f))
    write_npy(joinpath(out, "tau_state.npy"), Array(res.tau))
    @info "comparison" cmp.rrmse_u cmp.rrmse_ux cmp.near_share cmp.rrmse_far cmp.rrmse_all u_thr_K u_thr_Z re_throat mach

    # ---- H14: Guo forcing correction of the shear-rate estimator ----------
    fa_d = similar(res.f)
    stream_periodic_x_wall_y_2d!(fa_d, res.f, Nx, Ny)
    fa = Array(fa_d); tauh = Array(res.tau)
    dnu = 0.0; dg = 0.0
    for j in 1:Ny, i in 1:Nx
        solid[i, j] && continue
        fp = Kraken.SVector{9}(ntuple(q -> fa[i, j, q], 9))
        g0 = shear_rate_local(fp, G, 0.0, tauh[i, j]; corrected = false)
        gc = shear_rate_local(fp, G, 0.0, tauh[i, j]; corrected = true)
        n0 = effective_viscosity(cl, g0); nc = effective_viscosity(cl, gc)
        dnu = max(dnu, abs(nc - n0) / n0)
        g0 > 0 && (dg = max(dg, abs(gc - g0) / g0))
    end
    @info "H14" dnu dg

    # ---- timing: split vs fused, from the converged state -----------------
    solid_d = KernelAbstractions.allocate(backend, Bool, Nx, Ny); copyto!(solid_d, solid)
    Fx_d = KernelAbstractions.allocate(backend, Float64, Nx, Ny); fill!(Fx_d, G)
    Fy_d = KernelAbstractions.zeros(backend, Float64, Nx, Ny)
    i1 = KernelAbstractions.zeros(backend, Float64, Nx, Ny)    # I1 = S:S buffer
    nu = KernelAbstractions.zeros(backend, Float64, Nx, Ny)
    timing = Dict{String,Float64}()
    for fused in (false, true)
        fi = copy(res.f); fo = similar(fi); ta = copy(res.tau)
        for _ in 1:10
            gnf_step!(fo, fi, ta, i1, nu, solid_d, cl, Fx_d, Fy_d; fused); fi, fo = fo, fi
        end
        KernelAbstractions.synchronize(backend)
        t = @elapsed begin
            for _ in 1:timing_steps
                gnf_step!(fo, fi, ta, i1, nu, solid_d, cl, Fx_d, Fy_d; fused); fi, fo = fo, fi
            end
            KernelAbstractions.synchronize(backend)
        end
        timing[fused ? "fused" : "split"] = t / timing_steps
    end
    @info "timing (s/step, no residual checks)" timing

    # ---- T2b: decay of mass-conserving, symmetry-breaking perturbations ---
    n2b = round(Int, t2b_units / dt_star(U))
    every = max(1, n2b ÷ 200)
    eps = 1e-2 * res.u_max
    f0 = Array(res.f); rhoh = res.rho
    for j in 1:Ny, i in 1:Nx
        solid[i, j] && continue
        x = (i - 0.5) / m; y = (j - 0.5) / m
        dux = eps * sin(2π * y / CASE_H) * (1 + 0.5 * cos(2π * x / CASE_L))      # odd in y
        duy = eps * sin(2π * (x - CASE_L / 2) / CASE_L) * sin(π * y / CASE_H)  # odd about x = L/2
        fe0 = gnf_feq_cell(rhoh[i, j], res.ux[i, j], res.uy[i, j])
        fe1 = gnf_feq_cell(rhoh[i, j], res.ux[i, j] + dux, res.uy[i, j] + duy)
        for q in 1:9
            f0[i, j, q] += fe1[q] - fe0[q]
        end
    end
    mass_change = abs(sum(f0) - sum(Array(res.f))) / sum(Array(res.f))
    fi = KernelAbstractions.allocate(backend, Float64, Nx, Ny, 9); copyto!(fi, f0)
    fo = similar(fi); ta = copy(res.tau)
    uxs_d = KernelAbstractions.allocate(backend, Float64, Nx, Ny); copyto!(uxs_d, res.ux)
    uys_d = KernelAbstractions.allocate(backend, Float64, Nx, Ny); copyto!(uys_d, res.uy)
    rb = similar(uxs_d); ub = similar(uxs_d); vb = similar(uxs_d)
    unorm = sqrt(sum(res.ux .^ 2 .+ res.uy .^ 2))
    rec = Tuple{Int,Float64}[]
    gnf_macroscopic_post_2d!(rb, ub, vb, fi, solid_d, Fx_d, Fy_d)
    push!(rec, (0, sqrt(sum((ub .- uxs_d) .^ 2 .+ (vb .- uys_d) .^ 2)) / unorm))
    t2b_wall = @elapsed for s in 1:n2b
        gnf_step!(fo, fi, ta, i1, nu, solid_d, cl, Fx_d, Fy_d); fi, fo = fo, fi
        if s % every == 0
            gnf_macroscopic_post_2d!(rb, ub, vb, fi, solid_d, Fx_d, Fy_d)
            d = sqrt(sum((ub .- uxs_d) .^ 2 .+ (vb .- uys_d) .^ 2)) / unorm
            push!(rec, (s, d))
            isfinite(d) || break
        end
    end
    open(joinpath(out, "t2b.csv"), "w") do io
        println(io, "step,delta_rel")
        for r in rec; @printf(io, "%d,%.6e\n", r[1], r[2]); end
    end
    sel = [r for r in rec if r[1] >= 0.1 * n2b && r[2] > 0]
    half = [r for r in rec if r[1] >= 0.5 * n2b && r[2] > 0]
    slope10 = length(sel) >= 3 ? log_slope(first.(sel), last.(sel)) : NaN
    slope50 = length(half) >= 3 ? log_slope(first.(half), last.(half)) : NaN
    t2b_pass = isfinite(slope10) && slope10 < 0 && all(isfinite, last.(rec))
    efold_star = slope10 < 0 ? -dt_star(U) / slope10 : Inf
    @info "T2b" n2b eps mass_change d0 = rec[1][2] d_end = rec[end][2] slope10 slope50 efold_star t2b_pass t2b_wall

    results = Dict{String,Any}(
        "steps" => res.steps, "converged" => res.converged, "finite" => res.finite,
        "stop" => string(res.stop),
        "r_u" => res.r_u, "r_u2" => res.r_u2, "r_f" => res.r_f, "r_tau" => res.r_tau,
        "wall_s" => res.wall_time, "s_per_step" => s_per_step,
        "contraction_per_step" => rho1, "dist_to_fixed_point_rel_est" => dist_rel,
        "u_lat_max" => res.u_max, "mach" => mach, "mach_pass" => res.u_max <= 0.1 / sqrt(3),
        "u_star_max" => res.u_max / uref,
        "tau_min" => res.tau_min, "tau_max" => res.tau_max, "tau_min_ge_0p505" => res.tau_min >= 0.505,
        "rho_min" => res.rho_min, "rho_max" => res.rho_max,
        "u_throat_star_max_kraken" => u_thr_K, "u_throat_star_max_reference" => u_thr_Z,
        "nu_star_min" => nu_min_star, "re_throat" => re_throat, "re_throat_nu0" => u_thr_K,
        "rrmse_u_mask" => cmp.rrmse_u, "rrmse_ux_mask" => cmp.rrmse_ux,
        "near_wall_share_2dx" => cmp.near_share, "rrmse_far_2dx" => cmp.rrmse_far,
        "rrmse_u_all_cells" => cmp.rrmse_all, "n_mask" => cmp.n_mask,
        "h14_max_dnu_over_nu" => dnu, "h14_max_dgamma_over_gamma" => dg,
        "s_per_step_split" => timing["split"], "s_per_step_fused" => timing["fused"],
        "t2b_steps" => n2b, "t2b_eps_lat" => eps, "t2b_mass_change_rel" => mass_change,
        "t2b_delta0" => rec[1][2], "t2b_delta_end" => rec[end][2],
        "t2b_slope_after10pct" => slope10, "t2b_slope_last50pct" => slope50,
        "t2b_efold_time_star" => efold_star, "t2b_pass" => t2b_pass, "t2b_wall_s" => t2b_wall)
    open(joinpath(out, "results.toml"), "w") do io
        TOML.print(io, Dict(k => tomlsafe(v) for (k, v) in results))
    end
    println("RESULT ", tag, " ", join(["$k=$(results[k])" for k in sort(collect(keys(results)))], " "))
end

main()
