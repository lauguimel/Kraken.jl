# J4-prep forwards of the periodic obstacle channel for the fluids not yet run
# on the GPU (2, 3, 5, 6), with the grid rule of plan §4.4: (m; nu0) = (64; 0.34),
# and a second run at (72; 0.34) for every fluid whose Mach number at m = 64
# exceeds 0.1. Same driver, same observable as J1 (`run_j1_forward.jl`): fixed
# positions on the reference staggered grid, frozen mask, bilinear interpolation.
#
#   julia --project=. benchmarks/inverse_rheology/forward_runs.jl [runs=2+3+5+6] [backend=cuda]
#         [out=<repo>/benchmarks/inverse_rheology/out/forward] [tol=1e-10] [max_steps=2000000]
#         [check_every=1000] [max_wall=2400]
#
# A fluid that does not reach dist <= tol in max_steps, or whose tau_min < 0.505,
# is reported as such (flags in results.toml), nothing else is changed.
#
# Outputs, per fluid: <out>/run<r>/summary.toml (grid chosen, reference folder check,
# reference range) and, per grid, <out>/run<r>/m<m>_nu<nu0>/{manifest.toml,
# results.toml, history.csv, ux_lat.npy, uy_lat.npy, I1_lat.npy, tau.npy,
# err_ux.npy, err_uy.npy, walldist_cells_ux.npy, walldist_cells_uy.npy,
# ux_kraken_at_reference.npy, uy_kraken_at_reference.npy}.

include(joinpath(@__DIR__, "obstacle_channel_case.jl"))
using CUDA

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

argd = Dict{String,String}()
for a in ARGS
    k, v = split(a, "="; limit = 2)
    argd[k] = v
end
getarg(k, d) = get(argd, k, d)
tomlsafe(v::AbstractFloat) = isfinite(v) ? Float64(v) : string(v)
tomlsafe(v) = v

# Reference model of a run folder of the reference fields, read from its iteration_summary.txt (H17).
function reference_model_from_summary(run::Int)
    txt = read(joinpath(REF_FIELDS_DIR, CASE_RUNS[run].folder, "iteration_summary.txt"), String)
    blk = match(r"Reference model:(.*?)Solver configuration:"s, txt)
    blk === nothing && return nothing
    g(key) = (m = match(Regex("$key:\\s*([0-9.eE+-]+)"), blk[1]); m === nothing ? NaN : parse(Float64, m[1]))
    it = match(r"Iteration:\s*(\d+)", txt)
    return (; eta_inf = g("η∞"), eta0 = g("η₀"), lambda = g("λ"), n = g("n"), a = g("a"),
            iteration = it === nothing ? -1 : parse(Int, it[1]))
end

function forward_one(run::Int, m::Int, nu0::Float64, backend, device, bk, outroot;
                     tol, max_steps, check_every, max_wall)
    U = CaseUnits(m, nu0)
    Nx, Ny = grid_size(U)
    solid = case_solid_mask(U)
    cl = case_closure(U, run)
    G = g_lat(U); uref = u_ref_lat(U)
    out = joinpath(outroot, "run$run", @sprintf("m%d_nu%.3f", m, nu0))
    mkpath(out)
    sha, dirty = git_state(REPO)
    p = CASE_RUNS[run]
    manifest = Dict{String,Any}(
        "case" => "periodic obstacle channel, CY truth run $run", "reference_folder" => p.folder,
        "mask" => REF_MASK_FROZEN, "grid" => [Nx, Ny], "m" => m, "nu0_lat" => nu0,
        "G_star" => CASE_G, "G_lat" => G, "u_ref_lat" => uref, "dt_star" => dt_star(U),
        "cy_star" => Dict("eta0" => 1.0, "eta_inf" => p.eta_inf, "lambda" => p.lam, "n" => p.n, "a" => 2.0),
        "cy_lat" => Dict("eta_0" => cl.eta_0, "eta_inf" => cl.eta_inf, "lambda" => cl.lambda, "n" => cl.n, "a" => cl.a),
        "tau0_nominal" => 3 * nu0 + 0.5, "tau_inf_nominal" => 3 * cl.eta_inf + 0.5,
        "solid_nodes" => count(solid), "tol" => tol, "stop" => "distance",
        "check_every" => check_every, "max_steps" => max_steps, "max_wall_s" => max_wall,
        "dtype" => "Float64", "seed" => "none (deterministic)", "path" => "split (strain, closure_eval!, collide)",
        "backend" => bk, "device" => device, "commit" => sha, "worktree_dirty" => dirty,
        "julia" => string(VERSION), "date" => readchomp(`date -Iseconds`))
    open(joinpath(out, "manifest.toml"), "w") do io; TOML.print(io, manifest); end
    @info "forward" run m nu0 Nx Ny G uref device

    res = run_gnf_periodic_steady(; is_solid = solid, closure = cl, Fx = G, Fy = 0.0,
                                  backend, T = Float64, tol, stop = :distance, max_steps, check_every,
                                  max_wall, verbose = true)
    open(joinpath(out, "history.csv"), "w") do io
        println(io, "step,r_u,r_f,r_tau,r_u2")
        for r in res.history; @printf(io, "%d,%.6e,%.6e,%.6e,%.6e\n", r[1], r[2], r[3], r[4], r[5]); end
    end
    ux = Array(res.ux); uy = Array(res.uy)
    mach = res.u_max * sqrt(3)
    kx, ky = kraken_at_reference(ux ./ uref, uy ./ uref, m)
    zx, zy = load_reference_fields(run)
    mask = read_npy(REF_MASK_FROZEN)
    cmp = compare_fields(kx, ky, zx, zy, mask)
    write_npy(joinpath(out, "ux_lat.npy"), ux)
    write_npy(joinpath(out, "uy_lat.npy"), uy)
    write_npy(joinpath(out, "I1_lat.npy"), Array(res.I1))
    write_npy(joinpath(out, "tau.npy"), Array(res.tau))
    write_npy(joinpath(out, "ux_kraken_at_reference.npy"), kx)
    write_npy(joinpath(out, "uy_kraken_at_reference.npy"), ky)
    write_npy(joinpath(out, "err_ux.npy"), cmp.ex)
    write_npy(joinpath(out, "err_uy.npy"), cmp.ey)
    write_npy(joinpath(out, "walldist_cells_ux.npy"), cmp.dx_cells)
    write_npy(joinpath(out, "walldist_cells_uy.npy"), cmp.dy_cells)
    results = Dict{String,Any}(
        "run" => run, "m" => m, "nu0_lat" => nu0, "steps" => res.steps, "converged" => res.converged,
        "finite" => res.finite, "wall_s" => res.wall_time, "s_per_step" => res.wall_time / res.steps,
        "dist_final_rel_est" => res.dist_est, "contraction_per_step" => res.contraction,
        "r_u" => res.r_u, "r_u2" => res.r_u2, "tol" => tol, "dist_reached" => res.dist_est <= tol,
        "u_lat_max" => res.u_max, "u_star_max" => res.u_max / uref, "mach" => mach, "mach_over_0p1" => mach > 0.1,
        "tau_min" => res.tau_min, "tau_max" => res.tau_max, "tau_min_ge_0p505" => res.tau_min >= 0.505,
        "rrmse_u_mask" => cmp.rrmse_u, "rrmse_ux_mask" => cmp.rrmse_ux, "n_mask" => cmp.n_mask,
        "near_wall_share_2dx" => cmp.near_share, "rrmse_far_2dx" => cmp.rrmse_far,
        "level_5pct" => cmp.rrmse_u <= 0.05, "level_10pct" => cmp.rrmse_u <= 0.10)
    open(joinpath(out, "results.toml"), "w") do io
        TOML.print(io, Dict(k => tomlsafe(v) for (k, v) in results))
    end
    println("RESULT ", join(["$k=$(results[k])" for k in sort(collect(keys(results)))], " "))
    return results
end

function main()
    runs = parse.(Int, split(getarg("runs", "2+3+5+6"), '+'))
    bk = getarg("backend", "cuda")
    tol = parse(Float64, getarg("tol", "1e-10"))
    max_steps = parse(Int, getarg("max_steps", "2000000"))
    check_every = parse(Int, getarg("check_every", "1000"))
    max_wall = parse(Float64, getarg("max_wall", "2400"))
    outroot = getarg("out", joinpath(REPO, "benchmarks", "inverse_rheology", "out", "forward"))
    if bk == "cuda"
        CUDA.functional() || error("CUDA not functional on this node")
        backend = CUDABackend(); device = CUDA.name(CUDA.device())
    else
        backend = KernelAbstractions.CPU(); device = "cpu"
    end
    mask = read_npy(REF_MASK_FROZEN)
    for run in runs
        p = CASE_RUNS[run]
        ref = reference_model_from_summary(run)
        match_ok = ref !== nothing && isapprox(ref.eta_inf, p.eta_inf) && isapprox(ref.lambda, p.lam) &&
                   isapprox(ref.n, p.n) && isapprox(ref.eta0, 1.0) && isapprox(ref.a, 2.0)
        zx, zy = load_reference_fields(run)
        rng = Kraken.reference_shear_range(zx, zy, mask)
        share = Kraken.eta_inf_plateau_share((eta0 = 1.0, eta_inf = p.eta_inf, lambda = p.lam, n = p.n, a = 2.0), rng)
        @info "fluid" run p.folder match_ok ref rng share
        g64 = (m = 64, nu0 = 0.34)
        r64 = forward_one(run, g64.m, g64.nu0, backend, device, bk, outroot; tol, max_steps, check_every, max_wall)
        chosen = g64; rchosen = r64; r72 = nothing
        if r64["mach"] > 0.1
            r72 = forward_one(run, 72, 0.34, backend, device, bk, outroot; tol, max_steps, check_every, max_wall)
            chosen = (m = 72, nu0 = 0.34); rchosen = r72
        end
        summary = Dict{String,Any}(
            "run" => run, "reference_folder" => p.folder, "reference_iteration" => ref === nothing ? -1 : ref.iteration,
            "reference_matches_case" => match_ok,
            "reference_model" => ref === nothing ? "missing" : Dict("eta_inf" => ref.eta_inf, "eta0" => ref.eta0,
                                  "lambda" => ref.lambda, "n" => ref.n, "a" => ref.a),
            "mach_m64" => r64["mach"], "grid_rule" => "(64; 0.34), (72; 0.34) if Ma(64) > 0.1",
            "grid_chosen_m" => chosen.m, "grid_chosen_nu0" => chosen.nu0,
            "rrmse_u_m64" => r64["rrmse_u_mask"], "rrmse_u_chosen" => rchosen["rrmse_u_mask"],
            "ref_range_gd_lo" => rng.gd_lo, "ref_range_gd_hi" => rng.gd_hi, "ref_range_n_cells" => rng.n,
            "eta_inf_share_at_gd95" => share)
        mkpath(joinpath(outroot, "run$run"))
        open(joinpath(outroot, "run$run", "summary.toml"), "w") do io
            TOML.print(io, Dict(k => tomlsafe(v) for (k, v) in summary))
        end
        println("SUMMARY ", join(["$k=$(summary[k])" for k in sort(collect(keys(summary)))], " "))
    end
end

main()
