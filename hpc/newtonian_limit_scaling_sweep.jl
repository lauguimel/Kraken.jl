# Scaling sweep for the Newtonian-limit viscoelastic cylinder force error.
#
# Fits, for each (β, Wi, model),
#     1 - Cd_VE/Cd_Newt = A + B/R^p
# with p = 1 and p = 2.  The Newtonian baseline is computed once per R
# and reused across β.

include(joinpath(@__DIR__, "..", "src", "Kraken.jl"))

using .Kraken
using Dates
using KernelAbstractions
using Printf

const _CUDA_MOD = try
    @eval using CUDA
    getfield(Main, :CUDA)
catch
    nothing
end

const _METAL_MOD = if Sys.isapple()
    try
        @eval using Metal
        getfield(Main, :Metal)
    catch
        nothing
    end
else
    nothing
end

function _select_backend()
    requested = lowercase(get(ENV, "KRAKEN_BACKEND", "auto"))
    if requested in ("auto", "cuda") && _CUDA_MOD !== nothing
        try
            if Base.invokelatest(getfield(_CUDA_MOD, :functional))
                backend = Base.invokelatest(getfield(_CUDA_MOD, :CUDABackend))
                device = Base.invokelatest(getfield(_CUDA_MOD, :device))
                name = Base.invokelatest(getfield(_CUDA_MOD, :name), device)
                return backend, Float64, "CUDA $name"
            end
        catch err
            requested == "cuda" && rethrow(err)
        end
    end
    if requested in ("auto", "metal") && _METAL_MOD !== nothing
        try
            if Base.invokelatest(getfield(_METAL_MOD, :functional))
                backend = Base.invokelatest(getfield(_METAL_MOD, :MetalBackend))
                return backend, Float32, "Metal"
            end
        catch err
            requested == "metal" && rethrow(err)
        end
    end
    return KernelAbstractions.CPU(), Float64, "CPU"
end

function _parse_list(::Type{T}, raw::AbstractString) where {T}
    return [parse(T, strip(x)) for x in split(raw, ',') if !isempty(strip(x))]
end

function _parse_symbol_list(raw::AbstractString)
    return [Symbol(strip(x)) for x in split(raw, ',') if !isempty(strip(x))]
end

function _polymer_bc(name::Symbol)
    name === :cnebb && return CNEBB()
    name === :cnebb_qaware && return CNEBBQAware()
    name in (:cnebb_eq_gradient, :cnebb_eqgrad, :eq_gradient) && return CNEBBEqGradient()
    name in (:cnebb_cutlink_eq_gradient, :cnebb_cutlink_eqgrad, :cutlink_eq_gradient) &&
        return CNEBBCutLinkEqGradient()
    name in (:extrap_eq, :extrapeq, :extrap_eq_wall_bc) && return ExtrapEqWallBC()
    name === :ylw_a && return YLW_A()
    name === :ylw_b && return YLW_B()
    name === :ylw_balance && return YLWBalanceOnly()
    error("unknown polymer BC $(name); expected cnebb, cnebb_qaware, cnebb_eq_gradient, cnebb_cutlink_eq_gradient, extrap_eq, ylw_a, ylw_b, or ylw_balance")
end

function _run_newtonian(; backend, FT, R, u_mean, max_steps, avg_window,
                        drag_stride, momentum_exchange_mode, solvent_magic)
    ν_total = u_mean * R
    cy = (4R - 1) / 2
    return run_cylinder_libb_2d(;
        Nx=30R, Ny=4R, radius=R, cx=15R, cy=cy,
        u_in=FT(1.5 * u_mean), ν=FT(ν_total), inlet=:parabolic,
        max_steps, avg_window, drag_stride,
        momentum_exchange_mode, solvent_magic, backend, T=FT)
end

function _run_visco(; backend, FT, R, u_mean, beta, Wi, model_name,
                    polymer_bc,
                    max_steps, avg_window, drag_stride, drag_mode,
                    hermite_source_mode, solvent_source_mode,
                    momentum_exchange_mode, solvent_magic,
                    conformation_magic, conformation_collision,
                    conformation_divergence_mode,
                    conformation_gradient_mode,
                    source_stress_reconstruction,
                    source_stress_reconstruction_order,
                    source_scale_dynamics,
                    diagnostic_interval,
                    allow_diagnostic_conformation_collision)
    ν_total = u_mean * R
    ν_s = beta * ν_total
    ν_p = (1 - beta) * ν_total
    λ = Wi * R / u_mean
    G = ν_p / λ
    abs(G * λ - ν_p) ≤ 100eps(Float64) * max(ν_p, 1.0) ||
        error("polymer viscosity mismatch: G*λ=$(G * λ), ν_p=$ν_p")
    model = model_name === :logconf ?
        LogConfOldroydB(G=FT(G), λ=FT(λ)) :
        OldroydB(G=FT(G), λ=FT(λ))
    cy = (4R - 1) / 2
    return run_conformation_cylinder_libb_2d(;
        Nx=30R, Ny=4R, radius=R, cx=15R, cy=cy,
        u_mean=FT(u_mean), ν_s=FT(ν_s),
        polymer_model=model, polymer_bc,
        inlet=:parabolic, ρ_out=one(FT), tau_plus=one(FT),
        max_steps, avg_window, drag_stride,
        drag_mode, hermite_source_mode, solvent_source_mode,
        solvent_magic, conformation_magic, conformation_collision,
        conformation_divergence_mode, conformation_gradient_mode,
        momentum_exchange_mode,
        source_stress_reconstruction, source_stress_reconstruction_order,
        source_scale_dynamics, diagnostic_interval,
        allow_diagnostic_polymer_bc=polymer_bc isa CNEBBEqGradient ||
                                    polymer_bc isa CNEBBCutLinkEqGradient,
        allow_diagnostic_conformation_collision,
        allow_diagnostic_force_mode = drag_mode === :source_scaled_mea,
        allow_diagnostic_log_wall_bc = model_name === :logconf,
        backend, FT)
end

function _fit_A_B(rows, p)
    n = length(rows)
    n ≥ 2 || return (A=NaN, B=NaN, rmse=NaN)
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0
    for row in rows
        x = row.R^(-p)
        y = row.err
        sx += x
        sy += y
        sxx += x * x
        sxy += x * y
    end
    denom = n * sxx - sx * sx
    if abs(denom) < eps(Float64)
        return (A=NaN, B=NaN, rmse=NaN)
    end
    B = (n * sxy - sx * sy) / denom
    A = (sy - B * sx) / n
    rmse = sqrt(sum((row.err - (A + B * row.R^(-p)))^2 for row in rows) / n)
    return (A=A, B=B, rmse=rmse)
end

backend, FT, backend_label = _select_backend()

R_list = _parse_list(Int, get(ENV, "KRAKEN_R_LIST", "20,40,80"))
beta_list = _parse_list(Float64, get(ENV, "KRAKEN_BETA_LIST", "0.3,0.59,0.9"))
Wi_list = _parse_list(Float64, get(ENV, "KRAKEN_WI_LIST", "0.001"))
models = _parse_symbol_list(get(ENV, "KRAKEN_MODELS", "direct"))
polymer_bcs = _parse_symbol_list(get(ENV, "KRAKEN_POLYMER_BCS", "cnebb"))
drag_modes = _parse_symbol_list(get(ENV, "KRAKEN_DRAG_MODES", "post_source_mea"))
hermite_source_modes = _parse_symbol_list(get(ENV, "KRAKEN_HERMITE_SOURCE_MODES", "liu_direct"))
solvent_source_mode = Symbol(get(ENV, "KRAKEN_SOLVENT_SOURCE_MODE", "post_collision"))
momentum_exchange_mode = Symbol(get(ENV, "KRAKEN_MOMENTUM_EXCHANGE_MODE", "mei_reconstruct"))
solvent_magic = parse(Float64, get(ENV, "KRAKEN_SOLVENT_MAGIC", string(3/16)))
conformation_magic = parse(Float64, get(ENV, "KRAKEN_CONFORMATION_MAGIC", "1e-6"))
conformation_collision = Symbol(get(ENV, "KRAKEN_CONFORMATION_COLLISION", "trt"))
conformation_divergence_mode = Symbol(get(ENV, "KRAKEN_CONFORMATION_DIVERGENCE_MODE", "trace_free"))
conformation_gradient_mode = Symbol(get(ENV, "KRAKEN_CONFORMATION_GRADIENT_MODE", "wall_aware"))
source_stress_reconstruction = Symbol(get(ENV, "KRAKEN_SOURCE_STRESS_RECONSTRUCTION", "interior"))
source_stress_reconstruction_order = parse(Int, get(ENV, "KRAKEN_SOURCE_STRESS_RECONSTRUCTION_ORDER", "2"))
source_scale_dynamics = parse(Float64, get(ENV, "KRAKEN_SOURCE_SCALE_DYNAMICS", "1.0"))
diagnostic_interval = parse(Int, get(ENV, "KRAKEN_DIAGNOSTIC_INTERVAL", "0"))
allow_diagnostic_conformation_collision = get(ENV, "KRAKEN_ALLOW_DIAGNOSTIC_CONFORMATION_COLLISION", "0") == "1"
u_mean = parse(Float64, get(ENV, "KRAKEN_U_MEAN", "0.005"))
steps_per_R = parse(Int, get(ENV, "KRAKEN_STEPS_PER_R", "2000"))
steps_override = get(ENV, "KRAKEN_STEPS", "")
avg_divisor = parse(Int, get(ENV, "KRAKEN_AVG_DIVISOR", "10"))
drag_stride = parse(Int, get(ENV, "KRAKEN_DRAG_STRIDE", "100"))
control_beta = parse(Float64, get(ENV, "KRAKEN_CONTROL_BETA", "1.0"))
control_R = parse(Int, get(ENV, "KRAKEN_CONTROL_R", string(first(R_list))))
control_all_R = get(ENV, "KRAKEN_CONTROL_ALL_R", "0") == "1"

for model in models
    model in (:direct, :logconf) || error("unknown model $(model); expected direct or logconf")
end
foreach(_polymer_bc, polymer_bcs)

results_dir = get(ENV, "KRAKEN_RESULTS_DIR",
    joinpath("tmp", "newtonian_limit_scaling_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
mkpath(results_dir)
sweep_csv = joinpath(results_dir, "newtonian_limit_scaling_sweep.csv")
fit_csv = joinpath(results_dir, "newtonian_limit_scaling_fit.csv")

println("="^140)
println("Newtonian-limit scaling sweep")
println("Date/time: $(Dates.now())")
println("Backend: $backend_label, FT=$FT")
println("R_LIST=$(join(R_list, ",")) BETA_LIST=$(join(beta_list, ",")) WI_LIST=$(join(Wi_list, ","))")
println("MODELS=$(join(models, ",")) POLYMER_BCS=$(join(polymer_bcs, ",")) DRAG_MODES=$(join(drag_modes, ",")) HERMITE=$(join(hermite_source_modes, ","))")
println("CONFORMATION_COLLISION=$(conformation_collision), CONFORMATION_DIVERGENCE_MODE=$(conformation_divergence_mode), CONFORMATION_GRADIENT_MODE=$(conformation_gradient_mode), SOURCE_SCALE_DYNAMICS=$(source_scale_dynamics)")
println("DIAGNOSTIC_INTERVAL=$(diagnostic_interval)")
println("SOURCE_STRESS_RECONSTRUCTION=$(source_stress_reconstruction), ORDER=$(source_stress_reconstruction_order)")
println("CONTROL beta=$(control_beta), R=$(control_R), all_R=$(control_all_R)")
println("CSV: $sweep_csv")
println("Fit CSV: $fit_csv")
println("="^140)

rows = NamedTuple[]
newtonian_by_R = Dict{Int,Any}()

open(sweep_csv, "w") do io
    println(io, join((
        "kind", "R", "beta", "Wi", "model", "polymer_bc", "drag_mode", "hermite_source_mode",
        "conformation_gradient_mode",
        "Cd_Newt", "Cl_Newt", "Cd", "Cl", "ratio", "err", "pass",
        "Cd_s", "Cd_p", "Cd_post", "Cd_scaled", "Cd_split",
        "G", "lambda", "nu_total", "nu_s", "nu_p",
        "max_steps", "avg_window", "drag_stride", "time_s"
    ), ","))
    @printf("%-10s %-5s %-8s %-8s %-8s %-12s %-16s %-12s %-12s %-12s %-10s %-10s %-8s\n",
            "kind", "R", "beta", "Wi", "model", "bc", "drag", "Cd_Newt", "Cd", "ratio", "err", "pass", "time")
    println("-"^140)

    for R in R_list
        max_steps = isempty(steps_override) ? steps_per_R * R : parse(Int, steps_override)
        avg_window = max(1, max_steps ÷ avg_divisor)
        t0 = time()
        newt = _run_newtonian(; backend, FT, R, u_mean, max_steps, avg_window,
            drag_stride, momentum_exchange_mode, solvent_magic)
        dt = time() - t0
        newtonian_by_R[R] = newt
        println(io, join((
            "newtonian", R, 1.0, 0.0, "-", "-", "-", "-",
            "-",
            newt.Cd, newt.Cl, newt.Cd, newt.Cl, 1.0, 0.0, true,
            newt.Cd, NaN, NaN, NaN, NaN,
            0.0, 0.0, u_mean * R, u_mean * R, 0.0,
            max_steps, avg_window, drag_stride, dt
        ), ","))
        @printf("%-10s %-5d %-8.3f %-8.4g %-8s %-12s %-16s %-12.6f %-12.6f %-10.6f %-10.3g %-8s %-8.1f\n",
                "newtonian", R, 1.0, 0.0, "-", "-", "-", newt.Cd, newt.Cd, 1.0, 0.0, "PASS", dt)
        flush(stdout)
    end

    for Wi in Wi_list, model in models, bc_name in polymer_bcs, drag_mode in drag_modes,
        hermite_source_mode in hermite_source_modes, beta in beta_list, R in R_list
        polymer_bc = _polymer_bc(bc_name)
        newt = newtonian_by_R[R]
        max_steps = isempty(steps_override) ? steps_per_R * R : parse(Int, steps_override)
        avg_window = max(1, max_steps ÷ avg_divisor)
        ν_total = u_mean * R
        ν_s = beta * ν_total
        ν_p = (1 - beta) * ν_total
        λ = Wi * R / u_mean
        G = ν_p / λ
        t0 = time()
        result = _run_visco(; backend, FT, R, u_mean, beta, Wi, model_name=model,
            polymer_bc,
            max_steps, avg_window, drag_stride, drag_mode, hermite_source_mode,
            solvent_source_mode, momentum_exchange_mode, solvent_magic,
            conformation_magic, conformation_collision,
            conformation_divergence_mode, conformation_gradient_mode,
            source_stress_reconstruction, source_stress_reconstruction_order,
            source_scale_dynamics, diagnostic_interval,
            allow_diagnostic_conformation_collision)
        dt = time() - t0
        ratio = result.Cd / newt.Cd
        err = 1 - ratio
        pass = abs(err) ≤ 0.01
        row = (kind="sweep", R=R, beta=beta, Wi=Wi, model=model,
               polymer_bc=bc_name,
               drag_mode=drag_mode, hermite_source_mode=hermite_source_mode,
               conformation_gradient_mode=conformation_gradient_mode,
               Cd_Newt=newt.Cd, Cl_Newt=newt.Cl, Cd=result.Cd, Cl=result.Cl,
               ratio=ratio, err=err, pass=pass, Cd_s=result.Cd_s,
               Cd_p=result.Cd_p, Cd_post=result.Cd_mea_post_source,
               Cd_scaled=result.Cd_mea_source_scaled,
               Cd_split=result.Cd_split_explicit, G=G, λ=λ,
               ν_total=ν_total, ν_s=ν_s, ν_p=ν_p, max_steps=max_steps,
               avg_window=avg_window, drag_stride=drag_stride, time_s=dt)
        push!(rows, row)
        println(io, join((
            row.kind, row.R, row.beta, row.Wi, row.model, row.polymer_bc, row.drag_mode,
            row.hermite_source_mode, row.conformation_gradient_mode,
            row.Cd_Newt, row.Cl_Newt, row.Cd, row.Cl,
            row.ratio, row.err, row.pass, row.Cd_s, row.Cd_p, row.Cd_post,
            row.Cd_scaled, row.Cd_split, row.G, row.λ, row.ν_total, row.ν_s,
            row.ν_p, row.max_steps, row.avg_window, row.drag_stride, row.time_s
        ), ","))
        @printf("%-10s %-5d %-8.3f %-8.4g %-8s %-12s %-16s %-12.6f %-12.6f %-10.6f %-10.3g %-8s %-8.1f\n",
                "sweep", R, beta, Wi, string(model), string(bc_name), string(drag_mode),
                newt.Cd, result.Cd, ratio, err, pass ? "PASS" : "FAIL", dt)
        flush(stdout)
    end

    control_Rs = control_all_R ? R_list : [control_R]
    for Wi in Wi_list, model in models, bc_name in polymer_bcs, drag_mode in drag_modes,
        hermite_source_mode in hermite_source_modes, R in control_Rs
        polymer_bc = _polymer_bc(bc_name)
        R in R_list || error("KRAKEN_CONTROL_R=$(R) is not in KRAKEN_R_LIST")
        newt = newtonian_by_R[R]
        max_steps = isempty(steps_override) ? steps_per_R * R : parse(Int, steps_override)
        avg_window = max(1, max_steps ÷ avg_divisor)
        ν_total = u_mean * R
        ν_s = control_beta * ν_total
        ν_p = (1 - control_beta) * ν_total
        λ = Wi * R / u_mean
        G = ν_p / λ
        t0 = time()
        result = _run_visco(; backend, FT, R, u_mean, beta=control_beta, Wi,
            model_name=model, polymer_bc, max_steps, avg_window, drag_stride, drag_mode,
            hermite_source_mode, solvent_source_mode, momentum_exchange_mode,
            solvent_magic, conformation_magic, conformation_collision,
            conformation_divergence_mode, conformation_gradient_mode,
            source_stress_reconstruction, source_stress_reconstruction_order,
            source_scale_dynamics, diagnostic_interval,
            allow_diagnostic_conformation_collision)
        dt = time() - t0
        ratio = result.Cd / newt.Cd
        err = 1 - ratio
        pass = abs(err) ≤ 1e-10
        println(io, join((
            "control", R, control_beta, Wi, model, bc_name, drag_mode, hermite_source_mode,
            conformation_gradient_mode,
            newt.Cd, newt.Cl, result.Cd, result.Cl, ratio, err, pass,
            result.Cd_s, result.Cd_p, result.Cd_mea_post_source,
            result.Cd_mea_source_scaled, result.Cd_split_explicit,
            G, λ, ν_total, ν_s, ν_p, max_steps, avg_window, drag_stride, dt
        ), ","))
        @printf("%-10s %-5d %-8.3f %-8.4g %-8s %-12s %-16s %-12.6f %-12.6f %-10.6f %-10.3g %-8s %-8.1f\n",
                "control", R, control_beta, Wi, string(model), string(bc_name), string(drag_mode),
                newt.Cd, result.Cd, ratio, err, pass ? "PASS" : "FAIL", dt)
        flush(stdout)
    end
end

open(fit_csv, "w") do io
    println(io, join(("Wi", "beta", "model", "polymer_bc", "drag_mode", "hermite_source_mode",
                      "conformation_gradient_mode",
                      "p", "A", "B", "rmse", "n", "R_values", "err_values"), ","))
    println()
    println("Fits: 1 - ratio = A + B/R^p")
    @printf("%-8s %-8s %-8s %-12s %-16s %-8s %-14s %-14s %-14s %-4s\n",
            "Wi", "beta", "model", "bc", "drag", "p", "A", "B", "rmse", "n")
    println("-"^110)
    for Wi in Wi_list, model in models, bc_name in polymer_bcs, drag_mode in drag_modes,
        hermite_source_mode in hermite_source_modes, beta in beta_list
        group = [row for row in rows if row.Wi == Wi && row.model == model &&
                 row.polymer_bc == bc_name &&
                 row.drag_mode == drag_mode &&
                 row.hermite_source_mode == hermite_source_mode &&
                 row.conformation_gradient_mode == conformation_gradient_mode &&
                 row.beta == beta]
        sort!(group; by = row -> row.R)
        for p in (1.0, 2.0)
            fit = _fit_A_B(group, p)
            R_values = join((row.R for row in group), ";")
            err_values = join((row.err for row in group), ";")
            println(io, join((Wi, beta, model, bc_name, drag_mode, hermite_source_mode,
                              conformation_gradient_mode,
                              p, fit.A, fit.B, fit.rmse, length(group),
                              R_values, err_values), ","))
            @printf("%-8.4g %-8.3f %-8s %-12s %-16s %-8.1f %-14.6g %-14.6g %-14.6g %-4d\n",
                    Wi, beta, string(model), string(bc_name), string(drag_mode), p,
                    fit.A, fit.B, fit.rmse, length(group))
        end
    end
end

println("="^140)
println("Done. Inspect:")
println("  $sweep_csv")
println("  $fit_csv")
