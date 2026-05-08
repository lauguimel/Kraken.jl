#!/usr/bin/env julia

# Confined-cylinder Cd convergence harness for the legacy Liu-style
# viscoelastic cylinder driver. This is benchmark/audit code, not the
# production log-FV polymer backend.

using Dates
using KernelAbstractions
using Kraken
using Printf

const CUDA_MOD = try
    @eval using CUDA
    getfield(Main, :CUDA)
catch
    nothing
end

const METAL_MOD = if Sys.isapple()
    try
        @eval using Metal
        getfield(Main, :Metal)
    catch
        nothing
    end
else
    nothing
end

const LIU_REF = Dict(
    (20, "0.1") => 129.42,
    (20, "0.5") => 125.17,
    (20, "1") => 164.26,
    (25, "0.1") => 129.61,
    (30, "0.1") => 130.36,
    (30, "0.5") => 126.31,
    (30, "1") => 151.31,
    (35, "0.1") => 130.77,
    (35, "0.5") => 127.72,
    (35, "1") => 149.04,
    (40, "0.1") => 130.79,
    (48, "0.1") => 130.83,
)

const RHEOTOOL_REF_MEAN = Dict(
    "0.05" => 131.813036363,
    "0.1" => 130.428774404,
    "0.2" => 126.831069913,
    "0.5" => 119.288284200,
    "1" => 116.995396566,
)

const RHEOTOOL_REF_LAST = Dict(
    "0.05" => 131.813486383,
    "0.1" => 130.429053837,
    "0.2" => 126.839657948,
    "0.5" => 119.713781971,
    "1" => 120.400592705,
)

const RHEOTOOL_NEWTONIAN = 132.362236515

const CSV_COLUMNS = [
    :timestamp, :suite, :status, :error,
    :backend, :float_type, :case_name, :model, :polymer_bc, :gradient,
    :R, :Nx, :Ny, :cx, :cy, :beta, :u_mean, :nu_total, :nu_s, :nu_p,
    :Wi, :lambda, :G, :Sc, :tau_plus, :solvent_magic, :conformation_magic,
    :conformation_collision, :divergence_mode, :initial_condition,
    :wall_geometry, :momentum_exchange_mode, :solvent_source_mode,
    :hermite_source_mode, :drag_mode, :source_reconstruction,
    :source_reconstruction_order, :source_scale_dynamics,
    :source_on_domain_walls, :source_on_cutlinks, :steps, :avg_window,
    :drag_stride, :hydrodynamic_warmup_steps, :dt_s, :lups, :Cd_report,
    :Cd_s, :Cd_p, :Cd_post, :Cd_source_scaled, :Cd_split, :Cl,
    :first_nonfinite_step,
    :min_det_C, :max_tr_C, :nonfinite_C, :n_drag_samples,
    :liu_ref, :err_report_liu_pct, :err_post_liu_pct, :err_split_liu_pct,
    :rheotool_ref_mean, :err_report_rheotool_mean_pct,
    :err_post_rheotool_mean_pct, :err_split_rheotool_mean_pct,
    :rheotool_ref_last, :err_report_rheotool_last_pct,
    :newtonian_ref, :err_report_newtonian_pct,
]

function env_items(raw::AbstractString)
    return (strip(x) for x in split(replace(raw, ';' => ','), ',')
            if !isempty(strip(x)))
end

parse_list(::Type{T}, raw::AbstractString) where {T} =
    [parse(T, x) for x in env_items(raw)]

parse_symbol_list(raw::AbstractString) = [Symbol(x) for x in env_items(raw)]

function parse_bool_env(name::AbstractString, default::Bool)
    raw = lowercase(strip(get(ENV, name, default ? "1" : "0")))
    raw in ("1", "true", "yes", "on") && return true
    raw in ("0", "false", "no", "off") && return false
    error("$(name) must be boolean-like, got $(raw)")
end

function wi_key(wi::Real)
    return @sprintf("%.6g", Float64(wi))
end

function select_backend()
    requested = lowercase(get(ENV, "KRAKEN_BACKEND", "auto"))
    if requested in ("auto", "cuda") && CUDA_MOD !== nothing
        try
            if Base.invokelatest(getfield(CUDA_MOD, :functional))
                backend = Base.invokelatest(getfield(CUDA_MOD, :CUDABackend))
                device = Base.invokelatest(getfield(CUDA_MOD, :device))
                name = Base.invokelatest(getfield(CUDA_MOD, :name), device)
                return backend, "cuda", "CUDA $(name)"
            end
        catch err
            requested == "cuda" && rethrow(err)
        end
    end
    if requested in ("auto", "metal") && METAL_MOD !== nothing
        try
            if Base.invokelatest(getfield(METAL_MOD, :functional))
                backend = Base.invokelatest(getfield(METAL_MOD, :MetalBackend))
                return backend, "metal", "Metal"
            end
        catch err
            requested == "metal" && rethrow(err)
        end
    end
    requested in ("auto", "cpu") ||
        error("unknown or unavailable KRAKEN_BACKEND=$(requested)")
    return KernelAbstractions.CPU(), "cpu", "CPU"
end

function select_float_type(backend_kind::AbstractString)
    raw = lowercase(get(ENV, "KRAKEN_FT", "auto"))
    raw == "auto" && return backend_kind == "metal" ? Float32 : Float64
    raw in ("float32", "single", "f32") && return Float32
    raw in ("float64", "double", "f64") && return Float64
    error("unknown KRAKEN_FT=$(raw); expected auto, float32, or float64")
end

function benchmark_case(name::Symbol)
    name === :direct_cnebb &&
        return (; name, label="direct/CNEBB/wall_aware", model=:direct,
                bc=CNEBB(), gradient=:wall_aware, collision=:liu_eq26,
                allow_mixed_log_bc=false)
    name === :direct_cnebb_embedded &&
        return (; name, label="direct/CNEBB/embedded_axis", model=:direct,
                bc=CNEBB(), gradient=:embedded_axis, collision=:liu_eq26,
                allow_mixed_log_bc=false)
    name === :direct_cnebb_wallfit4 &&
        return (; name, label="direct/CNEBB/wallfit4", model=:direct,
                bc=CNEBB(), gradient=:wallfit4, collision=:liu_eq26,
                allow_mixed_log_bc=false)
    name === :direct_extrapeq_wallfit4 &&
        return (; name, label="direct/ExtrapEq/wallfit4", model=:direct,
                bc=ExtrapEqWallBC(), gradient=:wallfit4, collision=:liu_eq26,
                allow_mixed_log_bc=false)
    name === :logconf_logfield &&
        return (; name, label="logconf/LogField/wall_aware", model=:logconf,
                bc=LogFieldWallBC(), gradient=:wall_aware, collision=:trt,
                allow_mixed_log_bc=false)
    error("unknown case $(name); expected direct_cnebb, direct_cnebb_embedded, direct_cnebb_wallfit4, direct_extrapeq_wallfit4, or logconf_logfield")
end

function polymer_model(model_name::Symbol, G, lambda, ::Type{FT}) where {FT}
    model_name === :direct && return OldroydB(G=FT(G), λ=FT(lambda))
    model_name === :logconf && return LogConfOldroydB(G=FT(G), λ=FT(lambda))
    error("unknown model $(model_name)")
end

function tau_plus_for_collision(collision::Symbol, nu_s, Sc, override::AbstractString)
    !isempty(strip(override)) && return parse(Float64, override)
    collision === :trt && return 1.0
    collision in (:regularized, :liu_eq26) && return 0.5 + 3.0 * nu_s / Sc
    error("unknown conformation collision $(collision)")
end

function suite_defaults(suite::AbstractString, smoke::Bool)
    if smoke
        return "6", "0.1", "direct_cnebb", 60, 60
    elseif suite == "liu"
        return "20,30,35", "0.1,0.5,1.0", "direct_cnebb", 100_000, 200_000
    elseif suite == "rheotool"
        return "30", "0.05,0.1,0.2,0.5,1.0", "direct_cnebb,logconf_logfield",
               100_000, 200_000
    elseif suite == "both"
        return "20,30,35", "0.05,0.1,0.2,0.5,1.0",
               "direct_cnebb,logconf_logfield", 100_000, 200_000
    end
    error("unknown KRAKEN_CYLINDER_SUITE=$(suite); expected liu, rheotool, or both")
end

function csv_cell(x)
    x === nothing && return ""
    if x isa AbstractFloat
        return isfinite(x) ? @sprintf("%.16g", x) : string(x)
    end
    s = string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function append_csv_row(path::AbstractString, row::Dict{Symbol,Any})
    first = !isfile(path)
    open(path, first ? "w" : "a") do io
        if first
            println(io, join(string.(CSV_COLUMNS), ","))
        end
        println(io, join((csv_cell(get(row, col, "")) for col in CSV_COLUMNS), ","))
        flush(io)
    end
end

function pct_error(value, ref)
    (value isa Real && ref isa Real && isfinite(Float64(value)) &&
     isfinite(Float64(ref))) || return NaN
    return 100.0 * (Float64(value) - Float64(ref)) / Float64(ref)
end

function field_or(x, name::Symbol, default)
    return hasproperty(x, name) ? getproperty(x, name) : default
end

function conformation_stats(result)
    hasproperty(result, :C_xx) || return (NaN, NaN, true)
    Cxx = result.C_xx
    Cxy = result.C_xy
    Cyy = result.C_yy
    solid = result.is_solid
    min_det = Inf
    max_tr = -Inf
    nonfinite = false
    @inbounds for j in axes(Cxx, 2), i in axes(Cxx, 1)
        solid[i, j] && continue
        cxx = Float64(Cxx[i, j])
        cxy = Float64(Cxy[i, j])
        cyy = Float64(Cyy[i, j])
        if !(isfinite(cxx) && isfinite(cxy) && isfinite(cyy))
            nonfinite = true
            continue
        end
        min_det = min(min_det, cxx * cyy - cxy * cxy)
        max_tr = max(max_tr, cxx + cyy)
    end
    min_det = isfinite(min_det) ? min_det : NaN
    max_tr = isfinite(max_tr) ? max_tr : NaN
    return (min_det, max_tr, nonfinite)
end

function steps_for_R(R::Int, Wi::Float64, steps_low_wi::Int, steps::Int,
                     scale_steps::Bool, cap::Int)
    base = Wi < 0.01 ? steps_low_wi : steps
    scaled = scale_steps ? round(Int, base * (R / 30)^2) : base
    return cap > 0 ? min(scaled, cap) : scaled
end

function guard_cpu_run!(backend_kind, Nx, Ny, steps, allow_long_cpu, max_lups_cpu)
    backend_kind == "cpu" || return nothing
    allow_long_cpu && return nothing
    updates = Float64(Nx) * Float64(Ny) * Float64(steps)
    updates <= max_lups_cpu && return nothing
    error("refusing long CPU cylinder run ($(updates) lattice updates); set KRAKEN_ALLOW_LONG_CPU=1 or use KRAKEN_BACKEND=cuda/metal")
end

function row_base(; timestamp, suite, backend_label, FT, case_label, model,
                  bc, gradient, R, Nx, Ny, cx, cy, beta, u_mean, nu_total,
                  nu_s, nu_p, Wi, lambda, G, Sc, tau_plus, solvent_magic,
                  conformation_magic, collision, divergence_mode,
                  initial_condition, wall_geometry, momentum_exchange_mode,
                  solvent_source_mode, hermite_source_mode, drag_mode,
                  source_reconstruction, source_reconstruction_order,
                  source_scale_dynamics, source_on_domain_walls,
                  source_on_cutlinks, steps, avg_window, drag_stride,
                  hydrodynamic_warmup_steps)
    return Dict{Symbol,Any}(
        :timestamp => timestamp,
        :suite => suite,
        :status => "started",
        :error => "",
        :backend => backend_label,
        :float_type => string(FT),
        :case_name => case_label,
        :model => model,
        :polymer_bc => bc,
        :gradient => gradient,
        :R => R,
        :Nx => Nx,
        :Ny => Ny,
        :cx => cx,
        :cy => cy,
        :beta => beta,
        :u_mean => u_mean,
        :nu_total => nu_total,
        :nu_s => nu_s,
        :nu_p => nu_p,
        :Wi => Wi,
        :lambda => lambda,
        :G => G,
        :Sc => Sc,
        :tau_plus => tau_plus,
        :solvent_magic => solvent_magic,
        :conformation_magic => conformation_magic,
        :conformation_collision => collision,
        :divergence_mode => divergence_mode,
        :initial_condition => initial_condition,
        :wall_geometry => wall_geometry,
        :momentum_exchange_mode => momentum_exchange_mode,
        :solvent_source_mode => solvent_source_mode,
        :hermite_source_mode => hermite_source_mode,
        :drag_mode => drag_mode,
        :source_reconstruction => source_reconstruction,
        :source_reconstruction_order => source_reconstruction_order,
        :source_scale_dynamics => source_scale_dynamics,
        :source_on_domain_walls => source_on_domain_walls,
        :source_on_cutlinks => source_on_cutlinks,
        :steps => steps,
        :avg_window => avg_window,
        :drag_stride => drag_stride,
        :hydrodynamic_warmup_steps => hydrodynamic_warmup_steps,
    )
end

backend, backend_kind, backend_label = select_backend()
FT = select_float_type(backend_kind)
smoke = parse_bool_env("KRAKEN_SMOKE", false)
suite = lowercase(get(ENV, "KRAKEN_CYLINDER_SUITE", smoke ? "liu" : "both"))
default_R, default_Wi, default_cases, default_steps_low_wi, default_steps =
    suite_defaults(suite, smoke)

R_values = parse_list(Int, get(ENV, "KRAKEN_R_LIST", default_R))
Wi_values = parse_list(Float64, get(ENV, "KRAKEN_WI_LIST", default_Wi))
cases = [benchmark_case(name) for name in
         parse_symbol_list(get(ENV, "KRAKEN_CASES", default_cases))]
magic_values = parse_list(Float64, get(ENV, "KRAKEN_CONFORMATION_MAGIC_LIST",
                                       get(ENV, "KRAKEN_CONFORMATION_MAGIC", "1e-6")))

beta = parse(Float64, get(ENV, "KRAKEN_BETA", "0.59"))
u_mean = parse(Float64, get(ENV, "KRAKEN_U_MEAN", "0.005"))
Sc = parse(Float64, get(ENV, "KRAKEN_SC", "1e4"))
steps_low_wi = parse(Int, get(ENV, "KRAKEN_STEPS_LOW_WI",
                              string(default_steps_low_wi)))
steps_default = parse(Int, get(ENV, "KRAKEN_STEPS", string(default_steps)))
step_cap = parse(Int, get(ENV, "KRAKEN_MAX_STEPS_CAP", smoke ? "1000" : "0"))
scale_steps = parse_bool_env("KRAKEN_SCALE_STEPS_WITH_R", !smoke)
avg_divisor = parse(Int, get(ENV, "KRAKEN_AVG_DIVISOR", smoke ? "2" : "5"))
drag_stride = parse(Int, get(ENV, "KRAKEN_DRAG_STRIDE", smoke ? "5" : "200"))
hydrodynamic_warmup_steps =
    parse(Int, get(ENV, "KRAKEN_HYDRODYNAMIC_WARMUP_STEPS", "0"))
run_newtonian = parse_bool_env("KRAKEN_RUN_NEWTONIAN", true)
continue_on_error = parse_bool_env("KRAKEN_CONTINUE_ON_ERROR", !smoke)
allow_long_cpu = parse_bool_env("KRAKEN_ALLOW_LONG_CPU", false)
max_cpu_updates = parse(Float64, get(ENV, "KRAKEN_MAX_CPU_UPDATES", "2e7"))

solvent_magic = parse(Float64, get(ENV, "KRAKEN_SOLVENT_MAGIC", "0.25"))
solvent_source_mode = Symbol(get(ENV, "KRAKEN_SOLVENT_SOURCE_MODE",
                                 "integrated_collision"))
drag_mode = Symbol(get(ENV, "KRAKEN_DRAG_MODE",
                       solvent_source_mode === :integrated_collision ?
                       "post_source_mea" : "explicit_split"))
hermite_source_mode = Symbol(get(ENV, "KRAKEN_HERMITE_SOURCE_MODE",
                                 "liu_direct"))
source_reconstruction = Symbol(get(ENV, "KRAKEN_SOURCE_STRESS_RECONSTRUCTION",
                                   "interior"))
source_reconstruction_order =
    parse(Int, get(ENV, "KRAKEN_SOURCE_STRESS_RECONSTRUCTION_ORDER", "2"))
source_scale_dynamics =
    parse(Float64, get(ENV, "KRAKEN_SOURCE_SCALE_DYNAMICS", "1.0"))
source_on_domain_walls =
    parse_bool_env("KRAKEN_SOLVENT_SOURCE_ON_DOMAIN_WALLS", false)
source_on_cutlinks =
    parse_bool_env("KRAKEN_SOLVENT_SOURCE_ON_CUTLINKS", true)
divergence_mode = Symbol(get(ENV, "KRAKEN_CONFORMATION_DIVERGENCE_MODE",
                             "trace_free"))
initial_condition = Symbol(get(ENV, "KRAKEN_CONFORMATION_INITIAL_CONDITION",
                               "inlet_profile"))
wall_geometry = Symbol(get(ENV, "KRAKEN_WALL_GEOMETRY", "cutlink"))
momentum_exchange_mode = Symbol(get(ENV, "KRAKEN_MOMENTUM_EXCHANGE_MODE",
                                    "mei_reconstruct"))
collision_override = get(ENV, "KRAKEN_CONFORMATION_COLLISION", "")
tau_plus_override = get(ENV, "KRAKEN_TAU_PLUS", "")
diagnostic_interval = parse(Int, get(ENV, "KRAKEN_DIAGNOSTIC_INTERVAL", "0"))

out_dir = get(ENV, "KRAKEN_OUTPUT_DIR",
              joinpath("tmp", "viscoelastic_logfv", "cylinder_cd_convergence"))
mkpath(out_dir)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
csv_path = get(ENV, "KRAKEN_OUTPUT_CSV",
               joinpath(out_dir, "cylinder_cd_convergence_$(timestamp).csv"))

println("="^78)
println("Confined-cylinder Cd convergence benchmark")
println("Backend: $(backend_label), FT=$(FT), suite=$(suite), smoke=$(smoke)")
println("R=$(join(R_values, ",")) Wi=$(join(Wi_values, ","))")
println("cases=$(join((case.label for case in cases), ","))")
println("magic=$(join(magic_values, ","))")
println("output=$(csv_path)")
println("="^78)

errors = String[]

for R in R_values
    Re_target = 1.0
    nu_total = u_mean * R / Re_target
    nu_s = beta * nu_total
    nu_p = (1.0 - beta) * nu_total
    Nx = 30 * R
    Ny = 4 * R
    cx = 15 * R
    cy = (Ny - 1) / 2

    if run_newtonian
        max_steps = steps_for_R(R, 0.0, steps_low_wi, steps_default,
                                scale_steps, step_cap)
        avg_window = max(1, max_steps ÷ avg_divisor)
        row = row_base(; timestamp, suite, backend_label, FT,
            case_label="newtonian", model=:newtonian, bc=:none,
            gradient=:none, R, Nx, Ny, cx, cy, beta, u_mean, nu_total,
            nu_s=nu_total, nu_p=0.0, Wi=0.0, lambda=0.0, G=0.0, Sc,
            tau_plus=NaN, solvent_magic, conformation_magic=NaN,
            collision=:none, divergence_mode, initial_condition,
            wall_geometry, momentum_exchange_mode, solvent_source_mode=:none,
            hermite_source_mode=:none, drag_mode=:mea, source_reconstruction,
            source_reconstruction_order, source_scale_dynamics,
            source_on_domain_walls, source_on_cutlinks, steps=max_steps,
            avg_window, drag_stride, hydrodynamic_warmup_steps=0)
        try
            guard_cpu_run!(backend_kind, Nx, Ny, max_steps, allow_long_cpu,
                           max_cpu_updates)
            @printf("R=%d Newtonian steps=%d ...\n", R, max_steps)
            t0 = time()
            result = run_cylinder_libb_2d(;
                Nx, Ny, radius=R, cx, cy,
                u_in=FT(1.5 * u_mean), ν=FT(nu_total), inlet=:parabolic,
                max_steps, avg_window, drag_stride,
                momentum_exchange_mode, solvent_magic=FT(solvent_magic),
                backend, T=FT)
            dt = time() - t0
            row[:status] = "ok"
            row[:dt_s] = dt
            row[:lups] = Float64(Nx) * Float64(Ny) * Float64(max_steps) / dt
            row[:Cd_report] = result.Cd
            row[:Cd_s] = result.Cd
            row[:Cd_p] = 0.0
            row[:Cd_post] = NaN
            row[:Cd_source_scaled] = NaN
            row[:Cd_split] = result.Cd
            row[:Cl] = result.Cl
            row[:first_nonfinite_step] = -1
            row[:min_det_C] = NaN
            row[:max_tr_C] = NaN
            row[:nonfinite_C] = false
            row[:n_drag_samples] = result.n_drag_samples
            row[:newtonian_ref] = RHEOTOOL_NEWTONIAN
            row[:err_report_newtonian_pct] =
                pct_error(result.Cd, RHEOTOOL_NEWTONIAN)
            @printf("  Cd=%.9g  err_newt=%.4g%%  dt=%.1fs\n",
                    result.Cd, row[:err_report_newtonian_pct], dt)
        catch err
            row[:status] = "error"
            row[:error] = sprint(showerror, err)
            push!(errors, "R=$(R) newtonian: $(row[:error])")
            @warn "Newtonian case failed" R error=row[:error]
            append_csv_row(csv_path, row)
            continue_on_error || rethrow()
        end
        append_csv_row(csv_path, row)
    end

    for case in cases, Wi in Wi_values, conformation_magic in magic_values
        collision = isempty(strip(collision_override)) ? case.collision :
            Symbol(collision_override)
        if case.model === :logconf && collision in (:regularized, :liu_eq26)
            msg = "log-conformation cylinder only supports TRT collision"
            push!(errors, "R=$(R) Wi=$(Wi) $(case.label): $(msg)")
            continue_on_error || error(msg)
            continue
        end
        lambda = Wi * R / u_mean
        G = nu_p / lambda
        tau_plus = tau_plus_for_collision(collision, nu_s, Sc, tau_plus_override)
        max_steps = steps_for_R(R, Wi, steps_low_wi, steps_default,
                                scale_steps, step_cap)
        avg_window = max(1, max_steps ÷ avg_divisor)
        row = row_base(; timestamp, suite, backend_label, FT,
            case_label=case.label, model=case.model, bc=typeof(case.bc),
            gradient=case.gradient, R, Nx, Ny, cx, cy, beta, u_mean,
            nu_total, nu_s, nu_p, Wi, lambda, G, Sc, tau_plus, solvent_magic,
            conformation_magic, collision, divergence_mode,
            initial_condition, wall_geometry, momentum_exchange_mode,
            solvent_source_mode, hermite_source_mode, drag_mode,
            source_reconstruction, source_reconstruction_order,
            source_scale_dynamics, source_on_domain_walls,
            source_on_cutlinks, steps=max_steps, avg_window, drag_stride,
            hydrodynamic_warmup_steps)
        try
            guard_cpu_run!(backend_kind, Nx, Ny, max_steps, allow_long_cpu,
                           max_cpu_updates)
            model = polymer_model(case.model, G, lambda, FT)
            @printf("R=%d Wi=%.6g %-28s magic=%.6g steps=%d ...\n",
                    R, Wi, case.label, conformation_magic, max_steps)
            t0 = time()
            result = run_conformation_cylinder_libb_2d(;
                Nx, Ny, radius=R, cx, cy,
                u_mean=FT(u_mean), ν_s=FT(nu_s), polymer_model=model,
                polymer_bc=case.bc, tau_plus=FT(tau_plus),
                inlet=:parabolic, ρ_out=one(FT),
                max_steps, avg_window, drag_stride,
                drag_mode, hermite_source_mode, solvent_source_mode,
                solvent_magic=FT(solvent_magic),
                conformation_magic=FT(conformation_magic),
                conformation_collision=collision,
                conformation_divergence_mode=divergence_mode,
                conformation_gradient_mode=case.gradient,
                conformation_initial_condition=initial_condition,
                hydrodynamic_warmup_steps,
                wall_geometry, momentum_exchange_mode,
                source_stress_reconstruction=source_reconstruction,
                source_stress_reconstruction_order=source_reconstruction_order,
                source_scale_dynamics=source_scale_dynamics,
                solvent_source_on_domain_walls=source_on_domain_walls,
                solvent_source_on_cutlinks=source_on_cutlinks,
                diagnostic_interval,
                allow_diagnostic_polymer_bc=!(case.bc isa CNEBB ||
                                              case.bc isa ExtrapEqWallBC ||
                                              case.bc isa LogFieldWallBC),
                allow_diagnostic_force_mode=drag_mode === :source_scaled_mea ||
                    (drag_mode === :post_source_mea &&
                     solvent_source_mode === :post_collision),
                allow_diagnostic_conformation_collision=false,
                allow_diagnostic_log_wall_bc=case.allow_mixed_log_bc,
                backend, FT)
            dt = time() - t0
            min_det_C, max_tr_C, nonfinite_C = conformation_stats(result)
            key = wi_key(Wi)
            liu_ref = get(LIU_REF, (R, key), NaN)
            rheo_mean = get(RHEOTOOL_REF_MEAN, key, NaN)
            rheo_last = get(RHEOTOOL_REF_LAST, key, NaN)
            row[:status] = "ok"
            row[:dt_s] = dt
            row[:lups] = Float64(Nx) * Float64(Ny) * Float64(max_steps) / dt
            row[:Cd_report] = result.Cd
            row[:Cd_s] = result.Cd_s
            row[:Cd_p] = result.Cd_p
            row[:Cd_post] = result.Cd_mea_post_source
            row[:Cd_source_scaled] = result.Cd_mea_source_scaled
            row[:Cd_split] = result.Cd_split_explicit
            row[:Cl] = result.Cl
            row[:first_nonfinite_step] = result.first_nonfinite_step
            row[:min_det_C] = min_det_C
            row[:max_tr_C] = max_tr_C
            row[:nonfinite_C] = nonfinite_C
            row[:n_drag_samples] = result.n_drag_samples
            row[:liu_ref] = liu_ref
            row[:err_report_liu_pct] = pct_error(result.Cd, liu_ref)
            row[:err_post_liu_pct] =
                pct_error(result.Cd_mea_post_source, liu_ref)
            row[:err_split_liu_pct] =
                pct_error(result.Cd_split_explicit, liu_ref)
            row[:rheotool_ref_mean] = rheo_mean
            row[:err_report_rheotool_mean_pct] =
                pct_error(result.Cd, rheo_mean)
            row[:err_post_rheotool_mean_pct] =
                pct_error(result.Cd_mea_post_source, rheo_mean)
            row[:err_split_rheotool_mean_pct] =
                pct_error(result.Cd_split_explicit, rheo_mean)
            row[:rheotool_ref_last] = rheo_last
            row[:err_report_rheotool_last_pct] =
                pct_error(result.Cd, rheo_last)
            @printf("  Cd=%.9g Cd_post=%.9g Cd_split=%.9g Liu_err=%.4g%% Rheo_err=%.4g%% min_detC=%.4g bad=%d dt=%.1fs\n",
                    result.Cd, result.Cd_mea_post_source,
                    result.Cd_split_explicit, row[:err_report_liu_pct],
                    row[:err_report_rheotool_mean_pct], min_det_C,
                    result.first_nonfinite_step, dt)
        catch err
            row[:status] = "error"
            row[:error] = sprint(showerror, err)
            push!(errors, "R=$(R) Wi=$(Wi) $(case.label): $(row[:error])")
            @warn "Viscoelastic case failed" R Wi case=case.label error=row[:error]
            append_csv_row(csv_path, row)
            continue_on_error || rethrow()
        end
        append_csv_row(csv_path, row)
    end
end

println("="^78)
println("CSV: $(csv_path)")
if isempty(errors)
    println("Done without case errors.")
else
    println("Done with $(length(errors)) case error(s):")
    foreach(err -> println("  - ", err), errors)
    continue_on_error || error("cylinder convergence failed")
end
