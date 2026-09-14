# Newtonian-limit check for viscoelastic cylinder force modes.
#
# Criterion at Wi ≪ 1: Cd_VE / Cd_Newt should be in [0.99, 1.01].
# This is independent of the bulk constitutive checks and targets force
# accounting modes directly.

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

function _case_filter()
    raw = strip(get(ENV, "KRAKEN_CASES", ""))
    isempty(raw) && return nothing
    return Set(Symbol(strip(x)) for x in split(raw, ',') if !isempty(strip(x)))
end

function _model_list()
    raw = get(ENV, "KRAKEN_MODELS", get(ENV, "KRAKEN_FORMULATIONS", "direct"))
    models = [Symbol(strip(x)) for x in split(raw, ',') if !isempty(strip(x))]
    isempty(models) && push!(models, :logconf)
    for model in models
        model in (:direct, :logconf) ||
            error("unknown model $(model); expected direct or logconf")
    end
    return models
end

function _polymer_bc_from_env()
    name = Symbol(strip(get(ENV, "KRAKEN_POLYMER_BC", "cnebb")))
    name === :cnebb && return CNEBB()
    name === :extrap_eq && return ExtrapEqWallBC()
    error("unknown KRAKEN_POLYMER_BC=$(name); expected cnebb or extrap_eq")
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

function _run_visco(; backend, FT, R, u_mean, beta, Wi, max_steps, avg_window,
                    drag_stride, drag_mode, hermite_source_mode,
                    solvent_source_mode, momentum_exchange_mode,
                    solvent_magic, conformation_magic, model_name,
                    polymer_bc, conformation_gradient_mode,
                    source_stress_reconstruction,
                    source_stress_reconstruction_order,
                    source_scale_dynamics)
    ν_total = u_mean * R
    ν_s = beta * ν_total
    ν_p = (1 - beta) * ν_total
    λ = Wi * R / u_mean
    cy = (4R - 1) / 2
    G = ν_p / λ
    abs(G * λ - ν_p) ≤ 100eps(Float64) * max(ν_p, 1.0) ||
        error("polymer viscosity mismatch: G*λ=$(G * λ), ν_p=$ν_p")
    model = model_name === :logconf ?
        LogConfOldroydB(G=FT(G), λ=FT(λ)) :
        OldroydB(G=FT(G), λ=FT(λ))
    return run_conformation_cylinder_libb_2d(;
        Nx=30R, Ny=4R, radius=R, cx=15R, cy=cy,
        u_mean=FT(u_mean), ν_s=FT(ν_s),
        polymer_model=model, polymer_bc=polymer_bc,
        inlet=:parabolic, ρ_out=one(FT), tau_plus=one(FT),
        max_steps, avg_window, drag_stride,
        drag_mode, hermite_source_mode, solvent_source_mode,
        solvent_magic, conformation_magic,
        conformation_gradient_mode,
        source_stress_reconstruction,
        source_stress_reconstruction_order,
        source_scale_dynamics,
        momentum_exchange_mode,
        allow_diagnostic_force_mode = drag_mode === :source_scaled_mea,
        allow_diagnostic_log_wall_bc = model_name === :logconf,
        backend, FT)
end

function _node_value(field, solid, i::Int, j::Int)
    nx, ny = size(field)
    if i < 1 || i > nx || j < 1 || j > ny || solid[i, j]
        return 0.0
    end
    return Float64(field[i, j])
end

function _polymer_stress_from_velocity(ux, uy, is_solid, ν_p)
    nx, ny = size(ux)
    txx = zeros(Float64, nx, ny)
    txy = zeros(Float64, nx, ny)
    tyy = zeros(Float64, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        is_solid[i, j] && continue
        dux_dx = 0.5 * (_node_value(ux, is_solid, i + 1, j) -
                        _node_value(ux, is_solid, i - 1, j))
        duy_dy = 0.5 * (_node_value(uy, is_solid, i, j + 1) -
                        _node_value(uy, is_solid, i, j - 1))
        dux_dy = 0.5 * (_node_value(ux, is_solid, i, j + 1) -
                        _node_value(ux, is_solid, i, j - 1))
        duy_dx = 0.5 * (_node_value(uy, is_solid, i + 1, j) -
                        _node_value(uy, is_solid, i - 1, j))
        txx[i, j] = 2ν_p * dux_dx
        txy[i, j] = ν_p * (dux_dy + duy_dx)
        tyy[i, j] = 2ν_p * duy_dy
    end
    return txx, txy, tyy
end

function _polymer_drag_from_velocity(result, ν_p, cx, cy, R)
    txx, txy, tyy = _polymer_stress_from_velocity(result.ux, result.uy,
                                                  result.is_solid, ν_p)
    drag = Kraken.compute_polymeric_drag_2d(txx, txy, tyy,
                                            result.q_wall, size(txx, 1), size(txx, 2);
                                            cx=cx, cy=cy, radius=R)
    Cd = 2.0 * drag.Fx / (result.u_ref^2 * result.D)
    Cl = 2.0 * drag.Fy / (result.u_ref^2 * result.D)
    return (; Cd, Cl, Fx=drag.Fx, Fy=drag.Fy, txx, txy, tyy)
end

function _bilinear(field, x, y)
    nx, ny = size(field)
    x = clamp(Float64(x), 0.0, nx - 1.0)
    y = clamp(Float64(y), 0.0, ny - 1.0)
    i0 = clamp(floor(Int, x) + 1, 1, nx)
    j0 = clamp(floor(Int, y) + 1, 1, ny)
    i1 = min(i0 + 1, nx)
    j1 = min(j0 + 1, ny)
    ax = x - (i0 - 1)
    ay = y - (j0 - 1)
    return (1 - ax) * (1 - ay) * Float64(field[i0, j0]) +
           ax * (1 - ay) * Float64(field[i1, j0]) +
           (1 - ax) * ay * Float64(field[i0, j1]) +
           ax * ay * Float64(field[i1, j1])
end

_safe_ratio(a, b) = abs(b) > 1e-14 ? a / b : NaN
_stress_norm(xx, xy, yy) = sqrt(xx^2 + 2xy^2 + yy^2)

function _write_probe_rows(io, case_label, model_name, result, newt_fd,
                           G, cx, cy, R)
    for (θ_label, θ) in (("0", 0.0), ("pi_2", π / 2))
        nx = cos(θ)
        ny = sin(θ)
        for d in 1:5
            x = cx + (R + d) * nx
            y = cy + (R + d) * ny
            τxx = _bilinear(result.tau_p_xx, x, y)
            τxy = _bilinear(result.tau_p_xy, x, y)
            τyy = _bilinear(result.tau_p_yy, x, y)
            cxx = _bilinear(result.C_xx, x, y)
            cxy = _bilinear(result.C_xy, x, y)
            cyy = _bilinear(result.C_yy, x, y)
            gc_xx = G * (cxx - 1)
            gc_xy = G * cxy
            gc_yy = G * (cyy - 1)
            n_xx = _bilinear(newt_fd.txx, x, y)
            n_xy = _bilinear(newt_fd.txy, x, y)
            n_yy = _bilinear(newt_fd.tyy, x, y)
            τ_norm = _stress_norm(τxx, τxy, τyy)
            n_norm = _stress_norm(n_xx, n_xy, n_yy)
            println(io, join((
                case_label, model_name, θ_label, d, x, y,
                τxx, τxy, τyy, cxx, cxy, cyy,
                gc_xx, gc_xy, gc_yy,
                _safe_ratio(τxx, gc_xx), _safe_ratio(τxy, gc_xy), _safe_ratio(τyy, gc_yy),
                n_xx, n_xy, n_yy, _safe_ratio(τ_norm, n_norm)
            ), ","))
        end
    end
end

backend, FT, backend_label = _select_backend()
R = parse(Int, get(ENV, "KRAKEN_R", "20"))
u_mean = parse(Float64, get(ENV, "KRAKEN_U_MEAN", "0.005"))
beta = parse(Float64, get(ENV, "KRAKEN_BETA", "0.59"))
Wi = parse(Float64, get(ENV, "KRAKEN_WI", "0.001"))
steps_per_R = parse(Int, get(ENV, "KRAKEN_STEPS_PER_R", "2000"))
max_steps = parse(Int, get(ENV, "KRAKEN_STEPS", string(steps_per_R * R)))
avg_divisor = parse(Int, get(ENV, "KRAKEN_AVG_DIVISOR", "10"))
avg_window = max(1, max_steps ÷ avg_divisor)
drag_stride = parse(Int, get(ENV, "KRAKEN_DRAG_STRIDE", "100"))
solvent_magic = parse(Float64, get(ENV, "KRAKEN_SOLVENT_MAGIC", string(3/16)))
conformation_magic = parse(Float64, get(ENV, "KRAKEN_CONFORMATION_MAGIC", "1e-6"))
conformation_gradient_mode = Symbol(get(ENV, "KRAKEN_CONFORMATION_GRADIENT_MODE", "wall_aware"))
source_stress_reconstruction = Symbol(get(ENV, "KRAKEN_SOURCE_STRESS_RECONSTRUCTION", "interior"))
source_stress_reconstruction_order =
    parse(Int, get(ENV, "KRAKEN_SOURCE_STRESS_RECONSTRUCTION_ORDER", "2"))
source_scale_dynamics = parse(Float64, get(ENV, "KRAKEN_SOURCE_SCALE_DYNAMICS", "1.0"))
polymer_bc = _polymer_bc_from_env()
models = _model_list()
results_dir = get(ENV, "KRAKEN_RESULTS_DIR",
    joinpath("tmp", "newtonian_limit_cylinder_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
mkpath(results_dir)
csv_path = joinpath(results_dir, "newtonian_limit_cylinder.csv")
probe_path = joinpath(results_dir, "newtonian_limit_stress_probe.csv")

cases = [
    (:source_scaled_mea, :post_collision, :ce_corrected, :source_scaled_mea, :mei_reconstruct),
    (:post_liu_raw, :post_collision, :liu_direct, :post_source_mea, :mei_reconstruct),
    (:explicit_split_liu, :post_collision, :liu_direct, :explicit_split, :mei_reconstruct),
    (:explicit_split_ce, :post_collision, :ce_corrected, :explicit_split, :mei_reconstruct),
    (:integrated_liu, :integrated_collision, :liu_direct, :post_source_mea, :mei_reconstruct),
]
filter = _case_filter()
if filter !== nothing
    cases = [case for case in cases if case[1] in filter]
end

println("="^132)
println("Newtonian-limit cylinder force-mode check")
println("Date/time: $(Dates.now())")
println("Backend: $backend_label, FT=$FT")
@printf("R=%d u_mean=%.6g beta=%.6g Wi=%.6g steps=%d avg=%d drag_stride=%d\n",
    R, u_mean, beta, Wi, max_steps, avg_window, drag_stride)
println("polymer_bc=$(typeof(polymer_bc)) conformation_gradient_mode=$conformation_gradient_mode source_stress_reconstruction=$source_stress_reconstruction order=$source_stress_reconstruction_order source_scale=$source_scale_dynamics")
println("Models: $(join(string.(models), ","))")
println("CSV: $csv_path")
println("Stress probe CSV: $probe_path")
println("="^132)

t0 = time()
newt = _run_newtonian(; backend, FT, R, u_mean, max_steps, avg_window,
    drag_stride, momentum_exchange_mode=:mei_reconstruct, solvent_magic)
newt_time = time() - t0
ν_total = u_mean * R
ν_p = (1 - beta) * ν_total
λ = Wi * R / u_mean
G = ν_p / λ
cx = 15R
cy = (4R - 1) / 2
newt_fd = _polymer_drag_from_velocity(newt, ν_p, cx, cy, R)

open(csv_path, "w") do io
    open(probe_path, "w") do probe_io
        println(probe_io, join((
            "case", "model", "theta", "dist", "x", "y",
            "tau_xx", "tau_xy", "tau_yy", "C_xx", "C_xy", "C_yy",
            "G_CmI_xx", "G_C_xy", "G_CmI_yy",
            "tau_over_GC_xx", "tau_over_GC_xy", "tau_over_GC_yy",
            "tau_newt_xx", "tau_newt_xy", "tau_newt_yy",
            "tau_norm_over_newt_norm"
        ), ","))

    header = [
        "case", "model", "Cd", "Cl", "Cd_Newt", "ratio", "pass",
        "Cd_s", "Cd_p", "Cd_post", "Cd_scaled", "Cd_split",
        "Cd_p_newt_fd", "Cd_p_over_newt_fd", "Cd_p_ve_fd", "Cd_p_over_ve_fd",
        "G", "lambda", "nu_p",
        "drag_mode", "hermite_source_mode", "solvent_source_mode", "time_s"
    ]
    println(io, join(header, ","))
    @printf("%-30s %-8s %-12s %-12s %-12s %-10s %-6s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-8s\n",
        "case", "model", "Cd", "Cl", "Cd_Newt", "ratio", "pass", "Cd_s", "Cd_p",
        "Cd_post", "Cd_scaled", "Cd_split", "Cd_p_newt", "Cd_p/newt", "time")
    println("-"^132)
    @printf("%-30s %-8s %-12.6f %-12.6g %-12.6f %-10.6f %-6s %-12.6f %-12s %-12s %-12s %-12s %-12.6f %-12s %-8.1f\n",
        "newtonian", "-", newt.Cd, newt.Cl, newt.Cd, 1.0, "PASS",
        newt.Cd, "NaN", "NaN", "NaN", "NaN", newt_fd.Cd, "NaN", newt_time)

    for model_name in models
    for (base_label, solvent_source_mode, hermite_source_mode, drag_mode, force_mode) in cases
        label = length(models) > 1 ? Symbol("$(model_name)_$(base_label)") : base_label
        tcase = time()
        result = _run_visco(; backend, FT, R, u_mean, beta, Wi, max_steps,
            avg_window, drag_stride, drag_mode, hermite_source_mode,
            solvent_source_mode, momentum_exchange_mode=force_mode,
            solvent_magic, conformation_magic, model_name, polymer_bc,
            conformation_gradient_mode, source_stress_reconstruction,
            source_stress_reconstruction_order, source_scale_dynamics)
        dt = time() - tcase
        ratio = result.Cd / newt.Cd
        pass = 0.99 <= ratio <= 1.01
        ve_fd = _polymer_drag_from_velocity(result, ν_p, cx, cy, R)
        _write_probe_rows(probe_io, label, model_name, result, newt_fd, G, cx, cy, R)
        row = Any[
            label, model_name, result.Cd, result.Cl, newt.Cd, ratio, pass,
            result.Cd_s, result.Cd_p, result.Cd_mea_post_source,
            result.Cd_mea_source_scaled, result.Cd_split_explicit,
            newt_fd.Cd, result.Cd_p / newt_fd.Cd,
            ve_fd.Cd, result.Cd_p / ve_fd.Cd,
            G, λ, ν_p,
            result.drag_mode, result.hermite_source_mode,
            result.solvent_source_mode, dt
        ]
        println(io, join(string.(row), ","))
        @printf("%-30s %-8s %-12.6f %-12.6g %-12.6f %-10.6f %-6s %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f %-8.1f\n",
            string(label), string(model_name), result.Cd, result.Cl, newt.Cd, ratio, pass ? "PASS" : "FAIL",
            result.Cd_s, result.Cd_p, result.Cd_mea_post_source,
            result.Cd_mea_source_scaled, result.Cd_split_explicit,
            newt_fd.Cd, result.Cd_p / newt_fd.Cd, dt)
        flush(stdout)
    end
    end
    end
end

println("="^132)
println("Done. Inspect: $csv_path")
