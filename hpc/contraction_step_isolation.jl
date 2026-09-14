# Axis-aligned step-channel isolation matrix.
#
# Purpose: isolate viscoelastic coupling/wall bugs before returning to a
# curved obstacle. These cases have no curved wall and no Cd post-processing:
# metrics are vortex length, symmetry, conformation positivity, and stress
# levels in axis-aligned re-entrant-corner flows.
#
# Usage examples:
#   KRAKEN_BACKEND=metal julia --project=. hpc/contraction_step_isolation.jl
#   KRAKEN_GEOMETRIES=contraction,bfs KRAKEN_HOUT=20 KRAKEN_STEPS_PER_H=2000 \
#     KRAKEN_WI_LIST=0.05,0.1,0.2 \
#     julia --project=. hpc/contraction_step_isolation.jl

include(joinpath(@__DIR__, "..", "src", "Kraken.jl"))

using .Kraken
using Dates
using KernelAbstractions
using Printf
using Statistics

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

function _parse_list(::Type{T}, value::AbstractString) where {T}
    return [parse(T, strip(x)) for x in split(value, ',') if !isempty(strip(x))]
end

function _parse_symbols(value::AbstractString)
    return [Symbol(strip(x)) for x in split(value, ',') if !isempty(strip(x))]
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

function _polymer_bc(name::Symbol)
    name === :cnebb && return CNEBB()
    name in (:cnebb_eq_gradient, :cnebb_eqgrad, :eq_gradient) && return CNEBBEqGradient()
    name in (:none, :nop, :no_polymer_wall) && return NoPolymerWallBC()
    error("unknown polymer BC $name; expected cnebb, cnebb_eq_gradient, or none")
end

function _model(name::Symbol, FT, ν_p, λ)
    name === :newtonian && return OldroydB(G=zero(FT), λ=one(FT))
    G = FT(ν_p / λ)
    λ_ft = FT(λ)
    name === :direct && return OldroydB(G=G, λ=λ_ft)
    name === :logconf && return LogConfOldroydB(G=G, λ=λ_ft)
    error("unknown model $name; expected direct, logconf, or newtonian")
end

function _build_geometries(names, FT; H_out, β_c, L_up, L_down,
                           bfs_H_in, bfs_expansion_ratio, bfs_L_up, bfs_L_down)
    geometries = Any[]
    for name in names
        if name in (:contraction, :contract, :sudden_contraction)
            push!(geometries, contraction_step_geometry_2d(;
                H_out=H_out, β_c=β_c, L_up=L_up, L_down=L_down, FT=FT))
        elseif name in (:bfs, :backward_facing_step, :backward_step)
            push!(geometries, backward_facing_step_geometry_2d(;
                H_in=bfs_H_in, expansion_ratio=bfs_expansion_ratio,
                L_up=bfs_L_up, L_down=bfs_L_down, FT=FT))
        else
            error("unknown geometry $name; expected contraction or bfs")
        end
    end
    return geometries
end

function _tensor_stats(Cxx, Cxy, Cyy, is_solid)
    min_eig = Inf
    max_trace = -Inf
    max_abs_cxy = 0.0
    max_abs_n1 = 0.0
    @inbounds for k in eachindex(Cxx)
        is_solid[k] && continue
        cxx = Float64(Cxx[k])
        cxy = Float64(Cxy[k])
        cyy = Float64(Cyy[k])
        tr = cxx + cyy
        diff = cxx - cyy
        disc = sqrt(diff * diff + 4.0 * cxy * cxy)
        min_eig = min(min_eig, 0.5 * (tr - disc))
        max_trace = max(max_trace, tr)
        max_abs_cxy = max(max_abs_cxy, abs(cxy))
        max_abs_n1 = max(max_abs_n1, abs(diff))
    end
    return (; min_eig, max_trace, max_abs_cxy, max_abs_n1)
end

function _symmetry_error(field, is_solid; parity::Int)
    Nx, Ny = size(field)
    err = 0.0
    scale = 0.0
    @inbounds for i in 1:Nx, j in 1:(Ny ÷ 2)
        jr = Ny + 1 - j
        (is_solid[i, j] || is_solid[i, jr]) && continue
        a = Float64(field[i, j])
        b = Float64(field[i, jr])
        err = max(err, abs(a - parity * b))
        scale = max(scale, abs(a), abs(b))
    end
    return err / max(scale, eps(Float64))
end

function _masked_face_mean(field, mask, i::Int)
    rows = findall(mask)
    isempty(rows) && return NaN
    return mean(@view field[i, rows])
end

function _sign_with_tolerance(value, tolerance)
    value > tolerance && return 1
    value < -tolerance && return -1
    return 0
end

function _bfs_recirculation_length(result)
    Nx, Ny = size(result.ux)
    max_abs_ux = maximum(abs, result.ux[.!result.is_solid])
    tolerance = sqrt(eps(float(eltype(result.ux)))) * max_abs_ux
    i0 = min(result.i_step + 1, Nx)
    j_probe = nothing
    @inbounds for j in 2:(Ny - 1)
        if !result.is_solid[i0, j]
            j_probe = j
            break
        end
    end
    j_probe === nothing && return NaN

    first_negative_distance = nothing
    @inbounds for distance in 0:(Nx - result.i_step)
        i_probe = result.i_step + distance
        result.is_solid[i_probe, j_probe] && continue
        current_sign = _sign_with_tolerance(Float64(result.ux[i_probe, j_probe]), tolerance)
        if current_sign < 0
            first_negative_distance = distance
            break
        elseif current_sign > 0
            return 0.0
        end
    end
    first_negative_distance === nothing && return 0.0

    @inbounds for distance in (first_negative_distance + 1):(Nx - result.i_step)
        i_probe = result.i_step + distance
        result.is_solid[i_probe, j_probe] && continue
        current_sign = _sign_with_tolerance(Float64(result.ux[i_probe, j_probe]), tolerance)
        current_sign > 0 && return Float64(distance)
    end
    return Float64(Nx - result.i_step)
end

function _metrics(result)
    if result.geometry === :contraction
        X_R_s, _ = vortex_length_contraction_2d(result.ux, result.uy, result.is_solid;
            i_step=result.i_step, j_low=result.j_low, j_high=result.j_high, side=:south)
        X_R_n, _ = vortex_length_contraction_2d(result.ux, result.uy, result.is_solid;
            i_step=result.i_step, j_low=result.j_low, j_high=result.j_high, side=:north)
        xrec_s = X_R_s / result.H_ref
        xrec_n = X_R_n / result.H_ref
        xrec_asym = abs(X_R_s - X_R_n) / result.H_ref
        ux_sym = _symmetry_error(result.ux, result.is_solid; parity=1)
        uy_asym = _symmetry_error(result.uy, result.is_solid; parity=-1)
        Cxx_sym = _symmetry_error(result.C_xx, result.is_solid; parity=1)
        Cxy_asym = _symmetry_error(result.C_xy, result.is_solid; parity=-1)
    elseif result.geometry === :backward_facing_step
        xrec_s = _bfs_recirculation_length(result) / result.H_ref
        xrec_n = NaN
        xrec_asym = NaN
        ux_sym = NaN
        uy_asym = NaN
        Cxx_sym = NaN
        Cxy_asym = NaN
    else
        xrec_s = NaN
        xrec_n = NaN
        xrec_asym = NaN
        ux_sym = NaN
        uy_asym = NaN
        Cxx_sym = NaN
        Cxy_asym = NaN
    end
    j_center = (first(findall(result.east_hydro_mask)) + last(findall(result.east_hydro_mask))) ÷ 2
    N1_center = [result.tau_p_xx[i, j_center] - result.tau_p_yy[i, j_center]
                 for i in result.i_step:result.Nx]
    fluid = .!result.is_solid
    cstats = _tensor_stats(result.C_xx, result.C_xy, result.C_yy, result.is_solid)
    return (;
        xrec_s = xrec_s,
        xrec_n = xrec_n,
        xrec_asym = xrec_asym,
        ux_sym = ux_sym,
        uy_asym = uy_asym,
        Cxx_sym = Cxx_sym,
        Cxy_asym = Cxy_asym,
        min_eig_C = cstats.min_eig,
        max_trace_C = cstats.max_trace,
        max_abs_Cxy = cstats.max_abs_cxy,
        max_abs_N1 = cstats.max_abs_n1,
        max_abs_tau_xy = maximum(abs, result.tau_p_xy[fluid]),
        centerline_N1_max = maximum(abs, N1_center),
        inlet_u_mean = _masked_face_mean(result.ux, result.west_hydro_mask, 2),
        outlet_u_mean = _masked_face_mean(result.ux, result.east_hydro_mask, result.Nx),
        rho_min = minimum(result.ρ[fluid]),
        rho_max = maximum(result.ρ[fluid]))
end

backend, FT, backend_label = _select_backend()

newtonian_mode = lowercase(get(ENV, "KRAKEN_NEWTONIAN", "0")) in ("1", "true", "yes")
geometry_names = _parse_symbols(get(ENV, "KRAKEN_GEOMETRIES", "contraction,bfs"))
H_out = parse(Int, get(ENV, "KRAKEN_HOUT", "8"))
β_c = parse(Int, get(ENV, "KRAKEN_CONTRACTION_RATIO", "4"))
L_up = parse(Int, get(ENV, "KRAKEN_L_UP", "4"))
L_down = parse(Int, get(ENV, "KRAKEN_L_DOWN", "8"))
bfs_H_in = parse(Int, get(ENV, "KRAKEN_BFS_HIN", string(H_out)))
bfs_expansion_ratio = parse(Int, get(ENV, "KRAKEN_BFS_EXPANSION_RATIO", "2"))
bfs_L_up = parse(Int, get(ENV, "KRAKEN_BFS_L_UP", string(L_up)))
bfs_L_down = parse(Int, get(ENV, "KRAKEN_BFS_L_DOWN", string(L_down)))
u_ref_mean = parse(Float64, get(ENV, "KRAKEN_U_REF_MEAN",
    get(ENV, "KRAKEN_U_OUT_MEAN", "0.005")))
Re = parse(Float64, get(ENV, "KRAKEN_RE", "1.0"))
beta = newtonian_mode ? 1.0 : parse(Float64, get(ENV, "KRAKEN_BETA", "0.59"))
Wi_values = newtonian_mode ? [0.0] : _parse_list(Float64, get(ENV, "KRAKEN_WI_LIST", "0.1"))
models = newtonian_mode ? [:newtonian] : _parse_symbols(get(ENV, "KRAKEN_MODELS", "direct"))
polymer_bcs = newtonian_mode ? [:none] : _parse_symbols(get(ENV, "KRAKEN_POLYMER_BCS", "cnebb,none"))
source_modes = newtonian_mode ? [:liu_direct] : _parse_symbols(get(ENV, "KRAKEN_HERMITE_SOURCE_MODES", "liu_direct,ce_corrected"))
magic_values = _parse_list(Float64, get(ENV, "KRAKEN_MAGIC_LIST", "1e-6"))
geometries = _build_geometries(geometry_names, FT;
    H_out, β_c, L_up, L_down,
    bfs_H_in, bfs_expansion_ratio, bfs_L_up, bfs_L_down)
max_H_ref = maximum(geometry.H_ref for geometry in geometries)
steps_per_H = parse(Int, get(ENV, "KRAKEN_STEPS_PER_H", "200"))
max_steps = parse(Int, get(ENV, "KRAKEN_STEPS", string(steps_per_H * max_H_ref)))
avg_divisor = parse(Int, get(ENV, "KRAKEN_AVG_DIVISOR", "5"))
avg_window = max(1, max_steps ÷ avg_divisor)
results_dir = get(ENV, "KRAKEN_RESULTS_DIR",
    joinpath("tmp", "step_geometry_isolation_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
mkpath(results_dir)
csv_path = joinpath(results_dir, "step_geometry_isolation.csv")

println("="^176)
println("Axis-aligned step-channel isolation matrix")
println("Date/time: $(Dates.now())")
println("Backend: $backend_label, FT=$FT")
println("Newtonian mode: $newtonian_mode")
println("Geometries: $(join(string.(getfield.(geometries, :name)), ", "))")
@printf("contraction: H_out=%d beta_c=%d L_up=%d L_down=%d\n", H_out, β_c, L_up, L_down)
@printf("bfs: H_in=%d expansion_ratio=%d L_up=%d L_down=%d\n",
    bfs_H_in, bfs_expansion_ratio, bfs_L_up, bfs_L_down)
@printf("steps=%d avg=%d u_ref=%.6g Re_target=%.6g beta=%.6g\n",
    max_steps, avg_window, u_ref_mean, Re, beta)
println("Wi values: $(join(Wi_values, ", "))")
println("Models: $(join(string.(models), ", "))")
println("Polymer BCs: $(join(string.(polymer_bcs), ", "))")
println("Hermite source modes: $(join(string.(source_modes), ", "))")
println("Conformation magic values: $(join(magic_values, ", "))")
println("CSV: $csv_path")
println("="^176)

header = [
    "case", "geometry", "Nx", "Ny", "H_ref", "H_in", "H_out",
    "Wi_target", "Wi_actual", "Re_target", "Re_actual",
    "model", "polymer_bc", "source_mode", "magic", "tau_plus",
    "nu_s", "nu_p", "lambda",
    "xrec_s", "xrec_n", "xrec_asym", "ux_sym", "uy_asym", "Cxx_sym", "Cxy_asym",
    "min_eig_C", "max_trace_C", "max_abs_Cxy", "max_abs_N1", "max_abs_tau_xy",
    "centerline_N1_max", "inlet_u_mean", "outlet_u_mean", "rho_min", "rho_max", "time_s"
]

_csv_value(value) = value isa Symbol ? String(value) : string(value)

open(csv_path, "w") do io
    println(io, join(header, ","))
    @printf("%-46s %-22s %-8s %-8s %-14s %-10s %-10s %-10s %-10s %-10s %-10s %-8s\n",
        "case", "geometry", "Wi", "Re", "model", "bc", "source", "xrec_s", "minEig", "traceC", "rho_rng", "time")
    println("-"^176)

    for geometry in geometries, Wi in Wi_values, model_name in models, bc_name in polymer_bcs,
        source_mode in source_modes, magic in magic_values

        ν_total = u_ref_mean * geometry.H_ref / Re
        ν_s = beta * ν_total
        ν_p = (1.0 - beta) * ν_total
        λ = newtonian_mode ? 1.0 : Wi * (geometry.H_ref / 2) / u_ref_mean
        tau_plus = if haskey(ENV, "KRAKEN_SC")
            0.5 + 3.0 * ν_s / parse(Float64, ENV["KRAKEN_SC"])
        else
            parse(Float64, get(ENV, "KRAKEN_TAU_PLUS", "1.0"))
        end
        case_name = string(geometry.name, "_", model_name, "_", bc_name, "_", source_mode)
        t0 = time()
        result = try
            run_conformation_step_libb_2d(;
                geometry,
                u_ref_mean=FT(u_ref_mean), ν_s=FT(ν_s),
                polymer_model=_model(model_name, FT, ν_p, λ),
                polymer_bc=_polymer_bc(bc_name),
                ρ_out=one(FT), tau_plus=FT(tau_plus),
                hermite_source_mode=source_mode,
                conformation_magic=magic,
                allow_diagnostic_polymer_bc=_polymer_bc(bc_name) isa CNEBBEqGradient,
                allow_diagnostic_conformation_collision=abs(tau_plus - 1.0) > 1e-12,
                allow_diagnostic_log_wall_bc=model_name === :logconf,
                max_steps=max_steps, avg_window=avg_window,
                backend=backend, FT=FT)
        catch err
            @warn "case failed" case_name geometry=geometry.name Wi magic err
            nothing
        end
        dt = time() - t0

        if result === nothing
            row = Any[
                case_name, geometry.name, geometry.Nx, geometry.Ny,
                geometry.H_ref, geometry.H_in, geometry.H_out,
                Wi, NaN, Re, NaN, model_name, bc_name, source_mode,
                magic, tau_plus, ν_s, ν_p, λ, fill(NaN, 17)..., dt
            ]
            println(io, join(_csv_value.(row), ","))
            @printf("%-46s %-22s %-8.3g %-8.3g %-14s %-10s %-10s %-10s %-10s %-10s %-10s %-8.1f\n",
                case_name, string(geometry.name), Wi, Re, string(model_name),
                string(bc_name), string(source_mode), "FAILED", "NaN", "NaN", "NaN", dt)
            flush(stdout)
            continue
        end

        m = _metrics(result)
        Wi_report = newtonian_mode ? 0.0 : result.Wi
        row = Any[
            case_name, geometry.name, result.Nx, result.Ny,
            result.H_ref, geometry.H_in, result.H_out,
            Wi, Wi_report, Re, result.Re,
            model_name, bc_name, source_mode, magic, tau_plus, ν_s, ν_p, λ,
            m.xrec_s, m.xrec_n, m.xrec_asym, m.ux_sym, m.uy_asym, m.Cxx_sym, m.Cxy_asym,
            m.min_eig_C, m.max_trace_C, m.max_abs_Cxy, m.max_abs_N1, m.max_abs_tau_xy,
            m.centerline_N1_max, m.inlet_u_mean, m.outlet_u_mean, m.rho_min, m.rho_max, dt
        ]
        println(io, join(_csv_value.(row), ","))
        @printf("%-46s %-22s %-8.3g %-8.3g %-14s %-10s %-10s %-10.4g %-10.4g %-10.4g %-10.4g %-8.1f\n",
            case_name, string(geometry.name), Wi_report, result.Re,
            string(model_name), string(bc_name), string(source_mode),
            m.xrec_s, m.min_eig_C, m.max_trace_C, m.rho_max - m.rho_min, dt)
        flush(stdout)
    end
end

println("="^176)
println("Done. Inspect: $csv_path")
