include(joinpath(@__DIR__, "..", "src", "Kraken.jl"))
using .Kraken
using Printf
using Dates
using KernelAbstractions

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

function _case_filter(env_name::AbstractString)
    value = strip(get(ENV, env_name, ""))
    isempty(value) && return nothing
    return Set(strip.(split(value, ',')))
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

function cylinder_geometry(R::Int, mode::AbstractString)
    if mode == "exact_nodes"
        return (Nx = 30R + 1, Ny = 4R + 1, cx = 15R, cy = 2R)
    elseif mode == "centered_legacy"
        Nx = 30R
        Ny = 4R
        return (Nx = Nx, Ny = Ny, cx = 15R, cy = (Ny - 1) / 2)
    elseif mode == "legacy"
        return (Nx = 30R, Ny = 4R, cx = 15R, cy = 2R)
    else
        error("unknown KRAKEN_GEOMETRY_MODE=$mode; expected exact_nodes, centered_legacy, or legacy")
    end
end

function run_newtonian_case(; backend, FT, R, u_mean, steps_per_R,
                            avg_divisor, drag_stride, wall_mode,
                            momentum_exchange_mode, geometry_mode,
                            solvent_magic)
    max_steps = steps_per_R * R
    avg_window = max(1, max_steps ÷ avg_divisor)
    ν_total = u_mean * R
    geom = cylinder_geometry(R, geometry_mode)
    if wall_mode === :libb_v2
        return run_cylinder_libb_2d(;
            Nx = geom.Nx, Ny = geom.Ny, radius = R, cx = geom.cx, cy = geom.cy,
            u_in = FT(1.5 * u_mean), ν = FT(ν_total), inlet = :parabolic,
            max_steps, avg_window, drag_stride,
            momentum_exchange_mode, solvent_magic, backend, T = FT)
    elseif wall_mode === :halfway_staircase
        return run_cylinder_2d(;
            Nx = geom.Nx, Ny = geom.Ny, radius = R, cx = geom.cx, cy = geom.cy,
            u_in = Float64(u_mean), ν = Float64(ν_total),
            max_steps, avg_window, backend, T = FT)
    else
        error("unknown wall_mode $wall_mode")
    end
end

function run_visco_case(; backend, FT, R, u_mean, beta, Wi, formulation,
                        steps_per_R, avg_divisor, drag_stride,
                        drag_mode, hermite_source_mode, solvent_source_mode,
                        momentum_exchange_mode, geometry_mode,
                        solvent_magic, tau_plus, conformation_magic,
                        conformation_collision)
    max_steps = steps_per_R * R
    avg_window = max(1, max_steps ÷ avg_divisor)
    ν_total = u_mean * R
    ν_s = beta * ν_total
    ν_p = (1 - beta) * ν_total
    λ = Wi * R / u_mean
    G = ν_p / λ
    model = formulation == "logconf" ?
        LogConfOldroydB(G = FT(G), λ = FT(λ)) :
        OldroydB(G = FT(G), λ = FT(λ))
    geom = cylinder_geometry(R, geometry_mode)
    return run_conformation_cylinder_libb_2d(;
        Nx = geom.Nx, Ny = geom.Ny, radius = R, cx = geom.cx, cy = geom.cy,
        u_mean = FT(u_mean), ν_s = FT(ν_s),
        polymer_model = model, polymer_bc = CNEBB(),
        inlet = :parabolic, ρ_out = one(FT), tau_plus = FT(tau_plus),
        max_steps, avg_window, drag_stride,
        drag_mode, hermite_source_mode, solvent_source_mode,
        solvent_magic,
        conformation_magic,
        conformation_collision,
        momentum_exchange_mode,
        allow_diagnostic_force_mode = drag_mode === :source_scaled_mea,
        allow_diagnostic_conformation_collision = true,
        allow_diagnostic_log_wall_bc = formulation == "logconf",
        backend, FT)
end

backend, FT, backend_label = _select_backend()
R_values = _parse_list(Int, get(ENV, "KRAKEN_R_LIST", "30"))
u_mean = parse(Float64, get(ENV, "KRAKEN_U_MEAN", "0.005"))
beta = parse(Float64, get(ENV, "KRAKEN_BETA", "0.59"))
Wi = parse(Float64, get(ENV, "KRAKEN_WI", "0.1"))
steps_per_R = parse(Int, get(ENV, "KRAKEN_STEPS_PER_R", "4000"))
avg_divisor = parse(Int, get(ENV, "KRAKEN_AVG_DIVISOR", "10"))
drag_stride = parse(Int, get(ENV, "KRAKEN_DRAG_STRIDE", "200"))
formulation = get(ENV, "KRAKEN_FORMULATION", "direct")
geometry_mode = get(ENV, "KRAKEN_GEOMETRY_MODE", "centered_legacy")
solvent_magic = parse(Float64, get(ENV, "KRAKEN_SOLVENT_MAGIC", string(3/16)))
conformation_magic = parse(Float64, get(ENV, "KRAKEN_CONFORMATION_MAGIC", "1e-6"))
conformation_collision = Symbol(get(ENV, "KRAKEN_CONFORMATION_COLLISION", "trt"))
tau_plus_override = get(ENV, "KRAKEN_TAU_PLUS", "")
sc_override = get(ENV, "KRAKEN_SC", "")
newtonian_filter = _case_filter("KRAKEN_NEWTONIAN_CASES")
visco_filter = _case_filter("KRAKEN_VISCO_CASES")

rheotool_newt = 132.362236515
rheotool_visco = 130.428774404
liu_visco = Dict(20 => 129.42, 25 => 129.61, 30 => 130.36,
                 35 => 130.77, 40 => 130.79, 48 => 130.83)

newtonian_cases = [
    ("libb_mei", :libb_v2, :mei_reconstruct),
    ("libb_liu_eq63", :libb_v2, :liu_eq63),
    ("libb_simple", :libb_v2, :simple_halfway),
    ("libb_postpair", :libb_v2, :postpair),
    ("halfway_std", :halfway_staircase, :standard),
]

visco_cases = [
    ("post_ce_scaled", :post_collision, :ce_corrected, :source_scaled_mea, :mei_reconstruct),
    ("post_ce_scaled_liu", :post_collision, :ce_corrected, :source_scaled_mea, :liu_eq63),
    ("post_ce_raw", :post_collision, :ce_corrected, :post_source_mea, :mei_reconstruct),
    ("post_liu_raw", :post_collision, :liu_direct, :post_source_mea, :mei_reconstruct),
    ("integrated_ce_raw", :integrated_collision, :ce_corrected, :post_source_mea, :mei_reconstruct),
    ("integrated_ce_liu", :integrated_collision, :ce_corrected, :post_source_mea, :liu_eq63),
    ("integrated_liu_raw", :integrated_collision, :liu_direct, :post_source_mea, :mei_reconstruct),
]

println("="^156)
println("Newtonian/visco force-source scheme matrix")
println("Date/time: $(Dates.now())")
println("Backend: $backend_label, FT=$FT")
@printf("u_mean=%.6g beta=%.6g Wi=%.6g steps_per_R=%d avg_divisor=%d drag_stride=%d formulation=%s\n",
        u_mean, beta, Wi, steps_per_R, avg_divisor, drag_stride, formulation)
println("Geometry mode: $geometry_mode")
println("Solvent magic Λs: $solvent_magic")
println("Conformation magic Λp: $conformation_magic")
println("Conformation collision: $conformation_collision")
if !isempty(tau_plus_override)
    println("Conformation tau_plus override: $tau_plus_override")
elseif !isempty(sc_override)
    println("Conformation Schmidt Sc override: $sc_override")
else
    println("Conformation tau_plus default: 1.0")
end
println("R values: $(join(R_values, ", "))")
newtonian_filter !== nothing && println("Newtonian case filter: $(join(sort(collect(newtonian_filter)), ", "))")
visco_filter !== nothing && println("Visco case filter: $(join(sort(collect(visco_filter)), ", "))")
println("="^156)
@printf("%-10s %-20s %-4s %-18s %-18s %-16s %-12s %-12s %-12s %-10s %-10s %-10s %-10s %-9s\n",
        "family", "case", "R", "wall/source", "force", "hermite", "Cd",
        "Cl", "ref", "err%", "Re_R", "Re_D", "samples", "time")
println("-"^156)

for R in R_values
    for (label, wall_mode, force_mode) in newtonian_cases
        newtonian_filter !== nothing && !(label in newtonian_filter) && continue
        t0 = time()
        result = try
            run_newtonian_case(; backend, FT, R, u_mean, steps_per_R,
                               avg_divisor, drag_stride, wall_mode,
                               momentum_exchange_mode = force_mode,
                               geometry_mode, solvent_magic)
        catch err
            @warn "newtonian case failed" label R err
            nothing
        end
        dt = time() - t0
        if result === nothing
            @printf("%-10s %-20s %-4d %-18s %-18s %-16s %-12s %-12s %-12s %-10s %-10s %-10s %-10s %-9.1f\n",
                    "newtonian", label, R, String(wall_mode), String(force_mode),
                    "-", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "-", dt)
            continue
        end
        ref = wall_mode === :libb_v2 ? rheotool_newt : NaN
        err = isfinite(ref) ? (result.Cd - ref) / ref * 100 : NaN
        u_ref_result = hasproperty(result, :u_ref) ? result.u_ref : u_mean
        D_result = hasproperty(result, :D) ? result.D : 2R
        Cl = hasproperty(result, :Cl) ? result.Cl : 2 * result.Fy / (u_ref_result^2 * D_result)
        Re_R = hasproperty(result, :Re_R) ? result.Re_R : one(Float64)
        Re_D = hasproperty(result, :Re_D) ? result.Re_D : 2Re_R
        samples = hasproperty(result, :n_drag_samples) ? result.n_drag_samples : 0
        @printf("%-10s %-20s %-4d %-18s %-18s %-16s %-12.6f %-12.6g %-12.6f %-10.3f %-10.3f %-10.3f %-10d %-9.1f\n",
                "newtonian", label, R, String(wall_mode), String(force_mode),
                "-", result.Cd, Cl, ref, err, Re_R, Re_D, samples, dt)
        flush(stdout)
    end

    for (label, source_mode, hermite_mode, drag_mode, force_mode) in visco_cases
        visco_filter !== nothing && !(label in visco_filter) && continue
        ν_total_case = u_mean * R
        ν_s_case = beta * ν_total_case
        tau_plus_case = !isempty(tau_plus_override) ? parse(Float64, tau_plus_override) :
            (!isempty(sc_override) ? 0.5 + 3.0 * ν_s_case / parse(Float64, sc_override) : 1.0)
        t0 = time()
        result = try
            run_visco_case(; backend, FT, R, u_mean, beta, Wi, formulation,
                           steps_per_R, avg_divisor, drag_stride,
                           drag_mode, hermite_source_mode = hermite_mode,
                           solvent_source_mode = source_mode,
                           momentum_exchange_mode = force_mode,
                           geometry_mode, solvent_magic,
                           tau_plus = tau_plus_case,
                           conformation_magic,
                           conformation_collision)
        catch err
            @warn "visco case failed" label R err
            nothing
        end
        dt = time() - t0
        ref = get(liu_visco, R, rheotool_visco)
        if result === nothing
            @printf("%-10s %-20s %-4d %-18s %-18s %-16s %-12s %-12s %-12.6f %-10s %-10s %-10s %-10s %-9.1f\n",
                    "visco", label, R, String(source_mode), String(force_mode),
                    String(hermite_mode), "NaN", "NaN", ref, "NaN", "NaN", "NaN", "-", dt)
            continue
        end
        err = (result.Cd - ref) / ref * 100
        Cl = hasproperty(result, :Cl) ? result.Cl : 2 * result.Fy / (result.u_ref^2 * result.D)
        Re_R = hasproperty(result, :Re_R) ? result.Re_R : result.Re / 2
        Re_D = hasproperty(result, :Re_D) ? result.Re_D : result.Re
        @printf("%-10s %-20s %-4d %-18s %-18s %-16s %-12.6f %-12.6g %-12.6f %-10.3f %-10.3f %-10.3f %-10d %-9.1f\n",
                "visco", label, R, String(source_mode), String(force_mode),
                String(hermite_mode), result.Cd, Cl, ref, err, Re_R, Re_D,
                result.n_drag_samples, dt)
        flush(stdout)
    end
end

println("="^156)
println("Done.")
