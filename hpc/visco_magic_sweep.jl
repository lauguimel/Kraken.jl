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

function run_case(; backend, FT, R, formulation, conformation_magic,
                  u_mean, beta, Wi, steps_per_R, avg_divisor, drag_stride,
                  drag_mode, hermite_source_mode, solvent_source_mode,
                  momentum_exchange_mode)
    nu_total = u_mean * R
    nu_s = beta * nu_total
    nu_p = (1 - beta) * nu_total
    λ = Wi * R / u_mean
    G = nu_p / λ
    model = formulation == "logconf" ?
        LogConfOldroydB(G = FT(G), λ = FT(λ)) :
        OldroydB(G = FT(G), λ = FT(λ))
    max_steps = steps_per_R * R
    avg_window = max(1, max_steps ÷ avg_divisor)
    return run_conformation_cylinder_libb_2d(;
        Nx = 30R, Ny = 4R, radius = R, cx = 15R, cy = 2R,
        u_mean = FT(u_mean), ν_s = FT(nu_s),
        polymer_model = model, polymer_bc = CNEBB(),
        inlet = :parabolic, ρ_out = one(FT), tau_plus = one(FT),
        max_steps, avg_window, drag_stride,
        drag_mode, hermite_source_mode, conformation_magic,
        solvent_source_mode, momentum_exchange_mode,
        allow_diagnostic_log_wall_bc = formulation == "logconf",
        backend, FT)
end

backend, FT, backend_label = _select_backend()
R_values = _parse_list(Int, get(ENV, "KRAKEN_R_LIST", "30"))
formulations = split(get(ENV, "KRAKEN_FORMULATIONS", "direct"), ',')
formulations = [strip(x) for x in formulations if !isempty(strip(x))]
magic_values = _parse_list(Float64, get(ENV, "KRAKEN_MAGIC_LIST", "0.000001,0.0001,0.01,0.25"))

u_mean = parse(Float64, get(ENV, "KRAKEN_U_MEAN", "0.005"))
beta = parse(Float64, get(ENV, "KRAKEN_BETA", "0.59"))
Wi = parse(Float64, get(ENV, "KRAKEN_WI", "0.1"))
steps_per_R = parse(Int, get(ENV, "KRAKEN_STEPS_PER_R", "4000"))
avg_divisor = parse(Int, get(ENV, "KRAKEN_AVG_DIVISOR", "10"))
drag_stride = parse(Int, get(ENV, "KRAKEN_DRAG_STRIDE", "200"))
drag_mode = Symbol(get(ENV, "KRAKEN_DRAG_MODE", "post_source_mea"))
hermite_source_mode = Symbol(get(ENV, "KRAKEN_HERMITE_SOURCE_MODE", "liu_direct"))
solvent_source_mode = Symbol(get(ENV, "KRAKEN_SOLVENT_SOURCE_MODE", "post_collision"))
momentum_exchange_mode = Symbol(get(ENV, "KRAKEN_MOMENTUM_EXCHANGE_MODE", "mei_reconstruct"))

liu_visco = Dict(20 => 129.42, 25 => 129.61, 30 => 130.36,
                 35 => 130.77, 40 => 130.79, 48 => 130.83)

println("="^128)
println("Viscoelastic conformation TRT magic sweep")
println("Date/time: $(Dates.now())")
println("Backend: $backend_label, FT=$FT")
println("R values: $(join(R_values, ", "))")
println("Formulations: $(join(formulations, ", "))")
println("Magic values: $(join(magic_values, ", "))")
println("Drag mode: $(String(drag_mode))")
println("Hermite source mode: $(String(hermite_source_mode))")
println("Solvent source mode: $(String(solvent_source_mode))")
println("Momentum exchange mode: $(String(momentum_exchange_mode))")
println("="^128)
@printf("%-8s %-4s %-12s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-9s\n",
        "case", "R", "magic", "Cd_ref", "Cd", "err_ref%",
        "Cd_s", "Cd_post", "Cd_scaled", "Cd_split", "time")
println("-"^128)

for R in R_values
    ref = get(liu_visco, R, NaN)
    for formulation in formulations
        for magic in magic_values
            t0 = time()
            result = run_case(; backend, FT, R, formulation,
                              conformation_magic = magic,
                              u_mean, beta, Wi, steps_per_R, avg_divisor,
                              drag_stride, drag_mode, hermite_source_mode,
                              solvent_source_mode, momentum_exchange_mode)
            dt = time() - t0
            err = isfinite(ref) ? (result.Cd - ref) / ref * 100 : NaN
            @printf("%-8s %-4d %-12.6g %-10.4f %-10.4f %-10.3f %-10.4f %-10.4f %-10.4f %-10.4f %-9.1f\n",
                    formulation, R, magic, ref, result.Cd, err,
                    result.Cd_s, result.Cd_mea_post_source,
                    result.Cd_mea_source_scaled, result.Cd_split_explicit, dt)
            flush(stdout)
        end
    end
end

println("="^128)
println("Done.")
