include(joinpath(@__DIR__, "..", "src", "Kraken.jl"))
using .Kraken
using Printf
using Statistics
using KernelAbstractions
using Dates

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
    if requested in ("auto", "cuda")
        if _CUDA_MOD !== nothing
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
    end
    if requested in ("auto", "metal")
        if _METAL_MOD !== nothing
            try
                if Base.invokelatest(getfield(_METAL_MOD, :functional))
                    backend = Base.invokelatest(getfield(_METAL_MOD, :MetalBackend))
                    return backend, Float32, "Metal"
                end
            catch err
                requested == "metal" && rethrow(err)
            end
        end
    end
    return KernelAbstractions.CPU(), Float64, "CPU"
end

function _fluid_density_stats(result)
    fluid = .!result.is_solid
    rho = result.ρ[fluid]
    return mean(rho), minimum(rho), maximum(rho)
end

function _run_newtonian(; backend, FT, R, u_mean, max_steps, avg_window,
                        drag_stride, momentum_exchange_mode)
    nu = u_mean * R
    return run_cylinder_libb_2d(;
        Nx = 30R, Ny = 4R, radius = R, cx = 15R, cy = 2R,
        u_in = FT(1.5 * u_mean), ν = FT(nu), inlet = :parabolic,
        max_steps = max_steps, avg_window = avg_window,
        drag_stride = drag_stride,
        momentum_exchange_mode = momentum_exchange_mode,
        backend = backend, T = FT)
end

function _run_visco(; backend, FT, R, u_mean, beta, Wi, formulation, drag_mode,
                    hermite_source_mode, solvent_source_mode, conformation_magic,
                    momentum_exchange_mode, max_steps, avg_window, drag_stride)
    nu_total = u_mean * R
    nu_s = beta * nu_total
    nu_p = (1 - beta) * nu_total
    lambda = Wi * R / u_mean
    G = FT(nu_p / lambda)
    lambda_ft = FT(lambda)
    model = formulation == "logconf" ?
        LogConfOldroydB(G = G, λ = lambda_ft) :
        OldroydB(G = G, λ = lambda_ft)

    return run_conformation_cylinder_libb_2d(;
        Nx = 30R, Ny = 4R, radius = R, cx = 15R, cy = 2R,
        u_mean = FT(u_mean), ν_s = FT(nu_s),
        polymer_model = model, polymer_bc = CNEBB(),
        inlet = :parabolic, ρ_out = one(FT), tau_plus = one(FT),
        max_steps = max_steps, avg_window = avg_window,
        drag_stride = drag_stride, drag_mode = drag_mode,
        hermite_source_mode = hermite_source_mode,
        solvent_source_mode = solvent_source_mode,
        conformation_magic = conformation_magic,
        momentum_exchange_mode = momentum_exchange_mode,
        allow_diagnostic_log_wall_bc = formulation == "logconf",
        backend = backend, FT = FT)
end

backend, FT, backend_label = _select_backend()

R_values = _parse_list(Int, get(ENV, "KRAKEN_R_LIST", "20"))
formulations = split(get(ENV, "KRAKEN_FORMULATIONS", "newtonian,direct"), ',')
formulations = [strip(x) for x in formulations if !isempty(strip(x))]

u_mean = parse(Float64, get(ENV, "KRAKEN_U_MEAN", "0.005"))
beta = parse(Float64, get(ENV, "KRAKEN_BETA", "0.59"))
Wi = parse(Float64, get(ENV, "KRAKEN_WI", "0.1"))
steps_per_R = parse(Int, get(ENV, "KRAKEN_STEPS_PER_R", "2000"))
avg_divisor = parse(Int, get(ENV, "KRAKEN_AVG_DIVISOR", "10"))
drag_stride = parse(Int, get(ENV, "KRAKEN_DRAG_STRIDE", "100"))
drag_mode = Symbol(get(ENV, "KRAKEN_DRAG_MODE", "post_source_mea"))
hermite_source_mode = Symbol(get(ENV, "KRAKEN_HERMITE_SOURCE_MODE", "liu_direct"))
solvent_source_mode = Symbol(get(ENV, "KRAKEN_SOLVENT_SOURCE_MODE", "post_collision"))
conformation_magic = parse(Float64, get(ENV, "KRAKEN_CONFORMATION_MAGIC", "1e-6"))
momentum_exchange_mode = Symbol(get(ENV, "KRAKEN_MOMENTUM_EXCHANGE_MODE", "mei_reconstruct"))

rheotool_newt = 132.362236515
rheotool_visco = 130.428774404
liu_visco = Dict(20 => 129.42, 25 => 129.61, 30 => 130.36,
                 35 => 130.77, 40 => 130.79, 48 => 130.83)

println("="^96)
println("Low-Mach viscoelastic cylinder validation")
println("Date/time: $(Dates.now())")
println("Backend: $backend_label, FT=$FT")
@printf("u_mean=%.6g  beta=%.6g  Wi_R=%.6g  steps_per_R=%d  avg_divisor=%d  drag_stride=%d  drag_mode=%s  hermite_source=%s  solvent_source=%s  conformation_magic=%.6g  momentum_exchange=%s\n",
        u_mean, beta, Wi, steps_per_R, avg_divisor, drag_stride,
        String(drag_mode), String(hermite_source_mode), String(solvent_source_mode), conformation_magic,
        String(momentum_exchange_mode))
println("R values: $(join(R_values, ", "))")
println("Formulations: $(join(formulations, ", "))")
println("="^96)
@printf("%-10s %-5s %-9s %-8s %-12s %-12s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-8s\n",
        "case", "R", "steps", "samples", "Cd", "Cd_ref", "err_ref%",
        "rho_mean", "rho_max", "Cd_s", "Cd_p", "Cd_post", "Cd_scaled", "Cd_split", "time")
println("-"^96)

for R in R_values
    max_steps = steps_per_R * R
    avg_window = max(1, max_steps ÷ avg_divisor)
    for formulation in formulations
        t0 = time()
        result = try
            if formulation == "newtonian"
                _run_newtonian(; backend, FT, R, u_mean, max_steps, avg_window,
                               drag_stride, momentum_exchange_mode)
            elseif formulation in ("direct", "logconf")
                _run_visco(; backend, FT, R, u_mean, beta, Wi, formulation,
                           drag_mode, hermite_source_mode, solvent_source_mode, conformation_magic,
                           momentum_exchange_mode,
                           max_steps, avg_window, drag_stride)
            else
                error("unknown formulation '$formulation'")
            end
        catch err
            @warn "case failed" R formulation err
            nothing
        end
        dt = time() - t0
        if result === nothing
            @printf("%-10s %-5d %-9d %-8s %-12s %-12s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-8.1f\n",
                    formulation, R, max_steps, "-", "NaN", "NaN", "NaN",
                    "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", dt)
            flush(stdout)
            continue
        end

        rho_mean, _, rho_max = _fluid_density_stats(result)
        ref = formulation == "newtonian" ? rheotool_newt :
              get(liu_visco, R, rheotool_visco)
        err = (result.Cd - ref) / ref * 100
        Cd_s = hasproperty(result, :Cd_s) ? result.Cd_s : result.Cd
        Cd_p = hasproperty(result, :Cd_p) ? result.Cd_p : NaN
        Cd_post = hasproperty(result, :Cd_mea_post_source) ? result.Cd_mea_post_source : NaN
        Cd_scaled = hasproperty(result, :Cd_mea_source_scaled) ? result.Cd_mea_source_scaled : NaN
        Cd_split = hasproperty(result, :Cd_split_explicit) ? result.Cd_split_explicit : NaN
        samples = hasproperty(result, :n_drag_samples) ? result.n_drag_samples : 0
        @printf("%-10s %-5d %-9d %-8d %-12.6f %-12.6f %-10.3f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-8.1f\n",
                formulation, R, max_steps, samples, result.Cd, ref, err,
                rho_mean, rho_max, Cd_s, Cd_p, Cd_post, Cd_scaled, Cd_split, dt)
        flush(stdout)
    end
end

println("="^96)
println("Done.")
