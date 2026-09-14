# Bulk constitutive checks for Oldroyd-B conformation kernels.
#
# Purpose: validate the constitutive update before step geometries, walls,
# polymer feedback, force accounting, or Cd. These checks prescribe an
# analytical velocity gradient and compare Cxx/Cxy/Cyy against exact
# Oldroyd-B steady states.
#
# Usage examples:
#   julia --project=. hpc/bulk_constitutive.jl
#   KRAKEN_BACKEND=metal KRAKEN_BULK_CASES=shear,elongation,poiseuille \
#     KRAKEN_MODELS=direct,logconf KRAKEN_WI_LIST=0.05,0.1,0.2 \
#     julia --project=. hpc/bulk_constitutive.jl

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

function _to_backend(backend, ::Type{T}, host::AbstractArray) where {T}
    device = KernelAbstractions.allocate(backend, T, size(host)...)
    copyto!(device, T.(host))
    return device
end

function _to_backend_bool(backend, host::AbstractArray{Bool})
    device = KernelAbstractions.allocate(backend, Bool, size(host)...)
    copyto!(device, Matrix{Bool}(host))
    return device
end

function _velocity_exact(case_name::Symbol, ::Type{FT}, Nx, Ny, λ, Wi) where {FT}
    ux = zeros(FT, Nx, Ny)
    uy = zeros(FT, Nx, Ny)
    Cxx = ones(FT, Nx, Ny)
    Cxy = zeros(FT, Nx, Ny)
    Cyy = ones(FT, Nx, Ny)

    if case_name === :shear
        γdot = FT(Wi / λ)
        @inbounds for j in 1:Ny, i in 1:Nx
            ux[i, j] = γdot * (FT(j) - FT(0.5))
            Cxy[i, j] = FT(Wi)
            Cxx[i, j] = FT(1) + FT(2) * FT(Wi)^2
        end
        return (; ux, uy, Cxx, Cxy, Cyy, parameter=Float64(γdot), Wi_actual=Float64(Wi))
    elseif case_name === :elongation
        Wi < 0.5 || error("elongation requires Wi < 0.5 for finite Oldroyd-B steady state")
        εdot = FT(Wi / λ)
        x0 = FT(Nx - 1) / FT(2)
        y0 = FT(Ny - 1) / FT(2)
        cxx = FT(1) / (FT(1) - FT(2) * FT(Wi))
        cyy = FT(1) / (FT(1) + FT(2) * FT(Wi))
        @inbounds for j in 1:Ny, i in 1:Nx
            ux[i, j] = εdot * ((FT(i) - FT(1)) - x0)
            uy[i, j] = -εdot * ((FT(j) - FT(1)) - y0)
            Cxx[i, j] = cxx
            Cyy[i, j] = cyy
        end
        return (; ux, uy, Cxx, Cxy, Cyy, parameter=Float64(εdot), Wi_actual=Float64(Wi))
    elseif case_name === :poiseuille
        H = FT(Ny)
        max_base_shear = zero(FT)
        for j in 1:Ny
            y = FT(j) - FT(0.5)
            max_base_shear = max(max_base_shear, abs(FT(4) * (H - FT(2) * y) / (H * H)))
        end
        u_max = FT(Wi / λ) / max_base_shear
        max_shear = zero(FT)
        @inbounds for j in 1:Ny, i in 1:Nx
            y = FT(j) - FT(0.5)
            ux[i, j] = FT(4) * u_max * y * (H - y) / (H * H)
            dudy = FT(4) * u_max * (H - FT(2) * y) / (H * H)
            Cxy[i, j] = FT(λ) * dudy
            Cxx[i, j] = FT(1) + FT(2) * (FT(λ) * dudy)^2
            max_shear = max(max_shear, abs(dudy))
        end
        return (; ux, uy, Cxx, Cxy, Cyy, parameter=Float64(u_max),
                Wi_actual=Float64(λ) * Float64(max_shear))
    else
        error("unknown bulk case $case_name; expected shear, elongation, or poiseuille")
    end
end

function _min_eig_C(Cxx, Cxy, Cyy)
    min_eig = Inf
    @inbounds for k in eachindex(Cxx)
        cxx = Float64(Cxx[k])
        cxy = Float64(Cxy[k])
        cyy = Float64(Cyy[k])
        tr = cxx + cyy
        diff = cxx - cyy
        disc = sqrt(diff * diff + 4.0 * cxy * cxy)
        min_eig = min(min_eig, 0.5 * (tr - disc))
    end
    return min_eig
end

function _max_relative_error(num, exact, mask)
    err = 0.0
    @inbounds for k in eachindex(num)
        mask[k] || continue
        denom = max(abs(Float64(exact[k])), 1e-12)
        err = max(err, abs(Float64(num[k]) - Float64(exact[k])) / denom)
    end
    return err
end

function _bulk_mask(Nx, Ny, skip)
    mask = falses(Nx, Ny)
    ilo = min(max(skip + 1, 1), Nx)
    ihi = max(min(Nx - skip, Nx), ilo)
    jlo = min(max(skip + 1, 1), Ny)
    jhi = max(min(Ny - skip, Ny), jlo)
    mask[ilo:ihi, jlo:jhi] .= true
    return mask
end

function _run_case(case_name, model_name, backend, ::Type{FT};
                   Nx, Ny, λ, Wi, tau_plus, magic, steps, skip) where {FT}
    exact = _velocity_exact(case_name, FT, Nx, Ny, λ, Wi)
    ux = _to_backend(backend, FT, exact.ux)
    uy = _to_backend(backend, FT, exact.uy)
    is_solid = _to_backend_bool(backend, falses(Nx, Ny))

    Cxx = _to_backend(backend, FT, ones(FT, Nx, Ny))
    Cxy = _to_backend(backend, FT, zeros(FT, Nx, Ny))
    Cyy = _to_backend(backend, FT, ones(FT, Nx, Ny))

    if model_name === :direct
        Φxx = Cxx
        Φxy = Cxy
        Φyy = Cyy
    elseif model_name === :logconf
        Φxx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        Φxy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        Φyy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    else
        error("unknown model $model_name; expected direct or logconf")
    end

    g_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    init_conformation_field_2d!(g_xx, Φxx, ux, uy)
    init_conformation_field_2d!(g_xy, Φxy, ux, uy)
    init_conformation_field_2d!(g_yy, Φyy, ux, uy)

    for _ in 1:steps
        compute_conformation_macro_2d!(Φxx, g_xx)
        compute_conformation_macro_2d!(Φxy, g_xy)
        compute_conformation_macro_2d!(Φyy, g_yy)
        if model_name === :direct
            collide_conformation_2d!(g_xx, Φxx, ux, uy, Φxx, Φxy, Φyy, is_solid, tau_plus, λ; magic=magic, component=1)
            collide_conformation_2d!(g_xy, Φxy, ux, uy, Φxx, Φxy, Φyy, is_solid, tau_plus, λ; magic=magic, component=2)
            collide_conformation_2d!(g_yy, Φyy, ux, uy, Φxx, Φxy, Φyy, is_solid, tau_plus, λ; magic=magic, component=3)
        else
            collide_logconf_2d!(g_xx, Φxx, ux, uy, Φxx, Φxy, Φyy, is_solid, tau_plus, λ; magic=magic, component=1)
            collide_logconf_2d!(g_xy, Φxy, ux, uy, Φxx, Φxy, Φyy, is_solid, tau_plus, λ; magic=magic, component=2)
            collide_logconf_2d!(g_yy, Φyy, ux, uy, Φxx, Φxy, Φyy, is_solid, tau_plus, λ; magic=magic, component=3)
        end
    end
    compute_conformation_macro_2d!(Φxx, g_xx)
    compute_conformation_macro_2d!(Φxy, g_xy)
    compute_conformation_macro_2d!(Φyy, g_yy)

    if model_name === :direct
        Cxx = Φxx
        Cxy = Φxy
        Cyy = Φyy
    else
        psi_to_C_2d!(Cxx, Cxy, Cyy, Φxx, Φxy, Φyy)
    end

    Cxx_h = Array(Cxx)
    Cxy_h = Array(Cxy)
    Cyy_h = Array(Cyy)
    mask = _bulk_mask(Nx, Ny, skip)
    return (;
        case_name, model_name, Wi_target=Wi, Wi_actual=exact.Wi_actual,
        parameter=exact.parameter,
        err_Cxx=_max_relative_error(Cxx_h, exact.Cxx, mask),
        err_Cxy=_max_relative_error(Cxy_h, exact.Cxy, mask),
        err_Cyy=_max_relative_error(Cyy_h, exact.Cyy, mask),
        min_eig_C=_min_eig_C(Cxx_h, Cxy_h, Cyy_h),
        max_trace_C=maximum(Cxx_h .+ Cyy_h))
end

backend, FT, backend_label = _select_backend()

cases = _parse_symbols(get(ENV, "KRAKEN_BULK_CASES", "shear,elongation,poiseuille"))
models = _parse_symbols(get(ENV, "KRAKEN_MODELS", "direct,logconf"))
Wi_values = _parse_list(Float64, get(ENV, "KRAKEN_WI_LIST", "0.05,0.1,0.2"))
Nx = parse(Int, get(ENV, "KRAKEN_NX", "24"))
Ny = parse(Int, get(ENV, "KRAKEN_NY", "32"))
λ = parse(Float64, get(ENV, "KRAKEN_LAMBDA", "10.0"))
tau_plus = parse(Float64, get(ENV, "KRAKEN_TAU_PLUS", "1.0"))
magic = parse(Float64, get(ENV, "KRAKEN_MAGIC", "1e-6"))
steps = parse(Int, get(ENV, "KRAKEN_STEPS", "2000"))
skip = parse(Int, get(ENV, "KRAKEN_BULK_SKIP", "2"))
rtol = parse(Float64, get(ENV, "KRAKEN_BULK_RTOL", "0.02"))
results_dir = get(ENV, "KRAKEN_RESULTS_DIR",
    joinpath("tmp", "bulk_constitutive_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
mkpath(results_dir)
csv_path = joinpath(results_dir, "bulk_constitutive.csv")

println("="^150)
println("Bulk constitutive checks")
println("Date/time: $(Dates.now())")
println("Backend: $backend_label, FT=$FT")
@printf("Nx=%d Ny=%d lambda=%.6g tau_plus=%.6g magic=%.6g steps=%d skip=%d rtol=%.4g\n",
    Nx, Ny, λ, tau_plus, magic, steps, skip, rtol)
println("Cases: $(join(string.(cases), ", "))")
println("Models: $(join(string.(models), ", "))")
println("Wi values: $(join(Wi_values, ", "))")
println("CSV: $csv_path")
println("="^150)

header = [
    "case", "model", "Wi_target", "Wi_actual", "parameter",
    "err_Cxx", "err_Cxy", "err_Cyy", "min_eig_C", "max_trace_C", "status"
]

failures = Ref(0)
open(csv_path, "w") do io
    println(io, join(header, ","))
    @printf("%-12s %-8s %-8s %-8s %-11s %-11s %-11s %-11s %-11s %-11s %-8s\n",
        "case", "model", "Wi", "Wi_act", "param", "err_Cxx", "err_Cxy", "err_Cyy", "minEig", "traceC", "status")
    println("-"^150)
    for case_name in cases, model_name in models, Wi in Wi_values
        result = _run_case(case_name, model_name, backend, FT;
            Nx, Ny, λ, Wi, tau_plus, magic, steps, skip)
        status = result.min_eig_C > 0 &&
                 result.err_Cxx <= rtol &&
                 result.err_Cxy <= rtol &&
                 result.err_Cyy <= rtol ? "PASS" : "FAIL"
        failures[] += status == "FAIL" ? 1 : 0
        row = Any[
            result.case_name, result.model_name, result.Wi_target, result.Wi_actual,
            result.parameter, result.err_Cxx, result.err_Cxy, result.err_Cyy,
            result.min_eig_C, result.max_trace_C, status
        ]
        println(io, join(string.(row), ","))
        @printf("%-12s %-8s %-8.3g %-8.3g %-11.4g %-11.4g %-11.4g %-11.4g %-11.4g %-11.4g %-8s\n",
            string(result.case_name), string(result.model_name), result.Wi_target,
            result.Wi_actual, result.parameter, result.err_Cxx, result.err_Cxy,
            result.err_Cyy, result.min_eig_C, result.max_trace_C, status)
        flush(stdout)
    end
end

println("="^150)
println("Done. Inspect: $csv_path")
failures[] == 0 || exit(1)
