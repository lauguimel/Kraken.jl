using Kraken
using Printf
using Statistics
using Dates
using KernelAbstractions

import Kraken: stream_periodic_x_wall_y_2d!,
               init_conformation_field_2d!, compute_conformation_macro_2d!,
               collide_conformation_2d!, collide_conformation_regularized_2d!,
               collide_conformation_liu_eq26_2d!, apply_cnebb_conformation_2d!

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

function _parse_list(::Type{T}, s::AbstractString) where {T}
    vals = strip(s)
    isempty(vals) && return T[]
    return [parse(T, strip(x)) for x in split(vals, ",") if !isempty(strip(x))]
end

function _parse_symbols(s::AbstractString)
    vals = strip(s)
    isempty(vals) && return Symbol[]
    return [Symbol(strip(x)) for x in split(vals, ",") if !isempty(strip(x))]
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
                return backend, Float32, "Metal Float32"
            end
        catch err
            requested == "metal" && rethrow(err)
        end
    end
    requested in ("cuda", "metal") &&
        error("requested backend '$requested' is not available")
    return CPU(), Float64, "CPU"
end

function _analytic_profiles(::Type{FT}, Ny::Int, u_mean, λ) where {FT}
    H = FT(Ny)
    u_max = FT(1.5) * FT(u_mean)
    ux = zeros(FT, Ny)
    Cxx = ones(FT, Ny)
    Cxy = zeros(FT, Ny)
    Cyy = ones(FT, Ny)
    shear = zeros(FT, Ny)
    for j in 1:Ny
        y = FT(j) - FT(0.5)
        ux[j] = FT(4) * u_max * y * (H - y) / (H * H)
        shear[j] = FT(4) * u_max * (H - FT(2) * y) / (H * H)
        Cxy[j] = FT(λ) * shear[j]
        Cxx[j] = one(FT) + FT(2) * (FT(λ) * shear[j])^2
    end
    return (; ux, Cxx, Cxy, Cyy, shear)
end

function _l2_rel(num, ref, idx)
    den = sum(abs2, ref[idx])
    den == 0 && return sqrt(sum(abs2, num[idx] .- ref[idx]))
    return sqrt(sum(abs2, num[idx] .- ref[idx]) / den)
end

function _max_rel(num, ref, idx)
    den = maximum(abs.(ref[idx]))
    den == 0 && return maximum(abs.(num[idx] .- ref[idx]))
    return maximum(abs.(num[idx] .- ref[idx])) / den
end

function _min_eig_c(Cxx, Cxy, Cyy)
    mineig = Inf
    @inbounds for k in eachindex(Cxx)
        tr = Cxx[k] + Cyy[k]
        diff = Cxx[k] - Cyy[k]
        disc = sqrt(diff * diff + 4 * Cxy[k] * Cxy[k])
        mineig = min(mineig, 0.5 * (tr - disc))
    end
    return mineig
end

function run_case(; backend, FT, Nx::Int, Ny::Int, u_mean::Float64,
                  beta::Float64, Wi::Float64, R::Float64, steps::Int,
                  tau_plus::Float64, conformation_magic::Float64,
                  collision::Symbol, phi_mode::Symbol,
                  bneq_source_scale::Float64, bneq_mass_scale::Float64,
                  bneq_second_moment::Symbol)
    λ = Wi * R / u_mean
    ν_total = u_mean * R
    ν_s = beta * ν_total
    Sc = ν_s / ((tau_plus - 0.5) / 3)
    ana = _analytic_profiles(FT, Ny, u_mean, λ)

    ux_h = repeat(reshape(ana.ux, 1, Ny), Nx, 1)
    uy_h = zeros(FT, Nx, Ny)
    Cxx_h = repeat(reshape(ana.Cxx, 1, Ny), Nx, 1)
    Cxy_h = repeat(reshape(ana.Cxy, 1, Ny), Nx, 1)
    Cyy_h = repeat(reshape(ana.Cyy, 1, Ny), Nx, 1)
    is_solid_h = fill(false, Nx, Ny)

    ux = KernelAbstractions.allocate(backend, FT, Nx, Ny); copyto!(ux, ux_h)
    uy = KernelAbstractions.allocate(backend, FT, Nx, Ny); copyto!(uy, uy_h)
    ρ = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(ρ, one(FT))
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny); copyto!(is_solid, is_solid_h)

    Cxx = KernelAbstractions.allocate(backend, FT, Nx, Ny); copyto!(Cxx, Cxx_h)
    Cxy = KernelAbstractions.allocate(backend, FT, Nx, Ny); copyto!(Cxy, Cxy_h)
    Cyy = KernelAbstractions.allocate(backend, FT, Nx, Ny); copyto!(Cyy, Cyy_h)

    g_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    init_conformation_field_2d!(g_xx, Cxx, ux, uy)
    init_conformation_field_2d!(g_xy, Cxy, ux, uy)
    init_conformation_field_2d!(g_yy, Cyy, ux, uy)
    g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_yy_buf = similar(g_yy)
    Fe_xx_prev = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    Fe_xy_prev = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    Fe_yy_prev = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)

    t0 = time()
    for _ in 1:steps
        stream_periodic_x_wall_y_2d!(g_xx_buf, g_xx, Nx, Ny)
        stream_periodic_x_wall_y_2d!(g_xy_buf, g_xy, Nx, Ny)
        stream_periodic_x_wall_y_2d!(g_yy_buf, g_yy, Nx, Ny)

        apply_cnebb_conformation_2d!(g_xx_buf, g_xx, is_solid, Cxx, ux, uy; phi_mode)
        apply_cnebb_conformation_2d!(g_xy_buf, g_xy, is_solid, Cxy, ux, uy; phi_mode)
        apply_cnebb_conformation_2d!(g_yy_buf, g_yy, is_solid, Cyy, ux, uy; phi_mode)

        g_xx, g_xx_buf = g_xx_buf, g_xx
        g_xy, g_xy_buf = g_xy_buf, g_xy
        g_yy, g_yy_buf = g_yy_buf, g_yy

        compute_conformation_macro_2d!(Cxx, g_xx)
        compute_conformation_macro_2d!(Cxy, g_xy)
        compute_conformation_macro_2d!(Cyy, g_yy)

        if collision === :trt
            collide_conformation_2d!(g_xx, Cxx, ux, uy, Cxx, Cxy, Cyy, is_solid,
                                      FT(tau_plus), FT(λ); magic=FT(conformation_magic), component=1)
            collide_conformation_2d!(g_xy, Cxy, ux, uy, Cxx, Cxy, Cyy, is_solid,
                                      FT(tau_plus), FT(λ); magic=FT(conformation_magic), component=2)
            collide_conformation_2d!(g_yy, Cyy, ux, uy, Cxx, Cxy, Cyy, is_solid,
                                      FT(tau_plus), FT(λ); magic=FT(conformation_magic), component=3)
        elseif collision === :regularized
            collide_conformation_regularized_2d!(g_xx, Cxx, ux, uy, Cxx, Cxy, Cyy, is_solid,
                                                  FT(tau_plus), FT(λ); magic=FT(conformation_magic), component=1)
            collide_conformation_regularized_2d!(g_xy, Cxy, ux, uy, Cxx, Cxy, Cyy, is_solid,
                                                  FT(tau_plus), FT(λ); magic=FT(conformation_magic), component=2)
            collide_conformation_regularized_2d!(g_yy, Cyy, ux, uy, Cxx, Cxy, Cyy, is_solid,
                                                  FT(tau_plus), FT(λ); magic=FT(conformation_magic), component=3)
        elseif collision === :liu_eq26
            raw_second = bneq_second_moment === :raw
            collide_conformation_liu_eq26_2d!(g_xx, Fe_xx_prev, Cxx, ux, uy, ρ,
                                               Cxx, Cxy, Cyy, is_solid, FT(tau_plus), FT(λ);
                                               magic=FT(conformation_magic),
                                               bneq_source_scale=FT(bneq_source_scale),
                                               bneq_mass_scale=FT(bneq_mass_scale),
                                               bneq_second_moment_raw=raw_second,
                                               component=1)
            collide_conformation_liu_eq26_2d!(g_xy, Fe_xy_prev, Cxy, ux, uy, ρ,
                                               Cxx, Cxy, Cyy, is_solid, FT(tau_plus), FT(λ);
                                               magic=FT(conformation_magic),
                                               bneq_source_scale=FT(bneq_source_scale),
                                               bneq_mass_scale=FT(bneq_mass_scale),
                                               bneq_second_moment_raw=raw_second,
                                               component=2)
            collide_conformation_liu_eq26_2d!(g_yy, Fe_yy_prev, Cyy, ux, uy, ρ,
                                               Cxx, Cxy, Cyy, is_solid, FT(tau_plus), FT(λ);
                                               magic=FT(conformation_magic),
                                               bneq_source_scale=FT(bneq_source_scale),
                                               bneq_mass_scale=FT(bneq_mass_scale),
                                               bneq_second_moment_raw=raw_second,
                                               component=3)
        else
            error("unknown collision $collision")
        end
    end
    elapsed = time() - t0

    Cxx_profile = vec(mean(Array(Cxx), dims=1))
    Cxy_profile = vec(mean(Array(Cxy), dims=1))
    Cyy_profile = vec(mean(Array(Cyy), dims=1))
    N1_profile = Cxx_profile .- Cyy_profile
    N1_ref = ana.Cxx .- ana.Cyy

    all_idx = 1:Ny
    core_idx = max(1, Ny ÷ 8):min(Ny, Ny - Ny ÷ 8 + 1)
    wall_idx = [1, 2, Ny - 1, Ny]

    return (;
        Nx, Ny, steps, collision, phi_mode, tau_plus, Sc, conformation_magic,
        Cxy_l2 = _l2_rel(Cxy_profile, Float64.(ana.Cxy), all_idx),
        Cxy_core_l2 = _l2_rel(Cxy_profile, Float64.(ana.Cxy), core_idx),
        Cxy_max_rel = _max_rel(Cxy_profile, Float64.(ana.Cxy), all_idx),
        N1_l2 = _l2_rel(N1_profile, Float64.(N1_ref), all_idx),
        N1_core_l2 = _l2_rel(N1_profile, Float64.(N1_ref), core_idx),
        N1_max_rel = _max_rel(N1_profile, Float64.(N1_ref), all_idx),
        wall_Cxy_max_abs = maximum(abs.(Cxy_profile[wall_idx] .- Float64.(ana.Cxy[wall_idx]))),
        wall_N1_max_abs = maximum(abs.(N1_profile[wall_idx] .- Float64.(N1_ref[wall_idx]))),
        min_eig = _min_eig_c(Cxx_profile, Cxy_profile, Cyy_profile),
        elapsed)
end

backend, FT, backend_label = _select_backend()
R = parse(Float64, get(ENV, "KRAKEN_R", "30"))
Ny = parse(Int, get(ENV, "KRAKEN_NY", string(Int(4R))))
Nx = parse(Int, get(ENV, "KRAKEN_NX", "16"))
u_mean = parse(Float64, get(ENV, "KRAKEN_U_MEAN", "0.005"))
beta = parse(Float64, get(ENV, "KRAKEN_BETA", "0.59"))
Wi = parse(Float64, get(ENV, "KRAKEN_WI", "0.1"))
steps = parse(Int, get(ENV, "KRAKEN_STEPS", string(Int(4000R))))
collisions = _parse_symbols(get(ENV, "KRAKEN_COLLISIONS", "trt,liu_eq26"))
phi_modes = _parse_symbols(get(ENV, "KRAKEN_CNEBB_PHI_MODES", "pre_opp"))
conformation_magic = parse(Float64, get(ENV, "KRAKEN_CONFORMATION_MAGIC", "2.5e-7"))
bneq_source_scale = parse(Float64, get(ENV, "KRAKEN_BNEQ_SOURCE_SCALE", "0.0"))
bneq_mass_scale = parse(Float64, get(ENV, "KRAKEN_BNEQ_MASS_SCALE", "1.0"))
bneq_second_moment = Symbol(get(ENV, "KRAKEN_BNEQ_SECOND_MOMENT", "hermite"))

ν_s = beta * u_mean * R
tau_values = if haskey(ENV, "KRAKEN_TAU_PLUS_LIST")
    _parse_list(Float64, ENV["KRAKEN_TAU_PLUS_LIST"])
elseif haskey(ENV, "KRAKEN_SC_LIST")
    [0.5 + 3ν_s / Sc for Sc in _parse_list(Float64, ENV["KRAKEN_SC_LIST"])]
else
    [1.0, 0.5 + 3ν_s / 1e4]
end

println("="^150)
println("Poiseuille high-Sc CDE isolation")
println("Date/time: $(Dates.now())")
println("Backend: $backend_label, FT=$FT")
@printf("Nx=%d Ny=%d R=%.6g u_mean=%.6g beta=%.6g Wi=%.6g steps=%d Λp=%.6g\n",
        Nx, Ny, R, u_mean, beta, Wi, steps, conformation_magic)
println("Collisions: $(join(string.(collisions), ", "))")
println("CNEBB phi modes: $(join(string.(phi_modes), ", "))")
println("="^150)
@printf("%-12s %-9s %-13s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-10s\n",
        "collision", "phi", "tau_plus", "Sc", "Cxy_l2", "Cxy_core",
        "Cxy_max", "N1_l2", "N1_core", "N1_max", "wall_Cxy", "min_eig", "time")
println("-"^150)

for tau_plus in tau_values
    for phi_mode in phi_modes
        for collision in collisions
            result = run_case(; backend, FT, Nx, Ny, u_mean, beta, Wi, R, steps,
                              tau_plus, conformation_magic, collision, phi_mode,
                              bneq_source_scale, bneq_mass_scale,
                              bneq_second_moment)
            @printf("%-12s %-9s %-13.8g %-12.5g %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e %-10.1f\n",
                    string(collision), string(phi_mode), result.tau_plus, result.Sc,
                    result.Cxy_l2, result.Cxy_core_l2, result.Cxy_max_rel,
                    result.N1_l2, result.N1_core_l2, result.N1_max_rel,
                    result.wall_Cxy_max_abs, result.min_eig, result.elapsed)
            flush(stdout)
        end
    end
end

println("="^150)
println("Done.")
