# Frozen-velocity open-channel CDE canary.
#
# This is the surgical control for `cylinder_frozen_velocity_cde_patch.jl`:
# same stream/collide/reset pipeline and same open inlet/outlet treatment, but
# with a fully developed planar Poiseuille velocity field and the exact
# Oldroyd-B stationary conformation profile. If this drifts or loses SPD, the
# open-boundary CDE pipeline is broken before any curved-wall/cylinder physics.

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
    requested = lowercase(get(ENV, "KRAKEN_BACKEND", "cpu"))
    if requested in ("auto", "cuda") && _CUDA_MOD !== nothing
        try
            if Base.invokelatest(getfield(_CUDA_MOD, :functional))
                return Base.invokelatest(getfield(_CUDA_MOD, :CUDABackend)),
                       Float64, "CUDA"
            end
        catch err
            requested == "cuda" && rethrow(err)
        end
    end
    if requested in ("auto", "metal") && _METAL_MOD !== nothing
        try
            if Base.invokelatest(getfield(_METAL_MOD, :functional))
                return Base.invokelatest(getfield(_METAL_MOD, :MetalBackend)),
                       Float32, "Metal"
            end
        catch err
            requested == "metal" && rethrow(err)
        end
    end
    return KernelAbstractions.CPU(), Float64, "CPU"
end

_parse_list(::Type{T}, raw::AbstractString) where {T} =
    [parse(T, strip(x)) for x in split(raw, ',') if !isempty(strip(x))]

_parse_symbols(raw::AbstractString) =
    [Symbol(strip(x)) for x in split(raw, ',') if !isempty(strip(x))]

function _poiseuille_profile(::Type{FT}, Ny, u_mean, λ, use_logconf) where {FT}
    H = FT(Ny)
    u_max = FT(1.5) * FT(u_mean)
    ux = zeros(FT, Ny)
    Cxx = ones(FT, Ny)
    Cxy = zeros(FT, Ny)
    Cyy = ones(FT, Ny)
    for j in 1:Ny
        y = FT(j) - FT(0.5)
        ux[j] = FT(4) * u_max * y * (H - y) / (H * H)
        dudy = FT(4) * u_max * (H - FT(2) * y) / (H * H)
        Cxy[j] = FT(λ) * dudy
        Cxx[j] = one(FT) + FT(2) * (FT(λ) * dudy)^2
    end
    if use_logconf
        for j in 1:Ny
            tr = Cxx[j] + Cyy[j]
            diff = Cxx[j] - Cyy[j]
            disc = sqrt(diff * diff + FT(4) * Cxy[j] * Cxy[j])
            μ1 = FT(0.5) * (tr + disc)
            μ2 = FT(0.5) * (tr - disc)
            l1 = log(max(μ1, FT(1e-30)))
            l2 = log(max(μ2, FT(1e-30)))
            θ = FT(0.5) * atan(FT(2) * Cxy[j], diff)
            c = cos(θ)
            s = sin(θ)
            Cxx[j] = c * c * l1 + s * s * l2
            Cxy[j] = c * s * (l1 - l2)
            Cyy[j] = s * s * l1 + c * c * l2
        end
    end
    return ux, Cxx, Cxy, Cyy
end

function _run_case(; backend, FT, Nx, Ny, R, beta, Wi, model_name,
                   u_mean, steps, tau_plus, magic, divergence_mode,
                   diagnostic_interval, io)
    ν_total = u_mean * R
    ν_p = (1 - beta) * ν_total
    λ = Wi * R / u_mean
    G = ν_p / λ
    use_logconf = model_name === :logconf

    ux_prof, Cxx_prof, Cxy_prof, Cyy_prof =
        _poiseuille_profile(FT, Ny, u_mean, λ, use_logconf)
    ux_h = repeat(reshape(ux_prof, 1, Ny), Nx, 1)
    uy_h = zeros(FT, Nx, Ny)
    ρ_h = ones(FT, Nx, Ny)
    solid_h = fill(false, Nx, Ny)

    ux = KernelAbstractions.allocate(backend, FT, Nx, Ny); copyto!(ux, ux_h)
    uy = KernelAbstractions.allocate(backend, FT, Nx, Ny); copyto!(uy, uy_h)
    ρ = KernelAbstractions.allocate(backend, FT, Nx, Ny); copyto!(ρ, ρ_h)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny); copyto!(is_solid, solid_h)
    inlet_xx = KernelAbstractions.allocate(backend, FT, Ny); copyto!(inlet_xx, Cxx_prof)
    inlet_xy = KernelAbstractions.allocate(backend, FT, Ny); copyto!(inlet_xy, Cxy_prof)
    inlet_yy = KernelAbstractions.allocate(backend, FT, Ny); copyto!(inlet_yy, Cyy_prof)
    u_profile = KernelAbstractions.allocate(backend, FT, Ny); copyto!(u_profile, ux_prof)

    C_xx_h = repeat(reshape(Cxx_prof, 1, Ny), Nx, 1)
    C_xy_h = repeat(reshape(Cxy_prof, 1, Ny), Nx, 1)
    C_yy_h = repeat(reshape(Cyy_prof, 1, Ny), Nx, 1)
    Ψ_xx = KernelAbstractions.allocate(backend, FT, Nx, Ny); copyto!(Ψ_xx, C_xx_h)
    Ψ_xy = KernelAbstractions.allocate(backend, FT, Nx, Ny); copyto!(Ψ_xy, C_xy_h)
    Ψ_yy = KernelAbstractions.allocate(backend, FT, Nx, Ny); copyto!(Ψ_yy, C_yy_h)
    C_xx = use_logconf ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : Ψ_xx
    C_xy = use_logconf ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : Ψ_xy
    C_yy = use_logconf ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : Ψ_yy
    if use_logconf
        psi_to_C_2d!(C_xx, C_xy, C_yy, Ψ_xx, Ψ_xy, Ψ_yy)
    end

    g_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    init_conformation_field_2d!(g_xx, Ψ_xx, ux, uy)
    init_conformation_field_2d!(g_xy, Ψ_xy, ux, uy)
    init_conformation_field_2d!(g_yy, Ψ_yy, ux, uy)
    b_xx = similar(g_xx)
    b_xy = similar(g_xy)
    b_yy = similar(g_yy)

    first_bad_step = 0
    last_diag = nothing
    for step in 1:steps
        stream_2d!(b_xx, g_xx, Nx, Ny)
        stream_2d!(b_xy, g_xy, Nx, Ny)
        stream_2d!(b_yy, g_yy, Nx, Ny)
        reset_conformation_inlet_2d!(b_xx, inlet_xx, u_profile, Ny)
        reset_conformation_inlet_2d!(b_xy, inlet_xy, u_profile, Ny)
        reset_conformation_inlet_2d!(b_yy, inlet_yy, u_profile, Ny)
        reset_conformation_outlet_2d!(b_xx, Nx, Ny)
        reset_conformation_outlet_2d!(b_xy, Nx, Ny)
        reset_conformation_outlet_2d!(b_yy, Nx, Ny)

        g_xx, b_xx = b_xx, g_xx
        g_xy, b_xy = b_xy, g_xy
        g_yy, b_yy = b_yy, g_yy
        compute_conformation_macro_2d!(Ψ_xx, g_xx)
        compute_conformation_macro_2d!(Ψ_xy, g_xy)
        compute_conformation_macro_2d!(Ψ_yy, g_yy)

        if use_logconf
            collide_logconf_2d!(g_xx, Ψ_xx, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                                tau_plus, λ; magic, component=1, divergence_mode)
            collide_logconf_2d!(g_xy, Ψ_xy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                                tau_plus, λ; magic, component=2, divergence_mode)
            collide_logconf_2d!(g_yy, Ψ_yy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                                tau_plus, λ; magic, component=3, divergence_mode)
            psi_to_C_2d!(C_xx, C_xy, C_yy, Ψ_xx, Ψ_xy, Ψ_yy)
        else
            collide_conformation_2d!(g_xx, C_xx, ux, uy, C_xx, C_xy, C_yy,
                                     is_solid, tau_plus, λ;
                                     magic, component=1, divergence_mode)
            collide_conformation_2d!(g_xy, C_xy, ux, uy, C_xx, C_xy, C_yy,
                                     is_solid, tau_plus, λ;
                                     magic, component=2, divergence_mode)
            collide_conformation_2d!(g_yy, C_yy, ux, uy, C_xx, C_xy, C_yy,
                                     is_solid, tau_plus, λ;
                                     magic, component=3, divergence_mode)
        end

        reset_conformation_inlet_2d!(g_xx, inlet_xx, u_profile, Ny)
        reset_conformation_inlet_2d!(g_xy, inlet_xy, u_profile, Ny)
        reset_conformation_inlet_2d!(g_yy, inlet_yy, u_profile, Ny)
        reset_conformation_outlet_2d!(g_xx, Nx, Ny)
        reset_conformation_outlet_2d!(g_xy, Nx, Ny)
        reset_conformation_outlet_2d!(g_yy, Nx, Ny)

        if step == 1 || step % diagnostic_interval == 0 || step == steps
            diag = conformation_field_diagnostics_2d(C_xx, C_xy, C_yy, ρ, ux, uy, is_solid)
            last_diag = diag
            if first_bad_step == 0 && (!diag.finite || !(diag.min_eig > 0.0))
                first_bad_step = step
            end
            @printf("%-8s step=%d finite=%s min_eig=%.6g maxC=%.6g maxdiv=%.6g maxstrain=%.6g\n",
                    string(model_name), step, string(diag.finite), diag.min_eig,
                    diag.max_abs_C, diag.max_abs_divu, diag.max_strain_eig)
            flush(stdout)
        end
    end

    diag = last_diag === nothing ?
        conformation_field_diagnostics_2d(C_xx, C_xy, C_yy, ρ, ux, uy, is_solid) :
        last_diag
    @printf(io,
            "%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%d,%s,%.17g,%.17g,%.17g,%d\n",
            model_name, Nx, Ny, beta, Wi, λ, G, steps, diag.finite,
            diag.min_eig, diag.max_abs_C, diag.max_abs_divu, first_bad_step)
end

backend, FT, backend_label = _select_backend()
R = parse(Int, get(ENV, "KRAKEN_R", "30"))
Nx = parse(Int, get(ENV, "KRAKEN_NX", string(30R)))
Ny = parse(Int, get(ENV, "KRAKEN_NY", string(4R)))
beta = parse(Float64, get(ENV, "KRAKEN_BETA", "0.59"))
Wi = parse(Float64, get(ENV, "KRAKEN_WI", "0.1"))
u_mean = parse(Float64, get(ENV, "KRAKEN_U_MEAN", "0.005"))
steps = parse(Int, get(ENV, "KRAKEN_STEPS", "60000"))
tau_plus = parse(Float64, get(ENV, "KRAKEN_TAU_PLUS", "1.0"))
magic = parse(Float64, get(ENV, "KRAKEN_CONFORMATION_MAGIC", "1e-6"))
divergence_mode = Symbol(get(ENV, "KRAKEN_CONFORMATION_DIVERGENCE_MODE", "trace_free"))
diagnostic_interval = parse(Int, get(ENV, "KRAKEN_DIAGNOSTIC_INTERVAL", "10000"))
models = _parse_symbols(get(ENV, "KRAKEN_MODELS", "direct,logconf"))
results_dir = get(ENV, "KRAKEN_RESULTS_DIR",
    joinpath("tmp", "channel_frozen_velocity_cde_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
mkpath(results_dir)
csv_path = joinpath(results_dir, "channel_frozen_velocity_cde_patch.csv")

println("="^100)
println("Open-channel frozen-velocity CDE patch")
println("Backend=$backend_label FT=$FT Nx=$Nx Ny=$Ny R=$R Wi=$Wi beta=$beta steps=$steps")
println("tau_plus=$tau_plus magic=$magic divergence_mode=$divergence_mode")
println("CSV: $csv_path")
println("="^100)

open(csv_path, "w") do io
    println(io, "model,Nx,Ny,beta,Wi,lambda,G,steps,finite,min_eig,max_abs_C,max_abs_divu,first_bad_step")
    for model in models
        _run_case(; backend, FT, Nx, Ny, R, beta, Wi, model_name=model,
                  u_mean, steps, tau_plus, magic, divergence_mode,
                  diagnostic_interval, io)
    end
end

println("Done: $csv_path")
