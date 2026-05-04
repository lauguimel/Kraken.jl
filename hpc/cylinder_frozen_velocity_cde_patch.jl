# Frozen-velocity CDE canary for the viscoelastic cylinder.
#
# This isolates the conformation/log-conformation transport and wall BC on a
# realistic cylinder velocity field, without polymer feedback on the Navier-
# Stokes populations. If this fails, the bug is in CDE/BC/∇u. If this passes
# but the active macro flow fails, the bug is in feedback/source coupling.

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
                backend = Base.invokelatest(getfield(_CUDA_MOD, :CUDABackend))
                return backend, Float64, "CUDA"
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

_parse_list(::Type{T}, raw::AbstractString) where {T} =
    [parse(T, strip(x)) for x in split(raw, ',') if !isempty(strip(x))]

_parse_symbols(raw::AbstractString) =
    [Symbol(strip(x)) for x in split(raw, ',') if !isempty(strip(x))]

function _polymer_bc(name::Symbol)
    name === :cnebb && return CNEBB()
    name === :cnebb_qaware && return CNEBBQAware()
    name === :ylw_balance && return YLWBalanceOnly()
    name === :none && return NoPolymerWallBC()
    error("unknown polymer BC $(name); expected cnebb, cnebb_qaware, ylw_balance, or none")
end

function _inlet_conformation_profile(::Type{FT}, Ny, u_max, λ, use_logconf) where {FT}
    cxx = ones(FT, Ny)
    cxy = zeros(FT, Ny)
    cyy = ones(FT, Ny)
    H = FT(Ny)
    for j in 1:Ny
        y = FT(j) - FT(0.5)
        dudy = FT(u_max) * FT(4) * (H - FT(2) * y) / (H * H)
        cxy[j] = FT(λ) * dudy
        cxx[j] = one(FT) + FT(2) * (FT(λ) * dudy)^2
    end
    if use_logconf
        for j in 1:Ny
            tr = cxx[j] + cyy[j]
            diff = cxx[j] - cyy[j]
            disc = sqrt(diff * diff + FT(4) * cxy[j] * cxy[j])
            μ1 = FT(0.5) * (tr + disc)
            μ2 = FT(0.5) * (tr - disc)
            l1 = log(max(μ1, FT(1e-30)))
            l2 = log(max(μ2, FT(1e-30)))
            θ = FT(0.5) * atan(FT(2) * cxy[j], diff)
            c = cos(θ)
            s = sin(θ)
            cxx[j] = c * c * l1 + s * s * l2
            cxy[j] = c * s * (l1 - l2)
            cyy[j] = s * s * l1 + c * c * l2
        end
    end
    return cxx, cxy, cyy
end

function _run_frozen_case(; backend, FT, R, beta, Wi, model_name, bc_name,
                          u_mean, hydro_steps, cde_steps, tau_plus,
                          conformation_magic, divergence_mode,
                          diagnostic_interval, io)
    Nx = 30R
    Ny = 4R
    cx = 15R
    cy = (4R - 1) / 2
    ν_total = u_mean * R
    ν_s = beta * ν_total
    ν_p = (1 - beta) * ν_total
    λ = Wi * R / u_mean
    G = ν_p / λ
    u_max = 1.5 * u_mean

    hydro = run_cylinder_libb_2d(;
        Nx, Ny, radius=R, cx, cy,
        u_in=FT(u_max), ν=FT(ν_total), inlet=:parabolic,
        max_steps=hydro_steps, avg_window=max(1, hydro_steps ÷ 5),
        drag_stride=max(1, hydro_steps ÷ 20),
        momentum_exchange_mode=:mei_reconstruct,
        backend, T=FT)

    use_logconf = model_name === :logconf
    polymer_model = use_logconf ? LogConfOldroydB(G=FT(G), λ=FT(λ)) :
                                  OldroydB(G=FT(G), λ=FT(λ))
    polymer_bc = _polymer_bc(bc_name)

    q_wall = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    ux = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    uy = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    ρ = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    copyto!(q_wall, FT.(hydro.q_wall))
    copyto!(is_solid, hydro.is_solid)
    copyto!(ux, FT.(hydro.ux))
    copyto!(uy, FT.(hydro.uy))
    copyto!(ρ, FT.(hydro.ρ))

    inlet_xx_h, inlet_xy_h, inlet_yy_h =
        _inlet_conformation_profile(FT, Ny, u_max, λ, use_logconf)
    inlet_xx = KernelAbstractions.allocate(backend, FT, Ny)
    inlet_xy = KernelAbstractions.allocate(backend, FT, Ny)
    inlet_yy = KernelAbstractions.allocate(backend, FT, Ny)
    u_profile = KernelAbstractions.allocate(backend, FT, Ny)
    copyto!(inlet_xx, inlet_xx_h)
    copyto!(inlet_xy, inlet_xy_h)
    copyto!(inlet_yy, inlet_yy_h)
    copyto!(u_profile, FT.(hydro.ux[1, :]))

    C_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(C_xx, one(FT))
    C_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    C_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(C_yy, one(FT))
    Ψ_xx = use_logconf ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : C_xx
    Ψ_xy = use_logconf ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : C_xy
    Ψ_yy = use_logconf ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : C_yy

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
    for step in 1:cde_steps
        stream_2d!(b_xx, g_xx, Nx, Ny)
        stream_2d!(b_xy, g_xy, Nx, Ny)
        stream_2d!(b_yy, g_yy, Nx, Ny)

        apply_polymer_wall_bc!(b_xx, g_xx, is_solid, q_wall, Ψ_xx, ux, uy, polymer_bc)
        apply_polymer_wall_bc!(b_xy, g_xy, is_solid, q_wall, Ψ_xy, ux, uy, polymer_bc)
        apply_polymer_wall_bc!(b_yy, g_yy, is_solid, q_wall, Ψ_yy, ux, uy, polymer_bc)

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
                                tau_plus, λ; magic=conformation_magic,
                                component=1, divergence_mode)
            collide_logconf_2d!(g_xy, Ψ_xy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                                tau_plus, λ; magic=conformation_magic,
                                component=2, divergence_mode)
            collide_logconf_2d!(g_yy, Ψ_yy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                                tau_plus, λ; magic=conformation_magic,
                                component=3, divergence_mode)
            psi_to_C_2d!(C_xx, C_xy, C_yy, Ψ_xx, Ψ_xy, Ψ_yy)
        else
            collide_conformation_2d!(g_xx, C_xx, ux, uy, C_xx, C_xy, C_yy,
                                     is_solid, tau_plus, λ;
                                     magic=conformation_magic, component=1,
                                     divergence_mode)
            collide_conformation_2d!(g_xy, C_xy, ux, uy, C_xx, C_xy, C_yy,
                                     is_solid, tau_plus, λ;
                                     magic=conformation_magic, component=2,
                                     divergence_mode)
            collide_conformation_2d!(g_yy, C_yy, ux, uy, C_xx, C_xy, C_yy,
                                     is_solid, tau_plus, λ;
                                     magic=conformation_magic, component=3,
                                     divergence_mode)
        end

        reset_conformation_inlet_2d!(g_xx, inlet_xx, u_profile, Ny)
        reset_conformation_inlet_2d!(g_xy, inlet_xy, u_profile, Ny)
        reset_conformation_inlet_2d!(g_yy, inlet_yy, u_profile, Ny)
        reset_conformation_outlet_2d!(g_xx, Nx, Ny)
        reset_conformation_outlet_2d!(g_xy, Nx, Ny)
        reset_conformation_outlet_2d!(g_yy, Nx, Ny)

        if step == 1 || step % diagnostic_interval == 0 || step == cde_steps
            diag = conformation_field_diagnostics_2d(C_xx, C_xy, C_yy, ρ, ux, uy, is_solid)
            last_diag = diag
            if first_bad_step == 0 && (!diag.finite || !(diag.min_eig > 0.0))
                first_bad_step = step
            end
            @printf("%-8s %-10s R=%d Wi=%.4g step=%d finite=%s min_eig=%.6g at=(%d,%d) maxC=%.6g maxdiv=%.6g\n",
                    string(model_name), string(bc_name), R, Wi, step,
                    string(diag.finite), diag.min_eig, diag.min_i, diag.min_j,
                    diag.max_abs_C, diag.max_abs_divu)
            @printf("    max_strain=%.6g at=(%d,%d) lambda*max_strain=%.6g\n",
                    diag.max_strain_eig, diag.maxStrain_i, diag.maxStrain_j,
                    λ * diag.max_strain_eig)
            @printf("    minloc_grad=(%.6g, %.6g, %.6g, %.6g) minloc_strain=%.6g lambda*minloc_strain=%.6g\n",
                    diag.min_dudx, diag.min_dudy, diag.min_dvdx, diag.min_dvdy,
                    diag.min_strain_eig, λ * diag.min_strain_eig)
            flush(stdout)
        end
    end

    diag = last_diag === nothing ?
        conformation_field_diagnostics_2d(C_xx, C_xy, C_yy, ρ, ux, uy, is_solid) :
        last_diag
    @printf(io,
            "%s,%s,%d,%.17g,%.17g,%.17g,%.17g,%d,%d,%s,%.17g,%d,%d,%.17g,%.17g,%d,%d,%.17g,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,%.17g,%.17g\n",
            model_name, bc_name, R, beta, Wi, λ, G,
            hydro_steps, cde_steps, diag.finite, diag.min_eig,
            diag.min_i, diag.min_j, diag.max_abs_C, diag.max_abs_divu,
            diag.maxDiv_i, diag.maxDiv_j, diag.max_strain_eig,
            diag.maxStrain_i, diag.maxStrain_j, diag.min_dudx, diag.min_dudy,
            diag.min_dvdx, diag.min_dvdy, diag.min_strain_eig, first_bad_step,
            diag.n_fluid, hydro.Cd, hydro.Cl)
    flush(io)
    return nothing
end

backend, FT, backend_label = _select_backend()

R_list = _parse_list(Int, get(ENV, "KRAKEN_R_LIST", "15"))
Wi_list = _parse_list(Float64, get(ENV, "KRAKEN_WI_LIST", "0.1"))
models = _parse_symbols(get(ENV, "KRAKEN_MODELS", "direct,logconf"))
bcs = _parse_symbols(get(ENV, "KRAKEN_POLYMER_BCS", "cnebb"))
beta = parse(Float64, get(ENV, "KRAKEN_BETA", "0.59"))
u_mean = parse(Float64, get(ENV, "KRAKEN_U_MEAN", "0.005"))
hydro_steps = parse(Int, get(ENV, "KRAKEN_HYDRO_STEPS", "5000"))
cde_steps = parse(Int, get(ENV, "KRAKEN_CDE_STEPS", "20000"))
tau_plus = parse(Float64, get(ENV, "KRAKEN_TAU_PLUS", "1.0"))
conformation_magic = parse(Float64, get(ENV, "KRAKEN_CONFORMATION_MAGIC", "1e-6"))
divergence_mode = Symbol(get(ENV, "KRAKEN_CONFORMATION_DIVERGENCE_MODE", "trace_free"))
diagnostic_interval = parse(Int, get(ENV, "KRAKEN_DIAGNOSTIC_INTERVAL", "5000"))
results_dir = get(ENV, "KRAKEN_RESULTS_DIR",
    joinpath("tmp", "cylinder_frozen_velocity_cde_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
mkpath(results_dir)
csv_path = joinpath(results_dir, "cylinder_frozen_velocity_cde_patch.csv")

println("="^120)
println("Cylinder frozen-velocity CDE patch")
println("Backend=$backend_label FT=$FT R=$(join(R_list, ",")) Wi=$(join(Wi_list, ",")) beta=$beta")
println("hydro_steps=$hydro_steps cde_steps=$cde_steps tau_plus=$tau_plus magic=$conformation_magic divergence_mode=$divergence_mode")
println("models=$(join(models, ",")) bcs=$(join(bcs, ","))")
println("CSV: $csv_path")
println("="^120)

open(csv_path, "w") do io
    println(io, join(("model", "polymer_bc", "R", "beta", "Wi", "lambda",
                      "G", "hydro_steps", "cde_steps", "finite", "min_eig",
                      "min_i", "min_j", "max_abs_C", "max_abs_divu",
                      "maxDiv_i", "maxDiv_j", "max_strain_eig",
                      "maxStrain_i", "maxStrain_j", "min_dudx", "min_dudy",
                      "min_dvdx", "min_dvdy", "min_strain_eig",
                      "first_bad_step", "n_fluid",
                      "Cd_hydro", "Cl_hydro"), ","))
    for Wi in Wi_list, model in models, bc in bcs, R in R_list
        _run_frozen_case(; backend, FT, R, beta, Wi, model_name=model,
                         bc_name=bc, u_mean, hydro_steps, cde_steps,
                         tau_plus, conformation_magic, divergence_mode,
                         diagnostic_interval, io)
    end
end

println("Done: $csv_path")
