# Frozen-velocity planar-wall patch for CDE boundary conditions.
#
# This is the q=0.5 control for the curved-wall CDE patch: YLW_A/YLW_B must
# reduce to the same result as CNEBB on a halfway planar wall.

include(joinpath(@__DIR__, "..", "src", "Kraken.jl"))

using .Kraken
using Dates
using KernelAbstractions
using Printf

_parse_symbol_list(raw::AbstractString) =
    [Symbol(strip(x)) for x in split(raw, ',') if !isempty(strip(x))]

_parse_list(::Type{T}, raw::AbstractString) where {T} =
    [parse(T, strip(x)) for x in split(raw, ',') if !isempty(strip(x))]

function _polymer_bc(name::Symbol, tau_plus)
    name === :cnebb && return CNEBB()
    name === :cnebb_qaware && return CNEBBQAware()
    name in (:cnebb_eq_gradient, :cnebb_eqgrad, :eq_gradient) && return CNEBBEqGradient()
    name === :ylw_a && return YLW_A(tau_plus=tau_plus)
    name === :ylw_b && return YLW_B(tau_plus=tau_plus)
    name === :ylw_balance && return YLWBalanceOnly()
    name === :none && return NoPolymerWallBC()
    error("unknown polymer BC $(name); expected cnebb, cnebb_qaware, cnebb_eq_gradient, ylw_a, ylw_b, ylw_balance, or none")
end

function _build_planar_geometry(Nx, Ny, γ)
    is_solid = falses(Nx, Ny)
    is_solid[:, 1] .= true
    is_solid[:, Ny] .= true
    q_wall = zeros(Float64, Nx, Ny, 9)

    # Bottom fluid row j=2: missing q=3,6,7; outgoing links 5,8,9.
    # Top fluid row j=Ny-1: missing q=5,8,9; outgoing links 3,6,7.
    for i in 2:Nx-1
        q_wall[i, 2, 5] = 0.5
        q_wall[i, 2, 8] = 0.5
        q_wall[i, 2, 9] = 0.5
        q_wall[i, Ny - 1, 3] = 0.5
        q_wall[i, Ny - 1, 6] = 0.5
        q_wall[i, Ny - 1, 7] = 0.5
    end

    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    y0 = (Ny - 1) / 2
    for j in 1:Ny, i in 1:Nx
        ux[i, j] = γ * ((j - 1) - y0)
    end
    ux[:, 1] .= 0.0
    ux[:, Ny] .= 0.0
    return is_solid, q_wall, ux, uy
end

function _run_case(; Nx, Ny, γ, νp, λ, tau_plus, magic, max_steps,
                   sample_interval, convergence_tol, model_name, bc_name,
                   d_list, io)
    backend = KernelAbstractions.CPU()
    FT = Float64
    is_solid_h, q_wall_h, ux_h, uy_h = _build_planar_geometry(Nx, Ny, γ)
    G = νp / λ
    polymer_model = model_name === :logconf ?
        LogConfOldroydB(G=G, λ=λ) : OldroydB(G=G, λ=λ)
    polymer_bc = _polymer_bc(bc_name, tau_plus)
    use_logconf = uses_log_conformation(polymer_model)

    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    q_wall = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    ux = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    uy = KernelAbstractions.allocate(backend, FT, Nx, Ny)
    copyto!(is_solid, is_solid_h)
    copyto!(q_wall, q_wall_h)
    copyto!(ux, ux_h)
    copyto!(uy, uy_h)

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
    g_xx_buf = similar(g_xx)
    g_xy_buf = similar(g_xy)
    g_yy_buf = similar(g_yy)

    last_Cxx = Array(C_xx)
    last_Cxy = Array(C_xy)
    last_Cyy = Array(C_yy)
    max_delta = Inf
    converged_step = max_steps

    for step in 1:max_steps
        stream_2d!(g_xx_buf, g_xx, Nx, Ny)
        stream_2d!(g_xy_buf, g_xy, Nx, Ny)
        stream_2d!(g_yy_buf, g_yy, Nx, Ny)

        apply_polymer_wall_bc!(g_xx_buf, g_xx, is_solid, q_wall, Ψ_xx, ux, uy, polymer_bc)
        apply_polymer_wall_bc!(g_xy_buf, g_xy, is_solid, q_wall, Ψ_xy, ux, uy, polymer_bc)
        apply_polymer_wall_bc!(g_yy_buf, g_yy, is_solid, q_wall, Ψ_yy, ux, uy, polymer_bc)

        g_xx, g_xx_buf = g_xx_buf, g_xx
        g_xy, g_xy_buf = g_xy_buf, g_xy
        g_yy, g_yy_buf = g_yy_buf, g_yy

        compute_conformation_macro_2d!(Ψ_xx, g_xx)
        compute_conformation_macro_2d!(Ψ_xy, g_xy)
        compute_conformation_macro_2d!(Ψ_yy, g_yy)

        if use_logconf
            collide_logconf_2d!(g_xx, Ψ_xx, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                                tau_plus, λ; magic=magic, component=1)
            collide_logconf_2d!(g_xy, Ψ_xy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                                tau_plus, λ; magic=magic, component=2)
            collide_logconf_2d!(g_yy, Ψ_yy, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                                tau_plus, λ; magic=magic, component=3)
            psi_to_C_2d!(C_xx, C_xy, C_yy, Ψ_xx, Ψ_xy, Ψ_yy)
        else
            collide_conformation_2d!(g_xx, C_xx, ux, uy, C_xx, C_xy, C_yy,
                                     is_solid, tau_plus, λ; magic=magic, component=1)
            collide_conformation_2d!(g_xy, C_xy, ux, uy, C_xx, C_xy, C_yy,
                                     is_solid, tau_plus, λ; magic=magic, component=2)
            collide_conformation_2d!(g_yy, C_yy, ux, uy, C_xx, C_xy, C_yy,
                                     is_solid, tau_plus, λ; magic=magic, component=3)
        end

        if step % sample_interval == 0 || step == max_steps
            Cxx_h = Array(C_xx)
            Cxy_h = Array(C_xy)
            Cyy_h = Array(C_yy)
            max_delta = max(maximum(abs.(Cxx_h .- last_Cxx)),
                            maximum(abs.(Cxy_h .- last_Cxy)),
                            maximum(abs.(Cyy_h .- last_Cyy)))
            last_Cxx = Cxx_h
            last_Cxy = Cxy_h
            last_Cyy = Cyy_h
            if max_delta < convergence_tol
                converged_step = step
                break
            end
        end
    end

    Cxx_h = Array(C_xx)
    Cxy_h = Array(C_xy)
    Cyy_h = Array(C_yy)
    τxx_target = 2νp * λ * γ^2
    τxy_target = νp * γ
    τyy_target = 0.0
    target_norm = sqrt(τxx_target^2 + τxy_target^2 + τyy_target^2)

    i_probe = Nx ÷ 2
    for side in (:south, :north), d in d_list
        j = side === :south ? 1 + d : Ny - d
        τxx = G * (Cxx_h[i_probe, j] - 1.0)
        τxy = G * Cxy_h[i_probe, j]
        τyy = G * (Cyy_h[i_probe, j] - 1.0)
        err = sqrt((τxx - τxx_target)^2 +
                   (τxy - τxy_target)^2 +
                   (τyy - τyy_target)^2)
        rel = err / max(target_norm, eps(Float64))
        @printf(io,
                "%s,%s,%s,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g\n",
                model_name, bc_name, side, d, γ, νp, λ, tau_plus,
                τxx, τxy, τyy, τxx_target, τxy_target, τyy_target,
                converged_step, rel)
    end
end

Nx = parse(Int, get(ENV, "KRAKEN_NX", "64"))
Ny = parse(Int, get(ENV, "KRAKEN_NY", "32"))
γ = parse(Float64, get(ENV, "KRAKEN_GAMMA", "0.001"))
νp = parse(Float64, get(ENV, "KRAKEN_NU_P", "0.041"))
λ = parse(Float64, get(ENV, "KRAKEN_LAMBDA", "5.0"))
tau_plus = parse(Float64, get(ENV, "KRAKEN_TAU_PLUS", "1.0"))
magic = parse(Float64, get(ENV, "KRAKEN_CONFORMATION_MAGIC", "1e-6"))
max_steps = parse(Int, get(ENV, "KRAKEN_STEPS", "500"))
sample_interval = parse(Int, get(ENV, "KRAKEN_SAMPLE_INTERVAL", "100"))
convergence_tol = parse(Float64, get(ENV, "KRAKEN_CONVERGENCE_TOL", "1e-12"))
models = _parse_symbol_list(get(ENV, "KRAKEN_MODELS", "direct"))
polymer_bcs = _parse_symbol_list(get(ENV, "KRAKEN_POLYMER_BCS", "cnebb,cnebb_qaware,ylw_a,ylw_b"))
d_list = _parse_list(Int, get(ENV, "KRAKEN_D_LIST", "1,2,3"))
results_dir = get(ENV, "KRAKEN_RESULTS_DIR",
    joinpath("tmp", "cde_frozen_planar_wall_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
mkpath(results_dir)
csv_path = joinpath(results_dir, "cde_frozen_planar_wall_patch.csv")

println("="^100)
println("Frozen CDE planar-wall q=0.5 patch")
println("Nx=$Nx Ny=$Ny gamma=$γ nu_p=$νp lambda=$λ tau_plus=$tau_plus magic=$magic steps=$max_steps")
println("models=$(join(models, ",")) bcs=$(join(polymer_bcs, ","))")
println("CSV: $csv_path")
println("="^100)

open(csv_path, "w") do io
    println(io, join(("model", "polymer_bc", "side", "d", "gamma", "nu_p",
                      "lambda", "tau_plus", "tau_xx", "tau_xy", "tau_yy",
                      "target_xx", "target_xy", "target_yy",
                      "converged_step", "rel_err"), ","))
    for model in models, bc in polymer_bcs
        t0 = time()
        _run_case(; Nx, Ny, γ, νp, λ, tau_plus, magic, max_steps,
                  sample_interval, convergence_tol, model_name=model,
                  bc_name=bc, d_list, io)
        @printf("%-8s %-14s %.2fs\n", string(model), string(bc), time() - t0)
    end
end

println("Done: $csv_path")
