# Liu et al. 2025 cylinder benchmark — reproduction with Kraken.jl.
#
# Uses `run_conformation_cylinder_libb_2d`:
#   - Fused TRT + Bouzidi LI-BB V2 for the solvent flow (curved cylinder)
#   - Modular BCSpec: ZouHeVelocity(Poiseuille) inlet + ZouHePressure outlet
#   - Explicit split drag by default. Post-source MEA remains available through
#     KRAKEN_DRAG_MODE for audit-only Liu/Yu force-accounting comparisons.
#   - Selectable conformation collision + polymer wall BC + Hermite stress source
#
# Liu setup (Table 3, CNEBB, Sc=10⁴):
#   - Domain 30R × 4R, cylinder at (15R, 2R), B = 0.5
#   - Re = 1, β = 0.59
#   - Re = U_avg · R / ν_total (L_c = R, NOT D)
#   - Cd = Fx / (0.5 ρ U_avg² D)   — Liu Eq 64
#
# Reference values (R=30, CNEBB, Sc=10⁴):
#   Wi=0.1 → Cd ≈ 130.36
#   Wi=0.5 → Cd ≈ 126.31
#   Wi=1.0 → Cd ≈ 151.31
#
# By default this script uses the direct-C regularized Liu Eq. 26 collision
# window validated by the patch ladder: τp,+≈0.5 with Liu's small polymer TRT
# magic parameter (`Λp=1e-6`; the cylinder section also reports `2.5e-7`).
#
# Usage:
#   julia --project=. hpc/liu_cylinder_benchmark.jl
#   KRAKEN_LIU_VARIANTS=cnebb_wall_aware,extrap_eq_wallfit4 \
#       julia --project=. hpc/liu_cylinder_benchmark.jl

using Kraken, Printf, KernelAbstractions

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
    requested in ("auto", "cpu") ||
        error("unknown or unavailable KRAKEN_BACKEND=$(requested); expected auto, cuda, metal, or cpu")
    return KernelAbstractions.CPU(), Float64, "CPU"
end

backend, FT, backend_label = _select_backend()

function _env_items(raw::AbstractString)
    return (strip(x) for x in split(replace(raw, ';' => ','), ',')
            if !isempty(strip(x)))
end

function _parse_list(::Type{T}, raw::AbstractString) where {T}
    return [parse(T, x) for x in _env_items(raw)]
end

function _parse_symbol_list(raw::AbstractString)
    return [Symbol(x) for x in _env_items(raw)]
end

function _polymer_model(name::Symbol, G, λ, ::Type{FT}) where {FT}
    name === :direct && return OldroydB(G=FT(G), λ=FT(λ))
    name === :logconf && return LogConfOldroydB(G=FT(G), λ=FT(λ))
    error("unknown KRAKEN_MODELS entry $(name); expected direct or logconf")
end

function _liu_variant(name::Symbol)
    name === :cnebb_wall_aware &&
        return (; label="CNEBB/wall_aware", bc=CNEBB(), gradient=:wall_aware)
    name === :cnebb_embedded &&
        return (; label="CNEBB/embedded_axis", bc=CNEBB(), gradient=:embedded_axis)
    name === :cnebb_wallfit4 &&
        return (; label="CNEBB/wallfit4", bc=CNEBB(), gradient=:wallfit4)
    name === :cnebb_field_wall_aware &&
        return (; label="CNEBBField/wall_aware", bc=CNEBBField(),
                  gradient=:wall_aware)
    name === :cnebb_field_eq_wall_aware &&
        return (; label="CNEBBFieldEq/wall_aware", bc=CNEBBFieldEquilibrium(),
                  gradient=:wall_aware)
    name === :extrap_eq_wall_aware &&
        return (; label="ExtrapEq/wall_aware", bc=ExtrapEqWallBC(),
                  gradient=:wall_aware)
    name === :log_field_wall_aware &&
        return (; label="LogField/wall_aware", bc=LogFieldWallBC(),
                  gradient=:wall_aware)
    name === :extrap_eq_embedded &&
        return (; label="ExtrapEq/embedded_axis", bc=ExtrapEqWallBC(),
                  gradient=:embedded_axis)
    name === :extrap_eq_wallfit4 &&
        return (; label="ExtrapEq/wallfit4", bc=ExtrapEqWallBC(),
                  gradient=:wallfit4)
    error("unknown Liu benchmark variant $(name); expected cnebb_wall_aware, cnebb_embedded, cnebb_wallfit4, cnebb_field_wall_aware, cnebb_field_eq_wall_aware, extrap_eq_wall_aware, log_field_wall_aware, extrap_eq_embedded, or extrap_eq_wallfit4")
end

println("="^70)
println("Liu et al. 2025 cylinder benchmark — LI-BB V2 driver")
println("Backend: $backend_label, FT=$FT")
println("="^70)

# Liu Table 3 reference values (CNEBB, Sc=10⁴)
liu_ref = Dict(
    (20, 0.1) => 129.42, (20, 0.5) => 125.17, (20, 1.0) => 164.26,
    (30, 0.1) => 130.36, (30, 0.5) => 126.31, (30, 1.0) => 151.31,
    (35, 0.1) => 130.77, (35, 0.5) => 127.72, (35, 1.0) => 149.04,
)

R_values = _parse_list(Int, get(ENV, "KRAKEN_R_LIST", "20,30"))
Wi_values = _parse_list(Float64, get(ENV, "KRAKEN_WI_LIST", "0.001,0.1,0.5,1.0"))
variants = [_liu_variant(name) for name in
            _parse_symbol_list(get(ENV, "KRAKEN_LIU_VARIANTS", "cnebb_wall_aware"))]
models = _parse_symbol_list(get(ENV, "KRAKEN_MODELS",
                                get(ENV, "KRAKEN_FORMULATIONS", "direct")))
β = parse(Float64, get(ENV, "KRAKEN_BETA", "0.59"))
u_mean = parse(Float64, get(ENV, "KRAKEN_U_MEAN", "0.005"))
steps_low_wi = parse(Int, get(ENV, "KRAKEN_STEPS_LOW_WI", "100000"))
steps = parse(Int, get(ENV, "KRAKEN_STEPS", "200000"))
avg_divisor = parse(Int, get(ENV, "KRAKEN_AVG_DIVISOR", "5"))
drag_stride = parse(Int, get(ENV, "KRAKEN_DRAG_STRIDE", "200"))
run_newtonian = get(ENV, "KRAKEN_RUN_NEWTONIAN", "1") == "1"
drag_mode = Symbol(get(ENV, "KRAKEN_DRAG_MODE", "explicit_split"))
hermite_source_mode =
    Symbol(get(ENV, "KRAKEN_HERMITE_SOURCE_MODE", "liu_direct"))
solvent_source_mode =
    Symbol(get(ENV, "KRAKEN_SOLVENT_SOURCE_MODE", "post_collision"))
source_stress_reconstruction =
    Symbol(get(ENV, "KRAKEN_SOURCE_STRESS_RECONSTRUCTION", "interior"))
source_stress_reconstruction_order =
    parse(Int, get(ENV, "KRAKEN_SOURCE_STRESS_RECONSTRUCTION_ORDER", "2"))
source_scale_dynamics =
    parse(Float64, get(ENV, "KRAKEN_SOURCE_SCALE_DYNAMICS", "1.0"))
solvent_source_on_domain_walls =
    get(ENV, "KRAKEN_SOLVENT_SOURCE_ON_DOMAIN_WALLS", "0") == "1"
solvent_source_on_cutlinks =
    get(ENV, "KRAKEN_SOLVENT_SOURCE_ON_CUTLINKS", "1") == "1"
conformation_magic =
    parse(Float64, get(ENV, "KRAKEN_CONFORMATION_MAGIC", "1e-6"))
conformation_collision =
    Symbol(get(ENV, "KRAKEN_CONFORMATION_COLLISION", "liu_eq26"))
Sc = parse(Float64, get(ENV, "KRAKEN_SC", "1e4"))
tau_plus_override = get(ENV, "KRAKEN_TAU_PLUS", "")
conformation_divergence_mode =
    Symbol(get(ENV, "KRAKEN_CONFORMATION_DIVERGENCE_MODE", "trace_free"))
conformation_initial_condition =
    Symbol(get(ENV, "KRAKEN_CONFORMATION_INITIAL_CONDITION", "inlet_profile"))
allow_diagnostic_force_mode =
    drag_mode in (:post_source_mea, :source_scaled_mea)
allow_diagnostic_conformation_collision =
    get(ENV, "KRAKEN_ALLOW_DIAGNOSTIC_CONFORMATION_COLLISION", "0") == "1"
wall_geometry =
    Symbol(get(ENV, "KRAKEN_WALL_GEOMETRY", "cutlink"))
diagnostic_interval =
    parse(Int, get(ENV, "KRAKEN_DIAGNOSTIC_INTERVAL", "0"))

function _tau_plus_for_collision(conformation_collision, ν_s, Sc, tau_plus_override)
    !isempty(tau_plus_override) && return parse(Float64, tau_plus_override)
    conformation_collision === :trt && return 1.0
    return 0.5 + 3.0 * ν_s / Sc
end

println("R_LIST=$(join(R_values, ",")) WI_LIST=$(join(Wi_values, ","))")
println("VARIANTS=$(join((v.label for v in variants), ","))")
println("MODELS=$(join(models, ","))")
println("beta=$β u_mean=$u_mean Sc=$Sc steps_low_wi=$steps_low_wi steps=$steps avg_divisor=$avg_divisor drag_stride=$drag_stride run_newtonian=$run_newtonian drag_mode=$drag_mode hermite_source_mode=$hermite_source_mode solvent_source_mode=$solvent_source_mode source_reconstruction=$source_stress_reconstruction source_order=$source_stress_reconstruction_order source_scale=$source_scale_dynamics source_on_domain_walls=$solvent_source_on_domain_walls source_on_cutlinks=$solvent_source_on_cutlinks tau_plus_override=$(isempty(tau_plus_override) ? "none" : tau_plus_override) conformation_magic=$conformation_magic conformation_collision=$conformation_collision divergence_mode=$conformation_divergence_mode initial_condition=$conformation_initial_condition wall_geometry=$wall_geometry diagnostic_interval=$diagnostic_interval")
allow_diagnostic_force_mode &&
    println("WARNING: KRAKEN_DRAG_MODE=$drag_mode is audit-only; Cd_report is not the validation force path.")

for R in R_values
    # Liu uses Re = U_avg · R / ν → solve for ν_total given u_mean, R, Re
    # Then split ν_s, ν_p with β.
    Re_target = 1.0
    ν_total = u_mean * R / Re_target
    ν_s = β * ν_total
    ν_p = (1 - β) * ν_total
    tau_plus_R = _tau_plus_for_collision(
        conformation_collision, ν_s, Sc, tau_plus_override)
    Nx = 30 * R
    Ny = 4 * R
    cx = 15 * R
    cy = (Ny - 1) / 2

    println("\n>>> R = $R  (Nx=$Nx, Ny=$Ny, cx=$cx, cy=$cy, ν_s=$ν_s, ν_p=$ν_p, tau_plus=$tau_plus_R)")
    if run_newtonian
        max_steps_newt = steps_low_wi
        avg_window_newt = max_steps_newt ÷ avg_divisor
        t0 = time()
        rn = run_cylinder_libb_2d(;
            Nx=Nx, Ny=Ny, radius=R, cx=cx, cy=cy,
            u_in=FT(1.5 * u_mean), ν=FT(ν_total), inlet=:parabolic,
            max_steps=max_steps_newt, avg_window=avg_window_newt,
            drag_stride=drag_stride,
            backend=backend, T=FT,
        )
        dt = time() - t0
        @printf("%-22s %-6s %-10.3f %-10s %-8s %-14s (%.0fs, Cl=%.3g)\n",
                "Newtonian", "-", rn.Cd, "-", "-", "-", dt, rn.Cl)
    end
    @printf("%-22s %-9s %-6s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-9s %-14s\n",
            "variant", "model", "Wi", "Cd_report", "Cd_post", "Cd_split",
            "Cd_Liu", "err_rep%", "err_post%", "err_split%", "bad_step",
            "gradient")
    println("-"^126)

    for variant in variants, model_name in models, Wi in Wi_values
        # Liu: Wi = λ · U_c / R with U_c = U_avg = u_mean
        λ = Wi * R / u_mean
        G = ν_p / λ
        polymer_model = _polymer_model(model_name, G, λ, FT)
        max_steps = Wi < 0.01 ? steps_low_wi : steps
        avg_window = max_steps ÷ avg_divisor

        t0 = time()
        r = run_conformation_cylinder_libb_2d(;
                Nx=Nx, Ny=Ny, radius=R, cx=cx, cy=cy,
                u_mean=FT(u_mean), ν_s=FT(ν_s),
                polymer_model=polymer_model, tau_plus=FT(tau_plus_R),
                inlet=:parabolic, ρ_out=one(FT),
                max_steps=max_steps, avg_window=avg_window,
                drag_stride=drag_stride,
                polymer_bc=variant.bc,
                conformation_gradient_mode=variant.gradient,
                conformation_magic=conformation_magic,
                conformation_collision=conformation_collision,
                conformation_divergence_mode=conformation_divergence_mode,
                conformation_initial_condition=conformation_initial_condition,
                wall_geometry=wall_geometry,
                drag_mode=drag_mode,
                hermite_source_mode=hermite_source_mode,
                solvent_source_mode=solvent_source_mode,
                source_stress_reconstruction=source_stress_reconstruction,
                source_stress_reconstruction_order=source_stress_reconstruction_order,
                source_scale_dynamics=source_scale_dynamics,
                solvent_source_on_domain_walls=solvent_source_on_domain_walls,
                solvent_source_on_cutlinks=solvent_source_on_cutlinks,
                diagnostic_interval=diagnostic_interval,
                allow_diagnostic_polymer_bc=!(variant.bc isa CNEBB ||
                                              variant.bc isa ExtrapEqWallBC),
                allow_diagnostic_force_mode=allow_diagnostic_force_mode,
                allow_diagnostic_conformation_collision=
                    allow_diagnostic_conformation_collision,
                allow_diagnostic_log_wall_bc=model_name === :logconf,
                backend=backend, FT=FT)
        dt = time() - t0

        ref = get(liu_ref, (R, Wi), NaN)
        err_report = isnan(ref) ? NaN : (r.Cd - ref) / ref * 100
        err_post = isnan(ref) ? NaN : (r.Cd_mea_post_source - ref) / ref * 100
        err_split = isnan(ref) ? NaN : (r.Cd_split_explicit - ref) / ref * 100
        @printf("%-22s %-9s %-6.3f %-10.3f %-10.3f %-10.3f %-10.3f %-10.2f %-10.2f %-10.2f %-9d %-14s (%.0fs)\n",
                variant.label, string(model_name), Wi, r.Cd,
                r.Cd_mea_post_source, r.Cd_split_explicit, ref, err_report,
                err_post, err_split,
                r.first_nonfinite_step, string(variant.gradient), dt)
    end
end

println("\nDone.")
