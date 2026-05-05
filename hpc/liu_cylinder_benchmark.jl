# Liu et al. 2025 cylinder benchmark — reproduction with Kraken.jl.
#
# Uses `run_conformation_cylinder_libb_2d`:
#   - Fused TRT + Bouzidi LI-BB V2 for the solvent flow (curved cylinder)
#   - Modular BCSpec: ZouHeVelocity(Poiseuille) inlet + ZouHePressure outlet
#   - Mei-consistent MEA drag on cut-link q_wall
#   - TRT conformation LBM + selectable polymer wall BC + Hermite stress source
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
# Usage:
#   julia --project=. hpc/liu_cylinder_benchmark.jl
#   KRAKEN_LIU_VARIANTS=cnebb_wall_aware,extrap_eq_wallfit4 \
#       julia --project=. hpc/liu_cylinder_benchmark.jl

using Kraken, Printf, CUDA

backend = CUDABackend()
FT = Float64

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

function _liu_variant(name::Symbol)
    name === :cnebb_wall_aware &&
        return (; label="CNEBB/wall_aware", bc=CNEBB(), gradient=:wall_aware)
    name === :cnebb_embedded &&
        return (; label="CNEBB/embedded_axis", bc=CNEBB(), gradient=:embedded_axis)
    name === :cnebb_wallfit4 &&
        return (; label="CNEBB/wallfit4", bc=CNEBB(), gradient=:wallfit4)
    name === :extrap_eq_embedded &&
        return (; label="ExtrapEq/embedded_axis", bc=ExtrapEqWallBC(),
                  gradient=:embedded_axis)
    name === :extrap_eq_wallfit4 &&
        return (; label="ExtrapEq/wallfit4", bc=ExtrapEqWallBC(),
                  gradient=:wallfit4)
    error("unknown Liu benchmark variant $(name); expected cnebb_wall_aware, cnebb_embedded, cnebb_wallfit4, extrap_eq_embedded, or extrap_eq_wallfit4")
end

println("="^70)
println("Liu et al. 2025 cylinder benchmark — LI-BB V2 driver")
println("Backend: $(typeof(backend)), GPU: $(CUDA.name(CUDA.device()))")
println("="^70)

# Liu Table 3 reference values (CNEBB, Sc=10⁴)
liu_ref = Dict(
    (20, 0.1) => 129.42, (20, 0.5) => 125.17, (20, 1.0) => 164.26,
    (30, 0.1) => 130.36, (30, 0.5) => 126.31, (30, 1.0) => 151.31,
)

R_values = _parse_list(Int, get(ENV, "KRAKEN_R_LIST", "20,30"))
Wi_values = _parse_list(Float64, get(ENV, "KRAKEN_WI_LIST", "0.001,0.1,0.5,1.0"))
variants = [_liu_variant(name) for name in
            _parse_symbol_list(get(ENV, "KRAKEN_LIU_VARIANTS", "cnebb_wall_aware"))]
β = parse(Float64, get(ENV, "KRAKEN_BETA", "0.59"))
u_mean = parse(Float64, get(ENV, "KRAKEN_U_MEAN", "0.02"))
steps_low_wi = parse(Int, get(ENV, "KRAKEN_STEPS_LOW_WI", "100000"))
steps = parse(Int, get(ENV, "KRAKEN_STEPS", "200000"))
avg_divisor = parse(Int, get(ENV, "KRAKEN_AVG_DIVISOR", "5"))

println("R_LIST=$(join(R_values, ",")) WI_LIST=$(join(Wi_values, ","))")
println("VARIANTS=$(join((v.label for v in variants), ","))")
println("beta=$β u_mean=$u_mean steps_low_wi=$steps_low_wi steps=$steps")

for R in R_values
    # Liu uses Re = U_avg · R / ν → solve for ν_total given u_mean, R, Re
    # Then split ν_s, ν_p with β.
    Re_target = 1.0
    ν_total = u_mean * R / Re_target
    ν_s = β * ν_total
    ν_p = (1 - β) * ν_total
    Nx = 30 * R
    Ny = 4 * R
    cx = 15 * R
    cy = 2 * R

    println("\n>>> R = $R  (Nx=$Nx, Ny=$Ny, ν_s=$ν_s, ν_p=$ν_p)")
    @printf("%-22s %-6s %-10s %-10s %-8s %-14s\n",
            "variant", "Wi", "Cd_sim", "Cd_Liu", "err%", "gradient")
    println("-"^76)

    for variant in variants, Wi in Wi_values
        # Liu: Wi = λ · U_c / R with U_c = U_avg = u_mean
        λ = Wi * R / u_mean
        max_steps = Wi < 0.01 ? steps_low_wi : steps
        avg_window = max_steps ÷ avg_divisor

        t0 = time()
        r = run_conformation_cylinder_libb_2d(;
                Nx=Nx, Ny=Ny, radius=R, cx=cx, cy=cy,
                u_mean=u_mean, ν_s=ν_s, ν_p=ν_p, lambda=λ, tau_plus=1.0,
                inlet=:parabolic, ρ_out=1.0,
                max_steps=max_steps, avg_window=avg_window,
                polymer_bc=variant.bc,
                conformation_gradient_mode=variant.gradient,
                backend=backend, FT=FT)
        dt = time() - t0

        ref = get(liu_ref, (R, Wi), NaN)
        err = isnan(ref) ? NaN : (r.Cd - ref) / ref * 100
        @printf("%-22s %-6.3f %-10.3f %-10.3f %-8.2f %-14s (%.0fs)\n",
                variant.label, Wi, r.Cd, ref, err, string(variant.gradient), dt)
    end
end

println("\nDone.")
