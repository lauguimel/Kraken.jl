# =============================================================================
# Phase 2 helpers: Setup directive, Presets, Sanity, Sweeps, spell-correction
# =============================================================================
#
# These helpers extend the .krk DSL with convenience directives:
#   - `Setup reynolds = 1000 [L_ref = ...] [U_ref = ...]`
#   - `Setup rayleigh = 1e5 prandtl = 0.71`
#   - `Preset <name>`  (cavity_2d, poiseuille_2d, couette_2d, taylor_green_2d,
#                        rayleigh_benard_2d)
#   - `Sweep param = [a, b, c]` (expands into multiple SimulationSetups)
# Sanity checks (tau, Mach) run at parse time.
# Unknown identifiers get Levenshtein-based "did you mean?" suggestions.

"""
    _levenshtein(a::AbstractString, b::AbstractString) -> Int

Compute Levenshtein edit distance between two strings.
"""
function _levenshtein(a::AbstractString, b::AbstractString)
    m, n = length(a), length(b)
    m == 0 && return n
    n == 0 && return m
    av = collect(a)
    bv = collect(b)
    prev = collect(0:n)
    curr = zeros(Int, n + 1)
    for i in 1:m
        curr[1] = i
        for j in 1:n
            cost = av[i] == bv[j] ? 0 : 1
            curr[j+1] = min(curr[j] + 1, prev[j+1] + 1, prev[j] + cost)
        end
        prev, curr = curr, prev
    end
    return prev[n + 1]
end

"""
    _suggest_name(name, candidates) -> Union{String, Nothing}

Return the closest candidate by Levenshtein distance if within threshold
(distance ≤ max(2, length(name) ÷ 3)), otherwise `nothing`.
"""
function _suggest_name(name::AbstractString, candidates)
    best = nothing
    best_d = typemax(Int)
    for c in candidates
        d = _levenshtein(lowercase(String(name)), lowercase(String(c)))
        if d < best_d
            best_d = d
            best = String(c)
        end
    end
    threshold = max(2, length(name) ÷ 3)
    return best_d <= threshold ? best : nothing
end

"""
    _parse_setup(line, user_vars) -> Dict{Symbol,Float64}

Parse a `Setup key = value ...` directive. Known keys:
`reynolds`, `rayleigh`, `prandtl`, `L_ref`, `U_ref`.
"""
function _parse_setup(line::String, user_vars::Dict{Symbol,Float64})
    out = Dict{Symbol, Float64}()
    known = (:reynolds, :rayleigh, :prandtl, :L_ref, :U_ref)
    for m in eachmatch(r"(\w+)\s*=\s*([\w.eE+\-*/()]+)", line)
        m.captures[1] == "Setup" && continue
        key = Symbol(m.captures[1])
        if key ∉ known
            sug = _suggest_name(String(m.captures[1]), known)
            msg = "Unknown Setup key '$key'"
            sug !== nothing && (msg *= " (did you mean: $sug?)")
            throw(ArgumentError(msg))
        end
        val_str = strip(String(m.captures[2]))
        val = tryparse(Float64, val_str)
        if val === nothing
            val = Float64(evaluate(parse_kraken_expr(val_str, user_vars)))
        end
        out[key] = val
    end
    return out
end

"""
    _apply_setup_helpers!(physics_params, helpers, domain, boundaries)

Mutate `physics_params` to add auto-computed `nu`, `alpha`, `gbeta_DT` from
`reynolds`/`rayleigh`/`prandtl` helpers. Errors if conflicts exist.
"""
function _apply_setup_helpers!(physics_params::Dict{Symbol,Float64},
                               helpers::Dict{Symbol,Float64},
                               domain,
                               boundaries::Vector{BoundarySetup})
    isempty(helpers) && return

    # Determine L_ref: explicit > domain min(Nx, Ny) in lattice units
    L_ref = get(helpers, :L_ref, Float64(min(domain.Nx, domain.Ny)))

    # Determine U_ref: explicit > probe a velocity BC > default 0.1
    U_ref = get(helpers, :U_ref, _probe_U_ref(boundaries))

    if haskey(helpers, :reynolds)
        Re = helpers[:reynolds]
        if haskey(physics_params, :nu)
            throw(ArgumentError(
                "Setup reynolds conflicts with Physics nu (both specified). " *
                "Remove one: either use `Setup reynolds = $Re` or `Physics nu = ...`."))
        end
        ν = U_ref * L_ref / Re
        physics_params[:nu] = ν
        physics_params[:Re] = Re
    end

    if haskey(helpers, :rayleigh)
        Ra = helpers[:rayleigh]
        Pr = get(helpers, :prandtl, get(physics_params, :Pr, 0.71))
        # Standard scaling: ν = U_ref * L_ref / sqrt(Ra/Pr), α = ν/Pr,
        #                   gβΔT = Ra * ν * α / L_ref^3
        # Using U_ref = sqrt(gβΔT L) gives ν = sqrt(Pr/Ra) * U_ref * L
        ν_ra = sqrt(Pr / Ra) * U_ref * L_ref
        α_ra = ν_ra / Pr
        gβΔT = Ra * ν_ra * α_ra / L_ref^3
        if haskey(physics_params, :nu)
            throw(ArgumentError(
                "Setup rayleigh conflicts with Physics nu (both specified)."))
        end
        physics_params[:nu] = ν_ra
        physics_params[:alpha] = α_ra
        physics_params[:gbeta_DT] = gβΔT
        physics_params[:Ra] = Ra
        physics_params[:Pr] = Pr
    end

    return
end

"""
    _validate_faces_vs_lattice(setup)

Reject 3D-only face names (`:top`, `:bottom`) when the lattice is D2Q9.
Allowed:
- D2Q9:  `:north`, `:south`, `:east`, `:west` (+ `:front`/`:back` legacy aliases)
- D3Q19: all of the above + `:top`, `:bottom`

Axisymmetric cases (`Module axisymmetric`) express boundaries with the
(z, r) aliases `z`/`wall`/`axis` in the user-facing .krk text; those are
rewritten to `x`/`north`/`south` in `_parse_boundary` before this check
runs, so by the time we get here only standard face symbols remain.
"""
function _validate_faces_vs_lattice(setup::SimulationSetup)
    d2_faces = (:north, :south, :east, :west, :front, :back)
    if setup.lattice == :D2Q9
        for b in setup.boundaries
            if !(b.face in d2_faces)
                throw(ArgumentError(
                    "Boundary face ':$(b.face)' is not valid for D2Q9. " *
                    "2D face names are: north/south/east/west."))
            end
        end
    end
    return nothing
end

"""Scan boundary conditions for a velocity BC and return its magnitude, or 0.1."""
function _probe_U_ref(boundaries::Vector{BoundarySetup})
    for b in boundaries
        if b.type == :velocity
            ux = 0.0; uy = 0.0
            if haskey(b.values, :ux)
                try; ux = Float64(evaluate(b.values[:ux])); catch; end
            end
            if haskey(b.values, :uy)
                try; uy = Float64(evaluate(b.values[:uy])); catch; end
            end
            mag = sqrt(ux^2 + uy^2)
            mag > 0 && return mag
        end
    end
    return 0.1
end

# ══════════════════════════════════════════════════════════════════════
#  LBM parameter calculator / advisor
# ══════════════════════════════════════════════════════════════════════

"""
    LBMParams

Computed lattice-Boltzmann parameters with feasibility assessment.
Returned by [`lbm_params`](@ref).
"""
struct LBMParams
    # --- Inputs (lattice units) ---
    Re::Float64
    N::Int
    U_ref::Float64
    # --- Derived ---
    nu::Float64       # lattice viscosity
    tau::Float64      # relaxation time
    omega::Float64    # relaxation rate
    Ma::Float64       # Mach number
    # --- Quality assessment ---
    feasible::Bool           # all hard constraints satisfied
    regime::Symbol           # :optimal, :acceptable, :marginal, :diffusive, :infeasible
    warnings::Vector{String} # human-readable diagnostics
    # --- Recommendations ---
    recommended_N::Int           # best N for this Re at default U_ref
    recommended_U_ref::Float64   # best U_ref for this Re and N
end

function Base.show(io::IO, p::LBMParams)
    status = p.feasible ? "✓ feasible" : "✗ INFEASIBLE"
    println(io, "LBMParams ($status, regime = $(p.regime))")
    println(io, "  Re = $(p.Re), N = $(p.N), U_ref = $(round(p.U_ref, digits=6))")
    println(io, "  ν  = $(round(p.nu, digits=6)), τ = $(round(p.tau, digits=4)), ω = $(round(p.omega, digits=6))")
    println(io, "  Ma = $(round(p.Ma, digits=4))")
    if !isempty(p.warnings)
        println(io, "  Diagnostics:")
        for w in p.warnings
            println(io, "    ⚠ ", w)
        end
    end
    if !p.feasible || p.regime in (:diffusive, :marginal)
        # Compute τ for each recommendation
        τ_rec_N = 3.0 * 0.01 * p.recommended_N / p.Re + 0.5
        τ_rec_U = 3.0 * p.recommended_U_ref * p.N / p.Re + 0.5
        println(io, "  Recommendations:")
        println(io, "    → N = $(p.recommended_N) at U_ref = 0.01 → τ = $(round(τ_rec_N, digits=3))")
        println(io, "    → U_ref = $(round(p.recommended_U_ref, digits=6)) at N = $(p.N) → τ = $(round(τ_rec_U, digits=3))")
        # Low-Re advisory
        if p.Re < 0.1
            println(io, "  Note: Re = $(p.Re) < 0.1 — Stokes regime.")
            println(io, "    LBM is poorly suited for very low Re (τ grows as 1/Re).")
            println(io, "    Best achievable τ at N=10: $(round(3.0 * 0.058 * 10 / p.Re + 0.5, digits=1))")
        end
    end
end

# τ regime thresholds
const _TAU_UNSTABLE   = 0.5
const _TAU_MARGINAL   = 0.51
const _TAU_OPTIMAL_HI = 2.0
const _TAU_ACCEPT_HI  = 10.0
const _TAU_ABSURD     = 100.0

# Target τ values for recommendations (best → fallback)
const _TAU_TARGETS = [1.0, 1.5, 2.0, 5.0, 10.0]

"""Recommend best N for given (Re, U_ref), trying τ targets from ideal to fallback."""
function _recommend_N(Re, U_ref, N_min)
    for τ_t in _TAU_TARGETS
        N_rec = ceil(Int, Re * (τ_t - 0.5) / (3.0 * U_ref))
        N_rec >= N_min && return N_rec
    end
    return N_min  # best effort
end

"""Recommend best U_ref for given (Re, N), trying τ targets from ideal to fallback."""
function _recommend_U(Re, N, U_min, U_max)
    for τ_t in _TAU_TARGETS
        U_rec = Re * (τ_t - 0.5) / (3.0 * N)
        U_min ≤ U_rec ≤ U_max && return U_rec
    end
    # If all targets give U outside bounds, clamp to nearest feasible
    U_for_tau10 = Re * (_TAU_TARGETS[end] - 0.5) / (3.0 * N)
    return clamp(U_for_tau10, U_min, U_max)
end

"""
    lbm_params(; Re, N, U_ref=0.01, L_ref=N)

Compute all LBM lattice parameters from physical inputs and assess feasibility.

Returns an [`LBMParams`](@ref) with derived quantities, regime classification,
diagnostics, and concrete recommendations.

# Regimes (based on τ = 3ν + 0.5, where ν = U_ref × L_ref / Re)

| Regime       | τ range       | Meaning                                    |
|:-------------|:--------------|:-------------------------------------------|
| `:optimal`   | 0.51 – 2.0    | BGK accurate, best precision               |
| `:acceptable`| 2.0 – 10.0    | OK with MRT, some numerical diffusion       |
| `:marginal`  | 0.5 – 0.51    | Nearly unstable, MRT mandatory              |
| `:diffusive` | 10.0 – 100.0  | Collision quasi-inactive, MRT mandatory     |
| `:infeasible`| < 0.5 or > 100| Cannot run — parameters must change         |

# Additional constraints checked
- **Mach number**: Ma = U_ref × √3 ≤ 0.17 (compressibility)
- **CFL**: U_ref ≤ 0.3
- **Float32 precision**: U_ref ≥ 1e-5
- **Resolution**: N ≥ 10

# Examples
```julia
julia> lbm_params(Re=100, N=128)         # standard case
julia> lbm_params(Re=0.01, N=64)         # your problematic case
julia> lbm_params(Re=0.01, N=64, U_ref=0.001)  # with adjusted velocity
```
"""
function lbm_params(; Re::Real, N::Integer, U_ref::Real=0.01, L_ref::Real=N)
    Re = Float64(Re)
    N = Int(N)
    U_ref = Float64(U_ref)
    L_ref = Float64(L_ref)

    # --- Derived quantities ---
    ν   = U_ref * L_ref / Re
    τ   = 3ν + 0.5
    ω   = 1.0 / τ
    cs  = 1.0 / sqrt(3.0)
    Ma  = U_ref / cs

    warnings = String[]
    feasible = true

    # --- Regime classification ---
    regime = if τ < _TAU_UNSTABLE
        feasible = false
        push!(warnings, "τ = $(round(τ, digits=4)) < 0.5 — UNSTABLE (negative effective viscosity)")
        :infeasible
    elseif τ < _TAU_MARGINAL
        push!(warnings, "τ = $(round(τ, digits=4)) ≈ 0.5 — marginally stable, MRT mandatory")
        :marginal
    elseif τ ≤ _TAU_OPTIMAL_HI
        :optimal
    elseif τ ≤ _TAU_ACCEPT_HI
        push!(warnings, "τ = $(round(τ, digits=2)) — numerical diffusion significant, MRT recommended")
        :acceptable
    elseif τ ≤ _TAU_ABSURD
        push!(warnings, "τ = $(round(τ, digits=1)) — collision quasi-inactive (ω = $(round(ω, digits=5))), " *
                        "only $(round(ω*100, digits=1))% relaxation per step")
        :diffusive
    else
        feasible = false
        push!(warnings, "τ = $(round(τ, digits=0)) — absurd, advective physics dead " *
                        "(ω = $(round(ω, digits=6)))")
        :infeasible
    end

    # --- Mach / compressibility ---
    if Ma > 0.3 * sqrt(3.0)
        feasible = false
        push!(warnings, "Ma = $(round(Ma, digits=3)) > 0.52 — CFL violation, will diverge")
    elseif Ma > 0.1 * sqrt(3.0)
        push!(warnings, "Ma = $(round(Ma, digits=3)) > 0.17 — compressibility errors > 1%")
    end

    # --- Float32 precision ---
    if U_ref < 1e-5 && U_ref > 0
        push!(warnings, "U_ref = $(U_ref) — round-off dominates in Float32 " *
                        "(relative precision ~ $(round(eps(Float32)/U_ref, digits=0)))")
    elseif U_ref < 1e-3 && U_ref > 0
        push!(warnings, "U_ref = $(U_ref) — use Float64 for best accuracy")
    end

    # --- Resolution ---
    if N < 10
        push!(warnings, "N = $N — insufficient spatial resolution")
    end
    if Re > 0 && N / Re < 1.0
        push!(warnings, "N/Re = $(round(N/Re, digits=2)) < 1 — boundary layer under-resolved")
    end

    # --- Recommendations ---
    # Goal: find (U_ref, N) that gives τ in optimal range [0.55, 2.0]
    # Constraints: Ma ≤ 0.1 (U ≤ 0.058), U ≥ 1e-5, N ≥ 10
    #
    # τ = 3·U·N/Re + 0.5  ⟹  U·N = Re·(τ-0.5)/3
    #
    # Strategy: target τ = 1.0 (ideal). If that requires U < 1e-5 or N < 10,
    # relax τ target upward until feasible, capping at τ = 10 (MRT acceptable).
    U_max = 0.058  # Ma ≈ 0.1
    U_min = 1e-5
    N_min_rec = 10

    # For recommended_N: fix U = min(0.01, U_max) and solve N
    rec_U_fixed = min(0.01, U_max)
    rec_N = _recommend_N(Re, rec_U_fixed, N_min_rec)

    # For recommended_U: fix N and solve U
    rec_U_for_N = _recommend_U(Re, N, U_min, U_max)

    return LBMParams(Re, N, U_ref, ν, τ, ω, Ma, feasible, regime,
                     warnings, rec_N, rec_U_for_N)
end

"""
    lbm_params(setup::SimulationSetup)

Extract parameters from a parsed `.krk` setup and compute [`LBMParams`](@ref).
"""
function lbm_params(setup::SimulationSetup)
    params = setup.physics.params
    ν = get(params, :nu, NaN)
    isnan(ν) && error("No viscosity (nu) in setup — cannot compute LBM parameters.")
    Re = get(params, :Re, NaN)
    N = min(setup.domain.Nx, setup.domain.Ny)
    U_ref = _probe_U_ref(setup.boundaries)

    # If Re not stored, recompute from ν
    if isnan(Re) && U_ref > 0
        Re = U_ref * N / ν
    end

    return lbm_params(; Re=Re, N=N, U_ref=U_ref, L_ref=N)
end

"""
    lbm_params_table(; Re, N_range, U_ref=0.01)

Print a comparison table for multiple grid sizes at a given Re.

# Example
```julia
lbm_params_table(Re=0.01, N_range=[8, 16, 32, 64, 128])
```
"""
function lbm_params_table(; Re::Real, N_range, U_ref::Real=0.01)
    Re = Float64(Re)
    U_ref = Float64(U_ref)
    cs = 1.0 / sqrt(3.0)
    Ma = U_ref / cs

    println("LBM parameter space for Re = $Re, U_ref = $U_ref (Ma = $(round(Ma, digits=4)))")
    println("─"^78)
    println(rpad("N", 6), rpad("ν", 12), rpad("τ", 10), rpad("ω", 12),
            rpad("regime", 14), "status")
    println("─"^78)

    for N in N_range
        p = lbm_params(; Re=Re, N=Int(N), U_ref=U_ref)
        status = p.feasible ? "✓" : "✗"
        nw = length(p.warnings)
        extra = nw > 0 && p.feasible ? " ($(nw) warning$(nw>1 ? "s" : ""))" : ""
        println(rpad(N, 6),
                rpad(round(p.nu, digits=6), 12),
                rpad(round(p.tau, digits=4), 10),
                rpad(round(p.omega, digits=6), 12),
                rpad(p.regime, 14),
                status, extra)
    end

    println("─"^78)
    # Show best achievable configuration
    rec = lbm_params(; Re=Re, N=64, U_ref=U_ref)
    rec_τ = 3.0 * 0.01 * rec.recommended_N / Re + 0.5
    rec_regime = rec_τ < 0.51 ? "marginal" : rec_τ ≤ 2.0 ? "optimal" :
                 rec_τ ≤ 10.0 ? "acceptable" : rec_τ ≤ 100.0 ? "diffusive" : "infeasible"
    println("Best config: N = $(rec.recommended_N), U_ref = 0.01 → τ = $(round(rec_τ, digits=3)) ($rec_regime)")
    if Re < 0.1
        # Show the physical limit
        τ_min_possible = 3.0 * 0.058 * 10 / Re + 0.5  # N=10, Ma=0.1
        println("Low-Re limit: Re = $Re < 0.1, best possible τ ≈ $(round(τ_min_possible, digits=1)) (N=10, Ma=0.1)")
    end
end

