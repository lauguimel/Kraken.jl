# =====================================================================
# Modular viscoelastic specs — decouple polymer model and wall BC
# from the driver (same spirit as BCSpec2D for hydrodynamics).
#
# Dispatch compiles the right update/BC per spec type at first call.
# Adding a new polymer model or polymer wall BC means:
#   1. struct MyModel <: AbstractPolymerModel + fields
#   2. update_polymer_stress!(τ_p_*, C_*, model::MyModel)
#   3. (optional) custom C source override via collide_conformation_2d!
# No driver changes required.
# =====================================================================

"""
    AbstractPolymerModel

Constitutive closure mapping conformation tensor C to polymer stress τ_p.
Concrete subtypes: `OldroydB`, `FENEP` (planned), `Giesekus` (planned).
"""
abstract type AbstractPolymerModel end

"""
    OldroydB(; G, λ)

Linear Oldroyd-B: τ_p = G · (C − I) with G = ν_p / λ.
- `G`: polymer shear modulus
- `λ`: polymer relaxation time
"""
struct OldroydB{T<:AbstractFloat} <: AbstractPolymerModel
    G::T
    λ::T
end
OldroydB(; G, λ) = OldroydB(promote(float(G), float(λ))...)

polymer_modulus(m::OldroydB) = m.G
polymer_relaxation_time(m::OldroydB) = m.λ

"""
    LogConfOldroydB(; G, λ)

Oldroyd-B evolved in the log-conformation variable Ψ = log(C) (Fattal
& Kupferman 2004). Preserves positive-definiteness of C exactly at the
discrete level — required for stability at high Weissenberg numbers
where direct-C schemes blow up in extensional regions.

Stress reconstruction: C = exp(Ψ), τ_p = G · (C − I).

Use `collide_logconf_2d!` in place of `collide_conformation_2d!` in
the driver time loop when this model is active.
"""
struct LogConfOldroydB{T<:AbstractFloat} <: AbstractPolymerModel
    G::T
    λ::T
end
LogConfOldroydB(; G, λ) = LogConfOldroydB(promote(float(G), float(λ))...)

polymer_modulus(m::LogConfOldroydB) = m.G
polymer_relaxation_time(m::LogConfOldroydB) = m.λ

"""
    uses_log_conformation(model) -> Bool

Dispatch hook: `true` if the driver must evolve Ψ = log(C) instead of
C, and reconstruct C via `psi_to_C_2d!` before computing τ_p.
"""
uses_log_conformation(::AbstractPolymerModel) = false
uses_log_conformation(::LogConfOldroydB) = true

"""
    update_polymer_stress!(τ_p_xx, τ_p_xy, τ_p_yy, C_xx, C_xy, C_yy, model)

In-place update τ_p ← f(C) from the constitutive closure.
"""
function update_polymer_stress!(τ_p_xx, τ_p_xy, τ_p_yy,
                                  C_xx, C_xy, C_yy,
                                  m::OldroydB)
    G = eltype(τ_p_xx)(m.G)
    @. τ_p_xx = G * (C_xx - 1)
    @. τ_p_xy = G * C_xy
    @. τ_p_yy = G * (C_yy - 1)
    return nothing
end

# For LogConfOldroydB, callers pass C arrays that have already been
# reconstructed from Ψ via `psi_to_C_2d!`. The stress formula is then
# identical to the direct Oldroyd-B case.
function update_polymer_stress!(τ_p_xx, τ_p_xy, τ_p_yy,
                                  C_xx, C_xy, C_yy,
                                  m::LogConfOldroydB)
    G = eltype(τ_p_xx)(m.G)
    @. τ_p_xx = G * (C_xx - 1)
    @. τ_p_xy = G * C_xy
    @. τ_p_yy = G * (C_yy - 1)
    return nothing
end

# ---------------------------------------------------------------------
# Polymer wall BC
# ---------------------------------------------------------------------

"""
    AbstractPolymerWallBC

Boundary treatment for the conformation LBM populations g_* at fluid
cells adjacent to solid walls. Concrete types: `CNEBB`, `NoPolymerWallBC`.
"""
abstract type AbstractPolymerWallBC end

"""
    CNEBB()

Conservative non-equilibrium bounce-back (Liu 2025, Eq 38-39). Exactly
conserves each component of C at solid walls; required at high Wi.
"""
struct CNEBB <: AbstractPolymerWallBC end

"""
    NoPolymerWallBC()

Do nothing — let the underlying `stream_2d!` handle bounce-back as the
default. Use only if there are no solid walls adjacent to fluid cells.
"""
struct NoPolymerWallBC <: AbstractPolymerWallBC end

"""
    apply_polymer_wall_bc!(g_post, g_pre, is_solid, C, bc::AbstractPolymerWallBC)

Dispatch hook; default falls through for `NoPolymerWallBC`.
"""
apply_polymer_wall_bc!(g_post, g_pre, is_solid, C, ::NoPolymerWallBC) = nothing
function apply_polymer_wall_bc!(g_post, g_pre, is_solid, C, ::CNEBB)
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C)
    return nothing
end
