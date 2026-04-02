# --- Non-Newtonian rheology model hierarchy ---
#
# All structs are lightweight (no arrays) and can be passed directly
# as GPU kernel arguments. Julia's JIT specializes kernels per concrete
# type → zero-overhead dispatch on GPU.

# ============================================================
# Thermal coupling (temperature-dependent rheological parameters)
# ============================================================

"""
    AbstractThermalCoupling

Base type for temperature dependence of rheological parameters.
Dispatched at compile-time: `IsothermalCoupling` generates no extra code.
"""
abstract type AbstractThermalCoupling end

"""
    IsothermalCoupling()

No temperature dependence. The JIT eliminates all thermal-shift code paths.
"""
struct IsothermalCoupling <: AbstractThermalCoupling end

"""
    ArrheniusCoupling(T_ref, E_a)

Arrhenius-type thermal shift: `p(T) = p_ref · exp(E_a · (1/T - 1/T_ref))`.
All rheological parameters (η, λ, K, τ_y, …) are shifted by the same factor `a_T`.
"""
struct ArrheniusCoupling{T} <: AbstractThermalCoupling
    T_ref::T
    E_a::T
end

"""
    WLFCoupling(T_ref, C1, C2)

Williams-Landel-Ferry shift: `log(a_T) = -C1·(T - T_ref) / (C2 + T - T_ref)`.
"""
struct WLFCoupling{T} <: AbstractThermalCoupling
    T_ref::T
    C1::T
    C2::T
end

# ============================================================
# Rheology model hierarchy
# ============================================================

"""
    AbstractRheology

Root type for all rheological models (GNF and viscoelastic).
"""
abstract type AbstractRheology end

"""
    GeneralizedNewtonian <: AbstractRheology

Generalized Newtonian Fluid: viscosity depends on local strain rate γ̇.
"""
abstract type GeneralizedNewtonian <: AbstractRheology end

"""
    Viscoelastic <: AbstractRheology

Viscoelastic fluid: requires evolution of a conformation/stress tensor.
"""
abstract type Viscoelastic <: AbstractRheology end

# ============================================================
# Generalized Newtonian models
# ============================================================

"""
    Newtonian(nu; thermal=IsothermalCoupling())

Constant viscosity. Provided for API consistency so that two-phase
simulations can mix Newtonian and non-Newtonian phases.
"""
struct Newtonian{T,TC<:AbstractThermalCoupling} <: GeneralizedNewtonian
    nu::T
    thermal::TC
end
Newtonian(nu; thermal=IsothermalCoupling()) = Newtonian(nu, thermal)

"""
    PowerLaw(K, n; nu_min=1e-6, nu_max=10.0, thermal=IsothermalCoupling())

Power-law model: `η(γ̇) = K · γ̇^(n-1)`.
- `n < 1`: shear-thinning
- `n > 1`: shear-thickening
- `nu_min/nu_max`: stability cutoffs (clamped).
"""
struct PowerLaw{T,TC<:AbstractThermalCoupling} <: GeneralizedNewtonian
    K::T
    n::T
    nu_min::T
    nu_max::T
    thermal::TC
end
function PowerLaw(K, n; nu_min=1e-6, nu_max=10.0, thermal=IsothermalCoupling())
    T = promote_type(typeof(K), typeof(n), typeof(nu_min), typeof(nu_max))
    PowerLaw(T(K), T(n), T(nu_min), T(nu_max), thermal)
end

"""
    CarreauYasuda(eta_0, eta_inf, lambda, a, n; thermal=IsothermalCoupling())

Carreau-Yasuda model:
`η(γ̇) = η_∞ + (η_0 - η_∞) · (1 + (λγ̇)^a)^((n-1)/a)`.
Set `a = 2` for the standard Carreau model.
"""
struct CarreauYasuda{T,TC<:AbstractThermalCoupling} <: GeneralizedNewtonian
    eta_0::T
    eta_inf::T
    lambda::T
    a::T
    n::T
    thermal::TC
end
function CarreauYasuda(eta_0, eta_inf, lambda, a, n; thermal=IsothermalCoupling())
    T = promote_type(typeof(eta_0), typeof(eta_inf), typeof(lambda), typeof(a), typeof(n))
    CarreauYasuda(T(eta_0), T(eta_inf), T(lambda), T(a), T(n), thermal)
end

"""
    Cross(eta_0, eta_inf, K, m; thermal=IsothermalCoupling())

Cross model: `η(γ̇) = η_∞ + (η_0 - η_∞) / (1 + (Kγ̇)^m)`.
"""
struct Cross{T,TC<:AbstractThermalCoupling} <: GeneralizedNewtonian
    eta_0::T
    eta_inf::T
    K::T
    m::T
    thermal::TC
end
function Cross(eta_0, eta_inf, K, m; thermal=IsothermalCoupling())
    T = promote_type(typeof(eta_0), typeof(eta_inf), typeof(K), typeof(m))
    Cross(T(eta_0), T(eta_inf), T(K), T(m), thermal)
end

"""
    Bingham(tau_y, mu_p; m_reg=1000.0, thermal=IsothermalCoupling())

Bingham plastic with Papanastasiou regularization (branchless, GPU-safe):
`η(γ̇) = τ_y · (1 - exp(-m·γ̇)) / γ̇ + μ_p`.
Larger `m_reg` → closer to ideal Bingham behavior.
"""
struct Bingham{T,TC<:AbstractThermalCoupling} <: GeneralizedNewtonian
    tau_y::T
    mu_p::T
    m_reg::T
    thermal::TC
end
function Bingham(tau_y, mu_p; m_reg=1000.0, thermal=IsothermalCoupling())
    T = promote_type(typeof(tau_y), typeof(mu_p), typeof(m_reg))
    Bingham(T(tau_y), T(mu_p), T(m_reg), thermal)
end

"""
    HerschelBulkley(tau_y, K, n; m_reg=1000.0, thermal=IsothermalCoupling())

Herschel-Bulkley model with Papanastasiou regularization:
`η(γ̇) = τ_y · (1 - exp(-m·γ̇)) / γ̇ + K · γ̇^(n-1)`.
Reduces to Bingham when `n = 1`, to power-law when `τ_y = 0`.
"""
struct HerschelBulkley{T,TC<:AbstractThermalCoupling} <: GeneralizedNewtonian
    tau_y::T
    K::T
    n::T
    m_reg::T
    thermal::TC
end
function HerschelBulkley(tau_y, K, n; m_reg=1000.0, thermal=IsothermalCoupling())
    T = promote_type(typeof(tau_y), typeof(K), typeof(n), typeof(m_reg))
    HerschelBulkley(T(tau_y), T(K), T(n), T(m_reg), thermal)
end

# ============================================================
# Viscoelastic formulations
# ============================================================

"""
    VEFormulation

Compile-time tag for the viscoelastic evolution strategy.
"""
abstract type VEFormulation end

"""
    StressFormulation()

Evolve the polymeric stress tensor τ_p directly.
Simple but unstable at high Weissenberg numbers (loss of positive-definiteness).
"""
struct StressFormulation <: VEFormulation end

"""
    LogConfFormulation()

Evolve Θ = log(C) (log-conformation, Fattal & Kupferman 2004).
Guarantees positive-definite C → stable at high Wi.
Requires 2×2 eigen-decomposition (analytical in 2D, branchless).
"""
struct LogConfFormulation <: VEFormulation end

# ============================================================
# Viscoelastic models
# ============================================================

"""
    OldroydB(nu_s, nu_p, lambda; formulation=LogConfFormulation(), thermal=IsothermalCoupling())

Oldroyd-B model: linear viscoelastic with constant-viscosity solvent.
- `nu_s`: solvent kinematic viscosity
- `nu_p`: polymeric kinematic viscosity
- `lambda`: polymer relaxation time
"""
struct OldroydB{T,F<:VEFormulation,TC<:AbstractThermalCoupling} <: Viscoelastic
    nu_s::T
    nu_p::T
    lambda::T
    formulation::F
    thermal::TC
end
function OldroydB(nu_s, nu_p, lambda; formulation=LogConfFormulation(), thermal=IsothermalCoupling())
    T = promote_type(typeof(nu_s), typeof(nu_p), typeof(lambda))
    OldroydB(T(nu_s), T(nu_p), T(lambda), formulation, thermal)
end

"""
    FENEP(nu_s, nu_p, lambda, L_max; formulation=LogConfFormulation(), thermal=IsothermalCoupling())

FENE-P model: finitely extensible nonlinear elastic with Peterlin closure.
- `L_max`: maximum extensibility of polymer chains
- Peterlin function: `f(tr(C)) = L²/(L² - tr(C))`
- Reduces to Oldroyd-B when `L_max → ∞`.
"""
struct FENEP{T,F<:VEFormulation,TC<:AbstractThermalCoupling} <: Viscoelastic
    nu_s::T
    nu_p::T
    lambda::T
    L_max::T
    formulation::F
    thermal::TC
end
function FENEP(nu_s, nu_p, lambda, L_max; formulation=LogConfFormulation(), thermal=IsothermalCoupling())
    T = promote_type(typeof(nu_s), typeof(nu_p), typeof(lambda), typeof(L_max))
    FENEP(T(nu_s), T(nu_p), T(lambda), T(L_max), formulation, thermal)
end

"""
    Saramito(nu_s, nu_p, lambda, tau_y; n=1.0, m_reg=1000.0,
             formulation=LogConfFormulation(), thermal=IsothermalCoupling())

Saramito model: elasto-viscoplastic (combines yield stress + viscoelasticity).
- Below yield stress: Kelvin-Voigt solid behavior
- Above yield stress: Oldroyd-B-like viscoelastic flow with power-law viscosity
- Papanastasiou regularization for the yield criterion.
"""
struct Saramito{T,F<:VEFormulation,TC<:AbstractThermalCoupling} <: Viscoelastic
    nu_s::T
    nu_p::T
    lambda::T
    tau_y::T
    n::T
    m_reg::T
    formulation::F
    thermal::TC
end
function Saramito(nu_s, nu_p, lambda, tau_y; n=1.0, m_reg=1000.0,
                  formulation=LogConfFormulation(), thermal=IsothermalCoupling())
    T = promote_type(typeof(nu_s), typeof(nu_p), typeof(lambda), typeof(tau_y), typeof(n), typeof(m_reg))
    Saramito(T(nu_s), T(nu_p), T(lambda), T(tau_y), T(n), T(m_reg), formulation, thermal)
end

# ============================================================
# Thermal shift helper
# ============================================================

"""
    thermal_shift_factor(tc, T_local) → a_T

Compute the thermal shift factor for a given local temperature.
All rheological parameters are multiplied by `a_T`.
"""
@inline thermal_shift_factor(::IsothermalCoupling, T_local) = one(T_local)

@inline function thermal_shift_factor(tc::ArrheniusCoupling, T_local)
    return exp(tc.E_a * (one(T_local) / T_local - one(T_local) / tc.T_ref))
end

@inline function thermal_shift_factor(tc::WLFCoupling, T_local)
    dT = T_local - tc.T_ref
    return exp(-tc.C1 * dT / (tc.C2 + dT))
end
