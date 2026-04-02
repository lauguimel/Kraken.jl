# --- Effective viscosity functions for GNF models ---
#
# All functions are @inline for GPU kernel inlining.
# Julia's JIT specializes per concrete type → zero-cost abstraction.
#
# Each function returns the kinematic viscosity ν (not dynamic η).
# For LBM: ω = 1 / (3ν + 0.5).

"""
    effective_viscosity(model, gamma_dot) → ν

Compute the effective kinematic viscosity for a GNF model
at a given shear rate magnitude `γ̇`.
"""
function effective_viscosity end

"""
    effective_viscosity_thermal(model, gamma_dot, T_local) → ν

Compute the effective viscosity with thermal coupling.
Applies the thermal shift factor to the model parameters
before computing the viscosity from the shear rate.
"""
function effective_viscosity_thermal end

# ============================================================
# Newtonian
# ============================================================

@inline effective_viscosity(m::Newtonian, gamma_dot) = m.nu

@inline function effective_viscosity_thermal(m::Newtonian, gamma_dot, T_local)
    a_T = thermal_shift_factor(m.thermal, T_local)
    return m.nu * a_T
end

# ============================================================
# Power-law: η = K · γ̇^(n-1)
# ============================================================

@inline function effective_viscosity(m::PowerLaw, gamma_dot)
    T = typeof(m.K)
    gdot = max(gamma_dot, T(1e-30))  # avoid 0^(n-1) singularity
    nu = m.K * gdot^(m.n - one(T))
    return clamp(nu, m.nu_min, m.nu_max)
end

@inline function effective_viscosity_thermal(m::PowerLaw, gamma_dot, T_local)
    a_T = thermal_shift_factor(m.thermal, T_local)
    T = typeof(m.K)
    gdot = max(gamma_dot, T(1e-30))
    K_local = m.K * a_T
    nu = K_local * gdot^(m.n - one(T))
    return clamp(nu, m.nu_min, m.nu_max)
end

# ============================================================
# Carreau-Yasuda: η = η_∞ + (η_0 - η_∞) · (1 + (λγ̇)^a)^((n-1)/a)
# ============================================================

@inline function effective_viscosity(m::CarreauYasuda, gamma_dot)
    T = typeof(m.eta_0)
    lg = m.lambda * gamma_dot
    factor = (one(T) + lg^m.a)^((m.n - one(T)) / m.a)
    return m.eta_inf + (m.eta_0 - m.eta_inf) * factor
end

@inline function effective_viscosity_thermal(m::CarreauYasuda, gamma_dot, T_local)
    a_T = thermal_shift_factor(m.thermal, T_local)
    T = typeof(m.eta_0)
    eta_0_local = m.eta_0 * a_T
    eta_inf_local = m.eta_inf * a_T
    lambda_local = m.lambda * a_T
    lg = lambda_local * gamma_dot
    factor = (one(T) + lg^m.a)^((m.n - one(T)) / m.a)
    return eta_inf_local + (eta_0_local - eta_inf_local) * factor
end

# ============================================================
# Cross: η = η_∞ + (η_0 - η_∞) / (1 + (Kγ̇)^m)
# ============================================================

@inline function effective_viscosity(m::Cross, gamma_dot)
    T = typeof(m.eta_0)
    kg = m.K * gamma_dot
    return m.eta_inf + (m.eta_0 - m.eta_inf) / (one(T) + kg^m.m)
end

@inline function effective_viscosity_thermal(m::Cross, gamma_dot, T_local)
    a_T = thermal_shift_factor(m.thermal, T_local)
    T = typeof(m.eta_0)
    eta_0_local = m.eta_0 * a_T
    eta_inf_local = m.eta_inf * a_T
    K_local = m.K * a_T
    kg = K_local * gamma_dot
    return eta_inf_local + (eta_0_local - eta_inf_local) / (one(T) + kg^m.m)
end

# ============================================================
# Bingham (Papanastasiou): η = τ_y · (1 - exp(-m·γ̇)) / γ̇ + μ_p
# ============================================================

@inline function effective_viscosity(m::Bingham, gamma_dot)
    T = typeof(m.tau_y)
    gdot = max(gamma_dot, T(1e-30))
    return m.tau_y * (one(T) - exp(-m.m_reg * gdot)) / gdot + m.mu_p
end

@inline function effective_viscosity_thermal(m::Bingham, gamma_dot, T_local)
    a_T = thermal_shift_factor(m.thermal, T_local)
    T = typeof(m.tau_y)
    gdot = max(gamma_dot, T(1e-30))
    tau_y_local = m.tau_y * a_T
    mu_p_local = m.mu_p * a_T
    return tau_y_local * (one(T) - exp(-m.m_reg * gdot)) / gdot + mu_p_local
end

# ============================================================
# Herschel-Bulkley (Papanastasiou):
#   η = τ_y · (1 - exp(-m·γ̇)) / γ̇ + K · γ̇^(n-1)
# ============================================================

@inline function effective_viscosity(m::HerschelBulkley, gamma_dot)
    T = typeof(m.tau_y)
    gdot = max(gamma_dot, T(1e-30))
    yield_part = m.tau_y * (one(T) - exp(-m.m_reg * gdot)) / gdot
    power_part = m.K * gdot^(m.n - one(T))
    return yield_part + power_part
end

@inline function effective_viscosity_thermal(m::HerschelBulkley, gamma_dot, T_local)
    a_T = thermal_shift_factor(m.thermal, T_local)
    T = typeof(m.tau_y)
    gdot = max(gamma_dot, T(1e-30))
    tau_y_local = m.tau_y * a_T
    K_local = m.K * a_T
    yield_part = tau_y_local * (one(T) - exp(-m.m_reg * gdot)) / gdot
    power_part = K_local * gdot^(m.n - one(T))
    return yield_part + power_part
end
