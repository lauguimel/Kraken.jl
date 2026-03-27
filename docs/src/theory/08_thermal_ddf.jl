# # Thermal LBM: Double Distribution Function
#
# Many engineering flows involve heat transfer alongside fluid motion.
# The **Double Distribution Function** (DDF) approach introduces a second
# set of populations to track the temperature field, while the original
# populations handle momentum [He, Chen & Doolen (1998)](@cite he1998novel).
#
# ## Why a separate distribution?
#
# One could add energy as a higher moment of the momentum distribution,
# but this "multispeed" approach requires larger lattices (e.g., D2Q17) and
# suffers from numerical instability. The DDF approach is simpler and more
# robust: it uses two standard D2Q9 lattices, one for ``f_q`` (momentum) and
# one for ``g_q`` (temperature), each with its own relaxation rate.
#
# ## Temperature distribution: ``g_q``
#
# The temperature populations ``g_q`` evolve with their own collision--streaming
# cycle:
#
# ```math
# g_q(\mathbf{x} + \mathbf{e}_q, \, t+1)
# = g_q(\mathbf{x}, t)
#   - \omega_T \big[ g_q(\mathbf{x}, t) - g_q^{\mathrm{eq}} \big]
# ```
#
# The thermal equilibrium is a **first-order** expansion (no ``u^2`` term,
# since temperature is a passive scalar):
#
# ```math
# g_q^{\mathrm{eq}} = w_q \, T \left( 1 + \frac{\mathbf{e}_q \cdot \mathbf{u}}{c_s^2} \right)
# = w_q \, T \big( 1 + 3 \, \mathbf{e}_q \cdot \mathbf{u} \big)
# ```
#
# !!! note "Key difference from momentum equilibrium"
#     The thermal equilibrium is *linear* in velocity. The ``u^2`` terms are
#     absent because the temperature equation is an advection--diffusion
#     equation, not a momentum equation.
#
# ## Thermal relaxation and diffusivity
#
# The thermal relaxation frequency ``\omega_T`` relates to the thermal
# diffusivity ``\alpha`` exactly like the momentum case:
#
# ```math
# \alpha = c_s^2 \left( \tau_T - \frac{1}{2} \right)
#        = \frac{1}{3}\left( \frac{1}{\omega_T} - \frac{1}{2} \right)
# ```
#
# Inverting: ``\omega_T = 2 / (6\alpha + 1)``.
#
# The **Prandtl number** links both relaxations:
#
# ```math
# \mathrm{Pr} = \frac{\nu}{\alpha}
# = \frac{\tau - 1/2}{\tau_T - 1/2}
# ```
#
# This means ``\nu`` and ``\alpha`` (hence ``\omega`` and ``\omega_T``) can be
# set independently, unlike some thermal LBM approaches.
#
# ## Dirichlet temperature boundaries: anti-bounce-back
#
# For a fixed-temperature wall at ``T = T_w``, the **anti-bounce-back** rule
# sets the unknown incoming temperature populations:
#
# ```math
# g_{\bar{q}} = -g_q^{\star} + 2 \, w_q \, T_w
# ```
#
# This is the thermal analogue of the standard bounce-back for velocity and
# achieves second-order accuracy for the Dirichlet condition.
#
# ## Boussinesq coupling
#
# For natural convection, temperature drives the flow through buoyancy.
# Under the Boussinesq approximation, the density is constant except in
# the gravity term [Guo et al. (2002)](@cite guo2002boussinesq):
#
# ```math
# F_y = \rho \, \beta \, g \, (T - T_{\mathrm{ref}})
# ```
#
# where ``\beta`` is the thermal expansion coefficient, ``g`` the gravitational
# acceleration, and ``T_{\mathrm{ref}}`` a reference temperature. This force
# is injected into the momentum equation using the Guo forcing scheme.
#
# In Kraken.jl, this coupling is handled by `collide_boussinesq_2d!`, which
# combines BGK collision with the Boussinesq body force in a single kernel
# pass.
#
# ## Temperature-dependent viscosity
#
# For some applications, viscosity varies with temperature following an
# Arrhenius-type law. `collide_boussinesq_vt_2d!` extends the Boussinesq
# kernel with a spatially varying relaxation frequency ``\omega(\mathbf{x})``:
#
# ```math
# \omega(T) = \frac{2}{6\,\nu(T) + 1}
# ```
#
# The viscosity field is recomputed from the temperature at each time step.
#
# ## Kraken.jl thermal API

using Kraken

lattice = D2Q9()

## Thermal relaxation for Pr = 0.71 (air) with ν = 0.01
ν = 0.01
Pr = 0.71
α = ν / Pr
ω_T = 2.0 / (6.0 * α + 1.0)
@show α
@show ω_T

# Available thermal kernels:
#
# | Kernel | Purpose |
# |:-------|:--------|
# | `collide_thermal_2d!` | BGK collision for ``g_q`` populations |
# | `compute_temperature_2d!` | Recover ``T = \sum_q g_q`` |
# | `apply_fixed_temp_south_2d!` | Anti-bounce-back at south wall |
# | `apply_fixed_temp_north_2d!` | Anti-bounce-back at north wall |
# | `collide_boussinesq_2d!` | Momentum collision + buoyancy force |
# | `collide_boussinesq_vt_2d!` | Same, with ``\nu(T)`` coupling |
#
# A typical Rayleigh--Benard convection loop looks like:
#
# ```julia
# for step in 1:max_steps
#     stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
#     stream_periodic_x_wall_y_2d!(g_out, g_in, Nx, Ny)
#     apply_fixed_temp_south_2d!(g_out, T_hot, Nx)
#     apply_fixed_temp_north_2d!(g_out, T_cold, Nx, Ny)
#     compute_temperature_2d!(Temp, g_out)
#     compute_macroscopic_2d!(ρ, ux, uy, f_out)
#     collide_thermal_2d!(g_out, ux, uy, ω_T)
#     collide_boussinesq_2d!(f_out, Temp, is_solid, ω_f, β_g, T_ref)
#     f_in, f_out = f_out, f_in
#     g_in, g_out = g_out, g_in
# end
# ```
