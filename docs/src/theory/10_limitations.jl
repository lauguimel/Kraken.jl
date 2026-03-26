# # Limitations of Standard LBM
#
# The Lattice Boltzmann Method is powerful, but it comes with inherent
# limitations that practitioners must understand. This page summarises the
# main constraints and how to work within them
# [Kruger et al. (2017)](@cite kruger2017lattice).
#
# ## Compressibility error
#
# LBM solves a weakly compressible form of the Navier--Stokes equations.
# The pressure relates to density through an equation of state:
#
# ```math
# p = \rho \, c_s^2 = \frac{\rho}{3}
# ```
#
# This means density fluctuations act as a proxy for pressure. The error
# compared to a truly incompressible solver scales as:
#
# ```math
# \text{error} = O(\mathrm{Ma}^2), \qquad \mathrm{Ma} = \frac{u}{c_s}
# ```
#
# !!! warning "The Ma < 0.1 rule"
#     To keep compressibility errors below 1%, the Mach number must satisfy
#     ``\mathrm{Ma} < 0.1``, or equivalently ``u < 0.058`` in lattice units.
#     This is the **single most important constraint** in LBM simulations.
#
# In practice, this means the lattice velocity `u_lid`, `u_max`, or any
# characteristic velocity must remain well below ``0.1``:

using Kraken

lattice = D2Q9()
cs = sqrt(cs2(lattice))
@show cs                      # ≈ 0.577

## Good: Ma = 0.1/0.577 ≈ 0.17 (borderline)
u_safe = 0.05
@show u_safe / cs

## Bad: Ma = 0.3/0.577 ≈ 0.52 (too high, results will be wrong)
u_bad = 0.3
@show u_bad / cs

# ## Stability: relaxation frequency bound
#
# The BGK relaxation frequency must satisfy:
#
# ```math
# 0 < \omega < 2
# ```
#
# - ``\omega \to 0``: very high viscosity, overdamped (accurate but slow).
# - ``\omega \to 2``: very low viscosity, high Reynolds number (fast but
#   prone to numerical instability).
#
# In practice, ``\omega > 1.95`` almost always leads to divergence.
# For stable production runs, keep ``\omega \leq 1.9``, which corresponds
# to ``\nu \geq 0.00175`` in lattice units.
#
# ```math
# \nu = \frac{1}{3}\left(\frac{1}{\omega} - \frac{1}{2}\right) \geq 0
# ```

## Minimum safe viscosity
ω_max = 1.9
ν_min = (1.0/3.0) * (1.0/ω_max - 0.5)
@show ν_min  # ≈ 0.00877

# ## Resolution requirements
#
# Under grid refinement, the key constraint is that all non-dimensional
# numbers (Re, Ma, Pr) must remain constant. If we refine by a factor 2
# (double ``N``):
#
# - ``\Delta x \to \Delta x / 2``
# - ``u_{\text{latt}} \to u_{\text{latt}} / 2`` (to keep Ma constant)
# - ``\nu_{\text{latt}} \to \nu_{\text{latt}} / 2`` (to keep Re constant)
# - ``\Delta t \to \Delta t / 4`` (since ``\Delta t \propto \Delta x^2 / \nu``)
#
# This means **doubling the resolution costs 8x in 2D** (4x more nodes,
# 2x more time steps) and **16x in 3D** (8x more nodes, 2x more time steps).
#
# !!! tip "Practical consequence"
#     Always run the coarsest grid that gives acceptable accuracy. Use the
#     Ma < 0.1 constraint and the stability limit on ``\omega`` to choose
#     the lattice velocity and viscosity.
#
# ## No adaptive mesh refinement
#
# Standard LBM requires a **uniform Cartesian grid**. Unlike finite volume or
# finite element methods, there is no straightforward way to locally refine the
# mesh near walls or features of interest. Multi-block and adaptive approaches
# exist but break the simplicity that makes LBM attractive.
#
# Kraken.jl currently uses uniform grids only (AMR is planned for V2).
#
# ## Limited to low-Mach flows
#
# Because the equilibrium distribution is a second-order Taylor expansion of
# the Maxwell--Boltzmann distribution, LBM is restricted to low-Mach-number
# flows. Compressible flows (shocks, supersonic regimes) require either
# higher-order equilibria or fundamentally different approaches.
#
# ## Summary of constraints
#
# | Constraint | Requirement | Consequence |
# |:-----------|:------------|:------------|
# | Low Mach | ``\mathrm{Ma} < 0.1`` | ``u_{\text{latt}} < 0.058`` |
# | Stability | ``\omega < 2`` | ``\nu > 0`` in lattice units |
# | Practical stability | ``\omega \leq 1.9`` | ``\nu \geq 0.009`` |
# | Uniform grid | Cartesian, ``\Delta x = \text{const}`` | No local refinement |
# | Refinement cost | ``\propto N^{D+1}`` | Expensive for high Re |
#
# Despite these limitations, LBM remains an excellent choice for
# low-to-moderate Reynolds number flows on regular geometries, especially
# when GPU acceleration makes the explicit time stepping very fast.
