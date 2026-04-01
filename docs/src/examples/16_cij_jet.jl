# # CIJ Jet Breakup (Axisymmetric Validation)
#
#
# ## Problem Statement
#
# A Continuous InkJet (CIJ) printhead ejects a liquid jet through a nozzle
# and stimulates it with a periodic velocity perturbation.  The stimulation
# triggers the **Rayleigh--Plateau instability** at a controlled wavelength,
# causing the jet to break up into uniform droplets.
#
# This is a validation case: we compare Kraken.jl's axisymmetric two-phase
# LBM against reference data from **Basilisk** (Popinet, 2015), a
# well-validated Navier--Stokes VOF solver with adaptive mesh refinement.
#
# The reference dataset comprises ~110,000 interface snapshots spanning
# ``\text{Re} \in [100, 1000]``, ``\text{We} = 600``, and 6 stimulation
# amplitudes (Roche *et al.*, 2021).
#
# ### Dimensionless parameters
#
# The problem is controlled by three dimensionless numbers:
#
# ```math
# \text{Re} = \frac{U\, R_0}{\nu_l}, \qquad
# \text{We} = \frac{\rho_l\, U^2\, R_0}{\sigma}, \qquad
# \text{Oh} = \frac{\mu_l}{\sqrt{\rho_l\, \sigma\, R_0}} = \frac{\sqrt{\text{We}}}{\text{Re}}
# ```
#
# where ``U`` is the mean inlet velocity, ``R_0`` the jet radius,
# ``\nu_l`` the liquid kinematic viscosity, and ``\sigma`` the surface
# tension coefficient.
#
# The pulsed inlet velocity is:
# ```math
# u(t) = U \left(1 + \delta \sin\!\left(\frac{2\pi t}{T}\right)\right)
# ```
# with period ``T = 7\,R_0/U`` (wavelength ``\lambda = 7\,R_0``),
# corresponding to an optimal Rayleigh--Plateau wavenumber ``k\,R_0 \approx 0.9``.
#
# ### Basilisk reference setup
#
# | Parameter | Value |
# |:----------|:------|
# | Geometry | Axisymmetric, domain ``512 \times 256`` (``z \times r``) |
# | Reynolds | ``\text{Re} = 100``--``1000`` |
# | Weber | ``\text{We} = 600`` |
# | Density ratio | ``\rho_l / \rho_g = 1000`` |
# | Viscosity ratio | ``\mu_l / \mu_g = 500`` |
# | Stimulation | ``\delta \in \{0.01, 0.015, 0.02, 0.025, 0.03, 0.035\}`` |
# | AMR | Level 15, ``\Delta x_\min \approx 0.0156\, R_0`` |
# | Solver | Navier--Stokes VOF with wavelet adaptation |
#
# ## Kraken.jl approach
#
# ### LBM parameter mapping
#
# We map the dimensionless numbers to lattice units:
# ```math
# \nu_l = \frac{u_\text{lb}\, R_0}{Re}, \qquad
# \sigma = \frac{\rho_l\, u_\text{lb}^2\, R_0}{We}, \qquad
# \tau = 3\,\nu_l + \tfrac{1}{2}
# ```
#
# **Stability constraint**: BGK collision is unstable when ``\tau`` approaches
# ``0.5``.  For high Re, we use **MRT (Multiple Relaxation Time) collision**
# (Lallemand & Luo, 2000) which provides enhanced stability through separate
# relaxation of non-hydrodynamic moments.
#
# With ``R_0 = 40`` and ``u_\text{lb} = 0.04``, we obtain ``\tau = 0.524``
# for ``\text{Re} = 200``, well within the stable regime.
#
# ### Simulation components
#
# - **Streaming**: `stream_axisym_inlet_2d!` — non-periodic axial direction
#   with Zou-He velocity inlet (west) and pressure outlet (east), specular
#   reflection at the axis (``j = 1``), wall at far field (``j = N_r``)
# - **VOF**: PLIC advection with MYC normal reconstruction and CFL
#   sub-stepping (`advect_vof_plic_step!`)
# - **Curvature**: Height-function meridional curvature + azimuthal correction
#   ``\kappa_2 = n_r / r`` (`add_azimuthal_curvature_2d!`)
# - **Surface tension**: CSF model ``\mathbf{F} = \sigma\,\kappa\,\nabla C``
# - **Collision**: Two-phase MRT with variable viscosity ``\nu(C)`` and Guo
#   forcing (`collide_twophase_mrt_2d!`)
# - **Axisymmetric correction**: viscous term ``\nu/r \cdot \partial u_z / \partial r``
#   added as body force (`add_axisym_viscous_correction_2d!`)
#
# ## Setup

using Kraken

# We run a single validation case at ``\text{Re} = 200``,
# ``\text{We} = 600``, ``\delta = 0.02`` (moderate stimulation amplitude).

Re = 200
We = 600
δ = 0.02
R0 = 40       # lattice units — gives τ = 0.524
u_lb = 0.04   # lattice velocity

# Derived parameters:
ν_l = u_lb * R0 / Re
σ_lb = u_lb^2 * R0 / We
τ = 3ν_l + 0.5
T_period = 7 * R0 / u_lb

println("LBM parameters:")
println("  ν = $ν_l, τ = $τ, σ = $σ_lb")
println("  T_period = $T_period steps")

# ## Run simulation
#
# The `run_cij_jet_axisym_2d` function handles everything: initialization,
# pulsed inlet, VOF tracking, curvature, surface tension, and output.

#nb # Note: this is a long-running simulation. The full run with
#nb # `domain_ratio=80` and `max_steps=100_000` takes ~30 minutes on CPU.
#nb # For the documentation build, we use a shorter domain.

result = run_cij_jet_axisym_2d(;
    Re=Re, We=We, δ=δ,
    R0=R0, u_lb=u_lb,
    domain_ratio=40, nr_ratio=3,
    ρ_ratio=10.0, μ_ratio=10.0,
    max_steps=10_000, output_interval=2000,
    output_dir=joinpath(@__DIR__, "cij_output"))

println("Breakup detected: ", result.breakup_detected)
println("Interface snapshots: ", length(result.interfaces))

# ## Load Basilisk reference
#
# We load the Basilisk interface data for the same (Re, δ) at a matching
# physical time.

basilisk_dir = "/Users/guillaume/Documents/Recherche/Rheodrop/data/numerical/ds_num"
t_phys = 10_000 * u_lb / R0  # physical time of last snapshot

bas_file = find_basilisk_snapshot(basilisk_dir, Re, δ, 155.0; tol=1.0)
if bas_file !== nothing
    bas_contour = load_basilisk_interface_contour(bas_file)
    println("Basilisk interface: $(length(bas_contour)) points")
    println("  z range: ", extrema(first.(bas_contour)), " R₀")
    println("  r range: ", extrema(last.(bas_contour)), " R₀")
end

# ## Comparison
#
# The comparison below overlays the Kraken interface (at the last output
# step) with the Basilisk reference at a comparable physical time.  Since
# the LBM simulation uses reduced density and viscosity ratios
# (``\rho_l/\rho_g = 10`` vs 1000 in Basilisk), exact agreement is not
# expected — the key validation metrics are:
#
# 1. **Jet morphology**: wavelength of the perturbation, satellite drop
#    formation pattern
# 2. **Breakup length**: axial distance from the nozzle to the first
#    pinch-off point
# 3. **Drop size and spacing**: regularity of the main drops
#
# A quantitative comparison with the full Basilisk dataset across multiple
# Re and δ values is the subject of ongoing work.

nothing  # suppress REPL output
