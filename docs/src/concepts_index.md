# Concepts index

This page maps the concepts that are public in Kraken.jl v0.1.0. It is
deliberately narrower than the development branches: if a feature is not
listed here, do not assume it is usable from this branch.

If you are brand new to the package, start with
[Getting started](getting_started.md), then return here for the theory links.

## `.krk` files

A `.krk` file is the primary user-facing description of a simulation: domain,
physics, boundaries, initial conditions, outputs and diagnostics live in one
small text file.

- Reference: [.krk overview](krk/overview.md), [Directives](krk/directives.md)
- Example: [.krk configuration](examples/10_krk_config.md)

## Lattices and equilibrium

LBM evolves discrete populations `f_i(x, t)` on a finite set of velocities.
Kraken uses D2Q9 in 2D and D3Q19 in 3D. Equilibrium is a low-Mach expansion
that recovers incompressible Navier-Stokes in the hydrodynamic limit.

- Theory: [LBM fundamentals](theory/01_lbm_fundamentals.md),
  [D2Q9 lattice](theory/02_d2q9_lattice.md),
  [From 2D to 3D](theory/06_from_2d_to_3d.md)
- Examples: [Poiseuille 2D](examples/01_poiseuille_2d.md),
  [Lid-driven cavity 3D](examples/05_cavity_3d.md)

## BGK collision

BGK is the public collision model in this branch. The relaxation parameter
sets the kinematic viscosity, and the same conceptual model is used in 2D and
3D.

- Theory: [BGK collision](theory/03_bgk_collision.md)
- Examples: [Lid-driven cavity 2D](examples/04_cavity_2d.md),
  [Taylor-Green vortex](examples/03_taylor_green_2d.md)

## Streaming

Streaming shifts post-collision populations to neighboring nodes along each
lattice direction. This is the non-local part of each LBM step.

- Theory: [Streaming](theory/04_streaming.md)
- Examples: [Couette flow](examples/02_couette_2d.md),
  [Poiseuille 2D](examples/01_poiseuille_2d.md)

## Boundary conditions

The public boundary set covers bounce-back walls, Zou-He velocity/pressure
forms, periodic axes, and scalar fixed-temperature walls for thermal runs.
Spatial expressions are supported on selected 2D velocity/pressure paths.

- Theory: [Boundary conditions](theory/05_boundary_conditions.md),
  [Spatial BCs](theory/19_spatial_bcs.md)
- Reference: [BC types](krk/bc_types.md)
- Examples: [Cylinder 2D](examples/06_cylinder_2d.md),
  [Heat conduction](examples/07_heat_conduction.md)

## Body forces

Guo forcing is used for body-force-driven channels and for Boussinesq thermal
coupling.

- Theory: [Body forces](theory/07_body_forces.md)
- Examples: [Poiseuille 2D](examples/01_poiseuille_2d.md),
  [Rayleigh-Benard convection](examples/08_rayleigh_benard.md)

## Thermal coupling

The thermal model uses a second distribution function for temperature. The
temperature field can be passive or coupled back to the velocity field through
Boussinesq buoyancy.

- Theory: [Thermal DDF](theory/08_thermal_ddf.md)
- Examples: [Heat conduction](examples/07_heat_conduction.md),
  [Rayleigh-Benard convection](examples/08_rayleigh_benard.md)

## Outputs and post-processing

Kraken writes VTK/PVD files for ParaView and can produce image/GIF outputs
through the CairoMakie extension. Post-processing helpers extract lines,
probe fields and compute error metrics.

- API: [IO](api/io.md), [Postprocess](api/postprocess.md)

## Explicitly not public here

MRT, axisymmetric LBM, grid refinement, VOF, phase-field, Shan-Chen, species
transport, rheology, viscoelasticity and SLBM/body-fitted work are not public
features of this branch. Some are present in other branches, especially
`slbm-paper`, but they require their own validation and documentation pass.

See [Capabilities](capabilities.md) for the precise status matrix.
