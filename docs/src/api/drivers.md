# Drivers

Drivers are the high-level Julia entry points that combine initialization,
streaming, collision, boundary conditions and output. The `.krk` runner
eventually dispatches to the same public paths where possible.

## Quick reference

| Symbol | Purpose |
|---|---|
| `run_simulation` | Top-level `.krk` dispatcher |
| `load_kraken` / `parse_kraken` | Parse `.krk` files/text into `SimulationSetup` |
| `run_cavity_2d` | Lid-driven cavity, D2Q9 |
| `run_cavity_3d` | Lid-driven cavity, D3Q19 |
| `run_poiseuille_2d` | Body-force channel flow |
| `run_couette_2d` | Moving-wall Couette flow |
| `initialize_taylor_green_2d` | Taylor-Green analytic initial condition |
| `run_taylor_green_2d` | Taylor-Green vortex decay |
| `initialize_cylinder_2d` | Cylinder obstacle mask initializer |
| `run_cylinder_2d` | Flow past a cylinder, D2Q9 |
| `compute_drag_mea_2d` | Momentum-exchange drag helper |
| `run_rayleigh_benard_2d` | Rayleigh-Benard convection |
| `run_natural_convection_2d` | Natural convection cavity, 2D |
| `run_natural_convection_3d` | Natural convection cavity, 3D |
| `fused_natconv_step!` | Fused thermal step used by natural convection |
| `fused_natconv_vt_step!` | Fused thermal step with variable-viscosity path |

Not public in this branch: `run_hagen_poiseuille_2d`,
`run_natural_convection_refined_2d`, fused BGK/AA persistent kernels, VOF,
rheology and SLBM/body-fitted drivers.

## `.krk` dispatch

```julia
using Kraken

result = run_simulation("examples/cavity.krk")
result = run_simulation("examples/cavity.krk"; max_steps=100, Nx=64, Ny=64)
```

`max_steps` and parameter overrides are accepted by the file-path method,
which reparses the `.krk` file. The parsed-setup method is:

```julia
setup = load_kraken("examples/cavity.krk")
result = run_simulation(setup)
```

In v0.1.0, `run_simulation(setup)` rejects unsupported modules and refinement
blocks before running. That is intentional: the parser may know syntax that is
reserved for development branches, but the runner does not advertise it as
usable here.

## Examples

```julia
run_poiseuille_2d(; Nx=4, Ny=64, ν=0.1, Fx=2e-6, max_steps=100_000)
run_taylor_green_2d(; N=64, ν=0.01, u0=0.01, max_steps=1000)
run_natural_convection_2d(; N=64, Ra=1e3, Pr=0.71, max_steps=30_000)
```

Most public examples are better run through their `.krk` files first; use the
driver calls when you need loops, callbacks, custom arrays or benchmark code.
