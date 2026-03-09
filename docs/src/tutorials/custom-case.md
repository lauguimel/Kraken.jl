# Running from YAML Configuration

Kraken.jl supports YAML configuration files so you can define a simulation
without writing Julia code. This tutorial shows how to create, load, and use
a config file.

## Create a YAML file

Save the following as `my_case.yaml`:

```yaml
geometry:
  domain: [0.0, 1.0, 0.0, 1.0]
  resolution: [64, 64]

physics:
  Re: 400.0

boundary_conditions:
  top: {type: "moving_wall", velocity: 1.0}
  bottom: {type: "wall"}
  left: {type: "wall"}
  right: {type: "wall"}

study:
  type: "steady"
  max_steps: 20000
  tolerance: 1.0e-7

output:
  format: "vtk"
  directory: "results/"
  frequency: 1000
```

## Load and inspect

```julia
using Kraken

config = load_config("my_case.yaml")
println(config.physics)     # PhysicsConfig(Re=400.0)
println(config.geometry)    # GeometryConfig(64×64, domain [0,1]×[0,1])
```

The [`load_config`](@ref) function returns a [`SimulationConfig`](@ref) struct
with typed fields for geometry, physics, boundary conditions, study parameters,
and output settings.

## Run with configuration

Currently, you map the config to `run_cavity` keyword arguments:

```julia
u, v, p, converged = run_cavity(
    N  = config.geometry.resolution[1],
    Re = config.physics.Re,
    max_steps = config.study.max_steps,
    tol = config.study.tolerance,
    verbose = true,
)
```

A fully integrated YAML-driven simulation runner is planned for V1.1.

## VTK time series

For unsteady simulations, you can write a time series using PVD files:

```julia
pvd = create_pvd("results/cavity")

# Inside your time loop:
dx = 1.0 / 63
for step in 1:100
    # ... advance solution ...
    if step % 10 == 0
        time = step * dt
        write_vtk_to_pvd(pvd, "results/cavity", 64, 64, dx,
            Dict("velocity_x" => u, "velocity_y" => v, "pressure" => p),
            time)
    end
end
```

Open the resulting `results/cavity.pvd` file in ParaView to browse all
time steps with the animation controls.

## Next steps

- [Composing Your Own Solver](@ref) — build custom solvers from operators
- [API Reference](@ref) — full function documentation
