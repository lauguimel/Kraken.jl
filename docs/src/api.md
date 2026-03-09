# API Reference

## Operators

```@docs
laplacian!
gradient!
divergence!
advect!
```

## Solvers

```@docs
solve_poisson_fft!
solve_poisson_cg!
solve_poisson_neumann!
projection_step!
run_cavity
```

### Internal Types

```@docs
Kraken.NegLaplacianOperator
Kraken.NeumannLaplacianOperator
Kraken.JacobiPreconditioner
Kraken.NeumannJacobiPreconditioner
```

## I/O

```@docs
write_vtk
create_pvd
write_vtk_to_pvd
load_config
```

## Configuration Types

```@docs
SimulationConfig
GeometryConfig
PhysicsConfig
BCConfig
OutputConfig
StudyConfig
```

## Module

```@docs
Kraken
```

## Utilities

```@docs
available_backends
apply_velocity_bc!
apply_pressure_neumann_bc!
greet
```
