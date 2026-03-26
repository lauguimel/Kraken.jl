# API Reference

## Lattice Types

```@docs
AbstractLattice
D2Q9
D3Q19
```

## Lattice Functions

```@docs
lattice_dim
lattice_q
weights
cs2
equilibrium
opposite
```

## Configuration

```@docs
LBMConfig
omega
reynolds
```

## Initialization

```@docs
initialize_2d
initialize_3d
```

## Simulation Drivers

```@docs
run_cavity_2d
run_cavity_3d
run_poiseuille_2d
run_couette_2d
run_taylor_green_2d
run_cylinder_2d
run_rayleigh_benard_2d
run_hagen_poiseuille_2d
```

## Streaming Kernels

```@docs
stream_2d!
stream_3d!
stream_periodic_x_wall_y_2d!
stream_fully_periodic_2d!
```

## Collision Kernels

```@docs
collide_2d!
collide_3d!
collide_guo_2d!
collide_boussinesq_2d!
collide_boussinesq_vt_2d!
collide_thermal_2d!
collide_axisymmetric_2d!
collide_li_axisym_2d!
```

## Macroscopic Fields

```@docs
compute_macroscopic_2d!
compute_macroscopic_3d!
compute_macroscopic_forced_2d!
compute_temperature_2d!
```

## Boundary Conditions

```@docs
apply_zou_he_north_2d!
apply_zou_he_south_2d!
apply_zou_he_west_2d!
apply_zou_he_pressure_east_2d!
apply_zou_he_top_3d!
apply_fixed_temp_south_2d!
apply_fixed_temp_north_2d!
```

## Drag Computation

```@docs
compute_drag_mea_2d
```

## I/O

```@docs
write_vtk
create_pvd
write_vtk_to_pvd
```

## Module

```@docs
Kraken
```
