# Public API inventory

This inventory is derived from the exports in `src/Kraken.jl` for this
branch. Each exported name should either be documented in the API pages or
kept here with an explicit status.

## Lattices

| Export | Status |
|---|---|
| `AbstractLattice`, `D2Q9`, `D3Q19` | public |
| `lattice_dim`, `lattice_q`, `weights`, `velocities_x`, `velocities_y`, `velocities_z` | public helpers |
| `opposite`, `cs2`, `equilibrium` | public helpers |

See [Lattice](lattice.md).

## Collision and streaming

| Export | Status |
|---|---|
| `stream_2d!`, `stream_3d!` | public |
| `stream_periodic_x_wall_y_2d!`, `stream_fully_periodic_2d!` | public |
| `collide_2d!`, `collide_3d!` | public |
| `collide_guo_2d!`, `collide_guo_field_2d!` | public |
| `collide_guo_3d!`, `collide_guo_field_3d!` | public |

See [Streaming](streaming.md) and [Collision](collision.md).

## Macroscopic fields

| Export | Status |
|---|---|
| `compute_macroscopic_2d!`, `compute_macroscopic_3d!` | public |
| `compute_macroscopic_forced_2d!`, `compute_macroscopic_forced_3d!` | public |

See [Macroscopic](macroscopic.md).

## Boundary kernels

| Export | Status |
|---|---|
| `apply_zou_he_north_2d!`, `apply_zou_he_south_2d!`, `apply_zou_he_west_2d!` | public |
| `apply_zou_he_pressure_east_2d!`, `apply_extrapolate_east_2d!` | public |
| `apply_zou_he_top_3d!`, `apply_zou_he_bottom_3d!`, `apply_zou_he_west_3d!`, `apply_zou_he_east_3d!`, `apply_zou_he_south_3d!`, `apply_zou_he_north_3d!` | public |
| `apply_zou_he_pressure_east_3d!`, `apply_zou_he_pressure_top_3d!` | public |
| `apply_bounce_back_walls_2d!`, `apply_bounce_back_wall_2d!` | public |
| `apply_bounce_back_walls_3d!`, `apply_bounce_back_wall_3d!` | public |
| `apply_zou_he_north_spatial_2d!`, `apply_zou_he_south_spatial_2d!`, `apply_zou_he_west_spatial_2d!`, `apply_zou_he_pressure_east_spatial_2d!`, `apply_zou_he_pressure_inlet_west_2d!` | public but lower-level |

See [Boundary](boundary.md).

## Thermal kernels

| Export | Status |
|---|---|
| `collide_thermal_2d!`, `compute_temperature_2d!` | public |
| `collide_thermal_3d!`, `compute_temperature_3d!` | public |
| `apply_fixed_temp_south_2d!`, `apply_fixed_temp_north_2d!`, `apply_fixed_temp_west_2d!`, `apply_fixed_temp_east_2d!` | public |
| `apply_fixed_temp_west_3d!`, `apply_fixed_temp_east_3d!`, `apply_fixed_temp_south_3d!`, `apply_fixed_temp_north_3d!`, `apply_fixed_temp_bottom_3d!`, `apply_fixed_temp_top_3d!` | public |
| `collide_boussinesq_2d!`, `collide_boussinesq_vt_2d!`, `collide_boussinesq_vt_modified_2d!`, `collide_boussinesq_3d!` | public, lower-level |
| `fused_natconv_step!`, `fused_natconv_vt_step!` | public, used by thermal drivers |

## Drivers

| Export | Status |
|---|---|
| `LBMConfig`, `omega`, `reynolds` | public |
| `initialize_2d`, `initialize_3d` | public, lower-level |
| `run_cavity_2d`, `run_cavity_3d` | public |
| `run_poiseuille_2d`, `run_couette_2d` | public |
| `initialize_taylor_green_2d`, `run_taylor_green_2d` | public |
| `initialize_cylinder_2d`, `run_cylinder_2d`, `compute_drag_mea_2d` | public |
| `run_rayleigh_benard_2d`, `run_natural_convection_2d`, `run_natural_convection_3d` | public |

See [Drivers](drivers.md).

## I/O and diagnostics

| Export | Status |
|---|---|
| `write_vtk`, `create_pvd`, `write_vtk_to_pvd` | public |
| `setup_output_dir`, `write_snapshot_2d!`, `write_snapshot_3d!` | public |
| `open_paraview` | public |
| `DiagnosticsLogger`, `open_diagnostics`, `log_diagnostics!`, `close_diagnostics!` | public |

See [IO](io.md).

## `.krk` parser and setup types

| Export | Status |
|---|---|
| `KrakenExpr`, `parse_kraken_expr`, `evaluate`, `has_variable`, `is_time_dependent`, `is_spatial` | public |
| `SimulationSetup`, `DomainSetup`, `PhysicsSetup`, `GeometryRegion`, `BoundarySetup`, `InitialSetup`, `OutputSetup`, `DiagnosticsSetup`, `SanityIssue` | public data model |
| `load_kraken`, `parse_kraken`, `parse_kraken_sweep`, `load_kraken_sweep` | public |
| `sanity_check`, `sanity_check_sweep` | public |
| `LBMParams`, `lbm_params`, `lbm_params_table` | public parameter summary helpers |
| `run_simulation` | public `.krk` runner |

See [.krk overview](../krk/overview.md), [Config](config.md), and
[Sanity](../krk/sanity.md).

## Post-processing

| Export | Status |
|---|---|
| `extract_line`, `field_error`, `probe`, `domain_stats` | public |

See [Postprocess](postprocess.md).

## Known gaps

The inventory is now explicit, but several lower-level exports need fuller
docstrings and examples. Do not add new exports without updating this page and
the relevant API page.
