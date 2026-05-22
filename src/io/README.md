# io/

The package boundary with the outside world — `.krk` DSL parsing,
VTK output for ParaView, STL voxelisation for immersed-body simulations.

## Key entry points

| File | Symbol | Purpose |
|---|---|---|
| `kraken_parser.jl` | `parse_krk`, `run_simulation(path::String)` | Parses a `.krk` file → typed config |
| `expression.jl` | Spatial / temporal BC expressions | Evaluates `boundary north velocity (u_max * sin(2π*y/Ny), 0)` and similar |
| `vtk_writer.jl` | `write_vtk`, `write_pvd` | ParaView output (`.vti`, `.pvd` time series) |
| `diagnostics.jl` | `write_diagnostic_csv` | Per-step time series (force, Cd, mass drift, etc.) |
| `stl_reader.jl` | `read_stl` | Read triangulated surfaces |
| `voxelizer.jl` | `voxelize_stl` | STL → boolean voxel mask for immersed bodies |
| `stl_libb.jl` | STL → LI-BB geometry | Generate cut-link `q_wall` arrays for curved boundary kernels |

## Critical invariants

- **`.krk` is the canonical configuration surface**: every accepted
  syntax must be parseable by `kraken_parser.jl`. Adding a new flow
  capability requires adding the parser hook here, not in `drivers/`.
- **VTK output is timestepped and ParaView-compatible**: the `.pvd`
  collection MUST list `.vti` files in time order with explicit
  `timestep=` attributes.
- **No silent unit conversion**: the parser preserves the user's
  unit system; all unit conversion to lattice units happens explicitly
  in `drivers/` or in the `.krk` `Define` block.

## Cross-module dependencies

Reads from: nothing internal (this is a leaf module on the input side).
Provides to: `drivers` (parsed config), `postprocess.jl` (VTK writer
called after a run).

## Status / scope notes

- `.krk` DSL features: `Define`, `Boundary`, `Refine`, `Postprocess`,
  parametric kwargs, spatial expressions. See the
  [Concepts](../../docs/src/concepts_index.md) page for the syntax surface.
- A VSCode `.krk` syntax extension exists under `vscode-krk/`.
- STL voxelisation supports the immersed-body family but is independent
  of the LI-BB curved-boundary kernels (which work from `q_wall` cut
  links, not voxel masks).
