# curvilinear/

SLBM (Semi-Lagrangian Lattice Boltzmann Method) on body-fitted curvilinear
and Lagrange grids. Decouples the lattice geometry from the Cartesian
embedding so that walls can be resolved exactly instead of approximated by
bounce-back on stair-step grids.

## Key entry points

| File | Symbol | Purpose |
|---|---|---|
| `mesh.jl` | `CurvilinearMesh2D`, `CurvilinearMesh3D` | Mesh data structure (vertex positions, Jacobian) |
| `mesh_3d.jl` | 3D variant | |
| `generators.jl` | `make_annulus_mesh`, `make_cylinder_mesh`, … | Built-in mesh generators for common geometries |
| `mesh_from_arrays.jl` | `mesh_from_arrays` | Build a mesh from user-provided vertex arrays |
| `mesh_gmsh.jl` | Gmsh loader | Import curvilinear grids from Gmsh (BSpline + ForwardDiff loader) |
| `slbm.jl` | `PullSLBM` (2D), `PullSLBMBiquad`, departures | Semi-Lagrangian streaming on the curvilinear mesh |
| `slbm_3d.jl` | 3D analogue | Apple Metal validated at 112 MLUPS |

## Critical invariants

- **Local-CFL discipline**: departures use the local cell size for shear-aligned
  flows → ~2.5% error vs naive global CFL.
- **Wall velocity must be tangent** to the oblique wall (any non-tangent
  component causes O(Ma²) residual that looks like a numerical bug but is
  wrong test physics — see project memory `project_slbm_session5_movingwall`).
- **PullSLBMBiquad** fixes 2.86× anisotropic viscosity on closed/periodic
  benchmarks (`ng=3` + hybrid_z); **bilinear default** on open-BC layouts
  (square / step) until the BC layer is audited (see
  `project_slbm_biquad_status`).
- **ng (ghost cells)**: bilinear stencil requires ng=2 (fallback at α>26.57°);
  biquad stencil requires ng=3 for machine precision across all angles.

## Cross-module dependencies

Reads from: `lattice`, `kernels` (BCSpec, equilibrium helpers).
Provides to: `drivers` (via SLBM-flavoured `run_*` paths).
Optional: `multiblock` (when curvilinear blocks are coupled — block
exchange handles shared-node connectivity).

## Status / scope notes

- **SLBM 2D is frozen for the paper** (project memory
  `project_curvilinear_v02`); v0.3+ may explore stream-tube LBM
  variants (see `project_stream_tube_lbm`).
- Enzyme AD on geometric parameters validated at 0.0% error
  (`project_kernel_dsl` session 2026-04-17).
