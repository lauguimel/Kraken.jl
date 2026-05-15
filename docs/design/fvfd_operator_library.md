# FV/FD Operator Library

## Scope

Kraken's FV/FD library is an internal operator layer for structured Cartesian
patches. Its first production user is the cell-centered log-conformation
polymer CDE backend coupled to the LBM solvent.

This is not a general CFD solver and not a standalone package yet. Keep it
inside Kraken until the API is stable, the LBM adapter is thin, and the 2D
validation ladder is green.

## Contract

The same explicit specs must feed every affected operator:

- domain BCs: periodic, wall, inlet, outlet, open, symmetry;
- field BCs: velocity, conformation/log-conformation, stress, scalar values;
- geometry: solid mask, `q_wall`, cut cells, wall normals, wall distances;
- models: Oldroyd-B, FENE-P, and future constitutive laws through lowered codes;
- operators: advection, velocity gradients, tensor divergence, stabilization,
  traction/drag, diagnostics, restart, and field dumps.

High-level specs may be expressive, but kernels consume only lowered concrete
data: scalar codes, masks, SoA arrays, compact slots, and precomputed
coefficients.

## Initial Layout

```text
src/fvfd/
  FVFD.jl
  specs.jl
  lowering_2d.jl
  operators_2d.jl
```

The initial vertical slice exposes BC specs, field BC specs,
embedded-boundary lowering from `q_wall`, and the 2D face-velocity,
upwind-advection, velocity-gradient, tensor-divergence, and BSD force
operators used by log-FV. Existing `logfv_*` names remain as compatibility
wrappers during migration.

Field BCs are lowered with `fvfd_transfer_field_bc_2d`. This helper validates
active open boundaries, converts host boundary values to the chosen floating
type, fills inactive sides with a concrete default when needed, and returns
backend-resident vectors. GPU kernels should consume these lowered vectors,
not ad hoc host arrays built in drivers.

Embedded advection uses preallocated face arrays through
`fvfd_advect_upwind_embedded_2d!` and
`fvfd_sym2_advect_upwind_embedded_2d!`: the wrapper lowers cell-centered
velocity to q_wall-aperture face fluxes, then calls the regular upwind
operator with the same `FVFDGeometry2D`. This keeps geometry ownership in the
operator layer while avoiding hidden allocations in hot loops.

Embedded tensor divergence is available as
`fvfd_tensor_divergence_embedded_2d!`. It uses face apertures, cut-cell volume
fractions, and the wall-normal-length closure implied by those apertures so a
constant stress field has zero divergence in a cut cell.

Embedded wall traction is available as `fvfd_embedded_wall_traction_2d!`. It
integrates `tau * n` over the lowered wall segment in each cut cell, using the
stored normal convention from the embedded geometry.

Analytical embedded half-planes can be lowered with
`fvfd_geometry_from_halfplane_2d`. This is a validation helper for coherent
multi-cell cut-wall canaries before relying on case-specific `q_wall`
generation from an LBM geometry.

Analytical embedded circles can be lowered with `fvfd_geometry_from_circle_2d`.
The helper computes coherent face apertures for the fluid outside the circle
and stores wall normals pointing from the solid into the fluid, matching the
half-plane convention used by embedded no-slip gradient correction. This gives
a curved constant-stress and traction canary below the cylinder benchmark.

The coupled log-FV cylinder driver can opt into this geometry through
`embedded_geometry=:circle`. `embedded_advection=true` uses aperture-weighted
face velocities for polymer advection, `embedded_force=true` uses the embedded
tensor divergence for polymer forcing, and `embedded_drag=true` computes
polymer/BSD drag diagnostics from embedded wall traction. Circle runs also
report radial normal-alignment diagnostics so the coupled path can catch a
wrong wall-normal convention before interpreting drag. The defaults remain the
legacy q_wall lowering and Cartesian/staircase paths until the simple-flow
ladder is fully green.

Lowered embedded-boundary geometry currently carries:

- wall normal components: `wall_nx`, `wall_ny`;
- wall distance and inverse distance: `wall_distance`, `wall_inv_distance`;
- a clipped half-plane cell volume proxy: `cell_fraction`;
- a wall segment length proxy in cell units: `wall_fraction`;
- per-cell face aperture fractions: `west_fraction`, `east_fraction`,
  `south_fraction`, `north_fraction`;
- the number of included cut links: `cut_count`.

These fields are transferred together so CPU, Metal, and CUDA code see the
same concrete geometry payload.

## Validation Order

Use the lower-level FV/FD tests before coupled benchmarks:

1. constants and affine fields;
2. domain BC semantics;
3. `q_wall` lowering: normals, distances, half-way filtering;
4. velocity gradients with and without embedded walls;
5. advection;
6. tensor divergence and BSD/laplacian;
7. frozen-velocity log-conformation CDE;
8. coupled square/BFS;
9. cylinder/RheoTool.

If a macro benchmark fails, add the smallest missing FV/FD canary before
changing production kernels.

Current 2D canaries include field BC spec compatibility, open-boundary field
BC length validation, explicit field BC host-to-backend lowering, constant
fields for all domain BC codes, affine exactness, periodic sine second-order
convergence, analytical Couette and Poiseuille gradients, quadratic one-sided
gradients around an internal solid cell, embedded `q_wall` wall-normal
gradients, cylinder near-tangent `q_wall` lowering, embedded face-aperture
velocity lowering from `q_wall`, embedded scalar and symmetric-tensor
constant advection through q_wall apertures, face-velocity lowering with open,
wall, periodic, and solid-mask semantics, affine scalar upwind advection with
field BCs for both west/south and east/north inflow, explicit
non-unit-spacing log-FV advection wrapper equivalence, periodic scalar and
symmetric-tensor upwind wrapping, tensor-divergence exactness, embedded
constant-stress divergence balance on q_wall cuts, coherent half-plane
constant-stress divergence balance on CPU and local Metal when available,
analytical half-plane wall traction on CPU and local Metal when available,
coherent circle constant-stress divergence and zero total traction,
BSD/laplacian exactness, and log-FV wrapper equivalence.
