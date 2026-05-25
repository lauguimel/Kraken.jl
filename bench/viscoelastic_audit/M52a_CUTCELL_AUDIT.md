# M52a — halfwayBB cut-cell + cylinder-adjacent stencil audit
Date: 2026-05-26

## TL;DR
`wall_bc=:halfwayBB` does **not** force `q_w=0.5` in the current cylinder Guo-field path: it dispatches `ApplyLiBBPrePhase`, which reads precomputed `q_wall`. The FVFD velocity gradient near the cylinder is **not** q_w-aware when `embedded_gradient=false`; it uses the same solid-mask cell-center derivative class as the M49 bug.

## Question 1: halfwayBB BC at cut-cells
- Dispatch chain: analytic cylinder `q_wall` is built at `src/kernels/li_bb_2d.jl:266-309`; the driver passes it to `fused_trt_libb_v2_guo_field_step!` at `src/drivers/viscoelastic_logfv_2d.jl:482-485`; `Val(:halfwayBB)` selects `_TRT_LIBB_V2_GUO_FIELD_SPEC` at `src/kernels/li_bb_2d_v2.jl:172-185`.
- The selected spec is not `ApplyHalfwayBBPrePhase`; it is `ApplyLiBBPrePhase` (`src/kernels/li_bb_2d_v2.jl:49-53`):
```julia
const _TRT_LIBB_V2_GUO_FIELD_SPEC = LBMSpec(
    PullHalfwayBB(), SolidInert(),
    ApplyLiBBPrePhase(),
    Moments(), CollideTRTDirectGuoField(),
    WriteMoments(),
)
```
- Quoted kernel snippet (`src/kernels/dsl/bricks.jl:357-368`):
```julia
emit_code(::ApplyLiBBPrePhase) = quote
    # Pair (2, 4): link q=2 flagged → corrupted pop is fp4 (=q̄ of q=2).
    qw2 = q_wall[i, j, 2]
    if qw2 > zero(T)
        δ4 = -T(2/3) * uw_link_x[i, j, 2]
        fp4 = _libb_branch(qw2, f_in[i, j, 2], fp2, f_in[i, j, 4], δ4)
    end
    qw4 = q_wall[i, j, 4]
```
- Verdict: **q_w honored, but under a misleading `:halfwayBB` name**. The true fixed-halfway brick exists (`src/kernels/dsl/bricks.jl:285-325`) and only tests `q_wall > 0`, but this path does not dispatch it. `_libb_branch` branches on true `q_w` at `src/kernels/li_bb_2d.jl:70-82`.

## Question 2: FVFD velocity-gradient at cylinder-adjacent cells
- Code path with `embedded_gradient=false`: default kwarg at `src/drivers/viscoelastic_logfv_2d.jl:193`; branch calls `fvfd_velocity_gradient_2d!` at `src/drivers/viscoelastic_logfv_2d.jl:419-428`. The M51 correction after it targets only `WallGradientSides(uy_south, uy_north, nothing, nothing)` (`src/drivers/viscoelastic_logfv_2d.jl:385`, `430-433`), i.e. outer y-walls, not cylinder cut-links.
- Quoted snippet (`src/fvfd/operators_2d.jl:1077-1088`):
```julia
dudx[i, j] = _fvfd_solid_bc_derivative_x_2d(
    ux, is_solid, i, j, Nx, inv_dx, inv_2dx, west_bc, east_bc,
)
dudy[i, j] = _fvfd_solid_bc_derivative_y_2d(
    ux, is_solid, i, j, Ny, inv_dy, inv_2dy, south_bc, north_bc,
)
dvdx[i, j] = _fvfd_solid_bc_derivative_x_2d(
    uy, is_solid, i, j, Nx, inv_dx, inv_2dx, west_bc, east_bc,
)
dvdy[i, j] = _fvfd_solid_bc_derivative_y_2d(
```
- Verdict: **cell-center derivative (M49 bug class)**. The derivative helpers take `field, is_solid, i, j, ...` only (`src/fvfd/operators_2d.jl:13-16`, `45-48`); no `q_wall`, wall normal, or wall distance is available.

## Question 3: embedded_gradient=true path
- Driver builds `embedded_h` from `fvfd_embedded_boundary_from_qwall_2d` when `embedded_geometry=:qwall` (`src/drivers/viscoelastic_logfv_2d.jl:247-258`). That lowering reads `q_wall[i,j,q]`, accumulates wall normals, and stores `wall_inv_distance` (`src/fvfd/lowering_2d.jl:421-455`, `488-497`).
- The embedded gradient seeds the same solid-mask derivatives, then applies a normal correction using wall normal and inverse wall distance (`src/fvfd/operators_2d.jl:1122-1127`):
```julia
ux_gx, ux_gy = _fvfd_apply_embedded_wall_gradient_2d(
    ux_gx, ux_gy, ux, wall_nx, wall_ny, wall_inv_distance, i, j,
)
uy_gx, uy_gy = _fvfd_apply_embedded_wall_gradient_2d(
    uy_gx, uy_gy, uy, wall_nx, wall_ny, wall_inv_distance, i, j,
)
```
- Verdict: **q_w-aware normal correction**, but tangential components remain seeded by the solid-mask derivative. It is not default because `embedded_gradient::Bool=false` (`src/drivers/viscoelastic_logfv_2d.jl:193`); M50 classified this embedded path as ambiguous, not the M48 default path (`bench/viscoelastic_audit/M50_STENCIL_CALLER_AUDIT.md:24-27`, `44`).

## Question 4: Bouzidi-FL comparison
- `:bouzidi_fl_twopass` uses raw pass-1, pass-2 `ApplyBouzidiFLPostCollideTwoPass`, then cut-link rho recompute (`src/kernels/li_bb_2d_v2.jl:211-249`). Pass-2 reads `q_wall` directly (`src/kernels/dsl/bricks.jl:580-587`):
```julia
qw2 = q_wall[i, j, 2]
if qw2 > zero(T)
    delta4 = -(T(2) / T(3)) * rho_w * uw_link_x[i, j, 2]
    has_ff2 = false
    f2_ff = f2_here
    if qw2 <= half
        i2_ff = i - 1
        j2_ff = j
```
- Switching to `:bouzidi_fl_twopass` would bypass a hypothetical fixed-`q_w=0.5` halfway cut-cell bug, but that bug is **not present** in the current `:halfwayBB` Guo-field path. It would **not** bypass the Q2 FVFD velocity-gradient bug unless paired with `embedded_gradient=true` or a qwall-aware cylinder-adjacent gradient fix.

## Question 5: Existing q_w-aware helpers
- Yes. `fvfd_embedded_boundary_from_qwall_2d` converts cut-link `q_wall` into `wall_nx`, `wall_ny`, `wall_inv_distance`, and cell/face fractions (`src/fvfd/lowering_2d.jl:396-555`).
- Yes. `_fvfd_apply_embedded_wall_gradient_2d` applies a variable-distance wall-normal derivative target (`phi[i,j] * wall_inv_distance[i,j]`) at `src/fvfd/operators_2d.jl:127-139`.
- No direct analogue to `apply_halfway_wall_gradient_correction!` exists for all cylinder-adjacent gradient components as a standalone correction pass; the current qwall-aware precedent is the embedded-gradient kernel path.

## Implication for fix scope
- Smallest change to flatten the M48 U-shape is not to generalize the LBM `:halfwayBB` cut-link BC; it already reads `q_wall`. Target the cylinder-adjacent FVFD velocity gradient: either promote/validate `embedded_gradient=true` for the cylinder benchmark, or add a focused qwall-aware correction pass for cut-link cells analogous to M51. Files affected: `src/fvfd/operators_2d.jl`, maybe `src/fvfd/lowering_2d.jl`, `src/drivers/viscoelastic_logfv_2d.jl`, plus a cylinder-adjacent canary. Estimated LOC: 80-150 including tests; 20-40 for a benchmark-only toggle.

## Anti-pattern flags
- `wall_bc=:halfwayBB` currently dispatches a q_w-aware LI-BB pre-phase in this driver path; the name hides semantics.
- `fvfd_velocity_gradient_2d!` does not encode whether near-solid values are cell-center gradients or wall gradients.
- `embedded_gradient=false` coexists with `embedded_geometry=:qwall`, so a run can carry exact qwall geometry while the production gradient ignores it.

## Recommendation to Boss
- Next mission: build the M52b-style cylinder-adjacent velocity-gradient canary for an analytic field, comparing `embedded_gradient=false` vs `true` at explicit qwall cut-link cells. If embedded is GREEN and default is RED, promote the qwall-aware gradient path or implement the minimal cut-link correction before rerunning M48.
