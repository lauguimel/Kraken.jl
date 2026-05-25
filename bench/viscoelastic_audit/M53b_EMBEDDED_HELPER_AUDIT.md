# M53b - embedded wall-gradient helper audit
Date: 2026-05-26

## TL;DR
Bug is the distance convention at `src/fvfd/lowering_2d.jl:493-497`: `wall_inv_distance` is `1/(wall-to-fluid-centroid distance)`, but `_fvfd_apply_embedded_wall_gradient_2d` applies it to `phi[i,j]`, the lattice cell-center value. Fix by storing the wall-to-cell-center distance for this gradient path, i.e. `wall_distance = distance` before `wall_inv_distance = inv(...)`.

## Q1: wall_inv_distance convention
- Formula at `src/fvfd/lowering_2d.jl:481-497`:
```julia
distance = -(q_w * cqx * nx + q_w * cqy * ny)
if distance > zero(FT)
    distance_sum += distance
    distance_count += 1
end
distance = distance_sum / FT(distance_count)
moments = _fvfd_halfplane_square_fluid_moments_2d(nx, ny, distance, FT)
centroid_distance = distance +
                    nx * moments.centroid_x +
                    ny * moments.centroid_y
wall_distance[i, j] = max(centroid_distance, eps(FT))
wall_inv_distance[i, j] = inv(wall_distance[i, j])
```
- Conclusion: **other**. It first computes the wall-to-cell-center plane distance `distance ~= q_w*|c_q|`, then stores inverse wall-to-fluid-centroid distance. For a single axis link with `dx=1`, `q_w=0.21` gives fluid centroid offset `(-0.5+0.21)/2=-0.145`, so stored `wall_distance=0.21+0.145=0.355`, `wall_inv_distance=2.817`, not expected `1/0.21=4.762`. For `q_w=0.85`, the clipped cell is full, centroid offset `0`, so stored `wall_distance=0.85`, `wall_inv_distance=1.176`, matching `1/q_w`.

## Q2: wall_nx, wall_ny orientation
- Formula at `src/fvfd/lowering_2d.jl:441-459`:
```julia
cqx = FT(cxs[q])
cqy = FT(cys[q])
link_length = hypot(cqx, cqy)
link_length > zero(FT) || continue
nx_q = -cqx / link_length
ny_q = -cqy / link_length
nx_sum += nx_q
ny_sum += ny_q
```
- `src/lattice/d2q9.jl:25-26` defines `cxs,cys`; the normal is `-c/|c|`. If `q_wall` link vector `c` points from fluid cell center toward the wall/solid, `wall_n` points from wall/solid back into fluid. Thus `dot(wall_n, cell_to_wall) = dot(-c/|c|, q_w*c) = -q_w*|c| < 0`: **outward-from-solid / inward-to-fluid domain**, not cell-to-wall.

## Q3: helper formula (the actual arithmetic)
- `_fvfd_apply_embedded_wall_gradient_2d`, `src/fvfd/operators_2d.jl:127-139`:
```julia
@inline function _fvfd_apply_embedded_wall_gradient_2d(
    gx, gy, phi, wall_nx, wall_ny, wall_inv_distance, i, j,
)
    inv_distance = wall_inv_distance[i, j]
    if inv_distance > zero(inv_distance)
        nx = wall_nx[i, j]
        ny = wall_ny[i, j]
        target_normal_derivative = phi[i, j] * inv_distance
        current_normal_derivative = gx * nx + gy * ny
        correction = target_normal_derivative - current_normal_derivative
        return gx + correction * nx, gy + correction * ny
    end
    return gx, gy
end
```
- Walk-through: output is `g_new = g + (phi_cell*inv_distance - dot(g,n))*n`, so `dot(g_new,n)=phi_cell*inv_distance` and tangential gradient is preserved. This is correct only if `inv_distance` is inverse wall-to-`phi_cell` distance.

## Q4: seed-then-correct order
- Kernel at `src/fvfd/operators_2d.jl:1110-1131`:
```julia
ux_gx = _fvfd_solid_bc_derivative_x_2d(
    ux, is_solid, i, j, Nx, inv_dx, inv_2dx, west_bc, east_bc,
)
ux_gy = _fvfd_solid_bc_derivative_y_2d(
    ux, is_solid, i, j, Ny, inv_dy, inv_2dy, south_bc, north_bc,
)
ux_gx, ux_gy = _fvfd_apply_embedded_wall_gradient_2d(
    ux_gx, ux_gy, ux, wall_nx, wall_ny, wall_inv_distance, i, j,
)
uy_gx, uy_gy = _fvfd_apply_embedded_wall_gradient_2d(
    uy_gx, uy_gy, uy, wall_nx, wall_ny, wall_inv_distance, i, j,
)
```
- Order: **seed-with-default-then-overwrite normal component**. It does not add a second normal derivative; it replaces `dot(g,n)` through projection.

## Q5: the bug
- M53a observed small-`q_w` embedded error worse than default: `bench/viscoelastic_audit/M53a_CANARI_EMBEDDED.md:25-32` reports default `0.0376/0.0368` vs embedded `0.1013/0.1035` for `q_w=0.1-0.5`, while high bins improve.
- At `q_w=0.21`, `dx=1`, `R=4`: `r_cell=4.21`, `phi_cell=(4.21^2-16)/16=0.108`. Expected wall normal derivative is `(phi_cell-0)/(0.21)=0.514`, close to analytic `2/R=0.5`.
- Actual helper target with stored centroid distance is `0.108/0.355=0.304`, i.e. under by `0.196` before geometry averaging. This matches the low-`q_w` failure direction: the path under-corrects, giving an observed value nearer `0.4` than `0.6`.
- At `q_w=0.85`, `phi_cell=(4.85^2-16)/16=0.470`; stored distance is `0.85`, so target is `0.553` vs expected `0.553`. High `q_w` is therefore much less broken. Discrepancy points to `src/fvfd/lowering_2d.jl:493-497`, not the helper projection arithmetic.

## Proposed fix
- Change the three identical centroid-distance storage blocks at `src/fvfd/lowering_2d.jl:493-497`, `514-518`, and `536-540` so the gradient distance is the wall-to-cell-center plane distance:
```julia
wall_distance[i, j] = max(distance, eps(FT))       # or best_distance in fallback blocks
wall_inv_distance[i, j] = inv(wall_distance[i, j])
```
- Predicted canary outcome: low-`q_w` embedded bins should flip from worse-than-default to near first-order wall FD error, restoring positive `q_w` error scaling; high-`q_w` bins should remain close to current behavior.

## Anti-pattern flags
- `wall_distance` mixes two meanings: geometric cut-cell centroid distance for fractions/diagnostics and wall-to-cell-center distance needed by the gradient helper.
- Existing tests seed `ux[3,3] = embedded_h.wall_distance[3,3]`, so they validate internal self-consistency rather than the physical `phi[cell]/(q_w*dx)` convention.
- `include_axis_aligned=false` means pure axis-aligned cuts can be silently ignored unless another subcell cut is present.
