# M55 AUDIT codex

## 1. Implementation Reading

Codex did not sample `u2` by bilinear lookup. The embedded gradient kernel first builds a seed gradient from the ordinary solid-aware finite-difference path:
`_fvfd_solid_bc_derivative_x_2d/_y_2d` for each velocity component. The q-wall helper then computes fixed-stencil Hessian entries with `_fvfd_solid_bc_second_derivative_x_2d`, `_y_2d`, and `_fvfd_solid_bc_mixed_derivative_xy_2d`; the mixed derivative returns zero if the four diagonal fluid samples are unavailable.

For each valid D2Q9 cut link, it sets the wall normal from the link direction, takes `u1 = phi[i,j]`, and reconstructs
`u2_hat = u1 + h * (g . n) + h^2/2 * (n' H n)` with `h = min(abs(dx), abs(dy))`. It then applies the no-slip (`u_wall = 0`) M55 numerator using physical `distance = q_w * link_length` and `distance + h`. Multiple valid cut links are converted into equations `n . grad = target`; the code solves a 2D least-squares system, then blends it 30/70 with the best valid two-link pair.

Snippets:

`src/fvfd/qw_wall_gradient_2d.jl:41-43`
```julia
qw = wall_q[i, j, q]
if qw < q_w_min
    return false, zero(T), zero(T), zero(T)
```

`src/fvfd/qw_wall_gradient_2d.jl:63-65`
```julia
u1 = phi[i, j]
d2n = hxx * nx * nx + T(2) * hxy * nx * ny + hyy * ny * ny
u2 = u1 + h * (gx * nx + gy * ny) + (h * h / T(2)) * d2n
```

`src/fvfd/qw_wall_gradient_2d.jl:66-68`
```julia
target_normal = (
    (distance + h) * (distance + h) * u1 - distance * distance * u2
) / (h * distance * (distance + h))
```

`src/fvfd/operators_2d.jl:1122-1125`
```julia
ux_gx, ux_gy = _fvfd_apply_qw_quadratic_embedded_wall_gradient_2d(
    ux_gx, ux_gy, ux, is_solid, wall_q, wall_nx, wall_ny,
    wall_inv_distance_to_center, i, j, Nx, Ny,
```

Fallback: `q_w_min = T(0.01)` is hard-coded at `src/fvfd/qw_wall_gradient_2d.jl:130`. If all q-wall equations are invalid, or the seed gradient is degenerate, the helper calls `_fvfd_apply_embedded_wall_gradient_2d`, the existing first-order wall-normal projection.

The canary field is exactly quadratic:

`bench/viscoelastic_validation/patch_tests/PT_cylinder_adjacent_stencil.jl:18-20`
```julia
x = i - 1.0
y = j - 1.0
ux[i, j] = ((x - cx)^2 + (y - cy)^2 - R2) / R2
```

## 2. Algebraic Equivalence Test

Let `s=0` at the wall, `u(s) = u0 + c1*s + c2*s^2/2 + c3*s^3/6 + ...`, first cell at `s=d`, and the M55 second sample at `s=d+h`.

True M55 sample:

```text
u2 = u(d+h)
   = u(d) + h*u'(d) + h^2*u''(d)/2 + h^3*u'''(d)/6 + O(h^4)
```

Codex reconstruction, even granting exact seed gradient and Hessian:

```text
u2_hat = u(d) + h*u'(d) + h^2*u''(d)/2
u2_hat - u2 = -h^3*u'''(d)/6 + O(h^4)
```

M55 derivative with true `u2`:

```text
D_M55 = [(d+h)^2*u(d) - d^2*u(d+h) - h*(2d+h)*u0]
        / [h*d*(d+h)]
```

Codex derivative:

```text
D_codex = [(d+h)^2*u(d) - d^2*u2_hat - h*(2d+h)*u0]
          / [h*d*(d+h)]

D_codex - D_M55
  = -d*(u2_hat - u2) / [h*(d+h)]
  = d*h^2*u'''(d) / [6*(d+h)] + O(h^3)
```

For `d = q*h`, the leading difference is:

```text
D_codex - D_M55 = [q/(q+1)] * h^2*c3/6 + O(h^3)
```

This is before adding the actual seed-gradient and fixed-Hessian stencil errors; those stencils are also exact on quadratics and not exact on general smooth fields.

Conclusion: **GOALPOST-HIT**. The Taylor reconstruction matches M55 only when the field is locally quadratic, because it replaces the true off-grid `u2` by a second-order Taylor truncation. On a cubic or higher field, the leading implementation-vs-M55 error is `d*h^2*u'''(d)/(6*(d+h)) + O(h^3)` (`[q/(q+1)]*h^2*c3/6 + O(h^3)` when `d=q*h`).

Additional concern: for diagonal links, the code uses `distance = q_w*link_length` but advances `u2` by `h = min(dx,dy)`, not by `link_length`. If M55 intends the second sample one full D2Q9 link farther along a diagonal, this is another algebraic mismatch.

## 3. q_w_min Protection

| q_w | Branch | Behavior |
| --- | --- | --- |
| `0.01` | q-aware M55/Taylor branch | The check is `< q_w_min`, so the floor itself is accepted. Denominator is `O(0.01*h)`, giving about `100x` sensitivity to `u1-u_wall`. |
| `0.005` | invalid q equation | That cut link is skipped. If no other valid q-link remains, the code falls back to first-order embedded wall projection. |
| `0.001` | invalid q equation | Same as `0.005`; skipped unless other q-links drive the least-squares solve. |

The fallback is finite in floating point, but not a hard small-q regularization. `_fvfd_apply_embedded_wall_gradient_2d` uses `phi[i,j] * wall_inv_distance_to_center`; for a very small wall distance this is still a `1/distance` estimate. It is reasonable only when the no-slip field is already smooth and `phi = O(distance)`.

Cylinder `R=30` estimate: one quadrant has about `R` vertical plus `R` horizontal crossings (`~60` axis cut links), plus `O(2*sqrt(2)*R) ~= 85` diagonal-family crossings. With approximately uniform fractional cut positions, `q_w < 0.1` appears in about 10% of links: `O(10)` per quadrant, `O(40-60)` around the cylinder. `q_w_min = 0.01` excludes only about 1%, so most `0.01 <= q_w < 0.1` near-singular links still enter the q-aware formula.

q_w_min adequacy: **INADEQUATE** for R=30 promotion. The agreed derivation recommended `0.05-0.1`; this implementation keeps the singular branch active down to exactly `0.01`, and the fallback below that is finite but not strongly regularized.

## 4. New Field on `FVFDEmbeddedBoundary2D`

All constructors found in `src/fvfd/lowering_2d.jl` set the new field:

| Constructor/call site | wall_q status | Downstream effect |
| --- | --- | --- |
| `FVFDEmbeddedBoundary2D(wall_nx, wall_ny, wall_inv_distance)` at lines 17-32 | zero `(Nx,Ny,9)` | No crash; q-aware equations invalid, first-order fallback. |
| `fvfd_empty_embedded_boundary_2d` at lines 35-54 | zero `(Nx,Ny,9)` | No crash; fallback. |
| `fvfd_embedded_boundary_from_halfplane_2d` at lines 163-223 | zero `(Nx,Ny,9)` | No crash; q-aware path silently disabled for halfplanes. |
| `fvfd_embedded_boundary_from_circle_2d` at lines 308-389 | zero `(Nx,Ny,9)` | No crash; q-aware path silently disabled for analytic circle lowering. |
| `fvfd_embedded_boundary_from_qwall_2d` at lines 408-574 | allocated and filled at line 463 for included q-links | Only constructor that activates M55/Taylor branch. |
| `fvfd_transfer_embedded_boundary_2d` at lines 657-692 | device allocation and copy at lines 673 and 686 | Device geometry preserves `wall_q`. |

Downstream consumers are structurally prepared: `fvfd_velocity_gradient_embedded_2d!` passes `embedded.wall_q` into the kernel (`src/fvfd/operators_2d.jl:1175`), and the q-wall helper is included by `src/fvfd/FVFD.jl:6`. Other embedded consumers use the existing geometry fields and are unaffected.

Behavioral caveat: any A.3 driver using `fvfd_geometry_from_circle_2d` rather than `fvfd_geometry_from_lbm_2d`/`fvfd_embedded_boundary_from_qwall_2d` will not exercise the new M55/Taylor branch.

## 5. Verdict

**RED - block A.3.**

Defects:

| Defect | Why it blocks | Smallest patch |
| --- | --- | --- |
| Taylor `u2` is not algebraically equivalent to M55 | It is exact on the quadratic canary by construction, but differs from true M55 by `d*h^2*u'''(d)/(6*(d+h)) + O(h^3)` on cubic fields. | Replace Taylor `u2` with the actual off-grid M55 sample, e.g. bilinear interpolation at `wall + (distance+h)*n` with a documented choice of `h_n`. |
| `q_w_min = 0.01` is too low | R=30 should have many `q_w < 0.1` links; links at `0.01 <= q_w < 0.1` still use the singular branch. | Raise default floor to `0.05-0.1` and route below it to a bounded first-order/halfway closure. |
| Diagonal spacing is ambiguous | Code uses `distance=q_w*link_length` but `u2` offset `h=min(dx,dy)`, which may not be the intended D2Q9 second-link spacing. | Define `h_n` explicitly per link and use the same spacing in `distance`, `u2`, and denominator. |

Empirical signature if run anyway: the R-sweep can stay green on quadratic or near-quadratic manufactured fields, but show angle/q-correlated errors on real cylinder flow, especially for `q_w in [0.01,0.1]` and diagonal cut links. A driver using analytic circle lowering may show no M55 improvement at all because `wall_q` is zero-filled there.
