# M51 — Wall Gradient Fix Verdict

Date: 2026-05-25

## TL;DR

Status: PASS.

Helper file: `src/fvfd/halfway_wall_gradient_correction_2d.jl` (117 LOC).

M49 extended canary: PASS. Helper-quadratic P1/P2/P3 maximum
`abs_error = 4.9737991503207013e-14 < 1e-10`.

Existing FVFD operator test: PASS (`952/952`).

Mission source delta, excluding pre-existing dirty edits in this
worktree: about 154 insertions and 43 deletions under `src/`.

Drivers targeted:
- cavity: refactor existing first-order half-cell correction through `apply_halfway_wall_gradient_correction!(...; order=:linear)`.
- cylinder_coupled: correct outer south/north stationary walls only; x-open/inlet sides unchanged.
- frozen_channel: correct outer south/north stationary walls only.
- poiseuille_coupled: correct outer south/north stationary walls only.
- square_periodic: correct outer south/north stationary walls only; embedded square surface deferred.
- bfs_passive: correct outer south/north stationary walls only; BFS step surface deferred.

## Quadratic Wall Formula Derivation

Let the wall-normal coordinate be `s = y / dy`, with the halfway wall at
`s = 0`, the first fluid cell at `s = 1/2`, and the second fluid cell at
`s = 3/2`. Fit

```text
u(s) = u_wall + A s + B s^2
```

through `u(0) = u_wall`, `u(1/2) = u1`, and `u(3/2) = u2`.
Define `r1 = u1 - u_wall` and `r2 = u2 - u_wall`. The two equations are

```text
(1/2) A + (1/4) B = r1
(3/2) A + (9/4) B = r2
```

Multiply by 4:

```text
2 A + B = 4 r1
6 A + 9 B = 4 r2
```

From the first equation, `B = 4 r1 - 2 A`. Substitute:

```text
6 A + 9(4 r1 - 2 A) = 4 r2
-12 A + 36 r1 = 4 r2
A = 3 r1 - r2 / 3
```

The physical derivative is `(du/ds) / dy`, so

```text
du/dy|wall = (3(u1 - u_wall) - (u2 - u_wall)/3) / dy
           = (3 u1 - u2/3 - (8/3) u_wall) / dy
```

For the upper wall, use outward mirrored coordinate `s = (L - y) / dy`.
The derivative with respect to physical `y` changes sign:

```text
du/dy|north = ((8/3) u_wall - 3 u1 + u2/3) / dy
```

The same sign convention applies to west/east for `d/dx`.

## Per-Driver Edit Summary

- `run_viscoelastic_logfv_cylinder_coupled_2d`: after the selected velocity-gradient path, apply `WallGradientSides(uy_south, uy_north, nothing, nothing)` in `:quadratic` mode; embedded cylinder wall remains out of scope.
- `run_viscoelastic_logfv_frozen_channel_cde_2d`: after the initial frozen velocity-gradient call, apply zero south/north wall profiles in `:quadratic` mode.
- `run_viscoelastic_logfv_poiseuille_coupled_2d`: inside the step loop after velocity-gradient assembly, apply zero south/north wall profiles in `:quadratic` mode.
- `run_viscoelastic_logfv_square_periodic_2d`: after velocity-gradient assembly, apply outer south/north wall profiles in `:quadratic` mode; square obstacle surface remains out of scope.
- `run_viscoelastic_logfv_bfs_passive_2d`: after velocity-gradient assembly, apply outer south/north wall profiles in `:quadratic` mode; BFS step surface remains out of scope.
- `run_viscoelastic_logfv_cavity_coupled_2d`: replaced the cavity-specific call site with `apply_halfway_wall_gradient_correction!(...; order=:linear, skip_top_corners=skip_top_corners)`.

## M49 Canary Results

Command:

```bash
julia --project=. bench/viscoelastic_validation/patch_tests/PT_halfway_wall_stencil.jl
```

Result: PASS, wall time `0.583642 s`.

Helper-on quadratic rows:

| Case | Direction | Wall | Computed | Analytic | Abs error |
|---|---|---|---:|---:|---:|
| P1 | y | south | 1 | 1 | 0 |
| P1 | y | north | 0.99999999999999845 | 1 | 1.5543122344752192e-15 |
| P1 | x | west | 1 | 1 | 0 |
| P1 | x | east | 0.99999999999999845 | 1 | 1.5543122344752192e-15 |
| P2 | y | south | 0 | 0 | 0 |
| P2 | y | north | 15.999999999999995 | 16 | 5.3290705182007514e-15 |
| P2 | x | west | 0 | 0 | 0 |
| P2 | x | east | 15.999999999999995 | 16 | 5.3290705182007514e-15 |
| P3 | y | south | 1 | 1 | 0 |
| P3 | y | north | 32.99999999999995 | 33 | 4.9737991503207013e-14 |
| P3 | x | west | 1 | 1 | 0 |
| P3 | x | east | 32.99999999999995 | 33 | 4.9737991503207013e-14 |

## Regression Check

Existing raw M49 rows were kept as regression rows under
`:quadratic_raw` and `:linear_raw`. The representative raw values match
the M49 audit:

| Case | Mode | Max abs error |
|---|---|---:|
| P1 | `:quadratic_raw` | 0 |
| P2 | `:quadratic_raw` | 1 |
| P1 | `:linear_raw` | 0 |

The raw P2/P3 failures remain informational in this extended canary; the
new hard assertion is only on `:halfway_quadratic` for P1/P2/P3.

## Existing Tests Run

- `julia --project=. bench/viscoelastic_validation/patch_tests/PT_halfway_wall_stencil.jl`: PASS.
- `julia --project=. test/test_fvfd_operators_2d.jl`: PASS (`952/952`, 16.9 s).
- Cavity regression: PASS for a focused FP64 bitwise comparison between the old cavity wall-correction formulas and `apply_halfway_wall_gradient_correction!(...; order=:linear)`, with `skip_top_corners=false` and `true`. No dedicated full log-FV cavity byte-identical test was present in `test/`.

## Estimated M48 Impact

Qualitative expectation: replacing first-fluid-center wall-row gradients
with halfway-wall derivatives should reduce the wall-row source bias in
the log-conformation update. For M48, expect the U-shape to flatten; at
R=50 the cylinder Cd should rise from about 114 toward the 117-120 band.
M48 R-sweep is intentionally deferred.
