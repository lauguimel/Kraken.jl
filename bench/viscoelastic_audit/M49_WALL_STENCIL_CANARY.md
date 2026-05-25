# M49 — FVFD halfway wall stencil canary

Date: 2026-05-25
Script: `bench/viscoelastic_validation/patch_tests/PT_halfway_wall_stencil.jl`
CSV: `scratch/M49_canary/wall_stencil_canary.csv`
Backend: CPU Float64 raw arrays

## TL;DR

**FAIL**: the quadratic wall derivative stencil is exact for P1 but fails
P2/P3 at both wall rows and in both x/y directions against the halfwayBB
wall derivative.

## Per-case max abs error

Max is over direction x/y and lower/upper wall rows.

| Case | Mode | Max abs error | Verdict |
|---|---|---:|---|
| P1 `u=s` | quadratic | 0 | PASS |
| P2 `u=s^2` | quadratic | 1 | FAIL |
| P3 `u=s+2s^2` | quadratic | 2 | FAIL |
| P4 `u=sin(0.3s)` | quadratic | 0.028840994213476245 | INFO |
| P1 `u=s` | linear | 0 | INFO |
| P2 `u=s^2` | linear | 2 | INFO |
| P3 `u=s+2s^2` | linear | 4 | INFO |
| P4 `u=sin(0.3s)` | linear | 0.07033159654642548 | INFO |

## Selected rows

| Case | Dir | Wall | Mode | Computed | Analytic | Abs err |
|---|---|---|---|---:|---:|---:|
| P1 | y | south | quadratic | 1 | 1 | 0 |
| P2 | y | south | quadratic | 1 | 0 | 1 |
| P3 | y | south | quadratic | 3 | 1 | 2 |
| P2 | y | north | quadratic | 15 | 16 | 1 |
| P3 | y | north | quadratic | 31 | 33 | 2 |
| P2 | x | west | quadratic | 1 | 0 | 1 |
| P3 | x | west | quadratic | 3 | 1 | 2 |
| P4 | y | south | quadratic | 0.30495448950039444 | 0.3 | 0.0049544895003944545 |

## Interpretation

The failure is systematic and symmetric between `_x_2d` and `_y_2d`.
For P2 at the lower wall, the quadratic stencil returns `1`, which is
`d(s^2)/ds` at the first-fluid center `s=0.5`, not the halfwayBB wall
at `s=0`. For P3 it returns `3`, which is likewise the derivative at
`s=0.5`, not the wall derivative `1`.

This does not show a simple factor-2 error for all polynomials; it shows
a geometric offset. The current one-sided formula is a cell-center
derivative at the boundary-adjacent fluid cell. The M48 hypothesis
therefore narrows from "divisor assumes wall@dx" to "wall derivative
request is being answered by a first-fluid-center derivative while
halfwayBB places the wall at dx/2 from that center."

P4 is non-polynomial and informational by design.

## Recommendation to Boss

Next mission: fix attempt should derive and install a halfway-wall
one-sided derivative formula for samples at `s=0.5, 1.5, 2.5` with
wall value `u(0)=0`, then rerun this canary before any Metal mesh
convergence run. The candidate site is `src/fvfd/operators_2d.jl:25-31`
and the mirrored branches at `:33-40`, `:57-63`, and `:65-72`.

Do not just swap `dx` to `dx/2`: P2/P3 show that curvature terms also
matter.

## Wall time

Measured canary loop wall time: 0.204330 s.
