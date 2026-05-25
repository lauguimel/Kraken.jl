# M52b — FVFD cylinder-adjacent stencil canary

Date: 2026-05-25
Script: `bench/viscoelastic_validation/patch_tests/PT_cylinder_adjacent_stencil.jl`
CSV: `scratch/M52b_canary/cyl_adj_canary.csv`
Backend: CPU Float64 raw arrays

## TL;DR

**FAIL**: C1 cylinder-adjacent wall-gradient projection has max abs_err
`0.14125809929537925`, mean abs_err `0.07110950006654244`, and a strong
positive abs_err-vs-q_w correlation (`r = 0.7757844079551557`).

## Setup

- Domain: `32x32`, `dx = dy = 1.0`
- Cylinder: `R = 4.0`, `(cx, cy) = (16.5, 16.5)`
- Geometry source: `Kraken.precompute_q_wall_cylinder(Nx, Ny, cx, cy, R; FT=Float64)`
- Source convention: node `(i,j)` is at `(i-1,j-1)`, so the analytic
  field and wall normals use that same convention.
- Field C1: `ux = (r^2 - R^2) / R^2`, `uy = 0`
- Analytic wall-normal derivative:
  `d ux / dn = d((r^2 - R^2)/R^2)/dr = 2r/R^2 = 2/R = 0.5`
- Compared quantity: `du_dn = nx*dudx + ny*dudy` at each cut link with `0.1 <= q_w <= 0.9`.

## Summary stats

| Metric | Value |
|---|---:|
| Cut links tested | 76 |
| q_w range | 0.21611781858498924 to 0.8542486889354093 |
| mean abs_err | 0.07110950006654244 |
| median abs_err | 0.0625 |
| max abs_err | 0.14125809929537925 |
| std abs_err | 0.03400945177705279 |
| corr(q_w, abs_err) | 0.7757844079551557 |

## Worst rows

| cell | q_w | theta_deg | computed du/dn | analytic | abs_err | rel_err |
|---|---:|---:|---:|---:|---:|---:|
| (15,13) | 0.8542486889354093 | -114.29518894536459 | 0.6412580992953792 | 0.5 | 0.14125809929537925 | 0.2825161985907585 |
| (20,13) | 0.8542486889354093 | -65.70481105463544 | 0.6412580992953792 | 0.5 | 0.14125809929537925 | 0.2825161985907585 |
| (13,15) | 0.8542486889354093 | -155.70481105463546 | 0.6412580992953792 | 0.5 | 0.14125809929537925 | 0.2825161985907585 |
| (22,15) | 0.8542486889354093 | -24.295188945364565 | 0.6412580992953792 | 0.5 | 0.14125809929537925 | 0.2825161985907585 |
| (13,20) | 0.8542486889354093 | 155.70481105463546 | 0.6412580992953792 | 0.5 | 0.14125809929537925 | 0.2825161985907585 |

## abs_err vs q_w

| q_w bin | n | mean abs_err | max abs_err |
|---|---:|---:|---:|
| 0.1-0.3 | 8 | 0.0376029090153146 | 0.0376029090153146 |
| 0.3-0.5 | 8 | 0.036835828037412366 | 0.036835828037412366 |
| 0.5-0.7 | 44 | 0.06692353784729693 | 0.1187184335382292 |
| 0.7-0.9 | 16 | 0.11651102770964653 | 0.14125809929537925 |

Shape: errors are lowest for q_w below 0.5 and rise sharply for large
q_w, with the worst links at `q_w = 0.8542486889354093`.

## Interpretation

This confirms the M48 hypothesis at the smallest missing level: the current FVFD velocity-gradient stencil at cylinder-adjacent cells is not q_w-aware and returns a biased wall-gradient projection for a field whose analytic wall derivative is exactly known.

The strong q_w dependence means a q_w-aware embedded/cut-link stencil is a plausible fix target. The result does not prove the coupled cylinder Cd bias is only this stencil, but it does make this stencil a real standalone bug rather than an ambiguous audit item.

## Wall time

Measured script wall time: `0.501240 s`.

## Recommendation to Boss

Next mission should derive a q_w-aware cylinder cut-link velocity-gradient
stencil and rerun this canary before any coupled cylinder benchmark. C2
potential-flow tangential-gradient coverage is a useful extension after
C1 is green.
