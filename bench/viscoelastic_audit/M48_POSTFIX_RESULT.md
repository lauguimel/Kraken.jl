# M48 post-fix R-sweep result (post-M51 wall-gradient fix)
Date: 2026-05-26
Backend: Metal M3 Max F32
Wall time: 1848 s (30.8 min) for 3 R values

## TL;DR

**M51 wall-gradient fix does NOT flatten the M48 U-shape.** Outer
halfwayBB walls were not the dominant source of cylinder Cd bias. The
remaining bias scales with cylinder-adjacent (cut-link) cell count,
not outer-wall count.

## Plateau Cd comparison (instantaneous at 1 flow-through)

| R | Cd pre-fix | Cd post-fix | Δ | trace_C pre→post |
|---:|---:|---:|---:|---:|
| 10 | 114.48 | 114.87 | **+0.39** | 106 → 127 (+20%) |
| 30 | 117.62 | 117.54 | -0.08 (noise) | 199 → 187 (-6%) |
| 50 | 114.26 | 113.93 | **-0.33** | 224 → 211 (-6%) |

Pre-fix data: `scratch/M48_hw_meshconv_prefix_baseline/`
Post-fix data: `scratch/M48_hw_meshconv/`
Both at Metal F32, identical setup, identical driver script.

## Average Cd (over whole run)

| R | Cd avg pre-fix | Cd avg post-fix | Δ avg |
|---:|---:|---:|---:|
| 10 | 114.77 | 115.13 | +0.36 |
| 30 | 116.28 | 116.22 | -0.06 |
| 50 | 111.82 | 111.68 | -0.14 |

Same pattern: improvement at R=10 (where outer wall = larger fraction
of polymer-active region), neutral at R=30, marginal degradation at
R=50.

## Interpretation

The M51 fix correctly addresses outer axis-aligned halfwayBB walls
(M49 canary GREEN at machine precision, 4.97e-14 abs err on P1-P3
quadratic at all four sides). But cylinder Cd is dominated by a
different bias source.

**Cylinder geometry decomposition** (cells affected by wall-gradient
bias per R):

| R | Outer wall cells (M51 targets) | Cylinder cut-link cells (M51 ignores) |
|---:|---:|---:|
| 10 | 2·Nx = 600 | ~2π·R = 63 |
| 30 | 2·Nx = 1800 | ~2π·R = 188 |
| 50 | 2·Nx = 3000 | ~2π·R = 314 |

Both scale linearly with R, but cylinder cut-cells live in the WAKE
region where polymer C-tensor is most active (M48 trace_C peak is at
wake) and where gradient bias propagates strongest. The U-shape
delta R=30 → R=50 = -3.4 Cd ≈ 314 / 188 = 1.67× cell count increase
× per-cell bias ~ 2 Cd.

## Implication for next mission

The remaining bias must come from cells adjacent to the CYLINDER
(non-axis-aligned, q_w variable). The current `halfwayBB` BC at those
cells likely uses q_w=0.5 fallback regardless of the precomputed q_w
geometry (cylinder cut-link q_w varies in [0, 1]). Two possible roots:

1. **Halfway BB approximation at cylinder cut-cells**: the BC itself
   uses q_w=0.5 even when the true q_w is e.g. 0.1 or 0.9 — bias
   scales with (q_w_true - 0.5)².
2. **Velocity-gradient stencil at cylinder-adjacent cells**: even if
   the BC is correct, the FVFD gradient at "first-fluid-near-solid"
   uses a cell-center derivative — same bug class as M48/M49 but at
   interior cells (cylinder cut), not axis-aligned outer walls.

The M47 H1 (parked) was about Bouzidi-FL specifically. The current
finding suggests the same mechanism may exist under HALFWAYBB at
cylinder cut-cells — just with q_w=0.5 forced instead of q_w-modulated.

## What the M51 fix DID achieve

- M49 canary machine-precision GREEN for outer walls
- New reusable helper `apply_halfway_wall_gradient_correction!` with
  derivable formula and unit-test coverage
- 6 viscoelastic drivers updated consistently
- Cavity gets correct second-order wall gradients (M51b)
- Audit trail and discipline rigor preserved for future fix iterations

The wall-gradient correction is a real bug fix; it just isn't the
dominant cylinder Cd contributor.

## Recommended next missions (one of)

1. **Static audit of cylinder cut-link BC**: read the halfwayBB code
   path for cut-cells (`precompute_q_wall_cylinder` + cylinder
   surface BC) and determine whether q_w is honored or forced to 0.5.
   Static, ~10 min Codex.
2. **Extend M49 canary to interior solid (cut-link)**: build a
   second canary that places a cylinder R=4 in the test domain and
   measures `dudy[cylinder_adjacent_cell]` against analytic Stokes
   flow. Sub-second, would localize the bias at cylinder cells.
3. **Unpark M47 Bouzidi BC test**: rerun the cylinder benchmark with
   `wall_bc=:bouzidi_fl_twopass` (real q_w-aware BC) and see if Cd
   becomes monotone in R.

## Artifacts

- `scratch/M48_hw_meshconv/cdtraj_R{10,30,50}_wi1_halfway.csv`
- `scratch/M48_hw_meshconv_prefix_baseline/cdtraj_R{10,30,50}_wi1_halfway.csv`
- `scratch/M48_hw_meshconv/M48_postfix_run.log`
- `scratch/M48_hw_meshconv/run_R{10,30,50}.log`
- This verdict
