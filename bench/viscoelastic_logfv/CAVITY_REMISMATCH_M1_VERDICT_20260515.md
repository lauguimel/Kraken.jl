# Cavity Re-mismatch sweep — M1 verdict (2026-05-15)

## TL;DR

**Candidate 1 (Re mismatch) is NOT the dominant source of the cavity
profile gap.** Sweeping `u_max` from 0.005 to 0.001 (dropping `Re_LU`
from ~6.4 to ~1.3 by a factor of 5) leaves the centerline-velocity L2
unchanged at ~18 % and only marginally affects the horizontal `psi_xy`
L2 (24.4 % → 23.8 %). Move on to M2 (wall-gradient corner artifact)
and M3 (polymer upwind / frozen replay) per Mandate §6.

## Setup

- Branch: `dev-viscoelastic`, commit `faf71c78`.
- Aqua job: `21339238.aqua`, gpu_batch_exec, 1 GPU on `gpu1n014`,
  walltime 04:22:15, exit_status 0.
- Single PBS job:
  [`bench/viscoelastic_logfv/run_cavity_remismatch_sweep.pbs`](run_cavity_remismatch_sweep.pbs).
- Per-case wallclock (from output subdir timestamps):

  | u_max | started | wallclock |
  |-------|---------|-----------|
  | 0.005 | 17:21 | ~32 min |
  | 0.002 | 17:21 → 18:38 | ~77 min (≈ 2.4× baseline) |
  | 0.001 | 18:38 → 21:11 | ~153 min (≈ 4.8× baseline) |

  Scaling consistent with 1/u_max (smaller lid velocity needs more steps
  for the same physical end_time = 8).

- Common parameters (rheoTool match on `De=1`, `beta=0.5`):
  `N=64`, `lambda_phys=1.0`, `nu_s=nu_p=0.1`, `bsd_fraction=0.75`,
  `end_time=8.0`, polymer model = Oldroyd-B, `bench/rheotool/cavity_oldroydb_log_re001_de1_b05`.

## Observable: relative L2 vs rheoTool reference

Analysis script:
[`bench/viscoelastic_logfv/analyse_cavity_remismatch.jl`](analyse_cavity_remismatch.jl)
(stdlib-only, denominator = rheoTool L2 norm).

| u_max | L2(u centerline) | L2(psi_xy y=0.75) |
|-------|------------------|-------------------|
| 0.005 | 0.1797 | 0.2441 |
| 0.002 | 0.1796 | 0.2396 |
| 0.001 | 0.1795 | 0.2381 |

Centerline L2 changes by less than 0.2 %; horizontal psi_xy L2 changes
by ~2.5 %. Both stay in the same band as the original baseline
(18-24 % per `NEXT_SESSION_PROMPT_20260515_cavity_spatial.md`).

## Interpretation

The Re mismatch hypothesis predicted L2 drops monotonically and
substantially with smaller u_max. The observation is essentially flat.
Inertia plays a negligible role in the cavity profile gap at the
current De/beta/N. The gap is therefore not driven by the finite
Re_LU but by something local to the spatial coupling or polymer
discretisation.

## Decision

Per Mandate §6 dependency graph: launch M2 and M3 in parallel.

- **M2** — wall-gradient correction at the moving-lid corner (cheap CPU
  smoke at N=32, t=2, with the corner cell no-op).
- **M3** — polymer upwind / frozen-replay (adapt
  `run_rheotool_frozen_replay_2d.jl` to cavity).

M4 (Guo/FD divergence) and M5 (kinetic-moment BSD) remain deferred
until M2/M3 close or do not close the gap.

## Artefacts

- Raw Aqua results: `tmp/cavity_remismatch/u{0.005,0.002,0.001}/kraken_N64_*/`
- PBS stdout: spooled on Aqua, not synced (Exit_status 0; rsync only
  pulled the analysis outputs).
- Analysis stdout (also see header of this file):

  ```text
  u_max    | L2(u_centerline) | L2(psi_xy_y=0.75)
  -------- | ---------------- | -----------------
  0.005    | 1.797e-01        | 2.441e-01
  0.002    | 1.796e-01        | 2.396e-01
  0.001    | 1.795e-01        | 2.381e-01
  ```
