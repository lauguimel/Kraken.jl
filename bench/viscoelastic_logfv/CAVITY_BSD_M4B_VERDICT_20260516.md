# Cavity BSD-fraction sweep — M4b verdict (2026-05-16)

## TL;DR

**The BSD correction is helping, not hurting.** Sweeping
`bsd_fraction` from 0.0 to 0.75 at fixed `N=64`, `t=8`, `De=1`,
`beta=0.5`, `u_max=0.005`, the relative L2 vs rheoTool falls
monotonically as `bsd_fraction` increases. The current production
choice (`bsd_fraction = 0.75`) is the best of the swept set; reducing
BSD makes the cavity profile match worse, not better. The M4
hypothesis that the BSD operator mismatch drives the 18-24 % profile
gap is therefore **refuted**. The next lever to pull is **M6-B
(wall-BC stencil match with rheoTool's `linearExtrapolation` on τ)**.

## Setup

- Branch: `dev-viscoelastic`, commit `26bc0206` (post-M5-B
  infrastructure commit; behaviour unchanged since `bsd_kind=:fd`
  default).
- Aqua job: `21385031.aqua`, gpu_batch_exec, 1 GPU on `gpu0n008` (the
  job was requeued overnight from `gpu1n014` — explains the walltime
  reset). Final walltime 02:23:06, Exit_status 0.
- Per-case wallclock from output subdir timestamps (after requeue):

  | bsd_fraction | started | ended | wallclock |
  |--------------|---------|-------|-----------|
  | 0.0 | 11:33 | 12:08 | ~35 min |
  | 0.25 | 12:08 | 12:43 | ~35 min |
  | 0.5 | 12:43 | 13:18 | ~35 min |
  | 0.75 | 13:18 | 13:54 | ~36 min |

  Consistent ~35 min per case — confirming the constitutive cost is
  bsd-independent (only the body-force assembly changes).

- Common parameters: `N=64`, `lambda_phys=1.0`, `nu_s=0.1`,
  `nu_p=0.1`, `u_max=0.005`, polymer model Oldroyd-B,
  `bench/rheotool/cavity_oldroydb_log_re001_de1_b05`.

## Observable: relative L2 vs rheoTool reference

Analysis script: [`analyse_cavity_remismatch.jl`](analyse_cavity_remismatch.jl)
(stdlib-only, denominator = rheoTool L2 norm).

| bsd_fraction | L2(u centerline) | L2(psi_xy y=0.75) |
|--------------|------------------|-------------------|
| 0.0 | 0.2115 | 0.2741 |
| 0.25 | 0.2022 | 0.2639 |
| 0.5 | 0.1917 | 0.2538 |
| 0.75 | 0.1797 | 0.2441 |

Monotonic decrease across the sweep:
- Centerline u: 21.15 % → 17.97 % (improves by ~3.2 percentage points).
- Horizontal psi_xy: 27.41 % → 24.41 % (improves by ~3.0 pp).

Slope of the centerline trend `dL2/dζ ≈ −0.043` over `ζ ∈ [0, 0.75]`.
Linear extrapolation to `ζ = 1.0` (forbidden by the lid corner crash)
would project L2 ≈ 17.0 %, only ~1 pp below current.

## Interpretation

The hypothesis (M4) was that the BSD correction is the dominant
source of the 18-24 % profile gap because the operator mismatch
between FD-central `∇²u` and the LBM lattice stencil produces a
54 % discrepancy between the Guo-applied body force and the pure FD
divergence of `τ`. The sweep tests this directly: if the BSD term is
introducing harmful inconsistency, reducing it (towards `ζ = 0`)
should reduce the profile error.

The data shows the opposite: the BSD term **improves** the rheoTool
match. The 54 % F discrepancy measured in M4 reflects the magnitude
of the BSD correction operating as designed, not a defect.

Therefore:
- M4 was a correct measurement but mis-interpreted as a bug.
- M5-B (kinetic-moment BSD as infrastructure) was the right
  refactor architecturally but cannot close the gap because the
  existing FD-BSD path is not the problem.
- The remaining 18 % centerline gap at `ζ = 0.75` must come from
  some other source — wall stencil on `τ`, possibly polymer
  constitutive at the wall, possibly the Re_LU mismatch interacting
  non-linearly (M1 ruled out the linear Re effect but a coupling
  remainder could exist).

## Decision

Pivot to **M6-B (wall-BC stencil match with rheoTool)**. The audit
(M6-A) predicted a 15-30 percentage-point drop at the F-discrepancy
level at cell (16, 63) if Kraken adopts rheoTool's
`linearExtrapolation` on `τ` at the moving lid. If the profile L2
follows the same direction, M6-B alone could bring the gap from
18-24 % to single-digit percent — closing the mission.

Sequencing (per M6-A's warning about `operators_2d.jl` overlap with
M5-B): M5-B is now committed but it does NOT touch `operators_2d.jl`,
so M6-B can proceed directly without conflict.

## Artefacts

- Raw Aqua results:
  `tmp/cavity_bsd_sweep/bsd{0_0,0_25,0_5,0_75}/kraken_N64_*/`
- PBS stdout: spooled on Aqua, not synced.
- Analysis stdout (reproduced for the record):

  ```text
  u_max    | L2(u_centerline) | L2(psi_xy_y=0.75)
  -------- | ---------------- | -----------------
  bsd0_0   | 2.115e-01        | 2.741e-01
  bsd0_25  | 2.022e-01        | 2.639e-01
  bsd0_5   | 1.917e-01        | 2.538e-01
  bsd0_75  | 1.797e-01        | 2.441e-01
  ```

  (Column header shows `u_max` because the analyse_cavity_remismatch
  script infers labels from the parent dir name `u<value>`; for this
  sweep the dirs are `bsd<value>` so the label fallback shows the
  kraken timestamp instead. Mapping by sweep order: row 1 = bsd=0.0,
  row 4 = bsd=0.75.)
