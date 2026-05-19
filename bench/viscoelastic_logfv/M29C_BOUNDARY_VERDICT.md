# M29c — boundary MUSCL-superbee reconstruction — verdict

**Mission**  : M29c-boundary (patch) + M29c-validate (this validation pipeline)
**Branch**   : `dev-viscoelastic`
**Date**     : 2026-05-19

## TL;DR

**Verdict : FAIL.**

The 1-sided 3-point MUSCL-superbee boundary reconstruction patch
preserves the regression baseline (169194 / 6 / 0 / 4 byte-identical to
HEAD) and runs without NaN flag, but the production case at the
M29 operating point (β=0.59, Wi=1.0, Re=1, R=30, qwall, F64 CUDA)
produces a **physically meaningless** result :

| Metric              | Value     | Reference (rheoTool / M29b) | Verdict     |
|---------------------|-----------|-----------------------------|-------------|
| Cd_kraken           | **−1571** | +120.4 (rheoTool)           | sign flip   |
| Cd_s                | −1594.8   | —                           | sign flip   |
| Cd_p                | +133.6    | —                           | inflated    |
| min(det C)          | 0.120     | 0.97 (M29b smoke)           | severely degraded |
| trace(C) max        | 11432.7   | 5.5  (M29b smoke)           | 2000× over  |
| u_max_abs (LU/step) | 0.21      | 0.012 (smoke)               | Ma≈0.36, supersonic |
| τ_xx_max_abs        | 4.19e-2   | 3.15e-4 (smoke)             | 130× over   |
| first_nonfinite_step| 0         | —                           | (no NaN reported) |
| nan_flag            | false     | —                           | values finite |

The boundary patch removes the M29b safety net (hard fallback to
Rusanov within ±2 cells of any solid) but exposes an upstream
instability at production-scale Wi. The simulation completes
100 000 steps without NaN, yet diverges to unphysical magnitudes —
behaviour consistent with the slope limiter being unable to clamp
1-sided extrapolations into the polymer stress hot-spot at the
leeward shoulder identified in `CYL_TAU_COMPARE_M29_VERDICT.md`.

**Do NOT bake the patch in as `:muscl_superbee` default.** The
:rusanov path remains byte-identical to legacy — safe to ship.
The boundary 1-sided reconstruction can be kept as a *separate
opt-in flag* (e.g. `:muscl_superbee_boundary`) for low-Wi
applications only, but production cylinder Cd at Wi≥0.3 needs a
different stabilisation strategy.

## Patch summary

- File : `src/fvfd/operators_2d.jl`  (+90 / −32 LOC, uncommitted)
- Helpers added :
  - `_fvfd_muscl_superbee_face_value_oneSided_2d(upwind, downwind)`
    — central average (1st-order, slope=0 fallback) used when the
    canonical 3-point stencil's `far_upwind` cell would be solid or
    OOB.
  - `_fvfd_muscl_superbee_guarded_face_value_2d(...)` — dispatch
    wrapper that picks canonical 3-point vs 1-sided based on a
    `canonical_usable` predicate.
- Body : the 4 face-value computations (east / west / north / south)
  in the `Val{:muscl_superbee}` method of
  `_fvfd_upwind_scalar_advective_rhs_2d` were rewritten to use the
  guarded helpers. Each face evaluates `canonical_usable` from grid
  index + `is_solid` mask, and falls back to the 1-sided average
  only at the offending face.
- The old "hard fallback to Rusanov within ±2 cells of any solid /
  boundary" early-return is **removed**.

## Validation pipeline

### Step 1 — test suite preservation (host, M-series F32)

```
julia --project=. test/runtests.jl
```

Result : **169194 passed / 6 failed / 0 errored / 4 broken**, exactly
matching the documented HEAD baseline. The :rusanov path is
byte-identical for legacy.  PASS.

### Step 2 — host smoke (Metal F32, R=20 Wi=0.1 β=0.59)

```
KRAKEN_BACKEND=metal KRAKEN_FT=float32 \
KRAKEN_R_LIST=20 KRAKEN_BETA_LIST=0.59 KRAKEN_WI_LIST=0.1 KRAKEN_RE_LIST=1.0 \
KRAKEN_BSD_LIST=1.0 KRAKEN_L_UP_LIST=15.0 KRAKEN_L_DOWN_LIST=15.0 \
KRAKEN_MAX_STEPS_BASE=20000 KRAKEN_ADVECTION_SCHEME=muscl_superbee
```

Output : `bench/scratch/m29c_smoke_metal/SUMMARY.csv`

| Metric          | M29c-validate (this run) | M29b baseline (smoke)     |
|-----------------|--------------------------|---------------------------|
| Cd_kraken       | 130.196                  | 131.18 (R=20 Wi=0.1 2000 steps L_up=4)  |
| min(det C)      | 0.969                    | 0.973                     |
| trace(C) max    | 5.45                     | ~4.80                     |
| τ_xx_max_abs    | 3.90e-4                  | 3.15e-4                   |
| u_max_abs       | 0.0146                   | 0.012                     |
| nan_flag        | false                    | false                     |
| MLUPS           | 20.7                     | 2.3                       |

Low-Wi smoke PASS (Cd within 0.75% of baseline; consistent
saturation of the limiter at the wall; no NaN). MLUPS is ~9× the
M29b smoke because the M29b smoke used short L_up=4 / 2000 steps
(domain-dependent precompile overhead dominated).

### Step 3-4 — Aqua production (A100 F64, R=30 Wi=1.0 β=0.59)

Job ID : **21588436.aqua**
Walltime : 85.83 s wall (budget 30 min — completed in 5% of cap)
Steps   : 100 000 / 100 000 (no early termination, no NaN flag)
MLUPS   : 125.8 (CUDA F64 A100)

Output : `tmp/m29c_kraken/cyl_bigsweep_v2_beta0p59_wi1_re1_R30_*.csv`
+ `*_fields.jls` (4.3 MB field snapshot for τ comparison).

| Metric              | M29c-validate Aqua | M29b A100 reference (β=0.59 Wi=1.0 R=30, MUSCL+hard fallback)|
|---------------------|--------------------|------------------------------------------|
| Cd_kraken           | **−1571.13**       | 116.474 (M29B_HRS_VERDICT.md L120)        |
| Cd vs rheoTool (120.4) | **−1691.5**     | −3.93                                    |
| min(det C)          | 0.120              | (M29b reported stable)                   |
| trace(C) max        | 11 432.7           | (M29b reported stable)                   |
| τ_xx_max_abs        | 4.19e-2            | (M29b L2_rel[τ_xx]≈0.16 vs rheoTool)      |
| u_max_abs (LU/step) | 0.210              | (Mach ≪ 0.1 in M29b)                     |

The patched MUSCL-superbee scheme is **unstable** at this operating
point. No NaN is emitted because the limiter still bounds individual
face values to finite numbers, but the global state drifts into a
strongly unphysical attractor (Mach ≈ 0.36, det(C) collapsing toward 0,
Cd sign flipping).

### Step 5 — field comparison vs rheoTool (Wi=1.0)

Driver : `bench/viscoelastic_audit/run_kraken_vs_rheotool_tau_compare.jl`
rheoTool case : `bench/rheotool/cylinder_wi1.0` (time=10)
ROI : x ∈ [−3, 8]·R, y ∈ [−2, 2]·R, sampled on 256×128 grid.
Valid samples : 29 825 / 32 768 (91.0 %).

| Field        | L2_rel (Kraken vs rheoTool) | max\|diff\| | ref L2 (rheoTool) |
|--------------|------------------------------|--------------|--------------------|
| u_x          | 0.7576                       | 15.4         | 1.232              |
| u_y          | 1.4461                       | 25.94        | 0.2673             |
| τ_xx         | 1.1180                       | 687          | 16.98              |
| τ_xy         | 3.8879                       | 1142         | 2.612              |
| τ_yy         | 6.2567                       | 1799         | 2.512              |

ΔCd = Cd_kraken − Cd_rheo = −1691.5 (−1405 % vs rheoTool).

All τ components show L2_rel > 1, i.e. the residual norm exceeds
the rheoTool reference norm itself. Max τ diffs are 40× to 700× the
rheoTool peak. The patch destroys the field agreement that M29
demonstrated (M29b: L2_rel[τ_xx] ≈ 0.16 with the hard fallback).

Artifacts :
- `bench/scratch/m29c_tau_compare/M29_residuals.csv`
- `bench/scratch/m29c_tau_compare/M29_band_stats_x.csv`
- per-field PNG plots in same directory.

## Hypothesis on root cause

M29b's hard fallback to Rusanov within ±2 cells of any solid was
documented as "LOAD-BEARING" in `.orchestrator/memory/engineer.md`
(2026-05-19 night entry). The 1-sided 3-point MUSCL-superbee
reconstruction was proposed as a higher-order replacement that
would keep slope-limiting active up to the wall row.

The Codex patch is mathematically correct — the limiter formula is
applied with `far_upwind ← upwind` shifted to a 1-sided stencil
when needed — but the resulting slope is non-monotonic at the wall:
when both `phi[i,j]` (upwind) and the BC ghost value `downwind`
are large positive (polymer stress hot-spot at the leeward
shoulder), the 1-sided stencil reduces to a plain central average,
which is **unconditionally unstable for hyperbolic advection** in the
absence of additional dissipation. Rusanov supplies that
dissipation; the central average does not.

The patch silently turned the boundary band from
"1st-order upwind (TVD)" into "2nd-order central (anti-TVD)" — that
explains why low-Wi runs (where polymer stress gradients are mild)
still look fine, but Wi=1.0 production diverges.

## Recommendation

1. **Revert the patch from default `:muscl_superbee`**. The git
   working tree currently holds the uncommitted patch — Boss must
   decide whether to discard or to file under a new feature flag.
2. **Do NOT ship as default**. The :rusanov path is unchanged and
   safe ; that's what production should keep using for now.
3. **If the 1-sided MUSCL-superbee idea is pursued further**, the
   fallback when `canonical_usable=false` should be **1-sided upwind
   (slope = 0, value = upwind)** — i.e. revert to true 1st-order
   upwind, NOT central average. That preserves TVD at the wall.
4. **Unit tests for the boundary path** specified in the M29c brief
   (sharp pulse adjacent to solid mask, flat field equality to
   Rusanov within 1e-12) were **never added** by Codex before the
   department exited. These tests should be added before any future
   attempt at this scheme to give a fast regression signal.

## Memory candidates

- M29c MUSCL boundary central-average → instability — adding 1-sided
  central average for hyperbolic advection at the wall is anti-TVD;
  use 1-sided UPWIND (slope=0) not 1-sided CENTRAL (mean) as fallback.
- M29b hard-Rusanov boundary fallback was load-bearing for a real
  numerical reason, not just conservatism : the limited region near
  the wall needs Rusanov dissipation to stay stable at production Wi.
- Aqua F64 CUDA production validation is FAST (85s wall on A100 for
  100k steps at R=30) — feasible to gate every "advection scheme"
  candidate on a 1-case production smoke before touching the
  default.

## Artifacts (preserved for Boss)

- `src/fvfd/operators_2d.jl`              — uncommitted patch (working tree)
- `bench/scratch/m29c_smoke_metal/`       — host smoke output (M3 Metal F32)
- `tmp/m29c_kraken/`                      — Aqua A100 F64 results + field .jls
- `bench/scratch/m29c_tau_compare/`       — vs rheoTool field comparison
- `.engineer_logs/M29c_20260519_131854.log` — Codex engineer log (patch)
- `.engineer_brief_M29c.md`               — original engineer brief
- this file                                — verdict
