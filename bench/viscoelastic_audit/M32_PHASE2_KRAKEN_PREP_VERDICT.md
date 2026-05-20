# M32 Phase 2 — Kraken-side PBS preparation verdict

**Date:** 2026-05-21
**Department:** M32-Phase2-kraken-prep
**Mandate:** Prepare two PBS Aqua A100 F64 CUDA scripts (Newtonian
sanity + viscoelastic R x Wi matrix). Do NOT submit — the Boss will
run `qsub` on both.

---

## 1. Files prepared

### PBS 1 — Newtonian sanity check

* **Path:**
  `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/bench/viscoelastic_logfv/run_cyl_m32_newtonian_sanity_a100.pbs`
* **Job name:** `M32_newtonian_sanity`
* **Resources:** 1 node, 8 CPU, 1 A100 GPU, 64 GB, walltime 1 h
* **Cases:** 1 (R=30, beta=1.0, Re=1, Wi=1.0 placeholder, bsd=1.0,
  qwall, no embedded, rusanov)
* **Output dir:** `tmp/m32_newtonian_sanity/${PBS_JOBID:-manual}`
* **Snapshot:** `KRAKEN_SAVE_FIELDS=1` — persists (ux, uy, tauxx,
  tauxy, tauyy, is_solid, rho); at nu_p=0 only (ux, uy, rho, is_solid)
  are useful for wall decomposition.
* **Gate:** Kraken `Cd_s` vs rheoTool Newtonian Cd at R=30 Re=1 must
  agree to <2 %. If this fails, no viscoelastic comparison is
  meaningful — H1/H3 dominate.

### PBS 2 — Viscoelastic matrix R ∈ {30, 60} × Wi ∈ {0.1, 1.0}

* **Path:**
  `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/bench/viscoelastic_logfv/run_cyl_m32_matrix_a100.pbs`
* **Job name:** `M32_matrix`
* **Resources:** 1 node, 8 CPU, 1 A100 GPU, 64 GB, walltime 4 h
* **Cases:** 4 (R ∈ {30, 60} × Wi ∈ {0.1, 1.0}, beta=0.59, Re=1,
  bsd=1.0, qwall, no embedded, rusanov)
* **Output dir:** `tmp/m32_matrix/${PBS_JOBID:-manual}`
* **Snapshot:** `KRAKEN_SAVE_FIELDS=1` — per-case `.jls` includes the
  M30 schema extension (rho persisted) so the azimuthal wall-pressure
  profile `p(theta) = c_s^2 * rho(theta)` is reconstructible without
  re-running.

---

## 2. Exact `qsub` commands (Boss to run)

From the repo root on Aqua (remote path `~/Kraken.jl-viscoelastic-run/`):

```
qsub bench/viscoelastic_logfv/run_cyl_m32_newtonian_sanity_a100.pbs
```

```
qsub bench/viscoelastic_logfv/run_cyl_m32_matrix_a100.pbs
```

Both PBS scripts use the standard `gpu_batch_exec` queue (implicit
from the `select=...:ngpus=1:gpu_id=A100` resource line).

**Maintenance-window sanity** (engineer.md 2026-05-19 pattern 2): the
mandate states Aqua maintenance ended, so the 1 h / 4 h walltimes
should not be held at the boundary. Still, the Boss should run
`qstat -u maitreje` immediately after each `qsub` to verify the job
moves R within ~30 s (not stuck Q with "Job would cross dedicated
time boundary" comment).

---

## 3. Expected outputs

### PBS 1 — Newtonian sanity

Layout under `tmp/m32_newtonian_sanity/${JOBID}/`:

```
SUMMARY.csv                                       # 1 row
cyl_bigsweep_v2_beta1_wi1_re1_R30_bsd1_..._geomqwall.csv   # per-case CSV
cyl_bigsweep_v2_beta1_wi1_re1_R30_bsd1_..._geomqwall_fields.jls   # snapshot
```

Key CSV columns to inspect (per
`bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl` lines 129-141):
`Cd_kraken`, `Cd_s`, `Cd_p`, `Cd_bsd`, `nan_flag`, `walltime_s`,
`completed_steps`, `MLUPS`.

At beta=1.0, `nu_p = 0`, so `Cd_p = Cd_bsd = 0` and
`Cd_kraken = Cd_s` exactly. The single comparable number for the
gate is `Cd_s`.

### PBS 2 — Viscoelastic matrix

Layout under `tmp/m32_matrix/${JOBID}/`:

```
SUMMARY.csv                                                 # 4 rows
cyl_bigsweep_v2_beta0p59_wi0p1_re1_R30_bsd1_..._geomqwall.csv
cyl_bigsweep_v2_beta0p59_wi0p1_re1_R30_bsd1_..._geomqwall_fields.jls
cyl_bigsweep_v2_beta0p59_wi0p1_re1_R60_bsd1_..._geomqwall.csv
cyl_bigsweep_v2_beta0p59_wi0p1_re1_R60_bsd1_..._geomqwall_fields.jls
cyl_bigsweep_v2_beta0p59_wi1_re1_R30_bsd1_..._geomqwall.csv
cyl_bigsweep_v2_beta0p59_wi1_re1_R30_bsd1_..._geomqwall_fields.jls
cyl_bigsweep_v2_beta0p59_wi1_re1_R60_bsd1_..._geomqwall.csv
cyl_bigsweep_v2_beta0p59_wi1_re1_R60_bsd1_..._geomqwall_fields.jls   # may be absent if R=60 Wi=1 NaN's pre-snapshot
```

The bench loops `BETA × WI × RE × R × BSD × GEOM × EMBEDDED`
(`run_cyl_bigsweep_v2_2d.jl` line 411-413) — with all single-value
lists except `KRAKEN_R_LIST="30,60"` and `KRAKEN_WI_LIST="0.1,1.0"`,
this yields exactly 2 × 2 = 4 iterations.

Per-case CSV is always written (even on NaN, with `nan_flag=true`);
`.jls` snapshot is gated on `status == :ok && result !== nothing`
(line 341 of the bench), so failed cases produce a CSV but no
fields file.

---

## 4. Walltime estimates

| PBS | Cases | Per-case (F64 A100) | Total wall | Walltime budget |
|---|---|---|---|---|
| Newtonian sanity | 1 (R=30) | ~5-10 min | ~5-10 min | 1 h (15× safety) |
| Matrix | 4 (R=30 ×2, R=60 ×2) | ~5-15 min (R=30), ~20-60 min (R=60) | ~50-150 min | 4 h (~2× safety) |

Per-case time scales with Nx × Ny × max_steps. At R=30, the grid is
(15+15)·30 × 4·30 = 900 × 120 = 108 k cells × 100 k steps ≈ 1.08e10
cell-updates. At Aqua A100 F64 (~600-800 MLUPS measured on prior
M28/M30 runs), this is ~14-18 s for the LBM macro pass; the polymer
substep loop (up to 64 substeps/step) typically adds 5-10× → 5-15 min
total. R=60 is 4× more cells (432 k) and longer relaxation
(`lambda = Wi · R / u_mean = 1 · 60 / 0.005 = 12 000 LU`), so wall
time scales superlinearly → 20-60 min per case.

The 4 h matrix budget assumes the worst case (both R=60 cases
complete + low MLUPS). If R=60 Wi=1.0 NaN's early (step-0 or
mid-run), the matrix may finish in ~2 h.

---

## 5. R=60 stability risk (Boss must read)

Phase 1 R-sweep on Metal F32 showed **R=60 Wi=1 NaN'd**. F64 CUDA
provides more headroom (log_spd eigenvalue floor at ~1e-16 instead
of F32's ~1e-7), but cannot guarantee convergence at this
elasticity/resolution combination.

**Failure modes to watch:**

- **Step-0 NaN** (M28e observed at R=60/80 on F64) — root cause
  likely an init/psi→C exponentiation stability issue, not a
  max_steps issue (see engineer.md 2026-05-19 entry, `KRAKEN_MAX_STEPS_BASE`
  pattern). The case fails before any LBM stepping; SUMMARY.csv row
  has `completed_steps=0`, `nan_flag=true`, `first_nonfinite_step=0`.
- **Mid-run DomainError** (log_spd raises on substep) — the bench's
  `try / catch DomainError` wrapping (see department.md 2026-05-18
  pattern) ensures the remaining cases still run.

**Verdict logic on Boss return:**

- All 4 cases pass: matrix is conclusive; cross-check vs rheoTool
  wi0.1 (Cd_total 130.43) and wi1.0 (Cd_total ~120.40 with drift).
- R=60 Wi=1.0 only NaN: R-sweep extends to R=40 (interpolation from
  R=30→R=40 plus R=60 Wi=0.1 endpoint may suffice for resolution
  trend at low Wi).
- Both R=60 cases NaN: "R-sweep cannot reach R=60 in this regime,
  port confined to R ≤ 40" — Phase 2 verdict accepts this as a
  documented systematic and the M32 reference matrix runs at R ∈
  {30, 40} only.

---

## 6. Notes / open items

* **No `src/` touch** — preparation only, both PBS scripts invoke
  the existing `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl`
  which already consumes `KRAKEN_R_LIST` and `KRAKEN_WI_LIST` as
  comma-separated lists (verified at bench lines 99 / 97; double-
  loop at line 411-413).
* **No commits, no push** — mandate explicit.
* **No ssh / qsub attempts** — preparation only; the Boss will
  submit.
* **Cd_kraken normalisation** — Phase 1 audit C1 flagged that
  rheoTool reports `Cd = Fx / (etaS + etaP)` (Hulsen K convention),
  not the classical `2·Fx / (rho·U^2·D)`. Kraken's
  `result.Cd_kraken` convention must be verified by the Boss /
  follow-up Department before any cross-code comparison; this PBS
  prep does not address C1.

---

## 7. Files touched

* `bench/viscoelastic_logfv/run_cyl_m32_newtonian_sanity_a100.pbs` (new)
* `bench/viscoelastic_logfv/run_cyl_m32_matrix_a100.pbs` (new)
* `bench/viscoelastic_audit/M32_PHASE2_KRAKEN_PREP_VERDICT.md` (this file)

---

## 8. Memory candidates (for Boss / future Department)

* **M32 Phase 2 PBS template pattern**: comma-separated `KRAKEN_R_LIST`
  + `KRAKEN_WI_LIST` is the canonical way to express a 2D Kraken
  parameter matrix in a single PBS, reusing the bench's existing
  nested-loop semantics. No script duplication needed for grid
  sweeps. (Already documented in M28-cluster memory but worth
  re-stating: a sweep is one PBS, not N PBS files.)
* **Newtonian gate before viscoelastic comparison**: any future
  cross-code Cd comparison must first pass a Newtonian sanity check
  (beta=1.0, nu_p=0) on the same geometry. If Cd_s differs by >2 %
  between codes, the staircase/mesh/domain mismatch is the load-
  bearing gap and polymer comparisons are noise on top. Promotes
  Phase 1 audit gate G3 from "recommended" to "mandatory for any
  cross-code Department brief".
