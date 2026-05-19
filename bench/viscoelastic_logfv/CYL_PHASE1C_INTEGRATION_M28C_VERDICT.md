# Cylinder Cd Phase 1c — M28c integration-time sweep — 2026-05-19

## Mission

Verify whether the M28 Phase 1 baseline Cd_kraken=111.55 at
Wi=1.0, R=30, β=0.59, Re=1, bsd_fraction=1.0, `0000_qwall` (the
Liu reference mode) is **steady-state-converged** at the default
`KRAKEN_MAX_STEPS_BASE=100000` or only transiently under-resolved
because the BSD average window (`KRAKEN_AVG_WINDOW_FRAC=0.2` → last
20 000 steps) closes before the elastic transients have fully
relaxed at λ=6000 LU.

The driver averages `Cd_kraken`, `Cd_s`, `Cd_p`, `Cd_bsd`,
`trace_C_max`, etc. over the last `avg_window = round(max_steps *
0.2)` steps; if those windows differ across max_steps in Cd by more
than ~0.05 (≈ 0.05 % at Cd ≈ 111.5), the 100k baseline is itself
slipping and the Kraken-vs-Liu discrepancy interpretation needs
revisiting.

## Setup

Single case repeated at three integration times:

- R=30, Wi=1.0, β=0.59, Re=1, bsd_fraction=1.0
- L_up=L_down=15, embedded_{gradient,advection,force,drag}=0,
  embedded_geometry=`qwall` (Liu reference mode, bug-free path).
- `KRAKEN_MAX_STEPS_BASE` ∈ {100 000, 300 000, 1 000 000}.
- A100 F64, CUDA backend; jobs submitted via
  `/tmp/qsub_m28c.sh` with explicit `qsub -l walltime` override
  (the default 24 h PBS walltime crossed the 20-May maintenance
  boundary and held all jobs in Q).

## Results

| max_steps | avg_window | Cd_kraken | Cd_s | Cd_p | Cd_bsd | trace_C_max | walltime |
|---|---|---|---|---|---|---|---|
| 100 000 | 20 000 | **111.5469491** | 115.5558013 | 11.5779467 | 15.5867989 | 185.8630 | 77.45 s |
| 300 000 | 60 000 | **111.5503664** | 115.5582800 | 11.5788422 | 15.5867558 | 185.8647 | 169.37 s |
| 1 000 000 | 200 000 | **111.5503667** | 115.5582804 | 11.5788422 | 15.5867558 | 185.8647 | 490.48 s |

Job IDs (Aqua A100 F64):

- `21579957.aqua` — 100 000 steps (3 min 51 s wall incl. precompile).
- `21579958.aqua` — 300 000 steps (5 min 22 s wall incl. precompile).
- `21579959.aqua` — 1 000 000 steps (10 min 49 s wall incl. precompile).

## Deltas

| transition | Δ Cd_kraken | Δ Cd_s | Δ trace_C_max |
|---|---|---|---|
| 100k → 300k | **+0.00342** | +0.00248 | +0.00171 |
| 300k → 1M  | **+3.0e-7**  | +3.2e-7 | +1.3e-6 |

The 300k → 1M residual is at machine-epsilon level for F64
(`~3e-7 / 111.55 ≈ 3e-9` relative). The 100k → 300k step shows a
genuine drift of +0.003 Cd points, but this is **0.003 %** of the
absolute value — three orders of magnitude smaller than the M28
Phase 1 Kraken-vs-Liu discrepancy of ~17 Cd points
(Liu Wi=1 Cd_s expectation ≈ 128 vs Kraken 111.55).

## Verdict

**The M28 Phase 1 100k baseline is converged.** The Cd_kraken,
Cd_s, Cd_p, Cd_bsd, and trace_C_max windowed averages at 100k
steps differ from the fully-relaxed 1M reference by less than
0.004 Cd (0.003 % relative). At 300k steps they are already
indistinguishable from 1M to F64 round-off in the BSD-relevant
fields.

**The under-integration hypothesis is REFUTED.** The Wi=1
elastic transients fully decay within the first 80 000 steps of
the integration (max_steps − avg_window = 80 000), and the last-20%
window is a faithful steady-state estimator at λ=6000 LU.

## Implication for the Kraken-vs-Liu discrepancy

The Cd_kraken = 111.55 baseline at Wi=1.0, R=30 is **physical
(within the model)**, not a transient artefact. The Kraken
constitutive coupling at finite Wi genuinely deviates from Liu /
rheoTool. The discrepancy must be sought elsewhere:

- **BSD architecture / coupling** (e.g. BSD vs polymer-stress
  decomposition, half-step convective/diffusive split, MEA
  cut-link drag formula at curved boundary).
- **Constitutive (log-conformation) discretisation** at high
  Wi λ/dx (e.g. Kupferman vs Fattal-Kupferman vs Liu log-FV
  variant, ATU advection scheme, ψ→C exponentiation pathway).
- Possibly geometry-quadrature differences invisible at
  Newtonian / low-Wi (but Phase 0 Liu-match showed the Newtonian
  limit is at 0.7 % offset only, so any geometric bias is small).

## Memory candidates

1. **`KRAKEN_AVG_WINDOW_FRAC=0.2` is sufficient for Wi≤1.0 at λ=O(6000 LU)
   with `max_steps_base=100 000`** — gives ≤0.004 Cd drift vs 10× longer
   integration. Verified empirically at R=30, β=0.59. Probably extends
   to higher Wi *iff* trace_C does not blow up, but should be re-checked
   when Wi>1 cases are run.

2. **`max_steps_base=100 000` is the right operating point for the
   bigsweep PBS** — anything larger wastes A100 time with no extra
   accuracy. The driver's `max_steps = KRAKEN_MAX_STEPS_BASE` is a
   straight assignment, no R² scaling (despite the misleading comment
   at line 15 of `run_cyl_bigsweep_v2_2d.jl`).

3. **PBS walltime trap before maintenance windows** — Aqua's
   `gpu_batch_exec` queue defaults to 24 h walltime via the PBS
   header `#PBS -l walltime=24:00:00`. The day before a scheduled
   maintenance, jobs requesting longer than the remaining
   wall-clock are held with `comment = Not Running: Job would
   cross dedicated time boundary`. Override per-job via
   `qsub -l walltime=HH:MM:SS ...` to unblock. (First three
   M28c submissions, jobs 21579942/943/944, hit this and had to
   be `qdel`'d and resubmitted as 21579957/958/959 with explicit
   shorter walltimes 15 min / 30 min / 1 h.)

## Files

- 100k SUMMARY.csv:
  `~/Kraken.jl-viscoelastic-run/results/viscoelastic_logfv/cyl_bigsweep_v2_21579957.aqua/SUMMARY.csv`
  (Aqua).
- 300k SUMMARY.csv:
  `~/Kraken.jl-viscoelastic-run/results/viscoelastic_logfv/cyl_bigsweep_v2_21579958.aqua/SUMMARY.csv`
  (Aqua).
- 1M SUMMARY.csv:
  `~/Kraken.jl-viscoelastic-run/results/viscoelastic_logfv/cyl_bigsweep_v2_21579959.aqua/SUMMARY.csv`
  (Aqua).
- Launcher: `/tmp/qsub_m28c.sh` (local) and `~/qsub_m28c.sh` (Aqua).
- PBS template: `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_a100.pbs`
  (unchanged).
- Driver: `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl`
  (commit `e602726f`, unchanged).
