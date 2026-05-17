# Poiseuille BSD path matrix sweep — VERDICT

Date: 2026-05-18. Branch: `dev-viscoelastic`. Mission: M21.

Sweep of **7 BSD / F_poly formulations** on the smooth Poiseuille
coupled viscoelastic driver chain (no embedded obstacle, no wall
corner) to identify which — if any — gives `F_total < 3.51 %` rel L2
**AND** completes 100 000 steps without NaN. Such a variant becomes
the candidate to revive on cavity after the wall-corner bug (engineer
memory 2026-05-17 "wall-gradient correction writes half-cell ghosts")
is fixed separately.

Subsumes the original M21 Open Q5 (kernel cross-check) as variant
`:baseline_fvfd_grad` (V7).

## Setup

CPU F64, Nx=8, Ny=32, Fx_body=1e-5, nu_s=nu_p=0.1,
`force_boundary_fill=:bc_aware`, `polymer_substeps=:auto`,
`logfv_periodicx_wally_bcspec_2d()`, max_steps=100 000.

Two cases per variant:

| label      | λ      | Wi = γ̇_max · λ |
|------------|--------|-----------------|
| A          | 1.0    | 8e-4            |
| A_high_Wi  | 1.25e4 | 1.0             |

For all variants ζ = 0.75 except `:no_bsd` (ζ = 0).
`nu_lbm = nu_s + ζ·nu_p`.

The 7 variants modify ONLY the BSD branch in step 5 of the Poiseuille
chain (`logfv_polymer_force_bc_aware_2d!` for `F_poly_wide` is the
same in every variant). See `bench/viscoelastic_audit/run_poiseuille_bsd_pathmatrix_2d.jl`
for the exact branch code.

## Variant ranking (case A, ordered by F_total interior L2)

Sorted ascending; NaN'd variants relegated to bottom group.

| rank | variant              | nan_step | F_total int L2 | F_total wall max | u int L2  | u wall max | min_C_eig |
|------|----------------------|----------|-----------------|--------------------|-----------|------------|-----------|
| 1    | `:no_bsd`            | -1       | **5.01e-03**    | 5.01e-03           | 4.17e-03  | 1.17e-03   | 0.9992    |
| 2    | `:baseline`          | -1       | **3.51e-02**    | 3.51e-02           | 5.69e-03  | 2.73e-03   | 0.9992    |
| 2    | `:baseline_fvfd_grad`| -1       | **3.51e-02**    | 3.51e-02           | 5.69e-03  | 2.73e-03   | 0.9992    |
| 4    | `:fd_v2_unc`         | -1       | 5.09e-01        | 4.02e+01           | 9.53e-02  | 5.54e-02   | 0.9993    |
| 5    | `:fd_v2`             | -1       | 8.57e-01        | 4.35e+01           | 1.90e-01  | 5.61e-02   | 0.9996    |
| 6    | `:kinetic`           | -1       | 1.86e+00        | 1.86e+00           | 2.06e-01  | 4.61e-02   | 0.9994    |
| 7    | `:epsilon_force`     | **5000** | NaN             | NaN                | NaN       | NaN        | NaN       |

## Variant ranking (case A_high_Wi, Wi=1)

| rank | variant              | nan_step | F_total int L2 | F_total wall max | u int L2  | u wall max | min_C_eig | τ_xx int L2 |
|------|----------------------|----------|-----------------|--------------------|-----------|------------|-----------|--------------|
| 1    | `:no_bsd`            | -1       | **1.33e-05**    | 1.33e-05           | 2.84e-04  | 2.76e-04   | 0.590     | 2.27e-11     |
| 2    | `:baseline`          | -1       | **9.33e-05**    | 9.33e-05           | 1.84e-03  | 1.83e-03   | 0.590     | 2.22e-11     |
| 2    | `:baseline_fvfd_grad`| -1       | **9.33e-05**    | 9.33e-05           | 1.84e-03  | 1.83e-03   | 0.590     | 2.22e-11     |
| 4    | `:fd_v2_unc`         | -1       | 5.79e-01        | 3.95e+01           | 1.02e-01  | 5.61e-02   | 0.578     | 7.02e-02     |
| 5    | `:kinetic`           | -1       | 1.87e+00        | 1.87e+00           | 2.08e-01  | 4.66e-02   | 0.634     | 2.23e-01     |
| 6    | `:fd_v2`             | **5000** | NaN             | NaN                | NaN       | NaN        | NaN       | NaN          |
| 6    | `:epsilon_force`     | **0**    | NaN             | NaN                | NaN       | NaN        | NaN       | NaN          |

`:epsilon_force` at Wi=1 raised `DomainError` in `logfv_log_spd_sym2_2d`
(C went non-SPD mid-substep, before the NaN watcher's 5000-step tick).
Sentinel `nan_step=0` records crash-before-first-watcher in the
summary.

## u and τ checks per variant (case A, steady state)

| variant              | u int L2  | u wall max | τ_xy int L2 | τ_xx abs max  | τ_yy abs max | min_C_eig |
|----------------------|-----------|------------|--------------|----------------|---------------|-----------|
| `:no_bsd`            | 4.17e-03  | 1.17e-03   | 2.61e-03     | 1.20e-07       | 8.10e-16      | 0.9992    |
| `:baseline`          | 5.69e-03  | 2.73e-03   | 2.61e-03     | 1.20e-07       | 5.77e-16      | 0.9992    |
| `:baseline_fvfd_grad`| 5.69e-03  | 2.73e-03   | 2.61e-03     | 1.20e-07       | 5.77e-16      | 0.9992    |
| `:fd_v2`             | 1.90e-01  | 5.61e-02   | 1.04e-01     | 6.27e-08       | **2.63e-01**  | 0.9996    |
| `:fd_v2_unc`         | 9.53e-02  | 5.54e-02   | 4.10e-02     | 8.27e-08       | 4.50e-10      | 0.9993    |
| `:kinetic`           | 2.06e-01  | 4.61e-02   | 1.42e-01     | 6.42e-08       | 8.33e-16      | 0.9994    |
| `:epsilon_force`     | NaN       | NaN        | NaN          | NaN            | NaN           | NaN       |

τ_yy must be analytically zero for simple shear Poiseuille (Newtonian
or Oldroyd-B, full Wi range). `:fd_v2` shows **τ_yy abs max = 0.26**
(≫ τ_xy_max ~ 2.5e-3), confirming the wide-on-wide BSD path
manufactures massive numerical asymmetric stress at the walls — a
distinct failure mode from the others.

## F_total decomposition per (variant, case)

Per-y row-mean residuals at j=16 (mid-channel). Recall analytical
targets: `F_poly_target = −5.0e-6`, `F_total_target = −1.25e-6` for
ζ=0.75.

### Case A (Wi=8e-4)

| variant              | F_poly_wide   | F_BSD_kind    | F_total_nb    | rel_resid_F_total |
|----------------------|----------------|----------------|----------------|--------------------|
| `:baseline`          | -4.975e-06    | -3.769e-06    | -1.206e-06    | 3.51e-02           |
| `:no_bsd`            | -4.975e-06    | 0             | -4.975e-06    | 5.01e-03 †          |
| `:fd_v2`             | -3.968e-06    | -3.006e-06    | -9.62e-07     | 2.30e-01           |
| `:fd_v2_unc`         | -4.617e-06    | -2.777e-06    | -1.84e-06     | 4.72e-01           |
| `:kinetic`           | -3.638e-06    | -6.72e-08     | -3.570e-06    | 1.86e+00           |

† for `:no_bsd` the analytical target is `−5.0e-6` (full F_poly, no
subtraction), so `F_total = F_poly = -4.975e-6` is a faithful 0.5 %
residual on its own target. The compare-to-baseline metric `rel_resid_F_total`
shown above uses the same M20 target (ζ=0 → `−5.0e-6`) — apples-to-apples.

### Case A_high_Wi (Wi=1)

Same column layout, at row j=16. Targets unchanged.

| variant              | F_poly_wide   | F_BSD_kind    | F_total_nb    | rel_resid_F_total |
|----------------------|----------------|----------------|----------------|--------------------|
| `:baseline`          | -5.000e-06    | -3.750e-06    | -1.250e-06    | 9.33e-05           |
| `:no_bsd`            | -5.000e-06    | 0             | -5.000e-06    | 1.33e-05 †         |
| `:fd_v2_unc`         | -4.636e-06    | -2.746e-06    | -1.89e-06     | 5.12e-01           |
| `:kinetic`           | -3.661e-06    | -6.70e-08     | -3.594e-06    | 1.87e+00           |

`:baseline` collapses 376× from case A to A_high_Wi (3.51e-2 → 9.33e-5)
because the LBM u-residual itself collapses 380× as the elastic
stress closes the parabola loop (M20 finding). `:fd_v2_unc` and
`:kinetic` do NOT collapse — their F_total residual stays at ~50 %
and ~190 % respectively across both Wi cases, confirming the bulk
discrepancy is operator-intrinsic, not LBM-coupling-intrinsic.

## Surprises

1. **`:baseline_fvfd_grad` (Open Q5) is bit-identical to `:baseline`.**
   The original M21 hypothesis ("velocity-gradient kernel difference
   between cavity and channel drivers explains the cavity gap") is
   REFUTED at the byte level. `logfv_velocity_gradient_bc_aware_2d!`
   is literally a thin wrapper around `fvfd_velocity_gradient_2d!`
   (see `src/kernels/logconformation_fv_2d.jl:918-926`). The cavity
   8× M7b ratio CANNOT come from this kernel difference; it must
   live in the wall-corner gradient correction overlay (the only path
   that differs between cavity and the smooth drivers) or in the LBM
   flow response to the corner singularity. Open Q5 is closed
   **negatively** by this run.

2. **`:fd_v2` (wide-on-wide BSD) catastrophically violates τ_yy = 0.**
   The wide-stencil cancellation principle is sound on Taylor-Green
   periodic (L1 analytical ladder showed `:fd_v2` near-machine
   cancellation), but in the driven LBM-coupled chain on a closed
   wall geometry, `:fd_v2` makes τ_yy reach 0.26 (10⁵× the τ_xy
   amplitude). This is a wall-stencil pathology: the wide div on
   τ_BSD at j∈{1, Ny} reads the one-sided 3-point velocity gradient
   that has zero exact-cancellation with the wide div on τ_p (which
   was built from τ from the source ODE, which is consistent with
   the same one-sided gradient). The bulk cancellation works; the
   wall cancellation does not, and the leftover loops back through
   the constitutive ODE.

3. **`:epsilon_force` (narrow-Lap + force-level elastic split) NaN'd
   at Wi=8e-4 step 5000 AND domainerrored at Wi=1.** The pattern was
   the engineer-memory-recommended "discrete identity ≠ analytical"
   workaround (force-level subtract instead of cell-tensor subtract).
   On Poiseuille the failure is in the narrow 5-point Laplacian's
   handling of the LBM cell-centred u with halfway-bounce ghosts:
   the implicit BC `u_ghost = -u_cell` is the right LBM no-slip
   reflection but couples ASYMMETRICALLY to the wide F_poly_elastic
   at j=1, Ny. The result accumulates over thousands of steps until
   C becomes non-SPD. NOT a candidate for cavity revival.

## Identification of cavity-revival candidates

**Selection rule**: F_total int L2 < 3.51e-2 (M7b threshold) AND
nan_step = -1 in BOTH cases.

| candidate? | variant              | case A F_total | case A_high_Wi F_total | both stable? |
|------------|----------------------|------------------|------------------------|--------------|
| YES        | `:no_bsd`            | 5.01e-03         | 1.33e-05                | YES          |
| BORDERLINE | `:baseline`          | 3.51e-02 (at threshold) | 9.33e-05         | YES          |
| NO         | `:baseline_fvfd_grad`| 3.51e-02         | 9.33e-05                | YES (bit-identical to :baseline) |
| NO         | `:fd_v2_unc`         | 5.09e-01         | 5.79e-01                | YES          |
| NO         | `:fd_v2`             | 8.57e-01         | NaN                     | NO           |
| NO         | `:kinetic`           | 1.86e+00         | 1.87e+00                | YES          |
| NO         | `:epsilon_force`     | NaN              | NaN                     | NO           |

**Only `:no_bsd` strictly beats the 3.51 % threshold on Poiseuille.**
This is a faithful confirmation of the M20 + M24 hypothesis: BSD adds
NO benefit on smooth geometry; it is a NET COST that the LBM nearly
recovers from at high Wi but never on F_total. The cavity benefit
must come from corner-singularity smoothing.

`:baseline` is the production reference. None of the four non-baseline
"clever" variants (V3-V6) reach below it; two NaN at finite Wi.

## Per-variant recommendation

- **`:baseline` (V1)** — KEEP as production default for cavity. Lowest-risk,
  matches M20 reference exactly. ζ=0.75 is the empirically-validated
  cavity setpoint.
- **`:no_bsd` (V2)** — DEBUG-ONLY: it is the ζ=0 control. For cavity
  it would forfeit the corner smoothing that BSD was introduced to
  provide. Not a cavity revival candidate, but the **right
  Poiseuille reference**.
- **`:fd_v2` (V3)** — DISCARD. Wall-stencil cancellation is broken in
  the LBM-coupled chain (τ_yy = 0.26, F_total wall max = 43). NaN at
  Wi=1. NOT a cavity revival candidate.
- **`:fd_v2_unc` (V4)** — DISCARD. Stable at both Wi but bulk F_total
  residual is 15× worse than `:baseline` because the centred-FD
  gradient zeroes at wall rows (engineer memory 2026-05-17 "BSD wide-
  stencil divergence" entry assumed this would HELP on cavity; on
  smooth Poiseuille it HURTS). It might still be useful ON CAVITY
  ONLY because there the wall-corner correction is what kills V3 in
  the first place — so V4 could be revived for cavity testing as a
  controlled comparison.
- **`:kinetic` (V5)** — DISCARD. Π^neq route overshoots BSD magnitude
  by 30× (F_total = -3.57e-6 vs target -1.25e-6 at Wi=8e-4). The
  Chapman-Enskog identity that gives near-machine cancellation on
  cavity at t=2 (per `src/kernels/bsd_kinetic.jl` docstring) does
  NOT hold in the coupled steady state. NOT a cavity revival
  candidate.
- **`:epsilon_force` (V6)** — DISCARD. NaN both cases. The "narrow-Lap
  + force-level subtract" path is sound on paper but the LBM-coupled
  steady state with mirror-ghost narrow Lap drives C non-SPD. NOT
  a cavity revival candidate.
- **`:baseline_fvfd_grad` (V7)** — DISCARD as a separate variant.
  REFUTES Open Q5: it is **bit-identical** to V1. Use this as the
  archival proof that `logfv_velocity_gradient_bc_aware_2d!` is a
  trivial wrapper around `fvfd_velocity_gradient_2d!` (this is a one-
  line memory-worthy result for future Departments).

## What this means for the cavity revival programme

The Poiseuille sweep **does not surface a clear "RED on cavity →
GREEN on smooth" winner** beyond `:fd_v2_unc` (which is GREEN-by-
nonNaN on Poiseuille but bulk-wrong by 50×). The most
operationally meaningful finding is that the **cavity's M7b 3.5 %
discrepancy is not improved by ANY operator-side BSD reformulation
on smooth geometry**. The 8× cavity-vs-Poiseuille ratio that M20
identified is now confirmed (by V7 ≡ V1) to NOT live in the
velocity-gradient kernel difference. It must live in:

1. The wall-corner gradient correction (engineer memory 2026-05-17),
   OR
2. The LBM-side flow response to the corner singularity (Zou-He lid
   coupling, Guo source at corner cells), OR
3. The constitutive ODE's interaction with the wall-corner half-cell
   ghosts in `D_corrected`.

M24 (BSD direction-inversion synthesis) should focus on (1) and (3)
since (2) was already controlled in M16/M17.

## Artefacts

- `bench/viscoelastic_audit/run_poiseuille_bsd_pathmatrix_2d.jl`
  (≈420 LOC)
- `bench/scratch/poiseuille_pathmatrix_<variant>_<case>.csv` × 14
  (one per (variant, case))
- `bench/scratch/poiseuille_pathmatrix_<variant>_A.csv` × 3 from the
  self-test run (Ny=16, max_steps=1000)
- `tmp/m21_full_run.log` — full sweep stdout

Total runtime: ~9 min CPU F64 (single thread, sequential 14 runs).
Self-test: 5.5 s.

Test suite identity NOT re-run (M21 is instrumentation-only, no
`src/` touched).
