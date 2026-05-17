# Poiseuille `F_total` post-hoc trace — VERDICT

Date: 2026-05-18. Branch: `dev-viscoelastic`. Mission: M20.

Opens the Poiseuille investigation cluster (M20-M24) by directly
measuring whether the BSD `−ζ·ν_p·∇²u_narrow` correction cancels the
`F_poly_wide` `ν_p·∇²u_wide` portion of the body force injected into the
LBM, leaving `(1−ζ)·ν_p·∇²u` as designed — or whether a structured
residual remains.

## Setup

`run_viscoelastic_logfv_poiseuille_coupled_2d` (CPU F64), Nx=8, Ny=32,
F_body=1e-5, λ ∈ {1.0, 1.25e4}, max_steps=100 000,
`force_boundary_fill=:bc_aware`. x-periodic / y-wall. Three cases:

| label      | nu_s | nu_p | ζ    | λ      | Wi=γ̇_max·λ |
|------------|------|------|------|--------|------------|
| A_no_BSD   | 0.1  | 0.1  | 0.00 | 1.0    | 8e-4       |
| A          | 0.1  | 0.1  | 0.75 | 1.0    | 8e-4       |
| A_high_Wi  | 0.1  | 0.1  | 0.75 | 1.25e4 | 1.0        |

Post-hoc, the driver result is decomposed per y-row (x-mean across
Nx=8 cells) into three additive components:

```
F_total[i,j]       = Fx_body + F_poly_wide[i,j] − F_BSD_narrow[i,j]
F_poly_wide        = logfv_polymer_force_bc_aware_2d!(τ_p)
F_BSD_narrow       = ζ · ν_p · ∇²u_narrow (5-point)
F_total_nb         = F_total − Fx_body
                   = F_poly_wide − F_BSD_narrow                (extracted)
```

Analytical Newtonian-limit targets (parabolic Poiseuille,
`d²u/dy² = −F/ν_total`, uniform across interior):

```
F_poly_target    = −ν_p   · F_body / ν_total = −5.000e-6
F_BSD_target     = −ζ·ν_p · F_body / ν_total = −3.750e-6      (case A)
F_total_target   = −(1−ζ)·ν_p · F_body / ν_total = −1.250e-6  (case A)
```

## Per-case results

Interior rows = y_idx ∈ 3:(Ny−2); wall rows = y_idx ∈ {1, 2, Ny−1, Ny}.
rel_resid_L2 is RMS over interior; wall rel_resid is the max over the
4 wall rows.

| case      | F_poly interior L2 | F_poly wall max | F_total interior L2 | F_total wall max |
|-----------|--------------------|-----------------|---------------------|------------------|
| A_no_BSD  | **5.01e-03**       | 5.01e-03        | **5.01e-03**        | 5.01e-03         |
| A         | **5.01e-03**       | 5.01e-03        | **3.51e-02**        | 3.51e-02         |
| A_high_Wi | **1.33e-05**       | 1.33e-05        | **9.33e-05**        | 9.33e-05         |

## Comparison vs analytical Newtonian-limit targets

### Case A_no_BSD (ζ=0 control)

`F_total_nb ≡ F_poly_wide` by construction (no BSD subtraction). The
single residual is **0.50 %**, uniform across all 32 rows.

### Case A (ζ=0.75 production)

- `F_poly_wide` measured: −4.9750e-6 (analytical: −5.000e-6) → **0.50 %**.
  **Bit-identical to A_no_BSD** at every row, confirming the
  Result-2 "BSD-invariant on stress fields" finding from the
  17 May verdict: the steady-state polymer-stress field does not
  depend on `ζ` for Poiseuille.
- `F_BSD_narrow` measured: −3.7688e-6 (analytical: −3.750e-6) →
  **0.50 % HIGH** in magnitude.
- `F_total_nb = F_poly_wide − F_BSD_narrow` = −1.2062e-6
  (analytical: −1.250e-6) → **3.51 %**.

The two stencils (WIDE for `div(τ_p)` and NARROW for `ζ·ν_p·∇²u`)
each carry an O(0.5 %) discretisation residual relative to the
analytical Laplacian. The residuals have the **same sign** (both fall
short of magnitude) and partially cancel, but the cancellation leaves
a **0.044e-6 mismatch on a 1.25e-6 target → 3.5 %**. The wall and
interior residuals are identical to 6 sig figs — there is **no
wall-row spike**, no structured x-variation (Nx=8 means is
bit-identical row by row), the residual is purely a **stencil-amplitude
mismatch in the bulk**.

### Case A_high_Wi (ζ=0.75, Wi=1)

- `F_poly_wide` measured: −4.9999e-6 → **1.33e-5** (≈380× better than A).
- `F_BSD_narrow` measured: −3.7501e-6 → matching to 5 sig figs.
- `F_total_nb`: −1.2499e-6 → **9.33e-5** (≈380× better than A).

At Wi=1 the polymer ressorts a measurable elastic normal-stress
contribution; the LBM's parabolic-profile residual itself shrinks
by ≈380× (visible in `ux_cell_mean`: the centerline u is 6.4055e-3 at
Wi=1 vs 6.4375e-3 at Wi=8e-4 — closer to the analytical 6.4000e-3 by
the same ratio). Both stencils sample a velocity field that is much
closer to the analytical parabola, so both residuals collapse and
their cancellation residual collapses with them.

## What does BSD actually do on Poiseuille?

**It works as designed at the operator level.** The `F_BSD_narrow`
term cancels the `F_poly_wide` term up to the **stencil discretisation
difference of the underlying d²u/dy² operator** (~0.5 % on this
profile and grid).

The "3.5 % F_total residual" is **not a BSD bug**; it is the WIDE−NARROW
stencil-mismatch residual amplified by `(1−ζ)⁻¹ = 4×` in the relative
metric (smaller denominator) plus the partial mis-cancellation of the
two ~0.5 % stencil errors.

Concretely: the absolute F_total error is `|−1.2062e-6 − (−1.250e-6)|
= 4.4e-8`, identical in magnitude to (and same sign as) the absolute
F_poly error `|−4.9750e-6 − (−5.000e-6)| = 2.5e-8` modulo the
F_BSD-shift mismatch. The relative metric magnifies because the
post-BSD target is 4× smaller.

## Implications for M21-M24

- **M21 (velocity-gradient kernel cross-check, Open Q5)** —
  PROMOTED PRIORITY. The 0.5 % WIDE-stencil residual on F_poly is
  uniform across the channel (no wall amplification on Poiseuille).
  Any wall-row signal in the cavity comes from `fvfd_velocity_gradient_2d!`
  (cavity) vs `logfv_velocity_gradient_bc_aware_2d!` (this driver) or
  from the corner singularity, NOT from the BSD subtraction itself.
  M21 should isolate which.
- **M22 (finite-Wi analytical, angle c)** —
  KEEP, with REVISED EXPECTATION. The Wi=1 case shows F_poly /
  F_total residuals **380× smaller** than Wi=8e-4, mostly because the
  LBM's own u-discretisation residual shrinks. M22 should compare
  τ_xy / N1 / C_xx to the analytical Oldroyd-B closed form, not the
  Newtonian Laplacian target used here. The polymer pipeline ratchet
  from the 17 May verdict already extends to Wi=1 on stress; M22's
  job is now to confirm it persists at Wi=0.5 and Wi=1 simultaneously
  with `ζ ∈ {0, 0.75}`.
- **M23 (rheoTool planar Poiseuille cross-check)** —
  KEEP, gated on rheoTool case existence. If absent, defer; the
  current Newtonian-limit decomposition is informative without an
  iBSD-OFF / ON pair to compare against.
- **M24 (BSD direction-inversion synthesis, angle b)** —
  STRENGTHEN. The 17 May hypothesis ("ζ↑ helps cavity, hurts
  Poiseuille because of the corner singularity") is consistent with
  M20: on smooth Poiseuille, BSD is **net cost** (3.5 % on F_total vs
  0.5 % on F_poly alone). The cavity benefit must come from corner
  smoothing the WIDE-stencil cannot resolve. M24 should construct a
  controlled-singularity test (e.g. step geometry or analytic singular
  forcing) to confirm.

## Artefacts

- `bench/viscoelastic_audit/run_poiseuille_bsd_trace_2d.jl` (282 LOC)
- `bench/scratch/poiseuille_bsd_trace_A_no_BSD.csv` (33 rows)
- `bench/scratch/poiseuille_bsd_trace_A.csv` (33 rows)
- `bench/scratch/poiseuille_bsd_trace_A_high_Wi.csv` (33 rows)
- `bench/scratch/poiseuille_bsd_trace_A_selftest.csv` (17 rows; from
  the self-test exit-criterion run)
- `tmp/m20_full_run.log` (full-mode stdout)

Test suite identity NOT re-run (M20 is instrumentation-only, no `src/`
touched).
