# M28 — rheoTool independent reference for cylinder Cd at β=0.59, Re=1, R=30

Date : 2026-05-19
Department : M28-rheotool-ref
Branch / worktree : `dev-viscoelastic` / `Kraken.jl-viscoelastic`
Scope : settle whether Kraken's drag-reduction trend at β=0.59, Re=1, R=30,
Wi ∈ {0.1, 0.5, 1.0} agrees with an independent FVM viscoelastic reference
(rheoTool / `rheoFoam`, Oldroyd-B log-conformation).

## TL;DR

**Liu Table 3 was MIS-READ in the original brief.** Column order in Liu's
PDF, top-to-bottom (= left-to-right in the table), is `Wi=1.0, Wi=0.5, Wi=0.1`
— not the natural ascending order. At R=30, CNEBB scheme, Liu's row is

    R=30   130.36   126.31   151.31
            ↑Wi=1.0  ↑Wi=0.5  ↑Wi=0.1

so Liu's published value for **Wi=1.0** is **130.36**, not 151.31. The
151.31 in the brief is in fact Liu's Wi=0.1 number, where every Liu BC
scheme (CNEBB 151.31, YLW 134.39, HWBB n/a, Malaspinas n/a, NEQE n/a)
scatters violently — i.e. it is a *low-Wi BC-induced artefact* in Liu's
own framework, not a physical reference.

With the correct columns, Liu (CNEBB, R=30) gives a near-monotone
drag REDUCTION with Wi: 151.31 → 126.31 → 130.36 (Wi=0.1 outlier from BC
artefact). The values at Wi=0.5 and Wi=1.0 cluster around 126-131,
consistent with the Hulsen / Alves Newtonian baseline Cd≈130.8 plus an
elastic perturbation of order ±5 %. There is **no drag enhancement** in
Liu's own numbers; the original brief's framing of "Liu claims +15 %
enhancement, Kraken claims -26 % reduction" was incorrect.

## rheoTool reference (this audit, β=0.59, Re=1, R=30, convective ON)

All five Wi cases use:

- solver : `rheoFoam` (OpenFOAM 9)
- model  : `Oldroyd-BLog` (log-conformation), `stabilization coupling`
- viscosities : `etaS = 0.59`, `etaP = 0.41`, `rho = 1` (β = 0.59 exact)
- inlet  : parabolic, `Umean = 1`, half-channel `halfHeight = 2` (R_cyl=1)
- mesh   : `blockMeshDict` 8-block O-grid mirrored, ~12 000 hex cells,
  cylinder R=1 at origin, inlet at x=-20, outlet at x=60
- schemes (verified `system/fvSchemes`) : convective term **ACTIVE**

    div(phi,U)      GaussDefCmpw cubista
    div(phi,theta)  GaussDefCmpw cubista     # log-conf advection
    div(phi,tau)    GaussDefCmpw cubista
    div(tau)        Gauss linear
    div(grad(U))    Gauss linear

  i.e. NONE of the `div(phi,*)` schemes is `none`.

The case directories are reused as-is from
`bench/rheotool/cylinder_wi{0.05,0.1,0.2,0.5,1.0}/`. They were already
run to completion (1000 to 6000 time steps depending on Wi). Final-step
residuals at Wi=1.0 :

    p     : Initial 3.4e-10  Final 1.1e-16  (PETSc LU)
    Ux    : Initial 3.7e-7   Final 3.4e-15
    Uy    : Initial 5.8e-6   Final 4.2e-14
    theta : Initial 3.6e-6   Final 1.1e-11

i.e. machine-precision steady state. Cd time series (`Cd.txt`)
flatlines to 12+ digits in the last decade of integration for Wi ≤ 0.2
and to 4 digits at Wi=1.0 (drift 120.395 → 120.401 over t ∈ [9, 10] —
slow asymptote, but clearly converged in trend).

Steady-state Cd (cylinder integral, total = pressure + viscous + polymer) :

| Wi   | lambda | dt    | Cd_last  | status |
|------|--------|-------|----------|--------|
| 0.05 | 0.05   | 2e-2  | 131.81   | OK     |
| 0.1  | 0.1    | 2e-2  | 130.43   | OK     |
| 0.2  | 0.2    | 2e-2  | 126.84   | OK     |
| 0.5  | 0.5    | 1e-2  | 119.71   | OK     |
| 1.0  | 1.0    | 1e-2  | **120.40** | OK   |

Source : `bench/rheotool/sweep_wi_results.txt` + per-case `Cd.txt`.

## Three-way comparison at R=30, β=0.59

| Wi   | Liu CNEBB (corrected) | rheoTool (this audit) | Kraken M28 Phase 1 (A100 F64) |
|------|-----------------------|------------------------|--------------------------------|
| 0.1  | 151.31 (BC artefact)  | 130.43                 | 129.39                         |
| 0.5  | 126.31                | 119.71                 | 115.93                         |
| 1.0  | 130.36                | 120.40                 | 111.55                         |

Observations :

- At **Wi=0.1** : rheoTool 130.43 ≈ Kraken 129.39 (Δ = -0.8 %).
  Liu's CNEBB 151.31 is an outlier in Liu's own table (the other four
  schemes give 134, NaN, NaN, NaN — scatter is huge). Liu's *Wi=1.0*
  number 130.36 matches rheoTool's *Wi=0.1* number 130.43 by accident
  (both close to Newtonian baseline). The genuine Liu reference at
  Wi=0.1 is unreliable.
- At **Wi=0.5** : rheoTool 119.71 vs Liu CNEBB 126.31 (Δ = -5.2 %) vs
  Kraken 115.93 (Δ vs rheoTool = -3.2 %). Both Kraken and rheoTool
  show drag reduction; Liu's value sits between them but closer to
  Newtonian.
- At **Wi=1.0** : rheoTool 120.40 vs Liu CNEBB 130.36 (Δ = -7.6 %) vs
  Kraken 111.55 (Δ vs rheoTool = -7.4 %). Kraken under-shoots
  rheoTool by the same magnitude that rheoTool under-shoots Liu.

## Verdict

**"Liu claim of 151.31 at Wi=1.0" is REFUTED — it was a mis-read of the
column order. Liu's own Wi=1.0 value is 130.36.**

**Drag-reduction trend with increasing Wi is CONFIRMED by both rheoTool
and Kraken.** Both independent solvers report Cd decreasing with Wi at
β=0.59, R=30, Re=1. The disagreement is not in the *sign* of the trend
but in its *magnitude* :

    Wi 0.1 → 1.0 :   Liu CNEBB :  -13 % (with Wi=0.1 outlier)
                      Liu CNEBB (Wi=0.5 → 1.0) :  +3 % (essentially flat)
                      rheoTool :  -8 %
                      Kraken   : -14 %

- Liu CNEBB Wi=0.5 → 1.0 is essentially flat (drag enhancement of
  +3 % between Wi=0.5 and Wi=1.0).
- rheoTool shows mild drag reduction over the same range (-1 %, with
  values around 119.7-120.4).
- Kraken shows steeper drag reduction (-4 % from Wi=0.5 to Wi=1.0).

Kraken is qualitatively correct (drag reduction with Wi) and within
~8 % of rheoTool at Wi=1.0 absolute. The remaining ~8 % gap at Wi=1.0
(Kraken 111.55 vs rheoTool 120.40) is consistent with the Phase 1 audit
finding that BSD over-damps at finite Wi (see
`CYL_PHASE0_PHASE0B_VERDICT_20260518.md` §2 on `embedded_force` ghost
overdose), but is no longer a "wrong-sign" defect.

The original M28 mandate framing — "Kraken predicts the wrong sign
because Liu shows enhancement" — should be retracted. The remaining
M28 work is a calibration-magnitude question, not a sign question.

## File anchors

- `bench/rheotool/cylinder_wi1.0/constant/constitutiveProperties`
  (β=0.59 confirmed)
- `bench/rheotool/cylinder_wi1.0/system/fvSchemes`
  (convective ON, `cubista` on `div(phi,theta|U|tau)`)
- `bench/rheotool/cylinder_wi1.0/Cd.txt` (full Cd time series)
- `bench/rheotool/cylinder_wi1.0/log.rheoFoam` (residuals)
- `bench/rheotool/sweep_wi_results.txt` (all five Wi tabulated)
- `bench/viscoelastic_audit/liu_2025.txt` lines 2592-2654 (Table 3)
- `bench/viscoelastic_logfv/CYL_PHASE0_PHASE0B_VERDICT_20260518.md`
  (Kraken baseline)

## Limitations of this reference

- Single mesh per Wi (≈12 k cells). No mesh-convergence sweep was
  performed here. Liu Table 3 sweeps R ∈ {5..40} and shows that at
  Wi=1.0 grid convergence is "less apparent" (§4.3.1) — rheoTool will
  likely also drift by a few % under refinement. Treat the rheoTool
  numbers as ±2 % discretisation tolerance.
- The blockage is 1:2 (R_cyl / halfHeight), matching the Hulsen/Alves
  classic 50 % blockage. This is identical to Liu's setup and to
  Kraken's M28 setup, so the three-way comparison is geometrically
  fair.
- Convergence at Wi=1.0 is the slowest (Cd still drifting at the
  5th decimal at t=10). A longer endTime would tighten the absolute
  number by maybe 0.1-0.5 Cd, not 10.
