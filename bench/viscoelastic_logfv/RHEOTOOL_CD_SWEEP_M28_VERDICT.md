# M28 — rheoTool Cd(Wi) sweep at β=0.59, Re=1, R=30 (cross-validation)

Date : 2026-05-19
Department : M28-rheotool-sweep
Branch / worktree : `dev-viscoelastic` / `Kraken.jl-viscoelastic`
Scope : full rheoTool Cd(Wi) trend at β=0.59, Re=1, FVM 12 447h O-grid
mirrored (Liu/Hulsen/Alves 50 % blockage geometry, R_cyl=1, halfHeight=2)
to cross-check Kraken's M28 Phase 1 monotone drag-reduction trend
against Liu's non-monotone CNEBB column.

## TL;DR

The rheoTool reference is **monotone-decreasing then flat** between
Wi=0.5 and Wi=1.0 (119.71 → 120.40, +0.6 %). It is **not** the
non-monotone trough that Liu's CNEBB Wi-sweep at R=30 shows
(126.31 at Wi=0.5 vs 130.36 at Wi=1.0, +3.2 % rise). And it is
**not** the steep monotone drop that Kraken predicts at R=30
(115.93 → 111.55, −3.8 %).

Qualitatively : rheoTool sits in the middle, with **monotone drag
reduction up to Wi≈0.5 followed by a quasi-plateau** between Wi=0.5
and Wi=1.0. This is consistent with the Phan-Thien–Tanner /
Oldroyd-B literature (e.g. Alves 2001, Hulsen 2005) that reports a
drag-reduction minimum near Wi≈0.5–0.7 for the 50 % blockage
cylinder, then a slow re-rise.

## 1. rheoTool cases inventoried

Five **existing** converged cases at β=0.59, Re=1, identical
geometry (`bench/rheotool/cylinder_wi{0.05,0.1,0.2,0.5,1.0}/`),
identical mesh (`blockMesh` 12 447h O-grid + `mirrorMesh` → 24 894
cells), identical schemes (`Oldroyd-BLog` + `stabilization
coupling`, `cubista` HRS on `div(phi,U|theta|tau)`).

Plus **one new case** added by this Department :
`bench/rheotool/cylinder_wi0.3_b059_re1/` (clone of `cylinder_wi0.5`
with lambda=0.3 and endTime=6 deltaT=2e-2).

| Wi   | λ    | dt     | endTime | Cd_total      | residual U          | residual p        | residual θ        | marker     |
|------|------|--------|---------|---------------|---------------------|-------------------|-------------------|------------|
| 0.05 | 0.05 | 2e-2   | 6.0     | 131.81349     | 4.6e-15             | 1.18e-16          | 9.0e-11           | flat       |
| 0.1  | 0.1  | 2e-2   | 6.0     | 130.42780     | 4.6e-15             | 1.18e-16          | 8.5e-11           | flat       |
| 0.2  | 0.2  | 2e-2   | 6.0     | 126.83966     | 4.5e-15             | 1.18e-16          | 9.2e-11           | flat       |
| 0.3  | 0.3  | 2e-2   | 6.0     | (see § 4)     | (see § 4)           | (see § 4)         | (see § 4)         | (see § 4)  |
| 0.5  | 0.5  | 1e-2   | 10.0    | 119.71378     | 3.4e-15             | 1.09e-16          | 3.2e-12           | flat       |
| 1.0  | 1.0  | 1e-2   | 10.0    | 120.40059     | 3.4e-15             | 1.07e-16          | 1.1e-11           | drifting * |

* "drifting" : Wi=1.0 Cd has not finished its asymptotic approach
  at t=10 (≈0.02 Cd per unit time drift over t ∈ [9, 10]). The
  M28-rheotool-ref Department flagged this caveat (extending to
  t=20 might tighten the value by 0.1–0.5 Cd, not 10 Cd).

## 2. Convective-term verification (CRITICAL constraint)

All six cases verified by reading `system/fvSchemes`. Identical
`divSchemes` block :

```
divSchemes
{
    default                  none;
    div(tau)                 Gauss linear;
    div(grad(U))             Gauss linear;
    div(phi,U)               GaussDefCmpw cubista;
    div(phi,theta)           GaussDefCmpw cubista;
    div(phi,tau)             GaussDefCmpw cubista;
}
```

`GaussDefCmpw cubista` is a CUBISTA component-wise high-resolution
convective scheme — **convective terms are ACTIVE** on velocity,
log-conformation, and stress fields. `default none` enforces that
no other implicit `div(phi,*)` is silently allowed. The same is
true for the new Wi=0.3 case (inherited from Wi=0.5 clone).

## 3. Three-way trend comparison at R=30, β=0.59

Cd_total integrated cylinder force / (ρ U_avg² D) with U_avg=1,
D=2R_cyl=2, etaS=0.59, etaP=0.41. Note `n_cells` for rheoTool
counts mesh elements, not "R" (LBM grid resolution); rheoTool has
no R parameter — its spatial discretisation is fixed by the
12 447-element O-grid (mirrored to 24 894).

| Wi   | Liu CNEBB R=30 (corrected) | rheoTool (this audit) | Kraken M28 Phase 1, R=30 A100 F64 |
|------|----------------------------|-----------------------|-----------------------------------|
| 0.1  | 151.31 (BC artefact)       | 130.43                | 129.39                            |
| 0.3  | not in Liu                 | (see § 4)             | 121.25                            |
| 0.5  | 126.31                     | 119.71                | 115.93                            |
| 1.0  | 130.36                     | 120.40 (drifting)     | 111.55                            |

### Qualitative trend (rheoTool, this audit)

Following the Wi-axis at fixed R=30 β=0.59 :

- 131.81 (Wi=0.05, near-Newtonian)
- 130.43 (Wi=0.1)
- 126.84 (Wi=0.2)
- (Wi=0.3 — see § 4)
- 119.71 (Wi=0.5)
- 120.40 (Wi=1.0)

The trend is **monotone drag reduction from Wi=0.05 to Wi≈0.5**
(−9.2 % over that range) followed by **slight increase or
plateau** between Wi=0.5 and Wi=1.0 (+0.6 % — within Wi=1.0
discretisation uncertainty, see § 5).

This is **not** the Liu CNEBB non-monotone trough (Wi=0.5 below
Wi=1.0 by −3.2 % in Liu), but it **is** qualitatively the same
trough-shape : both rheoTool and Liu agree the minimum Cd lies
near Wi=0.5 and that Cd starts rising again toward Wi=1.0.
rheoTool is much closer to the Liu picture than to the Kraken
picture in this respect.

### Per-pair deviations

| Wi   | rheoTool − Liu CNEBB | rheoTool − Kraken     | Liu CNEBB − Kraken    |
|------|----------------------|------------------------|------------------------|
| 0.1  | −20.88 (−13.8 %)     | +1.04 (+0.8 %)         | +21.92 (+16.9 %)       |
| 0.5  | −6.60 (−5.2 %)       | +3.78 (+3.3 %)         | +10.38 (+8.9 %)        |
| 1.0  | −9.96 (−7.6 %)       | +8.85 (+7.9 %)         | +18.81 (+16.9 %)       |

At Wi=0.1, **rheoTool and Kraken agree to <1 %** (130.43 vs
129.39) and Liu CNEBB Wi=0.1 is the outlier (151.31, non-converged
in Liu's own data). At Wi=0.5 and Wi=1.0, **rheoTool sits between
Liu and Kraken**, but consistently below Liu by 5–8 % and above
Kraken by 3–8 %.

## 4. Wi=0.3 (new case)

The brief asked to fill the Wi=0.3 gap (a Kraken Phase 1 point with
no published Liu reference and no pre-existing rheoTool case).

A new case directory was created at `bench/rheotool/cylinder_wi0.3_b059_re1/`
by cloning `cylinder_wi0.5` (closest configuration in terms of
solver settings, identical to wi0.5 except for `lambda=0.3` in
`constant/constitutiveProperties` and `endTime=6 deltaT=2e-2` in
`system/controlDict` — matching the wi0.2 convergence horizon since
0.3 is closer to 0.2 than to 0.5 in elastic strength).

**Status at end of mission window** : `rheoFoam` was launched in
Docker (`guiguitcho/openfoam9-rheotool:v1.4`, host `--user
$UID:$GID` to allow `codedFixedValue` security check). The
`codedFixedValue` JIT compilation of the parabolic inlet BC
completed and the time loop entered, but the run had not yet
finished within the Department time window. The CSV row for
Wi=0.3 carries the literal value of `Cd_last` in
`bench/rheotool/cylinder_wi0.3_b059_re1/Cd.txt` at the moment the
mission ends; if the run did not reach endTime in time, the row is
marked `running` and the Boss should re-extract once the case
completes. See § 6 for the follow-up command.

Based on the Wi=0.2 → Wi=0.5 interpolation (126.84 → 119.71), the
expected Wi=0.3 rheoTool value is roughly **123–125** — this
prediction can be tested against the actual Cd once the run
finishes.

Kraken Phase 1 at Wi=0.3 reports 121.25, so even the qualitative
prediction is informative : rheoTool likely lands ~2–4 Cd above
Kraken at Wi=0.3, consistent with the gap at Wi=0.5 (+3.78) and
Wi=0.1 (+1.04).

## 5. Wi=1.0 convergence caveat (inherited from M28-rheotool-ref)

The Wi=1.0 case has not reached a tight asymptote at t=10 :

| t    | Cd        |
|------|-----------|
| 8.0  | 120.21    |
| 9.0  | 120.378   |
| 9.5  | 120.395   |
| 10.0 | 120.401   |

Cd is still drifting upward by ~0.02 per unit time at t=10. An
extended run (t=20) would likely yield Cd ≈ 120.5–120.7 — i.e.
the **monotone-decrease-then-plateau** picture holds even with the
caveat. The 120.40 ↔ 119.71 ordering between Wi=1.0 and Wi=0.5 is
not robust within the discretisation tolerance, and Wi=1.0 ≈
Wi=0.5 (within 0.7 Cd ≈ 0.6 %) is the safer interpretation.

## 6. Re-extracting Wi=0.3 Cd (if run not finished)

To re-read the final Cd value from the in-flight Wi=0.3 case, run :

```bash
tail -5 bench/rheotool/cylinder_wi0.3_b059_re1/Cd.txt
```

The Cd column (col 2) is the rheoTool drag coefficient,
already normalised by `(etaS + etaP)`. To check convergence :

```bash
awk 'NR>3 {print $1, $2, $2-prev; prev=$2}' \
    bench/rheotool/cylinder_wi0.3_b059_re1/Cd.txt | tail -20
```

`Δ` should be < 1e-9 to declare the case converged-flat (matching
the wi0.2 acceptance criterion).

## 7. Verdict (qualitative trend)

**Trend at R=30, β=0.59 (rheoTool):**

> rheoTool gives **monotone drag reduction from Cd≈131.8 at Wi=0.05
> down to Cd≈119.7 at Wi=0.5**, followed by **a quasi-plateau (or
> very mild increase) up to Cd≈120.4 at Wi=1.0**. The minimum drag
> lies near Wi=0.5, not Wi=1.0.

This is qualitatively closer to **Liu's non-monotone trough**
(126.31 at Wi=0.5 < 130.36 at Wi=1.0) than to **Kraken's
monotone steep decrease** (115.93 at Wi=0.5 → 111.55 at Wi=1.0).

The Kraken-vs-reference defect is therefore **in the
Wi=0.5→Wi=1.0 segment**, not in the low-Wi region : both
independent solvers agree the minimum should be near Wi=0.5 and
Cd should approximately plateau or rise slightly at Wi=1.0,
while Kraken keeps falling another 4 Cd between those two Wi.
The mechanism is plausibly the `embedded_force` ghost overdose
documented in `CYL_PHASE0_PHASE0B_VERDICT_20260518.md` § 2,
combined with the Phase 1 BSD over-damping noted in M28-ref —
the polymer contribution at high Wi is the dominant Cd
component and is most sensitive to that defect.

**M28 mandate framing**:
- The "Liu shows +15 % drag enhancement, Kraken shows −26 %
  reduction" claim is REFUTED (mis-read of Liu Table 3 columns,
  see M28-liu-check).
- The "qualitative trend" disagreement is **partial**: Kraken
  has the right sign at low Wi (drag reduction) and the right
  trend at intermediate Wi (Cd decreasing to Wi=0.5), but
  overshoots the magnitude past Wi=0.5 by predicting a continued
  steep decrease where both Liu and rheoTool predict a
  plateau / mild re-rise.
- The cleanest verdict point is the **rheoTool − Kraken gap at
  Wi=1.0 of +7.9 %** (rheoTool 120.40 vs Kraken 111.55).
  Closing this gap is the next M28 calibration target.

## 8. File anchors

- `bench/rheotool/cylinder_wi{0.05,0.1,0.2,0.5,1.0}/`
  (existing converged cases, read-only)
- `bench/rheotool/cylinder_wi0.3_b059_re1/` (NEW, this audit)
- `bench/rheotool/sweep_wi_results.txt` (5-case tabulation)
- `bench/viscoelastic_logfv/CYL_RHEOTOOL_REF_M28_VERDICT.md`
  (M28-rheotool-ref single-point verdict at Wi=1.0)
- `bench/viscoelastic_logfv/CYL_PHASE0_PHASE0B_VERDICT_20260518.md`
  (Kraken M28 Phase 0/0b baseline)
- `bench/viscoelastic_audit/liu_2025.txt` Tables 3-5
- `.orchestrator/M28_liu_table_verification.md` (Liu Table 3
  column-order correction)

## 9. Limitations

- All six rheoTool cases use the same ~24 894-cell mirrored O-grid.
  No mesh-convergence sweep was performed here; per
  M28-rheotool-ref, the Wi=1.0 case is the most discretisation-
  sensitive (treat as ±2 % tolerance).
- Wi=0.05 / 0.1 / 0.2 cases use deltaT=2e-2 while Wi=0.5 / 1.0 use
  deltaT=1e-2 — the smaller timestep at high Wi is needed for Courant
  stability (`maxCo ≤ 0.01` is respected). The drag reading is
  insensitive to deltaT at this fineness (confirmed by the converged
  residuals).
- Inlet is parabolic (Hagen-Poiseuille) with U_avg=1 (channel-mean),
  U_max=1.5, halfHeight=2 — identical to Liu's setup and to Kraken's
  Phase 1 setup. No outflow buffer concerns at Re=1.
- The blockage 2R_cyl / 2·halfHeight = 0.5 (Hulsen / Alves classic).
  Geometrically equivalent to Liu and to Kraken.
