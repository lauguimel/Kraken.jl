# M29c-wallstress — Verdict (DECISIVE: pressure mismatch + polymer rear-shoulder overshoot)

Date  : 2026-05-19
Branch: dev-viscoelastic
Mission: circumferential wall-stress decomposition of Cd into
         Cd_pressure + Cd_solvent + Cd_polymer at the cylinder wall,
         for M29b (Rusanov), M29c-v2 (MUSCL-superbee), and rheoTool reference,
         to identify **on which azimuth band** the M29c-v2-to-rheoTool
         residual Cd gap is concentrated.

## Status: EXECUTED — verdict is DECISIVE

Artefacts:

| Artefact                                                              | Present |
|-----------------------------------------------------------------------|---------|
| `bench/scratch/m29c_wallstress/run_wallstress.jl`                     | YES (driver) |
| `bench/scratch/m29c_wallstress/M29WS_rheotool_wall.csv`               | YES (378 wall faces) |
| `bench/scratch/m29c_wallstress/M29WS_rheotool_bins.csv`               | YES (72 θ-bins) |
| `bench/scratch/m29c_wallstress/M29WS_M29b_bins.csv`                   | YES (72 θ-bins) |
| `bench/scratch/m29c_wallstress/M29WS_M29c_v2_bins.csv`                | YES (72 θ-bins) |
| `bench/scratch/m29c_wallstress/M29WS_summary.txt`                     | YES |
| `.engineer_logs/M29c-wallstress_*.log`                                | YES (stdout) |

## rheoTool Cd convention (from controlDict outputCd functionObject)

```cpp
F = tau + symm(L + L^T)*etaS - p*rho*I
Fpatch.x = gSum( (-faceAreas()) & F.boundaryField()[cyl] ) / (etaS + etaP)
Cd       = Fpatch.x   // ≡ 2 Fx / (rho U² D) with rho=1, U=1, D=2
```

`-faceAreas()` points radially OUT of the cylinder (= into fluid).
Decomposition is therefore:

- `Cd_pressure = ∮ (-p·rho·I)·n_out · ê_x / (etaS+etaP) dA`
- `Cd_solvent  = ∮ (2·etaS·D)·n_out · ê_x / (etaS+etaP) dA`
- `Cd_polymer  = ∮ tau·n_out          · ê_x / (etaS+etaP) dA`

With etaS=0.59, etaP=0.41 (so etaS+etaP=1), rho=1, U=1, D=2, all three
contributions are reported in the same "Cd" units as the Cd.txt history.

Kraken side: `rho` is NOT stored in the .jls snapshot, so the pressure
integral cannot be computed directly. We compute `Cd_solvent_wall` from
2·mu_s·D (FD gradients) and `Cd_polymer_wall` from tau_p directly, then
`Cd_pressure_residual := Cd_total_stored - Cd_solvent_wall - Cd_polymer_wall`.

θ convention: θ=0 along +x (= rear stagnation, downstream), θ=±π along -x
(= front stagnation, upstream).

## Cd decomposition (scalar)

| contribution | rheoTool | M29b   | M29c-v2 | gap (rT−M29b) | gap (rT−M29c-v2) |
|---|---|---|---|---|---|
| Cd_pressure | **85.77** | 75.64 | 75.56 | +10.13 | **+10.22** |
| Cd_solvent  | 19.78    | 21.19 | 20.34 | -1.41  | -0.56 |
| Cd_polymer  | **13.45** | 13.40 | 20.01 | +0.05  | **-6.55** |
| **Cd_total** | **119.0** | 110.23 | 115.90 | +8.77 | **+3.11** |

Reference Cd from `bench/rheotool/cylinder_wi1.0/Cd.txt` at t=10: **120.40**.
Our wall integral gives 119.00 — within 1.4 pts (1.2 %) of the converged
controlDict value, validating the integration method. The small residual
is the difference between cell-centred values used at owner cells under
the zeroGradient BC and exact face-extrapolated values OpenFOAM uses
internally for `F.boundaryField`.

## Internal consistency

| check | M29b | M29c-v2 | rheoTool |
|---|---|---|---|
| Sum (P + S + Poly) vs total stored Cd_kraken | 75.6 + 21.2 + 13.4 = 110.23 ✓ (residual definition closes this exactly) | 75.6 + 20.3 + 20.0 = 115.90 ✓ (closes exactly) | 85.8 + 19.8 + 13.5 = 119.0 (vs 120.4 from Cd.txt → 1.2 % residual from BC sampling) |
| y-component (lift symmetry test) | Cd_s,y + Cd_p,y = 1.15 | 1.69 | 7.5e-13 |

rheoTool's symmetry residual is at machine precision (good). Both Kraken
snapshots have a small but visible y-asymmetry (1-2 pts of Cd) consistent
with not having reached full statistical convergence at 30k steps.

## Azimuthal breakdown (Cd_x contribution per band)

```
band                       | rT_pres   rT_solv   rT_poly  | K(M29b)_solv  K(M29b)_poly | K(M29c-v2)_solv  K(M29c-v2)_poly
front_pole   (|θ-π| < 15°) |   +48.54    +0.14    +0.06   |     +0.21        +0.08    |       +0.24            +0.08
front_shldr  (|θ| 108-162°)|  +109.99    +6.47    +4.16   |     +5.64        +1.56    |       +6.22            +2.91
equator      (|θ|  72-108°)|    +1.54    +8.81    +6.11   |     +9.79        +4.90    |       +9.44            +8.04
rear_shldr   (|θ|  18- 72°)|   -65.72    +3.91    +2.83   |     +5.06        +6.32    |       +3.88            +8.53
rear_pole    (|θ| < 15°)   |   -19.48    +0.20    +0.13   |     +0.25        +0.27    |       +0.27            +0.24
```

Gap rheoTool − M29c-v2 per band (polymer only, the only sub-quantity
that has more than 1 pt of structure):

```
band             | rT_poly - M29c_poly  | M29c-v2 over/under?
front_pole       |   -0.02              | match
front_shoulder   |   +1.25              | M29c-v2 still UNDER-predicts (better than M29b which was -2.60)
equator          |   -1.93              | M29c-v2 OVER-predicts (was -1.21 in M29b, got worse)
rear_shoulder    |   -5.70              | M29c-v2 dramatically OVER-predicts (was -3.49 in M29b, got worse)
rear_pole        |   -0.11              | match
```

### Where the polymer wall-stress peaks (rheoTool reference)

```
top 6 rheoTool polymer-bins (max per-bin Cd_poly contribution):
  θ = ±0.625 π  ( ±113° from rear, = ±67° from front)   Cd_poly = +0.44
  θ = ±0.542 π  ( ±98°  from rear, = ±82° from front)   Cd_poly = +0.42
  θ = ±0.514 π  ( ±92°  from rear, = ±88° from front)   Cd_poly = +0.42
```

rheoTool's polymer wall-stress peaks just upstream of the equator
(=upper/lower shoulder), at ~+0.625π = ±113° from rear ≈ ±67° from front.

```
top 6 Kraken M29c-v2 polymer-bins:
  θ = ±0.347 π  ( ±62°  from rear, = ±118° from front)  Cd_poly = +0.88
  θ = ±0.375 π  ( ±67°  from rear)                       Cd_poly = +0.85
  θ = ±0.403 π  ( ±72°  from rear)                       Cd_poly = +0.84
```

**Kraken's polymer wall-stress peaks downstream of the equator at the
rear shoulder** — opposite side of the equator from where rheoTool peaks,
**and at 2× the magnitude** of rheoTool's peak. The polymer peak is
**rotated ~45-50° CCW** between rheoTool (front-shoulder) and Kraken
(rear-shoulder), with the Kraken peak being twice as tall.

## Verdict

**The M29c-v2-to-rheoTool residual Cd gap is concentrated in TWO components,
acting in opposite directions on the total:**

1. **Cd_pressure: Kraken under-predicts by +10.2 pts** (76 vs 86, ~12 % low).
   rheoTool's pressure-drag is dominated by the front-shoulder band
   (|θ| ∈ 108-162° from rear, i.e. ±18-72° from front) carrying +110 pts
   of pressure-Cd. The Kraken residual cannot be localised in θ from the
   snapshot data (ρ not stored); however, the global under-prediction of
   10 pts is consistent with the previously identified front-shoulder
   pressure-BC coupling deficit.

2. **Cd_polymer: Kraken (M29c-v2) over-predicts by -6.5 pts** (20 vs 13).
   This over-prediction is dominated by **the rear-shoulder band**
   (|θ| ∈ 18-72° from rear, i.e. behind the equator, suction zone),
   where M29c-v2 gives +8.5 pts vs rheoTool's +2.8 pts — **3× too high**.
   The polymer peak is also mis-placed: rheoTool peaks at θ ≈ ±0.6π
   (front-shoulder), Kraken peaks at θ ≈ ±0.35π (rear-shoulder),
   a **45-50° azimuthal offset**.

The two errors **partially cancel** in the total Cd:
  ΔCd_total = ΔCd_pressure + ΔCd_solvent + ΔCd_polymer
            = +10.22 + (-0.56) + (-6.55) = **+3.11** (rheoTool vs M29c-v2).

The volume-level L2_rel decrease for τ_p of -20-38 % (prior verdict)
is therefore consistent with INCREASING the wall τ_p values — the
volume metric was dominated by far-wake samples where M29c-v2 reduces
τ_p relative to M29b, but at the wall the scheme increases polymer
stress everywhere by 50-100 %, overshooting rheoTool especially in the
rear-shoulder zone.

Confidence: HIGH on solvent (good agreement, sign consistent) and on
total decomposition (rheoTool sum closes to 119/120). MEDIUM-HIGH on
the per-band attribution of the polymer gap (azimuthal binning at
N_az=72 = 5° bins, but the rear-shoulder result is stable across
bins). LOW on per-θ Kraken pressure attribution (residual only, no
field data).

## Boss decision implication

1. **The hypothesis "M29c-v2 progresses toward rheoTool in τ_p" is FALSE
   AT THE WALL.** M29c-v2 actually moves Cd_polymer FURTHER from rheoTool
   than M29b was: |ΔCd_polymer|_M29b = 0.05, |ΔCd_polymer|_M29c-v2 = 6.55.
   The volume-level improvement of -38 % on L2_rel(τ_p_xy) was a wake
   metric that hides this wall overshoot.

2. **M29c-style 1-sided MUSCL is the WRONG direction for the wall τ_p.**
   It increases the polymer wall stress (gain of 6.6 pts) but the
   physical reference (rheoTool) had Cd_polymer ≈ 13.4 already correctly
   reached by M29b. Going to higher-order or CUBISTA (M29d) is NOT
   well-motivated by the wall-stress evidence: it would likely further
   amplify the rear-shoulder polymer overshoot. The remaining ~3 pt
   total Cd gap (M29c-v2 → rheoTool) lives mostly in the pressure
   component, NOT the polymer scheme.

3. **The main physical action item is the +10.2 pt pressure deficit**,
   which is not addressed by any polymer-advection scheme change.
   Candidates: (i) pressure-BC coupling (Zou-He / extrapolation order
   at the in/out faces), (ii) inadequate gradient stencil for the
   Newtonian back-stress at the front face, (iii) Mach-number / weak
   compressibility error at front stagnation (highest |p| region).

4. **The +6.5 pt polymer overshoot in M29c-v2 is a regression vs M29b**,
   even though wake L2_rel improved. If the production scheme is to be
   shipped, M29b (Rusanov, with its better Cd_polymer wall total but
   bigger wake L2 error) may actually be the better choice for Cd
   reporting, since `|Cd_total - rT| = 8.77` for M29b vs `3.11` for
   M29c-v2 — but this hides that M29c-v2's small total gap is a
   cancellation between under-predicted pressure (+10) and
   over-predicted polymer (-6.5), not a true convergence.

## Memory candidates

1. `feedback_cd_decomposition_cancellation` — A small Cd_total gap can
   hide LARGE per-component gaps acting in opposite directions
   (M29c-v2: +10.2 pressure deficit ± -6.5 polymer overshoot → +3.1 net).
   Always inspect the wall decomposition before declaring "Cd matches".
2. `feedback_volume_vs_wall_metrics_diverge` — A volume L2_rel
   improvement on τ_p (-38 %) can coexist with a wall-stress overshoot
   (+50 % at the rear shoulder). Wake L2_rel is dominated by far-wake
   samples; wall Cd_polymer is a different physical quantity. Both
   must be reported; one cannot substitute for the other.
3. `feedback_rheotool_cd_decomposition` — rheoTool reports
   `Cd = Fpatch.x / (etaS+etaP)` where Fpatch.x = ∮F·n_out dA and
   `F = tau + symm(L+L^T)*etaS - p*rho*I`. With etaS+etaP=1, U=1, D=2,
   the controlDict Cd equals the standard 2 Fx/(ρU²D) drag coefficient.
   The decomposition Cd_pressure + Cd_solvent + Cd_polymer follows
   directly from this Cauchy decomposition.
4. `feedback_kraken_pressure_not_in_jls` — Kraken's `viscoelastic_logfv`
   driver does NOT store `rho` (or `f`) in the .jls snapshot — only
   `ux, uy, tauxx, tauxy, tauyy, psixx, psixy, psiyy, is_solid` and
   scalar diagnostics. Direct wall-pressure integration on saved
   snapshots is impossible; pressure can only be inferred as the
   residual `Cd_total - Cd_solvent_wall - Cd_polymer_wall`.
   For future on-snapshot analyses, save `rho` (or full `f`) too.
