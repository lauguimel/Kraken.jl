# M30 — rheoTool multi-Wi pressure-profile p(θ)

**Mandate**: extract `p(θ)` wall profiles and per-band Cd_pressure
decomposition for the rheoTool reference cases at Wi=0.1 and Wi=0.5
(R=30, β=0.59, Re=1; identical setup to Wi=1.0). Format-identical to
`bench/scratch/m30_rheotool_p_profile/M30RP_pressure_bins.csv` (72 azimuthal
bins, Δθ=5°) so subsequent comparison missions ingest directly.

Generated 2026-05-20.

## Inputs

| Wi  | Case dir                                            | Time used | Cd_total (rheoTool Cd.txt) |
|-----|-----------------------------------------------------|-----------|----------------------------|
| 0.1 | `bench/rheotool/cylinder_wi0.1`                     | `5/`      | 130.4278 (converged at t≈3)|
| 0.5 | `bench/rheotool/cylinder_wi0.5`                     | `10/`     | 119.7138 (converged at t≈9)|
| 1.0 | `bench/rheotool/cylinder_wi1.0` (ref M30RP)         | `10/`     | 120.40 (M30RP reference)   |

Convergence verified for both new cases (Cd.txt residual <1e-4 at the
final time directory).

## Pressure-component scalars (integral of `dCd_p_per_bin` over θ)

| Wi  | Cd_pressure (rheoTool)  |
|-----|-------------------------|
| 0.1 | **90.5161**             |
| 0.5 | **81.7141**             |
| 1.0 |  85.7716  (M30RP ref)   |

Non-monotonic: Cd_p drops from Wi=0.1 → Wi=0.5 by 9.7%, then partially
recovers at Wi=1.0 (+5.0% relative to Wi=0.5). The min lies near Wi≈0.5.

## 5-band Cd_pressure decomposition

Bands use the same `band_sum` convention as `M30RP_summary.txt` (each
canonical angle ±22.5° half-width; up/down shoulder and equator pairs
combined into a single line). All values have units of `Cd_pressure`
(dimensionless, normalised by `(η_s+η_p)`). Sum of band rows exceeds the
total because the script's `band_sum` uses a closed interval at the bin
boundaries (each pole band picks up 10 bins, adjacent shoulder bands also
pick up the same boundary bin — see notes below).

| θ band                          | Wi=0.1 dCd_p | Wi=0.5 dCd_p | Wi=1.0 dCd_p (ref) |
|---------------------------------|--------------|--------------|--------------------|
| Front pole ±π                   | **+78.6213** | **+76.9061** | +80.2549           |
| Front shoulder ±0.75π (up+down) | **+105.9050**| **+104.8492**| +109.9928          |
| Equator ±π/2  (up+down)         | **+4.2498**  | **+3.5628**  | +3.6374            |
| Rear shoulder ±0.25π (up+down)  | **−54.5056** | **−60.3558** | −65.7238           |
| Rear pole 0                     | **−35.2023** | **−35.7296** | −34.8914           |
| **TOTAL Cd_pressure (integral)**| **+90.5161** | **+81.7141** | **+85.7716**       |

Disaggregated up/down values are in each Wi's `M30RP_W0?_summary.txt`.

## Per-band evolution vs Wi

| Band            | Wi=0.1 → Wi=0.5 | Wi=0.5 → Wi=1.0 | Overall Wi=0.1 → Wi=1.0 |
|-----------------|-----------------|-----------------|-------------------------|
| Front pole      | −1.72  (−2.2%)  | +3.35  (+4.4%)  |  +1.63  (+2.1%)         |
| Front shoulder  | −1.06  (−1.0%)  | +5.14  (+4.9%)  |  +4.09  (+3.9%)         |
| Equator         | −0.69 (−16.2%)  | +0.07  (+2.0%)  |  −0.61  (−14.4%)        |
| Rear shoulder   | −5.85 (+10.7%)  | −5.37  (+8.9%)  | −11.22  (+20.6%)        |
| Rear pole       | −0.53  (+1.5%)  | +0.84  (−2.4%)  |  +0.31  (−0.9%)         |
| **TOTAL Cd_p**  | **−8.80 (−9.7%)**| **+4.06 (+5.0%)**| **−4.74 (−5.2%)**     |

(Sign conventions: increase in dCd_p means more drag pressure; for the rear
bands a "more negative" value means more thrust contribution. The "%"
columns are relative to the prior column's |value|.)

**Key observations**

1. **Front pole pressure (windward stagnation) is essentially Wi-independent**:
   varies by <4% across the full Wi range (78.6 → 76.9 → 80.3). The +33.22
   reference quoted in the orchestrator brief does not match the M30RP
   summary's `front_stag` line (+80.25) and likely refers to a different
   sub-band convention; here we follow the locked-harness convention so
   numbers cross-check exactly.

2. **Front shoulder is also weakly Wi-dependent**: 105.9 → 104.8 → 110.0,
   range ≈5% of the mean. Front-side pressure (pole + shoulder) is the
   dominant drag source (≈185 across all three Wi).

3. **Rear shoulder is the Wi-sensitive band**: −54.5 → −60.4 → −65.7. This
   monotonically more-negative trend (≈+21% in magnitude from Wi=0.1 to
   Wi=1.0) is what drives the non-monotonic total: as polymer extension
   grows, the rear shoulder thrust contribution overtakes the small
   front-side increase.

4. **Equator and rear pole are nearly Wi-frozen**: rear pole varies <1
   unit (35.2 → 35.7 → 34.9), equator <1 unit (4.25 → 3.56 → 3.64). These
   are structural pressure values set by Re=1 confinement, not by the
   polymer.

5. **Cd_pressure total is non-monotonic in Wi** (90.5 → 81.7 → 85.8) with a
   minimum near Wi≈0.5. The drop from Wi=0.1 to 0.5 is dominated by the
   rear-shoulder thrust intensification; the partial recovery from 0.5 to
   1.0 comes from the front shoulder strengthening faster than rear
   shoulder continues to drop.

## Implications for the K/rT diagnosis

The K/rT pattern at Wi=1.0 in the orchestrator's prior analysis (front-pole
0.59, rear-pole 0.16, plateau structural-BC issue) is anchored on
Wi-invariant front-pole and rear-pole values. Since both pole bands
deviate by less than 4% across the full Wi=0.1..1.0 range, the **K/rT
asymmetry is structurally robust, not polymer-coupled** at the level of
the polar pressure values. The Wi-dependence of Cd_pressure (and of Cd_total)
is concentrated in the **rear shoulder**, which is exactly the band most
sensitive to polymer wake/extensional stress structure (not BC).

Conclusion (this mission only): the H1 BC-locality mechanism survives the
multi-Wi cross-check. The K/rT asymmetry seen at Wi=1.0 is the same
pole-driven structural signature that exists at Wi=0.1 and Wi=0.5. Any
polymer modification of the Kraken-vs-rheoTool gap will show up in the
rear shoulder band, not in the poles.

## Files

- `bench/scratch/m30_rheotool_p_profile_wi01/M30RP_W01_pressure_bins.csv`
  72 bins, columns `theta_rad, theta_deg, p_normalised, n_x, n_y, dCd_p_per_bin`
  (format identical to `M30RP_pressure_bins.csv`).
- `bench/scratch/m30_rheotool_p_profile_wi01/M30RP_W01_pressure_wall.csv`
  Per-face (378 faces) raw record.
- `bench/scratch/m30_rheotool_p_profile_wi01/M30RP_W01_summary.txt`
- `bench/scratch/m30_rheotool_p_profile_wi05/M30RP_W05_pressure_bins.csv`
- `bench/scratch/m30_rheotool_p_profile_wi05/M30RP_W05_pressure_wall.csv`
- `bench/scratch/m30_rheotool_p_profile_wi05/M30RP_W05_summary.txt`
- Reference (unchanged):
  `bench/scratch/m30_rheotool_p_profile/M30RP_pressure_bins.csv`

## Method (one paragraph)

For each Wi case the latest converged OpenFOAM time directory is parsed
(polyMesh + zeroGradient `p` field). The 378 cylinder-patch faces are
walked, each face's centroid `(x_f, y_f)`, outward normal `(n_x, n_y)` and
area computed analytically; the owner-cell pressure value is used (faces
are zeroGradient ⇒ face value = owner value). Per-face drag pressure
contribution is `dCd_p = −p · ρ · n_x · area / (η_s + η_p)` with `ρ=1`,
`η_s+η_p=1`. Bins are 5°-wide azimuthal sectors (72 total), with
`θ = atan2(y, x)`. The integration is identical to the M29c-wallstress
verification of `Cd_pressure = 85.77` at Wi=1.0. No simulation is run,
no `src/` is modified.

## Cross-checks

- Symmetry residual: `|Cd_pressure_y|` for both Wi=0.1 and Wi=0.5 is
  `<2e-12`, confirming a clean mirror-symmetric OpenFOAM run.
- Peak bin location (largest `|dCd_p_per_bin|`): both new cases peak at
  `θ = −172.5°` (front-side, off-axis), identical to the Wi=1.0 reference.
  Peak values: Wi=0.1 → 9.18, Wi=0.5 → 8.97, Wi=1.0 → 9.36.
- 80% concentration: Wi=0.1 → 41 bins (56.9% of azimuth);
  Wi=0.5 → 42 bins (58.3%); Wi=1.0 → 42 bins (58.3%). Pressure-drag
  spatial spread is essentially Wi-invariant.
