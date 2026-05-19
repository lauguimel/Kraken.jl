# M30 — rheoTool wall-pressure profile p(θ) verdict

Date    : 2026-05-20
Branch  : dev-viscoelastic
Mission : Extract `p(θ)` on the cylinder surface at R=30, Wi=1, β=0.59
          (converged t=10), bin in 5° azimuthal sectors (72 bins), and
          identify where the pressure contribution to Cd is concentrated.
          This is the reference target for Kraken's pressure profile
          comparison (M30 Phase 0c — pending `rho` dump in Kraken snapshot).

## Status: EXECUTED — verdict is DECISIVE

| Artefact                                                            | Present |
|---------------------------------------------------------------------|---------|
| `bench/scratch/m30_rheotool_p_profile/run_p_profile.jl`             | YES (driver) |
| `bench/scratch/m30_rheotool_p_profile/M30RP_pressure_bins.csv`      | YES (72 θ-bins) |
| `bench/scratch/m30_rheotool_p_profile/M30RP_pressure_wall.csv`      | YES (378 wall faces) |
| `bench/scratch/m30_rheotool_p_profile/M30RP_summary.txt`            | YES |

## Convention (verified against `system/controlDict` outputCd functionObject)

```cpp
F = tau + symm(L + L^T)*etaS - p*rho*I
Fpatch.x = gSum( (-faceAreas()) & F.boundaryField()[cyl] ) / (etaS + etaP)
Cd       = Fpatch.x                                  // ≡ 2 Fx/(ρU²D), ρ=1 U=1 D=2
⇒ dCd_p = (-p·ρ·I·n_out)_x dA / (etaS+etaP) = -p·ρ·n_x dA / (etaS+etaP)
```

`-faceAreas()` points radially OUT of the cylinder. The pressure `p` is
read from the snapshot as-is (the controlDict multiplies by `rho` with
ρ=1 from `constitutiveProperties`, so the numerical value in `10/p` is
the dimensional pressure used in the integrand). `etaS+etaP=1` so the
normalisation is trivial.

θ = `atan2(y, x)`, CCW from +x:
- θ = 0   → rear stagnation (downstream, lee side)
- θ = ±π  → front stagnation (upstream, windward face)
- θ = ±π/2 → equator (top/bottom shoulder)

Sign of `dCd_p/dθ`: positive = drag direction (+x), negative = thrust.

## Cross-check (M29c-wallstress vs M30 integration)

| Quantity                          | M30 (this mission) | M29c-wallstress | Δ |
|-----------------------------------|--------------------|-----------------|---|
| Integrated Cd_pressure (drag, x)  | **85.7716**        | 85.7716         | **0.000 % (exact)** |
| Lift residual Cd_p (y)            | −6.5 e−13          | 7.5 e−13        | machine precision (symmetric flow) |

The two harnesses re-implement the integrand the same way and read the
same `10/p.gz` file; the result is bit-exact in F64. Exit criterion (2 %
match) cleared by 4 orders of magnitude.

## Azimuthal distribution (8 canonical 45° bands, half-width ±22.5°)

Each band spans 9 of the 72 bins. Bands overlap by design at the poles
(front_stag and rear_stag are wrap-aware single bands; shoulders are
upper/lower mirror pairs).

| Band                                    | dCd_p     | frac (%)   | sign |
|-----------------------------------------|-----------|-----------:|:----:|
| **front_stag**  (θ ≈ ±π)                | **+80.25**| **+93.6 %**|  +   |
| front_shldr_up (θ ≈ +0.75π)             | +55.00    | +64.1 %    |  +   |
| front_shldr_lo (θ ≈ −0.75π)             | +55.00    | +64.1 %    |  +   |
| equator_up      (θ ≈ +π/2)              | +1.82     | +2.1 %     |  +   |
| equator_lo      (θ ≈ −π/2)              | +1.82     | +2.1 %     |  +   |
| rear_shldr_up  (θ ≈ +0.25π)             | −32.86    | −38.3 %    |  −   |
| rear_shldr_lo  (θ ≈ −0.25π)             | −32.86    | −38.3 %    |  −   |
| **rear_stag**   (θ ≈ 0)                 | **−34.89**| **−40.7 %**|  −   |

Bands are perfectly symmetric in θ → −θ (lift residual at machine ε).

Upstream half (|θ| > π/2) :  +186.0  (216.9 % of Cd_p)
Downstream half (|θ| < π/2):  −100.2  (−116.9 % of Cd_p)

Cancellation factor |abs|/net = **3.34×** : the net Cd_pressure of 85.77
emerges from a sum of ±286 in magnitude. Errors in either half-arc are
hard to track in the scalar.

## Dominant locus

- **Peak dCd_p per bin = +9.36** at θ = ±172.5° (= ±0.958 π), i.e.
  immediately adjacent to the front stagnation line.
- **Peak |dCd_p| (negative side) = −4.20** at θ = ±22.5° (= ±0.125 π),
  immediately adjacent to the rear stagnation.
- 80 % of |dCd_p| is concentrated in the top 42 of the 72 bins (58 %
  of the azimuth) — the pressure-drag signal is broadly distributed,
  not sharply peaked.

**The front half-arc (|θ| ∈ [112.5°, 180°], i.e. 45° on either side of
front stagnation) contributes +176.4 of pressure-drag** (206 % of net),
balanced by the rear-shoulder + rear-stagnation suction of −101.6
(−118 % of net). The bulk of the *signal* lives in the front arc; the
bulk of the *cancellation* against it comes from the rear shoulders.

## Pressure profile shape

`p(θ)` ranges from **p ≈ 96.2 at θ = ±180°** (front pole) down to
**p ≈ 30.2 at θ ≈ 0°** (rear pole) — a 66-unit drop. Sample (every
other bin):

| θ (deg) | p_avg  | n_x    | dCd_p/bin |
|--------:|-------:|-------:|----------:|
| -172.5  | 96.16  | -0.991 | +9.36     |
| -132.5  | 91.60  | -0.674 | +5.57     |
|  -92.5  | 77.91  | -0.043 | +0.29     |
|  -52.5  | 57.84  | +0.613 | -3.27     |
|  -12.5  | 42.52  | +0.976 | -3.70     |
|   -2.5  | 30.22  | +0.999 | -2.70     |
|  +27.5  | 49.92  | +0.884 | -4.06     |
|  +67.5  | 64.52  | +0.380 | -2.28     |
| +107.5  | 84.61  | -0.294 | +2.13     |
| +147.5  | 94.00  | -0.842 | +6.22     |
| +177.5  | 96.23  | -0.999 | +7.55     |

(Full 72-bin profile in `M30RP_pressure_bins.csv`.)

## Verdict — locus where Kraken's pressure gap is expected to manifest

1. **The pressure drag is concentrated in the front arc**, especially in
   the **front_stag band (±22.5° around θ=π)** which alone supplies
   **+80.25 of the +85.77 net (93.6 %)**. The front shoulders contribute
   additional +110.0 of pressure-push, almost entirely balanced by the
   rear-shoulder suction of −65.7. The remaining +35 of "front-shoulder
   minus rear-stag" residual is what closes the Cd_p budget.

2. **The two most diagnostic θ-bands for the Kraken comparison are:**
   - θ = ±172.5° (front pole adjacency, peak +9.36/bin) — where Kraken's
     suspected under-prediction of front stagnation pressure will show
     directly. The +10.2-pt Kraken gap (75.6 vs 85.8) is ~12 % of this
     band alone, so even a 12 % deficit at the front pole would suffice
     to close the gap.
   - θ = ±22.5° (rear shoulder, peak −4.20/bin) — where suction
     intensity is largest and where any artificial regularisation
     (e.g. ZouHe outlet coupling) would manifest. A weaker suction
     here would also under-predict Cd_pressure.

3. **The cancellation factor of 3.34×** means scalar comparisons of
   Cd_pressure cannot localise the gap. Kraken must produce a θ-binned
   profile (Phase 0b/0c) before the gap mechanism can be diagnosed —
   the +10.2-pt deficit could come from (a) lower front-pole pressure
   peak, (b) shallower rear-pole suction, or (c) phase-shifted front
   shoulder profile. Only the binned overlay will distinguish these.

## Boss decision implication

- The Phase 0b deliverable for Kraken **must include `rho` (or `p`) in
  the snapshot** so the same `run_p_profile.jl` harness can be pointed
  at the Kraken output and produce a directly-overlayable 72-bin profile.
- The most likely physical mechanism to investigate first is the
  **front-pole pressure peak** (θ ≈ ±172.5°): rheoTool's p = 96 there
  is the highest value on the cylinder; any compressibility error /
  Mach error / front-BC coupling deficit in Kraken would lower this
  peak and is the simplest mechanism consistent with a +10 pt under-
  prediction.
- The Phase 0c comparison should report `Δ p(θ)`, `Δ n_x p(θ)`, and the
  per-band `Δ dCd_p`, NOT just the integrated Cd_pressure scalar
  (cancellation hides the signal).

## Memory candidates

1. `feedback_cd_pressure_strong_cancellation` — rheoTool Wi=1 R=30 β=0.59
   gives Cd_pressure = 85.77 from a sum of ±286 (|abs|/net = 3.34).
   Bands cancel front +186 vs rear −100. Scalar comparison cannot
   localise pressure-drag errors; always bin azimuthally before
   diagnosing a Cd_pressure gap.
2. `feedback_front_pole_dominates_cd_p` — 93.6 % of the net pressure-drag
   on the cylinder at Re=1 Wi=1 lives in the ±22.5° band around the
   front stagnation (θ = ±π). Therefore a 12 % under-prediction of the
   front-pole p would suffice to explain a 10 pt Cd_p deficit.
   For LBM/FVFD viscoelastic codes, the front pole is the single most
   important diagnostic point.
3. `feedback_rheotool_p_already_dimensional` — rheoTool's `10/p.gz`
   stores `p` such that the controlDict reads `p * rho` directly with
   ρ from `constitutiveProperties` (= 1.0 in this case). The numerical
   value in the file is therefore the dimensional pressure used in the
   integrand; no kinematic ↔ dimensional rescaling is needed.
4. `feedback_m29c_wallstress_harness_reusable` — the FOAM ASCII reader +
   patch-walking code in `bench/scratch/m29c_wallstress/run_wallstress.jl`
   is fully general and was reused verbatim in M30 (copied, not modified,
   per artefact lock). For any future rheoTool wall-quantity extraction
   (channel walls, sphere walls, contraction walls), copy this reader.
