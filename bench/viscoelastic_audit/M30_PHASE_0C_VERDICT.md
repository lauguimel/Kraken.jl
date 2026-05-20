# M30 Phase 0c — Kraken vs rheoTool wall-pressure profile p(θ)

**Date** : 2026-05-20
**Snapshot** : `tmp/m30_rho_metal/run01/cyl_bigsweep_v2_beta0p59_wi1_re1_R30_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls`
- Backend : Metal F32 (ρ stored F64)
- Advection scheme on log-Ψ : `:rusanov` (the M29b BASELINE first-order upwind, not the M29b-MUSCL variant)
- Wi = 1, β = 0.59, Re = 1, R_lu = 30, bsd_fraction = 1, L_up = L_down = 15, all `embedded_*` flags OFF (`qwall` geometry)
- 100 k transient steps, 20 % avg window. `Cd_kraken = 111.091`.

**Reference** : `bench/scratch/m30_rheotool_p_profile/M30RP_pressure_bins.csv` (rheoTool t=10, 378 cylinder-patch faces → 72 bins of 5°). `Cd_pressure_rT = 85.7716`.

---

## Cd_pressure scalar (cross-check)

| quantity | value | source |
|---|---|---|
| Kraken Cd_pressure (this run, wall-ring ρ integral) | **76.64** | new `c_s²·(ρ−1)·n_x` integral, 242 ring cells |
| M29c-wallstress M29b residual reference | 75.64 | Aqua F64, 30 k steps, ρ-free residual = `Cd_kraken − Cd_solvent_wall − Cd_polymer_wall` |
| Δ this run − M29c reference | +1.00 (+1.33 %) | direct (Metal F32) vs residual (Aqua F64) — agrees |
| Δ wall_ring vs Cd_kraken stored | +76.64 (direct) is partial decomposition only; `Cd_kraken = 111.09 = Cd_s + Cd_p − Cd_bsd` includes solvent-viscous in `Cd_s` | — |
| **rheoTool** Cd_pressure | **85.77** | M30RP, 378 faces × `−p·n_x·area / (etaS+etaP)` |
| **Gap rT − K** | **+9.13 pts (+11.9 %)** | Kraken under by 11.9 % on the wall-pressure integral |

Cross-check verdict : the new direct ρ-integral agrees with the M29c-wallstress residual (1.3 % Δ — within F32/F64+transient noise). The +10.2-pt gap quoted in the brief is reproduced at +9.13 pts on this snapshot. Y-asymmetry residual `Cd_pressure_y = +0.27` (~0.3 % of x-integral) → ring sampling is essentially symmetric.

## Side-by-side 5-band table (the headline)

| band | rheoTool dCd_p | Kraken dCd_p | gap (rT − K) | % of NET gap | % of Σ|gap| |
|---|---:|---:|---:|---:|---:|
| Front pole ±π   (\|θ\| > 0.875π) | +80.25 | +48.01 | **+32.24** | +353.2 % | 20.95 % |
| Front shldr     (\|θ\| ∈ [0.625, 0.875]π) | +109.99 | +61.83 | **+48.16** | +527.5 % | 31.30 % |
| Equator         (\|θ\| ∈ [0.375, 0.625]π) | +3.64  | +2.85  | +0.79      | +8.6 %    | 0.51 %  |
| Rear shldr      (\|θ\| ∈ [0.125, 0.375]π) | −65.72 | −21.52 | **−44.20** | −484.2 %  | 28.73 % |
| Rear pole 0     (\|θ\| < 0.125π) | −34.89 | −6.42  | **−28.47** | −311.9 %  | 18.50 % |
| **TOTAL (5 bands ≈ 80 % of azim.)** | +93.27 | +84.76 | +8.51 | 100 % | 100 % |
| FULL (∫ all 72 bins) | +85.77 | +76.64 | **+9.13** | — | — |

**Key observation** : the NET gap (+9.13) is the **DIFFERENCE OF TWO MUCH LARGER PER-BAND GAPS** that cancel because rheoTool's p(θ) is anti-symmetric across the equator (front pos / rear neg). The sign-preserved per-band fractions exceed ±100 % because of this cancellation. The amplitude-fraction column (|gap| / Σ|gap|) is the trustworthy locus indicator.

## Gap localisation

- **Front-arc** (\|θ\| ≥ 0.625π) gap = **+73.65** (806.8 % of NET; **52.25 % of Σ|gap|**)
- **Mid-arc**                   gap = **+1.85**   (20.2 %  of NET; 0.51 % of Σ|gap|)
- **Rear-arc**  (\|θ\| ≤ 0.375π) gap = **−66.37** (−727.0 % of NET; **47.24 % of Σ|gap|**)

The front-arc and rear-arc contribute essentially **equal magnitudes** of the disagreement (52 % vs 47 %, parity to within ±3 %). The signs are opposite, hence the small NET.

But the **per-bin K/rT amplitude ratio** is **strongly inhomogeneous** :

| arc | mean K/rT | std | n | interpretation |
|---|---:|---:|---:|---|
| Front (\|θ\|>0.625π, large \|rT\|) | **0.578** | 0.061 | 26 | Kraken under by ~42 %, fairly uniform |
| Rear  (\|θ\|<0.375π, large \|rT\|) | **0.281** | 0.105 | 27 | Kraken under by ~72 %, more scattered |
| Rear pole (\|θ\|<0.125π) | **0.155** | 0.094 | 8  | Kraken's pressure at rear stagnation collapses to ~15 % of rheoTool |

The pressure profile is NOT a uniform damping. The **rear (lee) side of the cylinder is damped 2× more strongly than the front (windward) side**. The rear-stagnation pole drops to <16 % of the rheoTool value (bin 36/37 : rT = −2.70, K = −0.22 and −0.02 respectively).

## Verdict

**INHOMOGENEOUS DAMPING** — K/rT_front ≈ 0.58 vs K/rT_rear ≈ 0.28 (ratio inhomogeneity = 0.30 ≫ 0.10 threshold).

The +9.13-pt NET gap is the residual of two opposite-sign band gaps of ~70 pts each, but the **shape of p(θ)** in Kraken is **flatter than rheoTool's** and the **rear half collapses harder**. Specifically the rear stagnation pressure is essentially zero in Kraken but rheoTool sustains a finite suction there.

### Implication for the H1/H2/H3/H4 ranking

| Hypothesis | Definition | Ranked by Phase 0c |
|---|---|---|
| **H1** (LBM ρ-BC at the cylinder wall) | Halfway BB + Mei MEA at the staircased cut yields a ρ field whose values immediately outside the solid are a one-sided extrapolation; if this extrapolation under-damps the upstream pressure rise and over-damps the wake recovery, the ratio K/rT will be asymmetric front-vs-rear. **The K/rT_front=0.58 / K/rT_rear=0.28 pattern is the SIGNATURE of a 1-sided BC stencil that does NOT respect the lee-side relaxation properly.** | **RANKED PRIMARY** |
| **H3** (pressure-gradient stencil near wall) | The FVFD ∇p stencil used in the polymer body-force assembly may bias near-wall p reconstruction; same diagnostic but coupling-mediated. | **RANKED SECONDARY** — likely correlated with H1 since H3 acts via ρ ⟷ τ_p coupling on the same ring. |
| **H2** (BSD wide-vs-narrow stencil) | Acts mostly on the polymer side, would show in Cd_polymer / wall-τ_p balance, NOT in the LBM ρ wall-trace. | **DOWN-RANKED** — not consistent with the inhomogeneous K/rT shape (would give a more diffuse pattern). |
| **H4** (embedded-mode cell-fraction overdose) | OFF on this snapshot (all `embedded_*` flags = false, `qwall` geometry). | **EXCLUDED** by construction for this run. |

**Operational ranking** : H1 first (LBM ρ-BC / wall halfway-BB pressure reconstruction asymmetry between windward and lee side), H3 second (FVFD ∇p stencil on the staircased wall ring). H2 demoted; H4 not applicable.

## Files

- `bench/scratch/m30_kraken_p_profile/run_kraken_p_profile.jl` — harness (adapted from m30_rheotool_p_profile + m29c_wallstress)
- `bench/scratch/m30_kraken_p_profile/M30KP_pressure_bins.csv` — 72-bin Kraken
- `bench/scratch/m30_kraken_p_profile/M30KP_pressure_ring.csv` — 242 ring cells per-cell
- `bench/scratch/m30_kraken_p_profile/M30KP_sidebyside.csv` — 72-bin Kraken vs rheoTool side-by-side
- `bench/scratch/m30_kraken_p_profile/M30KP_summary.txt` — auto summary

## Notes for downstream

1. The wall-ring is **242 ring cells** (≥1 solid neighbour in 8-stencil) over the 188-cell circumference (`2π·R = 188 LU`). About 1.3 ring cells per LU of perimeter — the staircase has a 2-3× density variation around the circle. The harness handles this by **averaging traction within each 5° bin then multiplying by the single arc_dl** — summing per-cell `arc_dl` (the first naive version of the harness) double-counts by the cell occupancy.
2. The cylinder centre is at `cy = 59.5` (half-integer) → ring is exactly symmetric in Y, but the staircase pattern has 1-cell offset between left-right and top-bottom halves. The 0.3 % Y-asymmetry residual confirms the ring sampling is unbiased.
3. The Metal F32 snapshot at 100 k steps is in the same regime as the M29b Aqua F64 at 30 k steps (residual cross-check 1.3 % apart). Both share the `:rusanov` upwind on Ψ-advection. The follow-up to validate the H1 ranking should ideally be repeated on an Aqua F64 snapshot of the SAME run config (Cd_kraken ≈ 111, same `qwall` flags) once the maintenance window closes.
