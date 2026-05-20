# M30 Phase 1 R-sweep — Claude step 1 (solo, pre-Codex)

Date : 2026-05-20
Engine : Claude (Anthropic Opus 4.7, 1M)
Harness : `bench/scratch/m30_phase1_R_sweep_claude/run_p_vs_R.jl`
Frame : `:idx` (kernel-correct per M31 verdict, `dx = (i−1) − cx_phys`)
N_az  : 72 (5°/bin), aligned element-by-element with rheoTool
rT ref : `bench/scratch/m30_rheotool_p_profile/M30RP_pressure_bins.csv` (total Cd_p = **85.7716**)

Snapshots : Metal F32, BSD=1.0, Wi=1, β=0.59, M29b `:rusanov`, 100k steps, ρ persisted.

---

## Q1 — Per-R scalars

| R  | Cd_kraken (stored) | Cd_s (stored) | Cd_p (driver) | Cd_bsd (stored) | **Cd_pressure ring (`:idx`)** | gap rT − K |
|----|---:|---:|---:|---:|---:|---:|
| 20 | 111.8202 | 118.5025 |  9.3270 | 16.0093 | **78.6006** | +7.1710 |
| 30 | 111.0910 | 115.2047 | 11.4895 | 15.6032 | **76.6220** | +9.1496 |
| 40 | 110.7633 | 114.0125 | 12.0972 | 15.3464 | **76.4621** | +9.3094 |

Internal consistency check (`:idx` ring vs stored `Cd_kraken`) — the ring decomposition is
`pressure + solvent + polymer`, the stored `Cd_kraken` is the LBM cut-link MEA
decomposition `Cd_s + Cd_p − Cd_bsd`. They report the same total drag in different splits;
the per-R relative gap of `Cd_kraken stored − (Cd_pressure_ring + Cd_solvent_ring + Cd_polymer_ring)`
is **out of scope** for this mission (M30 Phase 0c / M31 already audited it). What matters here is:

- **Cd_kraken total stabilises across R** : 111.82 → 111.09 → 110.76 (Δ_R20→40 = −1.06, < 1 % of 111).
  So the M29b production target itself is converged to better than 1 % wrt R.
- **Cd_p driver-stored INCREASES with R** : 9.33 → 11.49 → 12.10 (+30 %).
- **Cd_pressure ring DECREASES with R** : 78.60 → 76.62 → 76.46 (Δ_R20→40 = −2.14).
- **Cd_s + Cd_bsd both DECREASE with R**, partially offsetting the polymer rise.

Consistent picture : as R increases, more of the total Cd is captured as polymer wall drag
(higher `Cd_p`) and less as the LBM cut-link `Cd_s` bundle. **Cd_pressure** (the ring
quantity probed here) does NOT converge to rT 85.77 — it MOVES AWAY from it.

---

## Q2 — 5-band table

| band            |       rT |    K_R20 |    K_R30 |    K_R40 | K/rT R20 | K/rT R30 | K/rT R40 |
|-----------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| Front pole ±π   |  +33.223 |  +19.379 |  +19.584 |  +19.729 |  +0.583  |  +0.589  |  +0.594  |
| Front shoulder  |  +89.522 |  +53.780 |  +53.949 |  +54.950 |  +0.601  |  +0.603  |  +0.614  |
| Equator         |   +1.541 |   +3.107 |   +2.900 |   +2.910 |  +2.017  |  +1.882  |  +1.888  |
| Rear shoulder   |  −52.751 |  −15.673 |  −18.216 |  −19.955 |  +0.297  |  +0.345  |  +0.378  |
| Rear pole 0     |  −26.484 |   −4.698 |   −4.220 |   −3.809 |  +0.177  |  +0.159  |  +0.144  |
| **TOTAL Cd_p**  | **+85.772** | **+78.601** | **+76.622** | **+76.462** | **+0.916** | **+0.893** | **+0.891** |

CSV : `bench/scratch/m30_phase1_R_sweep_claude/M30P1R_bands.csv`

---

## Q3 — K/rT amplitude ratio, front-pole vs rear-pole, vs R

- **Front pole** (rT = +33.22) : `K/rT = 0.5833 → 0.5895 → 0.5938`
  - Δ_R20→40 = **+0.0105** (very small, < 2 %).
  - Trend : monotonically increasing but **near-flat**. Linear extrapolation:
    front-pole K/rT would still be ≈ 0.61 at R=80 and ≈ 0.63 at R=160.
- **Rear pole** (rT = −26.48) : `K/rT = 0.1774 → 0.1593 → 0.1438`
  - Δ_R20→40 = **−0.0335** (*degrading*, not improving).
  - Trend : monotonically **decreasing** with R, moving AWAY from 1.0.

The front-pole pressure peak (where Kraken under-recovers rT) is essentially R-independent
at this resolution range : refining the lattice does NOT recover the missing pressure
amplitude. The rear-pole pressure (also missing in K) gets WORSE with R, not better. Both
are red flags for "resolution-limited" since a resolution-limited deficit should monotonically
shrink with R.

CSV : `M30P1R_bands.csv` ratio columns.

---

## Q4 — Cd_pressure scalar gap vs R

`gap(R) = rT_total − Cd_pressure_ring(R)`

| R  | Cd_pressure ring | gap |
|----|---:|---:|
| 20 | 78.6006 | **+7.1710** |
| 30 | 76.6220 | **+9.1496** |
| 40 | 76.4621 | **+9.3094** |

**The gap GROWS, not shrinks, as R increases.** The fraction (gap_R20 − gap_R40) / gap_R20
is **−29.8 %** (negative — divergent, not convergent).

Log-log slope analysis (pairwise) :
- R=20 → R=30 : slope = +0.601
- R=30 → R=40 : slope = +0.060
- LS over 3 points : +0.391

A *positive* slope means `gap ∝ R^+0.39`, i.e. **divergent in R**, not convergent. There is
no R→∞ asymptote at which the gap closes. Naïve extrapolation gives gap ≈ 12.6 at R=80 and
16.6 at R=160 (still using the LS slope). The 30→40 pairwise slope is much flatter (+0.06)
than the 20→30 one (+0.60), suggesting the gap is approaching a plateau near 9.3 rather
than continuing to grow ; but it does NOT decrease.

Convergence rate verdict : **no convergence detected** (best-case interpretation: plateau
at ≈ 9.3 ± 0.2 for R ≥ 40).

---

## Q5 — Verdict (Claude solo)

**`structural-BC`** (front-pole K/rT plateau at < 1, R-independent ; gap does not shrink with R).

Evidence :
1. Front-pole K/rT only moves by +0.0105 over a doubling of R (20 → 40). Δ-per-R-doubling
   ≈ 0.01 ; if this were the dominant convergence rate, R=80 would only push it to ≈ 0.60,
   R=160 to ≈ 0.61, R=320 to ≈ 0.62. Asymptote estimate ≲ 0.65, far short of 1.0.
2. Rear-pole K/rT goes the **wrong way** with R (0.177 → 0.144).
3. Total `Cd_pressure_ring` does NOT converge to rT — it diverges *away* from it (78.60 →
   76.46 ; gap 7.17 → 9.31).
4. Meanwhile the stored `Cd_kraken` total IS well-converged (111.82 → 110.76, < 1 %), so
   this is not a "all quantities still moving" regime. The total drag is converged ;
   it's the *decomposition* across pressure/solvent/polymer that doesn't match rT.
5. The polymer wall drag `Cd_p` driver-stored INCREASES with R (9.33 → 12.10), partially
   compensating the missing pressure. This is consistent with a halfway-BB ↔ no-slip ↔
   ρ-extrapolation mechanism that shifts traction between pressure and polymer components
   without changing the total — a structural BC class issue, not a discretisation error.

**Implication for M30 H1 ranking** : if the verdict survives Codex cross-check, H1 (ρ-BC
class) is confirmed as the primary mechanism. Production targets should NOT chase higher R
to close the gap (Boss decision : abandon the "R=60-80 needed" branch). Instead, investigate
ρ-BC alternatives (Inamuro pressure BC, interpolated bounce-back, Zou-He extended for
curved walls) — none of which is part of the current Kraken viscoelastic stack.

The lone caveat : front-pole K/rT does increase monotonically by +0.01 over R=20 → R=40, so
"structural-BC" should be qualified as "structural-BC dominant + small resolution component".
The verdict is *not* "intermediate" because the resolution component is too small to ever
close the gap.

---

## Files produced (Claude step 1)

- `bench/scratch/m30_phase1_R_sweep_claude/run_p_vs_R.jl`        — harness
- `bench/scratch/m30_phase1_R_sweep_claude/M30P1R_bins_R20.csv`  — per-bin (R=20)
- `bench/scratch/m30_phase1_R_sweep_claude/M30P1R_bins_R30.csv`  — per-bin (R=30)
- `bench/scratch/m30_phase1_R_sweep_claude/M30P1R_bins_R40.csv`  — per-bin (R=40)
- `bench/scratch/m30_phase1_R_sweep_claude/M30P1R_bands.csv`     — 5-band × 3-R table
- `bench/scratch/m30_phase1_R_sweep_claude/M30P1R_scalars.csv`   — Q1 scalars
- `bench/scratch/m30_phase1_R_sweep_claude/M30P1R_stdout.log`    — execution log
