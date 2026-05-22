# M32 Phase 4 — Wi=1 Cd gap localization (R=30, R=40)

Date: 2026-05-21T14:24:44.684
Branch: dev-viscoelastic
Mission: D1 (Department: M32-Phase4-walldecomp)
Verdict: **CONTRADICTS M33 premise**

## TL;DR
- Total gap (Cd_rT − Cd_Kraken) at R=30 Wi=1: **+10.3728 Cd** (target rT=120.38, K=108.72 → +9.54%)
- **Dominant bucket: (`pres`, `front_pole`) carries 80.4% of the gap.**
- M33 premise bucket (polymer × wake): ΔCd = -1.0540  (**-10.2%** of total gap)
- M33 premise threshold (≥70%): **FAILED**
- R=40 cross-check: R=40 dominant=(pres,front_pole) frac=71.4%; rel diff vs R=30 = 11.0%; same bucket=true

## Setup
- Kraken R=30 snapshot: `tmp/m30_rho_metal/run01/cyl_bigsweep_v2_beta0p59_wi1_re1_R30_bsd1_*.jls`
  - Metal F32, :rusanov, BSD=1.0, Cd_stored = 111.0910, has `rho`. (F64 Aqua would give Cd≈111.55.)
- Kraken R=40 snapshot: `tmp/m30_R_sweep_metal/cyl_bigsweep_v2_beta0p59_wi1_re1_R40_bsd1_*.jls`
  - Metal F32, :rusanov, BSD=1.0, Cd_stored = 110.7633, has `rho`.
- rT reference: `bench/rheotool/cylinder_wi1.0_shrunk15R/20/{U,p,tau}.gz` (t=20 converged, Cd.txt last = 120.383)
- Frame: `:idx` (kernel-correct, `cx_lu = cx_phys + 1`, per `[[feedback_wall_ring_idx_frame]]`)
- N_az = 36 bins (Δθ = 10.0°)
- Region split: front-pole `|θ ± π| < π/4` (90°), shoulder `π/4 ≤ |θ ± π/2|` (180°), wake `|θ| < π/4` (90°)
- θ convention: θ=0 at rear-pole (+x), θ=±π at front-pole (-x), θ=±π/2 at shoulders

## Step 1 — Kraken R=30 wall decomposition (:idx frame)

| component        | front_pole | shoulder | wake     | total |
|------------------|------------|----------|----------|-------|
| Cd_pressure      |   +71.5240 | +18.4078 | -13.0347 | +76.8971 |
| Cd_visc_solvent  |    +1.4908 | +18.4580 |  +1.1091 | +21.0579 |
| Cd_polymer       |    +0.3586 |  +8.5140 |  +1.8925 | +10.7651 |
| **column total** |   +73.3734 | +45.3798 | -10.0330 | **+108.7201** |

Reconciliation: Σ ring components = 108.7201 vs stored Cd_kraken = 111.0910 → drift = -2.13%

## Step 2 — rheoTool Wi=1 shrunk15R wall decomposition (t=20)

| component        | front_pole | shoulder | wake     | total |
|------------------|------------|----------|----------|-------|
| Cd_pressure      |   +79.8624 | +20.0749 | -14.1371 | +85.8002 |
| Cd_visc_solvent  |    +1.5574 | +17.1189 |  +1.1797 | +19.8559 |
| Cd_polymer       |    +0.9466 | +11.6517 |  +0.8385 | +13.4368 |
| **column total** |   +82.3664 | +48.8454 | -12.1188 | **+119.0930** |

Reconciliation: Σ ring components = 119.0930 vs Cd.txt last = 120.383 → drift = -1.07%

## Step 3 — 3×3 bucket gap matrix (ΔCd = rT − Kraken R=30)

| ΔCd              | front_pole | shoulder | wake     | row sum |
|------------------|------------|----------|----------|---------|
| pressure         |    +8.3384 |  +1.6671 |  -1.1023 |  +8.9031 |
| visc_solvent     |    +0.0666 |  -1.3392 |  +0.0706 |  -1.2020 |
| polymer          |    +0.5880 |  +3.1377 |  -1.0540 |  +2.6717 |
| **col sum**      |    +8.9930 |  +3.4656 |  -2.0858 | **+10.3728** |

Fraction of total gap (sign-aware, denominator = 10.3728):

| fraction (%)     | front_pole | shoulder | wake     |
|------------------|------------|----------|----------|
| pressure         |     +80.4% |   +16.1% |   -10.6% |
| visc_solvent     |      +0.6% |   -12.9% |    +0.7% |
| polymer          |      +5.7% |   +30.2% |   -10.2% |

## Step 4 — Dominant bucket + M33 premise check

Top 5 buckets by |fraction|:

| rank | component | region | ΔCd | fraction |
|------|-----------|--------|-----|----------|
| 1 | pressure | front_pole | +8.3384 | +80.4% |
| 2 | polymer | shoulder | +3.1377 | +30.2% |
| 3 | pressure | shoulder | +1.6671 | +16.1% |
| 4 | visc_solvent | shoulder | -1.3392 | -12.9% |
| 5 | pressure | wake | -1.1023 | -10.6% |

**Dominant bucket: (`pres`, `front_pole`)** with fraction = **80.4%** of total gap.

**M33 premise** asserts the locus is `(polymer, wake)` (wake-side `:rusanov`-over-dissipation):
- Observed (polymer, wake) fraction = **-10.2%** of total gap
- Threshold for premise CONFIRMED: ≥ 70%
- **Premise **NOT CONFIRMED****

**The actual dominant bucket is `(pressure, front_pole)`. This CONTRADICTS the M33 premise.**

## Step 5 — R=40 cross-check

Kraken R=40 ring totals: Cd_pres=76.1497, Cd_solv=20.9740, Cd_poly=11.3527, Σ=108.4764 (stored=110.7633, drift=-2.06%)

Total gap rT − K40 = **+10.6165 Cd** (vs R=30 gap = +10.3728)

| ΔCd (R=40)       | front_pole | shoulder | wake     | row sum |
|------------------|------------|----------|----------|---------|
| pressure         |    +7.5784 |  +2.5229 |  -0.4508 |  +9.6505 |
| visc_solvent     |    +0.1384 |  -1.3066 |  +0.0501 |  -1.1181 |
| polymer          |    +0.4685 |  +2.3788 |  -0.7632 |  +2.0841 |

R=40 dominant=(pres,front_pole) frac=71.4%; rel diff vs R=30 = 11.0%; same bucket=true

## Caveats

- Both Kraken snapshots are **Metal F32**, not Aqua F64. Per the M32 Phase 3 mandate table, F32 Cd is typically ~0.5 Cd off F64 (R=30 F32=111.09 vs F64=111.55). The **bucket identity** is expected stable; the **magnitude** of |fraction| may shift by ~5% relative.
- Reconciliation drift between ring Σ and stored Cd_kraken: the ring integral on the staircased boundary is a different decomposition than the LBM-MEA cut-link integral (`Cd_kraken = Cd_s + Cd_p − Cd_bsd`). Drift up to ~5% is structural and not a methodology error (see `[[feedback_wall_ring_idx_frame]]`).
- The rT FOAM cell-gradient is reconstructed via kNN affine fit, not the bit-for-bit OpenFOAM Gauss-linear gradient. Solvent contribution is the most sensitive to this — expect ~5% variation on the solvent component alone.
- Both pressure constants float (Kraken: ρ-1 LBM with no reference fix, rT: zeroGradient on inlet); only **gradients** of p around the cylinder are physically meaningful for Cd_pressure. Both codes use the same convention.

## Files

- `bench/scratch/m32_phase4_wi1_walldecomp/M32P4_kraken_R30_bins_idx.csv` — 36 bins, Kraken R=30 :idx-frame
- `bench/scratch/m32_phase4_wi1_walldecomp/M32P4_kraken_R40_bins_idx.csv` — 36 bins, Kraken R=40 :idx-frame
- `bench/scratch/m32_phase4_wi1_walldecomp/M32P4_rheotool_wi1_shrunk_bins.csv` — 36 bins, rT Wi=1 shrunk15R t=20
- `bench/scratch/m32_phase4_wi1_walldecomp/M32P4_bucket_matrix.csv` — flat 3×3 bucket table
- `bench/scratch/m32_phase4_wi1_walldecomp/run_walldecomp.jl` — full driver

## Memory candidates

1. **M32 Phase 4 bucket-attribution template** — the (component × region) 3×3 matrix on the `:idx` ring is a reusable template for any Kraken-vs-rT viscoelastic Cd-gap attribution. Pattern: K-side via `kraken_wall_decomp` on a `.jls` snapshot (requires `rho` field — check schema), rT-side via `rheotool_wall_decomp` on the `constant/polyMesh` + last-time `(U, p, tau).gz`, both binned with the SAME `N_az` and the same θ convention, then `aggregate_regions` collapses to 3 angular buckets. Total dev cost: 1 mission once the locked harnesses (`m30_centering_audit`, `m29c_wallstress`) exist.
2. **Kraken `.jls` schema variability** — different sweep generations produce dumps with different field sets. `tmp/m29c_kraken/` (M29c rolled-back run) has `:rusanov`? No — it's `muscl_superbee` with Cd=-1571 (catastrophe), missing `rho`. `tmp/m30_rho_metal/run01/` is the canonical Wi=1 R=30 :rusanov F32 dump with `rho`. ALWAYS inspect `propertynames(snap)`, `snap.advection_scheme`, and `snap.Cd_kraken` BEFORE using any dump — do not trust the directory name alone.
3. **3-region split convention** — front-pole `|θ±π|<π/4`, wake `|θ|<π/4`, shoulder `π/4≤|θ|≤3π/4`. Each region is 90° (front, wake) or 180° (shoulder, both flanks combined). For finer resolution use 5° bins (N_az=72) and post-aggregate; the 36-bin (10°) granularity is the minimum that distinguishes pole/shoulder/wake cleanly without staircase aliasing.
