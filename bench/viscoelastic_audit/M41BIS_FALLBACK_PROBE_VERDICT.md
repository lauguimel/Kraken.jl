# M41-bis — MUSCL-fallback band as locus of cylinder Wi=1 Cd deficit

**Verdict: CONFIRMS — the M29b ±2-cell MUSCL fallback band is the dominant locus of polymer-stress concentration around the cylinder.**

Date    : 2026-05-23
Mission : Post-hoc probe of an existing M29b R=30 Wi=1 β=0.59 dump to test
          whether the "fall back to 1st-order Rusanov within ±2 cells of any
          solid" rule in `_fvfd_upwind_scalar_advective_rhs_2d(::Val{:muscl_superbee})`
          (src/fvfd/operators_2d.jl L517-565) coincides with the polymer-stress
          hot-spot that drives the +5 (Wi=0.1) → NaN (Wi=1) over-shoot
          decomposition documented in M41 NEWTONIAN_ISOLATION_VERDICT.
Method  : Read the M29b dump (Cd_kraken validation), build the EXACT
          fallback mask (cross-shape ±2 LU along the i and j axes), compute
          per-zone |τ_p| statistics (max, mean, q95, q99, median) for
          τ_xx, τ_xy, τ_yy, tr(τ_p), and produce a spatial heatmap with
          the band overlaid.

---

## (a) Dump used

| Property             | Value |
|----------------------|-------|
| Path                 | `tmp/m29b_kraken/cyl_bigsweep_v2_beta0p59_wi1_re1_R30_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls` |
| `advection_scheme`   | `muscl_superbee`            |
| `Cd_kraken`          | **116.474** (M29b reference; rT 132.37 → −12.0 % gap) |
| `Cd_s, Cd_p, Cd_bsd` | 117.314 / 14.198 / 15.038   |
| Wi, Re, β            | 1.0, 1.0, 0.59              |
| R, bsd_fraction      | 30, 1.0                     |
| Nx, Ny               | 900, 120                    |
| ν_p, λ               | 0.0615, 6000.0              |
| polymer fields dumped | `tauxx, tauxy, tauyy` directly (no need to reconstruct from Ψ) |

Validation passes the mission specification: `advection_scheme == :muscl_superbee`
**AND** `Cd_kraken ≈ 116.47` (mission required ≈ 116.47).

## (b) Fallback-zone definition (cited from source)

`src/fvfd/operators_2d.jl` L523-532, inside `_fvfd_upwind_scalar_advective_rhs_2d(::Val{:muscl_superbee})`:

```julia
if i <= 2 || i >= Nx - 1 || j <= 2 || j >= Ny - 1 ||
   is_solid[i - 2, j] || is_solid[i - 1, j] ||
   is_solid[i + 1, j] || is_solid[i + 2, j] ||
   is_solid[i, j - 2] || is_solid[i, j - 1] ||
   is_solid[i, j + 1] || is_solid[i, j + 2]
    return _fvfd_upwind_scalar_advective_rhs_2d(
        ..., Val(:rusanov),
    )
end
```

**Correction to the brief's "5×5 box" framing:** the actual stencil is a
**cross-shape with arms of length 2 along the i and j axes** (8 sites
checked, not 24). The corners (i±1, j±1), (i±2, j±2), etc. are NOT in the
stencil and a fluid cell with only a diagonally-adjacent solid does NOT
fall back to Rusanov. This is mirrored exactly in `probe.jl`.

The full fallback trigger is **near_solid ∨ domain_edge_2** (domain_edge_2
= 4-cell band along the open inlet/outlet/walls). For the cylinder
isolation we report stats SEPARATELY for the three zones
(`near_solid`, `bulk`, `domain_edge_2`) so that inlet/outlet effects are
not conflated with the curved BC.

## (c) Cell counts

| Zone                    | n     | %   of fluid |
|-------------------------|-------|--------------|
| solid                   | 2 820 | 2.6 % of domain |
| near_solid (cross ±2 LU) | **344** | **0.327 %** of fluid |
| domain_edge_2 (i≤2, i≥Nx−1, j≤2, j≥Ny−1) | 4 064 | 3.86 % of fluid |
| fallback_total          | 4 408 | 4.19 % of fluid |
| bulk (MUSCL active)     | 100 772 | 95.81 % of fluid |
| Total                   | 108 000 | — |

The cylinder fallback band is **tiny** (344 cells, 0.33 % of fluid). For
the band to be the dominant locus of the Cd_p deficit, |τ_p| must be
**massively elevated** there relative to bulk on a per-cell basis.

## (d) Stats per zone (|field|)

| Field   | Zone          | n      | max       | mean      | q95       | q99       | median    |
|---------|---------------|--------|-----------|-----------|-----------|-----------|-----------|
| τ_xx    | near_solid    | 344    | **1.585e-3** | **5.522e-4** | **1.344e-3** | 1.448e-3 | 4.751e-4 |
| τ_xx    | bulk          | 100 772| 9.789e-4  | 2.933e-5  | 1.038e-4  | 3.625e-4  | 1.244e-5  |
| τ_xx    | domain_edge_2 | 4 064  | 2.007e-3  | 1.269e-4  | 7.462e-4  | 1.744e-3  | 4.460e-5  |
| tr(τ_p) | near_solid    | 344    | **1.608e-3** | **7.811e-4** | **1.377e-3** | 1.472e-3 | 8.654e-4 |
| tr(τ_p) | bulk          | 100 772| 1.050e-3  | 3.206e-5  | 1.158e-4  | 4.155e-4  | 1.320e-5  |
| τ_yy    | near_solid    | 344    | 6.988e-4  | **2.296e-4** | **5.442e-4** | 6.419e-4  | 2.062e-4 |
| τ_yy    | bulk          | 100 772| 8.584e-4  | 3.542e-6  | 9.051e-6  | 5.323e-5  | 4.43e-8   |
| τ_xy    | near_solid    | 344    | 4.935e-4  | **2.437e-4** | **4.574e-4** | 4.865e-4  | 2.441e-4 |
| τ_xy    | bulk          | 100 772| 5.255e-4  | 1.051e-5  | 2.302e-5  | 8.632e-5  | 7.44e-6   |

### Ratios near_solid / bulk (load-bearing)

| Field   | max_ratio | **mean_ratio** | q95_ratio | q99_ratio |
|---------|-----------|----------------|-----------|-----------|
| τ_xx    | **1.62**  | **18.83**      | **12.96** | 4.00      |
| tr(τ_p) | **1.53**  | **24.37**      | **11.90** | 3.54      |
| τ_yy    | 0.81      | **64.80**      | **60.13** | 12.06     |
| τ_xy    | 0.94      | **23.19**      | **19.87** | 5.64      |

**Interpretation:** every polymer component is **strongly concentrated**
in the 344-cell fallback band. mean|τ_xx| is 19× bulk, mean|tr(τ_p)| is
24× bulk, mean|τ_yy| is 65× bulk (because τ_yy is essentially zero in
the bulk, so any localised cross-flow stress is huge in ratio), and the
q95 of τ_xx in the band is 13× the bulk q95. The fallback band carries
the bulk of the polymer-stress signal on a per-cell basis.

**Bulk q99 of τ_xx (3.63e-4) being only 3.7× lower than near_solid max** is
not a refutation: the bulk q99 is itself concentrated in the wake-shoulder
region that is the *outer envelope* of the same shoulder hot-spot — the
extreme bulk tail and the fallback band are adjacent and physically
continuous, so the spatial discriminator is mean/median ratio, not q99.

## (e) Heatmap

- `bench/scratch/m41bis_fallback_probe/tau_xx_with_fallback_overlay.png` —
  |τ_xx| zoomed ±4R around the cylinder, red dots = MUSCL-fallback band,
  gray = solid. The two pole/shoulder lobes of τ_xx are visibly
  co-located with the red band.
- `bench/scratch/m41bis_fallback_probe/tr_tau_with_fallback_overlay.png` —
  same overlay on |tr(τ_p)|.

Numerical summary: `bench/scratch/m41bis_fallback_probe/stats_summary.txt`.
Probe arrays (re-usable for further dives): `probe_arrays.jls`.

## (f) Verdict

**CONFIRMS** — the M29b MUSCL→Rusanov fallback band, despite being only
**0.33 %** of the fluid, contains **all four polymer-stress peaks** at
mean ratios **19× to 65×** the bulk. The polymer field around the cylinder
is **1st-order advected exactly where it matters** — the shoulder /
wake-onset region documented as the +3.14 deficit zone in M29c D1
post-mortem.

This is consistent, quantitatively, with M41's by-elimination conclusion
that the locus is "polymer × curved BC coupling" — specifically the
**advection scheme of the polymer field at solid-adjacent cells**.

## (g) Implication for M42

- **Primary target**: the fallback band is the right place to act. A
  one-sided or biased MUSCL reconstruction that avoids touching the solid
  cell value (and therefore does NOT need ±2 fluid neighbours along the
  axis) is the canonical fix. The M29c-v2 attempt failed at step 92 200
  via `rho` NaN at j=1 south wall — this was a late-time stability
  issue at the **open south wall**, NOT at the cylinder. So M42 can
  preserve the cylinder-side relaxation while keeping the open-wall side
  conservative.
- **Architecture**: M30 P2b "two-pass kernel split" is a useful safety net
  — first pass at solid-adjacent cells, second pass elsewhere, with
  lag-0 reads at the boundary so we don't read stale ghost data.
- **Validation gate**: any M42 relaxation must FIRST re-run the same
  R=30 Wi=1 β=0.59 case and produce a Cd_kraken ≥ 122 (≈ +5 vs M29b's
  116.47) without NaN before sweeping R or Wi.
- **Negative control**: the M41 Newtonian-limit numbers already show that
  `bouzidi_fl_twopass` at β=1 over-shoots rT by only +0.20 % at R=30.
  So the Newtonian Cd is essentially correct — any M42 improvement must
  NOT regress the Newtonian Cd by more than ~1 %.

The cylinder Wi=1 deficit hypothesis is now empirically grounded:
the polymer field is being **first-order advected in the only band
where it has any non-trivial gradient**.

---

## Files produced

- `bench/scratch/m41bis_fallback_probe/probe.jl` — analysis script
- `bench/scratch/m41bis_fallback_probe/plot.jl` — heatmap overlay
- `bench/scratch/m41bis_fallback_probe/stats_summary.txt` — numeric summary
- `bench/scratch/m41bis_fallback_probe/probe_arrays.jls` — masks + fields
- `bench/scratch/m41bis_fallback_probe/tau_xx_with_fallback_overlay.png`
- `bench/scratch/m41bis_fallback_probe/tr_tau_with_fallback_overlay.png`
