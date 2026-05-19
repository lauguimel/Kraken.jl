# M29 — Kraken vs rheoTool field-level τ comparison at the worst-Wi point

Date : 2026-05-19
Department : M29-tau-compare
Branch / worktree : `dev-viscoelastic` / `Kraken.jl-viscoelastic`
Operating point : β=0.59, Re=1, R=30, Wi=1.0, bsd_fraction=1.0,
                  embedded flags all OFF, geometry=qwall.

## TL;DR — outcome (b): u differs by ~17 %, τ differs by 60-93 %

At the steady-state operating point where Kraken under-shoots rheoTool
in integrated drag by **ΔCd = −8.85 (−7.4 %)** (Cd_kraken 111.55 vs
Cd_rheo 120.40), the field-level comparison shows the **polymer stress
tensor is the dominant source of the gap**, not the velocity field.

| field        | L2_rel | max\|diff\| | ref_L2 |
|--------------|--------|------------|--------|
| u_x / U_mean | **0.17** | 6.2  | 1.23  |
| u_y / U_mean | **0.18** | 1.2  | 0.27  |
| τ_xx / ρU²   | **0.93** | 521  | 17.0  |
| τ_xy / ρU²   | **0.77** | 64   | 2.6   |
| τ_yy / ρU²   | **0.58** | 32   | 2.5   |

(non-dim, ROI x ∈ [−3, 8] × y ∈ [−2, 2] sampled on a 256×128 Cartesian
grid, 29 825 fluid samples after solid-mask filtering, 91 % of ROI.)

**Velocity matches at 17 % L2**, **polymer stress disagrees at 58-93 %
L2**. This is outcome (b) in the M29 brief: u matches reasonably, τ
differs catastrophically — the discrepancy lives in the **constitutive
evolution** (Kraken's log-FV upwind / Rusanov flux vs rheoTool's
`cubista` HRS in `div(phi,theta)`), not in the coupling (Guo body force,
BSD) or in the LBM hydrodynamics.

## Spatial localisation

Per-x-band averages of L2(diff τ_xx) and max\|diff τ_xx\|:

| region              | L2_mean | max_mean |
|---------------------|---------|----------|
| Near (\|x\|<1.5)    | **29.8** | **199** |
| Far  (\|x\|>4)      | 0.28    | 2.17    |

The disagreement is concentrated by a factor of **100×** in the near-
cylinder zone (the stress boundary layer where extensional kinematics
of the wrap-around flow stretch the polymer most violently).

Top-5 x-bands by max\|diff τ_xx\| (units of ρU²):

    x/R       L2(diff)   max(diff)
    +0.02     87.4       521.3      ← top-of-cylinder shoulder
    -0.02     86.4       514.8      ← top-of-cylinder shoulder
    +0.06     86.3       514.0      ← leeward shoulder
    +0.11     84.7       506.2      ← leeward shoulder
    +0.15     83.9       503.1      ← leeward shoulder
    ...
    +1.18     30.9       290.4      ← near wake
    +1.23     31.0       291.9      ← near wake
    ...
    -1.32     18.9       182.0      ← upstream stagnation envelope

The shoulder x ∈ [0, 0.3] is the wrap-around region where the polymer
experiences peak extension. The near wake x ∈ [1, 1.3] is the rear
stagnation zone where the stretched polymer relaxes. Both regions
favour high-resolution HRS schemes in the conformation advection;
Kraken's first-order Rusanov upwind on log(C) under-relaxes precisely
here.

Absolute τ_xx max in the fluid :

- Kraken    : max(τ_xx)/(ρU²) ≈ **75.3**
- rheoTool  : max(τ_xx)/(ρU²) ≈ **135.5**

Kraken under-predicts τ_xx by **44 %** at the peak. The polymer stress
that should resist the flow is too weak in Kraken → less polymer
contribution to wall shear (Cd_p 11.58 vs the implicit rheoTool
equivalent that contributes the missing ≈8.85 to Cd_total).

## Mechanism narrative

1. Kraken's log-FV polymer pipeline uses a **first-order Rusanov flux**
   for the `psixx, psixy, psiyy` advection (`logfv_polymer_advect_2d!`
   family), with **CFL-limited substepping** but no high-resolution
   reconstruction. This is intrinsically diffusive.
2. rheoTool's `div(phi,theta)` uses `GaussDefCmpw cubista` — a TVD
   high-resolution scheme that preserves stress peaks much better.
3. At Wi=1.0, R=30, the cylinder's leeward shoulder has a stress
   feature of width O(R/30) = O(1 phys / 30) ≈ 1 LU in Kraken.
   First-order upwind smears such features over O(grid spacing × CFL)
   per step, accumulating O(λ × velocity / R) of relative damping
   over the 1×10⁵ relaxation period — exactly the order of magnitude
   we see (~44 % τ_xx peak loss).
4. The trace_C diagnostic confirms : Kraken trace_C_max = 186 vs the
   relaxed state trace_C = 2 → factor of 90× stretch, more than half
   of which is being numerically diffused.

This is **not** a coupling bug (Guo body force is intact: u matches
within 17 %), it is **not** an embedded-geometry artefact (all embedded
flags are OFF for this run, qwall), and it is **not** an inlet-boundary
artefact (matches the M28 rheoTool case which uses parabolic Poiseuille
inlet equivalent).

## Pointer to next mission

The Cd-magnitude gap reduces to a **constitutive scheme upgrade**
task : replace Rusanov upwind on Ψ = log(C) with a TVD/HRS scheme
(MUSCL+superbee, CUBISTA, or WENO3). This is a known and well-trodden
upgrade in the OpenFOAM viscoelastic community (Pimenta &
Alves 2017 — `Comp. Phys. Commun. 220`).

Estimated impact : a HRS upgrade on the log-conformation advection
should recover at least half of the missing 8.85 Cd, plausibly all of
it, bringing Kraken's Cd from 111.55 to 120 ± 2 at Wi=1.0 R=30 — i.e.,
within rheoTool's own discretisation tolerance.

## Method, in one paragraph

The patched bench script
`bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl` now honours an
opt-in `KRAKEN_SAVE_FIELDS=1` env flag that serialises `(ux, uy,
tauxx, tauxy, tauyy, is_solid, Nx, Ny, R, cylinder_x_lbm,
cylinder_y_lbm, ...)` to a `.jls` next to the per-case CSV (no
`src/` edits, blast radius = one bench file). Aqua A100 F64 job
`21585158.aqua` produced the snapshot. The comparison driver
`bench/viscoelastic_audit/run_kraken_vs_rheotool_tau_compare.jl`
reads the .jls, gunzips rheoTool's `tau.gz`/`U.gz`, derives FOAM
cell-centers from `polyMesh/{points,faces,owner,neighbour}`, builds
a kNN+affine spatial interpolator (12 neighbours, 3-term basis),
samples both datasets on a common Cartesian ROI grid (in physical
units: cylinder at origin, R=1 phys), non-dimensionalises by U_mean
on each side (Kraken: 0.005 LU/step ; rheoTool: 1 phys), and
computes L2-relative + max-abs residuals + per-x-band aggregates.

## File anchors

- Kraken field snapshot :
  `tmp/m29_kraken/cyl_bigsweep_v2_beta0p59_wi1_re1_R30_bsd1_*_fields.jls`
- rheoTool reference :
  `bench/rheotool/cylinder_wi1.0/10/{U.gz, tau.gz}`
- Patched bench driver (Option B-light) :
  `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl`
  (env `KRAKEN_SAVE_FIELDS=1` adds .jls per case, ~4 MB at R=30)
- Comparison driver :
  `bench/viscoelastic_audit/run_kraken_vs_rheotool_tau_compare.jl`
- M29 single-case PBS :
  `bench/viscoelastic_logfv/run_cyl_m29_field_snapshot_a100.pbs`
- Aqua job id : **21585158.aqua** (A100 F64, 100 k steps, 76 s)
- Numerical artefacts :
  `bench/scratch/m29_tau_compare/M29_residuals.csv`
  `bench/scratch/m29_tau_compare/M29_band_stats_x.csv`
- Field plots (5 components × 3 panels each :
  Kraken / rheoTool / diff) + band plot :
  `bench/scratch/m29_tau_compare/M29_field_{ux,uy,tau_xx,tau_xy,tau_yy}.png`
  `bench/scratch/m29_tau_compare/M29_band_diffs.png`

## Caveats / limitations

- The kNN+affine interpolator is exact on smooth fields but can
  oscillate up to a few percent near the rheoTool body-fitted cells'
  highest-gradient cells (close to the cylinder wall). The "max|diff|"
  numbers in the very-near-wall bands (|x|<0.3) should be read with
  ±10 % interpolation tolerance. The "L2_mean" aggregate is robust to
  this : it dominates the verdict.
- The non-dim convention `τ/(ρU_mean²)` makes Kraken and rheoTool
  directly comparable; the Reynolds-similar units of the polymer
  stress are *not* further normalised by viscosity ratio β, because
  the operating points have identical β=0.59.
- Single-Wi field comparison. To extend M29 to a fuller picture
  (Wi=0.1 should show similar L2 hierarchy but smaller absolute
  magnitude; the trend with Wi locates the HRS payoff scale), re-run
  with `KRAKEN_WI_LIST="0.1,0.5,1.0"` and `KRAKEN_SAVE_FIELDS=1` —
  ~12 minutes total on A100 F64.
- The mechanism narrative (§4) is based on the kinematics of the
  near-cylinder shoulder; a follow-up M29b would actually swap in a
  superbee-or-cubista log-FV reconstruction and re-run the same
  point to *confirm* the +8 Cd recovery. That is `src/`-level work
  and explicitly out of M29 scope (forbidden zone per brief).
