# M29c-tau-decompose — Verdict (PARTIAL — polymer carries most of the gain)

Date  : 2026-05-19
Branch: dev-viscoelastic
Mission: decompose M29b vs M29c-v2 vs rheoTool field comparison into
         u, tau_s = mu_s (grad u + grad u^T), tau_p, and test the
         falsifiable claim that "the entire M29c-v2 improvement is
         carried by tau_p; u and tau_s do not change".

## Status: EXECUTED — verdict is PARTIAL

Re-run by Department `M29c-tau-decompose-exec`. One minimal fix applied to
`run_decompose.jl` (de-referencing `rheo_samp[]` inside `process_one`; the
prior author referenced the module-level `Ref{Any}` rather than the held
NamedTuple). No other changes.

Artefacts produced this run:

| Artefact                                                              | Present | Notes                                          |
|-----------------------------------------------------------------------|---------|------------------------------------------------|
| `bench/scratch/m29c_tau_decompose/run_decompose.jl`                   | YES     | +5 lines (one-line `rs = rheo_samp[]` alias, 6 access rewrites) |
| `bench/scratch/m29c_tau_decompose/M29DEC_decomposition.csv`           | YES     | 8 fields x 2 snapshots = 16 rows               |
| `bench/scratch/m29c_tau_decompose/M29DEC_uy_bands.csv`                | YES     | 4 bands x 2 snapshots = 8 rows                 |
| `.engineer_logs/M29c-tau-decompose-exec_20260519_*.log`               | YES     | Stdout captured                                 |

Inputs used (all present, unchanged): JLS snapshots
`21588714.aqua_m29b_30k_f64/result_m29b_30k_f64.jls` and
`21588713.aqua_m29c_v2_30k_f64/result_m29c_v2_30k_f64.jls`; rheoTool case
`bench/rheotool/cylinder_wi1.0`, time `10`. ROI x in [-3, 8], y in [-1.9, 1.9],
256 x 128 grid, 30206 valid samples (92.2%). beta = 0.59, Re_R = 1, mu_s_phys = 0.59.

Configuration confirmed: Nx=900, Ny=120, R=30, cx=450, cy=59.5. M29b
scheme=rusanov Cd=110.227. M29c-v2 scheme=muscl_superbee Cd=115.898.

## Decomposition table

```
field      M29b L2_rel   M29c-v2 L2_rel   Delta L2_rel   Delta %    peak_R
u_x        0.0860        0.0864           +0.0004        +0.44 %    3.287
u_y        0.1472        0.1613           +0.0141        +9.59 %    0.9455
tau_s_xx   0.2698        0.2680           -0.0018        -0.67 %    5.415
tau_s_xy   1.0233        1.0221           -0.0012        -0.12 %    25.82
tau_s_yy   1.0291        1.0286           -0.0005        -0.05 %    22.57
tau_p_xx   0.6223        0.4965           -0.1257        -20.21 %   125.9
tau_p_xy   0.6093        0.3785           -0.2308        -37.88 %   47.89
tau_p_yy   0.5998        0.3730           -0.2268        -37.81 %   49.37
```

Notes on tau_s: both Kraken runs UNDER-resolve the magnitude of tau_s,xy
and tau_s,yy versus rheoTool (peak_K ≈ 7 vs peak_R ≈ 25; peak_K ≈ 5 vs
peak_R ≈ 22), which is why their L2_rel sits at ~1.02 in both. This is a
shared baseline gradient-fidelity gap independent of the M29b -> M29c-v2
transition, and the relevant question (does the gap CHANGE?) answers
flat: < 1 % drift in all three solvent components.

Cross-check vs prior M29c_tau_compare (rolled-up tau): the prior compare
mission reported L2_rel(tau_xy) 0.609 -> 0.379 (-37.7 %) and L2_rel(tau_yy)
0.600 -> 0.373 (-37.9 %). The new tau_p_xy = -37.88 % and tau_p_yy =
-37.81 % match those numbers to within rounding, confirming that the
tau-field improvement seen at the compare stage was entirely the polymer
sub-tensor.

## u_y band breakdown

```
x/R band       M29b L2rel   M29c-v2 L2rel   Delta      n_samples
[-3.0, -1.0]   0.1163       0.1210          +0.0047    6016    (upstream)
[-1.0,  1.0]   0.2092       0.2318          +0.0226    3345    (near body)
[ 1.0,  3.0]   0.0998       0.1143          +0.0145    5997    (near wake)
[ 3.0,  8.0]   0.2219       0.0830          -0.1390    14720   (far wake)
```

The +9.59 % u_y degradation is concentrated near the body and in the
upstream/near-wake region; the far wake (where 49 % of valid samples
live) IMPROVES by -0.14 absolute (-62.7 %). Sign of the integral u_y
displacement is therefore not monotone — band-wise reading flips at
x/R ~ 3. Hypothesis: the additional MUSCL-superbee dissipation on Psi
slightly sharpens the polymer wake (so far-wake tau_p backflow is
better tracked) at the cost of a small polymer-momentum recirculation
near the cylinder shoulder. Consistent with Cd going UP +5.1 % (more
polymer drag transfer to the body) while wake-Cd_p relaxation
improves.

## Verdict thresholds

| Threshold                                                         | Value   | Pass? |
|-------------------------------------------------------------------|---------|-------|
| u L2_rel max-component drift <= 5 %                               | 9.59 %  | NO    |
| tau_s L2_rel max-component drift <= 5 %                           | 0.67 %  | YES   |
| tau_p L2_rel max-component drop >= 15 %                           | 37.88 % | YES   |

**Outcome: PARTIAL.** The polymer-only hypothesis is FALSIFIED on the
u_y component (+9.59 % drift, above the 5 % gate), but two of three
sub-claims pass cleanly: tau_s is unchanged (< 1 % drift on all three
components) and tau_p drops sharply (-20 to -38 % on its three
components). The u_y degradation is a real residual signal, not
numerical noise — it is concentrated in the near-body / upstream
region and is REVERSED in sign in the far wake.

## Boss decision implication

- A pure "M29d BSD = 0" experiment is well-posed for the **tau_p**
  question: tau_s is established as a non-actor (< 1 % across all
  three components), so any further M29d run that improves
  tau_p L2_rel by another 15 %+ is correctly attributed to the
  polymer model regardless of BSD.
- But "BSD = 0" cannot be assumed sufficient to also explain u_y, since
  9.6 % drift slipped through the gate. M29d should report u_y
  band-wise (4 bands as above) so the near-body / far-wake sign flip is
  not hidden by global averaging.
- Cd accounting: M29c-v2 Cd = 115.9 vs rheoTool ~ 130.8 (12 % gap)
  cannot be closed by the polymer-only narrative alone. Roughly one
  third of the remaining gap is plausibly velocity-coupling (the
  near-body u_y residual that survived this decomposition); two
  thirds is residual polymer-stress underprediction (peak_K =
  66 vs peak_R = 126 on tau_p_xx).

## Memory candidates

1. `feedback_decompose_partial_uy_residual` — When testing a
   "polymer-only" hypothesis in viscoelastic LBM, the velocity field
   itself can drift several percent under a scheme change even if
   the solvent stress (which is grad-u based) does not, because the
   gradient sampler is the SAME for both M29b and M29c-v2 and
   damps high-frequency content equally. The velocity drift therefore
   cannot be diagnosed via solvent stress; it needs a direct u-component
   band breakdown.
2. `feedback_band_breakdown_reveals_sign_flip` — A 9.6 % global L2
   degradation in u_y can hide a far-wake IMPROVEMENT (-62.7 %)
   coexisting with near-body deterioration (+10.8 %). Band-wise
   L2_rel along x/R must be the default reporting unit in the wake
   benchmark family, NOT a global scalar.
3. `feedback_minimal_fix_ref_dereference` — Module-level
   `const x = Ref{Any}(...)` accessed without `[]` inside a nested
   function returns the Ref struct (with field `.x`), not the held
   value. Symptom: `FieldError: type Base.RefValue has no field
   <yours>`. Always pass through an explicit `local = x[]` alias at
   function head.
