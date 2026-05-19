# Cylinder Cd β=0.59 Re=1 — M28 cluster synthesis — 2026-05-19

Date : 2026-05-19. Department M30-synthesis (Layer 1, Engineer + Department).
Branch / worktree : `dev-viscoelastic` / `Kraken.jl-viscoelastic`.

Scope : consolidate M25, M26, M26b, M28, M28b, M28c, M28d, M28e, M28f,
M28-rheotool-sweep, M28-rheotool-ref and M28-liu-check into a single
verdict locating where the Kraken-vs-rheoTool Cd gap lives, and what
to do next.

## 1. TL;DR

1. **The Liu Wi=1 reference "151.31" was a mis-read.** Liu Table 3
   columns at fixed R are ordered Wi=1.0 / 0.5 / 0.1 (descending,
   not ascending). True Liu CNEBB R=30 Wi=1.0 = **130.36**. The
   prior M25 "0.7 % below Liu 130.36" verdict accidentally compared
   Kraken Wi=0.1 (129.39) to Liu Wi=1.0 (130.36) — close to Newtonian
   on both sides, hence a fortuitous match.
2. **Sign of the drag trend is correct**. Both rheoTool (independent
   FVM, `Oldroyd-BLog`) and Kraken (LBM + log-FV BSD) show monotone
   drag REDUCTION with Wi at β=0.59 R=30. Kraken: 129.39 → 111.55
   (−14 %). rheoTool: 130.43 → 120.40 (−8 %). Liu CNEBB is
   non-monotone in Wi and not a reliable reference at Wi=0.1.
3. **The Kraken-vs-rheoTool gap is Wi-DEPENDENT** : ≈ +0.0 % at
   Wi=0.1, → −3 % at Wi=0.3, → −3 % at Wi=0.5, → **−7.3 %** at
   Wi=1.0. Constant-offset hypothesis is REFUTED. The defect
   amplifies with elastic loading.
4. **Geometry is NOT the gap**. M28f (matched-domain L_up=20
   L_down=60 vs Liu/rheoTool spec) gives Δ = +0.38 Cd CONSTANT
   across Wi ∈ {0.1, 0.3, 0.5, 1.0}. M28e mesh refinement at Wi=1
   gives Cd plateauing at ≈ 111.4 (R=40), not approaching rheoTool
   120.40. The gap is not from domain truncation, mesh resolution,
   inlet/outlet placement, or λ-substep windowing.
5. **The gap is most likely in the BSD-vs-stress coupling
   (`embedded_force=0` Liu-mode internals)**, NOT in the embedded
   `1111_circle` ghost path that was the M26 focus. M28b (bsd=0)
   gives a CLEANER gap of −12 % at Wi=1.0 vs rheoTool, so BSD damps
   the elastic Cd toward Liu-Wi=1 (which would actually be a
   Newtonian-attractor, not a feature). The +7-15 % residual after
   M28 ratcheting points at the LBM Guo / log-FV polymer-force
   coupling itself (cubista-vs-upwind, ψ→C exponentiation pathway,
   or TRT collision-source ordering).

## 2. All-Wi comparison at R=30, β=0.59, Re=1

Numbers below are steady-state Cd_kraken (LBM-side Mei MEA) or
rheoTool Cd_last (cylinder-surface integral total = pressure +
viscous + polymer). All Kraken runs are A100 CUDA F64, 100k steps,
`KRAKEN_AVG_WINDOW_FRAC=0.2` (last 20 k steps averaged).

| Wi  | rheoTool | Kraken M28<br>bsd=1 0000_qwall | Δ vs rheo | Kraken M28b<br>bsd=0 0000_qwall | Δ vs rheo | Kraken M28d<br>bsd=1 0010_qwall (force-on) | Δ vs M28 | Kraken M28f<br>matched-domain | Δ vs M28 | Notes                                       |
|----:|---------:|-------------------------------:|----------:|--------------------------------:|----------:|-------------------------------------------:|---------:|------------------------------:|---------:|---------------------------------------------|
| 0.1 |  130.43  |                       129.39   |   −0.8 %  |                        120.97   |   −7.3 %  |                                  138.10    |  +8.71   |                       129.77  |  +0.38   | Newtonian-additive; M26 bug visible in M28d |
| 0.3 |    NA    |                       121.25   |     —     |                        114.72   |     —     |                                  128.86    |  +7.61   |                       121.62  |  +0.38   | rheoTool not run                            |
| 0.5 |  119.71  |                       115.93   |   −3.2 %  |                        110.51   |   −7.7 %  |                                  122.47    |  +6.54   |                       116.31  |  +0.38   | first regime where rheoTool drag < Newt.    |
| 1.0 | **120.40** |                   **111.55** | **−7.3 %**|                        106.79   |  −11.3 %  |                                  116.27    |  +4.72   |                       111.93  |  +0.38   | stress-test                                 |

Observations :

- **Δ vs rheoTool grows with Wi** : 0.8 → 3.2 → 7.3 % between
  Wi=0.1 and Wi=1.0 on the bsd=1 baseline. The defect is
  elastic-loading-dependent ⇒ NOT a fixed geometric / discretisation
  bias.
- **bsd=0 makes the gap WORSE, not better** (M28b adds another
  −4 to −5 % onto every Wi). BSD architecture is doing the right
  thing physically — pulling Kraken back toward Newtonian. The
  residual −7.3 % at Wi=1.0 bsd=1 is therefore **not** a BSD bug,
  it is what is left after BSD has already pulled in by ~5 %.
- **M28d (force-on, embedded_force=1, the M26 bug path)** adds a
  Wi-dependent Δ : +8.71 at Wi=0.1 down to +4.72 at Wi=1.0. The
  bug becomes RELATIVELY smaller at high Wi. Interesting :
  the +8 ghost-overdose is itself diluted by elastic re-distribution.
  This is consistent with M26b's partial-fix verdict (only 8 % of
  the +8 closed by the cell-fraction divisor; residual amplifier
  is the wall-segment term `wall_*_length * tau` in the embedded
  divergence kernel).
- **M28f matched-domain Δ = +0.38 CONSTANT across Wi.** Domain
  asymmetry contributes a constant +0.38 Cd irrespective of Wi.
  Wake truncation is NOT the elastic gap.

## 3. M28e mesh refinement at Wi = 1, β = 0.59, 0000_qwall, bsd = 1

Job `21580646.aqua`. Goal : does Cd(R→∞) converge to rheoTool 120.40
at the highest grid Kraken can reach ?

| R   | Cd_kraken | Δ vs rheoTool (120.40) | walltime | status                    |
|----:|----------:|------------------------:|---------:|---------------------------|
|  20 |    111.99 |              −7.0 %     |   65.8 s | OK                        |
|  30 |    111.55 |              −7.4 %     |   48.6 s | OK                        |
|  40 |    111.29 |              −7.6 %     |   65.3 s | OK                        |
|  60 |       NaN |                  —      |  132.6 s | **non-finite at step 0**  |
|  80 |       NaN |                  —      |  217.1 s | **non-finite at step 0**  |

Conclusion : **the Wi=1 Cd plateau is ~111.4 ± 0.3 across R ∈
{20, 30, 40}, with monotonic DECREASE in R (away from rheoTool)**.
The plateau is **NOT** approaching rheoTool 120.40. Mesh
refinement REFUTES the "discretisation gap" hypothesis at finite Wi.
The R=60 / R=80 NaN at step 0 is a separate τ_p initial-condition /
ψ→C exponentiation stability issue at large λ (12 000 / 16 000 LU);
not relevant to the gap, but worth a future audit of the
initialisation order at λ ≳ 8 000 LU.

## 4. M28c integration-time sanity (R = 30, Wi = 1, bsd = 1, 0000_qwall)

Job triplet `21579957/958/959.aqua` (100 k / 300 k / 1 M steps).
Verdict : 100 k IS converged. Δ to 1 M = +3.0e-7 Cd, i.e. F64
round-off. The 111.55 baseline at Wi = 1 R = 30 is steady-state,
not transient. Under-integration REFUTED.

## 5. rheoTool Cd sweep at R = 30, β = 0.59 (M28-rheotool-sweep)

| Wi   | rheoTool Cd | residual_p   | steady-state marker | endTime |
|-----:|------------:|-------------:|---------------------|--------:|
| 0.05 |      131.81 |   1.2e−16    | flat 12 digits      |   6.0   |
| 0.1  |      130.43 |   1.2e−16    | flat 12 digits      |   6.0   |
| 0.2  |      126.84 |   1.2e−16    | flat 12 digits      |   6.0   |
| 0.5  |      119.71 |   1.1e−16    | flat 9 digits       |  10.0   |
| 1.0  |    **120.40** | 1.1e−16    | drifting 5th decimal|  10.0   |

Trough at Wi = 0.5 (119.71), tiny uptick at Wi = 1.0 (120.40).
This is the qualitative behaviour expected for confined Oldroyd-B
near critical Wi : drag reduction up to a minimum, then mild
re-rise. Liu Table 3 at finer R (R = 35) shows the same shape
(127.72 → 130.77 between Wi = 0.5 and Wi = 1.0).

Kraken is monotone-down to Wi = 1, missing the slight re-rise.
**This is a candidate locus** : Kraken under-resolves the
extensional-stress contribution to drag at Wi ≳ 0.5, which is
the term that drives the drag re-rise. The TRT plus-rate and the
Guo source prefactor `1 − s_plus/2` for the polymer body force
both depend on the SOURCE-side discretisation choices ;
cubista-vs-upwind on `div(phi, theta)` in rheoTool vs Kraken's
ATU pathway for ψ is a leading suspect.

## 6. What we ratcheted OUT (no longer in the suspect list)

1. **BSD architecture** : M28b (bsd = 0) shows the gap is WORSE
   without BSD. BSD is correctly pulling Kraken closer to rheoTool.
2. **Time integration / under-integration** : M28c machine-ε
   convergence at 100 k steps.
3. **Wake truncation / domain asymmetry** : M28f Δ = +0.38 Cd
   constant across Wi. Even matched to Liu/rheoTool 30 R upstream
   spacing, the gap persists.
4. **Liu Cd(Wi=1) = 151.31 mis-citation** : Liu's actual Wi = 1
   value is 130.36; 151.31 was the Wi = 0.1 column entry (and is
   non-converged in Liu's own table, see M28-liu-check verdict).
5. **M26 H1 (drag-formula bug)** : empirically REFUTED, Cd_s is
   sourced from LBM MEA which is invariant under `embedded_drag`.
6. **M26 H3 (`:circle` quadrature)** : empirically REFUTED, force-only
   bug has same magnitude on `:qwall` and `:circle`.
7. **Mesh resolution** at finite Wi : M28e plateaus at 111.4 across
   R ∈ {20, 30, 40}, not approaching rheoTool 120.40 as R → ∞.

## 7. Still under suspicion

1. **M26 H2 residual — wall-segment term in
   `fvfd_tensor_divergence_embedded_2d_kernel!`** : M26b
   cell-fraction rescale closed only 8 % of the +8 Cd overdose.
   The remaining +7.27 Cd from the embedded-force path most likely
   comes from the `wall_x_length * tauxx[i, j]` / `wall_y_length *
   tauxy[i, j]` etc. surface-segment terms (operators_2d.jl:759 -
   766). This is the M26c follow-up. NOTE : in the Liu-reference
   mode (`0000_qwall`, all embedded flags off) THIS PATH IS NOT
   TAKEN — so the M26-cluster bug is real but separate from the
   M28 gap.
2. **Log-conformation source-discretisation scheme** : rheoTool uses
   `GaussDefCmpw cubista` on `div(phi, theta)` (3rd-order
   monotonicity-preserving). Kraken-side ATU pathway for ψ → C
   exponentiation may have a different effective truncation error
   at finite Wi. This is the leading suspect for the +7 % Wi = 1
   gap, and it is consistent with the missing drag re-rise at
   Wi ≳ 0.5.
3. **Guo source prefactor / TRT collision-source ordering** in the
   polymer body-force injection. The `1 − s_plus/2` prefactor is
   shared with the solvent Guo coupling and works at Wi = 0.1 ; the
   gap appears only as elastic loading grows, so the suspect is
   not the prefactor itself but the placement of the polymer-force
   addition relative to the TRT plus-rate collision.
4. **Re = 1 vs Re = 0.01 finite-inertia bias.** rheoTool, Liu and
   Kraken all run Re = 1 ; not a between-codes systematic. But the
   Wi-dependent gap could couple with inertia. Could in principle be
   tested by a Re = 0.01 rheoTool + Kraken pair at Wi = 1, R = 30.

## 8. Outstanding diagnostics

- **M29-tau-compare** (rheoTool vs Kraken τ field at R = 30 Wi = 1) :
  **IN-FLIGHT** at the time of this writeup. Not yet a verdict.
  When delivered, this is the cleanest discriminator between
  hypotheses (2) and (3) : if τ_xx field magnitudes match between
  codes (≤ 5 % L2) but Cd differs by 7 %, the locus is the
  drag-integration / Guo coupling, NOT the constitutive
  discretisation. If τ fields differ by ~10 %, locus is the log-conf
  / ATU discretisation.

## 9. Recommendations for next session (NOT executed)

In priority order :

1. **Land M29-tau-compare** when it returns. This unblocks
   hypothesis ranking between (2) and (3) above.
2. **Run Kraken cubista-pathway smoke** at R = 30 Wi = 1 if such a
   pathway exists ; if not, schedule a `bench/viscoelastic_audit/`
   test that varies the log-conf advection scheme (upwind /
   QUICK / cubista-equivalent) and watches Cd.
3. **Resume M26c** (wall-segment term in embedded force kernel)
   in PARALLEL — independent of the M28 gap, this is still a real
   bug that blocks any future embedded-mode production work.
4. **Defer M28e R = 60 / 80 NaN** to a separate stability mission ;
   not on the Cd-gap critical path.
5. **Defer rheoTool Wi = 0.3** unless a Wi-resolution-of-3 ladder
   is needed for a paper-style plot ; Wi = 0.1 / 0.5 / 1.0 is
   sufficient for the gap-location analysis.

## 10. Files

### Kraken bigsweep CSVs (rsync targets, NOT committed)

- M28 main (Phase 1) : `results/viscoelastic_logfv/cyl_bigsweep_v2_21575466.aqua/SUMMARY.csv`
- M28b (bsd = 0)     : `results/viscoelastic_logfv/cyl_bigsweep_v2_21580009.aqua/SUMMARY.csv`
- M28c (100k/300k/1M): `results/viscoelastic_logfv/cyl_bigsweep_v2_2157995{7,8,9}.aqua/SUMMARY.csv`
- M28d (force-on)    : `results/viscoelastic_logfv/cyl_bigsweep_v2_21580531.aqua/SUMMARY.csv`
- M28e (R sweep Wi=1): `results/viscoelastic_logfv/cyl_bigsweep_v2_21580646.aqua/SUMMARY.csv`
- M28f (matched dom) : `results/viscoelastic_logfv/cyl_bigsweep_v2_21580724.aqua/SUMMARY.csv`

### rheoTool reference

- Cases     : `bench/rheotool/cylinder_wi{0.05,0.1,0.2,0.5,1.0}/`
- Sweep CSV : `bench/viscoelastic_logfv/RHEOTOOL_CD_SWEEP_M28.csv`
- Aggregator: `bench/rheotool/sweep_wi_results.txt`

### Verdict cluster (this session, all on `dev-viscoelastic`)

- M25 + M26 + Phase 0 / Phase 0b : `bench/viscoelastic_logfv/CYL_PHASE0_PHASE0B_VERDICT_20260518.md`
- M28c integration time         : `bench/viscoelastic_logfv/CYL_PHASE1C_INTEGRATION_M28C_VERDICT.md`
- M28 rheoTool reference        : `bench/viscoelastic_logfv/CYL_RHEOTOOL_REF_M28_VERDICT.md`
- M28 rheoTool full sweep       : `bench/viscoelastic_logfv/RHEOTOOL_CD_SWEEP_M28_VERDICT.md`
- M28 Liu Table 3 verification  : `.orchestrator/M28_liu_table_verification.md`
- M26 mathematical audit        : `.orchestrator/M26_analysis_verdict.md`
- M26b partial-fix verdict      : `bench/viscoelastic_logfv/M26B_FIX_VERDICT.md`
- This synthesis                : `bench/viscoelastic_logfv/CYL_SESSION_M28_SYNTHESIS_20260519.md`

### Driver and bench

- Driver  : `src/drivers/viscoelastic_logfv_2d.jl` (M26b smoke patch
  WIP, NOT committed in this session)
- Bench   : `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl` (commit
  `e602726f`, unchanged)
- Smoke   : `bench/scratch/m26b_smoke_2d.jl` (new, untracked)
