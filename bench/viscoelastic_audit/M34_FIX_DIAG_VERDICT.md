**Recommendation**: M34-second-bug

# M34-fix-diag — NaN spatial fingerprint triage (verdict)

**Date**     : 2026-05-22
**Mission**  : M34-fix-diag-finalize (write verdict from existing artifacts; no new computation)
**Branch**   : `dev-viscoelastic` (uncommitted)
**Status**   : **YELLOW — triage CLASSIFIED. NaN cause is BC-side residual over-bounce + amplification, NOT a polymer-scheme failure. M35 (Ψ-upgrade) remains PARKED; the next intervention is M34v3 (cut-link re-WriteMoments after pass-2).**

## Triage protocol applied

`[[feedback_nan_uniform_vs_arc_diagnostic]]` — per-case NaN spatial fingerprint via `bench/scratch/m34_fix_diag/triage.jl`. Reference field = `rho` when `psixx` is absent NaN-wise; classification thresholds: ≥0.9 → uniform, <0.30 with bilateral-arc dominance → bilateral-arcs, <0.30 other → localised-other, else mixed.

## Triage table (3 NaN cases + 1 clean sanity)

| Case               |  R | Wi  |  Nx × Ny  | rho_nan_frac | psi_nan_frac | classification |  pos_front-pole | pos_front-shoulder-arc | pos_wake | pos_other | inlet_col_nan | outlet_col_nan |
| ------------------ | -: | --: | :-------: | -----------: | -----------: | :------------: | --------------: | ---------------------: | -------: | --------: | ------------: | -------------: |
| R30_Wi1            | 30 | 1.0 |  900×120  |       0.9739 |        0.000 |    uniform     |             168 |                    347 |    47323 |     57258 |       120/120 |        120/120 |
| R40_Wi1            | 40 | 1.0 | 1200×160  |       0.9739 |        0.000 |    uniform     |             224 |                    440 |    84122 |    102084 |       160/160 |        160/160 |
| R60_Wi0p1          | 60 | 0.1 | 1800×240  |       0.9738 |        0.000 |    uniform     |             324 |                    633 |   189236 |    230321 |       240/240 |        240/240 |
| R30_Wi0p1 (clean)  | 30 | 0.1 |  900×120  |       0.000  |        0.000 | localised-other|               0 |                      0 |        0 |         0 |             0 |              0 |

NaN-mask heatmaps exist on disk for the three NaN cases:
`bench/scratch/m34_fix_diag/R30_Wi1_nan_mask.png`,
`R40_Wi1_nan_mask.png`,
`R60_Wi0p1_nan_mask.png`.

## Key empirical observations

1. **Ψ remains FINITE in all three NaN cases** (`psi_nan_frac = 0.000`) while `rho_nan_frac` ≈ 0.974. The polymer log-conformation field is innocent — the `f → ρ` chain diverges FIRST. This rules out a polymer-scheme failure as the proximate cause.

2. **NaN is UNIFORM (≥97 %) — not bilateral-arc.** The narrow `front-pole / front-shoulder-arc` bins hold only ~0.5 % of NaN cells each. The mass is in `wake` + `other`, i.e. the whole channel post-divergence. This is the signature of a divergence that *propagates through advection*, not a localised stress-pole singularity.

3. **Inlet AND outlet columns are FULLY NaN** (Ny cells each: 120/120, 160/160, 240/240). The blow-up has had time to reach both ends of the domain → the NaN happened well before snapshot time and the post-divergence f-field has fully propagated.

4. **Clean R30_Wi0p1 baseline is bit-exact zero NaN** — confirms the M34-fix RAW spec works for the low-(R,Wi) corner and that the divergent cases are not a generic regression.

## Coherent image (Cd + triage)

Combining the M34_FIX_VERDICT Aqua matrix outcome with this triage:

|  R | Wi  |        Cd_kraken         | Mechanism (single coherent root cause)                              |
| -: | --: | ------------------------ | ------------------------------------------------------------------- |
| 30 | 0.1 | 132.51 (+1.6 % vs rT 130.43) | residual BC over-bounce survives at cut-link cells                 |
| 40 | 0.1 | 133.46 (+2.3 % vs rT)        | over-bounce amplified by R (more cut-link cells)                  |
| 60 | 0.1 |   NaN                        | over-bounce × R crosses divergence envelope                       |
| 30 | 1.0 |   NaN                        | over-bounce × Wi (polymer back-force amplifies) crosses envelope  |
| 40 | 1.0 |   NaN                        | over-bounce × R × Wi → fastest divergence                          |

Single coherent root cause: **M34-fix still over-corrects at cut-link cells**. At low (R, Wi) the residual is a finite +1.6 % Cd bias; at high R or high Wi the cumulative amplification through the f-stream diverges `ρ` first, while `Ψ` (which is integrated in log-form with bounded RHS clamps) remains finite. The polymer is a passenger, not a driver.

## Candidate residual bugs (ranked)

1. **HIGH — pass-2 reads ρ_out written by pass-1 *post-halfwayBB*, NOT *post-Bouzidi-FL***. The RAW pass-1 spec ends with `WriteMoments()` after `PullHalfwayBB() + SolidInert() + Moments() + CollideTRTDirectGuoField()`. The Bouzidi-FL post-collision overwrite is applied by pass-2's `_TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_TWOPASS_PASS2_SPEC`. The `rho_w` used inside `_bouzidi_fl_post_value` in pass-2 is therefore taken from pass-1's `WriteMoments` output, which was computed before the cut-link f's were overwritten. Inconsistency: pass-2's f-update uses a `rho_w` that does not match the f-set after pass-2 finishes. **Fix candidate**: add a cut-link-only `WriteMoments`-equivalent at the end of pass-2 so subsequent kernels see ρ consistent with the cut-link f's.

2. **MEDIUM — args-trim 15 → 12 in dispatch may have dropped something the polymer pipeline indirectly needs.** The drop was justified for the RAW spec's `required_args`, but the polymer back-force kernel (Guo field) reads `q_wall, uw_link_x, uw_link_y` via a separate channel; verify those are still wired downstream and not implicitly relied upon by pass-1. Likely OK because `ApplyLiBBPrePhase` (which consumed them) was the only pass-1 user, but verify explicitly.

3. **LOW — q ≤ 0.5 branch formula in `_bouzidi_fl_post_value` could be a Yu-Mei-Shyy variant that slightly over-corrects vs the canonical Bouzidi-FL** (Bouzidi-Firdaouss-Lallemand 2001 eq. 12 vs. Yu-Mei-Shyy 2003 §3.2). Unlikely to be the proximate cause given the R30 Wi=0.1 clean run; only worth checking if HIGH and MEDIUM are refuted.

## Next mission spec (M34v3)

- **Objective**: implement the HIGH-confidence fix candidate — add a cut-link-only re-`WriteMoments` after pass-2 of `:bouzidi_fl_twopass`, so the ρ field the polymer side reads is consistent with the post-Bouzidi-FL f's.
- **Smoke**: re-run the existing cut-link cylinder R=8 Newtonian Re=1 testset (regression sentinel) PLUS add a Wi=0.1 polymer smoke — must exercise polymer back-force, not just Newtonian flow.
- **Re-submit**: full Aqua matrix on a single PBS (the 5 acceptance cases of M34_FIX_VERDICT §"Acceptance criteria"), gate on `Cd_kraken` envelope per case.
- **Failure handoff**: if HIGH refuted by M34v3 (still NaN at R≥40 Wi=1.0), escalate MEDIUM (args audit) before considering M35.

## NOT M35 yet

Per the triage classification (uniform NaN, `psi_nan_frac = 0.0` on all three NaN cases), a Ψ-scheme upgrade (M35: log-conformation variant or polymer-side stabiliser) is NOT the right intervention. The polymer field is FINITE everywhere on the snapshots. M35 stays parked. The next intervention is on the BC pass / ρ consistency between passes.

## Files

- `bench/scratch/m34_fix_diag/{R30_Wi1, R40_Wi1, R60_Wi0p1, R30_Wi0p1_CLEAN}_summary.txt` (existing)
- `bench/scratch/m34_fix_diag/{R30_Wi1, R40_Wi1, R60_Wi0p1}_nan_mask.png` (existing visual heatmaps)
- `bench/scratch/m34_fix_diag/triage.jl` (analysis script)
- `bench/viscoelastic_audit/M34_FIX_VERDICT.md` (M34-fix context)
- `bench/viscoelastic_audit/M34_FIX_DIAG_VERDICT.md` (this verdict)
