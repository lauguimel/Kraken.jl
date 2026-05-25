# M46 + M46-B — K Newtonian R-sweep & R=60 Wi=1 temporal convergence

Date: 2026-05-25
Branch: dev-viscoelastic
Aqua jobs: `21861929` (M46 Newt sweep) + `21862685` (M46-B tconv probe)
Status: **CRITICAL FINDING** — R=60 Wi=1 NOT temporally converged at 100k (the M44 sweep duration); β-residual hypothesis ELIMINATED but mesh-R "decrease with R" reframed as time-effect not mesh-effect.

---

## TL;DR

Two diagnostics in parallel on Aqua A100 F64:

1. **M46 K Newt sweep** (8 cases): Newtonian via β=1.0 placeholder.
   Both BCs (halfwayBB and bouzidi_fl_twopass) give **Cd INCREASING
   monotonically with R**. **Opposite direction** to the M44 sweep Wi=1
   Cd-DECREASE.
2. **M46-B temporal convergence probe** (4 cases): R=30 @ 400k +
   R=60 @ {100k, 200k, 400k} Wi=1. **R=30 fully converged at 100k**
   (Cd=118.099 vs M44 anchor 118.10, Δ=0.003). **R=60 NOT converged
   even at 400k**: Cd drops 113.23 → 112.13 → 109.42 over 100k→200k→400k,
   accelerating (−1.1 then −2.7).

**Verdict on M45 residual**:
- **β eliminated**: Newt opposite-direction trend rules out lattice-distance,
  TRT-relaxation, or domain-scaling explanations.
- **M44 anchor R=30 confirmed**: 100k is enough at R=30; the 78% closure
  is solid.
- **R-trend reframing**: the M44 sweep "Cd decreases with R at Wi=1" is
  NOT a pure mesh effect — it is a **time-effect that scales with R**.
  At R=30, 100k steps = ~0.56 flow-through is enough. At R=60, 100k =
  0.28 flow-through is far from steady state; 400k = 1.10 is still
  drifting.
- **M45 γ candidate (FVFD non-q_wall-aware stencil) NOT confirmed**:
  the residual "signature" was measured on a non-converged snapshot.

---

## (a) M46 K Newt sweep (`21861929.aqua`, 15 min walltime)

8 cases via viscoelastic_logfv driver, β=1.0 (zero polymer), Wi=1
placeholder (driver requires λ>0; β=1 makes polymer chain dormant
LBM-side, no F_poly injection).

### halfwayBB + qwall

| R | Cd_kraken | Cd_s | trace_C_max | nan |
|---:|---:|---:|---:|---:|
| 30 | 132.076 (matches M41) | 132.076 | 209 | false |
| 40 | 132.380 | 132.380 | 227 | false |
| 50 | 132.558 | 132.558 | 237 | false |
| 60 | 132.675 | 132.675 | 321 | false |

**Cd INCREASES monotonically**, +0.60 from R=30→60. Reasonable mesh
convergence behavior for Newtonian flow (better-resolved boundary layer
→ higher viscous drag).

### bouzidi_fl_twopass + qwall

| R | Cd_kraken | Cd_s | trace_C_max | nan |
|---:|---:|---:|---:|---:|
| 30 | 132.637 (matches M41) | 132.637 | 1938 | false |
| 40 | 133.537 (matches M41) | 133.537 | 3358 | false |
| 50 | 134.313 | 134.313 | 9610 | false |
| 60 | 135.436 (matches M41) | 135.436 | **1.4e7** | false |

**Cd INCREASES monotonically**, +2.80 from R=30→60. **trace_C_max
explosion at R=60 = 14 million** despite β=1.0 (polymer should be
dormant). This is an anomaly — the polymer chain (C tensor evolution)
is unstable under Bouzidi-FL Newt even though ν_p=0 makes F_poly=0
LBM-side. Cd_kraken is NOT contaminated (Cd_s = Cd_kraken, no polymer
contribution to LBM-MEA drag), but the polymer chain instability
suggests a separate Bouzidi-FL bug.

### Comparison vs M44 viscoelastic sweep direction

| Sweep | R=30→R=60 | Direction |
|---|---|---|
| Newt halfwayBB (β=1.0) | 132.08 → 132.68 | **↑ +0.60** |
| Newt Bouzidi-FL (β=1.0) | 132.64 → 135.44 | **↑ +2.80** |
| Wi=1 halfwayBB β=0.59 (M44 sweep, 100k) | 118.10 → 113.23 | **↓ −4.87** |

**Direction OPPOSITE**. The Wi=1 R-decrease is NOT a Newtonian
baseline (BC/lattice-distance/TRT) effect.

---

## (b) M46-B temporal convergence probe (`21862685.aqua`, 24 min)

4 cases, all viscoelastic Wi=1 β=0.59 muscl_superbee halfwayBB qwall
(same setup as M44 sweep).

| Case | Cd_kraken | Δ vs 100k | Cd_s | Cd_p |
|---|---:|---:|---:|---:|
| R=30 @ 400k | **118.099** | =100k ±0.003 | 118.891 | 13.99 |
| R=60 @ 100k | 113.234 | (baseline) | 113.037 | 13.97 |
| R=60 @ 200k | 112.130 | **−1.10** | 111.587 | 14.48 |
| R=60 @ 400k | **109.424** | **−3.81** | 108.187 | 15.30 |

**Key observations**:
- **R=30 is rock-solid temporally**: 118.099 at 400k vs 118.10 at 100k.
  M44 anchor confirmed beyond doubt.
- **R=60 is NOT converging**: drift is **accelerating** (−1.10 then
  −2.71 per 200k window). No plateau visible at 400k.
- **Cd_p INCREASES with time**: 13.97 → 14.48 → 15.30 (+1.33).
- **Cd_s DECREASES faster**: 113.04 → 111.59 → 108.19 (−4.85).
- Net Cd drops because solvent drops faster than polymer grows.
- **trace_C_max stable** ~230 across all R=60 cases (no polymer blow-up).
- 0 NaN. Run completes cleanly, just doesn't converge.

Flow-through time analysis (u_mean=0.005, domain = 30R LU):
- R=30: 180k steps per flow-through → 100k = 0.56, 400k = 2.22 flow-throughs
- R=60: 360k steps per flow-through → 100k = 0.28, 400k = 1.11 flow-throughs

**The drift correlates with flow-through count, not with polymer
relaxation count.** At R=30 with 2.22 flow-throughs, the wake is
established and Cd has plateaued. At R=60 with only 1.11 flow-throughs,
the wake (and Cd) is still developing.

---

## (c) Hypothesis matrix for the R=60 Wi=1 drift

| Hypothesis | Evidence for | Evidence against | Verdict |
|---|---|---|---|
| Slow polymer transient | Cd_p grows with t | trace_C stable; λ_LU=12000 should give ~5λ=60k convergence (well within 100k) | Weak |
| Vortex shedding (Hopf bif crossed at R=60 Wi=1) | Cd is non-monotone in time at R=60 | Drift is accelerating, not oscillating | Possible — would need Cd(t) series |
| Numerical drift (mass/momentum slow violation) | Cd shifts past 200k, accelerating | trace_C stable; no NaN | Possible |
| Wake length growth (wake hasn't reached outlet at 400k) | Cd_p increase matches expectation if wake extends with time | At R=30, wake establishes in <100k | **Best fit** |
| BC bug (Bouzidi or halfwayBB) | Bouzidi Newt trace_C blow-up | M44+M46-B used halfwayBB, not Bouzidi | Halfway: low evidence; Bouzidi: separate signal |

**Most parsimonious**: at R=60 Wi=1 with L=15R, the polymer-loaded wake
needs >1 flow-through (more than 400k steps) to establish. The Cd
"decrease with R" in M44 sweep is the snapshot of a slow wake build-up
at a fraction of flow-through.

---

## (d) Implications for prior verdicts

### M44 fix (commit 9fd92ab0) — **VALIDATED at R=30**

The R=30 anchor closure of 78% (Cd 111→118 vs rT 120.38) is **temporally
robust**. The Guo half-step fix is the right fix. M28-M42 cluster
remains CLOSED.

### M44 sweep R-trend at Wi=1 — **REFRAMED**

Not a "mesh effect" or "domain-size effect". It is **incomplete temporal
convergence at R>30**, scaling with flow-through requirements. The
M44 sweep at R=40/50/60 max_steps=100k under-sampled the wake.

### M45 residual decomposition — **PARTIALLY INVALIDATED**

The per-θ decomposition at R=60 (B finding: residual in Cd_solv shoulder
+ Cd_pres wake) was on a **non-converged snapshot**. The decomposition
may not represent a steady-state residual. **The "γ FVFD non-q_wall-aware
stencil" candidate is unsupported by this data** — we don't know the
steady-state R=60 Wi=1 Cd, so we can't claim there's a residual.

### Open items

1. **Bouzidi-FL Newt trace_C explosion** : with β=1.0 (ν_p=0), polymer
   F_poly is zero so it doesn't matter for Cd. But the C tensor reaching
   1.4e7 in Newt R=60 Bouzidi is a polymer-chain bug that may not surface
   at viscoelastic Wi=0.1/1 because it's smaller magnitude. Worth a
   separate audit (`feedback_bouzidi_polymer_chain_blowup` candidate).
2. **Guo fix completeness** : the M44 G1+G3 fix removed `+F/2` from
   2 readouts. Codex G2/G4/G5/G6/G7 inventory of similar getters
   remains unfixed. If any of those run in the active cylinder Newt path
   (Codex said no, but worth re-checking after this finding), Newt Cd
   could be subtly off.
3. **R=60 Wi=1 true steady-state Cd**: needs Cd time-series logging or
   run at 1.6M-3.2M steps to confirm plateau (if it ever plateaus). Or
   per-step Cd dumps from existing infrastructure.

---

## (e) Open questions for next session

The cylinder Wi=1 R-sweep narrative is now:
- M44 fix solid at R=30: validates the Guo half-step bug + fix
- R≥40 results from M44 sweep need re-run with longer max_steps
- The user-suspected Bouzidi BC issues (separate from the M44 Guo fix)
  may explain the Newt trace_C blow-up; worth investigating
- "Guo in Newtonian" — does the fix completely close the Newtonian
  path, or does the M46 Newt Cd at R=30 (132.076 vs rT 132.37 =
  −0.22%) hint at a residual Newt-side bug?

---

## (f) Artifacts

- `bench/viscoelastic_logfv/run_cyl_m46_newt_sweep_a100.pbs` (NEW)
- `bench/viscoelastic_logfv/run_cyl_m46b_tconv_a100.pbs` (NEW)
- `tmp/m46_newt_sweep/21861929.aqua/` (8 cases CSV + jls)
- `tmp/m46b_tconv/21862685.aqua/` (4 cases CSV + jls)
- This verdict markdown
