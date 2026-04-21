# Reference values — viscoelastic benchmarks

**Single source of truth** for all Liu / Lunsmann / Alves reference Cd values
used in the dev-viscoelastic branch. Any script, test or doc must cite from
this file — not reinvent a value inline. If a discrepancy with the original
paper is found, update this file and propagate.

Last re-audit : 2026-04-21 (Guillaume).

---

## Liu et al. 2025 — confined cylinder Oldroyd-B

**Paper** : Liu, Zhou, Grecov, Wang — arxiv **2508.16997** (Aug 2025),
"TRT lattice Boltzmann scheme for Oldroyd-B viscoelastic fluid flow".

**Setup (Table 3)** :
- Domain : 30R × 4R, cylinder at centreline (15R, 2R)
- Blockage ratio B = R / (Ny/2) = 0.5
- Re = U_avg · R / ν_total = 1  (characteristic length = R, not D)
- β = ν_s / ν_total = 0.59
- Wi = λ · U_avg / R
- CNEBB conformation wall BC, Sc = 10⁴
- Inlet : fully-developed Poiseuille parabolic profile
- Cd = F_x / (0.5 · ρ · U_avg² · D) with D = 2R  (Liu Eq 64)

**Table 3 — Cd values at Wi = 0.1** (source: `hpc/liu_R_convergence.jl`
dictionary — to be cross-checked against published PDF Table 3) :

| R  | Cd    |
|----|-------|
| 20 | 129.42 |
| 25 | 129.61 |
| 30 | 130.36 |
| 35 | 130.77 |
| 40 | 130.79 |
| 48 | 130.83 |

⚠ **Verification status** : extracted from previous-session work, not yet
cross-checked against the arxiv PDF Table 3. TODO: once PDF is re-read,
confirm values and remove this warning.

**Table 3 — Cd values at fixed R (source: `hpc/liu_cylinder_benchmark.jl`)** :

| R  | Wi    | Cd    |
|----|-------|-------|
| 20 | 0.1   | 129.42 |
| 20 | 0.5   | 125.17 |
| 20 | 1.0   | 164.26 |
| 30 | 0.1   | 130.36 |
| 30 | 0.5   | 126.31 |
| 30 | 1.0   | 151.31 |

---

## Resolved inconsistencies in prior documentation

- `NEXT_SESSION_PROMPT.md:22` claimed "130.78 vs Liu 130.83 → 0.32% error"
  at R=48, Wi=0.1. Liu reference is **130.83** (confirmed from
  `liu_R_convergence.jl`). The simulated 130.78 gives
  |130.78 − 130.83| / 130.83 = **0.038%**, not 0.32%. The "0.32%" figure
  in the prompt appears to be mis-stated.

- **Cylinder convergence measurement (2026-04-21 audit)** : the Apr 18
  production run (job 20142038) gave Cd = {119.85, 126.41, 129.68, 131.29}
  at R = {20, 30, 40, 48}. Monotone increasing (deltas 6.56, 3.27, 1.61,
  ratio ~2 per R-step). **Richardson extrapolation Cd(R→∞) ≈ 131.5**,
  which is **~1.5% above Liu's 130.83**. The "0.35% at R=48" is a
  sign-crossing artifact, not a convergence proof. True asymptotic bias
  is ~1.5% — still acceptable for an LBM benchmark, but prior docs
  systematically misrepresented the accuracy as sub-1%.

- `VISCOELASTIC_FINDINGS.md:49` claimed "130.83 → 131.29" (0.35% err)
  as the R=48 validation. 131.29 is inconsistent with the best
  log-conformation result quoted at §3:143 (Cd_logconf = 130.78).

- `VISCOELASTIC_FINDINGS.md §3:143` stated "Cd_Liu = 130.36" at Wi=0.1 —
  that is the **R=30** reference, not R=48. It was compared against
  a R=48 simulation, which is inconsistent.

- The comment block in `liu_R_convergence.jl:8` states "R=40 Cd = 129.42"
  in the comment but the dict on line 24 gives 130.79. The dict value
  is monotone with neighbours (130.77 → 130.79 → 130.83) and is almost
  certainly the correct one. The comment is a copy-paste typo.

**Single canonical value** used from now on : **Liu Table 3, Wi=0.1, R=48 = 130.83**.

---

## Lunsmann et al. 1993 — ducted sphere Oldroyd-B

**Paper** : Lunsmann, Genieser, Armstrong, Brown, J. Non-Newt. Fluid
Mech. 50 (1993) 135–155.

**Setup** :
- Sphere in a cylindrical pipe (approximated here by a square duct
  of side 4R_s for the LBM implementation — Schäfer-Turek convention)
- Blockage β_geom = R_s / H = 0.5
- β_visco = ν_s / ν_total = 0.5
- Oldroyd-B, Wi = λ · U_avg / R_s

**Expected trend** : Cd_visco **increases** with Wi (drag enhancement),
monotonically with Wi up to Wi ~ 1 where the schemes typically break
down without log-conformation or refined stress resolution.

Qualitative reference values (order-of-magnitude, for the R_s/H=0.5
blockage) :
- Wi = 0.1 : Cd/Cd_Newt ≈ 1.02 – 1.05
- Wi = 0.5 : Cd/Cd_Newt ≈ 1.10 – 1.15
- Wi = 1.0 : Cd/Cd_Newt ≈ 1.15 – 1.25

Precise values from Lunsmann Table vary with mesh and discretization ;
for the Kraken validation the **sign** of the trend is the primary
qualitative check. The magnitude will be compared once step 1-4
(canal, cylinder) convergence are firmly established.

---

## Alves & Pinho 2003 — 4:1 contraction

**Paper** : Alves, Pinho, J. Non-Newt. Fluid Mech. 110 (2003) 45–75.

Reserved for a later audit step. Not on the critical path for the sphere
validation.

---

## Analytical references (exact, no measurement uncertainty)

### Poiseuille channel, body-force-driven, Newtonian
For a channel of width H (wall-to-wall), body force F_x, viscosity ν_total,
periodic in x, no-slip at ±H/2 (HWBB at half-cell outside the last fluid
cell) :

- u_max = F_x · (H/2)² / (2 · ν_total)
- u(y)  = u_max · (1 − (2y/H)²)
- γ̇(y) = −F_x · y / ν_total = −u_max · 4y / H²

For LBM with HWBB on rows j=1 and j=Ny (fluid cells), the effective
wall-to-wall distance is H = Ny (walls located at j=0.5 and j=Ny+0.5).

### Poiseuille Oldroyd-B steady state (Liu Eq 62)
With u(y) as above and steady-state Oldroyd-B (∂_t C = 0 and solvent
inertia negligible), the Maxwell relaxation admits the closed-form
solution :

- C_xy(y)  = λ · ∂u/∂y(y)  =  λ · γ̇(y)
- C_xx(y)  = 1 + 2 · (λ · γ̇(y))²
- C_yy(y)  = 1
- τ_p_xx   = G · (C_xx − 1)  = 2 · ν_p · λ · γ̇²
- N1(y)    = τ_p_xx − τ_p_yy = 2 · ν_p · λ · γ̇²(y)

This is the ground truth used for all step-1 / step-2 convergence tests
(no measurement uncertainty).
