# Equations cross-check — Kraken vs Liu 2025 (+ Kraken 2D vs 3D)

Session 2026-04-22. Companion to `equations_cross_check_kraken_side.md`
(Kraken-side extraction by sub-agent) and `REFERENCES_STUDY.md`.

**Critical context** : Liu 2025 arxiv 2508.16997 is **2D-only** by
explicit statement of the authors. So Liu cross-check applies to 2D
only ; Kraken's 3D is an independent extension.

---

## Finding 1 — Hermite stress prefactor **inconsistent between Kraken 2D and 3D**

### 2D (`src/kernels/collide_viscoelastic_source_2d.jl:61`)

```julia
pre = -ω * T(9.0/2.0)
```

→ NO division by (1 − ω/2).

### 3D (`src/kernels/viscoelastic_3d.jl:26`)

```julia
pre = -s_plus * T(9.0/2.0) / (one(T) - s_plus / T(2))
```

→ HAS division by (1 − s_plus/2).

### The 2D file's own comment (`collide_viscoelastic_source_2d.jl:50-60`) says :

> The source T_i adds −ω·τ_αβ to the 2nd-order non-equilibrium moment
> Π_αβ. Chapman-Enskog recovers the stress as :
>   σ_p = (1 − ω/2) · Π_source / ω = −(1 − ω/2) · τ_αβ
>
> To get σ_p = −τ_αβ (full polymer stress), we must divide by
> (1 − ω/2), giving :
>   pre = −ω · 9/2 / (1 − ω/2)
>
> In Liu's regularized scheme, (1−ω/2) is already in the
> reconstruction, so their formula omits this factor. For standard BGK
> (our case), it must be included.

**The comment explicitly states that the factor SHOULD be included,
but the actual code omits it.** This is either :

(a) a copy-paste / derivation-vs-implementation error (real bug), or
(b) a deliberate choice that the 2D fused kernel uses a regularized
    interpretation, making the factor unnecessary, and the comment
    is stale.

The 3D kernel is consistent with its own comment (line 6 of
`viscoelastic_3d.jl` : "with the (1 − s_plus/2) division for standard
BGK/TRT consistency"). So 2D and 3D apply different effective stress
amplitudes :

At step5d β=0.5 R=30, ν_s=0.3, so ω_s = 1/(3·0.3+0.5) = 0.714 →
(1−ω_s/2) = 0.643. The 2D kernel applies 64.3% of the stress that 3D
would apply with the same τ_p input.

### Expected sign of the effect

At blockage 0.5 with confined Oldroyd-B, polymer stress causes **drag
reduction** (ratio < 1). Under-applying polymer stress → less drag
reduction → ratio closer to 1.

Kraken gives Cd_visco/Cd_Newt = 0.901 (2D) and 0.892 (3D). If 2D is
under-applying (as the code suggests vs its own comment), fixing the
2D prefactor would **push the ratio further below 0.901**, away from
the literature's 0.96. So this fix alone does NOT close the gap.

### Further complication

2D (0.901) and 3D (0.892) give **almost the same ratio** despite
using different prefactors. Two possibilities :

1. The 2D fused kernel IS effectively regularized in such a way that
   the missing factor is compensated — in which case 2D is already
   correct and 3D might be over-applying (extra factor = wrong).
2. Both 2D and 3D are biased (in different directions) by other
   errors that partially cancel against the prefactor discrepancy,
   converging both to ~0.89.

### Concrete tests to disambiguate

**Test A — Poiseuille τ_p verification** (cheap, local) :
At β=0.5, Wi=0.1, R=30, 2D Poiseuille, compare computed τ_p_xx(y) and
τ_p_xy(y) against the analytic Oldroyd-B values (REFERENCES.md §143):
- τ_p_xx = 2·ν_p·λ·γ̇²(y)
- τ_p_xy = ν_p·λ·γ̇(y)·(1+2(λγ̇)²) ... etc

The conformation field C is already validated (audit step 1c, O(2)
convergence on C_xy). So if τ_p = G·(C−I) matches analytic, the
INPUT to the Hermite source is right. The question is then PURELY
the injection prefactor.

**Test B — prefactor swap** (cheap, targeted) :
Toggle `collide_viscoelastic_source_2d_kernel!` line 61 between :
  `pre = -ω * T(9.0/2.0)` (current)
  `pre = -ω * T(9.0/2.0) / (one(T) - ω / T(2))` (3D-consistent)
Rerun step5d 2D at β=0.5 Wi=0.1. Compare ratios.

- Expected if current 2D code has the bug : fixed prefactor → ratio
  drops from 0.901 to ~0.85 (more reduction). Confirms bug direction.
- Expected if current 2D is correct (regularized) : fixed prefactor
  breaks the computation, may even diverge or give |ratio| far from
  0.96. Suggests current 2D is right and 3D is wrong.

**Test C — N1 sanity** (local, Poiseuille analytic) :
Rerun step 2 of the audit (`step2_trt_hermite.jl`) and check if N1(y)
is recovered at the analytic value (2·ν_p·λ·γ̇²). If Kraken 2D N1 is
0.64× analytic → prefactor is under-applying. If 1.00× → 2D is
correctly regularized.

---

## Finding 2 — TRT magic parameter convention

### Kraken (`conformation_lbm_2d.jl:133-138` and _3d.jl:202-206)

```julia
magic = 0.25    # default
tau_minus = magic / (tau_plus - 0.5) + 0.5
```

### Liu 2025 (§3 via HTML) — Eq. 38

```
τ_{p,2} = Λ_p / (τ_{p,1} − 0.5) + 0.5
```

with **Λ_p ≈ 10⁻⁶ for periodic ; optimised per case for wall-bounded**.

### Note

These are the same algebraic relation, but Kraken's default Λ = 0.25
is the standard hydrodynamic TRT magic (from d'Humières). Liu uses
Λ_p orders of magnitude smaller for the conformation field. At
τ_plus = 1.0 (step5d setting) :

| Magic | τ_minus | ω_minus | regime |
|-------|---------|---------|--------|
| Kraken 0.25 | 1.0 | 1.0 | BGK-like |
| Liu 1e-6 | 0.5 + 2e-6 ≈ 0.5 | ≈ 2.0 | maximum over-relaxation |
| Liu 1e-4 | 0.5002 | ≈ 1.999 | near-max |

**Not necessarily a bug** — the two choices represent different
stabilisation philosophies for the conformation advection. But it is
a TUNABLE DIFFERENCE that affects Cd. Kraken's step5d uses
tau_plus=1.0 and default magic=0.25, giving ω_minus = 1.0 (BGK-like
on the anti-symmetric modes). Liu's canonical LBM conformation uses
ω_minus ≈ 2.0 with a tiny Λ_p.

### Concrete test

Sweep `magic ∈ {0.25, 1e-2, 1e-4, 1e-6}` at fixed β=0.59 Wi=0.1 R=30
was run on AQUA (`visco_magic_20260428_131539`). Result:

| Magic | Cd_scaled |
|-------|----------:|
| 0.25 | 128.3937 |
| 1e-2 | 174.5943 |
| 1e-4 | 182.1699 |
| 1e-6 | NaN |

Resolution: the small Liu-style Λ_p is not a drop-in fix for this
wall-bounded Kraken driver. It strongly changes/destabilises the flow.
Keep Kraken's `0.25` until Liu/Yu's wall-bounded Λ_p optimisation is
implemented case-by-case.

---

## Finding 3 — D3Q19 for 3D conformation (Kraken-only choice)

Liu 2025 defers 3D to future work. Kraken's D3Q19 for the
conformation tensor is an independent implementation choice. The open
question (§4.2 of REFERENCES_STUDY.md) — does D3Q19 preserve the 2nd-
order Hermite moments required by the Oldroyd-B tensor advection ? —
remains unanswered and needs a hand calculation of Σ w_i c_iα c_iβ
for D3Q19 weights.

**Concrete check** (paper calculation, 15 min) :
verify for D3Q19 that Σ_i w_i c_iα c_iβ = c_s²·δ_αβ and Σ_i w_i c_iα
c_iβ c_iγ c_iδ = c_s⁴·(δ_αβ·δ_γδ + δ_αγ·δ_βδ + δ_αδ·δ_βγ). If the
4th-order moment fails → D3Q19 is insufficient, must go D3Q27.

---

## Finding 4 — Drag normalisation matches Liu

Liu Eq 64 : Cd = F_x / (0.5·ρ·U_avg²·D)
Kraken 2D : Cd = 2·Fx / (u_ref²·D)   with u_ref = (2/3)·u_in = U_avg
            (ρ = 1 implicitly)

Equivalent. ✓

---

## Summary and action for debug plan

| Finding | Severity | Resolution |
|---------|----------|-----------|
| 1. 2D vs 3D Hermite prefactor inconsistency | **HIGH** — real code discrepancy, direction unclear | Test A + B + C (see above) |
| 2. TRT magic default 0.25 vs Liu 10⁻⁶ | MEDIUM — not drop-in compatible | Sweep done; tiny Λₚ destabilises |
| 3. D3Q19 moment preservation | MEDIUM — untested, affects 3D only | Hand calculation of D3Q19 moments |
| 4. Drag normalisation | OK — matches Liu | — |

These findings become **M1.5** in the debug plan, to be executed
locally while M0 (Aqua Cd_Newt convergence) runs.

---

## Proposed M1.5 — prefactor disambiguation (local, 1-2 h)

1. **Test C (easiest)** — rerun `step2_trt_hermite.jl` on local Metal
   F32 at β=0.5 Wi=0.1, Poiseuille periodic, and dump N1(y). Compare
   to analytic 2·ν_p·λ·γ̇²(y). If Kraken N1 / analytic ≈ 0.64 → 2D
   missing factor. If ≈ 1.00 → 2D is regularized and correct.

2. **Test B (cheap)** — only if Test C is ambiguous. Create a git
   branch, toggle the 2D prefactor line to match 3D. Run step5d 2D
   Wi=0.1. Ratio drops → 3D-consistent is "more polymer" but may
   over-shoot literature. Ratio goes up → factor was compensating
   something else.

Before M2 (β sweep on Aqua), Finding 1 must be understood. Running
M2 with an ambiguous prefactor produces ambiguous data.
