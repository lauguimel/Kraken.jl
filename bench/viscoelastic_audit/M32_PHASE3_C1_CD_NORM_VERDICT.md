# M32 Phase 3 — C1 Cd normalisation audit (Kraken vs rheoTool)

Date    : 2026-05-21
Branch  : dev-viscoelastic
Mission : Verify bit-for-bit that Kraken's stored `Cd_kraken` (driver-level
          assembly `Cd_s + Cd_p − Cd_bsd`) uses the same normalisation as
          rheoTool's reported `Cd` (Hulsen `K = Fx / (etaS + etaP)`). If
          they differ, compute the conversion factor.
Method  : Read-only code audit (no simulations, no edits in src/).

## Status: EXECUTED — verdict is **SAME (at the canonical R=1 U_mean=1 ρ=1 η_total=1 benchmark)**

The two codes do **NOT** use the same algebraic formula. They produce the
same numerical value **only because the parameter choice ρ=U=1, D=2,
η_total=1 collapses both formulas to `Cd ≡ Fx_lbm`**. Outside that
canonical regime they differ by a known factor `k = (ρ·U²·D) / (2·η_total)`.
For the Wi=1 R=30 β=0.59 production this factor is **1.0 (machine ε)**,
so existing M30/M31 numerical comparisons stand. **A warning is filed in
the memory candidates** because any future case with `ρU²D ≠ 2η_total`
will desynchronise the two scalars and silently invalidate Cd parity.

## Kraken `Cd_kraken` formula

**Source** : `src/drivers/viscoelastic_logfv_2d.jl:594–607`

```julia
# Lines 594-606 (per-component averages over avg_window, then nondim)
Fx_s   = Fx_s_sum   / n_drag
Fx_p   = Fx_p_sum   / n_drag
Fx_bsd = Fx_bsd_sum / n_drag

drag_diameter = 2.0 * drag_radius        # = 2R, lattice cells
drag_speed    = drag_u_ref               # = u_mean, LBM units

Cd_s   = 2.0 * Fx_s   / (drag_speed^2 * drag_diameter)
Cd_p   = 2.0 * Fx_p   / (drag_speed^2 * drag_diameter)
Cd_bsd = 2.0 * Fx_bsd / (drag_speed^2 * drag_diameter)
Cd     = Cd_s + Cd_p - Cd_bsd
```

**Formula (math)**:

```
Cd_kraken = 2 · (Fx_s + Fx_p − Fx_bsd) / (u_mean² · D)        [ρ = 1 implicit in LBM]

            wall stresses
            ┌──────────────┐
Fx_s   = ∮  σ_LBM,wall · n_x dA       (cut-link MEA momentum exchange, src/drivers/cylinder_libb.jl:129-163)
Fx_p   = ∮  τ_p(x) · n_x dA           (gradient-extrapolated polymer-stress wall integral, viscoelastic.jl:65-147)
Fx_bsd = ∮  (2·ζν_p·D[u]) · n_x dA    (BSD double-count subtraction, viscoelastic_logfv_2d.jl:13-47)
```

The MEA Fx_s **already** includes the total wall momentum exchange (solvent
+ BSD-injected polymer), hence `Fx_p − Fx_bsd` adds the *non-BSD* fraction
of the polymer stress (the part not double-counted by the LBM solver). At
`bsd_fraction = 0`, `Fx_bsd = 0` and the assembly reduces to `Cd_s + Cd_p`.

Production callsite (`viscoelastic_logfv_2d.jl:879-880`):
```julia
drag_radius = Float64(radius)              # = 30 LBM cells at R=30
drag_u_ref  = Float64(get(kwargs, :u_mean, 0.01))   # = u_mean LBM (e.g. 0.02)
```

## rheoTool `Cd` formula

**Source** : `bench/rheotool/cylinder_wi1.0/system/controlDict:83-91`

```cpp
volTensorField    L(fvc::grad(U));
volSymmTensorField F( tau + symm(L + L.T())*etaS - p*symmTensor::I*rho );

vector Fpatch = gSum( (-mesh().boundaryMesh()[cyl].faceAreas())
                       & F.boundaryField()[cyl] ) / (etaS_ + etaP_).value();

list.append(Fpatch.x());    // Written to Cd.txt column 1
```

**Formula (math)**:

```
Cd_rT = Fx,wall / (η_S + η_P)

Fx,wall = ∮ [ τ_p  +  2·η_S·D[u]  −  p·ρ·I ] · n_out  dA  · ê_x
```

with `n_out = -faceAreas() / |faceAreas()|` (OpenFOAM area vectors point
out of the cell; the `-` flips them to point out of the cylinder, so they
are the outward solid normal).

Physical parameters (`bench/rheotool/cylinder_wi1.0/constant/constitutiveProperties:21-23`,
`0/U:32`, blockMeshDict R=1 D=2):
- `rho = 1`, `etaS = 0.59`, `etaP = 0.41`  →  `etaS + etaP = 1.0`
- `Umean = 1.0`, `halfHeight = 2.0`  →  `Umax = 1.5`, channel H = 4
- cylinder R = 1, D = 2

So `Cd_rT = Fx,wall / 1.0 = Fx,wall`.

## Term-by-term comparison

| Term | Kraken | rheoTool | Match? |
|---|---|---|---|
| Numerator: solvent wall stress | `∮ σ_LBM,wall·n_x dA` via cut-link MEA (Mei) | `∮ (2η_S·D[u] − p·I)·n_x dA` via wall FV interpolation | YES (same physics; M31 verdict confirmed frame match) |
| Numerator: polymer wall stress | `∮ τ_p·n_x dA` via gradient extrapolation | `∮ τ_p·n_x dA` via wall FV interpolation | YES |
| Numerator: BSD subtraction | `−∮ 2·ζν_p·D[u]·n_x dA` (double-count correction) | not applicable (rT has no BSD scheme) | N/A — Kraken-specific (≈0 at bsd_fraction=0) |
| Outward normal convention | `nx = (xw − cx) / R`, points out of cylinder | `n = -faceAreas / |faceAreas|`, points out of cylinder | YES (both outward) |
| Sign of pressure traction | `−p·n_x` inside `σ_LBM` (MEA), `+0` in Cd_p path | `−p·ρ·n_x` (in `F`) | YES (both subtract `p·n_x`; ρ=1) |
| **Denominator** | `(1/2)·ρ·U²·D = (1/2)·1·u_mean²·(2R)` (LBM units) | `(etaS + etaP) = η_total` | **DIFFERENT FORMULA** |
| Denominator value at canonical case | `(1/2)·1·1²·2 = 1.0` (in non-dim where U≡1, D≡2, ρ≡1) | `1.0` (etaS+etaP=1) | **NUMERICALLY EQUAL** |
| Sign of total | `+Cd_s + Cd_p − Cd_bsd` (drag direction = +x) | `+Fpatch.x` (drag direction = +x) | YES |

### The two denominators

The conversion factor between the two scalars is:
```
Cd_kraken = 2·Fx,LBM / (ρU²D)
Cd_rT     =   Fx     / η_total

⇒  Cd_kraken / Cd_rT = 2·η_total / (ρ·U²·D)
```

If the user works **in lattice units where ρ_LBM = 1, U_LBM = u_mean,
D_LBM = 2R**, and rT works in **physical units where ρ_phys = 1, U_phys = U_mean,
D_phys = 2R_phys, η_total,phys = ηS+ηP**, the standard Newtonian/Oldroyd-B
non-dim group `Re = ρUD/η_total` makes the conversion explicit:

```
Cd_kraken / Cd_rT = 2 / Re
```

So in general **the two scalars differ by `2/Re`**. At Re=1 (the rheoTool
Hulsen benchmark by construction), `2/Re = 2`. **That contradicts the
M30 verdict claim that K ≡ 2·Fx/(ρU²D).**

Resolution: at the M30/rheoTool benchmark, ρU²D = 1·1²·2 = 2 (not 1),
so `2·Fx/(ρU²D) = 2·Fx/2 = Fx`. **And** `Fx / η_total = Fx / 1 = Fx`.
Both yield `Cd ≡ Fx` because *both* denominators happen to equal 1. The
M30 statement "K ≡ 2Fx/(ρU²D)" is a happy coincidence of ρU²D = 2·η_total
in the canonical Hulsen setup (Re_Hulsen = ρU·R/η = 0.5, not 1).

## Numerical cross-check at R=30 Wi=1 β=0.59 (production)

Reading the production parameters:
- Kraken (`hpc/liu_logconf_sweep.jl:18-19`): `u_mean = 0.02`, `R = 30`,
  `ν_total = u_mean·R/1 = 0.6` (Re_Liu = 1, defined as `ρ·u_mean·R/η_total`).
- rT (this controlDict): `Umean = 1.0`, `R = 1`, `η_total = 1.0`
  (Re_Hulsen = `ρ·U·R/η_total = 1·1·1/1 = 1`).

Both runs target the same dimensionless `Re = ρUR/η_total = 1`.

| Quantity            | Value at Re=1 Wi=1 β=0.59 |
|---------------------|---------------------------|
| Cd_kraken (Metal F32 run01, stored) | 111.09 |
| Cd_rT  (Cd.txt last value)          | 120.40 |
| Cd_rT  (M30 wall-stress integration) | 119.0 |
| Δ                   | rT − Kraken = 9.31  (or 7.91 with the 119.0 reference) |

**Conversion sanity check**: do both scalars genuinely represent the same
non-dim drag coefficient?

```
Kraken:  Cd_kraken = 2 · Fx_LBM / (u_mean² · D_LBM)
                   = 2 · Fx_LBM / (0.02² · 60)
                   = 2 · Fx_LBM / 0.024
                   ≈ 83.33 · Fx_LBM
```

```
rT:      Cd_rT = Fx,wall,rT / 1.0  =  Fx,wall,rT
```

These look different, but Fx_LBM (LBM units, ρ_LBM=1, c_LBM=1) and
Fx,wall,rT (physical SI-like units) are **different physical quantities**.
The dimensionless `2·Fx/(ρU²D)` is invariant under choice of lattice
units → both `Cd_kraken` and `Cd_rT` are the **same** non-dimensional drag
coefficient `Cd_classical = 2·Fx/(ρU²D)`, because at the rT benchmark
`ρU²D / 2 = 1 = η_total`, and at the Kraken benchmark the dimensionless
ratio is computed explicitly via `2/(u_mean²·D)`. They are numerically
comparable: gap = 9.31 (or 7.91) is real and matches M30's reported gap.

**Verdict on the unit check**: the 111.09 vs 120.40 comparison is
apples-to-apples. The +8.4 % gap is the actual physics mismatch, not a
normalisation artefact.

## Verdict

**SAME normalisation at this benchmark — both compute the standard
non-dimensional drag coefficient `Cd = 2·Fx/(ρU²D) ≡ Fx,wall/η_total`
when ρU²D = 2·η_total**, which holds by parameter choice in the
rheoTool Hulsen setup (ρ=1, U=1, D=2, η_total=1). Kraken's lattice-unit
formula `2·Fx_LBM/(u_mean²·2R)` produces the same dimensionless number
because the LBM Fx_LBM scales with `ρ_LBM·U_LBM²·R_LBM` exactly the same
way as the physical Fx scales with `ρ·U²·R`.

**Conversion factor (general case)**:
```
Cd_kraken / Cd_rT = 2·η_total / (ρ·U²·D) = 2/Re
```

At Re=1 (Hulsen benchmark): `Cd_kraken / Cd_rT = 2/1 = 2`...
**WAIT.** Let me re-derive. At Re=1 in *rheoTool's* benchmark
`ρ=1, U=1, D=2, η_total=1`:
- `ρU²D = 1·1·2 = 2`
- `2·η_total = 2·1 = 2`
- Both equal 2 → ratio = 1.

At Re=1 in *Kraken's* benchmark `ρ_LBM=1, u_mean=0.02, D_LBM=60, η_total_LBM=ν_total=0.6`:
- `ρ U² D = 1 · 0.0004 · 60 = 0.024`
- `2·η_total = 1.2`
- `Cd_kraken/Cd_rT_hypothetical = 2·1.2/0.024 = 100`

But Kraken doesn't *use* the rT formula `Fx/η_total`. Kraken uses
`2Fx/(ρU²D)`. So the only relevant question is: **does Kraken's
`2·Fx_LBM/(u_mean²·D_LBM)` equal rT's `2·Fx_phys/(ρ·U_phys²·D_phys)`
which in turn equals `Fx_phys/η_total,phys` by the canonical-setup
coincidence?** Yes, because dimensionless Cd is invariant under unit
scaling.

**Final verdict**:
- Algebraically, Kraken's formula is `Cd_classical = 2·Fx/(ρU²D)`.
- rT's formula is `K_Hulsen = Fx/η_total`.
- The two formulas are **equivalent** iff `ρU²D = 2·η_total`, i.e. iff
  `Re = ρUD/η_total = 2`. **At the rheoTool Hulsen benchmark we have
  Re_Hulsen = ρUR/η = 1, i.e. ρUD/η = 2 → equivalence HOLDS.**
- Existing M30/M31/M32 Cd numerical comparisons (Kraken 111.09 vs rT
  120.40, gap 9.31, ≈8.4 %) are unaffected — the gap is real physics,
  not normalisation.

### Boxed verdict for the Boss

**SAME** at the canonical Hulsen benchmark (ρ=1, U=1, D=2, η_total=1, Re=1).
**No correction needed** — existing M30/M32 numerical comparisons stand.

**Conversion (if ever needed for a non-canonical case)**:
```
Cd_kraken  ≡  2·Fx/(ρU²D)
Cd_rT      ≡  Fx/η_total

⇒  Cd_kraken / Cd_rT = 2·η_total / (ρU²D) = 2 / Re_diameter
       where Re_diameter = ρUD/η_total

⇒  At Re_diameter = 2 (≡ Re_radius = 1, the Hulsen benchmark), ratio = 1.
   At other Re, the two scalars differ and a conversion factor is required.
```

## Files

- `src/drivers/viscoelastic_logfv_2d.jl:482-525, 594-607, 870-883`
  (Cd assembly + production drag_u_ref / drag_radius wiring)
- `src/drivers/cylinder_libb.jl:98-190` (`compute_drag_libb_mei_2d` host
  + GPU-cached MEA cut-link integral)
- `src/drivers/viscoelastic.jl:65-147` (`compute_polymeric_drag_2d`
  gradient-extrapolated wall integral of τ_p)
- `src/drivers/viscoelastic_logfv_2d.jl:13-47` (`_logfv_compute_bsd_drag_2d`
  BSD double-count subtraction)
- `bench/rheotool/cylinder_wi1.0/system/controlDict:65-111`
  (outputCd functionObject)
- `bench/rheotool/cylinder_wi1.0/constant/constitutiveProperties:21-23`
  (rho=1, etaS=0.59, etaP=0.41)
- `bench/rheotool/cylinder_wi1.0/0/U:22-44` (Umean=1, halfHeight=2)
- `hpc/liu_logconf_sweep.jl:18-19` (Kraken Wi sweep at R=30, u_mean=0.02)
- `bench/viscoelastic_audit/M30_RHEOTOOL_P_PROFILE_VERDICT.md:25`
  (M30's statement K ≡ 2Fx/(ρU²D), confirmed valid only at the
  ρU²D=2η_total canonical setup)

## Memory candidates

1. `feedback_cd_norm_equivalence_canonical_only` — Kraken's
   `Cd = 2·Fx/(ρU²D)` and rheoTool's Hulsen `K = Fx/η_total` are
   **only numerically equivalent when ρU²D = 2·η_total**, i.e. when the
   diameter-based Reynolds number Re_D = ρUD/η_total = 2 (equivalently
   Re_radius = 1, the Hulsen canonical setup). For any benchmark with a
   different Re_D the two scalars differ by the factor `2/Re_D` and a
   manual conversion is required. Always check parameter setup before
   comparing Cd values between LBM (classical Cd) and rheoTool (Hulsen K).

2. `feedback_hulsen_K_is_classical_Cd_at_Re1` — At the Hulsen 2005
   benchmark (ρ=1, U=1, D=2, η_total=1, Re=1 diameter-based-as-radius=2
   diameter), Hulsen's K coincides bit-for-bit with the classical drag
   coefficient `2·Fx/(ρU²D)`. This is by design of the benchmark
   (ρU²D = 2 = 2·η_total), not a property of the Hulsen K formula in
   general. Outside this setup the two diverge by `2/Re_D`.

3. `feedback_kraken_cd_3component_assembly` — Kraken's viscoelastic
   `Cd_kraken = Cd_s + Cd_p − Cd_bsd` where:
   (a) `Cd_s` is the MEA cut-link integral of the **full** wall momentum
       exchange (solvent + BSD-injected polymer), src/drivers/cylinder_libb.jl;
   (b) `Cd_p` is the gradient-extrapolated wall integral of τ_p,
       src/drivers/viscoelastic.jl `compute_polymeric_drag_2d`;
   (c) `Cd_bsd` is the wall integral of the BSD-injected solvent stress
       `2·ζν_p·D[u]·n`, subtracted to avoid double-counting (it appears
       once in Cd_s via the LBM kernel and once again in Cd_p via the
       FV τ_p path). At `bsd_fraction = 0`, `Cd_bsd = 0`. All three
       components share the same denominator `(1/2)·u_mean²·D` applied
       at lines 604-606.

4. `feedback_m30_cd_norm_clarification` — The M30 verdict statement
   "Cd = Fpatch.x ≡ 2Fx/(ρU²D), ρ=1 U=1 D=2" is correct numerically but
   misleading algebraically: rT's formula is `Fx/η_total`, not
   `2Fx/(ρU²D)`. The two coincide only because the chosen parameters
   make both denominators equal 1. Document this in M30 to prevent
   confusion when applying the same K-extraction harness to a
   non-canonical setup (different ρ, U, D, or η_total).
