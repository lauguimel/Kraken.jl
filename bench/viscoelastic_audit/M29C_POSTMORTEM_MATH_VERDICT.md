# M29c post-mortem — mathematical verdict on the central-average fallback

**Branch**: `dev-viscoelastic`
**Date**: 2026-05-19
**Scope**: independent first-principles audit of the M29c patch in
`src/fvfd/operators_2d.jl` (uncommitted, +90/−32) after M29c-validate Dept
reported FAIL at R=30 Wi=1.0 β=0.59 F64 (Cd = −1571).

This document is a pure math / theory analysis. No engineer was spawned;
no simulation was rerun. It exists to check whether the FAIL diagnosis
("central-average fallback, anti-TVD, unconditionally unstable") is
*correct, complete, and quantitatively sufficient* to explain the
observed Cd = −1571.

---

## 1. The patched code — exact quote and per-branch semantics

### 1.1 Helper functions (operators_2d.jl, lines 483–502)

```julia
@inline function _fvfd_muscl_superbee_face_value_2d(far_upwind, upwind, downwind)
    d_up   = upwind   - far_upwind
    d_down = downwind - upwind
    r = ifelse(d_down == zero(d_down), zero(d_down), d_up / d_down)
    return upwind + (one(r) / (one(r) + one(r))) * _fvfd_superbee_limiter_2d(r) * d_down
end

@inline function _fvfd_muscl_superbee_face_value_oneSided_2d(upwind, downwind)
    return (upwind + downwind) / (one(upwind) + one(upwind))
end

@inline function _fvfd_muscl_superbee_guarded_face_value_2d(
    far_upwind, upwind, downwind, canonical_usable,
)
    if canonical_usable
        return _fvfd_muscl_superbee_face_value_2d(far_upwind, upwind, downwind)
    else
        return _fvfd_muscl_superbee_face_value_oneSided_2d(upwind, downwind)
    end
end
```

The name `_fvfd_muscl_superbee_face_value_oneSided_2d` is **misleading**.
Mathematically the function returns

    f_face = (φ_upwind + φ_downwind) / 2

which is the **arithmetic mean of the two cells adjacent to the face**,
i.e. the **2-point central-differencing face value** (CD2). It is *not*
1-sided upwind. 1-sided upwind would have returned `upwind` alone.

### 1.2 Per-face dispatcher (lines 547–613, east face shown; W/N/S analogous)

```julia
phie = if ue >= 0
    upwind = phi[i, j]
    downwind = east_value
    canonical_usable = i > 1 && !is_solid[i - 1, j]
    far_upwind = canonical_usable ? phi[i - 1, j] : upwind
    _fvfd_muscl_superbee_guarded_face_value_2d(
        far_upwind, upwind, downwind, canonical_usable,
    )
else
    upwind = east_value
    downwind = phi[i, j]
    canonical_usable = i + 2 <= Nx && !is_solid[i + 2, j]
    far_upwind = canonical_usable ? phi[i + 2, j] : upwind
    _fvfd_muscl_superbee_guarded_face_value_2d(
        far_upwind, upwind, downwind, canonical_usable,
    )
end
```

Semantics per branch:

| condition                                   | far-upwind                | face value returned                    |
|---------------------------------------------|---------------------------|----------------------------------------|
| `canonical_usable = true`                   | `phi[i-1,j]` (or symm.)   | full MUSCL-superbee (TVD, 2nd order)   |
| `canonical_usable = false` (solid / OOB)    | unused                    | `(phi[i,j] + east_value) / 2`  — CD2   |

### 1.3 What changed vs M29b

M29b (commit `42d2177a`) used a **cell-level** dispatch: if *any* of the
8 stencil neighbours was solid or OOB, the entire cell fell back to
`:rusanov`, which is **pure 1st-order upwind on all four faces**
(donor cell, with the `divu` correction term).

M29c keeps MUSCL on the "good" faces (canonical 4-point stencil
available) and only falls back **per face** — but the fallback is CD2,
not 1st-order upwind. So at every cell touching the cylinder, M29c
*upgrades* the worst face from donor-cell (1st-order, dissipative) to
CD2 (2nd-order, non-dissipative).

---

## 2. TVD analysis of the boundary cell

Consider 1D linear advection ∂φ/∂t + u ∂φ/∂x = 0, u > 0, uniform Δx.
Finite-volume update for cell i with explicit Euler:

    φᵢⁿ⁺¹ = φᵢⁿ − (uΔt/Δx) (f_{i+1/2} − f_{i−1/2})

with CFL number ν = uΔt/Δx ∈ (0, 1].

Let i = 1 be the wall-adjacent cell (i = 0 = solid, value unavailable).
Face f_{1+1/2} must be reconstructed from {φ_1, φ_2, …}; far-upwind
φ_0 is missing.

### 2.1 The three candidate face reconstructions

(a) **1st-order donor (upwind)**: f = φ_1.

(b) **CD2 (central average)**: f = (φ_1 + φ_2) / 2.

(c) **MUSCL with central-average fallback (M29c)**: f = (φ_1 + φ_2) / 2.

(c) is identical to (b) — the patch falls back to CD2.

### 2.2 TVD on a monotone profile

Take φᵢ = i − 1 (monotone), u = 1, ν = 1/2.

Compute the update of cell 2 (interior, MUSCL is fully usable there).
With superbee on r = (φ_2 − φ_1) / (φ_3 − φ_2) = 1, ψ(1) = 1 (superbee
hits ψ = 2 at r = 0.5 and 1 at r = 1, then increases again above r = 1;
the standard form is ψ(r) = max(0, min(2r,1), min(r,2)) giving ψ(1) = 1).
Hence f_{2+1/2} = φ_2 + ½·ψ·(φ_3 − φ_2) = 1.5. The full MUSCL recipe is
TVD by Sweby's theorem.

The **interesting cell is i = 1**. With fallback (c)=(b),
f_{1+1/2} = (0 + 1)/2 = 0.5. The west face f_{1−1/2} is the wall
(no flux, or some boundary condition — call it φ_w). The TV
contribution from the i=1 face pair is

    Δφ_1 = φ_1 − ν (f_{3/2} − f_{1/2}) = 0 − ½(0.5 − φ_w).

For a no-flux wall (φ_w = 0), Δφ_1 = −0.25, so φ_1ⁿ⁺¹ = −0.25 < 0 = min(φⁿ).
**The minimum has been pushed below the original minimum**, violating
TVD (more precisely the LED / monotonicity-preserving property). The
analogous argument with 1st-order upwind (a) gives f = 0 and
Δφ_1 = 0 — no violation.

For (a) the boundary update is **monotonicity-preserving**.
For (b)=(c) it is **not**, even on a perfectly monotone profile.

### 2.3 Checkerboard amplification

Take φ = {0, 1, 0, 1, 0, …} (the highest-frequency mode on a Cartesian
grid). Standard CD2 face values: f_{i+1/2} = (φᵢ + φ_{i+1})/2 = 1/2
everywhere. The flux differences (f_{i+1/2} − f_{i−1/2}) = 0 →
**zero numerical dissipation on the checkerboard mode**. The mode is
neutrally stable under pure CD2; *any* implicit nonlinear coupling
(diffusion sign error, exponential map Ψ→C, BSD subtraction, LBM
relaxation feedback) can flip it from neutral to growing.

By contrast 1st-order upwind on the same checkerboard gives
f_{i+1/2} = φᵢ → flux divergence ν (φᵢ − φ_{i−1}), which alternates in
sign cell-to-cell and **damps the checkerboard at rate (1 − cos π)
ν = 2ν per step** (in Fourier-amplification language, the amplification
factor for mode kΔx = π is |1 − 2ν|, exactly zero at ν = 1/2).

**Conclusion of §2**: replacing the unavailable canonical 4-point
stencil by CD2 destroys the numerical dissipation that is the *only*
mechanism preserving boundedness at the high-wavenumber tail. Pure CD2
is, in von Neumann analysis, **neutrally stable** for linear advection
*without diffusion*; coupled to a stiff nonlinear source it becomes
*linearly unstable*.

---

## 3. Quantitative magnitude of the M29c FAIL

### 3.1 Where the fallback fires

In the cylinder geometry (R=30, F64, β=0.59, Wi=1.0), `canonical_usable`
is `false` for every face whose far-upwind neighbour is either
out-of-domain (×) or solid (S). Around the cylinder, that is the
**first FV cell-ring adjacent to the solid (≈ 2πR / Δx ≈ 190 cells)**,
and at most 2 faces per cell in that ring. Of those, roughly half are
in the leeward shoulder x/R ∈ [0, 0.3] where M29-tau-compare already
identified the polymer stress peak.

So **CD2 is active on O(200) faces concentrated exactly where Ψ has its
steepest gradient**. This is the worst possible place for it.

### 3.2 Expected artifact magnitude — central-average alone

A pure von-Neumann linear analysis of CD2 on a monotone Ψ ramp gives a
boundary error O(Δx) (CD2 face value lags the upwind cell-centre by
½(φ_d − φ_u)). On a smooth flow this stays bounded.

But Ψ is log-conformation; the *actual* stress field is C = exp(Ψ),
so a fluctuation δΨ of order 1 produces a fluctuation δC of order e ≈ 2.7.
A fluctuation δΨ ≈ 3 (one wavelength of an unstable mode amplifying
over O(100) steps at rate even 1.03) produces δC ≈ e³ ≈ 20.

If the actually-observed τ_xx is 130× too large, that requires δC ≈ 130,
i.e. δΨ ≈ ln 130 ≈ 4.87. In other words **about 5 e-folds of an
unstable mode in log-conformation space**. At an amplification rate of
1.03/step that is ~165 steps; at 1.10/step that is ~50 steps. Both
are quite achievable within a Cd-converged window.

If trace_C peaks at 60×, that is δΨ ≈ ln 60 ≈ 4.1 — same regime.

**The observed magnitudes are quantitatively consistent with a linear
instability in Ψ-space, with rate amplified by the Ψ→C exponential.**

### 3.3 Expected magnitude if the fallback were 1-sided upwind

1st-order upwind on the same O(200) faces gives a face value f = φ_1
(no contribution from the downwind cell). This is dissipative; it is
the **same recipe as M29b's full-cell Rusanov fallback on those faces**.
The only difference vs M29b would be that the **other** faces of the
ring cells (where the canonical 4-point stencil *is* available) get
MUSCL instead of being demoted to 1st-order by the whole-cell rule.

That difference is the +56 % Cd improvement M29c was *meant* to deliver
(MUSCL on more faces of the ring, while keeping a TVD fallback on the
truly-unreachable ones). The residual ~4 Cd gap to RheoTool that M29c
targeted (M29b reached Cd = 116.47 at this case) is plausible at this
order, *if and only if* the fallback is genuinely TVD.

### 3.4 Could the −1571 magnitude be CD2 alone, or does it require feedback?

CD2 in isolation, on a *linear* scalar field, is neutrally stable, not
growing. The factor ~14 explosion of Cd in 1 run cannot come from a
neutrally-stable scheme on its own. The amplifier sequence is

1. CD2 face leaks a high-wavenumber component into Ψ that 1st-order
   upwind would have killed.
2. The leak rides on the boundary-cell ring → its spatial structure
   has a strong projection onto the 1st polymer-stress eigenmode of
   the cylinder (this is essentially the analogous instability
   mechanism documented in Fattal–Kupferman 2005 for the unstabilised
   linear-conformation tensor).
3. Ψ → C = exp(Ψ) **exponentially amplifies** any δΨ. A δΨ of order
   ln 130 ≈ 4.9 → trace_C 130× larger.
4. The BSD subtraction in the polymer stress (τ_p = (η_p/λ)(C − I) for
   Oldroyd-B, or with BSD: τ_p − τ_BSD) feeds back into the momentum
   equation. With a sign-correct closure τ_p > 0 → drag increase. With
   an inflated and asymmetric C distribution (more polymer stretch on
   the leeward shoulder than physical), the **pressure-drop signature
   on the cylinder reverses**, which is exactly the Cd sign flip
   (−1571 instead of +120).
5. The LBM TRT-RLB step then propagates the spurious stress field back
   into u, closing the loop.

So the verdict is: **CD2 is the *trigger*, but the observed magnitude
(Cd = −1571) requires the Ψ→C exponential amplifier plus the
stress-feedback loop.** Without those amplifiers the same patch on a
pure scalar test would have shown only an O(1) artifact, not O(10).

---

## 4. Is the M29c-validate Department's diagnosis correct?

### 4.1 Claim "fallback is central-average"

**Verdict: CORRECT.** The function literally returns `(upwind +
downwind) / 2`. The misleading name `_oneSided_2d` does not change the
arithmetic. (See §1.1.)

### 4.2 Claim "anti-TVD"

**Verdict: CORRECT.** §2.2 exhibits a monotone profile whose minimum is
pushed below its original minimum by a single Euler-explicit step at
ν = 1/2. CD2 violates Harten's TVD criterion at the boundary cell.
The Sweby region for a TVD limiter requires ψ(r) ≤ 2r AND ψ(r) ≤ 2;
CD2 corresponds to ψ ≡ 1 *regardless of r*, which exits the TVD region
for r < 0.5 (where it gives ψ = 1 > 2r). At a sharp shoulder where r is
small or negative, this is the relevant regime.

### 4.3 Claim "unconditionally unstable for hyperbolic advection without dissipation"

**Verdict: STRONG VERSION OVERSTATED, WEAK VERSION CORRECT.** Pure CD2
on linear advection is *neutrally stable* in von Neumann analysis; it
is "unconditionally unstable" only in the colloquial sense that it
admits unbounded oscillations under any nonlinear coupling. In the
present setting (Ψ→C exponential + stress feedback + LBM coupling) the
nonlinear coupling is severe enough that "unconditional instability"
is operationally true. The Department's framing is correct in effect
but slightly loose in pure-PDE terms.

### 4.4 Is the diagnosis complete?

**Verdict: NEARLY.** The Department correctly identifies CD2 as the
trigger but does *not* explicitly mention the **Ψ → C = exp(Ψ)**
exponential amplifier, which is what converts a δΨ ≈ 5 instability
into a 130× stress overshoot. Without this amplifier the magnitudes
would not match. The exponential amplifier is intrinsic to
log-conformation transport (Fattal–Kupferman 2004); it is *the
reason* log-conformation requires a TVD advection scheme, while
direct-conformation can sometimes tolerate CD2.

The BSD subtraction and TRT-RLB feedback are secondary; they could in
principle compound the instability but a clean CD2 → Ψ-growth → C-explosion
→ Cd-flip narrative is sufficient.

---

## 5. Is Cd = −1571 consistent with the diagnosis?

§3 shows: **yes**, but only because of the Ψ→C amplifier. A naïve
"CD2 is unstable, so Cd is wrong by O(1)" would give Cd in [50, 200],
not −1571. The factor 14× plus sign flip requires that the boundary
CD2 leak excites a polymer-stress mode in a region whose contribution
to drag is dominant (leeward shoulder) and that the Ψ→C exponential
turns δΨ ≈ 5 into δC ≈ 130. Both are well-documented in the
log-conformation literature.

**Confidence**: medium-high. We have not run a Fourier-amplification
calculation on the exact discrete operator stack; we have argued by
order-of-magnitude. But the orders match cleanly: ln 130 ≈ 4.9, ln 60
≈ 4.1, both 5 e-folds of a discrete-mode instability over a
several-tens-of-steps window.

---

## 6. Proposed fix — M29c-v2 (1-sided upwind fallback)

### 6.1 Replacement helper

```julia
@inline function _fvfd_muscl_superbee_face_value_oneSided_2d(upwind, downwind)
    # 1-sided upwind: ZERO slope reconstruction when the far-upwind
    # cell is unavailable (solid / OOB). This is the canonical TVD-safe
    # fallback for MUSCL at boundary-adjacent cells (Toro 2009 §13;
    # LeVeque 2002 §6).
    return upwind
end
```

Only the **body** changes. The dispatcher
`_fvfd_muscl_superbee_guarded_face_value_2d` and the per-face logic in
the kernel are unchanged. The fix is **one line**.

### 6.2 Why this is TVD

In the MUSCL reconstruction
`f = φ_u + ½ ψ(r) (φ_d − φ_u)`
the slope `½ ψ(r) (φ_d − φ_u)` is the part that lifts the scheme to
2nd order. Setting `f = φ_u` is equivalent to imposing `ψ ≡ 0` at the
boundary face, which trivially lies inside the Sweby region (ψ(r) ∈
[0, min(2r, 2)] for r > 0; ψ(r) = 0 for r ≤ 0). Hence 1st-order
upwind is the **minimum-dissipation TVD reconstruction**.

It is also exactly what M29b did at those faces (via the full-cell
Rusanov fallback) — so M29c-v2 reproduces M29b's behaviour on the
*unreachable* faces while keeping MUSCL on the *reachable* ones,
which is the original M29c design intent.

### 6.3 Expected outcome on the R=30 Wi=1.0 case

- Cd should land near M29b's 116.47, possibly slightly higher (because
  MUSCL is now active on faces of the ring cells that M29b also
  demoted to 1st-order). The +56 % gain M29b delivered at lower R
  should remain.
- τ_xx_max and trace_C peak should match M29b's values within a few
  percent.
- No sign flip, no factor-14 overshoot.

If Cd lands inside [110, 130] the patch is recoverable as designed.
If Cd still drifts above 140 there may be a *second* issue (e.g. a
sign-error in `canonical_usable` for one of the four faces), but
nothing in §1.2 suggests that is the case — the four-face logic looks
symmetric and correct.

### 6.4 What the patched helper would look like in full

```julia
# Replace lines 490-492 in src/fvfd/operators_2d.jl

@inline function _fvfd_muscl_superbee_face_value_oneSided_2d(upwind, downwind)
    return upwind  # 1-sided upwind = MUSCL with zero slope; TVD by construction
end
```

The function signature is preserved (`downwind` is now unused but
kept for the dispatcher's calling convention). The compiler will
elide the unused argument.

---

## 7. CUBISTA boundary alternative (optional, lower priority)

The CUBISTA scheme (Alves, Oliveira & Pinho, *Int. J. Numer. Meth.
Fluids* 41:47–75, 2003) is a normalised-variable (NV) bounded high-
resolution scheme designed *specifically* for viscoelastic transport.
Its face value is given in the NV formulation by

    φ̂_f = φ̂_C                                            if φ̂_C ∉ [0, 1]
    φ̂_f = (7/4) φ̂_C                                       if 0   ≤ φ̂_C < 3/8
    φ̂_f = (3/4) φ̂_C + 3/8                                 if 3/8 ≤ φ̂_C < 3/4
    φ̂_f = (1/4) φ̂_C + 3/4                                 if 3/4 ≤ φ̂_C ≤ 1

where the normalised variable is
    φ̂ = (φ − φ_U) / (φ_D − φ_U)
with U = far-upwind, D = downwind, C = upwind. CUBISTA satisfies the
Convection Boundedness Criterion (CBC) and lies inside the TVD region.

**At a boundary cell where φ_U is unavailable**, CUBISTA's standard
recipe is exactly the same as MUSCL's: set the upstream slope to zero,
i.e. take φ_U = φ_C, giving φ̂_C = 0/0 → undefined → the scheme falls
back to 1st-order upwind φ_f = φ_C. (Alves et al. 2003 §4.)

So CUBISTA's boundary fallback is **identical to the MUSCL fix of
§6**. The choice between MUSCL-superbee+1st-order-fallback and
CUBISTA is therefore independent of the boundary question; it is a
choice about *which 2nd-order scheme runs in the interior*. Superbee
is sharper (more compressive); CUBISTA is smoother. For Oldroyd-B at
moderate Wi, both have been used successfully in the literature.

For the immediate Cd recovery, **stay with MUSCL-superbee + 1st-order
fallback (§6)**. Migrating to CUBISTA is a separate research question.

---

## 8. Confidence levels

| Claim                                                                | Confidence | Basis                                                       |
|----------------------------------------------------------------------|------------|-------------------------------------------------------------|
| Fallback returns CD2 = (upwind + downwind) / 2                       | HIGH       | Direct quote of lines 490–492                                |
| CD2 violates TVD at the boundary cell                                | HIGH       | Sweby region argument; explicit counterexample §2.2          |
| CD2 amplifies checkerboard modes                                     | HIGH       | von Neumann analysis (amplification factor = 1 for kΔx = π)  |
| The leeward shoulder is exactly where the fallback fires             | MEDIUM-HIGH| Geometric argument; matches M29-tau-compare audit            |
| Cd = −1571 magnitude is consistent with CD2 + Ψ→C exponential        | MEDIUM     | Order-of-magnitude; ln 130 ≈ 4.9 = 5 e-folds of a linear mode|
| Department's "anti-TVD" verdict is correct                           | HIGH       | §4.1, §4.2                                                   |
| Department's diagnosis is complete enough to act on                  | MEDIUM-HIGH| Misses the Ψ→C exponential as the magnitude amplifier        |
| 1-line fix (§6) recovers Cd ≈ 116–130 on R=30 Wi=1.0 F64             | MEDIUM-HIGH| Reduces to M29b on boundary faces; MUSCL on interior faces   |
| The +56 % gain M29b delivered is preserved by M29c-v2                | MEDIUM     | Logical: M29c-v2 ⊇ M29b on the unreachable-face subset       |
| CUBISTA boundary handling = MUSCL+1st-order at boundary cells        | HIGH       | Alves et al. 2003 §4, standard NV recipe                     |

---

## 9. TL;DR

The M29c patch falls back to **CD2 (= central average)** at every
face whose canonical 4-point MUSCL stencil reaches into a solid or
out-of-domain cell. The Department's diagnosis ("central-average
fallback, anti-TVD") is **correct**. The magnitude of the FAIL
(Cd = −1571) is **consistent** with CD2 once one accounts for the
Ψ → C = exp(Ψ) exponential amplifier intrinsic to log-conformation
transport — pure CD2 alone would only give an O(1) error.

The fix is **one line**: change the body of
`_fvfd_muscl_superbee_face_value_oneSided_2d` from
`(upwind + downwind) / 2` to `upwind`. This makes the fallback
**1st-order upwind**, which is TVD by construction, matches what
M29b did on those faces, and preserves MUSCL-superbee on every face
where it can run. Expected Cd post-fix: ≈ 116–130 on R=30 Wi=1.0 F64,
i.e. the +56 % gain M29b delivered plus a small extra closure from
running MUSCL on additional ring-cell faces.

**Recommendation**: implement §6 as M29c-v2 and re-run the canary.
Do **not** abandon M29c.
