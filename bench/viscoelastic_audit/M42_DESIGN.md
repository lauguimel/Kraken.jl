# M42 — MUSCL boundary relaxation via two-pass split — DESIGN

Date    : 2026-05-23
Branch  : `dev-viscoelastic`
Scope   : implementation design (NOT code). Boss decides whether to spawn
          M42-impl or M42-prework after reviewing this document.
Author  : M42-design Department (Claude inline).

Inputs (required reading verified during drafting) :
- `bench/viscoelastic_audit/M41BIS_FALLBACK_PROBE_VERDICT.md` — locus
- `bench/viscoelastic_audit/M30_PHASE2B_AUDIT_VERDICT.md` — two-pass spec
- `bench/viscoelastic_audit/M29C_POSTMORTEM_EMPIRICAL_VERDICT.md` — CD2 anti-TVD
- `bench/viscoelastic_audit/M29C_POSTMORTEM_MATH_VERDICT.md` — Sweby/TVD math
- `bench/viscoelastic_audit/M29C_V2_BC_AUDIT_VERDICT.md` — H-LATE-STIFF at j=1
- `src/fvfd/operators_2d.jl` L470-565 — MUSCL scheme + fallback
- `src/kernels/logconformation_fv_2d.jl` L1154-1289 — psi_advect dispatch
- `src/drivers/viscoelastic_logfv_2d.jl` L200-230, L416, L664 — driver kwarg
- `src/kernels/dsl/bricks.jl` (Bouzidi-FL two-pass template)

---

## 1. Problem statement

### 1.1 M29b fallback (currently shipping)

`src/fvfd/operators_2d.jl` L523-532, inside
`_fvfd_upwind_scalar_advective_rhs_2d(::Val{:muscl_superbee})`,
demotes the ENTIRE cell to `:rusanov` (1st-order upwind everywhere)
when ANY of the 8 cross-shape stencil neighbours (±1 and ±2 along the
i and j axes) is solid OR if the cell sits within 2 LU of the domain
edge (`i ≤ 2 ∨ i ≥ Nx − 1 ∨ j ≤ 2 ∨ j ≥ Ny − 1`).

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

This is **whole-cell, not per-face**. A cell adjacent to a single
solid neighbour falls back on all four of its faces. The fallback is
conservative TVD (donor-cell `:rusanov`) but **dissipative on every
face of the ring**, even those whose canonical 4-point stencil is
fully fluid.

### 1.2 M41-bis empirical locus

M41-bis (post-hoc probe of an M29b R=30 Wi=1 β=0.59 dump) demonstrates
this fallback band is the dominant locus of the polymer-stress signal :

| Field   | n (band) | mean ratio band/bulk | q95 ratio | max ratio |
|---------|----------|----------------------|-----------|-----------|
| τ_xx    | 344      | **18.83**            | 12.96     | 1.62      |
| tr(τ_p) | 344      | **24.37**            | 11.90     | 1.53      |
| τ_yy    | 344      | **64.80**            | 60.13     | 0.81      |
| τ_xy    | 344      | **23.19**            | 19.87     | 0.94      |

The cylinder-side near-solid band is **0.33 %** of fluid cells but
carries 19-65× bulk mean polymer stress. The polymer field is
**first-order advected exactly where it matters**.

### 1.3 M29c-v2 failure mode (CRITICAL NEGATIVE EVIDENCE)

The M29c-v2 attempt to relax the fallback (per-face MUSCL +
`(upwind+downwind)/2` CD2 fallback on the broken face) failed
catastrophically (Cd = −1571 on R=30 Wi=1 β=0.59 F64 100k). Two
mechanisms identified post-hoc:

1. **M29c-postmortem-math + empirical**: the CD2 fallback is
   **anti-TVD**. On a checkerboard adjacent to the wall, CD2
   produces an 8.8 % negative undershoot of φ within 11 steps. The
   log-conformation Ψ → C = exp(Ψ) exponential amplifier then
   converts a δΨ ≈ 5 instability into δC ≈ 130, large enough to
   reverse the cylinder pressure-drop sign. **Diagnosis: CD2 trigger
   + Ψ → C amplifier + LBM/BSD feedback loop.**

2. **M29c-v2-BC-audit + LOCATE**: at step 92,200 F64 (102,800 F32)
   the NaN actually surfaces in `rho` at **j=1 south open wall**,
   ~4R upstream of the cylinder, NOT at the cylinder surface. The
   audit Departments (Claude + Codex adversarial) ruled out the
   originally hypothesised "Ψ_solid → fluid spurious-flux"
   mechanism (face velocity `ue` is identically zero at solid-fluid
   faces). The actual mechanism is **H-LATE-STIFF**: M29c-v2's
   reduced numerical diffusion exposes the **physical** Wi=1, β=0.59,
   λ=6000 LU elastic feedback at a remote open-wall location.

### 1.4 What M42 must achieve

- Replace the whole-cell `:rusanov` demotion at the **cylinder-side**
  ring with a per-face higher-order reconstruction that does NOT
  read solid-cell values along the broken axis.
- Use a **TVD-preserving** boundary face value (1-sided upwind /
  zero slope, NOT CD2 — the M29c lesson is forcing).
- **Preserve** the conservative `:rusanov` fallback on the
  **open-wall ring** (`j ∈ {1, 2, Ny−1, Ny}`, and optionally
  `i ∈ {1, 2, Nx−1, Nx}` for inlet/outlet) — the H-LATE-STIFF
  failure mode is open-wall-side, not cylinder-side, and the M29b
  numerical diffusion at j=1 is load-bearing for stability.
- Architecturally avoid same-step lag-1 reads in the new branch
  (the M30 P2b lesson). Use a two-pass kernel split: pass-1 writes
  ψ_out at non-fallback (bulk MUSCL) cells; pass-2 reads pass-1's
  lag-0 outputs at fallback-band cells and writes ψ_out there.

The locus (M41-bis) tells us **where** to act. The math (M29c
postmortems) tells us **what** the safe face-value recipe is. The
H-LATE-STIFF (M29c-v2-BC-audit) tells us **where NOT to act**. The
two-pass (M30 P2b) tells us **how to structure the kernel launches**.

---

## 2. Architectural choice : two-pass kernel split

### 2.1 Why a single-pass relaxation is wrong

A single-pass kernel reads `phi` (lag-1, last step's input) and
writes `phi_out` (current step). If the boundary-relaxation branch
needs **other fluid cells' current-step values** to compute a biased
slope (e.g. `phi[i+1, j_out]` because `phi[i-1, j]` is solid), then
either :
- the read is from `phi_in` (lag-1) → mathematically inconsistent
  with the bulk MUSCL's lag-0 expectation at the same iteration; or
- the read is from `phi_out[i+1, j]` in the same kernel → **cross-thread
  race**, unbounded data hazard on GPU.

This is the exact failure class M30 P2b documented for Bouzidi-FL at
q ≤ 0.5 (and that we now know causes a quantitatively different but
structurally identical class of late-time NaN).

For Ψ-advection at the cylinder boundary, a one-sided reconstruction
needs at most `phi[i_n, j]` and `phi[i, j_n]` (the two fluid
neighbours along the axis), both of which are **available in `phi_in`
at lag-0 if we explicitly synchronise**. The two-pass split with
KernelAbstractions.synchronize() between passes accomplishes this.

### 2.2 Two-pass spec

**Pass 1** (existing, unchanged in behaviour) :
- Launch the existing `fvfd_advect_upwind_2d_kernel!` with
  `advection_scheme = Val(:muscl_superbee)`.
- Each (i, j) thread reads `phi` (lag-0 input to this Ψ-advection
  step), evaluates the M29b fallback predicate, and either writes
  full MUSCL ψ_out OR writes 1st-order `:rusanov` ψ_out. **No
  behaviour change in this pass.**
- After kernel finish: `KernelAbstractions.synchronize(backend)`.

**Pass 2** (NEW, written for M42) :
- Launch a new kernel `fvfd_advect_muscl_relax_boundary_2d_kernel!`.
- Each (i, j) thread evaluates a NEW predicate
  `is_cylinder_band(i, j, is_solid, Nx, Ny)` (defined §3).
- If the predicate is FALSE → no-op (pass-1's value of `phi_out` is
  preserved). Most threads are no-op : 0.33 % of fluid cells are
  cylinder band per M41-bis on R=30.
- If TRUE → compute the per-face one-sided MUSCL face values per
  §4, and OVERWRITE `phi_out[i, j]`.
- Pass-2 reads `phi` (same lag-0 input as pass 1) — NOT `phi_out`.
  This is the M30 P2b discipline : **lag-0 read, write to the
  same output array** ; the synchronise ensures pass-1 finished
  but the read source is still `phi`, so there is **no
  cross-thread race** with pass-1's writes.
- After kernel finish: `KernelAbstractions.synchronize(backend)`.

**Why pass-2 doesn't read `phi_out`** : pass-2 is replacing the
fallback at cylinder cells with a higher-order reconstruction of
the SAME advection step from the SAME input `phi`. There is no
need to read pass-1's outputs — they are equivalent at every
non-cylinder-band cell to what pass-2 would compute, and at
cylinder-band cells pass-1's output is exactly what we are
replacing. Pass-2 is a strict OVERWRITE on a sparse subset.

This is **strictly stronger** than M30 P2b's Bouzidi-FL spec, which
needed lag-0 reads at `x_ff`. Here pass-2 only needs lag-0 at
`phi[i, j]` and immediate fluid neighbours — the cleanest possible
case of the two-pass discipline.

### 2.3 Why the M29c-v2 NaN doesn't recur

M29c-v2 NaN'd because :
(a) CD2 fallback at the cylinder admits high-frequency content into
    Ψ, which Ψ → C = exp(Ψ) amplifies (M29c-postmortem-math).
(b) Reduced numerical diffusion at the open south wall lets the
    physical Wi=1 elastic feedback reach rho-positivity violation
    at j=1 by step 92,200 (M29c-v2-BC-audit H-LATE-STIFF).

M42 prevents (a) by using **1-sided upwind (zero-slope MUSCL)** at
the cylinder face, which is TVD by construction (slope ≡ 0 lies
inside the Sweby region). M42 prevents (b) by **keeping the M29b
`:rusanov` whole-cell fallback at the open-wall ring** (j ≤ 2,
j ≥ Ny − 1, and optionally i ≤ 2, i ≥ Nx − 1) — this is the same
numerical diffusion M29b had, which empirically reaches 200k+ steps.

The two-pass architecture additionally eliminates the
single-kernel lag-1 / race-on-`phi_out` class entirely.

---

## 3. Boundary classification

### 3.1 The three zones (per M41-bis (c))

| Zone | Definition | Pass-1 fallback | Pass-2 action |
|------|------------|-----------------|---------------|
| **bulk** (MUSCL active) | not in cylinder band AND not in open-wall band | none (full MUSCL) | no-op |
| **cylinder band** (NEAR-SOLID) | fluid cell with `is_solid[i±1, j] ∨ is_solid[i, j±1] ∨ is_solid[i±2, j] ∨ is_solid[i, j±2]` AND not in open-wall band | `:rusanov` | **OVERWRITE** with one-sided MUSCL (§4) |
| **open-wall band** | `i ≤ 2 ∨ i ≥ Nx − 1 ∨ j ≤ 2 ∨ j ≥ Ny − 1` | `:rusanov` (PRESERVED) | no-op (preserves pass-1's `:rusanov`) |

If a cell is **in both** zones (cylinder corner overlapping with
domain edge — rare for R=30 with R_offset bsd=1.0 but possible at
extreme bsd_fraction < 0.2) → classify as open-wall (more
conservative). The open-wall zone is a **strict superset**
overriding cylinder-band for the pass-2 decision.

### 3.2 The new predicate (pass-2 kernel)

```julia
@inline function is_cylinder_band_2d(
    is_solid, i, j, Nx, Ny,
)
    # Exclude open-wall band (preserved as :rusanov).
    if i <= 2 || i >= Nx - 1 || j <= 2 || j >= Ny - 1
        return false
    end
    # Cylinder-band test : cross-shape, arms 1 and 2.
    # (Arms 1 are sufficient to trigger one-sided fallback ;
    #  arms 2 are the M29b fallback predicate ; we use arms 2
    #  for stencil-compatibility with §4.)
    return is_solid[i - 2, j] | is_solid[i - 1, j] |
           is_solid[i + 1, j] | is_solid[i + 2, j] |
           is_solid[i, j - 2] | is_solid[i, j - 1] |
           is_solid[i, j + 1] | is_solid[i, j + 2]
end
```

### 3.3 Robustness to driver kwargs

- `wall_bc=:halfwayBB` (default) — `is_solid` is the standard
  cylinder mask. Predicate works.
- `wall_bc=:bouzidi_fl_twopass` (M30 P2b) — `is_solid` mask is the
  same ; only the Bouzidi q-aware streaming changes. Predicate works.
- `embedded_geometry=:circle` — same mask convention. Works.
- `bsd_fraction < 1` — body-force radius is smaller than cylinder
  radius ; `is_solid` is still the geometric cylinder. Predicate
  is per-cell and doesn't see BSD. Works.

---

## 4. One-sided MUSCL face-value formula

### 4.1 Derivation

MUSCL face value (standard, per `_fvfd_muscl_superbee_face_value_2d`
L483-488) :

```
f_face = upwind + ½ · ψ_superbee(r) · (downwind − upwind)
where r = (upwind − far_upwind) / (downwind − upwind)
```

When `far_upwind` is unavailable (solid or OOB along the axis), set
**slope = 0** :

```
f_face = upwind             (1-sided "upwind" = zero-slope MUSCL)
```

This is :
- **TVD by Sweby**: corresponds to ψ ≡ 0, trivially inside
  [0, min(2r, 2)] for r > 0 and inside [0, 0] for r ≤ 0.
- **Identical to M29b's `:rusanov`** on that single face.
- **NOT identical to M29c-v2's `(upwind+downwind)/2`** (which was
  anti-TVD, see M29c-postmortem-math §2).

### 4.2 Per-face dispatch (pass-2 kernel core)

```julia
# East face (sign of ue determines upwind/downwind direction)
ue = ux_face[i + 1, j]
phie = if ue >= 0
    upwind   = phi[i, j]
    downwind = phi[i + 1, j]          # available (cylinder-band excl. open-wall)
    canonical_usable_e_pos = (i > 2) && !is_solid[i - 1, j] &&
                                       !is_solid[i - 2, j]
    if canonical_usable_e_pos
        _fvfd_muscl_superbee_face_value_2d(
            phi[i - 1, j], phi[i, j], phi[i + 1, j],
        )
    else
        upwind                        # 1-sided upwind (zero-slope MUSCL)
    end
else
    upwind   = phi[i + 1, j]
    downwind = phi[i, j]
    canonical_usable_e_neg = (i + 2 <= Nx) && !is_solid[i + 1, j] &&
                                              !is_solid[i + 2, j]
    if canonical_usable_e_neg
        _fvfd_muscl_superbee_face_value_2d(
            phi[i + 2, j], phi[i + 1, j], phi[i, j],
        )
    else
        upwind
    end
end
# Repeat for west, north, south analogous (with j-axis predicate for n/s).
```

Note: pass-2 is entered when AT LEAST ONE of the 8 cross-shape
neighbours is solid. For each of the 4 faces, the per-face
canonical_usable test is evaluated **independently**. Out of the 4
faces, typically 0-2 will have non-canonical stencil ; the others
get full MUSCL. This is strictly better than M29b's all-or-nothing
demotion.

### 4.3 Velocity at solid-fluid faces is zero (load-bearing)

Per M29c-v2-BC-audit Codex argument : `ux_face[i+1, j]` at any
solid-adjacent face is identically zero
(`_fvfd_xface_average_or_zero_2d` operators_2d.jl L142-146 returns
zero if either side is solid). So even when `phi[i+1, j]` is read
at a solid-adjacent face, the contribution to the flux divergence
is `ue * phie = 0 * (anything) = 0`. **The one-sided MUSCL
formulation in §4.2 is safe : even if a `phi[i+1, j]` term accidentally
read a solid cell value, multiplication by zero kills the flux.**

### 4.4 Corner cells (both axes blocked)

A cylinder-staircase corner can have `is_solid[i+1, j] ∧ is_solid[i, j+1]`
simultaneously. In that case both east and north faces fall back to
1-sided upwind independently. No coupling needed — the per-axis
formulation in §4.2 handles it.

For more extreme cases (e.g. a 1-cell isthmus between two cylinders
— not a real benchmark geometry), if BOTH `is_solid[i+1, j]` AND
`is_solid[i-1, j]` are simultaneously true, then both east AND
west fall back. The remaining bulk update through n/s faces is the
only non-trivial advection ; this is mathematically well-posed.

---

## 5. New DSL spec / dispatch

### 5.1 File layout

`src/fvfd/operators_2d.jl` is 1190 LOC > 700 LOC threshold per brief.
Add new file **`src/fvfd/muscl_boundary.jl`** containing :

- `is_cylinder_band_2d` (predicate, ~12 LOC).
- `_fvfd_muscl_superbee_oneSidedAxis_face_value_2d` helper (~5 LOC)
  – returns `upwind` (named for clarity ; could inline).
- `fvfd_advect_muscl_relax_boundary_2d_kernel!` (the pass-2
  KernelAbstractions kernel, ~80 LOC : 4-face per-face dispatch
  reusing existing `_fvfd_muscl_superbee_face_value_2d` for the
  canonical-stencil sub-cases).
- `fvfd_advect_muscl_relax_boundary_2d!` (the kernel launch
  wrapper, ~15 LOC).
- New scalar-advection dispatch `fvfd_advect_upwind_2d!(...,
  advection_scheme=:muscl_superbee_relax)` that runs pass-1 then
  pass-2 (~15 LOC : a thin wrapper that calls the existing
  `fvfd_advect_upwind_2d!` with `:muscl_superbee` then calls the
  new pass-2 launcher).
- `include("muscl_boundary.jl")` line in `src/Kraken.jl` or
  `src/fvfd/operators_2d.jl` (top, after operator includes).

### 5.2 Symbol whitelist updates

Three sites accept `:muscl_superbee_relax` as a new valid value :

1. `src/fvfd/operators_2d.jl` L472 :
   `scheme in (:rusanov, :muscl_superbee, :muscl_superbee_relax)`.
2. `src/drivers/viscoelastic_logfv_2d.jl` L227 :
   same whitelist update.
3. Any analogous whitelist in V&V drivers if exposed (e.g.
   `bench/viscoelastic_logfv/*.jl`) — search for `:muscl_superbee`
   and update where the user-facing kwarg is validated.

### 5.3 Dispatch wiring

In `src/fvfd/operators_2d.jl` add a third method of
`fvfd_advect_upwind_2d!` (or a one-line conditional inside the
existing) that catches `:muscl_superbee_relax` and routes through
pass-1 + sync + pass-2. Alternative : add a `Val{:muscl_superbee_relax}`
to `_fvfd_advection_scheme_val` and let dispatch happen at kernel
launch. Either is acceptable ; the wrapper approach is simpler and
more readable (~10 LOC).

For the **symmetric-tensor** path (`fvfd_sym2_advect_upwind_2d!`
operators_2d.jl L648-688), the relax-scheme dispatch is already
forwarded by passing `advection_scheme` through to the scalar
launcher — **no new code path needed** at the sym2 level as long as
the scalar launcher in §5.2 handles the new symbol. This is the
beauty of the existing layered design.

### 5.4 Driver kwarg

`src/drivers/viscoelastic_logfv_2d.jl` already accepts
`advection_scheme::Symbol=:rusanov` and threads it down to the
psi_advect calls (L416, L664). No change beyond the L227
whitelist. Users can call
`run_viscoelastic_logfv_2d(... ; advection_scheme=:muscl_superbee_relax)`
to opt in.

Default stays `:rusanov` until G5 validation gate passes (§7).

---

## 6. Smoke test design

Smoke tests live in `test/test_muscl_boundary_relax.jl` (NEW file)
and `test/runtests.jl` include.

### 6.1 Smoke 1 — cylinder R=8 Wi=0.1 200 steps CPU F64

Cheap (~ a few seconds). Asserts :
- No NaN at any step.
- Mass conservation : `|sum(ρ) − sum(ρ_init)| < 1e−10`.
- Cd is finite and within ±10 % of `:rusanov` reference at same
  R, Wi (this is sanity, not validation : R=8 is sub-grid for Cd
  benchmarking).
- `max|Ψ_xx|` does not grow exponentially (track a 3-checkpoint
  ratio ; flag if `max_step100 / max_step50 > 10`).

### 6.2 Smoke 2 — cylinder R=8 Wi=1.0 1000 steps CPU F64 (proxy for the 92k NaN)

Cheaper proxy for the M29c-v2 R=30 step-92,200 failure mode. Same
acceptance criteria as Smoke 1. The 1000 steps × R=8 scaling gives
roughly the same dimensionless time as the 92,200 steps × R=30
case (factor (30/8)^2 ≈ 14, so 1000 / 14 ≈ 70 R=30-equivalent
steps — well short of 92k, but enough to catch immediate
checkerboard or rho-positivity violations).

### 6.3 Smoke 3 — NEGATIVE CONTROL : open-wall relaxation enabled

**Critical to validate the open-wall conservative scope (§3).**
Add a `:muscl_superbee_relax_openwall` symbol that toggles the
zone check to INCLUDE the open-wall band in pass-2 (purely for
this test ; not user-facing). Expected behaviour : reproduces the
M29c-v2 step ~92k NaN at j=1 south wall on R=30 Wi=1 β=0.59 (or a
scaled-down R=8 version, ~5k steps).

The test :
- Run R=8 Wi=1.0 β=0.59 F64 with both
  `:muscl_superbee_relax` AND `:muscl_superbee_relax_openwall`.
- Assert: `:muscl_superbee_relax` does NOT NaN at 5000 steps.
- Assert: `:muscl_superbee_relax_openwall` **DOES** NaN before
  5000 steps.
- If the negative control fails to NaN at R=8 (too small a stiff
  signal), demote to a `@test_skip` with a TODO to run R=30 on
  Aqua, but keep the symbol in the public surface for §7
  validation.

This is the dispositive evidence that the cylinder-only scope is
load-bearing, not over-conservative.

### 6.4 Smoke 4 — bulk equivalence

When NO cell is in the cylinder band (e.g. solid-free channel),
`:muscl_superbee_relax` should produce **bit-identical** output to
`:muscl_superbee`. Test on a 32×32 periodic channel with random
initial Ψ.

### 6.5 Smoke 5 — V&V hierarchy
Per `[[feedback_localize_via_vv_hierarchy]]`. Add a row to
`bench/viscoelastic_audit/vv_log.csv` (or equivalent) for the new
scheme on a canonical sub-bench (e.g. Couette analytical bench at
R=8 if applicable, otherwise just the cylinder smoke).

---

## 7. Validation gate (G5)

### 7.1 Matrix

Single Aqua A100 F64 submission, 4-cell matrix :

| Case | R  | Wi  | β   | scheme                 | wall_bc      | max_steps | Acceptance |
|------|----|-----|-----|------------------------|--------------|-----------|------------|
| G5-1 | 30 | 1.0 | 0.59| `:muscl_superbee_relax`| `:halfwayBB` | 100 000   | Cd ∈ [122, 124] (rT 120.40, +2 % margin ; closes the +5.5 Cd gap of M29b 116.47) |
| G5-2 | 30 | 0   | 1.0 | `:muscl_superbee_relax`| `:halfwayBB` | 100 000   | Cd within ±1 % of M41 R=30 `:halfwayBB` Newtonian 132.08 |
| G5-3 | 30 | 0.1 | 0.59| `:muscl_superbee_relax`| `:halfwayBB` | 100 000   | Cd within ±1 % of rT 130.43 (no regression at low Wi) |
| G5-4 | 60 | 0.1 | 0.59| `:muscl_superbee_relax`| `:halfwayBB` | 100 000   | No NaN at 100k (R=60 envelope test, demonstrates the scheme scales) |

### 7.2 Decision tree

- All 4 PASS → ship `:muscl_superbee_relax`. Promote default in a
  follow-up patch (don't auto-default — keep `:rusanov` default
  until at least one β-Wi sweep has been re-run).
- G5-1 fail HIGH (Cd > 124) → suggests over-relaxation at the
  cylinder ; spawn M42-debug to investigate per-face slope sign.
- G5-1 fail LOW (Cd ∈ [116, 121]) → close-but-not-enough ; either
  iterate slope-limiter choice (replace 1-sided upwind by 1-sided
  minmod = ψ(r) = max(0, min(r, 1)) on the broken-axis face, a
  trivial 1-line tweak) and re-G5, or accept M42 as partial and
  open a parallel research direction.
- G5-2 fail → Newtonian regression : a bug in the cylinder-band
  predicate or in the per-face dispatch. Block ship.
- G5-3 fail → low-Wi regression : same root cause class as G5-2.
- G5-4 NaN → scaling failure : suggests a parameter-dependent
  instability not caught at R=30. Investigate before broader rollout.

### 7.3 Cost estimate

Each cell ≈ 85 s A100 F64 (per M29c-validate `21588436.aqua`
calibration). Four cells, serial submission, ≈ 6-7 min wall on
Aqua. One PBS job total.

---

## 8. Risk analysis

### 8.1 Risk 1 — One-sided slope ≡ 0 is too dissipative (under-correction)

**Symptom**: G5-1 fail LOW (Cd ∈ [116, 121] only ; not quite 122).
**Cause**: 1-sided upwind on the broken face is identical to
`:rusanov` ; the only Cd uplift comes from the other 3 faces
getting MUSCL where M29b had whole-cell `:rusanov`. If the 1
broken face per cell carries 60 % of the Cd-contributing gradient
(consistent with M41-bis τ_xx near-solid q95 = 13× bulk q95),
then the uplift is only ~40 % of full MUSCL gain.
**Mitigation**: upgrade fallback to **1-sided minmod**
`ψ(r) = max(0, min(r, 1))` on the broken-axis face. Uses two
fluid cells (`upwind`, `downwind`) instead of three. Still TVD ;
strictly less dissipative than zero-slope. 5 LOC change. Already
documented in M29c-postmortem-math §7 as an alternative.

### 8.2 Risk 2 — Open-wall band is too wide (4 cells), under-correcting near the cylinder if cylinder is near the inlet

**Symptom**: A cylinder placed at e.g. `Lup = 5R` upstream extent
might be within the j ≤ 2 or i ≤ 2 band at its leading edge → those
cells go to `:rusanov` even though they are far from the south
wall (only 1 LU from inlet, not from south wall).
**Cause**: the open-wall predicate is geometric/total, not
per-edge.
**Mitigation**: §3.2 predicate uses combined edge check ; could be
refined to track per-edge if needed. For the canonical R=30
Lup=Ldn=15R, the cylinder is ≥ 450 LU from i=1, well outside the
j ≤ 2 band. So the cylinder ring is fully covered by pass-2.
**Likelihood**: LOW for benchmarks ≥ Lup = 10R. Document the
limitation ; suggest a per-edge predicate in M43 if needed.

### 8.3 Risk 3 — Open-wall H-LATE-STIFF still fires under M42

**Symptom**: G5-4 (R=60 Wi=0.1 100k) NaN at j=1 south wall.
**Cause**: M42 preserves the M29b open-wall `:rusanov` (and
M29b reached 200k+ steps at R=30 Wi=1). But R=60 Wi=0.1 has different
stiffness ratio (lower Wi → weaker elastic feedback, but R-doubling
→ longer integration, ~4× more steps to reach the same dimensionless
time). If R=60 Wi=0.1 is in the H-LATE-STIFF envelope of M29b
already (untested), M42 won't help and won't hurt at j=1.
**Mitigation**: M42 is **not designed** to fix H-LATE-STIFF.
G5-4 is a regression check, not a feature test. If R=60 NaN's at
j=1 with `:muscl_superbee_relax`, we have *also* gained data on
M29b's H-LATE-STIFF envelope (rerun M29b at R=60 to compare). The
NaN is then a known-known of `:halfwayBB` + open-wall, not an M42
defect.

### 8.4 Risk 4 — Cross-thread race in pass-2

**Symptom**: non-deterministic results, intermittent NaN, GPU-only
failure.
**Cause**: a coding bug in pass-2 that accidentally reads
`phi_out` instead of `phi`.
**Mitigation**: pass-2 kernel signature **does not accept
`phi_out` as a Const-array read source**, only as a writeable
output. Pin the read source explicitly in the kernel arguments
list (`@Const(phi)` for the read, `phi_out` for the write).
Compile-time-checked by KernelAbstractions.

### 8.5 Risk 5 — pass-2 over-fires (predicate too inclusive)

**Symptom**: bulk-equivalence smoke (§6.4) fails.
**Cause**: predicate accidentally returns true in a clean-bulk
channel.
**Mitigation**: §6.4 catches this immediately. Required pre-merge
gate.

### 8.6 Risk 6 — Polymer stress at the wall (FVFD `divu` term) inconsistent with pass-2 result

**Symptom**: subtle Cd offset (~1 %) that survives at all R, Wi.
**Cause**: the FVFD divergence term in the scalar advection RHS
`flux_div − phi[i, j] * divu` (operators_2d.jl L514, L564) uses
the **same** `divu` for bulk and boundary cells. Pass-2 doesn't
touch `divu`. If the divu computation was implicitly relying on
the dissipative `:rusanov` smoothing, it might not be smooth
enough for the MUSCL-relaxed pass-2 output.
**Mitigation**: this is an inherent limit of the FV scheme, not
M42-specific. M29b has the same issue at non-fallback cells. If
G5-1 lands at Cd ≈ 122 ± 0.5 the residual is within margin. If
it lands at Cd ≈ 120.5 ± 0.5 then a follow-up M42b may need to
re-examine the divu term — separate research question.

---

## 9. LOC budget + implementation effort

| Component | New file? | LOC est | Notes |
|-----------|-----------|---------|-------|
| `src/fvfd/muscl_boundary.jl` (NEW) | Y | ~120 | predicate + helper + kernel + launcher |
| `src/fvfd/operators_2d.jl` (whitelist + dispatch wiring) | N | ~15 | L470-475 whitelist + scheme dispatch wrapper |
| `src/Kraken.jl` (include) | N | 1 | `include("fvfd/muscl_boundary.jl")` |
| `src/drivers/viscoelastic_logfv_2d.jl` (driver whitelist) | N | ~5 | L226-228 update |
| `test/test_muscl_boundary_relax.jl` (NEW) | Y | ~150 | 5 smoke tests § 6 |
| `test/runtests.jl` (include) | N | 1 | `include("test_muscl_boundary_relax.jl")` |
| **Total** | | **~292** | comfortably inside orchestrator soft ceiling 500 LOC |

Implementation effort estimate :
- Engineer (Codex) write-time : ~90 min for 292 LOC at the
  observed Codex Kraken-skill cadence (kernel + dispatch + tests).
- Pre-merge testing : ~5 min on CPU F64 (5 smoke tests).
- G5 Aqua submission + post : 6-7 min wall + 10 min post-eval.
- Total wall : ~2 hours for impl ; ~30 min for G5 post-analysis.

---

## 10. NOT in scope

Explicitly OUT of M42 :

- **Open-wall side relaxation** — preserved as `:rusanov` for the
  H-LATE-STIFF reason (§1.3, §3.1). A future M44 may revisit the
  open-wall band with the same two-pass discipline, but ONLY after
  the south-wall stiffness mechanism is understood independently.
- **Higher-than-MUSCL boundary schemes** (WENO, CWENO, CUBISTA-3,
  RK4-spatial) — overkill ; we have a 1-line slope-zero recipe
  that is provably TVD. Migration to CUBISTA is a separate paper.
- **Adaptive switching based on local Wi** — out of scope ;
  log-conformation already handles Wi-stiffness ; per-cell
  scheme switching is a v0.4+ research direction.
- **3D extension** — out of scope for M42 (cylinder is 2D). A 3D
  analogue (`muscl_boundary_3d.jl`) is straightforward but
  separate ; needed for `viscoelastic_3d.jl` driver only when
  3D benchmarks become a target.
- **Replacing the FVFD `divu` correction term** — separate
  question (§8.6 Risk 6) ; if G5 lands within margin, ignore ;
  if not, spawn M42b.
- **Changing the default `advection_scheme` from `:rusanov` to
  `:muscl_superbee_relax`** — separate Boss decision after G5
  ships ; this design only adds the new option.
- **Re-validating the Bouzidi-FL two-pass (M30 P2b)** — M42 is
  orthogonal ; both two-pass mechanisms compose cleanly because
  they touch different fields (Bouzidi: f populations ; MUSCL-relax:
  Ψ fields) at different phases of the LBM step.

---

## Memory candidates (to be promoted by Boss if M42-impl ships green)

1. **feedback_muscl_boundary_two_pass_recipe** — When a per-face
   reconstruction needs other-cell reads at solid-adjacent fluid
   cells, the two-pass kernel split (lag-0 read of `phi`, pass-2
   overwrite of `phi_out` at the sparse band) is the canonical
   GPU-safe pattern. Compose with M30 P2b Bouzidi-FL two-pass —
   both run cleanly side-by-side (different fields, different
   phases). Single-pass with same-step writes is a guaranteed
   data-hazard ; single-pass with lag-1 reads is mathematically
   inconsistent.

2. **feedback_open_wall_diffusion_load_bearing** — Numerical
   diffusion at the open-wall band (j ≤ 2 ∨ j ≥ Ny − 1) is
   load-bearing for late-time stability of viscoelastic LBM at
   Wi=1 (M29c-v2-BC-audit H-LATE-STIFF). Any scheme that reduces
   diffusion at the open wall MUST be paired with an independent
   stability mechanism (boundary filtering, density-positivity
   limiter, etc.) BEFORE shipping. Cylinder-side relaxation is
   safe ; open-wall relaxation is not, on the current solver stack.

3. **feedback_locus_first_then_relax** — Validate the locus
   hypothesis BEFORE designing the relaxation (M41-bis →
   M42-design, not the other way round). M29c-v2 designed
   relaxation without confirming locus and shipped a system-wide
   per-face relaxation that NaN'd. M42 confines the change to
   0.33 % of fluid cells empirically pre-confirmed as the
   gradient hot-spot. Locus-first design halves the search space
   for failure mode hypotheses.

---

## Files referenced

- `src/fvfd/operators_2d.jl` — L470-565 (whitelist + MUSCL +
  fallback), L142-146 (`_fvfd_xface_average_or_zero_2d`).
- `src/kernels/logconformation_fv_2d.jl` — L1154-1289 (psi_advect
  dispatch).
- `src/drivers/viscoelastic_logfv_2d.jl` — L195-230, L416, L664
  (driver kwarg + threading).
- `src/kernels/dsl/bricks.jl` — Bouzidi-FL two-pass template.
- `bench/viscoelastic_audit/M41BIS_FALLBACK_PROBE_VERDICT.md`
- `bench/viscoelastic_audit/M30_PHASE2B_AUDIT_VERDICT.md`
- `bench/viscoelastic_audit/M29C_POSTMORTEM_EMPIRICAL_VERDICT.md`
- `bench/viscoelastic_audit/M29C_POSTMORTEM_MATH_VERDICT.md`
- `bench/viscoelastic_audit/M29C_V2_BC_AUDIT_VERDICT.md`

---

## End of design — recommended next step

**Spawn M42-impl** as Department + Codex engineer with this
document as the brief annex. Codex receives :
- The §5 file layout + §4 formula as the implementation spec.
- The §6 smoke tests as gate-before-PR.
- The §7 G5 matrix as ship-gate.
- The §8 risks as fall-back triggers.

No pre-work (M42-prework) is needed : the design has no
unresolved sub-questions. M41-bis confirmed the locus ; M29c
verdicts confirmed the TVD recipe ; M30 P2b confirmed the
architecture. M42-impl is a direct implementation mission.
