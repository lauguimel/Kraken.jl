# M29c-v2 BC-helper audit — Department verdict (Claude + Codex adversarial)

Mission: M29c-v2-audit-bc-helpers.
Date: 2026-05-19.
Branch: `dev-viscoelastic`, working tree.
Sources of evidence:
- Independent Claude pass: `bench/viscoelastic_audit/M29C_V2_BC_AUDIT_CLAUDE.md`
- Independent Codex pass: `bench/viscoelastic_audit/M29C_V2_BC_AUDIT_CODEX.md`
  (log: `.engineer_logs/M29c-audit-codex_20260519_151441.log`)

The two passes were performed independently (Codex was instructed NOT
to read Claude's file before writing its own). Both reached a
`DIFF DOES NOT HOLD (or PARTIAL)` verdict via overlapping but
**non-identical** arguments. Codex's argument is **stronger** and
**disposes** of the question more cleanly; Claude's argument is
**consistent** but weaker. Both arrive at the same physical conclusion.

## Per-question comparison

### Q1 — `:muscl_superbee` entry point

| | Claude | Codex |
|---|---|---|
| Dispatch fn | `_fvfd_upwind_scalar_advective_rhs_2d(..., ::Val{:muscl_superbee})` | same |
| Line range | 537–629 | 537–629 |
| Kernel | `fvfd_advect_upwind_2d_kernel!` 631–653 | same (via call chain) |
| Cylinder call chain | driver → `logfv_advect_upwind_bc_aware_2d!` → `fvfd_sym2_advect_upwind_2d!` → `fvfd_advect_upwind_2d!` → kernel | same |

**Agreement: YES, fully identical.**

### Q2 — BC helpers and `is_solid` test

| helper | Claude: accepts `is_solid`? | Codex: accepts `is_solid`? | Claude: returns when neighbour solid? | Codex: same? |
|---|---|---|---|---|
| east  | NO | NO | `phi[i+1, j]` blind | YES |
| west  | NO | NO | `phi[i-1, j]` blind | YES |
| north | NO | NO | `phi[i, j+1]` blind | YES |
| south | NO | NO | `phi[i, j-1]` blind | YES |

**Agreement: YES, fully identical.** Both passes confirm DIFF's
structural read of the helpers.

### Q3 — `phi[solid] = 0` enforcement and call ordering

| | Claude | Codex |
|---|---|---|
| Zeroing site | `fvfd_advect_upwind_2d_kernel!` line 641–642 | same (line 641–642) |
| Per-substep frequency | Once per LBM step (Ψ-advection); NOT per constitutive substep | Same; constitutive kernel has no `is_solid` guard (line 417–455 of logconformation_fv_2d.jl) |
| Before/after MUSCL? | Same-kernel write to `phi_out` is to a DIFFERENT array from the read source `phi`; no race | Same conclusion: "fluid neighbor sees the pre-existing solid-cell value in `phi`, not a same-substep write to `phi_out`" |
| Algebraic Ψ_solid → 0 mechanism after the first LBM step | Claude: small O(λ⁻¹·dt·n_substeps) residue; Codex: zero gradient + zero Ψ + Oldroyd-B source = preservation at zero | Different but compatible derivations; both converge on "Ψ_solid stays ≈ 0" |

**Agreement: YES on the structural facts, with mildly different
derivations of why Ψ_solid stays at 0 after the first LBM step. Both
agree that the MUSCL stencil reads `Ψ_solid ≈ 0` via the unguarded
helper.**

### Q4 — algebraic flux trace (cylinder-west fluid cell)

Setup: cell `(i, j)` fluid, `is_solid[i+1, j] = true`, Ψ_xx(fluid)=5,
Ψ_xx(solid)=0.

| Wind sign | Claude `phie` | Codex `phie` | Claude M29b `phie` | Codex M29b `phie` |
|---|---|---|---|---|
| ue > 0 | 5 (limiter zero, returns upwind) | 5 (locally flat W=5 → r=0) | 5 | 5 |
| ue < 0 | 0 (oneSided fallback or limiter zero) | 0 (E=5 → r=-1 → limiter 0) | 0 (Rusanov reads `east_value`) | 0 |

**Agreement on numerical answers: YES.** Both Claude and Codex get
`phie = 5` for ue>0 and `phie = 0` for ue<0, in BOTH M29c-v2 and M29b.

**Disagreement on the killer step (THE CRITICAL POINT):**

Claude argued: `phie = 0` × `ue` = `ue × 0 = 0` → no Ψ advected into
the fluid through this face → DIFF's "advects Ψ=0 INTO the fluid"
sentence is the algebraic error.

**Codex argued harder**: at the solid-fluid face, `ue = ux_face[i+1, j]`
is ITSELF identically zero, **regardless of `phie`**, because
`_fvfd_xface_average_or_zero_2d` (operators_2d.jl:142–146) returns
`zero(T)` if either side is solid, and this helper is exactly what
`fvfd_cell_velocity_to_faces_2d_kernel!` line 215 uses to populate
`ux_face` at every interior face. So `ue * phie = 0 * (anything) = 0`,
including the case where `phie` would have been some non-zero
intermediate that one might worry about.

**Codex's argument is the stronger one.** Claude's argument relied on
the specific value `phie = 0` at the solid face (case B); Codex's
relies on `ue = 0` at every solid-adjacent face, which is a stronger
invariant of the code (not just an algebraic coincidence at a
specific cell). Both arguments agree on the final claim (no spurious
flux through the solid face), but Codex's is more robust and verified
directly. **The verdict here is: DIFF's mechanism is dead two ways
over.**

**Boss-level agreement classification: YES (numerical), with Codex
providing a stronger justification that Claude did not surface.**

### Q5 — Reconciliation with LOCATE

| | Claude | Codex |
|---|---|---|
| DIFF mechanism | DOES NOT HOLD | PARTIAL: structural read holds, dynamical claim doesn't |
| Specific DIFF error line | section 4, "advects Ψ=0 INTO the fluid" conflates face value with flux | same conclusion via the stronger ue=0 argument: "for the actual solid-fluid face, dt * ue * phie / dx = 0" |
| NaN root cause from this audit | UNKNOWN, but H-LATE-STIFF (reduced numerical diffusion → physical Wi=1 stiffness exposed) | UNKNOWN |
| Boss decision recommendation | escalate to M29d (force-coupling / rho-positivity audit) | "NaN root cause remains unknown from this audit" |

**Agreement on DIFF verdict: YES** (both classify as DIFF-falsified,
nominally Codex says PARTIAL because the helper does read `Ψ_solid`
as DIFF claimed, but this read is dynamically inert; Claude says
DOES-NOT-HOLD because the load-bearing claim is the dynamical
spurious-flux one, and that's wrong. Same physical conclusion, different
labelling.)

**Agreement on NaN root cause from this audit: YES, UNKNOWN.**

## Synthesised verdict

### What DIFF got right (uncontested by both passes)

The four BC helpers `_fvfd_bc_{east,west,north,south}_scalar_2d`
(operators_2d.jl:422–468) do NOT accept `is_solid` and do read
`phi[neighbour]` blindly when the neighbour index is in-domain,
including when that neighbour is a solid cell. DIFF section 1 H3 is a
correct structural read.

### What DIFF got wrong (the key claim that motivates the proposed fix)

DIFF section 4 claims:

> "M29c-v2 ... feeds the solid Ψ into MUSCL ... The flux `ue * phie`
> then advects Ψ = 0 INTO the fluid."

This is wrong on two independent grounds:

1. (Claude) When `ue < 0`, the MUSCL face value at the solid-fluid
   face is `phie = upwind = Ψ_solid ≈ 0`. The flux `ue * phie` is then
   `ue * 0 = 0`. Nothing is advected.

2. (Codex, stronger) The face velocity `ue` at any face with at least
   one solid-adjacent cell is identically zero, because
   `fvfd_cell_velocity_to_faces_2d_kernel!` populates `ux_face[i+1, j]`
   via `_fvfd_xface_average_or_zero_2d(ux, is_solid, i, i+1, j)` which
   returns `zero(T)` when `is_solid[i, j] || is_solid[i+1, j]`. The
   solid-fluid face is therefore a no-slip + no-flux face for the
   FVFD advection routine. The face-value `phie` (whatever it is) is
   multiplied by zero. **The DIFF mechanism cannot produce ANY
   spurious flux through the cylinder surface, regardless of what
   the BC helpers return.**

DIFF's proposed minimal fix (section 5, options A/B re-introducing a
±1 solid-neighbour band or guarding the BC helpers for solid) is
therefore **fixing a non-bug**. The fix would not be incorrect per se
(adding extra diffusion near the cylinder), but it would not address
the actual NaN root cause, and it would re-introduce M29b's numerical
diffusion that M29c-v2 was specifically designed to remove.

### What LOCATE saw and what it actually means

LOCATE's empirical observations (NaN at j=1 south wall ~4R upstream,
in `rho` first, at step 92,200 F64 / 102,800 F32, asymmetric,
preceded by a slow build-up of wake polymer stress) are consistent
with the **H-LATE-STIFF** hypothesis: M29c-v2's reduced numerical
diffusion (relative to M29b's H2 boundary band + similar bulk MUSCL)
allows the **physical** Wi=1, β=0.59, λ=6000 LU elastic feedback to
reach a magnitude that breaks the LBM density-positivity envelope at
a wall location remote from the cylinder. M29b's "stability" at 200k
steps is then better characterised as "numerical-diffusion masking of
late-stage stiffness" than as "correct closure".

This is consistent with the literature pattern at Wi ≈ 1 cylinder
benchmarks (see LOCATE section "Cross-check with M29c-asis").

### What the Boss should do next

Both passes agree the BC-helper audit alone cannot identify the NaN
root cause. The natural next mission is a force-coupling / LBM
rho-positivity audit (call it M29d):

- Q: what does the body force `fx_total, fy_total` look like at j=1
  south wall, ~4R upstream, in the last clean snapshot before
  `rho` loses positivity?
- Q: is the BSD correction (`logfv_bsd_correct_force_bc_aware_2d!`)
  computing a stable answer in the wake, or amplifying the polymer
  body force?
- Q: at LBM rho ~ 1 ± O(few × Ma²), what is the actual `divu`
  produced by the LBM macro-flow at the south wall, and is the FVFD
  `divu` correction term (line 627–628) consistent with it?
- Q: is the diagnostic asymmetry (south wall vs north wall) a
  consequence of the LBM TRT magic-parameter coupling with the
  half-way bounce-back? An asymmetric phenomenon at j=1 vs j=Ny
  would point at this.

These are independent of the BC-helper / MUSCL question.

## Confidence assessment

| Aspect | Confidence |
|---|---|
| Q1 (entry point) | HIGH — Claude and Codex agree on every line number |
| Q2 (helpers don't test `is_solid`) | HIGH — verbatim function bodies match |
| Q3 (`phi[solid] = 0` enforcement and ordering) | HIGH — both agree on the per-LBM-step ordering and the absence of an `is_solid` guard in the constitutive kernel |
| Q4 (algebraic trace) | HIGH — numerical answers identical; Codex's `ue = 0` argument STRENGTHENS Claude's `phie = 0` argument by an extra layer (any non-zero `phie` would still be killed by `ue = 0`) |
| Q5 (DIFF verdict + Boss recommendation) | HIGH on DIFF-falsified; HIGH on NaN-root-cause-unknown from this audit; MEDIUM on H-LATE-STIFF as the physical explanation (consistent with LOCATE but not directly tested) |

**Overall: HIGH confidence that DIFF's structural-correctness claim
is wrong and that DIFF's proposed minimal fix is targeted at a
non-bug.** The 5-question adversarial pass converged cleanly with no
load-bearing disagreement.

## Memory candidates

1. **department.md**: Adversarial pair on a STRUCTURAL claim works
   best when one engine searches for the **dispositive** invariant
   (Codex: "ue=0 at solid faces") rather than just confirming the
   other's argument. Encourage briefs to ask explicitly: "what
   would have to be true for DIFF to be wrong on a stronger ground
   than the one Claude already found?" or equivalent. This M29c
   audit found a 2nd kill-shot Claude missed.

2. **engineer.md** (for future Codex BC-audit briefs): always
   trace face-velocity lowering AND face-value computation. A
   "BC helper reads solid value" finding is incomplete without
   asking what face velocity multiplies that value. The
   `_fvfd_{x,y}face_average_or_zero_2d` pattern (returns 0 when
   either neighbour is solid) is project-wide and should be the
   default sanity check on any "solid neighbour read" hypothesis.

3. **department.md**: Postmortem-style verdicts (M29C_V2_DIFF) that
   propose a "minimal fix" without exhibiting a sentinel canary (a
   ≤30-line test that fails before the fix and passes after)
   risk fixing non-bugs. Future DIFF-style postmortems should be
   accompanied by a `bench/scratch/<mission>/canary_*.jl` that
   isolates the predicted bug; if the canary cannot be constructed,
   the mechanism is likely wrong.
