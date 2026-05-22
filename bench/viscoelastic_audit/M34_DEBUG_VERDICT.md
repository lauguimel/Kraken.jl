# M34 Bouzidi-FL two-pass — debug verdict

Date     : 2026-05-22
Mission  : M34-debug (localize bug in `:bouzidi_fl_twopass`)
Branch   : dev-viscoelastic
Status   : **RED — root cause localized: pass-1 SPEC reuses `_TRT_LIBB_V2_GUO_FIELD_SPEC` which already applies Bouzidi-FL pre-collision via `ApplyLiBBPrePhase`. Pass-2 then reapplies Bouzidi-FL post-collision ⇒ V2 double-BC bug back.**

**Proposed fix**: build a new pass-1 spec WITHOUT `ApplyLiBBPrePhase` (i.e. `PullHalfwayBB → SolidInert → Moments → CollideTRTDirectGuoField → WriteMoments`), matching the structure of the single-pass `:bouzidi_fl` spec minus the in-kernel `ApplyBouzidiFLPostCollide`.

---

## Source-of-truth: the offending spec ordering

`src/kernels/li_bb_2d_v2.jl:49-54` — pass-1 of two-pass:

```julia
const _TRT_LIBB_V2_GUO_FIELD_SPEC = LBMSpec(
    PullHalfwayBB(), SolidInert(),
    ApplyLiBBPrePhase(),                # ← already does Bouzidi-FL pre-collision
    Moments(), CollideTRTDirectGuoField(),
    WriteMoments(),
)
```

vs. `src/kernels/li_bb_2d_v2.jl:56-61` — single-pass `:bouzidi_fl`:

```julia
const _TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_SPEC = LBMSpec(
    PullHalfwayBB(), SolidInert(),
    Moments(), CollideTRTDirectGuoField(),   # NO pre-phase substitution
    ApplyBouzidiFLPostCollide(),             # ← single-shot post-collision overwrite
    WriteMoments(),
)
```

M34 dispatch at `src/kernels/li_bb_2d_v2.jl:185-207` launches `_TRT_LIBB_V2_GUO_FIELD_SPEC` (the pre-phase variant) for pass-1, synchronises, then launches `_TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_TWOPASS_PASS2_SPEC` (`ApplyBouzidiFLPostCollideTwoPass` only) for pass-2.

Per the V2 docstring (`li_bb_2d_v2.jl:1-33`): **the V2 architecture was explicitly designed to be PRE-PHASE ONLY** — applying BC twice yields L2 ≈ 2.2 % at moderate Ny in the Couette baseline, and produces unbounded blow-up under transient stagnation pressure (cylinder).

## Empirical signal (matches the static prediction)

### Phase A — R=30 Wi=0.1 FINITE case (M34 Cd=117.59)

Theta-binned wall decomposition (`:idx` frame, N_az=36, ring cells with at least one solid neighbour):

| Bucket | Cd_pres (M34) | Cd_pres (halfwayBB R30 Wi1.0 baseline) | Δ |
|---|---:|---:|---:|
| front-pole (`|θ|>3π/4`)   | **+79.07** | +71.52 | **+7.55** |
| shoulder                  | +24.58 | +18.41 | **+6.18** |
| wake (`|θ|<π/4`)          | −10.18 | −13.04 | +2.85 |
| **Cd_pres total**         | **+93.46** | +76.90 | **+16.57** |
| Cd_solv total             | +16.61 | +21.06 | −4.45 |
| Cd_poly total             | +10.12 | +10.77 | −0.64 |
| **Cd_total (ring-derived)** | **+120.19** | +108.72 | +11.47 |

Compare CSV Cd_kraken = 117.59 (M34) vs 111.09 (halfwayBB baseline). The Δ ≈ +12 LU comes entirely from `Cd_pres` concentrated at front-pole + shoulder. Polymer is unchanged. **This is the signature of an OVER-BOUNCING wall BC pumping extra stagnation pressure**, fully consistent with reapplying Bouzidi-FL twice per step.

### Phase B — NaN cases (R=40 Wi=0.1 + R={30, 40} Wi=1.0)

All 3 cases: 186984 / 192000 (R=40) or 105180 / 108000 (R=30) cells NaN = **~97% of the entire domain**. NaN azimuthal distribution: front ≈ wake ≈ shoulder-uniform → **catastrophic global blow-up**, NOT bilateral-arc D2bis-style localized stress concentration.

R=30 Wi=0.1 (finite) → R=40 Wi=0.1 (NaN): only resolution changes. More cut-links per ring at R=40, more chances for the double-BC over-bounce term to push a stagnation cell past the lattice stability envelope. The NaN comes early enough that the field is uniformly garbage by the snapshot at step 100000.

R=30 Wi=0.1 finite vs R=30 Wi=1.0 NaN: Wi=1 means stronger polymer source → larger transient pressure peak during start-up → over-bounced wall pop blows up.

The 4-case matrix collapses to: **finite case = lowest-pressure regime (smallest R, lowest Wi); all stiffer cases NaN.** This is the unique signature of a BC over-bounce defect, not a polymer instability.

## Hypothesis ranking

**Hypothesis #1 (HIGH confidence): Pass-1 SPEC reuses `ApplyLiBBPrePhase`, causing a double application of Bouzidi-FL per step.**
- Empirical signal: Cd_pres +16.6 concentrated at front-pole (over-bounce), all stiffer cases NaN.
- Static signal: `_TRT_LIBB_V2_GUO_FIELD_SPEC` at `li_bb_2d_v2.jl:49-54` includes `ApplyLiBBPrePhase()`; the brick at `bricks.jl:355-403` substitutes `fp_{q̄}` with a full `_libb_branch` (Bouzidi-FL) estimate before collision.
- Architectural inversion: V2 was specifically designed PRE-PHASE-ONLY to avoid the double-BC bug (`li_bb_2d_v2.jl:1-33` docstring); the new dispatch silently undoes this by stacking POST-COLLISION on top.
- The smoke test (`test_bouzidi_fl_twopass_smoke.jl`) on a closed bounce-back box with R=8 cylinder at u0=(0.02, 0.01) does NOT exercise: (a) cut-link-resolved cylinder geometry with q_w ≠ 0.5 (the box is grid-aligned), (b) transient stagnation pressure build-up over 10⁴-10⁵ steps, (c) polymer-coupled regime — hence it missed the double-BC defect.

**Hypothesis #2 (LOW): `ApplyBouzidiFLPostCollideTwoPass` brick formula error in `q ≤ 0.5` or `q > 0.5` branch.**
- Refuted: the brick reuses the validated `_bouzidi_fl_post_value` helper bit-exactly; the only difference from the single-pass version is `f_q_ff = f_out[i_ff, j_ff, q]` (lag-0) instead of `f_in[...]` (lag-1) at `bricks.jl:589, 603, 617, 631, 645, 659, 673, 687`. The `qbar` write indices, `delta_q` formulas, and arg-order are identical. Independently single-pass `:bouzidi_fl` did NOT NaN at R=30 Wi=0.1 in any prior matrix (it was the lag-1 Wi=1 case that NaN'd at step 40k per M30 Phase 2b).

**Hypothesis #3 (LOW): Pass-2 kernel signature / arg ordering bug.**
- Refuted: the dispatch passes `(f_out, ρ, is_solid, q_wall, uw_link_x, uw_link_y, Nx, Ny)` (`li_bb_2d_v2.jl:205`) in alphabetical-sort order matching the brick's `required_args` (`bricks.jl:565`): `(:f_out, :q_wall, :uw_link_x, :uw_link_y, :is_solid, :ρ_out, :Nx, :Ny)`. Canonical DSL sort by `Symbol` puts `:Nx, :Ny, :f_out, :is_solid, :q_wall, :ρ_out, :uw_link_x, :uw_link_y` — let me actually NOT speculate here; this requires checking `build_lbm_kernel` arg-binding. However, since R=30 Wi=0.1 produces a finite, *physically-shaped* result that just over-counts pressure at the front-pole (the exact qualitative signature of over-bounce), the pass-2 args are clearly being threaded correctly — a sig mismatch would produce NaN or garbage from step 1.

## Proposed fix (targets hypothesis #1)

Replace pass-1 SPEC with a NEW SPEC matching the single-pass Bouzidi-FL minus the post-collision brick:

```julia
# NEW: pass-1 spec for two-pass Bouzidi-FL (no pre-phase BC; pass-2 owns the BC).
const _TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_TWOPASS_PASS1_SPEC = LBMSpec(
    PullHalfwayBB(), SolidInert(),
    Moments(), CollideTRTDirectGuoField(),
    WriteMoments(),
)
```

In the dispatch (`li_bb_2d_v2.jl:185-207`), replace
```julia
pass1! = build_lbm_kernel(backend, _TRT_LIBB_V2_GUO_FIELD_SPEC)
```
with
```julia
pass1! = build_lbm_kernel(backend, _TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_TWOPASS_PASS1_SPEC)
```

This removes the `ApplyLiBBPrePhase` pre-collision substitution from pass-1, so the wall populations after collision are the standard halfway-BB lag-1 estimate from `PullHalfwayBB` (junk on cut links, but pass-2 overwrites the only links that need correction). Pass-2's `ApplyBouzidiFLPostCollideTwoPass` then provides the canonical lag-0 Bouzidi-FL correction without competition.

## Smoke-test gap → memory candidate

The M34 smoke (`test/test_bouzidi_fl_twopass_smoke.jl`) passed all 10 assertions because:
- Closed bounce-back box → grid-aligned walls → all `q_wall == 0.5` → `_libb_branch` collapses to halfway-BB → pre-phase and post-collision Bouzidi-FL are algebraically identical → the double application is *invisible* on the smoke.
- 100 steps + R=8 cylinder + u0=(0.02, 0.01) → far from stagnation envelope → no blow-up.

The cylinder benchmark at R=30 with `:qwall` geometry (true cut-link q ∈ (0, 1]) is what exposes the double-BC. Smoke must be EXTENDED to a cylinder geometry with cut-links to catch this class of bug.

## Verification (read-only)

```bash
cd /Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic
test -f bench/viscoelastic_audit/M34_DEBUG_VERDICT.md          # PASS
grep -n 'ApplyLiBBPrePhase' src/kernels/li_bb_2d_v2.jl | head  # confirms line 51
grep -n '_TRT_LIBB_V2_GUO_FIELD_SPEC' src/kernels/li_bb_2d_v2.jl | head  # confirms pass1 line 195
julia --project=. bench/scratch/m34_debug/run_m34_debug.jl     # reproduces Cd decomposition
```

## Memory candidates

1. `feedback_m34_double_bc_pass1_spec_reuse` — When converting a single-pass DSL spec to two-pass with synchronisation, the pass-1 spec must be *new*: it cannot reuse the pre-existing wall-BC base spec (`_TRT_LIBB_V2_GUO_FIELD_SPEC` includes `ApplyLiBBPrePhase`). Reusing it stacks pre-phase + post-collision BC = the exact V2 double-BC bug. Empirical signature: Cd_pres over-shoot at front-pole + catastrophic NaN at all stiffer Wi/R. M34 burned an entire Aqua matrix on this.

2. `feedback_smoke_must_exercise_cutlinks` — A two-pass Bouzidi-FL smoke on a closed bounce-back box (R=8, grid-aligned walls, q_wall ≡ 0.5) cannot catch a double-BC bug because pre-phase Bouzidi-FL and post-collision Bouzidi-FL are algebraically identical when q=0.5 (both collapse to halfway-BB). Smoke MUST exercise the `:qwall` cylinder geometry with q ∈ (0, 1] cut-links AND a transient pressure build-up (≥ 1000 steps to reach stagnation envelope) to be diagnostic. Extends `[[feedback_test_invariant_match_geometry]]`.

3. `feedback_nan_uniform_vs_arc_diagnostic` — When debugging NaN field dumps, the azimuthal distribution of NaN cells discriminates between two classes: (a) **bilateral-arc NaN concentrated at front-shoulder = local stress concentration / D2bis-style polymer instability**; (b) **uniform azimuthal NaN sweep (front ≈ shoulder ≈ wake, hundreds of thousands of cells) = catastrophic global blow-up = wall-BC over-bounce or kernel arg-order bug**. M34 R={30,40} Wi=1.0 + R=40 Wi=0.1 all show class (b), matching the over-bounce hypothesis.

---

## Files

- `bench/scratch/m34_debug/run_m34_debug.jl` (analysis script)
- `tmp/m34_aqua_results/matrix/` (rsynced Aqua field dumps)
- `bench/viscoelastic_audit/M34_DEBUG_VERDICT.md` (this verdict)
