# M30 Phase 2b port audit — Claude solo (step 1)

Date    : 2026-05-20
Engine  : Claude (Anthropic Opus 4.7, 1M)
Mission : adversarial step 1 — answer Q1-Q6 BEFORE consulting Codex.

Inputs read :
- `src/kernels/dsl/bricks.jl` (lines 1-660, focus 349-550)
- `src/kernels/li_bb_2d_v2.jl` (lines 1-166)
- `src/drivers/viscoelastic_logfv_2d.jl` (lines 270-560)
- `src/kernels/li_bb_2d.jl` (lines 260-340, `precompute_q_wall_cylinder`)
- `bench/scratch/m30_phase2a_interpBB_claude/m30_phase2a.jl` (lines 1-520)
- `bench/scratch/m30_phase2a_interpBB_codex/m30_phase2a_codex.jl` (lines 75-225)
- `bench/viscoelastic_audit/M30_PHASE2A_VERDICT.md`
- `bench/viscoelastic_audit/M30_PHASE2B_VERDICT.md`
- `.orchestrator/memory/engineer.md` (full, 850 lines)

## Q1 — production code locations

| Item | Path | Lines |
|---|---|---|
| `_bouzidi_fl_post_value` helper | `src/kernels/dsl/bricks.jl` | 404-419 |
| `ApplyBouzidiFLPostCollide` struct + `required_args` + `phase` | `src/kernels/dsl/bricks.jl` | 421-425 |
| `ApplyBouzidiFLPostCollide` `emit_code` | `src/kernels/dsl/bricks.jl` | 426-550 |
| `_TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_SPEC` | `src/kernels/li_bb_2d_v2.jl` | 56-61 |
| `wall_bc=:bouzidi_fl` dispatch (public wrapper) | `src/kernels/li_bb_2d_v2.jl` | 123-135 |
| `_fused_trt_libb_v2_guo_field_step!(::Val{:bouzidi_fl}, ...)` | `src/kernels/li_bb_2d_v2.jl` | 152-165 |
| Kwarg threading in driver | `src/drivers/viscoelastic_logfv_2d.jl` | 475 (call site uses default; kwarg passed via outer driver, +3 LOC per port verdict) |

## Q2 — Phase 2a q-convention (Claude ref vs Codex ref)

Claude `compute_q_inner` (`m30_phase2a.jl:173-195`):
```
solve r(x_f + s·c_q) = R_in for s ∈ (0, 1]
return s (smallest positive root in range)
```
Codex `cut_fraction` (`m30_phase2a_codex.jl:82-98`):
```
solve |x_f + s·c_q|² = radius² for s ∈ [0, 1]
return s
```
Both use the **identical convention**: `q = s = fraction of link length from the
fluid cell x_f along c_q TO the wall hit point`. q ∈ (0, 1], q=0.5 ≡ halfway-BB.
This is Lallemand-Luo 2003 eq. 18 / Bouzidi-Firdaouss-Lallemand 2001 eq. 18.

**Q2 AGREE: YES** (cross-validated 2026-05-20 in M30_PHASE2A_VERDICT.md §"Setup chosen").

Phase 2a Claude branches (lines 444-470):
```
if q <= 0.5:
    f[i,j,qb] = 2q·f_post[i,j,q] + (1-2q)·f_post[i_ff,j_ff,q]
              − 2·W_q·rho_w·(c_q·u_w)/cs²
else:                                        # q > 0.5
    f[i,j,qb] = (1/(2q))·f_post[i,j,q] + ((2q-1)/(2q))·f_post[i,j,qb]
              − (1/q)·W_q·rho_w·(c_q·u_w)/cs²
```
Phase 2a Codex matches structurally (lines 182-192). Both use **lag-0** (current
step post-collision) on EVERY f read, including `x_ff`.

## Q3 — production q-convention vs Phase 2a

`precompute_q_wall_cylinder` (`src/kernels/li_bb_2d.jl:276-307`) computes:
```
t = solve |x_f + t·c_q − c_wall|² = R² in (0, 1]
q_wall[i,j,q] = t
```
where `x_f = (i−1, j−1)` in the `:idx`-frame convention (M31 audit memory).
This is **exactly the same `q`** as Phase 2a Claude/Codex: fraction from x_f
along c_q to the wall.

**No 1-q swap risk**: q_wall stores the INWARD distance (from fluid cell to wall
along the link), not the outward distance from the boundary cell.

In `_bouzidi_fl_post_value` (`bricks.jl:404-419`), the parameter `qw` is fed
directly from `q_wall[i, j, q]` with no transformation, so the convention
threads through correctly.

**Q3 MATCH: YES.**

## Q4 — branch logic + lag

### q ≤ 0.5 branch

Phase 2a (both engines) on the q ≤ 0.5 branch reads:
- `f_post[i, j, q]` (lag-0, current step post-collision)
- `f_post[i_ff, j_ff, q]` (**lag-0**, current step post-collision at neighbour)

Production `_bouzidi_fl_post_value` (line 410):
```
return 2qw·f_q_here + (1−2qw)·f_q_ff + delta
```
where `f_q_here` is sourced from `f2_here = f_out[i, j, 2]` (lag-0, just
written by `CollideTRTDirectGuoField`) and `f_q_ff = f_in[i2_ff, j2_ff, 2]`
(line 448).

`f_in` is the buffer-swap partner of `f_out` (driver line 554:
`f_in, f_out = f_out, f_in`). At the entry of the next LBM step, `f_in[*, *, *]`
holds what `f_out[*, *, *]` held at the end of the previous step (i.e.
**post-collision t-1, lag-1**).

**LAG MISMATCH on x_ff**: production reads lag-1 at x_ff, Phase 2a reads lag-0.

The brick docstring explicitly admits this:
```
"the q <= 0.5 far-fluid term uses lag-1 f_in at x_f - c_q, with halfway-BB
 fallback when that neighbour is unavailable"
```
And the M30_PHASE2B_VERDICT.md §"Diff summary" item 2 confirms:
```
"The lag-1 on x_ff is unavoidable in a single-pass fused kernel; matches the
 architectural pattern of ApplyLiBBPrePhase"
```
**This is wrong.** `ApplyLiBBPrePhase` is a PRE-collision substitution that
reads lag-1 BY CONSTRUCTION (it acts on `f_in` populations that have just been
pulled and which are lag-1 post-collision values from the previous step — this
is the correct semantics for a pre-phase fix). `ApplyBouzidiFLPostCollide` is a
POST-collision overwrite where the canonical formula expects lag-0 on the
far-fluid term. The two cases are not symmetric.

### q > 0.5 branch

Phase 2a reads:
- `f_post[i, j, q]` (lag-0)
- `f_post[i, j, qb]` (lag-0, opposite pop at SAME cell)

Production reads:
- `f_q_here = f_out[i, j, q]` (lag-0 ✓, snapshotted at top of `emit_code`)
- `f_qbar_here = f_out[i, j, qbar]` (lag-0 ✓, snapshotted at top — the
  snapshot of all 9 directions is at lines 430-437 BEFORE any overwrite)

**Match: YES.** The q > 0.5 branch is correctly lag-0 on both reads.

### Summary of Q4

| Branch | Phase 2a x_f lag | Phase 2a x_ff lag | Production x_f lag | Production x_ff lag | Match |
|---|---|---|---|---|---|
| q ≤ 0.5 | lag-0 | lag-0 | lag-0 ✓ | **lag-1 ✗** | **NO** |
| q > 0.5 | lag-0 (x_qb at same cell) | n/a | lag-0 ✓ | n/a | YES |

Moving-wall δ scaling: q ≤ 0.5 uses `+ delta` (no rescale); q > 0.5 uses
`+ delta * inv_two_qw` (= `δ/(2q)`). Computation check:
- Phase 2a q > 0.5 correction: `-(1/q)·W·ρ·(c·u_w)/cs²`. For q_dir=2 (East,
  W=1/9, cs²=1/3, c·u_w=u_wx): `-(1/q)·(1/3)·ρ·u_wx = -(1/(3q))·ρ·u_wx`.
- Production: `delta · inv_two_qw = -(2/3)·ρ·u_wx · (1/(2q)) = -(1/(3q))·ρ·u_wx`. ✓

Moving-wall correction MATCHES on both branches.

## Q5 — moving-wall correction term

Production reads `uw_link_x[i, j, q]`, `uw_link_y[i, j, q]` which encode the
wall velocity along link q.

For the cylinder benchmark (stationary cylinder), the driver populates these
with zeros, so `delta_q = 0` and the term vanishes harmlessly. No buggy term
fires for u_w = 0.

For the Phase 2a Couette validation, `uw_link_*` would carry the rotor
velocity, and as shown above the algebra reduces to the Phase 2a literal
`-(1/q)·W·ρ·(c·u)/cs²` (q > 0.5 branch) or `-2·W·ρ·(c·u)/cs²` (q ≤ 0.5 branch).

The `rho_w = ρ_out[i, j]` read at line 427 is **lag-1** (the WriteMoments brick
runs AFTER ApplyBouzidiFLPostCollide — so ρ_out is the previous step's density
at that moment). Phase 2a Claude uses `rho_w = s.rho[i, j]` which is the
**current** macroscopic from the moment loop. Phase 2a Codex same.

This is a **second lag mismatch**: `rho_w` should be lag-0 (current step
density, just computed by `Moments`), but production reads `ρ_out` which is
still the lag-1 value.

However the impact on the Couette validation is small because:
- Couette is steady-state → ρ varies slowly.
- The moving-wall correction term is O(Ma) on the wall-velocity scale.

For the cylinder at Wi=1, ρ on the front shoulder has a steady-state pressure
peak; lag-1 ρ_w introduces a slow temporal bias rather than a static error.
Not the primary suspect for the late NaN at step 36k.

**Q5 verdict**: present, but rho_w is lag-1 instead of lag-0 (minor secondary
issue; primary issue is Q4 lag-1 on x_ff).

## Q6 — root cause hypothesis

**Class (B) Lag mismatch on x_ff in q ≤ 0.5 branch.**

Evidence:
1. The brick docstring and verdict explicitly state the lag-1 design choice.
2. Phase 2a Claude and Codex BOTH use lag-0 on x_ff (cross-validated GO).
3. The Phase 2a Couette analytical bench would NOT have caught this because
   at steady-state Couette ALL populations are time-invariant ⇒ lag-1 ≡ lag-0
   to machine precision. The Couette validation does not exercise time-varying
   regimes.
4. For the cylinder benchmark with Wi=1 R=30, the front-shoulder stagnation
   region builds up a transient pressure peak over ~10⁴ steps. A lag-1 error
   on x_ff at q ≤ 0.5 cut links amplifies as ∂p/∂t accumulates near the wall.
5. The β=1 (Newtonian) NaN at step 40k is the SMOKING GUN: zero polymer means
   no nu_p coupling, so the only difference vs default `:halfwayBB` (which is
   stable at the same Re per regression tests) is the Bouzidi-FL substitution
   itself. Class (E) "algorithm fragility at Wi=1" cannot explain a Newtonian
   NaN; class (A) q-convention bug would NaN immediately on the first
   asymmetric cut link; class (C) q→0 division would NaN at any step where
   such a link exists; class (D) race condition — refuted, no read-after-write
   on the same memory location.

Confidence: **HIGH**. The lag mismatch is documented in the port verdict
itself; the only question is whether the architectural justification ("lag-1
is unavoidable in a single-pass fused kernel") is correct.

**Counter-argument**: is the lag-1 on x_ff truly unavoidable?

Looking at the kernel structure:
- All cells process in parallel on GPU.
- Cell (i, j) writes `f_out[i, j, *]`; cell (i_ff, j_ff) = (i±1, j±1, …) writes
  `f_out[i_ff, j_ff, *]` IN THE SAME KERNEL LAUNCH.
- Reading `f_out[i_ff, j_ff, *]` from cell (i, j) would be a cross-thread
  data race (lag-0 on a neighbour's `f_out` is non-deterministic — it depends
  on GPU thread scheduling whether the neighbour has written yet).

So lag-0 on x_ff genuinely requires either:
(a) a two-pass kernel: pass 1 = collision writes f_out; pass 2 = BC reads
    f_out (now safe) and writes back to f_out. This DOUBLES the launch
    overhead but is the canonical pattern for Bouzidi-FL on GPU (used by
    rheoTool's OpenFOAM dual-pass scheme).
(b) An auxiliary `f_post_buffer` that holds the just-finished collision
    values, separate from `f_out`. Cost: extra allocation and one extra
    write per cell.

Option (a) is the standard fix. The port chose to bypass this by using
`f_in[i_ff, j_ff, q]` (lag-1), which is mathematically inconsistent with the
canonical formula and causes the late NaN.

## Proposed minimal fix

Convert `ApplyBouzidiFLPostCollide` into a **two-pass** scheme:
1. Build SPEC #1 = `[PullHalfwayBB, SolidInert, Moments, CollideTRTDirectGuoField,
   WriteMoments]` (i.e. the existing `_TRT_LIBB_V2_GUO_FIELD_SPEC` minus the
   BC brick).
2. Build SPEC #2 = `[ApplyBouzidiFLPostCollideTwoPass]` — a new brick that
   reads `f_out` (now safe because pass 1 is done) at both x_f and x_ff and
   writes back to `f_out`.
3. The `_fused_trt_libb_v2_guo_field_step!(::Val{:bouzidi_fl}, ...)` dispatch
   launches kernel 1 then kernel 2 sequentially.

Alternative (less invasive): keep `ApplyBouzidiFLPostCollide` as-is but rename
to `ApplyBouzidiFLPostCollideHalfwayFallback` and document it as a degraded
single-pass variant valid ONLY for q ≈ 0.5 (where lag-1 ≡ lag-0 to leading
order). Currently used by production should be flagged as INCORRECT for
arbitrary q.

LOC estimate: ~50-80 LOC for the two-pass refactor; preserves the brick DSL
discipline.

---

## Memory candidates

1. `feedback_bouzidi_fl_single_pass_lag_trap` — A single-pass GPU kernel
   CANNOT implement Bouzidi-FL interpBB at arbitrary q without a lag mismatch
   on the far-fluid x_ff term. The canonical formula needs lag-0 (current
   step post-collision) on BOTH x_f and x_ff; reading neighbours from `f_out`
   is a cross-thread race; reading from `f_in` is lag-1 ⇒ mathematically
   inconsistent. Two-pass scheme is mandatory. Validated by Couette is
   misleading because steady-state ⇒ lag-1 ≡ lag-0. Late NaN at Wi=1 R=30
   cylinder step ~36k-40k is the smoking gun.

2. `feedback_phase2a_steady_state_misses_lag_bugs` — Phase 2a Couette
   analytical bench validated Bouzidi-FL GO at 5000 steps because steady-state
   makes the lag-1 vs lag-0 distinction invisible. Any future BC port whose
   correctness depends on a multi-cell pop-at-time-t reference MUST be tested
   on a transient benchmark (e.g. start-up Couette, oscillating cylinder, or
   wall-impulse) to expose lag mismatches. The Couette → cylinder leap was
   the bug-revealing transition for M30 Phase 2b.
