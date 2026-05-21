# M30 Phase 2b port audit — adversarial verdict

Date    : 2026-05-20
Engines : Claude (Anthropic Opus 4.7, 1M) + Codex (OpenAI gpt-5)
Step 1 (Claude solo) : `bench/viscoelastic_audit/M30_PHASE2B_AUDIT_CLAUDE.md`
Step 2 (Codex solo)  : `bench/viscoelastic_audit/M30_PHASE2B_AUDIT_CODEX.md`
Codex log : `.engineer_logs/M30P2b-audit-codex_*.log`

**Status: FULL AGREEMENT on root cause class (B) and proposed minimal fix.**

---

## Q1 — production code locations

| Item | Path | Lines | Agree |
|---|---|---|---|
| `_bouzidi_fl_post_value` helper | `src/kernels/dsl/bricks.jl` | 404-419 | YES |
| `ApplyBouzidiFLPostCollide` brick (struct + emit_code) | `src/kernels/dsl/bricks.jl` | 421-550 | YES |
| `_TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_SPEC` | `src/kernels/li_bb_2d_v2.jl` | 56-61 | YES |
| `wall_bc=:bouzidi_fl` dispatch | `src/kernels/li_bb_2d_v2.jl` | 123-134 + 152-164 | YES |

## Q2 — Phase 2a q-convention

| Engine | Ref formula | Source |
|---|---|---|
| Claude Phase 2a | solve `r(x_f + s·c_q) = R_in` for `s ∈ (0, 1]`, return `s` | `m30_phase2a.jl:173-194` |
| Codex Phase 2a  | solve `\|x + s·c\|² = R²` for `s ∈ [0, 1]`, return `s` | `m30_phase2a_codex.jl:82-98` |

Both conventions: `q = s = fraction-of-link from fluid x_f along c_q TO the
wall hit point`. q ∈ (0, 1]. q=0.5 ≡ halfway-BB.

Note: Claude Phase 2a's HEADER COMMENT (lines 29-31) describes the OPPOSITE
convention (`|x_b-x_w|/|x_b-x_f|`), but the actual implementation uses the
fluid-to-wall fraction — Codex flagged this comment/code mismatch. The code,
not the comment, is what matters for behaviour, and the code uses the
Lallemand-Luo / Bouzidi-FL fluid-to-wall convention.

**AGREE (cross-validated 2026-05-20): YES.**

## Q3 — production q-convention vs Phase 2a

`precompute_q_wall_cylinder` (`src/kernels/li_bb_2d.jl:276-307`) solves
`|x_f + t·c_q − c_wall|² = R²` for `t ∈ (0, 1]` and stores
`q_wall[i, j, q] = t` ⇒ identical to Phase 2a `s`. No `1-q` swap, no
complement.

`_bouzidi_fl_post_value(qw, ...)` (`bricks.jl:404-419`) consumes `qw` from
`q_wall` with no transformation. Convention threads through correctly.

**Q3 MATCH: YES** (Claude + Codex AGREE).

## Q4 — branch logic + lag

| Branch | Phase 2a x_f | Phase 2a x_ff | Production x_f | Production x_ff | Production x_qb | Match |
|---|---|---|---|---|---|---|
| q ≤ 0.5 | lag-0 | **lag-0** | lag-0 ✓ | **lag-1 ✗** | n/a | **NO** |
| q > 0.5 | lag-0 | n/a | lag-0 ✓ | n/a | lag-0 ✓ | YES |

- Production reads `f2_here..f9_here = f_out[i, j, *]` at the top of
  `emit_code` (lines 430-437). After `CollideTRTDirectGuoField` has written
  `f_out[i, j, *]` (lines 190-198 of `bricks.jl`), these are **lag-0** ✓.
- Production reads `f_q_ff = f_in[i_ff, j_ff, q]` (lines 448, 462, 476, 490,
  504, 518, 532, 546 of `bricks.jl`). After the previous-step buffer swap
  (`f_in, f_out = f_out, f_in` at driver line 554), `f_in` holds the
  previous step's post-collision values ⇒ **lag-1 ✗**.
- Phase 2a Claude lines 452-469: `f_xff = s.f_post[iff_i, iff_j, q_dir]` is
  lag-0 (current step post-collision).
- Phase 2a Codex lines 183-189: `fpost[link.xff, link.yff, k]` is lag-0.

The brick docstring (line 421) admits "the q <= 0.5 far-fluid term uses lag-1
f_in at x_f - c_q". The M30_PHASE2B_VERDICT.md justifies this as architecturally
unavoidable; the canonical formula requires lag-0 on x_ff (both Phase 2a
engines confirm).

Moving-wall δ scaling — q > 0.5 branch: `delta · inv_two_qw = -(2/3)·ρ·u_wx ·
(1/(2q)) = -(1/(3q))·ρ·u_wx`, matches Phase 2a `-(1/q)·W·ρ·(c·u)/cs² =
-(1/(3q))·ρ·u_wx`. ✓ (Claude + Codex independently algebra-checked.)

**Both engines AGREE: q ≤ 0.5 branch has a lag mismatch on x_ff (defect).
q > 0.5 branch correct.**

## Q5 — moving-wall correction term

- Production `delta_q` (lines 441, 455, 469, 483, 497, 511, 525, 539) includes
  the canonical `-(2/3 or 1/6)·ρ_w·(c_q·u_w)` term. Correctly scales by
  `inv_two_qw` in the q > 0.5 branch (line 417). Term VANISHES when u_w=0
  (stationary cylinder). No buggy term fires at u_w = 0.
- `rho_w = ρ_out[i, j]` (line 427) is read BEFORE `WriteMoments` runs (which
  is later in the SPEC: line 60). So `ρ_out[i, j]` holds the **previous
  step's density** at the moment of the read ⇒ **lag-1 ρ_w**.
- For the cylinder benchmark u_w ≡ 0, so the moving-wall δ term is identically
  zero regardless of ρ_w lag. The lag-1 ρ_w is a latent issue exposed only
  on moving-wall geometries (Couette-style); it does not contribute to the
  current cylinder NaN.

**Both engines AGREE: present + correctly guarded for u_w=0; secondary
lag-1 issue on ρ_w noted, not the dominant defect.**

## Q6 — root cause hypothesis

| Engine | Pick | Confidence | Notes |
|---|---|---|---|
| Claude | **B (lag mismatch on x_ff)** | HIGH | A refuted by Q3; C refuted by guard; D refuted by `f_in` vs `f_out` separation; E refuted by Newtonian β=1 NaN at step 40k. |
| Codex  | **B (single-pass lag/storage mismatch)** | medium-high | A ok by Q3; q=0 guarded; no cross-cell `f_out` race; stationary wall removes Q5 as primary. |

**AGREE: YES.** Both engines independently pick class (B) and explicitly
refute A, C, D, E with the same arguments. The β=1 Newtonian NaN at step
40k is the smoking gun that rules out polymer-related instabilities (E)
and any algorithm-fragile-at-high-Wi narrative.

Why the late NaN at step 36k-40k (not instant): Phase 2a Couette is steady
⇒ lag-1 ≡ lag-0 to machine precision (populations time-invariant). The
cylinder benchmark builds up a transient stagnation pressure peak over
~10⁴ steps; lag-1 reads at the front-shoulder cut links accumulate a
phase error that destabilises the LBM mass equation at sufficiently large
flux magnitude. Wi=0.5 stable because the elastic relaxation damps the
transient; Wi=1 + β=1 (no elastic damping) → NaN.

**Confidence (joint): HIGH.**

## Proposed minimal fix (both engines, independently)

Convert the single-pass `:bouzidi_fl` spec to **two-pass kernel launches**:
1. Pass 1 = existing `_TRT_LIBB_V2_GUO_FIELD_SPEC` (collision writes `f_out`).
2. Pass 2 = new `ApplyBouzidiFLPostCollideTwoPass` brick that reads `f_out`
   at **both** `x_f` and `x_ff` (now safe — pass 1 has globally synchronised)
   and overwrites `f_out[i, j, qbar]`.

Dispatch in `_fused_trt_libb_v2_guo_field_step!(::Val{:bouzidi_fl}, …)`:
launches kernel 1 then kernel 2 sequentially.

Additional small fix: read `rho_w` in pass 2 from the freshly-written
`ρ_out` to eliminate the secondary lag-1 ρ_w issue.

LOC estimate: ~50-80 LOC, no architectural change beyond the brick DSL.

---

## Memory candidates

1. `feedback_bouzidi_fl_single_pass_lag_trap` — A single-pass GPU kernel
   CANNOT implement Bouzidi-FL interpBB at arbitrary q without a lag
   mismatch on the far-fluid x_ff term. The canonical formula needs lag-0
   (current step post-collision) on BOTH x_f and x_ff; reading neighbours
   from `f_out` in the same kernel is a cross-thread race; reading from
   `f_in` is lag-1 ⇒ mathematically inconsistent. Two-pass scheme is
   mandatory. Late NaN at Wi=1 R=30 cylinder step ~36k-40k is the smoking
   gun. Validated cross-engine 2026-05-20 (Claude + Codex independently
   localise the same defect at `bricks.jl:448 (and analogues)`).

2. `feedback_phase2a_steady_state_misses_lag_bugs` — Phase 2a Couette
   analytical bench validated Bouzidi-FL GO at 5000 steps because
   steady-state makes the lag-1 vs lag-0 distinction invisible
   (populations time-invariant ⇒ f_in[i, j, q] ≡ f_out[i, j, q] to
   machine precision). Any future BC port whose correctness depends on
   a multi-cell pop-at-time-t reference MUST include a transient
   benchmark (e.g. start-up Couette, oscillating cylinder, or
   wall-impulse) to expose lag mismatches. The Couette → cylinder leap
   was the bug-revealing transition for M30 Phase 2b.

3. `feedback_dsl_brick_phase_ordering_subtle` — In the LBM DSL builder
   (`src/kernels/dsl/lbm_builder.jl`), bricks within the same phase
   (default `:fluid`) execute in spec order. `WriteMoments` placed AFTER
   `ApplyBouzidiFLPostCollide` means the brick reads `ρ_out[i, j]` as
   **lag-1** (previous step's value, written by the previous kernel
   launch). For correctness, any brick that needs the current density
   must either (a) accept the `ρ` local variable (set by `Moments`)
   directly, or (b) be placed AFTER `WriteMoments`. Found 2026-05-20
   in `_bouzidi_fl_post_value` rho_w read.

---

## Files

- `bench/viscoelastic_audit/M30_PHASE2B_AUDIT_CLAUDE.md`  (Claude step 1)
- `bench/viscoelastic_audit/M30_PHASE2B_AUDIT_CODEX.md`   (Codex step 2)
- `bench/viscoelastic_audit/M30_PHASE2B_AUDIT_VERDICT.md` (this synthesis)
- `.engineer_brief_M30P2b_audit_codex.md`                 (Codex brief, ephemeral)
- `.engineer_logs/M30P2b-audit-codex_*.log`               (Codex run log)
