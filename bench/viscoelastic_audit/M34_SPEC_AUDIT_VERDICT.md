# M34 spec audit — adversarial re-derivation verdict

Date    : 2026-05-22
Engine  : Claude (Anthropic Opus 4.7, 1M) — inline 2nd-pass adversarial
Inputs  : `bench/viscoelastic_audit/M30_PHASE2B_AUDIT_VERDICT.md` (1st pass),
          `M30_PHASE2B_AUDIT_CLAUDE.md`, `M30_PHASE2B_AUDIT_CODEX.md`,
          `bench/viscoelastic_audit/M34_BOUZIDI_FL_TWOPASS_VERDICT.md`,
          `src/kernels/dsl/bricks.jl` 280-700,
          `src/kernels/li_bb_2d_v2.jl` (full),
          `src/kernels/dsl/lbm_builder.jl`, `src/kernels/dsl/lbm_spec.jl`,
          `bench/scratch/m30_phase2a_interpBB_claude/m30_phase2a.jl`
          (lines 200-477; full `stream_interpBB!` and `collide!`),
          `bench/scratch/m30_phase2a_interpBB_codex/m30_phase2a_codex.jl`
          (lines 155-225).
Scratch : `bench/scratch/m34_spec_audit/derivation_q_le_half.md`.

**Recommendation: AMEND** the M30 P2b "Proposed minimal fix" — the spec
prescribes a pass-1 that itself contains a Bouzidi pre-phase, causing a
double Bouzidi correction at every cut link, identical pathology to the
double-BC bug that motivated the V2 refactor. The implementation
(`M34_BOUZIDI_FL_TWOPASS_VERDICT.md`) faithfully follows the spec — the
defect is in the spec, not in the port.

---

## (a) Independent first-principles derivation — q ≤ 0.5 branch

Lattice D2Q9. Wall hit point `x_w = x_f + q · c_q`, `q ∈ (0, 1]` measured
from fluid node `x_f` toward the wall. Far-fluid neighbour `x_ff = x_f − c_q`.

Bouzidi-Firdaouss-Lallemand 2001 eq. 18-19 (re-derived independently from the
quadratic interpolation of `f̃_q(x − q·c_q)` evaluated at the upwind side
of the wall):

```
q ≤ 0.5 :  f_q̄(x_f, t+dt) = 2q · f̃_q(x_f, t) + (1 − 2q) · f̃_q(x_ff, t)
                            − 2 W_q · ρ_w · (c_q · u_w) / c_s²        (A)
q > 0.5 :  f_q̄(x_f, t+dt) = (1/(2q)) · f̃_q(x_f, t)
                            + ((2q − 1)/(2q)) · f̃_q̄(x_f, t)
                            − (1/q) · W_q · ρ_w · (c_q · u_w) / c_s²  (B)
```

where `f̃ = f_post-collision` at the CURRENT step (lag-0). Phase 2a Claude
(m30_phase2a.jl:444-470) and Phase 2a Codex (m30_phase2a_codex.jl:175-192)
both implement this identically, both reading `s.f_post` (resp. `fpost`)
which is the buffer just written by `collide!` IN THE SAME STEP. **No
pre-collision Bouzidi substitution exists in Phase 2a** — collision writes
`f_post`, BC reads `f_post` and writes `s.f` (the new buffer).

Reads required: `f̃_q(x_f, t)`, `f̃_q(x_ff, t)`, `f̃_q̄(x_f, t)`, `ρ_w`.
All canonical reads are lag-0 ⇔ "current step post-collision, with NO prior
BC substitution".

## (b) Match / mismatch with M30 P2b verdict

| Claim (M30 P2b) | Status | Citation |
|---|---|---|
| Canonical formula q ≤ 0.5 reads lag-0 on x_f and x_ff | **MATCH** | M30 P2b verdict §Q4 lines 53-66; my derivation (A) |
| Production single-pass reads lag-1 on x_ff | **MATCH** | `bricks.jl:448` (and analogues `:462,:476,:490,:504,:518,:532,:546`) |
| Two-pass fix: pass-1 = `_TRT_LIBB_V2_GUO_FIELD_SPEC`, pass-2 reads f_out everywhere | **MISMATCH** | `li_bb_2d_v2.jl:49-54` shows pass-1 spec ALREADY contains `ApplyLiBBPrePhase()` (full Bouzidi pre-phase) — see below |
| Pass-2 `required_args` excludes `f_in` ⇒ lag-1 architecturally forbidden | **MATCH** | `lbm_builder.jl:31-39` `_collect_args` only emits args from the union of `required_args`; `f_in` not in pass-2 brick's required_args (`bricks.jl:564-565`) ⇒ would `UndefVarError` at build time if referenced. **Real, not paper.** |
| Global `KernelAbstractions.synchronize(backend)` flushes pass-1 writes | **MATCH** | Standard backend semantics; same idiom in `ghost_fluid_2d.jl:109`, `logconformation_lbm_2d.jl:337/358/396/432` |
| Phase 2a Couette is steady-state ⇒ lag-1 ≡ lag-0 invisible | **MATCH** | Confirmed — Phase 2a converges to a time-invariant solution where lag-1 = lag-0 to machine precision; this is why the Couette analytical bench passed despite the canonical formula needing lag-0 |

## (c) Ranked hypotheses on missed defects

### Hypothesis 1 — DOUBLE BOUZIDI-FL CORRECTION (HIGH confidence)

**Claim**: pass-1 already applies `ApplyLiBBPrePhase` (full Bouzidi
pre-collision substitution, `bricks.jl:355-402`). Then pass-2 applies a
SECOND Bouzidi-FL substitution post-collision on the same cut links.

**Mechanism**:
- pass-1 substitutes pulled pop `fp_q̄` ← `_libb_branch(qw, f_in[i,j,q], fp_q, f_in[i,j,q̄], δ)` (lag-1 Bouzidi)
- `Moments` builds `ρ, ux, uy` from the Bouzidi-substituted `fp*`
- `CollideTRTDirectGuoField` writes `f_out[i, j, q̄]` based on those moments
- `WriteMoments` stores ρ into `ρ_out`
- pass-2 then OVERWRITES `f_out[i, j, q̄]` using `_bouzidi_fl_post_value(qw, …)`, a SECOND Bouzidi-FL — discarding pass-1's work

**Citation**: `li_bb_2d_v2.jl:49-54` (pass-1 spec includes `ApplyLiBBPrePhase()`); `bricks.jl:591` (`f_out[i, j, 4] = _bouzidi_fl_post_value(…)`); `li_bb_2d_v2.jl:12-20` header comment explicitly warns that "running BOTH pre-phase and post-phase LI-BB resulted in L2 ≈ 2.2 %: a *second* double-BC".

**Empirical prediction (falsifier)**:
- The matrix R=30 Wi=0.1 should produce a Cd substantially OFF from rheoTool's ~130.43 (the double-BC adds ~O(1)% to the wall friction; my prior estimate from the V2 header comment: ~2% L2 on velocity → several percent on Cd).
- R ≥ 40 / Wi ≥ 0.1 NaN at moderate steps consistent with the double correction destabilising the front-shoulder stagnation region (same physical mechanism as the original single-pass NaN, but now even worse).
- Falsifier: build a pass-1 variant that REMOVES `ApplyLiBBPrePhase` from the spec — call it `_TRT_LIBB_V2_GUO_FIELD_RAW_SPEC`. If pass-2 over that gives Cd ≈ rheoTool and no NaN on R=60 Wi=0.1, hypothesis confirmed.

### Hypothesis 2 — ρ_w double-corruption + WriteMoments timing (MEDIUM confidence)

**Claim**: After pass-1 (which includes `ApplyLiBBPrePhase` then collision then
`WriteMoments`), `ρ_out[i, j]` at cut-link cells is built from
Bouzidi-substituted pulled pops. The "raw" canonical `ρ_w` (the density at
`x_f` from un-modified pulled pops) does not exist anywhere in pass-1's
output. Pass-2 reads `rho_w = ρ_out[i, j]` (`bricks.jl:568`) which is a
ρ derived from substituted pops, not the canonical `ρ_w`.

**Mechanism**: minor — wall correction is O(Ma) on `u_w` and `u_w = 0` on
the stationary cylinder benchmark. Not the primary NaN driver, but a
latent issue for any moving-wall geometry (Couette analytical, oscillating
sphere).

**Empirical prediction (falsifier)**:
- For the cylinder benchmark (u_w = 0), this contributes ZERO. So even if H1 were fixed, the cylinder Cd should not move significantly from this.
- For a Couette-style transient bench at finite u_w, expect ~1% error on the wall stress in addition to H1.
- Falsifier: run Phase 2a Couette with the two-pass and check residual at u_w ≠ 0.

### Hypothesis 3 — Pass-2 kernel runs over ALL cells (including solids) without `:solid` brick (LOW confidence)

**Claim**: pass-2 SPEC = `LBMSpec(ApplyBouzidiFLPostCollideTwoPass())` has no
`:solid` brick. `lbm_builder.jl:76-84` says: "If the spec has NO `:solid`
brick, the generated kernel skips the `if/else` entirely and emits all
`:fluid` bricks flat". So pass-2 runs on ALL cells, including the cylinder
interior solid cells AND the box border solids.

**Mechanism**: inside solid cells, `q_wall[i, j, q] = 0` (precomputed as
NaN/zero for non-fluid cells in `precompute_q_wall_cylinder`), so every
`if qw > zero(T)` guard fails → no overwrites. **HOWEVER** the snapshot
reads `f2_here = f_out[i, j, 2]` etc. happen unconditionally at lines
571-578, and `rho_w = ρ_out[i, j]`. On solid cells, those reads pull
`SolidInert`'s rest-equilibrium pops (correct) and ρ=1 (correct). No
write happens because all `qw > 0` guards fail. So this is correct in
principle — pass-2 is a no-op on solids.

**Caveat**: this assumes the precomputed `q_wall` is uniformly 0 on solid
cells; needs verification on the actual codepath. Defect would manifest as
spurious overwrites in the cylinder interior, which would be visible as
non-rest-equilibrium pops in solid cells at the end of the run.

**Empirical prediction (falsifier)**:
- Dump `f_out` at solid cells after a step; check all 9 pops match
  `feq(1, 0, 0)`. If not, this hypothesis fires.
- Low confidence because the smoke test (M34 verdict) shows no NaN and
  drift comparable to halfwayBB — if H3 were firing strongly we'd see
  drift much larger than the halfwayBB baseline.

## (d) Recommendation: AMEND

**Recommendation**: AMEND.

The M30 P2b "Proposed minimal fix" is half-right. The two-pass split is
indeed the canonical Bouzidi-FL GPU pattern, and the `required_args`
gating of `f_in` from pass-2 is architecturally sound. **But the spec
prescribes the wrong pass-1**: re-using `_TRT_LIBB_V2_GUO_FIELD_SPEC`
(which contains `ApplyLiBBPrePhase`) introduces a double-Bouzidi
pathology that is the same class as the V2 motivating bug.

**Concrete amendment**:

1. Add a new spec `_TRT_LIBB_V2_GUO_FIELD_RAW_SPEC = LBMSpec(
   PullHalfwayBB(), SolidInert(),
   Moments(), CollideTRTDirectGuoField(),
   WriteMoments())` — i.e. drop `ApplyLiBBPrePhase()`.

2. In `_fused_trt_libb_v2_guo_field_step!(::Val{:bouzidi_fl_twopass}, …)`,
   replace `pass1! = build_lbm_kernel(backend, _TRT_LIBB_V2_GUO_FIELD_SPEC)`
   with `_TRT_LIBB_V2_GUO_FIELD_RAW_SPEC`.

3. Re-run the M34 smoke test + Aqua matrix.

**Why M30 P2b missed this**: the verdict §"Proposed minimal fix" describes
pass-1 as "collision writes `f_out`" — true in a narrow sense, but elides
the fact that this spec ALSO does a Bouzidi pre-phase substitution before
collision. Both Claude and Codex in the original audit referenced
`_TRT_LIBB_V2_GUO_FIELD_SPEC` by name without enumerating its bricks. The
1st-pass adversarial audit was tight on the lag question but assumed the
"existing collide spec" was a vanilla pull+collide+write. The brick-level
expansion was the gap.

**Why the M34 smoke test passed without revealing this**: the smoke is a
60×40 stagnant box (u₀ = 0.02, no inlet/outlet, no transient pressure
build-up). The double-Bouzidi adds an O(Ma²) systematic bias to the
wall populations, which over 100 steps drifts mass by O(1e-5) — within
the 10× halfwayBB envelope of the smoke. The bug surfaces only on the
cylinder transient with realistic flow strength (R≥30 + finite Wi → NaN).

## Files

- `bench/viscoelastic_audit/M34_SPEC_AUDIT_VERDICT.md` — this verdict
- `bench/scratch/m34_spec_audit/derivation_q_le_half.md` — derivation +
  brick-by-brick comparison

## Memory candidates

1. **`feedback_double_bouzidi_two_pass_trap`** — When porting a single-pass
   Bouzidi-FL kernel to a two-pass GPU scheme, the spec for pass-1 MUST NOT
   itself contain a Bouzidi correction (pre-phase or otherwise). Re-using
   `_TRT_LIBB_V2_GUO_FIELD_SPEC` as pass-1 in M34 introduced a double-BC
   identical to the V2-motivating bug. Always cite the bricks of pass-1
   explicitly in the spec proposal, not just the spec name.

2. **`feedback_m30_p2b_audit_gap_brick_enumeration`** — The 1st-pass M30
   P2b adversarial audit (Claude + Codex CONCORDANT-HIGH) was tight on
   lag-vs-canonical analysis but missed the brick composition of the
   re-used pass-1 spec. Adversarial audits must enumerate the brick list
   of every referenced spec, not just the spec name. Same defect class
   surfaced two missions later in [[feedback_adversarial_default_uncertain]].

3. **`feedback_dsl_required_args_real_enforcement`** — The DSL `required_args`
   is REAL enforcement, not documentation. `lbm_builder.jl:_collect_args`
   builds the kernel signature from the union of `required_args` across
   the spec's bricks; any symbol referenced in emit_code that is NOT in
   the union → `UndefVarError` at build time. Pass-2 brick excluding
   `:f_in` from `required_args` does architecturally forbid lag-1 reads.
   Useful for future "is the brick's API contract enforced?" audits.
