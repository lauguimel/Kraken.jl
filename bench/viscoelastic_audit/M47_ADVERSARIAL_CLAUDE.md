# M47 — Adversarial verdict (CLAUDE)
Date: 2026-05-26  ·  Engine: Claude Code (Opus 4.7 1M)

## 0. WRITE-FIRST sketch (dropped before deep dive)

**Top hypothesis (one sentence)**: At β=1.0 placeholder Bouzidi-FL, the polymer log-Ψ chain consumes ∇u from a non-q_wall-aware FVFD stencil that reads `ux[first_fluid]` values which are q_w-dependent under Bouzidi-FL (q_w-invariant under halfwayBB), producing a spurious tangential shear along the wall arc that the upper-convected term D_t C = (∇u)·C + C·(∇u)ᵀ amplifies exponentially — even at β=1.0 (no LBM back-coupling).

**1-paragraph mechanism**: For halfwayBB at q_w=0.5, `f_out[i,j,qbar]=fp_q_post-collision` (q_w-invariant), so `ux[first_fluid]` varies smoothly along the wall. For `bouzidi_fl_twopass`, pass-2 overwrites `f_out[i,j,qbar]` with `_bouzidi_fl_post_value(q_w, …)` — a q_w-dependent algebraic rescaling. The first-fluid moment `ux = (sum f_out)/ρ` then varies as q_w varies along the arc. The FVFD `_fvfd_solid_bc_derivative_x_2d` one-sided quadratic stencil treats `ux[first_fluid]` as a cell-center sample at distance dx regardless of actual wall offset, manufacturing a large spurious tangential ∂u/∂s. The log-Ψ chain runs at every cell unconditionally and integrates this via the upper-convected term: trace_C ∝ exp(γ̇_spurious·Wi·N). At R=60 the wall-cell count doubles, so the arc accumulates more spurious shear → trace_C scales ~7000× over R=30→60. At β=1.0 no Cd impact (no F_poly); at β<1 a slow feedback amplifies the existing R=60 wake-buildup drift.

**Smallest patch test concept (CPU <60s)**: Frozen-u 64×4 straightened arc; prescribe analytic Stokes-around-cyl ux at j=2..4; at j=1 toggle between (A) halfwayBB-rule `ux=U·cos(θ)/2` and (B) Bouzidi-FL-rule `ux=f(q_w[i])` with q_w∈[0.05,0.95]. Frozen ∇u from `fvfd_velocity_gradient_2d!`. 1000 steps of `logfv_step_constitutive_log_2d!` at λ=12k LU (R=60 Wi=1). Discriminator: max_i(trace_C_B − trace_C_A) > 10·max(trace_C_A) at step 1000 confirms H1.

---

## 1. Top 3 hypotheses ranked

| # | Hypothesis | Mechanism (≤3 sentences) | Discriminator |
|---|------------|--------------------------|---------------|
| **1** | **Bouzidi-FL twopass + non-q_w-aware FVFD gradient at first-fluid arc → q_w-modulated ∂u/∂s feeds log-Ψ → trace_C grows ~exp(Wi·γ̇_spurious·N).** | Pass-2 `ApplyBouzidiFLPostCollideTwoPass` (bricks.jl:567-691) overwrites `f_out[i,j,qbar]` with a q_w-dependent algebraic post-value; ux at cut-link first-fluid cells consequently varies along arc as q_w varies. `fvfd_velocity_gradient_2d!` (operators_2d.jl:1062) uses `_fvfd_solid_bc_derivative_x_2d` (line 13-43): NO q_w awareness, only is_solid. The log-Ψ chain (logconformation_fv_2d.jl:417) runs at every cell unconditionally, exponentially amplifying spurious ∇u via the upper-convected term. | PT_bouzidi_fl_polymer_qw_arc (§2). Predict: case B trace_C grows ∝ |q_w−0.5|; case A bounded. |
| **2** | **R=60 wake-buildup is genuine (synthesis H2) but compounded by H1 leaking polymer back-stress into Cd_s.** | Cd_p grows monotonically with t at R=60 (13.97→15.30); Cd_s drops faster (−4.85). H1+H2: arc trace_C buildup creates a slow div(τ_p) perturbation that decreases solvent drag (Reynolds-stress feedback). The −1.10→−2.71 acceleration per 200k window is consistent with positive feedback. | Pure H2 = decelerating drift to asymptote; H1+H2 = compounding. After PT confirms H1, re-run R=60 Wi=1 with `polymer_wall_extrap=:linear` (already a kwarg). |
| **3** | **ux/uy staleness on cut-link cells across the two-pass barrier — pass-3 (`ApplyCutLinkRhoRecompute`, bricks.jl:708) re-sums ρ_out only, NOT ux/uy.** | After pass-1 WriteMoments at cut-link cells, `ux_out/uy_out` reflect pre-Bouzidi-FL pop sums. Pass-2 overwrites `f_out[qbar]`, pass-3 overwrites `ρ_out`. The cached `ux/uy` are inconsistent until line 528 recomputes them from `f_out`. *Within the LBM step* nothing reads `ux/uy` between line 477 and 528, so this is benign — demoted. | Static trace: confirmed nothing reads `ux/uy` in lines 478-527. Real but inert bug. |

H3 is structurally present but inert under current driver topology; treat H1 as load-bearing, H2 as secondary amplifier. Both H1 and H3 trace to the M30 Phase 2b/M34v3 architectural choice: Bouzidi-FL overwrites happen OUTSIDE the moment-computation phase — downstream consumers re-reading from f are safe, cached moments are stale.

## 2. Discriminator patch test

**File**: `bench/viscoelastic_validation/patch_tests/PT_bouzidi_fl_polymer_qw_arc.jl` (NEW, ~45 LOC skeleton committed in this verdict).

- **Geometry**: 64×4 "straightened arc": i = arc position, j = radial. q_w[i] ∈ [0.05,0.95] (cosine ramp).
- **Field**: prescribe analytic u_t(s,r)=U·cos(θ)(1−R/r) at j=2..4; at j=1 either (A) halfwayBB rule (q_w-invariant) or (B) Bouzidi-FL rule (q_w-dependent, derived by inverting `_bouzidi_fl_post_value`).
- **Pipeline**: `fvfd_velocity_gradient_2d!` (once, frozen) → 1000× `logfv_step_constitutive_log_2d!` (λ=12k, model_code=:oldroydb, dt=1) → `logfv_stress_from_log_2d!`.
- **Logged**: trace_C[i,1](step). **Baseline**: trace_C at q_w=0.5 cells (Bouzidi-FL ≡ halfwayBB there). Subtract → residual proxy per `[[feedback_residual_proxy_required]]`.
- **Predictions**: H1 → case B residual grows exponentially, peak at |q_w−0.5|→max; H2 → identical; H3 → identical (driver re-reads f anyway).
- **Pass criterion**: H1 confirmed if `max_i(trace_C_B[i] − trace_C_A[i]) > 10·max_i trace_C_A[i]` at step 1000; refuted if within 10%.
- **Cost**: ~45 LOC, CPU F64, wall time <15s. No Aqua, no Wi sweep, no F64-dependence.

Limitation: this test does not run the actual `fused_trt_libb_v2_guo_field_step!`; it is a constitutive-side discriminator. Upgrade later to PT_bouzidi_fl_polymer_full_step (>30 LOC, single-cut-link 32×32 box) only after H1 is confirmed.

## 3. Code-path provenance evidence

For both failing cases (Wi=1 R=60 halfwayBB drift AND β=1 R=60 Bouzidi-FL trace_C explosion), the dispatch chain is verified by static read of `Val{wall_bc}` dispatch + brick specs (NOT grep alone):

1. Driver `run_viscoelastic_logfv_cylinder_coupled_2d` (driver:868) → `_run_viscoelastic_logfv_step_channel_coupled_2d` (line 173).
2. Per step:
   - Line 477: `fused_trt_libb_v2_guo_field_step!` dispatches on `Val{wall_bc}`. `:halfwayBB` → spec `_TRT_LIBB_V2_GUO_FIELD_SPEC` (li_bb_2d_v2.jl:49); `:bouzidi_fl_twopass` → 3 passes: RAW spec (line 69), TWOPASS_PASS2 (line 80), TWOPASS_PASS3 (line 93).
   - Line 528: `logfv_compute_macroscopic_forced_field_2d!` (logconformation_fv_2d.jl:1025) — **M44-fixed at lines 1049-1050 (no +F/2)** — recomputes ux/uy from `f_out`.
   - Line 418-428: `fvfd_velocity_gradient_2d!` (default `embedded_gradient=false`, per M46 PBS line 67 `KRAKEN_EMBEDDED_GRADIENT="0"`) → `_fvfd_solid_bc_derivative_x_2d` (operators_2d.jl:13) one-sided quadratic stencil, NO q_w awareness.
3. Polymer chain consumer line 432: `logfv_step_constitutive_log_2d_kernel!` (line 417, no `is_solid` early return, upper-convected `D_t Ψ`).

Other Codex M44 G2/G4/G5/G6 getters: grep confirms NONE called from `src/drivers/viscoelastic_logfv_2d.jl` or `src/drivers/viscoelastic.jl`. H3-of-synthesis (other Guo bugs) refuted at provenance.

**Files read (10)**: M47 synthesis, brief, M46 verdict, M44 Codex audit, `src/drivers/viscoelastic_logfv_2d.jl` (lines 173-528, 868-892), `src/kernels/li_bb_2d_v2.jl` (1-252), `src/kernels/dsl/bricks.jl` (80-200, 560-740, 820-836), `src/fvfd/operators_2d.jl` (1-90, 1062-1196), `src/kernels/logconformation_fv_2d.jl` (417-460, 600-668, 1020-1065), `bench/viscoelastic_logfv/run_cyl_m46_newt_sweep_a100.pbs`.

Key lines: bricks.jl:152-157, 567-691, 708-727, 820-836; operators_2d.jl:13-43, 1062-1135; logconformation_fv_2d.jl:417-435, 1025-1065.

## 4. Skills used

- **`kraken-codebase-map`** (loaded). **Useful** for confirming voie-F (viscoelastic_logfv) vs voie-D (AMR-D) separation — route_sampling/c2f noise correctly ignored. The "production-vs-debug" framing pushed me to verify the actual `Val{wall_bc}` dispatch via static read rather than grep, which is exactly the anti-pattern shield needed here.
- **`kraken-trace`** — not loaded (Codex-side instrumentation; for Claude, manual static dispatch substitutes).
- **`rheotool`** — not needed (rheoTool Newt 132.37 ref already in synthesis; no constitutive-model interpretation required).

## 5. Confidence

**MED-HIGH** on H1.

One-sentence why: static dispatch is solid (FVFD gradient not q_w-aware ✓, Bouzidi-FL post-value q_w-dependent ✓, polymer Ψ chain consumes ∇u exponentially ✓), and the M46 trace_C R-scaling 1938→14M (~7000× per 2× resolution) is quantitatively consistent with arc-multiplicative amplification — but the exact functional form of `ux[first_fluid](q_w)` under Bouzidi-FL must be measured by PT_bouzidi_fl_polymer_qw_arc before claiming HIGH.
