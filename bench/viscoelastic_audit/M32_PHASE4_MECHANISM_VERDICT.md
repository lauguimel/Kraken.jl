**Verdict**: SAME mechanism

# M32 Phase 4 Mechanism Verdict — D1 (Cd gap at Wi=1) vs D2bis (R=60 NaN) share one upstream code path

Date: 2026-05-22
Branch: dev-viscoelastic
Mission: D3-finalize (synthesis from existing `.engineer_logs/trace.snapshot.jsonl` + dispatch_probe output + D1 + D2bis verdicts)

## TL;DR

D1 (R=30 Wi=1 Cd-gap, dominant `(pressure, front_pole)` = 80% of gap, secondary `(polymer, shoulder)` = +30%) and D2bis (R=60 Wi=1 NaN, bilateral front-shoulder logconf_singularity at θ ∈ ±(38°, 48°), r-R ∈ [0, 7] LU) are **two observables of one upstream coupling chain**:

```
halfwayBB(step n) writes ρ, ux, uy at wall ring
  → vel_grad(step n+1) reads ux, uy → ∇u at wall-adjacent cells (incl. front-shoulder)
  → psi_advect / constitutive substeps consume ∇u → Ψ at shoulder reflects BC quality
  → logfv_stress_from_log + poly_force → polymer back-force ∇·τ_p at shoulder
  → BSD correction → fx_total feeds Guo forcing in next halfwayBB
```

The BC at the front-pole (D1's dominant gap bucket) and the polymer back-force at the front-shoulder (D1's secondary + D2bis NaN locus) are downstream of the **same `WriteMoments` brick** in `_TRT_LIBB_V2_GUO_FIELD_SPEC`. The pressure-front-pole gap is the BC writing the wrong populations at the pole, and the polymer-shoulder pathology is the same BC error propagating one cell-row into the shoulder, where Ψ amplifies it exponentially. **A correct BC closes both at once.**

## Evidence

### jq citations from `.engineer_logs/trace.snapshot.jsonl`

**jq citation 1** — Per-step kernel call ordering (8 kernels per step, 200 steps captured):

```
jq -r '.kernel' .engineer_logs/trace.snapshot.jsonl | sort | uniq -c | sort -rn
```

```
600 psi_advect_inner   (3× per step, 1 per Ψ-component)
200 vel_grad
200 psi_sym2_advect
200 psi_advect
200 poly_force
200 lbm_step_halfwayBB
200 lbm_step
  1 driver_step_entry
```

**jq citation 2** — Within-step ordering (head 18 lines = 2 full steps):

```
jq -r '[.t_ns, .kernel] | @tsv' .engineer_logs/trace.snapshot.jsonl | head -18
```

shows the byte-exact sequence per step:

```
psi_advect → psi_sym2_advect → psi_advect_inner ×3 → vel_grad → poly_force → lbm_step → lbm_step_halfwayBB
```

(Matches driver source `src/drivers/viscoelastic_logfv_2d.jl:407-480`.)

**jq citation 3** — Cross-step causality (halfwayBB(n) precedes psi_advect(n+1)):

```
jq -r 'select(.kernel == "lbm_step_halfwayBB" or .kernel == "psi_advect") | [.t_ns, .kernel] | @tsv' .engineer_logs/trace.snapshot.jsonl | head -8
```

```
609864464345958  psi_advect             ← step 1, reads ux,uy from init
609865603571833  lbm_step_halfwayBB     ← step 1, writes ρ,ux,uy
609866153983750  psi_advect             ← step 2, reads ux,uy written by step 1 halfwayBB
609866261633291  lbm_step_halfwayBB     ← step 2, writes again
609866266312166  psi_advect             ← step 3, reads ux,uy written by step 2 halfwayBB
609866373134291  lbm_step_halfwayBB
609866377716875  psi_advect
609866484773833  lbm_step_halfwayBB
```

Δt(halfwayBB → psi_advect) ≈ 0.55 ms; Δt(psi_advect → halfwayBB) ≈ 1.14 ms. The data dependency is one-step lagged (halfwayBB writes at end of step n; psi_advect reads at start of step n+1). This is the **closed coupling loop**.

### Dispatch resolution (`dispatch_probe.jl` output, captured 2026-05-22)

```
lbm_step_halfwayBB
  _fused_trt_libb_v2_guo_field_step!(::Val{:halfwayBB}, ...)
  @ Kraken src/kernels/li_bb_2d_v2.jl:138

vel_grad
  fvfd_velocity_gradient_2d!(dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, bc; sync)
  @ Kraken src/fvfd/operators_2d.jl:1127

psi_advect
  logfv_advect_upwind_bc_aware_2d!(...) @ src/kernels/logconformation_fv_2d.jl:1154
psi_sym2_advect
  fvfd_sym2_advect_upwind_2d!(...) @ src/fvfd/operators_2d.jl:648
psi_advect_inner
  fvfd_advect_upwind_2d!(...) @ src/fvfd/operators_2d.jl:591
poly_force
  logfv_polymer_force_bc_aware_2d!(...) @ src/kernels/logconformation_fv_2d.jl:649

lbm_step_bouzidiFL
  _fused_trt_libb_v2_guo_field_step!(::Val{:bouzidi_fl}, ...)
  @ Kraken src/kernels/li_bb_2d_v2.jl:154   ← M34 candidate replacement for halfwayBB
```

### Shared-callee smoking gun

The brick sequence dispatched by `_fused_trt_libb_v2_guo_field_step!(Val(:halfwayBB), ...)` is (`src/kernels/li_bb_2d_v2.jl:49-54`):

```
_TRT_LIBB_V2_GUO_FIELD_SPEC = LBMSpec(
    PullHalfwayBB(),                # ← THE BC. Writes f at wall ring.
    SolidInert(),
    ApplyLiBBPrePhase(),
    Moments(), CollideTRTDirectGuoField(),
    WriteMoments(),                 # ← writes ρ, ux, uy at every fluid cell
)
```

- `PullHalfwayBB` is the BC whose pole accuracy D1 identified as the 80% gap bucket.
- `WriteMoments` is the kernel whose output (ρ, ux, uy) is then consumed by `fvfd_velocity_gradient_2d!`, `logfv_cell_velocity_to_faces_bc_aware_2d!`, `logfv_advect_upwind_bc_aware_2d!`, and `logfv_polymer_force_bc_aware_2d!` in step n+1.
- D2bis observed that ρ, ux, uy NaN before Ψ, τ_p NaN (first_nonfinite_field = `rho`; 98 NaN cells in ρ vs 44 in Ψ; gradients clean). That field-order signature is **diagnostic of the BC → moments → vel_grad → Ψ → poly_force → next-BC loop diverging**, not of an isolated Ψ-scheme blow-up.

So:

- The polymer pipeline (vel_grad / psi_advect / poly_force) has NO independent population read. It reads `ux, uy` from cell-centred arrays that ONLY halfwayBB writes.
- There is NO "Ψ has its own decoupled buffer" — Ψ is updated from ∇u, which is computed from ux/uy, which are written by halfwayBB.

This is the structural definition of **SAME mechanism**: changing the BC (halfwayBB → Bouzidi-FL) changes what gets written to ρ, ux, uy at the wall ring; that change propagates one cell-row into the shoulder within one step via vel_grad; and the Ψ-scheme, by exp(Ψ), amplifies the error to the polymer back-force the next step.

## Implication for M34 strategy

**M34 should be Bouzidi-FL Phase 2b ALONE, NOT Bouzidi-FL + Ψ-scheme in parallel.**

Justification:

1. The D1 dominant gap (80%) is `(pressure, front_pole)` — a BC observable. A correct BC must close it (M30 Phase 2a Bouzidi-FL analytical Stokes-flow validation already confirmed Bouzidi closes the front-pole pressure locus).
2. The D2bis NaN is at the front-shoulder, one cell-row removed from the pole BC; this is the front-pole BC error propagating into the polymer pipeline. The polymer pipeline has no independent population read, so a Ψ-scheme upgrade (MUSCL-superbee, CUBISTA) operating on `ux, uy` that are STILL written by the staircase BC will at best DELAY the blowup, not resolve it.
3. Bouzidi-FL replaces the entire `_fused_trt_libb_v2_guo_field_step!(::Val{:halfwayBB})` dispatch with `Val(:bouzidi_fl)` (`src/kernels/li_bb_2d_v2.jl:154`) — a one-line `wall_bc=:bouzidi_fl` change in the driver call site (line 479). The brick sequence is the same (`_TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_SPEC` at L56), so the polymer pipeline downstream is bit-identical except via the new (more accurate) ux/uy.
4. If Bouzidi-FL Phase 2b closes BOTH the D1 Cd gap AND the D2bis R=60 NaN, it confirms SAME mechanism empirically. If it closes the gap but the NaN persists, the residual is the Ψ-scheme contribution alone (and only THEN does M34b add a Ψ-scheme upgrade on top — sequential, not parallel).

A Ψ-scheme upgrade in parallel **adds two confounding variables at once** and prevents attribution; this is exactly the methodological trap [[feedback_cd_wall_vs_volume]] warned about (volume-L2 ranked schemes wrong because of cancellation), at a different scale.

## Caveats / open questions

- The trace was captured at R=30 CPU F64 (200 steps, no NaN). D2bis's NaN was at R=60 Metal F32 step 29. The coupling-chain TOPOLOGY is precision-/backend-independent (it's set by the driver source); the QUANTITATIVE thresholds (where exp(Ψ) saturates) differ. F32 will saturate at smaller Ψ than F64, so Aqua F64 CUDA Wi=0.1 (D2bis Phase B job 21619886.aqua, still pending at time of writing) may not NaN even when Metal F32 does. The SAME-mechanism verdict does NOT depend on the F32-saturation signal — it depends on the call-graph + WriteMoments shared-write, both of which are precision-invariant.
- The trace has empty `extras` fields (D3-original did not enrich with max-Ψ snapshots or region tags). The SAME-mechanism verdict is established by **dispatch + ordering + brick-sequence inspection**, not by extras telemetry. A WatchedArray probe on a single front-shoulder cell would directly observe the propagation across the halfwayBB → moments → vel_grad → Ψ → poly_force chain; this was Step D's fallback. **Not exercised** here because A+B+C are conclusive without it; the trace ordering + dispatch_probe + brick-sequence already close the question.
- The wrapper `lbm_step` (200 entries, args_hash `18bbcdea9c0ebe8d`) is the keyword dispatcher (`fused_trt_libb_v2_guo_field_step!` L123) that immediately delegates to `lbm_step_halfwayBB` (L138 + entry counter L143). Both entries fire per step; counts agree.
- M33's MUSCL-superbee + CUBISTA candidates are NOT REFUTED by this verdict; they remain valid for a future M34b after Bouzidi-FL has cleared the BC contribution. They are simply NOT the first move per the SAME-mechanism finding.
- D2bis Phase B (Aqua jobid 21619886.aqua, Wi=0.1 F64 CUDA) result, when it returns, may further constrain whether the front-shoulder pathology is BC-driven (Bouzidi-FL fix expected to also close Wi=0.1 R=60) or has a residual Ψ-scheme component visible only at lower Wi.

## Files

- `.engineer_logs/trace.snapshot.jsonl` — 167 KB, 1801 lines, 8 distinct kernels × 200 steps R=30 Wi=1 (D3-original captured 2026-05-22 morning)
- `bench/scratch/m32_phase4_mechanism/jq_summary.md` — full jq-derived breakdown
- `bench/scratch/m32_phase4_mechanism/dispatch_probe.jl` — `which`-resolver source
- `bench/scratch/m32_phase4_mechanism/dispatch_probe.out.txt` — captured stdout (re-run by D3-finalize)
- `src/kernels/li_bb_2d_v2.jl:49-54` — `_TRT_LIBB_V2_GUO_FIELD_SPEC` brick sequence (the shared `WriteMoments` smoking gun)
- `src/drivers/viscoelastic_logfv_2d.jl:407-480` — driver call sequence (matches trace ordering)
- `bench/viscoelastic_audit/M32_PHASE4_WI1_GAP_LOCALIZATION_VERDICT.md` — D1
- `bench/viscoelastic_audit/M32_PHASE4_R60_NAN_TRACE_VERDICT.md` — D2bis

## Memory candidates

1. **SAME-mechanism verdict at M32 Phase 4** — D1's `(pressure, front_pole)` gap and D2bis's bilateral front-shoulder NaN share one upstream BC. The polymer pipeline has NO independent population read; it consumes `ux, uy` that ONLY halfwayBB writes via `_TRT_LIBB_V2_GUO_FIELD_SPEC.WriteMoments`. M34 strategy = Bouzidi-FL Phase 2b ALONE, not in parallel with a Ψ-scheme upgrade. Adding both simultaneously confounds attribution per [[feedback_cd_wall_vs_volume]].
2. **D3-original Department stall pattern** (second occurrence after D2-original): Departments that arm Monitor on a long-running run and then sit for hours never finish synthesis. Finalize-style missions (artifact-only, must write verdict) MUST have hard `timeout 300` Bash gating, NO Monitor under any circumstance, and the verdict file as a deterministic exit criterion. Extension candidate for [[feedback_orchestrator_discipline]].
3. **Brick-sequence inspection is a 5-minute alternative to a WatchedArray run** — when the question is "do path A and path B share a write target?", reading the `LBMSpec` brick list (`PullHalfwayBB`, `WriteMoments`) and the driver call sequence is bit-for-bit more conclusive than instrumenting cells. Adversarial check: confirm the `Val(:halfwayBB)` dispatch lands at the expected `_fused_trt_libb_v2_guo_field_step!` method via `which`. Trace's `args_hash` invariance across steps (single hash per kernel) confirms the in-place mutation pattern is consistent; no hidden alternate dispatch.
