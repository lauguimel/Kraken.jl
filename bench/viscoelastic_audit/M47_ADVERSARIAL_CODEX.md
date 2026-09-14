# M47 — Adversarial verdict (CODEX)
Date: 2026-05-26  ·  Engine: Codex

## 0. WRITE-FIRST sketch (drop within 5 min, before deep dive)
- Top hypothesis (one sentence): The Bouzidi-FL two-pass boundary path leaves non-equilibrium wall-neighbor moment defects that the log-FV polymer chain reads as velocity-gradient/source data, producing a C-side residual that is dormant at beta=1 but feeds back at beta<1.
- 1-paragraph mechanism: The key asymmetry is not simply "curved wall bad"; it is that beta=1 Bouzidi-FL has healthy Newtonian Cd while trace_C explodes with R, so the bad signal must enter after hydrodynamic drag accounting but before or inside polymer evolution. The plausible path is LBM f -> forced macroscopic u -> log-FV advection/Hermite source -> tau_p/div(tau_p): a small near-wall velocity-gradient residual around cut links, amplified by q-dependent two-pass reconstruction and wall-cell count, can remain invisible to Cd_s at beta=1 yet generate huge C; at beta=0.59 the same residual can inject polymer force and distort solvent drag over long R=60 runs.
- Smallest patch test concept (CPU/Metal <60s): Build a tiny R=6 or R=8 cylinder box, initialize C=I and a known frozen analytic no-polymer velocity field, run only the log-FV polymer update for halfwayBB-vs-Bouzidi-like wall masks on the same CPU Float32 path, and log max(trace(C)-2) minus the t=0/bulk baseline in a one-cell wall band. H1 predicts a Bouzidi-only wall-band residual orders above halfway; a genuine slow wake transient predicts no growth in this frozen-U zero-polymer discriminator.

## 1. Top 3 hypotheses ranked
| # | Hypothesis | Mechanism (≤3 sentences) | Discriminating evidence required |
|---|---|---|---|
| 1 | Bouzidi-FL cut-link moment residual contaminates the log-FV polymer chain. | Two-pass Bouzidi overwrites cut-link populations and recomputes cut-link rho; downstream log-FV reads the resulting u field, then FVFD gradients/source evolve C. This matches beta=1: Cd_s is healthy, but C explodes internally. | Same-backend frozen-U/cut-link residual: Bouzidi wall-band `trace(C)-2` grows while halfway stays near baseline. |
| 2 | R=60 Wi=1 is a real slow wake transient. | Flow-through count is low at 100k-400k, and Cd_p grows with time. This explains halfwayBB drift but not beta=1 Bouzidi C blow-up. | Frozen-U patch should stay zero; live Cd(t) should decelerate with additional flow-throughs. |
| 3 | Remaining Guo/readout convention bug on the cylinder path. | M44 fixed G3/G1, but similar getters remain elsewhere. Static driver slice shows the active 2D log-FV cylinder loop still uses fixed G3, making this a runner-up rather than top. | Dispatch/trace must show an unfixed getter enters R=60 cylinder; otherwise reject. |

## 2. Discriminator patch test design
- Geometry, grid size, BC, run length: CPU Float32, R=6 cylinder in open-x/wall-y box, about 180x48 cells, 200 log-FV-only steps; compare two like-for-like wall masks: halfway q=0.5 vs Bouzidi q_wall from `precompute_q_wall_cylinder`.
- What field/quantity is logged: one-cell wall-band max and p99 of `trace(C)-2`, plus max norm of velocity-gradient residual.
- Baseline subtracted: subtract both t=0 wall-band value and bulk-fluid max at each log point.
- Predicted output: H1 gives Bouzidi residual >1e-3 and ratio >10 vs halfway; H2 gives both ≤1e-4 because U is frozen/no wake; H3 gives no Bouzidi-specific split unless the readout path is deliberately included.
- Pass/fail criterion: fail if Bouzidi residual >1e-4 or Bouzidi/halfway ratio >10 after baseline subtraction.
- Estimated LOC + dev cost: skeleton 17 LOC now; full test about 45 LOC, under 1 hour if helper APIs are reused.
- Filepath: `bench/viscoelastic_validation/patch_tests/PT_M47_bouzidi_logfv_wall_residual.jl`.

## 3. Code-path provenance evidence
- Empirical trace attempt: blocked locally because the Julia launcher could not create its lockfile under sandbox permissions. I therefore do not claim an empirical trace.
- Static dispatch table inspection, not grep alone:
| Stage | Evidence |
|---|---|
| Cylinder entry | `run_viscoelastic_logfv_cylinder_coupled_2d` builds q_wall cylinder geometry and calls `_run_viscoelastic_logfv_step_channel_coupled_2d` (`src/drivers/viscoelastic_logfv_2d.jl:868-890`). |
| Wall-BC selection | Driver accepts `:bouzidi_fl_twopass` (`:229`) and passes `wall_bc` into `fused_trt_libb_v2_guo_field_step!` (`:477-480`). |
| Bouzidi twopass | `_fused_trt_libb_v2_guo_field_step!(Val(:bouzidi_fl_twopass), ...)` runs pass1/pass2/pass3, with pass3 recomputing cut-link rho (`src/kernels/li_bb_2d_v2.jl:211-249`; `src/kernels/dsl/bricks.jl:563-724`). |
| Polymer reads u | After LBM, `logfv_compute_macroscopic_forced_field_2d!` reads `f_out` into rho/u (`src/drivers/viscoelastic_logfv_2d.jl:528`; fixed getter at `src/kernels/logconformation_fv_2d.jl:1025-1050`). |
| Polymer source/force | The loop maps u to faces, advects psi, computes FVFD velocity gradients, evolves log-C, computes tau, then divergence (`src/drivers/viscoelastic_logfv_2d.jl:395-468`). The default force path is non-embedded (`embedded_force=false`) and uses solid-mask FVFD divergence (`src/fvfd/operators_2d.jl:724-880`); gradients use solid-mask FVFD (`:1062-1154`). |

Files read: synthesis; M46 verdict; M44 fix verdict; M44 Codex audit; `logconformation_fv_2d.jl`; `operators_2d.jl`; `viscoelastic_logfv_2d.jl`; `li_bb_2d_v2.jl`; `dsl/bricks.jl`; branch contract; trace/FVFD/governor skills.

## 4. Skills you actually used
- `kraken-branch-governor`: useful for allowed-zone, small-test, no-HPC discipline.
- `kraken-trace`: useful; its required empirical trace could not run because Julia was sandbox-blocked, so I fell back to static dispatch inspection.
- `kraken-fvfd-operator-library`: useful for focusing the discriminator on lowered wall geometry, FVFD gradients/divergence, and baseline-subtracted canaries.

## 5. Confidence (LOW/MED/HIGH) on your top hypothesis + 1-sentence why
MED: H1 uniquely explains the beta=1 Bouzidi trace_C explosion and has a concrete static path into log-FV C, but the R=60 halfwayBB temporal drift may still be a separate slow-wake effect until the frozen wall residual test is implemented.
