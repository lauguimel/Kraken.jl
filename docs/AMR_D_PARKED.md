# AMR voie D — PARKED 2026-05-25

## Status: PARKED on `slbm-paper` branch

The conservative-tree ledger architecture (`src/refinement/`, 43 files, ~19,500 LOC) is **parked** as of 2026-05-25. No further active development on this architecture.

## What is preserved

All code, tests, and benchmarks remain on:
- Branch `slbm-paper` (HEAD: `5cf27368` at parking time)
- Tagged commits visible via `git log --all --oneline -- src/refinement/`

The 9 locked tests (`test/test_amr_d_*.jl`, `test/test_refinement*.jl`, `test/test_c2f_prolongation.jl`) continue to pass in CI. The 2D nested-1..4 corner non-regression is the production-ready snapshot.

## Why parked

Strategic step-back 2026-05-25 with adversarial Codex+Claude verdicts (`tmp/strategic_review_brief.md`, `tmp/strategic_review_claude_output.md`, `.engineer_logs/AMR-STRAT-REVIEW-CODEX_20260525_150721.log`):

1. **Mathematical**: the Rohde §5.4.2 reflux was empirically falsified (memory `feedback_rohde_reflux_simple_variant_defect`); the ledger sidesteps it via `:leaf_equivalent` time-scaling rather than a derived replacement. Momentum-per-level and wall c/f flux balance remain underived.

2. **GPU**: the production GPU subcycled runner hard-rejects `:limited_linear` (the c2f scheme empirically needed to suppress the H 3-way bug). Bug-suppression is CPU-only on the most production path.

3. **Bug-N+1 risk**: 162 commits since 2026-04-01 (22 refactor / 21 fix / 1 abandon), 8+ debug sessions converging on a recurrent pattern. The user diagnosis: "on tourne en rond parce qu'entre les sessions on n'arrive pas à avoir une discipline assez forte sur les canaries et l'analytique."

4. **Extensibility**: the c2f operator is not physics-agnostic. Each new physics (viscoelasticity, multiphase, thermal) would require its own prolongation operator and its own bug cycle.

5. **2 open production defects** (Couette H wall-row +26%, Poiseuille xband peak −11.5%) and **1 open 3D defect** (corner Δρ 14%) — closeable but representative of the structural pattern.

## What replaces it

Branch `feat/amr-port-sr` ports the **Schornbaum-Rüde 2016 algorithm** (waLBerla's `src/lbm_generated/refinement/` ~340 LOC + `NonuniformGeneratedPdfPackInfo.impl.h` ~500 LOC) to Julia + KA.jl (CUDA + Metal + CPU).

Decision basis (2026-05-25, two rounds of adversarial Codex+Claude audit on 4 candidates: waLBerla / Palabos / OpenLB / Neon):
- **Schornbaum-Rüde** chosen for **mathematical rigor** (Σ f_q algebraic mass conservation, independent of collision) + **versatility** (collision-agnostic sweep contract for multiphysics) + **production GPU scaling** (JUWELS Booster 10000+ GPUs published)
- Lagrava (OpenLB) was a candidate MVP rung but **skipped** because the discipline infrastructure (`kraken-amr-canary` + `kraken-port-fidelity` skills + `test_amr_port_validation.jl` ladder + UserPromptSubmit hook) already enforces canary + bit-exactness without needing an intermediate target
- waLBerla blockforest framework is **NOT** ported; Kraken's existing `conservative_tree_*` block infrastructure (parked but functional) is reused as the block substrate
- Codegen (lbmpy/pystencils) is **bypassed**; KA.jl kernels are hand-written from the published algorithm + the waLBerla `.impl.h` files as reference

Discipline: `kraken-amr-canary` + `kraken-port-fidelity` Codex skills + `test/test_amr_port_validation.jl` ladder (C0..C6) + UserPromptSubmit hook in `.claude/settings.local.json`. Port mission to be executed via Codex Cloud in autonomous mode.

## How to revive (if needed)

The parked architecture is available for:
- SLBM paper publication (the SLBM paper does NOT depend on AMR-D)
- Reference comparison during the port (validate the new implementation against the same fixtures)
- Cherry-picking specific algorithmic pieces (e.g., the GPU pack SoA layout in `conservative_tree_gpu_pack_2d.jl`)

To revive, `git checkout slbm-paper` and resume from `tmp/NEXT_SESSION_PROMPT.md` (measure-first plan U7/U8/U9 micro-canaries).

## Memory pointers

- `project_amr_strategy_pivot` (2026-05-17 original decision)
- `project_amr_d_parked_20260525` (this parking event)
- `feedback_rohde_reflux_simple_variant_defect`
- `feedback_math_sufficiency_check`
- `feedback_amr_micro_canary_pattern`
