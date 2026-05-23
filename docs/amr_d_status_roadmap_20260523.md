# AMR-D status + roadmap (post-M-H-ETA-7+8)

**Branch**: slbm-paper
**Date**: 2026-05-23
**Anchor commit**: b9f13194 (post Phase 6)
**Predecessor**: bfe77743 (M-H-ETA-7+8 ship, 3-way corner correction)

This doc consolidates the post-M-H-ETA-7+8 validation campaign (Phases 1-7) and lists the open missions for the next sessions.

## Executive summary

**2D AMR-D is production-ready** for the SLBM paper. The M-H-ETA-7+8 fix (committed bfe77743) closes the 3-way vertex corner artifact at nested-1 (1e-7 residual), and Phase 4 confirms the correction holds at nested-2/3/4 with residuals improving 4-6 orders deeper (1e-13).

**3D AMR-D has a c/f mass-balance defect** unrelated to the 2D 3-way bug. Single-patch 3D Poiseuille shows 14% corner Δρ even at Fx=0. Open mission for dedicated 3D investigation; 3D is NOT ready for the paper.

**No regression locked by** the new unit test `test/test_amr_d_corner_nonregression_2d.jl` (committed 7703a734). 5 fixtures, 10/10 PASS in 14.3s. Any future refactor that breaks the corner correction will be caught by `Pkg.test()`.

## Phase deliverables (timeline 2026-05-23)

| Phase | Goal | Status | Commit | Doc |
|---|---|---|---|---|
| 1 | Cleanup tmp/ (41 missions archived, .gitignore patterns) | ✓ | cb0ed338 | — |
| 2 | Unit test non-regression (locks M-H-ETA-7+8) | ✓ | 7703a734 | — |
| 3 | Capability matrix 2D × CPU Float64 (18 fixtures) | ✓ | efa85af7 | [amr_d_capability_matrix_2d.md](amr_d_capability_matrix_2d.md) |
| 4 | Nested level 2/3/4 validation + extended unit test | ✓ | (phase4 commit) | [amr_d_nested_validation_2d.md](amr_d_nested_validation_2d.md) |
| 5 | 3D AMR-D validation | ✓ FAIL documented | 3bd98d23 | [amr_d_3d_validation.md](amr_d_3d_validation.md) |
| 6 | Limits + perf scaling + cyl/bfs NaN resolution | ✓ | b9f13194 | [amr_d_limits_perf_2d.md](amr_d_limits_perf_2d.md) |
| 7 | Synthesis + roadmap (this doc) | ✓ | (this commit) | (this file) |

## Validated capability (current state)

### 2D AMR-D (CPU Float64)
| Layout | Levels tested | Corner peak \|ρ-1\| | Status |
|---|---|---|---|
| couette xband/yband single-band | nested-1 | 1e-6 to 2e-14 | PASS |
| couette H (3-patch) | nested-1 | 3.1e-5 | PASS |
| couette yband_h_full | nested-2,3,4 | 7.8e-13 to 1.5e-11 | PASS |
| poiseuille xband/yband/H | nested-1 | 5e-7 to 4e-6 | PASS |
| cylinder_scale1 | nested-1 | 6.6e-6 | PASS (solid-NaN by-design) |
| cylinder_nested4_probe | nested-4 | 2.75e-4 | DEGRADED |
| bfs_scale1 | nested-1 | NaN at corners | YELLOW (under investigation) |

### 3D AMR-D (CPU Float64)
| Layout | Level | Corner peak | Status |
|---|---|---|---|
| Poiseuille single-patch | nested-1 | 0.144 (14%) | **FAIL** (c/f mass-balance defect, independent of Fx) |

### Backend coverage
| Backend | 2D | 3D |
|---|---|---|
| CPU Float64 | ✓ validated | ✓ tested (3D FAIL documented) |
| Metal Float32 | NOT-TESTED (wrapper kwarg) | NOT-TESTED |
| CUDA Float64 | NOT-TESTED (no Aqua) | NOT-TESTED |

## Open missions prioritized

### High priority
1. **3D AMR-D c/f mass-balance defect** — single-patch Poiseuille shows 14% corner Δρ stable at Fx=0. Distinct mechanism from 2D 3-way. Phase 5 doc has the discriminator. Suggested approach: instrument `stream_composite_routes_periodic_x_wall_yz_F_3d!` mass accounting at the c/f interface.
2. **bfs_scale1 corner NaN** — non-solid NaN at corners requires investigation. Likely inflow/outflow + solid step interaction. Phase 6 deferred.

### Medium priority
3. **GPU validation (Metal + CUDA)** — current unit test is CPU-only. Add a Metal Float32 variant of `test_amr_d_corner_nonregression_2d.jl` (threshold loosened to 1e-3) once host has working Metal Kraken. CUDA Aqua perf scaling.
4. **Perf scaling cell-equivalent** — cartesian uniform vs AMR L1/L2/L3 at constant N_leaf_total, on CUDA H100 (Aqua). MLUPS overhead curve for the paper.
5. **Long-step stability** — 10k+ step accumulation on the worst-case AMR layouts to confirm no slow drift.

### Lower priority
6. **Capability matrix verdict logic** — refine NaN check to mask solid cells, update Phase 3 doc.
7. **3D fixture catalog** — currently 0 production `.krk` fixtures for 3D AMR-D. Building them is prerequisite for systematic 3D validation post-fix.
8. **Subcycled vs route-native dispatch documentation** — Phase 2 found `streaming_runners_channels_2d.jl` hardcodes `:limited_linear` (vs subcycled honoring the kwarg). Document or unify in the runner API.

## Decisions strategic

### For the SLBM paper
- **2D AMR-D**: ship as production. Cite nested-1 to nested-4 validation. Use `couette_yband_h_full` and `couette_H` as flagship layouts.
- **3D AMR-D**: NOT in the paper. Cite 2D only, with a roadmap mention that 3D is "in progress" pending the c/f mass-balance defect resolution.

### For the codebase
- Don't ship new AMR-D features until 3D defect is investigated.
- Keep the 2D non-regression test (`test_amr_d_corner_nonregression_2d.jl`) as the canary — any deeper refactor of `streaming_packets_2d.jl` / `subcycling_explosion_2d.jl` / `three_way_corner_correction_2d.jl` must keep this green.

### Process learnings (memory)
- **Julia cache stale baseline**: a dashboard generated at time T may reflect a stale compiled state if patches were applied at T-N but no recompile triggered. Always re-run the worst-offender fresh before any fix brief. Captured in `feedback_julia_cache_stale_baseline_20260523.md` (global memory).
- **Department subagent bail-out pattern (3/3 occurrences this session)**: subagents stop after capturing the fact Codex is running, expecting a notification that never reaches them. For mission-critical phases, prefer Boss-direct invocation of Codex via `bash run-engineer.sh` to keep control of timing and verification.

## Reference documents

- [Capability matrix 2D](amr_d_capability_matrix_2d.md)
- [Nested level validation 2D](amr_d_nested_validation_2d.md)
- [3D validation (FAIL documented)](amr_d_3d_validation.md)
- [Limits + perf 2D](amr_d_limits_perf_2d.md)

Tests:
- `test/test_corner_periodic_wall_canary_2d.jl` (quiescent canary, pre-existing)
- `test/test_amr_d_corner_nonregression_2d.jl` (shear-active, 5 fixtures, M-H-ETA-7+8 gate)
- `test/test_amr_d_ladder.jl` (ladder convergence, pre-existing)
