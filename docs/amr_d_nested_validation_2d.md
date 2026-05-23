# AMR-D 2D nested-level validation

**Commit**: efa85af7 (Phase 4)
**Branch**: slbm-paper
**Date**: 2026-05-23
**Backend**: CPU Float64
**Fixture base**: couette_yband_h_full layout (16×12 coarse, U=0.05, periodic-x + wall-y)
**Steps**: 200 (nested-1: 1500, kept dashboard config)

## Hypothesis
The M-H-ETA-7+8 3-way corner correction was validated empirically only on nested-1 (ratio=2). Does it generalize to nested-2 (ratio=4), nested-3 (ratio=8), nested-4 (ratio=16) — auto-cascaded 2:1 levels?

## Matrix
| Level | leaf_nx | leaf_ny | n_cells | corner_peak |ρ-1| | mass_drift | speed_max | rho_lvl_jump | rho_lvl_dev | NaN? | Wall(s) | MLUPS | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| nested1 | 32 | 24 | 768 | 9.5017e-07 | 6.3653e-14 | 4.8931e-02 | 0.0000e+00 | 0.0000e+00 | no | 4.99 | 0.031 | PASS |
| nested2 | 64 | 48 | 3072 | 7.7949e-13 | 0.0000e+00 | 4.6242e-02 | 0.0000e+00 | 0.0000e+00 | no | 4.21 | 0.146 | PASS |
| nested3 | 128 | 96 | 12288 | 1.4524e-11 | 0.0000e+00 | 4.3419e-02 | 0.0000e+00 | 0.0000e+00 | no | 6.29 | 0.391 | PASS |
| nested4 | 256 | 192 | 49152 | 1.3917e-11 | 0.0000e+00 | 3.9790e-02 | 0.0000e+00 | 0.0000e+00 | no | 47.85 | 0.205 | PASS |

## Verdict per level
- **nested1**: PASS (corner peak = 9.5017e-07)
- **nested2**: PASS (corner peak = 7.7949e-13)
- **nested3**: PASS (corner peak = 1.4524e-11)
- **nested4**: PASS (corner peak = 1.3917e-11)

## Open questions for follow-up
- If a level FAILs : isolate whether 3-way correction needs porting to deeper levels, or the c2f cascade itself breaks (cf `subcycling_explosion_2d.jl` flagged route mechanism)
- Compare residual interior quadrupole scaling with level (3-way vertices multiply at higher levels)
- Repeat with H multi-patch layout (currently only yband tested at depth)
