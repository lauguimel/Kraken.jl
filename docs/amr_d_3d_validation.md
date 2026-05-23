# AMR-D 3D validation (Phase 5)

**Commit**: post-Phase 4 (HEAD)
**Branch**: slbm-paper
**Date**: 2026-05-23
**Backend**: CPU Float64
**Runner**: low-level primitives (collide_Guo + stream_composite_routes_periodic_x_wall_yz, D3Q19)
**Layout**: single fine patch in 3D, ratio=2 (nested-1)

## Hypothesis
The M-H-ETA-7+8 3-way corner correction module is 2D-only by name (`three_way_corner_correction_2d.jl`). Does the same corner artifact exist in 3D (8 cube corners) on Poiseuille AMR-D? If yes, the module needs a 3D port.

## Matrix
| Test | Nx | Ny | Nz | leaf | corner_peak | mass_drift | speed_max | rho_range | NaN? | Wall(s) | Verdict |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---|
| 3D_small_n1 | 8 | 8 | 6 | 16×16×12 | 1.4453e-01 | 9.8588e-14 | 3.8883e-02 | [8.3176e-01, 9.2553e-01] | no | 0.60 | FAIL (corner) |
| 3D_medium_n1 | 12 | 10 | 8 | 24×20×16 | 1.3021e-01 | -4.6908e-13 | 4.0619e-02 | [8.4524e-01, 9.2053e-01] | no | 0.10 | FAIL (corner) |

## Per-corner detail (Test 1)
8 cube corners (leaf coords) of `3D_small_n1`:
- corner (1, 1, 1): rho = 8.5548e-01, |ρ-1| = 1.4452e-01
- corner (16, 1, 1): rho = 8.5547e-01, |ρ-1| = 1.4453e-01
- corner (1, 16, 1): rho = 8.5548e-01, |ρ-1| = 1.4452e-01
- corner (16, 16, 1): rho = 8.5547e-01, |ρ-1| = 1.4453e-01
- corner (1, 1, 12): rho = 8.5548e-01, |ρ-1| = 1.4452e-01
- corner (16, 1, 12): rho = 8.5547e-01, |ρ-1| = 1.4453e-01
- corner (1, 16, 12): rho = 8.5548e-01, |ρ-1| = 1.4452e-01
- corner (16, 16, 12): rho = 8.5547e-01, |ρ-1| = 1.4453e-01

## Verdict per test
- **3D_small_n1**: FAIL (corner) (corner peak = 1.4453e-01)
- **3D_medium_n1**: FAIL (corner) (corner peak = 1.3021e-01)

## Discriminating tests — body force does NOT cause the drop

To rule out a body-force / Mach-number explanation, we re-ran the small fixture with varying Fx and step counts:

| Fx | Steps | rho at coarse corner (1,1,1) | |Δρ| |
|---|---:|---:|---:|
| 0.0 | 200 | 0.855476 | 1.445e-01 |
| 1e-7 | 200 | 0.855476 | 1.445e-01 |
| 1e-7 | 50 | 0.854877 | 1.451e-01 |
| 1e-5 | 50 | 0.854880 | 1.451e-01 |
| 1e-5 | 10 | 0.803532 | 1.965e-01 |

The defect is **independent of the body force** (Fx=0 produces the same corner drop) and reaches a steady-state ≈ 0.855 within 50 steps. Total mass is conserved to 1e-13 — the missing density at coarse corners has accumulated inside the fine patch.

Mass-accounting estimate: coarse cells outside patch = 320, total active mass conserved at 384, so coarse cells lose ≈ 46 mass units to the fine patch (initial fine total = 64; steady ≈ 110). This is a ~71% over-accumulation in the fine patch — a c/f mass-balance bug, NOT a corner-vertex artifact.

## Implication

The 3D AMR-D path exhibits a **mass-balance defect at the coarse-fine interface** that is **distinct from the 2D 3-way vertex bug** addressed by M-H-ETA-7+8. The 2D fix (`three_way_corner_correction_2d.jl`) targets the planar c/f vertex prolongation; the 3D defect appears to be a different class (volumetric c/f mass transfer, present even on simple single-patch Poiseuille with zero body force).

**Open mission for follow-up**: investigate the 3D coarse-fine mass transfer in `stream_composite_routes_periodic_x_wall_yz_F_3d!` and the coarse-fine equilibrium initialization in `_fill_rest_composite_F_3d!`. Not a port of the 2D 3-way module — likely a separate diagnostic + fix.

## Status declared in Phase 5
- **2D AMR-D**: production-ready (Phase 1-4 validated, nested-1 to nested-4 all PASS)
- **3D AMR-D**: NOT production-ready (single-patch Poiseuille shows 14% corner Δρ; mass-balance defect at c/f interface). Use 2D only for the SLBM paper pending dedicated 3D investigation.
