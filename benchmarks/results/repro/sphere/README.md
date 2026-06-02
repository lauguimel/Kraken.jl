# Reproducibility bundle — 3D STL sphere drag

Self-contained bundle for the **sphere drag** benchmark page
(`docs/src/users/benchmarks/sphere-drag-3d.md`). Regenerates the blockage
convergence figure with the quadratic free-stream extrapolation, in the locked
house plotting style (seaborn `crest`, LaTeX, serif whitegrid).

## Case

Flow past a sphere defined by an STL surface, Re = 20, in a confined channel of
cross-section `W`. The blockage ratio `β = D/W` is swept; the measured drag
coefficient `Cd = 2·Fx / (ρ·u²·A)` converges to the free-stream value as β → 0.
LI-BB cut-link wall treatment on the STL, momentum-exchange drag.

| Series | Method | Notes |
|--------|--------|-------|
| **R = 16 sweep** | Kraken STL, CUDA F64 | the convergence series: β = 6, 8, 10, 14.3, 20 %. |
| **R = 8 probe** | Kraken STL, CUDA F64 | coarser resolution check at β = 20 %. |
| **Clift (1978)** | published standard drag curve | `Cd = (24/Re)(1 + 0.15·Re^0.687) = 2.61` at Re = 20 (here written `1.2·(1 + 0.15·20^0.687)`). |

## Files

- `sphere_drag_conv.csv` — main R = 16 (+ R = 8 smoke) sweep: blockage, Re, Cd,
  drag force, frontal area, solid volume, backend.
- `sphere_drag_conv_lowblock.csv` — the two lowest-blockage R = 16 points
  (β = 6, 8 %) that complete the convergence tail.
- `plot.py` — self-contained reproducer (csv + numpy + matplotlib + seaborn).
  Reads both CSVs, fits the quadratic `Cd = c0 + c1·β + c2·β²`, and writes
  `comparison.png`. Run `python plot.py`.
- `sphere_stl_3d_drag.krk` — the Kraken case file for the STL sphere drag run.
- `comparison.png` — Cd vs blockage `D/W` (%): R = 16 circles (colour = β via
  `crest`), R = 8 open square probe, quadratic fit, the β → 0 extrapolated
  diamond, and the Clift dashed reference. Also copied to the page PNG.

## Headline result

Quadratic LSQ fit of the R = 16 sweep (R² = 0.9998) extrapolates to a
free-stream drag coefficient of **Cd(β→0) ≈ 2.84**, vs the Clift (1978) standard
drag curve **Cd = 2.61** at Re = 20 — a **+8.9 %** confined-to-free-stream
residual at this resolution.

## Reference

Clift, R., Grace, J.R., Weber, M.E. (1978), *Bubbles, Drops, and Particles*,
Academic Press — standard drag correlation `Cd = (24/Re)(1 + 0.15·Re^0.687)`.
