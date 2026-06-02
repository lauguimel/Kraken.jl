# Reproducibility bundle — thermal natural convection

Self-contained bundle for the **thermal natural-convection** benchmark page
(`docs/src/users/benchmarks/thermal-natural-convection.md`). Regenerates the
Nusselt-vs-Rayleigh comparison and the Ra = 10⁵ convergence ladder from the
shipped CSVs, in the locked house plotting style (seaborn `crest`, LaTeX, serif
whitegrid).

## Case

Differentially-heated square cavity: hot west wall (`T = 1`), cold east wall
(`T = 0`), adiabatic north/south walls, no-slip everywhere, Boussinesq buoyancy.
Double-distribution thermal LBM (D2Q9 flow + D2Q9 temperature). Reynolds-free;
the dynamics are set by the Rayleigh and Prandtl numbers. Ra = 10³, 10⁴, 10⁵.

| Solver | Method | Grid | Notes |
|--------|--------|------|-------|
| **Kraken** | thermal LBM, D2Q9×2 | 128² (Ra ≤ 10⁴), 192² (Ra = 10⁵) | CPU F64 and Metal F32 agree to < 0.01 % at the canonical grid; F64 row preferred. |
| **de Vahl Davis (1983)** | published benchmark | de-singularised FD | Canonical mean-Nu reference. |
| **OpenFOAM `buoyantBoussinesqSimpleFoam`** | FVM, steady SIMPLE | 128² / 192² | Ra = 10⁴ point omitted (under-converged, residual plateau ~8×10⁻⁵). |

## Files

- `kraken_natconv_results.csv` — Kraken Nu / u_max / v_max and the de Vahl Davis
  reference per (case, backend, Ra, N). Includes the full Metal F32
  `cavity_finer` N-ladder at Ra = 10⁵ used for the convergence panel.
- `of_natconv_results.csv` — OpenFOAM `buoyantBoussinesqSimpleFoam` Nu (surface
  and volume gradient) per Ra; the `_UNCONVERGED` row is skipped by the plotter.
- `plot.py` — self-contained reproducer (csv + matplotlib + seaborn). Reads the
  two CSVs in this directory and writes `comparison.png`. Run `python plot.py`.
- `natural_convection.krk` — the Kraken case file. Reproduce the Kraken Nu(Ra)
  with `run_simulation("natural_convection.krk")` (sweep Ra via the preset /
  `Physics nu` line; the grids in the table above).
- `comparison.png` — left: Nu vs Ra (Kraken squares, de Vahl Davis open circles,
  OpenFOAM triangles, colour = Ra via `crest`); right: Ra = 10⁵ Kraken Nu-error
  ladder crossing the 1 % gate near N ≈ 384. Also copied to the page PNGs.

## Headline result

| Ra   | Nu (Kraken) | Nu (de Vahl Davis 1983) | err |
|------|-------------|-------------------------|-----|
| 10³  | 1.130       | 1.117                   | +1.2 % |
| 10⁴  | 2.300       | 2.238                   | +2.8 % (128²) |
| 10⁵  | 4.655       | 4.509                   | +3.2 % (192²) |

The Ra = 10⁵ convergence ladder (Metal F32, `cavity_finer`) drives the Nu error
monotonically from 3.1 % (N = 192) below the 1 % gate at N ≈ 384 (0.79 %).

## Reference

de Vahl Davis, G. (1983), *Natural convection of air in a square cavity: a bench
mark numerical solution*, Int. J. Numer. Methods Fluids **3**, 249–264.
