# Validation matrix

The single summary of where each public Kraken module stands against an external
reference. Every numeric cell is the headline metric of the linked benchmark page —
no value is reproduced here that is not on one of those pages. Each row gives the
**published/analytical reference**, an **independent other-solver comparison** where one
exists, and the **tolerance met** against the mandate §3 acceptance bar.

Where RheoTool covers a case it is the literal reference solver; where it does not, §3's
escape clause anchors the row to the canonical published data set (Ghia 1982 for the
cavity, de Vahl Davis 1983 for thermal, Clift et al. 1978 for sphere drag), each with an
independent OpenFOAM cross-check where available.

## Matrix

| Module | Analytical / published reference | Other-solver comparison | Tolerance target (§3) | Result actually met | Artifact |
|---|---|---|---|---|---|
| **Newtonian — lid-driven cavity** (D2Q9 BGK) | Ghia, Ghia & Shin (1982), centreline `u`/`v` | OpenFOAM `icoFoam` (FVM); independently clears < 1 % vs Ghia (0.49 / 0.23 / 0.46 %) | rel-L2 < 1 % on u-centreline (strict); ≤ 5 % local field | **PASS** — rel-L2 **0.47 % / 0.41 % / 1.05 %** at Re = 100 / 400 / 1000 (Re = 1000 marginal, abs-RMS 0.43 %) | [Cartesian cavity](cartesian-cavity.md); literal both-solver bundle `benchmarks/results/rheotool_compare/newtonian/` (`.krk` + CSVs + `plot.py` + figure + README) |
| **Thermal — natural convection** (double-distribution D2Q9) | de Vahl Davis (1983), `Nu` at Pr = 0.71 | OpenFOAM `buoyantBoussinesqSimpleFoam` (FVM SIMPLE); corroborates dVD at Ra = 10³ (−0.56 %) and 10⁵ (−0.60 %) | < 1 % on Nu; < 2 % on velocity extrema | **PASS** — Nu err **+0.79 % / +0.93 % / +0.79 %** at Ra = 10³ / 10⁴ / 10⁵; velocity extrema all < 1 % | [Thermal natural convection](thermal-natural-convection.md); reproducible bundle (`.krk` + CSV + plot script) forthcoming |
| **Geometry / STL — 3D sphere drag** (D3Q19 + LI-BB cut-link) | Clift, Grace & Weber (1978), free-stream `C_d ≈ 2.61` at Re = 20 | none (no RheoTool unbounded-sphere case); blockage extrapolation `D/W → 0` is the internal cross-check | ≤ 5 % local; resolution-limited geometry gate | **+8.9 %** — extrapolated free-stream `C_d ≈ 2.84` vs Clift 2.61 (quadratic LSQ, R² = 0.9998); residual is finite lattice resolution | [Sphere drag 3D](sphere-drag-3d.md); CSV `bench/geometry_stl/sphere_drag_conv*.csv` |
| **AMR — conservative-tree (route-native D)** | analytic mass conservation (zero net flux ⇒ roundoff drift) | dense-leaf "oracle" Cartesian reference at finest leaf-equivalent resolution | mass drift at roundoff (< 1e-12); route-native field parity vs oracle | **PASS (conservation)** — relative mass drift **~1.7e-13** (square/cylinder), all obstacle rows < 1e-12 (worst 2.08e-12); full-domain refined-patch parity vs oracle **L2 = 0, L∞ = 0** (bit-exact). Compact-patch cylinder Cd is 1.06× the oracle (near-interface transport gap, documented, not a gate) | [Refinement showcase](../../benchmarks/refinement_showcase.md); CSV `benchmarks/results/amr_obstacle_convergence_2d_aqua_conv_20757949.csv`; design note `docs/design/amr_d_publication_validation.md` |
| **Viscoelastic — Oldroyd-B cylinder** (TRT-LBM, halfway BB) | none published at this confinement (Oldroyd-B `C_d` is solver-dependent) | RheoTool `rheoFoam` fine-mesh (`Cylinder/Oldroyd-BLog`), log-conformation | < 1 % on `C_d` at Wi ≤ 1, R = 50, Re = 1 | **PASS** — `C_d` within **−0.40 % / −0.86 % / −0.96 %** of rheoFoam at Wi = 0.1 / 0.5 / 1.0 (diffusive scaling τ = 0.95, halfway BB); validated only for Wi ≤ 1 | [Viscoelastic cylinder](../tutorials/viscoelastic-cylinder.md) |
| **GPU — single-GPU efficiency** (BGK D2Q9, CUDA F64) | memory-bandwidth roofline (A100-40GB, 1.555 TB/s ⇒ 5115 MLUPS ceiling) | Palabos single-GPU F64 D3Q19 TRT (4656 MLUPS, published) | ≥ 0.5 of roofline | **PASS** — **3461 MLUPS** at N = 2048 ⇒ **0.68** of A100-40GB roofline, **0.74** of the Palabos number (0.52 even vs the stricter 80 GB ceiling) | [GPU certification](gpu-certification.md); CSV `benchmarks/results/certification_a100.csv` |

## Notes

- **Follow the artifact link** for the full error-norm tables, methodology, mesh/backend
  details and caveats — the matrix carries only the headline metric.
- **3D rows carry larger residuals** and are documented as resolution-limited on their
  pages (sphere drag +8.9 %). 3D natural convection is wired (`natural_convection_3d`
  preset) but only smoke-validated; a quantitative 3D thermal benchmark is a future item.
  The 2D rows are the strict acceptance gates.
- **AMR conservation ≠ AMR accuracy.** Mass conservation is exact to roundoff and
  full-domain patch parity is bit-exact, but the compact-patch cylinder `C_d` still
  carries a near-interface transport gap (~6 %) — reported, not hidden, and motivating
  the subcycling integration tracked for the next release.
- **Literal reference-solver backfill.** The Newtonian cavity row ships a complete
  `benchmarks/results/rheotool_compare/newtonian/` bundle (both-solver CSVs, per-Re error
  norms, the `.krk`, and a self-contained `plot.py`). The thermal and viscoelastic
  literal bundles are the next legs of the same backfill.
