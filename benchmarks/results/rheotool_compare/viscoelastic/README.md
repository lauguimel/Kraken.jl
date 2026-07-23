# Viscoelastic reference-solver comparison — confined-cylinder drag (Oldroyd-B)

§3 release-mandate artifact for the **viscoelastic** module: a literal comparison of
Kraken's Oldroyd-B solver against the viscoelastic reference solver **RheoTool**
(rheoFoam, OpenFOAM-9, log-conformation) on the canonical confined-cylinder problem —
the standard High-Weissenberg-Number-Problem (HWNP) testbed of Alves, Oliveira & Pinho
(2001).

## Case

Confined cylinder of radius `R` on the channel centreline, blockage ratio `D/H = 0.5`
(half-height `2R`), upstream/downstream length `15R`. Oldroyd-B fluid, `Re = 1`, solvent
fraction `β = η_s/η₀ = 0.59` (the Boger-fluid value used throughout the cylinder
literature). Diffusive scaling: fixed lattice viscosity `ν_total = 0.15`, `τ = 0.95`;
`u_mean = ν_total·Re/R`, `λ = Wi·R/u_mean`. Half-way bounce-back cylinder wall.

| Solver | Method | Grid | Notes |
|--------|--------|------|-------|
| **Kraken** | TRT-LBM + log-conformation finite-volume polymer transport | `R = 50` (cylinder diameter `100` LU), `±15R` channel | MUSCL–superbee advection of the conformation field, half-way bounce-back wall, v0.3 cut-link momentum-exchange drag. Diffusive scaling `ν_total = 0.15` (`τ = 0.95`), `300 000` steps. Backend **CUDA Float64 (Aqua A100)**, job 22199679. |
| **RheoTool `rheoFoam`** | FVM, log-conformation Oldroyd-BLog | `Cylinder/Oldroyd-BLog` tutorial mesh | OpenFOAM-9 base, 2-core MPI, steady at `t = 20`. `β = 0.59`, `Re = 1`, `D/H = 0.5`. |
| **Alves–Oliveira–Pinho (2001)** | published reference | finite-volume high-resolution | the canonical cylinder HWNP benchmark family. |

## Files

- `kraken_cd_vs_wi.csv` — Kraken drag at `Wi ∈ {0.1, 0.5, 1.0}`, R = 50, β = 0.59.
  Columns: `Wi, Cd, Cd_s, Cd_p`. `Cd` is the total drag coefficient (`Cd_kraken`,
  source column 26); `Cd_s`/`Cd_p` are the raw solvent/polymer wall-ring integrator
  diagnostics (source columns 27/28), reported verbatim. Source:
  `bench/viscoelastic_logfv/v03_cutlink_reval_22199679/reval_R50/cyl_bigsweep_v2_*.csv`.
- `rheotool_cd_vs_wi.csv` — RheoTool reference drag at the same `Wi`. Columns: `Wi, Cd`.
- `error_norms.csv` — per `Wi`, the relative error `(Cd_kraken − Cd_rheotool)/Cd_rheotool`
  (and its absolute percentage). This is the §3 integrated-quantity gate.
- `cylinder_oldroyd_b.krk` — the Kraken case file (smoke wiring; the production driver
  rebuilds the cylinder channel from `R`, `L_up`, `L_down`). Reproduce with
  `run_simulation("cylinder_oldroyd_b.krk")`.
- `comparison.png` — Cd vs Wi: Kraken filled markers + connecting line, RheoTool dashed
  reference line + open squares, per-point relative error annotated. Dark Documenter
  theme. Also copied to `docs/src/users/benchmarks/viscoelastic-cylinder.png`.
- `plot.py` — **self-contained reproducer**: reads the two `*_cd_vs_wi.csv` files in this
  directory and regenerates `comparison.png` (csv + matplotlib + seaborn; LaTeX if a
  system `latex` is present, else mathtext — no external paths). Run
  `conda run -n kraken-v0-3-figures python plot.py`.
- `m8_refs/` — the prior dev-viscoelastic "M8" study CSVs reproduced by this revalidation
  (`matrix_halfwayBB_R{10,30,50}.csv` = Cd vs Wi per R; `beta_R50.csv` = β-sweep;
  `staircase_R{30,50}.csv` = high-Wi stability envelope; `N1_comparison.csv` = wake-N1
  Kraken/RheoTool ratio). The v0.3 cut-link drag reproduces the M8 `matrix_halfwayBB_R50`
  Cd to 4–5 significant figures.

## Headline result — Cd vs Wi (R = 50, β = 0.59)

| Wi  | Kraken Cd | Kraken Cd_p | RheoTool Cd | rel. error |
|-----|-----------|-------------|-------------|------------|
| 0.1 | 129.9155  | 16.097      | 130.43      | **−0.40 %** |
| 0.5 | 118.6770  | 14.837      | 119.71      | **−0.86 %** |
| 1.0 | 119.2410  | 14.155      | 120.40      | **−0.96 %** |

All three clear the strict **< 1 %** integrated-quantity gate. Kraken reproduces the
characteristic elastic signature reported across the cylinder literature: a shallow drag
**minimum near `Wi ≈ 0.5`** followed by an **elastic upturn** toward `Wi = 1`, not just a
single matched value.

## Tolerance met

§3 requires **≤ 1 % integrated**. The integrated drag `Cd` lands at −0.40 / −0.86 / −0.96 %
across `Wi ∈ {0.1, 0.5, 1.0}` — **all three under 1 %**.

**Verdict: PASS at the strict < 1 % integrated-Cd gate.**

## Provenance

**NOT CI-reproducible.** All Kraken numbers are **Float64 on an NVIDIA A100 (CUDA)**,
300 000 steps, run on the Aqua HPC cluster (job **22199679**). These are not produced by
the local test suite or by GitHub CI — reproducing them requires an A100-class GPU and an
overnight run. The RheoTool references are rheoFoam (OpenFOAM-9, Oldroyd-BLog,
log-conformation), 2-core MPI, steady at `t = 20`. This artifact re-derives the comparison
CSVs, error norms and comparison plot from those raw outputs. The v0.3 cut-link drag
reproduces the prior dev-viscoelastic "M8" study (`m8_refs/`) to 4–5 significant figures.

## References

- M. A. Alves, P. J. Oliveira, F. T. Pinho (2001), *The flow of viscoelastic fluids past
  a cylinder: finite-volume high-resolution methods*, J. Non-Newtonian Fluid Mech. **97**,
  207–232.
- M. A. Hulsen, R. Fattal, R. Kupferman (2005), *Flow of viscoelastic fluids past a
  cylinder at high Weissenberg number: stabilized simulations using matrix logarithms*,
  J. Non-Newtonian Fluid Mech. **127**, 27–39.
- R. Fattal, R. Kupferman (2004), *Constitutive laws for the matrix-logarithm of the
  conformation tensor*, J. Non-Newtonian Fluid Mech. **123**, 281–285.
- S. Claus, T. N. Phillips (2013), *Viscoelastic flow around a confined cylinder using
  spectral/hp element methods*, J. Non-Newtonian Fluid Mech. **200**, 131–146.
