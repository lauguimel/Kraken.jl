# Reproducibility bundle — patch-based AMR refinement

Self-contained bundle for the **refinement showcase** page
(`docs/src/benchmarks/refinement_showcase.md`). Regenerates the AMR
mass-conservation figure from the shipped CSVs, in the locked house plotting
style (seaborn `crest`, LaTeX, serif whitegrid).

## What this shows

The validated v0.1.0 claim for patch-based grid refinement is **stability /
conservation**, not yet a cost-vs-accuracy win. The figure makes the
conservation claim concrete: the relative mass drift `|Δm|/m` stays at Float64
machine precision (~10⁻¹³, dominated by accumulated round-off over the run, well
above the per-op `ε_mach`) across grid scales and across **both** AMR paths —
the `leaf_oracle` reference and the production `amr_route_native` — for the
square- and cylinder-obstacle channels.

Both methods sit on top of each other in the plot: that overlap is the point —
the route-native AMR conserves mass identically to the oracle.

## Files

- `amr_obstacle_convergence_2d_aqua_conv_20757949.csv`,
  `amr_obstacle_convergence_2d_aqua_conv_20761330.csv` — the two Aqua
  `obstacle_convergence_2d` runs. Columns: `flow, method, scale, Nx, Ny, steps,
  ux_mean, uy_mean, Fx_drag, Fy_drag, Cd, mass_rel_drift, elapsed_s`. The
  plotter reads `mass_rel_drift` vs cells (`Nx·Ny`) and averages the two runs
  per (flow, method, cells).
- `plot.py` — self-contained reproducer (csv + matplotlib + seaborn). Globs both
  CSVs in this directory and writes `comparison.png`. Run `python plot.py`. No
  `.krk` case — this benchmark uses the `obstacle_convergence_2d` Julia driver
  (`benchmarks/convergence_natconv_refinement.jl` family), not the `.krk` runner.
- `comparison.png` — per-flow (cylinder / square) panels: `|Δm|/m` vs lattice
  cells, log–log, colour = cell count via `crest`, marker = AMR method, with the
  Float64 `ε_mach` reference band. Also copied to the page PNG.

## Headline result

Across both AMR paths, both obstacle flows and both grid scales the relative mass
drift stays in the **10⁻¹³** band (e.g. cylinder route-native 1.71×10⁻¹³ at
scale 1, 7.52×10⁻¹³ at scale 2) — i.e. machine precision. Refinement is
conservative; the throughput and accuracy items remain v0.2.0 work, as the page
text documents.
