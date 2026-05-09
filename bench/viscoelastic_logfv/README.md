# Viscoelastic Log-FV Benchmarks

This directory holds benchmark and audit harnesses for the viscoelastic branch.
The current cylinder script uses the legacy Liu-style population driver. It is
not the production cell-centered log-FV polymer backend.

## Legacy Cylinder Cd Convergence

`cylinder_cd_convergence.jl` runs confined-cylinder `Cd` sweeps against the
local Liu 2025 and RheoTool reference tables.

It records one CSV row per completed case, including:

- `Cd_report`, `Cd_post`, `Cd_split`, and force-path metadata.
- Liu and RheoTool reference values when available.
- mesh, Wi, beta, relaxation, TRT magic, source, wall, and gradient settings.
- basic stability diagnostics: first nonfinite step, `min_det_C`, `max_tr_C`.

The script refuses long CPU runs unless `KRAKEN_ALLOW_LONG_CPU=1` is set.

Local smoke:

```bash
KRAKEN_SMOKE=1 KRAKEN_BACKEND=metal KRAKEN_R_LIST=4 KRAKEN_WI_LIST=0.1 \
  KRAKEN_STEPS=8 KRAKEN_STEPS_LOW_WI=8 KRAKEN_MAX_STEPS_CAP=8 \
  julia --project=. bench/viscoelastic_logfv/cylinder_cd_convergence.jl
```

Liu Table 3 sweep:

```bash
KRAKEN_BACKEND=cuda KRAKEN_CYLINDER_SUITE=liu \
  KRAKEN_CASES=direct_cnebb KRAKEN_R_LIST=20,30,35 \
  KRAKEN_WI_LIST=0.1,0.5,1.0 KRAKEN_CONFORMATION_MAGIC_LIST=1e-6 \
  julia --project=. bench/viscoelastic_logfv/cylinder_cd_convergence.jl
```

RheoTool comparison at `R=30`:

```bash
KRAKEN_BACKEND=cuda KRAKEN_CYLINDER_SUITE=rheotool \
  KRAKEN_CASES=direct_cnebb,logconf_logfield KRAKEN_R_LIST=30 \
  KRAKEN_WI_LIST=0.05,0.1,0.2,0.5,1.0 \
  julia --project=. bench/viscoelastic_logfv/cylinder_cd_convergence.jl
```

Notes:

- `solvent_source_mode=integrated_collision` makes `Cd_report == Cd_post`.
  `Cd_split` is only meaningful for the standalone `post_collision` audit path.
- `KRAKEN_CONFORMATION_MAGIC_LIST` is explicit by design. Do not treat it as a
  hidden fitting knob.
- `direct_cnebb` is the Liu-reproduction case. `logconf_logfield` is a
  diagnostic legacy log-population case, not the new cell-centered log-FV path.

## Production Log-FV Cylinder Cd Convergence

`logfv_cylinder_cd_convergence.jl` runs the cell-centered log-conformation
FV/FD backend through `Kraken.run_viscoelastic_logfv_cylinder_coupled_2d`.
It uses the benchmark geometry by default:

- `H = 4R`
- `L_up = 15R`
- `L_down = 15R`
- `Fx_body = 0`
- `Re_R = U_mean R / nu_total = 1`

The reported `Cd` is the physical split drag corrected for BSD:

```text
Cd = Cd_s + Cd_p - Cd_bsd
```

where `Cd_s` is the LI-BB/Mei solvent drag and `Cd_p` is the polymer traction
drag reconstructed from the log-FV stress. `Cd_bsd` removes the artificial
Newtonian traction already embedded in the LBM viscosity when
`bsd_fraction > 0`. This corrected split is the comparison target for
RheoTool. Liu references are still recorded, but they are secondary because the
production backend is not the Liu population CDE.

The harness writes `lups` to the CSV and prints `MLUPS` for each completed
case. If any scalar stability or drag diagnostic is non-finite, the row status
is `nonfinite` rather than `ok`.

Local Metal smoke:

```bash
KRAKEN_BACKEND=metal KRAKEN_SMOKE=1 \
  julia --project=. bench/viscoelastic_logfv/logfv_cylinder_cd_convergence.jl
```

Near-Newtonian audit, still local and short unless explicitly allowed:

```bash
KRAKEN_BACKEND=metal KRAKEN_LOGFV_CYLINDER_SUITE=nearnewtonian \
  KRAKEN_R_LIST=4 KRAKEN_STEPS=200 KRAKEN_STEPS_LOW_WI=200 \
  julia --project=. bench/viscoelastic_logfv/logfv_cylinder_cd_convergence.jl
```

Long benchmark runs should use CUDA/HPC or set `KRAKEN_ALLOW_LONG_LOCAL=1`
intentionally. The script refuses long CPU/Metal runs by default.
