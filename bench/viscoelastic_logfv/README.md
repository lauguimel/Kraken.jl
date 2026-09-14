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

For failure localization, set `KRAKEN_DIAGNOSTIC_STRIDE=N`. This opt-in path
copies diagnostic fields back to the host every `N` steps, stops the case at
the first observed non-finite field, and records `completed_steps` plus
`first_nonfinite_step/field/i/j` in the CSV. Leave it at the default `0` for
production timing runs.

`KRAKEN_FORCE_BOUNDARY_FILL=bc_aware` is the default for the open-x log-FV
coupled driver. It applies the domain boundary conditions directly when forming
the polymer force chain. The older `nearest` fill remains available only for
defect-isolation audits.

`KRAKEN_LOGFV_EMBEDDED_GRADIENT=0` is the production default while the FVFD
cut-cell migration is incomplete. Setting it to `1` injects the lowered
`q_wall` geometry into the velocity-gradient source term only. That mode is
diagnostic, because advection, stress divergence, BSD correction, volume
weighting, and drag are not yet using the same embedded FV geometry.

`KRAKEN_MAX_MEMORY_DEFORMATION_INCREMENT` controls the `:auto`
`polymer_substeps` memory-time criterion:

```text
memory_deformation_increment = lambda * max_grad_norm_estimate
memory_deformation_substeps =
    ceil(memory_deformation_increment / KRAKEN_MAX_MEMORY_DEFORMATION_INCREMENT)
```

The default is `0.07`, which makes the Liu/RheoTool `Wi=0.5` cylinder choose
eight source substeps instead of one. This is a temporal convergence control,
not a physical or benchmark-fitting parameter.

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

A100/H100 PBS run:

```bash
qsub bench/viscoelastic_logfv/run_cylinder_cd_convergence_a100.pbs
```

For the memory-deformation sensitivity canary used on `R35/Wi=0.5`, override
the step count and threshold explicitly:

```bash
KRAKEN_R_LIST=35 KRAKEN_WI_LIST=0.5 KRAKEN_RUN_NEWTONIAN=0 \
  KRAKEN_SCALE_STEPS_WITH_R=0 KRAKEN_STEPS=272222 \
  KRAKEN_MAX_MEMORY_DEFORMATION_INCREMENT=0.07 \
  qsub bench/viscoelastic_logfv/run_cylinder_cd_convergence_a100.pbs
```

Field dumps produced by `KRAKEN_SAVE_FIELDS=1` can be inspected with:

```bash
julia --project=. bench/viscoelastic_logfv/analyze_logfv_field_dump.jl \
  results/viscoelastic_logfv/<run>/fields/<case>/fields.jls

julia --project=. bench/viscoelastic_logfv/analyze_logfv_gradient_dump.jl \
  results/viscoelastic_logfv/<run>/fields/<case>/fields.jls
```

The first script reports non-finite counts, extrema, and distance to the
cylinder wall. The second recomputes regular vs embedded velocity-gradient
statistics on a saved dump.

## Simple Log-FV Validation Outputs

`run_simple_validation_outputs.jl` is the gate before RheoTool macro-flow
comparisons. It runs four short controlled cases and writes ParaView-ready
snapshots plus a summary CSV:

- coupled Poiseuille channel;
- periodic square obstacle;
- open-x square-obstacle channel;
- open-x backward-facing-step coupled flow.

Each case saves `rho`, `ux`, `uy`, `speed`, `Psi`, reconstructed `C`,
reconstructed or native `tau`, polymer/total forces, and `is_solid` to VTK.

Local CPU:

```bash
julia --project=. bench/viscoelastic_logfv/run_simple_validation_outputs.jl
```

Local Metal:

```bash
KRAKEN_BACKEND=metal julia --project=. \
  bench/viscoelastic_logfv/run_simple_validation_outputs.jl
```

Override output location with `KRAKEN_OUTPUT_DIR=/path/to/output`. RheoTool
cylinder convergence should only be interpreted after these simple cases pass.

Build the static HTML dashboard for the latest or a selected output directory:

```bash
python3 bench/viscoelastic_logfv/make_simple_validation_dashboard.py \
  tmp/logfv_simple_validation_outputs/<run-id>
```

The dashboard shows `|u|`, `rho`, the grid/solid mask, and the Poiseuille
centerline profile against the analytical channel solution. If an OpenFOAM
case is available, an additional mesh/velocity panel can be added through
`fluidfoam`:

```bash
python3 bench/viscoelastic_logfv/make_simple_validation_dashboard.py \
  tmp/logfv_simple_validation_outputs/<run-id> \
  --foam-case /path/to/openfoam/case --foam-time latestTime
```

After an Aqua run, sync the latest `simple_validation_*` directory and build
the dashboard locally:

```bash
bash bench/viscoelastic_logfv/sync_simple_validation_from_aqua.sh
```

## Quantitative Frozen-Flow Ladder

Before interpreting RheoTool cylinder convergence, run the quantitative
frozen-channel CDE gate:

```bash
julia --project=. bench/viscoelastic_logfv/run_quantitative_simple_ladder.jl
```

It writes `summary.csv`, per-case profile CSVs, and `dashboard.html` under
`tmp/logfv_quantitative_simple_ladder/<run-id>`. The cases replay analytical
Couette and Poiseuille velocity fields through the production FVFD/log-FV
operator path without LBM feedback:

```text
frozen u -> FVFD face velocities -> FVFD advection -> log-C source
         -> tau_p -> div(tau_p) + BSD diagnostics
```

Useful overrides:

```bash
KRAKEN_BACKEND=metal julia --project=. \
  bench/viscoelastic_logfv/run_quantitative_simple_ladder.jl

KRAKEN_SIMPLE_LADDER_NY=16,32,64 \
KRAKEN_SIMPLE_LADDER_SUBSTEPS=128 \
julia --project=. bench/viscoelastic_logfv/run_quantitative_simple_ladder.jl
```
