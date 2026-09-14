# Contraction41 Axis-Aligned Pivot — 2026-05-14

## Objective

Pivot away from the curved-cylinder/cut-cell ratchet and test the same log-FV
polymer pipeline on an axis-aligned 4:1 contraction.

The intended discriminator is:

```text
axis-aligned contraction matches RheoTool
=> cylinder drift is curved cut-cell/wall handling

axis-aligned contraction disagrees or diverges
=> core FV polymer pipeline needs bisection
```

## Implemented This Session

- Added `run_viscoelastic_logfv_contraction_coupled_2d`, mirroring the BFS
  driver and calling `contraction_step_geometry_2d`.
- Exported the new contraction driver from `src/Kraken.jl`.
- Added the project-local RheoTool case:
  `bench/rheotool/contraction41_oldroydb_log/`.
- Added a Docker helper for that case:
  `bench/rheotool/contraction41_oldroydb_log/run_docker.sh`.
- Added a centerline sample to the copied RheoTool `sampleDict`.
- Added the quantitative Kraken harness:
  `bench/viscoelastic_logfv/run_contraction41_oldroydb_vs_rheotool.jl`.

## Local Smoke Result

Command:

```bash
KRAKEN_CONTRACTION_H_OUT_LIST=4 \
KRAKEN_L_UP=2 \
KRAKEN_L_DOWN=2 \
KRAKEN_STEPS=2 \
KRAKEN_MAX_LOCAL_UPDATES=1e8 \
KRAKEN_OUTPUT_DIR=tmp/contraction41_axis_aligned_smoke \
julia --project=. bench/viscoelastic_logfv/run_contraction41_oldroydb_vs_rheotool.jl
```

Result:

```text
case=kraken_Hout4
status=ok
Nx=16
Ny=16
steps=2
min_c_eig=0.9572662711
max_speed=0.1116417050
tau_xx_peak=1.7374697e-4
centerline_u_peak/u_mean=4.0931556
```

The smoke only validates the new glue and reporting path; it is not a physics
comparison.

## RheoTool Reference Status

The project-local RheoTool case was launched with `./run_docker.sh`, but the
stock tutorial is too slow for an interactive local refresh:

```text
time reached: 0.0204
wall-clock: about 112 s
target: t = 20
```

The Docker job was stopped deliberately; no container was left running.

This means the four-observable RheoTool comparison is still pending. The
project-local case is ready for an overnight or HPC run.

## Next Action

Run the project-local RheoTool contraction case to `t = 20`, then rerun:

```bash
KRAKEN_CONTRACTION_H_OUT_LIST=4,8,16 \
KRAKEN_L_UP=64 \
KRAKEN_L_DOWN=64 \
KRAKEN_STEPS=<steady Kraken step count> \
julia --project=. bench/viscoelastic_logfv/run_contraction41_oldroydb_vs_rheotool.jl
```

If Kraken diverges before the RheoTool comparison is available, reduce the
test to the axis-aligned contraction toggle matrix:

1. source-only with frozen velocity,
2. advection-only from a known `Psi`,
3. force-only coupling from a prescribed stress field.
