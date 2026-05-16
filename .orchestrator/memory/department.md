# Department memory — Kraken.jl viscoelastic cavity spatial debug

Patterns useful for every Department on this project. Initialised
2026-05-15.

## 2026-05-15 — Engineer-brief conventions for log-FV cavity work

- Always specify `KRAKEN_BACKEND` explicitly in the brief's validation
  env section. Default detection picks Metal on macOS (F32) which is
  NOT comparable to the Aqua A100 F64 baseline.
- The cavity comparison harness
  `bench/viscoelastic_logfv/run_cavity_oldroydb_vs_rheotool.jl` already
  consumes env vars `KRAKEN_N_LIST`, `KRAKEN_U_MAX`,
  `KRAKEN_OUTPUT_DIR`, `KRAKEN_LAMBDA_PHYS`, `KRAKEN_BSD_FRACTION`.
  Engineer briefs should set these via env, never patch the script.
- For PBS wrappers: keep `Pkg.instantiate(); Pkg.precompile()` out of
  the per-case loop. Run it once at the top of the job.

**Why**: M1 brief writing will repeatedly hit these. Skipping the env
discipline pollutes the baseline.

## 2026-05-15 — rheoTool reference loaders

`run_cavity_oldroydb_vs_rheotool.jl` defines:
- `read_rheotool_vertical_U(path)` — 4 cols (y, Ux, Uy, Uz)
- `read_rheotool_horizontal_tautheta(path)` — 13 cols (x, tau×6, theta×6)

Any new analysis script should import / replicate these instead of
re-deriving column ordering. Reference layout:
`bench/rheotool/cavity_oldroydb_log_re001_de1_b05/postProcessing/...`.

**Why**: column ordering is non-obvious; getting it wrong silently
inverts the comparison.

## 2026-05-15 — Per-case CSVs already carry rheoTool aligned to Kraken grid

`run_cavity_oldroydb_vs_rheotool.jl` writes
`profile_vertical_x0.5.csv` and `profile_horizontal_y0.75.csv` with
both Kraken AND rheoTool columns already interpolated onto the
rheoTool sample grid. Downstream analyses should read those columns,
NOT re-load the rheoTool `.xy` files.

**Why**: avoids the 4/13-col loader trap; also prevents
interpolation-method drift between scripts. Found during M1.

## 2026-05-16 — Codex sandbox cannot run julia (lockfile EPERM)

Codex CLI's `workspace-write` sandbox blocks `juliaup` / `julia` from
creating the launcher lockfile (`Operation not permitted (os error 1)`).
The Engineer therefore CANNOT execute the exit-criterion command for
any Julia mission and will stop there with a clean diff. The Department
MUST re-run the exit criterion itself on the host shell. Plan briefs
accordingly: write a self-test that prints a grep-able summary so the
Department's re-run is the single source of truth for success.

**Why**: avoids treating "Engineer stopped at validation" as failure
when the only blocker is a sandbox capability. Found during M2.

## 2026-05-16 — rheoTool cavity persisted snapshots at t≈8

The cavity reference case
`bench/rheotool/cavity_oldroydb_log_re001_de1_b05/` writes
**full cell-data** fields (not only sampleDict probes) at ~1 phys
time stride, including the time directory closest to t=8 which is
`7.999786329655222/`. Persisted gzipped fields: `U.gz`, `theta.gz`
(symmTensor = log-conformation), `tau.gz`, `p.gz`, `phi.gz`, plus
`eigVals.gz`, `eigVecs.gz`, `ddt0(U).gz`, `ddt0(theta).gz`. Mesh is
127×127 (= 16129 cells) on [0,1]² with z extruded ±0.5. Both `U` and
`theta` can be parsed by the cylinder harness's FOAM reader
(`parse_vol_vector`, `parse_vol_symmtensor`) — same regex / path
conventions.

**Why**: avoids the assumption in earlier prompts that only the 1D
sampleDict probes survive at t=8. The full field is available, so
frozen-replay across geometries does not need to extrapolate from
profile samples.

## 2026-05-16 — rheoTool `theta` IS log(C), not C

In rheoTool OldroydBLog the persisted symmTensor `theta` is already
the log-conformation `log(C) = Psi` in Kraken wording. Initialising
the Kraken polymer state from `theta` requires NO exp/log transform
— direct copy of the 6 components in symmTensor order
`(xx, xy, xz, yy, yz, zz)`. Confirmed by the cavity harness loader
docstring (~line 94 of `run_cavity_oldroydb_vs_rheotool.jl`) and
used directly in `run_rheotool_frozen_replay_cavity_2d.jl`. For
geometries where only `tau` is persisted, the cylinder harness's
`psi_from_tau` (Newtonian prefactor → C → log) remains the fallback.

**Why**: the cylinder frozen-replay re-derives Psi from `tau`. For
cavity (and any other rheoTool log-conformation case) reading
`theta` directly is shorter, exact, and skips the SPD-positivity
check.

## 2026-05-16 — Reusable polymer-pipeline frozen-replay call pattern

For a frozen-U replay of ONLY the log-FV polymer pipeline on any
axis-aligned 2D geometry without an embedded obstacle, per step:

```
logfv_cell_velocity_to_faces_bc_aware_2d!(..., logfv_bc)
logfv_advect_upwind_bc_aware_2d!(..., dummies, ux_face, uy_face,
                                 is_solid, dx, dy, logfv_bc, one(T))
fvfd_velocity_gradient_2d!(..., is_solid, dx, dy, logfv_bc)
for k in 1:n_substeps:
    logfv_step_constitutive_log_2d!(..., lambda_lu, dt_poly,
                                    LOGFV_MODEL_OLDROYDB, T(0.0))
logfv_stress_from_log_2d!(..., prefactor)
```

with `logfv_bc = logfv_wallxwally_bcspec_2d()` for closed boxes,
`logfv_periodicx_wally_bcspec_2d()` for x-periodic channels,
`logfv_openx_wally_bcspec_2d()` for inlet/outlet. DO NOT call
`_logfv_cavity_apply_wall_gradient_correction!` in a frozen replay:
rheoTool's U already encodes the lid shear at cells adjacent to the
wall, and the LBM ghost-correction would double-count.
`dt_poly = 1 / n_substeps` in LU (source kernel expects LU time).

**Why**: this is the third frozen-replay harness in the project
(cylinder, channel, now cavity). The pattern is identical modulo BC
spec; future contraction / BFS replays can copy it directly.
