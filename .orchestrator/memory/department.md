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
