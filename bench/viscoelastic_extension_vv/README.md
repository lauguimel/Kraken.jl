# M44-VV-A viscoelastic extension V&V

This bench tree contains the 4-roll mill extension validation suite for the log-FV Oldroyd-B polymer chain.

## Test A: periodic 4-roll mill

`A_extensional/run_4roll_mill_sweep.jl` runs one case or the 48-case Wi x N x beta matrix. It reuses the existing Kraken log-conformation FV entry points and the periodic LBM Hermite stress-source kernel without editing `src/`.

Smoke:

```bash
julia --project=. bench/viscoelastic_extension_vv/A_extensional/run_4roll_mill_sweep.jl --smoke
```

Full matrix and analysis:

```bash
julia --project=. bench/viscoelastic_extension_vv/A_extensional/run_4roll_mill_sweep.jl --all
julia --project=. bench/viscoelastic_extension_vv/A_extensional/analyse_4roll_mill.jl
```

Raw profiles are written to `bench/scratch/m44_vv_a/profiles/`; per-case scalar CSVs and the aggregate result CSV are written under `A_extensional/csv/`.

## Tests B and C

Reserved for follow-up missions after Test A determines whether the clean extension path is green enough to justify more complex cross-code or geometry-coupled validation.
