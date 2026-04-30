# Kraken.jl benchmark suite

Benchmark scripts write timestamped CSV files into `benchmarks/results/`.
Documentation should cite only numbers that can be traced to those CSVs.

The public v0.1.0 benchmark scope is conservative:

- single-phase BGK flows;
- thermal DDF checks;
- CPU baseline plus H100 once fresh H100 throughput CSVs are committed.

Grid refinement, VOF, rheology, viscoelasticity and SLBM/body-fitted work are
development-branch topics and should not be mixed into this branch's public
benchmark tables.

## Quick start

```bash
# Fast/default CPU-oriented run
julia --project=. benchmarks/run_all.jl --quick --hardware-id=my_machine

# Full CPU sweep
julia --project=. benchmarks/run_all.jl --hardware-id=my_machine

# GPU sweep, after selecting a real hardware id from hardware.toml
julia --project=. benchmarks/run_all.jl --gpu --hardware-id=aqua_h100
```

## Flags

| Flag | Purpose |
|---|---|
| `--quick` | Reduced subset |
| `--gpu` | Run GPU benchmarks when the backend is available |
| `--hardware-id=<key>` | Label matching `benchmarks/hardware.toml` |
| `--skip-existing` | Skip cases whose CSV already exists |
| `--output-dir=<path>` | Override `benchmarks/results` |

## Hardware ids

Declared hardware currently includes:

- `apple_m3max` — local Metal development artifact;
- `aqua_h100` — QUT Aqua H100 GPU node;
- `aqua_a100` — QUT Aqua A100 GPU node.

Some old CSVs still use `apple_m2`. Keep them as historical raw data, but do
not make them the public benchmark story.

## Scripts

| Script | What it measures | Publication status |
|---|---|---|
| `convergence_poiseuille.jl` | L2 error vs grid size | verified locally, CSV-backed |
| `convergence_taylor_green.jl` | Periodic vortex decay | verified locally |
| `convergence_thermal.jl` | Heat conduction and `Ra=1e3` natural convection | verified locally |
| `convergence_cavity.jl` | Ghia centerline error | rerun before citing |
| `perf_mlups.jl` | BGK throughput | cite only committed CSVs |
| `perf_gpu_physics.jl` | Advanced physics throughput | out of v0.1.0 public scope |
| `perf_optimizations.jl` | Optimization experiments | development artifact |

## CSV schema

Convergence files should include at least:

```text
case,N,error_L2,observed_order,hardware_id
```

Performance files should include at least:

```text
case,N,backend,precision,mlups,hardware_id
```

Before adding a number to the docs, record the command, hardware id, commit,
date and CSV filename in the relevant benchmark page or validation note.
