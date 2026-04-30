# Validation notes - 2026-04-30

Local checks were run from `release/v0.1.0` on 2026-04-30.

## Commands and results

```bash
julia --project=. -e 'using Kraken; println("Kraken loaded")'
```

Result: package loaded and precompiled successfully.

```bash
julia --project=. benchmarks/convergence_poiseuille.jl
```

Result:

| Ny | L2 error | Order |
|---:|---:|---:|
| 16 | 1.4977e-03 | - |
| 32 | 3.7442e-04 | 2.00 |
| 64 | 9.3605e-05 | 2.00 |
| 128 | 2.3401e-05 | 2.00 |

```bash
julia --project=. benchmarks/convergence_taylor_green.jl
```

Result:

| N | L2 error | Order |
|---:|---:|---:|
| 16 | 2.5419e-02 | - |
| 32 | 6.3782e-03 | 1.99 |
| 64 | 1.5897e-03 | 2.00 |
| 128 | 3.9755e-04 | 2.00 |

```bash
julia --project=. benchmarks/convergence_thermal.jl
```

Result:

| Ny | L_inf error | Order |
|---:|---:|---:|
| 8 | 6.2500e-02 | - |
| 16 | 3.1250e-02 | 1.00 |
| 32 | 1.5625e-02 | 1.00 |
| 64 | 7.8125e-03 | 1.00 |
| 128 | 3.9062e-03 | 1.00 |

Natural convection: `Ra = 1e3`, `Nu = 1.1423`, reference `1.1180`,
relative error `2.17%`.

```bash
julia --project=. -e 'using Kraken; r=run_simulation("examples/cavity.krk"; max_steps=10); @show maximum(abs.(r.ux)) maximum(abs.(r.uy))'
```

Result: ran successfully and returned nonzero velocity fields.

```bash
julia --project=. test/runtests.jl
```

Result: `379` tests passed in `3m16.3s`.

## Interpretation

Poiseuille and Taylor-Green convergence are healthy for this branch.
Thermal conduction is first-order with the current wall treatment; this is
expected for the documented half-cell boundary error, but it must not be
described as second-order. Natural convection at `Ra = 1e3` is within the
existing tolerance.

H100 throughput claims were not revalidated here because no matching MLUPS CSV
is present in `benchmarks/results/`.
