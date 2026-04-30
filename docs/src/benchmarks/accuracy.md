# Accuracy and convergence

This page reports only checks that were rerun locally on 2026-04-30 or are
backed by CSV artifacts in `benchmarks/results/`.

The convergence concern is real: several development-branch features in
`slbm-paper` had bugs, and many old documentation claims were ahead of this
branch. Those claims are not repeated here.

## Poiseuille channel

Plane Poiseuille flow is driven by a uniform body force between two no-slip
walls. The analytical half-way bounce-back profile is

```math
u_x(y) = \frac{F_x}{2\nu}(y - 1/2)(H + 1/2 - y).
```

Local rerun:

```bash
julia --project=. benchmarks/convergence_poiseuille.jl
```

| `Ny` | `L2` error | Order |
|---:|---:|---:|
| 16 | 1.4977e-03 | - |
| 32 | 3.7442e-04 | 2.00 |
| 64 | 9.3605e-05 | 2.00 |
| 128 | 2.3401e-05 | 2.00 |

CSV artifacts:

- `benchmarks/results/convergence_poiseuille_apple_m2_20260410_115121.csv`
- `benchmarks/results/convergence_poiseuille_aqua_h100_20260410_115434.csv`

Both CSVs match the local rerun values.

## Taylor-Green vortex

Taylor-Green vortex decay checks the effective viscosity in a fully periodic
domain.

Local rerun:

```bash
julia --project=. benchmarks/convergence_taylor_green.jl
```

| `N` | `L2` error | Order |
|---:|---:|---:|
| 16 | 2.5419e-02 | - |
| 32 | 6.3782e-03 | 1.99 |
| 64 | 1.5897e-03 | 2.00 |
| 128 | 3.9755e-04 | 2.00 |

## Thermal conduction

The current fixed-temperature wall treatment has a half-cell geometric error,
so first-order convergence in `L_inf` is expected here.

Local rerun:

```bash
julia --project=. benchmarks/convergence_thermal.jl
```

| `Ny` | `L_inf` error | Order |
|---:|---:|---:|
| 8 | 6.2500e-02 | - |
| 16 | 3.1250e-02 | 1.00 |
| 32 | 1.5625e-02 | 1.00 |
| 64 | 7.8125e-03 | 1.00 |
| 128 | 3.9062e-03 | 1.00 |

This is not a second-order thermal boundary result. Do not describe it as
such unless the boundary scheme changes.

## Natural convection

For the De Vahl Davis `Ra = 1e3`, `Pr = 0.71` square-cavity reference:

| `N` | `Nu_Kraken` | `Nu_ref` | Relative error |
|---:|---:|---:|---:|
| 64 | 1.1423 | 1.1180 | 2.17% |

This is a useful smoke validation for the 2D thermal path. Higher Rayleigh
numbers, 3D natural convection, refinement and body-fitted cases should be
rerun and CSV-backed before being documented as benchmark results.
