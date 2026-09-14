# Oldroyd-B 0D constitutive audit — 2026-05-15

## Why

The cavity comparison at `N = 64`, `De = 1` gave a `~18-24%` L2 profile
error against rheoTool while reproducing the velocity peak to `0.1%`.
The wallclock cost on Aqua H100 was `~33 min` for a single resolution
due to `1829` polymer substeps per LBM step.

Two questions to bisect:

1. Does Kraken's per-cell constitutive integration reproduce the
   analytical Oldroyd-B simple-shear trajectory and steady state?
2. Does rheoTool's rheoTestFoam reproduce the same trajectory and
   steady state on the same shear?

If both yes, the cavity profile drift cannot come from the constitutive
itself — it must originate in the spatial coupling (polymer advection,
velocity-gradient extraction near the moving lid, Guo body-force
discretisation, BSD truncation, Re mismatch).

## Setup

- `lambda = 1`, `etaS = etaP = 0.5` (matching cavity beta=0.5).
- Simple shear: `gradU` has the single non-zero entry `(grad U)_{2,1} = gamma`
  giving `du_x/dy = gamma` (rheoTool convention `(grad u)_ij = du_j/dx_i`,
  Kraken convention `dudy = gamma`).
- Sweep `gamma in {0.01, 0.1, 1.0, 10.0, 100.0}`. The cavity covers
  `gamma_LU = 0.01` (bulk) to `gamma_LU ~ 100` (top-corner lid region).
- `t_end = 12 * lambda` (well past transient).
- Adaptive `dt = 0.01 * min(lambda, 1/|gamma|)` to keep both relaxation
  per-step and deformation per-step below 0.01.

Analytical Oldroyd-B simple shear from rest `C(0) = I`:
```
Cxy(t) = gamma * lambda * (1 - exp(-t/lambda))
Cxx(t) = 1 + 2 * (gamma * lambda)^2 * (1 - exp(-t/lambda) - (t/lambda) * exp(-t/lambda))
Cyy(t) = 1
```

Steady state: `Cxx_ss = 1 + 2*Wi^2`, `Cxy_ss = Wi`, `Cyy_ss = 1` with
`Wi = gamma * lambda`.

## Kraken result

CPU Float64, `bench/viscoelastic_logfv/run_constitutive_0d_vs_rheotest.jl`:

| gamma | Wi | Cxx Kraken | Cxx analytical | Cxy Kraken | Cxy analytical | rel_Linf Cxx | rel_Linf Cxy | n_steps | wall |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.01 | 1.0002 | 1.0002 | 9.95e-3 | 9.99994e-3 | 1.0e-6 | 5.0e-3 | 1200 | <1 ms |
| 0.1 | 0.1 | 1.0199 | 1.0200 | 0.0995 | 0.0999994 | 1.0e-4 | 5.0e-3 | 1200 | <1 ms |
| 1.0 | 1.0 | 2.9899 | 2.9998 | 0.9950 | 0.999994 | 3.3e-3 | 5.0e-3 | 1200 | <1 ms |
| 10  | 10 | 200.88 | 200.98 | 9.9949 | 9.99994 | 5.0e-4 | 5.0e-4 | 12000 | 1 ms |
| 100 | 100 | 19998 | 19999 | 99.994 | 99.9994 | 5.0e-5 | 5.0e-5 | 120000 | 11 ms |

Kraken constitutive matches the analytical trajectory to:

- Cxx: `1e-6 to 3e-3` relative L_inf (best at Wi extremes, worst at
  Wi=1 — Wi=1 is the slowest decaying mode of the polynomial growth
  pre-factor in Cxx, so finite-dt truncation hits its peak there).
- Cxy: roughly constant `~5e-3` relative L_inf at low Wi (small
  absolute scale amplifies relative error) and `5e-5 to 5e-4` at high Wi.

**Conclusion (Kraken side)**: the per-cell log-conformation Oldroyd-B
step is correct. Truncation error scales as expected with the chosen
`dt` step.

## rheoTool result

To be filled after running `bench/rheotool/rheotest_oldroydb/`.

## Performance

- Kraken on CPU Float64: `~10 M constitutive steps / s / cell` at
  `gamma = 100` (120000 steps in 11 ms, single-cell).
- The cavity coupled run on Aqua H100 spends `~33 min` per
  resolution at `N = 64` for `1829 * 102400 ~= 1.9 * 10^8` substep
  cell-ops. Effective throughput on H100 CUDA for this coupled
  pipeline: `0.1 G cell-ops / s` (rough estimate), well below the
  constitutive-only Kraken CPU rate of `10 M steps / s` extrapolated to
  `64*64 = 4096` cells = `40 G cell-ops / s`. The coupled pipeline is
  bottlenecked by the per-substep kernel launches and synchronisation,
  not by constitutive arithmetic.

## Next questions if 0D agrees

- What fraction of the `18-24%` cavity profile error scales with
  resolution? Run `N = 32, 48, 64, 96` (without N=128 if time-limited)
  to see if it's mesh-truncation.
- Does `bsd_fraction in {0.25, 0.5, 0.75}` shift the profile? If a
  smaller bsd makes the profile closer to rheoTool, then the BSD-LBM
  discretisation mismatch is a real contributor.
- Does lowering `u_max` (from 0.005 to 0.001) and thus `Re_LU` from 6.4
  to 1.3 shift the profile? If yes, the Re mismatch is the dominant
  source.

## Files

- Driver: bench/viscoelastic_logfv/run_constitutive_0d_vs_rheotest.jl
- rheoTool 0D case: bench/rheotool/rheotest_oldroydb/
- Kraken trajectories: tmp/constitutive_0d/constitutive_0d_<date>/trajectory_gamma_*.csv
- Summary CSV: same directory, summary.csv
