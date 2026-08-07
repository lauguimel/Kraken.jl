# EHD electroconvection — critical parameter Tc

Validation evidence for the coupled electroconvection solver: the onset of
electroconvective instability in a 2D dielectric liquid layer with unipolar
charge injection, measured as the electric Rayleigh number T at which the
perturbation growth rate sigma changes sign.

Only the definitive post-fix runs are kept here. Earlier rounds (the
phi-frozen round and everything before the MRT half-force fix, `bb066bcc1`)
were invalidated and have been removed; new sweep output is gitignored by
default, see the rule in the repo `.gitignore`.

## Configuration

All runs below share the same setup:

| | |
|:--|:--|
| grid | 197 x 321 |
| NS collision | MRT (`--ns-scheme=mrt`) |
| charge scheme | regularized |
| potential solve | `--phi-scheme=direct` (assembled Laplacian, cuDSS on GPU) |
| force projection | `--force-projection=xy` |
| cycles | 600 000 |
| hardware | 1x H100 (Aqua), one PBS job per T (`hpc/ehd_tc_one.pbs`) |

Each `tc_sweep_T<T>_mrt_<stamp>.csv` is the amplitude history for one T:
`step, max_abs_u, growth_rate_estimate`, sampled every 100 steps. The
`growth_rate_estimate` column is the *cumulative* estimate
log(|u|/|u|_0)/t, so it stays positive for any run whose amplitude ever grew
above its initial value — the sign of the instability is read from the
late-time slope of `max_abs_u`, not from that column. Each
`tc_sweep_summary_<stamp>.{csv,md}` is the one-line digest of its job.

## Retained runs

Main sweep, 2026-07-31 (stamps `20260731_20*`):

| T | file stamp | max u at 6e5 | late-time trend |
|---:|:--|---:|:--|
| 150 | `20260731_200046` | 9.23e-08 | decaying |
| 160 | `20260731_200031` | 2.20e-06 | decaying |
| 163.5 | `20260731_200031` | 6.72e-06 | decaying |
| 165 | `20260731_200904` | 1.07e-05 | decaying |
| 170 | `20260731_202205` | 4.98e-05 | growing |
| 190 | `20260731_202300` | 1.57e-02 | growing (nonlinear) |

Tc refinement, 2026-08-01/02 (stamps `20260801_23*`, `20260802_00*`), rerun
with per-run ms/step instrumentation:

| T | file stamp | max u at 6e5 | late-time trend |
|---:|:--|---:|:--|
| 165 | `20260802_000007` | 1.07e-05 | decaying |
| 166 | `20260801_235924` | 1.47e-05 | decaying |
| 167 | `20260802_000427` | 2.00e-05 | growing |
| 170 | `20260802_003905` | 4.98e-05 | growing |

The T=165 refinement run reproduces the main-sweep T=165 run bit for bit,
which is the cross-check that the two rounds are the same configuration.

Note: the T=160 and T=163.5 jobs started within the same second, so they
share a timestamp and only one summary file (`20260731_200031`, holding the
T=160 line) survived the collision; both amplitude histories are intact.

## Result

Over the last 1e5 cycles the perturbation amplitude decays for T <= 166 and
grows for T >= 167, so sigma changes sign in that interval:

**Tc ~ 166.5**, against the reference value **163.5** from Luo, Wu, Yi &
Tan, Phys. Rev. E **93**, 023309 (2016) — a **+1.8%** deviation.

Timing at this resolution is ~4.5 ms/step on H100 with the direct phi solve.
