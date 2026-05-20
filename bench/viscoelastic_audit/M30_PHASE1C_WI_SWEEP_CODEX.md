# M30 Phase 1c Wi Sweep Independent Verdict

Date: 2026-05-20

## Method

Harness: `bench/scratch/m30_phase1c_codex/run_p_vs_Wi.jl`

Outputs:

- `bench/scratch/m30_phase1c_codex/M30P1c_codex_scalars.csv`
- `bench/scratch/m30_phase1c_codex/M30P1c_codex_bands.csv`
- `bench/scratch/m30_phase1c_codex/M30P1c_codex_stdout.log`

Frame and binning:

- Correct frame is `:idx`: `cx_lu = cx_phys + 1`, so `dx = i - cx_lu = (i - 1) - cx_phys`.
- Wall ring is every fluid cell with at least one solid 8-connectivity neighbour.
- Pressure traction is averaged per azimuthal bin, multiplied by `arc_dl = R * dtheta`, and normalised by `u_mean^2 * R`.
- Bin centres are `theta_k = -pi + (k - 0.5) * dtheta`, `N_az = 72`.
- The 5 bands use the locked half-width 22.5 deg convention around front pole, front shoulder, equator, rear shoulder, and rear pole.

## Q1: Cd_pressure Scalars and Stored Cd Reconciliation

| Wi | Kraken Cd_pressure ring `:idx` | rheoTool Cd_pressure | rT - K | stored Cd_kraken | Cd_s + Cd_p - Cd_bsd | diff | closes <= 0.01 |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.1 | 88.785155 | 90.516136 | 1.730981 | 129.477995 | 129.477995 | 0.000000 | yes |
| 0.5 | 78.171951 | 81.714092 | 3.542141 | 115.885743 | 115.885743 | 0.000000 | yes |
| 1.0 | 76.622000 | 85.771570 | 9.149570 | 111.091012 | 111.091012 | 0.000000 | yes |

The stored scalar identity `Cd_kraken = Cd_s + Cd_p - Cd_bsd` closes within 0.01 for all three snapshots.

## Q2: 5-Band Cd_pressure Decomposition

| band | K Wi=0.1 | rT Wi=0.1 | K Wi=0.5 | rT Wi=0.5 | K Wi=1.0 | rT Wi=1.0 |
|---|---:|---:|---:|---:|---:|---:|
| Front pole | 20.449315 | 32.564000 | 19.324740 | 31.842590 | 19.584290 | 33.223140 |
| Front shoulder | 57.053701 | 86.220940 | 54.210701 | 85.340720 | 53.949057 | 89.521640 |
| Equator | 3.448585 | 1.865116 | 2.982326 | 1.516052 | 2.900284 | 1.540810 |
| Rear shoulder | -11.790429 | -43.373720 | -15.909828 | -48.365960 | -18.215936 | -52.750860 |
| Rear pole | -5.116505 | -27.876400 | -5.246513 | -27.980600 | -4.220261 | -26.484400 |
| TOTAL Cd_p | 88.785155 | 90.516136 | 78.171951 | 81.714092 | 76.622000 | 85.771570 |

## Q3: K/rT Amplitude Ratios

| band | K/rT @ Wi=0.1 | K/rT @ Wi=0.5 | K/rT @ Wi=1.0 | Delta_max |
|---|---:|---:|---:|---:|
| Front pole | 0.627973 | 0.606883 | 0.589477 | 0.038496 |
| Front shoulder | 0.661715 | 0.635227 | 0.602637 | 0.059078 |
| Equator | 1.848992 | 1.967166 | 1.882311 | 0.118174 |
| Rear shoulder | 0.271833 | 0.328947 | 0.345320 | 0.073487 |
| Rear pole | 0.183543 | 0.187505 | 0.159349 | 0.028156 |
| TOTAL Cd_p | 0.980877 | 0.956652 | 0.893326 | 0.087550 |

## Q4: H1 Verdict

Verdict: **H1 pure-BC confirmed** under the specified pole criterion.

The pole ratios are invariant across Wi within the required 0.05 envelope:

- Front pole Delta_max = 0.038496
- Rear pole Delta_max = 0.028156

The total Cd_pressure ratio is not invariant by the same threshold:

- TOTAL Cd_p Delta_max = 0.087550

So the falsifiable pole test supports a boundary-condition origin at the poles: Phase 2b interpBB is necessary and sufficient for the pole amplitude defect under this criterion. The shoulder/equator/total rows still move with Wi, so this verdict should not be read as global Wi-invariance of the entire pressure decomposition.
