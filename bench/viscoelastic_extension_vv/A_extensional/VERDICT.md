# M44-VV-A VERDICT

Setup: periodic [0, 2pi]^2 4-roll mill, log-FV Oldroyd-B polymer chain, no walls, no solid mask, no BSD/cut-cell path.
Matrix: Wi={0.1,0.3,0.5,1.0}, N={32,64,96,128}, beta={0.59,0.80,0.90}; 48 cases.
Note: for the specified velocity formula, the listed pi/2 points are elliptic; the analytic gate samples the actual x-extensional stagnation point at (0,0).

## 48-case matrix
| Wi | beta | PASS | FAIL | NaN | convergence p |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.59 | 0 | 4 | 0 | -0.447 |
| 0.10 | 0.80 | 0 | 4 | 0 | -1.468 |
| 0.10 | 0.90 | 0 | 4 | 0 | -2.280 |
| 0.30 | 0.59 | 0 | 4 | 0 | 0.229 |
| 0.30 | 0.80 | 0 | 4 | 0 | 0.412 |
| 0.30 | 0.90 | 0 | 4 | 0 | 0.465 |
| 0.50 | 0.59 | 0 | 4 | 0 | -0.760 |
| 0.50 | 0.80 | 0 | 4 | 0 | -0.280 |
| 0.50 | 0.90 | 0 | 4 | 0 | 0.087 |
| 1.00 | 0.59 | 0 | 4 | 0 | 1.030 |
| 1.00 | 0.80 | 0 | 4 | 0 | 0.862 |
| 1.00 | 0.90 | 0 | 4 | 0 | 0.546 |

## Plots
- Convergence grid: `../plots/M44_VV_A_convergence_grid.png`
- Profile grid: `../plots/M44_VV_A_profiles_beta0p59_n128.png`

## Verdict synthesis
- Overall: 0/48 PASS, 48 FAIL, 0 NaN.
- Wi 0.10 convergence p range: -2.280 to -0.447.
- Wi 0.30 convergence p range: 0.229 to 0.465.
- Wi 0.50 convergence p range: -0.760 to 0.087.
- Wi 1.00 convergence p range: 0.546 to 1.030.
- Wi-dependent signature: low-Wi non-PASS count 24; Wi>=0.5 non-PASS count 24.
- beta-dependent signature: inspect PASS fractions above against (1-beta); stronger low-beta failures indicate polymer-coupling scaling.
- De Gennes note: Wi=1 is above the 0.5 coil-stretch threshold; stagnation growth is treated as a risk signature, not an automatic NaN failure.
- Recommendation to Boss: chain is RED in extension -> fundamental polymer-pressure coupling/advection issue should be isolated before Tests B + C.
