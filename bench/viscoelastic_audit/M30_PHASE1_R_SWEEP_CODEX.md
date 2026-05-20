### Q1 - scalars
R=20: Cd_pressure=78.60, Cd_p=9.33, Cd_kraken=111.82, ladder rel err=0% (holds)
R=30: Cd_pressure=76.62, Cd_p=11.49, Cd_kraken=111.09, ladder rel err=0% (holds)
R=40: Cd_pressure=76.46, Cd_p=12.10, Cd_kraken=110.76, ladder rel err=0% (holds)

### Q2 - 5-band table
| theta band | rT dCd_p | K R=20 | K R=30 | K R=40 |
|---|---:|---:|---:|---:|
| Front pole +-pi  |   +33.22 |   +19.38 |   +19.58 |   +19.73 |
| Front shoulder   |   +89.52 |   +53.78 |   +53.95 |   +54.95 |
| Equator          |    +1.54 |    +3.11 |    +2.90 |    +2.91 |
| Rear shoulder    |   -52.75 |   -15.67 |   -18.22 |   -19.96 |
| Rear pole 0      |   -26.48 |    -4.70 |    -4.22 |    -3.81 |
| TOTAL Cd_press   |   +85.77 |   +78.60 |   +76.62 |   +76.46 |

### Q3 - K/rT ratios
- Front pole : R=20 0.583, R=30 0.589, R=40 0.594 -> trend : plateau
- Rear pole  : R=20 0.177, R=30 0.159, R=40 0.144 -> trend : regressing

### Q4 - Cd_pressure scalar
- gap rT-K : R=20 7.17, R=30 9.15, R=40 9.31
- Convergence rate / log-log slope : 0.39 for |gap| vs R
- Asymptote : 1/R fit gives Cd_pressure(R->infinity) = 73.97; still below rT by 11.80.

### Q5 - Verdict
- Codex pick : structural-BC
- The front-pole amplitude ratio is effectively R-independent and remains below the rheoTool reference.
