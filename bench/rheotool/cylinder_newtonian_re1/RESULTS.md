# Results — finite-inertia rheoTool Newtonian cylinder

Date: 2026-04-27.

## Case

- Solver: `rheoFoam`.
- Model: Newtonian limit through `Oldroyd-BLog` with `etaS = 1`, `etaP = 0`.
- Geometry: cylinder radius `R = 1`, channel height `H = 4`, blockage `D/H = 0.5`.
- Inlet: parabolic, `U_mean = 1`.
- Dimensionless numbers: `Re_R = 1`, `Re_D = 2`.
- Momentum convection: active with `div(phi,U) GaussDefCmpw cubista`.

## Drag

`Cd.txt`:

| time | `Cd` |
|---:|---:|
| 0.2 | 132.460538216 |
| 0.4 | 132.298573986 |
| 0.6 | 132.339775957 |
| 0.8 | 132.358607028 |
| 1.0 | 132.365866001 |

| Quantity | Value |
|---|---:|
| `Cd(t >= 0.6)` | `132.354749662` |
| `Cd(t >= 0.8)` | `132.362236515` |
| `Cd(last)` | `132.365866001` |

For `Re_R = 1`, the rheoTool normalization `F_x/(eta0*U_mean)` is
numerically equal to Kraken's inertial drag normalization
`F_x/(0.5*rho*U_mean^2*D)`.

## Interpretation

This Newtonian finite-inertia target is `Cd ≈ 132.36`. It is the baseline
Kraken must match before using the Oldroyd-B result `Cd ≈ 130.43` as a
viscoelastic validation target.
