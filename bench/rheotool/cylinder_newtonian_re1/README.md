# rheoTool cylinder benchmark — Newtonian limit, finite inertia

This case is a Newtonian-limit copy of the corrected finite-inertia
Oldroyd-BLog cylinder case. It keeps `type Oldroyd-BLog` with `etaP = 0` so the
same fields and drag function object can be reused, but the momentum equation
is Newtonian because the polymer stress is identically zero.

## Target parameters

- Geometry: 2D confined cylinder, `R = 1`, channel height `H = 4`, blockage `D/H = 0.5`.
- Inlet: fully developed parabolic profile with mean velocity `U_mean = 1`.
- Properties: `rho = 1`, `etaS = 1`, `etaP = 0`, `eta0 = 1`.
- Reynolds number: `Re_R = rho*U_mean*R/eta0 = 1`.
- Diameter Reynolds number: `Re_D = rho*U_mean*D/eta0 = 2`.
- Momentum convection: active through `div(phi,U) GaussDefCmpw cubista`.

## Run

```sh
./run_docker.sh
./summarize_cd.sh
```

The coded function object writes `Cd.txt` with columns:

1. time
2. `F_x/(eta0*U_mean)`

At `Re_R = 1`, this equals Kraken's inertial drag normalization
`F_x/(0.5*rho*U_mean^2*D)`.
