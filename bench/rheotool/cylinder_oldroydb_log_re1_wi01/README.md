# rheoTool cylinder benchmark — Oldroyd-BLog, finite inertia

This case is a Kraken-side copy of rheoTool's `rheoFoam/Cylinder/Oldroyd-BLog`
tutorial, adjusted to provide an independent finite-Reynolds reference for the
Kraken viscoelastic cylinder validation.

## Target parameters

- Geometry: 2D confined cylinder, `R = 1`, channel height `H = 4`, blockage `D/H = 0.5`.
- Inlet: fully developed parabolic profile with mean velocity `U_mean = 1`.
- Viscosity split: `etaS = 0.59`, `etaP = 0.41`, so `beta = 0.59` and `eta0 = 1`.
- Density: `rho = 1`.
- Relaxation time: `lambda = 0.1`.
- Reynolds number: `Re_R = rho*U_mean*R/eta0 = 1`.
- Diameter Reynolds number: `Re_D = rho*U_mean*D/eta0 = 2`.
- Weissenberg number: `Wi_R = lambda*U_mean/R = 0.1`.
- Deborah number in Kumar/Mohanty convention: `De_D = lambda*U_mean/D = 0.05`.

The stock `rheoFoam` solver contains the inertial terms `fvm::ddt(U)` and
`fvm::div(phi, U)` in `UEqn.H`/`pUEqn.H`. The original tutorial nevertheless
neutralises the convective term through `div(phi,U) GaussDefCmpw none`, where
`none` means "no convection" in rheoTool's `GaussDefCmpw` implementation. This
case reactivates `v·grad(v)` by setting `div(phi,U) GaussDefCmpw cubista`.

## Run

Interactive wrapper:

```sh
/Users/guillaume/Documents/Clouds/UGA/Recherche/QUT/Rheotool/openfoam9-rheotoolv12.sh -d bench/rheotool/cylinder_oldroydb_log_re1_wi01
./Allclean
./Allrun
```

Non-interactive helper:

```sh
./run_docker.sh
```

## Output

The coded function object writes `Cd.txt` with columns:

1. time
2. `F_x/(eta0*U_mean)`

Because this case uses `Re_R = 1`, this viscous resistance coefficient is
numerically equal to Kraken's inertial drag normalization
`F_x/(0.5*rho*U_mean^2*D)`. For a future Reynolds sweep, convert using
`Cd_inertial = Cd_viscous/Re_R`.

## First local result with active inertia

Run date: 2026-04-27.

The Docker helper was run until `t = 1`, then stopped after the drag had become
flat over the last output points. The measured values are:

- `Cd_viscous(t >= 0.6) = 130.419752802`
- `Cd_viscous(t >= 0.8) = 130.428774404`
- `Cd_viscous(last) = 130.429053837`

This is within `0.05%` of the Liu-style LBM target `Cd = 130.36` at
`Re_R = 1`, `Wi_R = 0.1`, `beta = 0.59`. The previous local value
`Cd = 130.167153558` was from the inherited tutorial setting
`div(phi,U) GaussDefCmpw none`; it should be treated as the convection-free
Stokes-like reference, not the finite-inertia target.
