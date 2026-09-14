# Results — finite-inertia rheoTool cylinder

Date: 2026-04-27.

## Case

- Solver: `rheoFoam`.
- Model: `Oldroyd-BLog`.
- Geometry: cylinder radius `R = 1`, channel height `H = 4`, blockage `D/H = 0.5`.
- Inlet: parabolic, `U_mean = 1`.
- Properties: `rho = 1`, `etaS = 0.59`, `etaP = 0.41`, `lambda = 0.1`.
- Dimensionless numbers: `Re_R = 1`, `Re_D = 2`, `Wi_R = 0.1`, `De_D = 0.05`.

## Run Notes

The stock `rheoFoam` source includes the inertial terms `fvm::ddt(U)` and
`fvm::div(phi, U)`, but the original cylinder tutorial disables the convective
term with `div(phi,U) GaussDefCmpw none`. In rheoTool's `GaussDefCmpw`
implementation, `none` explicitly means no convection. The finite-inertia run
therefore uses:

- `div(phi,U) GaussDefCmpw cubista`
- `updateMatrixCoeffs true`
- `updatePrecondFrequency 1`

The local Docker image also needed explicit environment setup in
`run_docker.sh`:

- source `/opt/openfoam9/etc/bashrc`
- add `/home/openfoam/platforms/linux64GccDPInt32Opt/bin` to `PATH`
- add rheoTool and PETSc library paths to `LD_LIBRARY_PATH`

## Drag

`Cd.txt` with `v·grad(v)` active:

| Quantity | Value |
|---|---:|
| `Cd_viscous(t >= 0.6)` | `130.419752802` |
| `Cd_viscous(t >= 0.8)` | `130.428774404` |
| `Cd_viscous(last)` | `130.429053837` |

For this case, `Re_R = 1`, so the rheoTool viscous normalization
`F_x/(eta0*U_mean)` is numerically equal to Kraken's inertial normalization
`F_x/(0.5*rho*U_mean^2*D)`.

## Interpretation

This gives a clean finite-inertia target:

- rheoTool finite-Re reference: `Cd ≈ 130.43`
- Liu-style LBM target at `Re_R = 1`, `Wi_R = 0.1`, `beta = 0.59`: `Cd ≈ 130.36`
- difference using last value: `+0.05%`

The earlier `Cd ≈ 130.17` value came from the inherited `GaussDefCmpw none`
setting and is a convection-free reference, not the finite-inertia result. The
correct finite-inertia result supports using `Cd ≈ 130.3–130.5` as the
practical validation band for
Kraken's 2D Oldroyd-B cylinder at `Re_R = 1`, `Wi_R = 0.1`, `beta = 0.59`,
before doing lower-Re extrapolation.
