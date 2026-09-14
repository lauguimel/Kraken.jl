# Metal results — 2D Oldroyd-B confined cylinder

Date: 2026-04-27.

## References

- rheoTool Newtonian, `Re_R = 1`: `Cd ≈ 132.36`.
- rheoTool Oldroyd-BLog, `Re_R = 1`, `Wi_R = 0.1`, `beta = 0.59`: `Cd ≈ 130.43`.
- Liu-style R=20 Oldroyd-B target used in the existing scripts: `Cd ≈ 129.42`.

Kraken reports `Re = u_ref*D/nu_total = 2` for these runs. This is the same
case as `Re_R = u_ref*R/nu_total = 1`; it is only a diameter-vs-radius
convention difference.

## Kraken Metal setup

- Backend: `MetalBackend()`, `Float32`.
- Geometry: `Nx = 30R`, `Ny = 4R`, `cx = 15R`, `cy = 2R`.
- Flow: `u_mean = 0.02`, parabolic inlet.
- Rheology: `Wi_R = 0.1`, `beta = 0.59`, `CNEBB`.
- Drag measurement: LI-BB/cut-link Mei MEA on `f_out` after the Hermite source.

## Results after drag fix at `u_mean = 0.02`

| Case | R | steps | window | stride | `Cd` | `Cd_p` | samples |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oldroyd-B direct | 10 | 10,000 | 1,000 | 20 | 128.257316007 | 4.584863318 | 51 |
| Oldroyd-B direct | 20 | 10,000 | 1,000 | 20 | 128.759416792 | 3.904343735 | 51 |
| Oldroyd-B direct | 20 | 40,000 | 5,000 | 50 | 128.770168025 | 3.904599816 | 101 |
| Oldroyd-B log-conf | 20 | 10,000 | 1,000 | 20 | 127.786859155 | 3.836146980 | 51 |
| Newtonian LI-BB | 20 | 10,000 | 1,000 | 20 | 141.884803979 | — | 51 |

## Geometry checks

| Check | R | Geometry | `Cd` |
|---|---:|---|---:|
| Newtonian height-exact | 20 | `Nx=600`, `Ny=81`, `cy=41` | 135.671432918 |
| Oldroyd-B height-exact | 20 | `Nx=600`, `Ny=81`, `cy=41` | 124.018613315 |
| Newtonian long-domain | 20 | `Nx=1600`, `Ny=80`, `cx=400` | 146.771951474 |
| Newtonian phase check | 20 | `Cd(f_in)` vs `Cd(f_out)` | 141.884799012 vs 141.884803979 |

The long-domain run is not converged at 10,000 steps because the flow has not
convected through the full outlet length.

## Interpretation

The previous very low viscoelastic drag came from using the boolean-mask
staircase MEA in the viscoelastic LI-BB driver. After switching to the same
cut-link Mei MEA as the Newtonian LI-BB driver, the R=20 direct Oldroyd-B result
is stable at `Cd ≈ 128.77` at `u_mean = 0.02`.

This is close to the existing Liu-style R=20 target (`129.42`, error about
`-0.5%`) and within about `-1.3%` of the finite-inertia rheoTool Oldroyd-B
reference (`130.43`). The direct viscoelastic path is therefore no longer
blocked by the drag accounting bug.

The apparent Newtonian LI-BB failure at `u_mean = 0.02` is mostly a
compressibility/Mach issue. The density field is not close enough to the
incompressible limit (`rho_mean ≈ 1.043`, `rho_max ≈ 1.086`), and the measured
drag is correspondingly too high (`Cd ≈ 141.88`).

## Mach sweep at fixed `Re_R = 1`

All rows keep `Re_R = u_mean R/nu = 1`; only the lattice velocity is changed.

| Case | `u_mean` | steps | `Cd` | `rho_mean` | `rho_max` |
|---|---:|---:|---:|---:|---:|
| Newtonian LI-BB | 0.020 | 10,000 | 141.866360723 | 1.042998 | 1.086029 |
| Newtonian LI-BB | 0.010 | 20,000 | 133.223627824 | 1.010084 | 1.020175 |
| Newtonian LI-BB | 0.005 | 40,000 | 131.337508295 | 1.002484 | 1.007456 |
| Newtonian LI-BB | 0.005 | 120,000 | 131.336719948 | 1.002484 | 1.007455 |
| Oldroyd-B direct | 0.010 | 20,000 | 126.477694497 | 1.011871 | 1.023382 |
| Oldroyd-B direct | 0.005 | 40,000 | 131.269639054 | 1.003407 | 1.007084 |
| Oldroyd-B log-conf | 0.005 | 40,000 | 129.997222525 | 1.003504 | 1.006802 |

The Newtonian baseline is therefore not fundamentally broken. It becomes
quantitative once the validation is run at lower Mach. For the R=20 grid,
`u_mean = 0.005` gives `Cd_Newt ≈ 131.34`, about `-0.8%` from the rheoTool
Newtonian reference `132.36`.

The Oldroyd-B low-Mach results are in the right validation band:
direct-C gives `Cd ≈ 131.27`, while log-conformation gives `Cd ≈ 130.00`.
The log-conformation value is `+0.45%` from the Liu R=20 target `129.42` and
`-0.33%` from the finite-inertia rheoTool Oldroyd-B reference `130.43`. A
longer low-Mach viscoelastic run and an R-convergence should be used for the
final claim.
