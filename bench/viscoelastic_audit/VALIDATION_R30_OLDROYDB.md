# Validation R30 — Oldroyd-B cylinder

Date: 2026-04-29.

> Superseded on 2026-05-01. The numbers below are still a valid numerical
> record for the single case `R=30`, `Wi_R=0.1`, but the subsequent Wi sweep
> shows this is a point crossing, not a solver-level validation. See
> `AUDIT_WI_SWEEP_20260430.md` and `BUGS_20260501.md`.

## Historical single-point record

Kraken was numerically close for the finite-inertia 2D Oldroyd-B cylinder at
`R=30`, within the following scope:

- Geometry: confined cylinder, blockage `D/H = 0.5`.
- Flow: parabolic inlet, `Re_R = U_mean R / ν = 1`, therefore `Re_D = 2`.
- Viscoelastic parameters: `Wi_R = λ U_mean / R = 0.1`, `β = ηs/η0 = 0.59`.
- Kraken formulation: log-conformation Oldroyd-B, CNEBB polymer wall condition,
  LI-BB hydrodynamic wall, `τp,1 = 1` diffusive conformation setting.
- Force normalization: `Cd = Fx / (0.5 ρ U_mean^2 D)`, with `D = 2R`.

This records the practical Kraken viscoelastic cylinder calculation at one
operating point. It does not validate exact reproduction of Liu's
high-Schmidt setting `Sc = 1e4`,
`τp,1 ≈ 0.50003`, `Λp ≈ 2.5e-7`; that remains a separate scheme-reproduction
problem.

## Reference values

| Reference | Quantity | Value |
|---|---:|---:|
| rheoTool finite-inertia Newtonian | `Cd(t >= 0.8)` | `132.362236515` |
| rheoTool finite-inertia Oldroyd-BLog | `Cd_viscous(t >= 0.8)` | `130.428774404` |
| Liu/Yu Table 3, `R=30`, `Wi=0.1` | `Cd` | `130.36` |

The rheoTool cases use active inertia (`v·∇v`) and `Re_R = 1`. At this Reynolds
number rheoTool's viscous normalization is numerically equal to Kraken's
inertial drag normalization.

## Kraken result

Canonical run:
`tmp/force_matrix_cnebb_u_tau1_20260429_1130/results/force_scheme_matrix.txt`.

Configuration:

- Backend: CUDA A100, `Float64`.
- Grid: `Nx = 900`, `Ny = 120`, `R = 30`.
- Flow: `u_mean = 0.005`, `Re_R = 1`, `Re_D = 2`.
- Visco: `β = 0.59`, `Wi_R = 0.1`, log-conformation.
- Averaging: `61` drag samples.

| Case | Kraken `Cd` | Reference | Error | `Cl` |
|---|---:|---:|---:|---:|
| Newtonian, vs rheoTool | `132.076371` | `132.362236515` | `-0.216%` | `3.97e-13` |
| Oldroyd-B, vs Liu/Yu | `130.739168759` | `130.36` | `+0.291%` | `3.37e-13` |
| Oldroyd-B, vs rheoTool | `130.739168759` | `130.428774404` | `+0.238%` | `3.37e-13` |

## Numerical closeness rationale

- The Newtonian baseline is within `0.22%` of the finite-inertia rheoTool
  reference.
- The Oldroyd-B result is within `0.30%` of both independent references:
  Liu/Yu and rheoTool.
- The lift is numerically zero (`|Cl| < 4e-13`), so the centered geometry is not
  introducing a measurable asymmetry.
- rheoTool and Liu/Yu agree at the target level: `130.428774404` vs `130.36`,
  a `0.053%` difference.

Therefore the R30 Oldroyd-B cylinder point is recorded as close for the
diffusive conformation setting. It is not a solver-level validation after the
2026-04-30 Wi sweep.

## Source files

- Kraken result: `tmp/force_matrix_cnebb_u_tau1_20260429_1130/results/force_scheme_matrix.txt`.
- rheoTool Newtonian result: `bench/rheotool/cylinder_newtonian_re1/RESULTS.md`.
- rheoTool Oldroyd-B result: `bench/rheotool/cylinder_oldroydb_log_re1_wi01/RESULTS.md`.
- Liu/Yu reference values: `REFERENCES.md`.
