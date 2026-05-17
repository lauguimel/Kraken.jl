# BSD Analytical Ladder - 2026-05-17

Mission: M17-canary-analytical

Params: U_0 = 1.0, nu_p = 0.1, zeta = 0.75, N = (32, 64, 128).

Taylor-Green velocity: `Ux = U_0 sin(2*pi*x) cos(2*pi*y)`, `Uy = -U_0 cos(2*pi*x) sin(2*pi*y)` on cell centers of `[0, 1]^2`. The references are `lap(U) = -8*pi^2 U`, `tau_p = 2*nu_p*D`, `F_poly = nu_p*lap(U)`, and `F_total = (1 - zeta)*nu_p*lap(U)`.

```text
=========================================================
BSD analytical ladder - Taylor-Green vortex, CPU F64
nu_p = 0.1, zeta = 0.75, U_0 = 1.0
=========================================================

L0 - div(tau_p) vs nu_p * lap(U) (periodic, no walls)
  N=32   rel_L2 = 6.413e-03
  N=64   rel_L2 = 1.606e-03
  N=128  rel_L2 = 4.015e-04    (order ~= 2.00)

L1 - F_total = F_poly - F_BSD vs (1 - zeta) * nu_p * lap(U) (periodic)
  kind=:fd     N=32 rel_L2 = 1.603e-02
               N=64 rel_L2 = 4.014e-03  (order ~= 2.00)
               N=128 rel_L2 = 1.004e-03
  kind=:fd_v2  N=32 rel_L2 = 1.270e-02
               N=64 rel_L2 = 3.203e-03  (order ~= 2.00)
               N=128 rel_L2 = 8.026e-04
  Delta(:fd vs :fd_v2)  N=64 rel_L2 = 7.217e-03

L2 - F_total at walls (closed box, lid profile matches TG)
  kind=:fd     interior rel_L2 = 4.01e-03 | wall rel_L2 = 1.03e-01 |
               max|F_BSD| ratio wall/interior = 9.95e-01
  kind=:fd_v2  interior rel_L2 = 3.20e-03 | wall rel_L2 = 3.03e+02 |
               max|F_BSD| ratio wall/interior = 1.58e+02

=========================================================
VERDICT: :fd_v2 gives the tighter periodic cancellation, while :fd has the lower wall-band BSD spike; the M17 implication is to favor Option A D_uncorrected wall handling unless a separate kinetic default is validated.
=========================================================
```

Interpretation: At N=64, the periodic ladder identifies which BSD cancellation strategy matches the analytical Newtonian-limit force, while the closed-box wall band quantifies the penalty from applying cavity wall gradients to the BSD stress. The wall/interior BSD maxima are 9.95e-01 for :fd and 1.58e+02 for :fd_v2, so the M17 architecture should keep the BSD stabilizer on an uncorrected D path at walls unless a separately validated kinetic default replaces it.
