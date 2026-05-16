# BSD/Guo Wi=0 force audit (2026-05-16)

Scope: audit the body force applied to the LBM by the coupled cavity
driver in the `Wi -> 0` limit.  This document is analysis only.

## Empirical anchor

M7b isolates a Wi-independent polymer-path residual.  At
`lambda_phys=0.001`, `N=64`, `t=8`, `u_max=0.005`, and
`bsd_fraction=0.75`, the matched-viscosity comparison was:

- A: `nu_s=0.1`, `nu_p=0.1`, `nu_total=0.2`, `Re_LU=1.6`;
- B: `nu_s=0.2`, `nu_p=0.0`, `nu_total=0.2`, `Re_LU=1.6`;
- C: `nu_s=0.1`, `nu_p=0.0`, `nu_total=0.1`, `Re_LU=3.2`.

The A-vs-B centerline `u` relative L2 is `3.42 %`, while the Newtonian
B-vs-C control is `0.014 %`
(`bench/viscoelastic_logfv/CAVITY_LOWWI_M7B_VERDICT_20260516.md:5-13`,
`:20-37`).  The A-vs-B max absolute difference is `4.22e-2`
(`CAVITY_LOWWI_M7B_VERDICT_20260516.md:39-41`).

A and B have identical `nu_total` and identical `Re_LU`; the only
meaningful difference is the polymer code path.  M7b therefore proves
that a term surviving as `Wi -> 0` perturbs the LBM momentum equation
(`CAVITY_LOWWI_M7B_VERDICT_20260516.md:51-63`).  This is the first
concrete localization of the cavity-gap bug since M1.

## Design-intent body force at Wi

All quantities here are in lattice units: `dx = dy = 1`, `dt = 1`, and
`cs^2 = 1/3`.

In the Oldroyd-B steady-shear limit,

```text
tau_p = 2 * nu_p * D + O(Wi),
D_ab = (partial_a u_b + partial_b u_a) / 2.
```

For incompressible flow,

```text
div(tau_p)_a
  = 2 * nu_p * partial_b D_ab
  = nu_p * nabla^2 u_a + nu_p * partial_a(partial_b u_b)
  = nu_p * nabla^2 u_a.
```

BSD is intended to subtract a fraction `zeta` of this Newtonian
polymer contribution from the explicit Guo force and move that fraction
into the LBM relaxation viscosity:

```text
nu_LBM = nu_s + zeta * nu_p

implicit lattice:
  (nu_s + zeta * nu_p) * nabla^2 u

explicit body force:
  F_poly - zeta * nu_p * nabla^2 u
  = (1 - zeta) * nu_p * nabla^2 u

sum:
  (nu_s + nu_p) * nabla^2 u.
```

Thus case A should collapse to case B as `Wi -> 0`, up to ordinary
discretization and boundary noise.

## Implementation walk-through

The cavity harness passes `nu_s`, `nu_p`, `u_max`, `lambda_phys`, and
`bsd_fraction` into `run_viscoelastic_logfv_cavity_coupled_2d`
(`bench/viscoelastic_logfv/run_cavity_oldroydb_vs_rheotool.jl:225-266`).
The driver defaults to `bsd_kind = :fd` and
`polymer_wall_extrap = :quadratic`
(`src/drivers/viscoelastic_logfv_2d.jl:860-883`).  It sets
`prefactor_t = nu_p_t / lambda_lu`, `dx = dy = 1`, and
`nu_lbm_t = nu_s_t + bsd_t * nu_p_t`
(`src/drivers/viscoelastic_logfv_2d.jl:898-912`).

First, `logfv_stress_from_log_2d!` reconstructs stress from `Psi`.
The inline formula is
`tau_xx = prefactor * (f * C_xx - 1)`,
`tau_xy = prefactor * f * C_xy`,
`tau_yy = prefactor * (f * C_yy - 1)` after `C = exp(Psi)`, so for
Oldroyd-B it is `tau_p = (nu_p / lambda) * (exp(Psi) - I)`
(`src/kernels/logconformation_fv_2d.jl:296-303`, `:459-492`).  The
cavity loop calls it at step 5
(`src/drivers/viscoelastic_logfv_2d.jl:1067-1071`).

Second, `logfv_polymer_force_bc_aware_2d!` computes
`F_poly = div(tau_p)` at step 6
(`src/drivers/viscoelastic_logfv_2d.jl:1073-1077`).  The wrapper
delegates to `fvfd_tensor_divergence_2d!`
(`src/kernels/logconformation_fv_2d.jl:649-658`), whose kernel computes
`F_poly_x = d_x tau_xx + d_y tau_xy` and
`F_poly_y = d_x tau_xy + d_y tau_yy`
(`src/fvfd/operators_2d.jl:633-659`, `:772-790`).  Interior cells use:

```text
F_poly_x = (tau_xx[i+1,j] - tau_xx[i-1,j]) / (2 dx)
         + (tau_xy[i,j+1] - tau_xy[i,j-1]) / (2 dy)
F_poly_y = (tau_xy[i+1,j] - tau_xy[i-1,j]) / (2 dx)
         + (tau_yy[i,j+1] - tau_yy[i,j-1]) / (2 dy).
```

Wall rows use the solid-aware one-sided derivative helpers, with
quadratic 3-point extrapolation by default
(`src/fvfd/operators_2d.jl:13-75`).

Third, the default FD-BSD branch subtracts a narrow velocity laplacian.
Step 7 calls `logfv_bsd_correct_force_bc_aware_2d!`
(`src/drivers/viscoelastic_logfv_2d.jl:1079-1091`), which delegates to
`fvfd_bsd_force_2d!`
(`src/kernels/logconformation_fv_2d.jl:710-718`).  The kernel computes:

```text
F_total_x = F_poly_x - zeta * nu_p * Lap_5pt u_x
F_total_y = F_poly_y - zeta * nu_p * Lap_5pt u_y
```

with `_fvfd_solid_bc_second_derivative_{x,y}_2d`
(`src/fvfd/operators_2d.jl:77-125`, `:886-933`).  The M4 audit
reconstructs the same split as
`F_Guo = F_FD - zeta * nu_p * lap_u`
(`bench/viscoelastic_logfv/analyse_cavity_guo_vs_fd_2d.jl:203-222`).

Fourth, the fused TRT + LI-BB + Guo step consumes `fx_total, fy_total`
(`src/drivers/viscoelastic_logfv_2d.jl:1094-1098`).  The generated
kernel uses `CollideTRTDirectGuoField`
(`src/kernels/li_bb_2d_v2.jl:49-54`, `:115-127`).  The Guo brick reads
`Fx_field, Fy_field`, applies the local half-force velocity correction,
sets `guo_pref = 1 - s_plus / 2`, and multiplies all nine `S_q` terms
by that same prefactor (`src/kernels/dsl/bricks.jl:145-199`).  The
macroscopic update uses the matching `+ F/2` half-step
(`src/kernels/logconformation_fv_2d.jl:1024-1062`).

Finally, `trt_rates` defines the lattice viscosity convention:

```text
s_plus = 1 / (3 * nu + 1/2)
nu = (1 / s_plus - 1/2) / 3.
```

(`src/kernels/fused_trt_2d.jl:99-129`).  Since the driver passes
`nu_lbm_t = nu_s_t + bsd_t * nu_p_t`, the assembled `Wi -> 0`
contribution is:

```text
(nu_s + zeta * nu_p) * Lap_lattice u
  + nu_p * Lap_wide u
  - zeta * nu_p * Lap_5pt u.
```

Here `Lap_lattice` is the implicit TRT viscous operator, `Lap_wide` is
the operator generated by centered FD velocity gradients followed by
centered FD tensor divergence, and `Lap_5pt` is the narrow FD-BSD
laplacian.

## Algebraic discrepancy at Wi

The continuum identity `div(2 * nu_p * D) = nu_p * nabla^2 u` is exact,
but the discrete operators are not the same.  The velocity-gradient
kernel computes centered derivatives at cell centers
(`src/fvfd/operators_2d.jl:947-977`):

```text
dudx(i,j) = (u[i+1,j] - u[i-1,j]) / (2 dx)
dudy(i,j) = (u[i,j+1] - u[i,j-1]) / (2 dy)
dvdx(i,j) = (v[i+1,j] - v[i-1,j]) / (2 dx)
dvdy(i,j) = (v[i,j+1] - v[i,j-1]) / (2 dy).
```

At `Wi -> 0`,

```text
tau_xx(i,j) ~= 2 * nu_p * dudx(i,j)
             = nu_p * (u[i+1,j] - u[i-1,j]) / dx
tau_xy(i,j) ~= nu_p * (dudy(i,j) + dvdx(i,j))
tau_yy(i,j) ~= 2 * nu_p * dvdy(i,j).
```

Applying the polymer divergence to `tau_xx` gives:

```text
d_x tau_xx(i,j)
  = (tau_xx[i+1,j] - tau_xx[i-1,j]) / (2 dx)
  = nu_p * (u[i+2,j] - 2 u[i,j] + u[i-2,j]) / (2 dx^2).
```

This is a wide 3-point stencil with spacing `2 dx`, not the narrow
second difference
`(u[i+1,j] - 2 u[i,j] + u[i-1,j]) / dx^2`.  Its leading truncation
error is `dx^2 / 3 * d4u/dx4`; the narrow stencil error is
`dx^2 / 12 * d4u/dx4`.  The wide stencil therefore has four times the
leading fourth-derivative error.

FD-BSD subtracts the narrow operator:

```text
F_total_x = nu_p * (Lap_wide u + cross_xy)_x
          - zeta * nu_p * Lap_narrow u_x.
```

At Chapman-Enskog order, the implicit TRT viscous operator is the
standard lattice/narrow operator associated with `s_plus`.  M5-A gives
the matching kinetic-moment identity and the same Guo half-step
prefactor (`bench/viscoelastic_audit/BSD_KINETIC_MOMENT_DESIGN_20260516.md:45-110`).
Thus:

```text
nu_LBM * Lap_narrow u + nu_p * Lap_wide u
  - zeta * nu_p * Lap_narrow u

= nu_s * Lap_narrow u + nu_p * Lap_wide u.
```

The polymer Newtonian contribution is applied through the wide stencil,
not through the same narrow operator used by the LBM viscosity.  The
intended cancellation is broken at `O(nu_p * dx^2)`.

The leading residual includes:

```text
R_x = nu_p * (Lap_wide - Lap_narrow) u_x
    + nu_p * (cross_xy_wide - cross_xy_lattice)
    ~= nu_p * dx^2 * d4u_x/dx4 / 4 + mixed terms.
```

For `u_max = 0.005` and `N = 64`, the lid-scale estimate
`u_max * 64^4 ~= 8.4e4` makes the naive local residual
`0.1 * 8.4e4 / 4 ~= 2e3`.  This number is not a physical force
prediction because bulk cancellation and limited wall support are
ignored; it only shows why a local stencil inconsistency can integrate
into the observed few-percent centerline signal.

The cross terms are also not cancelled by the FD-BSD laplacian.  For
example,
`d_y tau_xy = nu_p * d_yy u_x + nu_p * d_xy u_y`, and the discrete
mixed derivative uses

```text
(v[i+1,j+1] - v[i-1,j+1] - v[i+1,j-1] + v[i-1,j-1]) / (4 dx dy),
```

which is not the narrow lattice laplacian stencil.  Continuum
incompressibility removes `grad(div u)` analytically, but the discrete
gradient-divergence path is still a different operator.

Common scalar explanations are ruled out:

- No sign error: `F_total = F_poly - zeta * nu_p * Lap u`
  (`src/fvfd/operators_2d.jl:900-911`).
- No double `guo_pref`: the Guo brick applies it once to all nine
  source terms (`src/kernels/dsl/bricks.jl:170-198`).
- No missing explicit `1 - zeta`: it should emerge from cancellation.
- No unit mismatch: the driver uses `dx = dy = 1`
  (`src/drivers/viscoelastic_logfv_2d.jl:898-912`).
- No staggering: `F_poly` and FD-BSD are evaluated at the same
  cell-centered `(i,j)` before Guo forcing
  (`src/drivers/viscoelastic_logfv_2d.jl:1041-1098`).

Conclusion: the bug is a stencil mismatch between the wide/mixed
operator implicit in `div(tau_p)` at `Wi -> 0` and the narrow 5-point
laplacian used by FD-BSD.  It is not a sign, unit, or Guo-prefactor
error.

## Proposed fix

Option A, recommended primary: replace the narrow-stencil FD-BSD
laplacian by the same tensor-divergence path used by `F_poly`.

The existing `logfv_bsd_stress_from_gradient_2d!` builds
`tau_BSD = 2 * zeta * nu_p * D`
(`src/kernels/logconformation_fv_2d.jl:678-708`).  Dividing that tensor
with the same `fvfd_tensor_divergence_2d!` operator used by `F_poly`
makes the zeta-fraction subtraction use the same wide/mixed stencil as
the Newtonian part of the polymer force.

Target `src/drivers/viscoelastic_logfv_2d.jl:1088-1091`:

```julia
logfv_bsd_stress_from_gradient_2d!(
    tau_bsd_xx, tau_bsd_xy, tau_bsd_yy,
    dudx, dudy, dvdx, dvdy, bsd_t * nu_p_t; sync=false,
)
fvfd_tensor_divergence_2d!(
    fx_bsd, fy_bsd, tau_bsd_xx, tau_bsd_xy, tau_bsd_yy,
    is_solid, dx, dy, logfv_bc;
    polymer_wall_extrap=polymer_wall_extrap, sync=false,
)
# fx_total = fx_poly - fx_bsd; fy_total = fy_poly - fy_bsd
```

This needs persistent `tau_bsd_xx`, `tau_bsd_xy`, `tau_bsd_yy`,
`fx_bsd`, and `fy_bsd` buffers beside the existing hot-loop arrays
(`src/drivers/viscoelastic_logfv_2d.jl:956-970`).  It is the smallest
FD-architecture fix: the zeta piece cancels against a zeta-scaled copy
of the same tensor-divergence operator that generated `F_poly`.

Option B: switch the cavity default from `bsd_kind = :fd` to
`bsd_kind = :kinetic`.  The kinetic path extracts `Pi_neq` from the
D2Q9 populations and builds the BSD term from the lattice's own viscous
moment.  M5-A documents the theory
(`BSD_KINETIC_MOMENT_DESIGN_20260516.md:45-110`, `:126-182`), and the
current implementation reconstructs `nu_eff` from `s_plus`, applies
the Guo half-step factor once, and assembles the force
(`src/kernels/bsd_kinetic.jl:1-120`).

Option B is formally clean but has wall-adjacent LI-BB risk: M5-A notes
that pre-phase cells may differ unless extraction is fused into the
same generated kernel (`BSD_KINETIC_MOMENT_DESIGN_20260516.md:112-124`,
`:180-182`).  Use Option A first; use Option B if the post-fix M7b
residual remains above threshold.

## Predicted impact

For Option A, M7b A-vs-B centerline `u` relative L2 should drop from
`3.42 %` toward the `0.014 %` Newtonian noise floor.  Pass if A-vs-B is
below `0.1 %`.  The max absolute difference should drop from `4.22e-2`
toward `1e-3`.

The `Wi = 1` cavity production gap should close partially.  A current
representative gap is about `17.97 %` centerline `u` L2 and `24.41 %`
`psi_xy` L2 (`bench/viscoelastic_logfv/CAVITY_M6B_CONFIRM_VERDICT_20260516.md:5-14`,
`:31-38`).  Removing a `3.4 %` Wi-independent floor should move the
centerline `u` gap to roughly `14-18 %`.  Remaining error then belongs
to finite-Wi polymer-stress discretization and spatial-discretization
floor, not the Newtonian BSD/Guo split alone.

Option B should have the same low-Wi direction through a lattice-moment
cancellation mechanism.

## Validation plan

1. Re-run `bench/viscoelastic_logfv/run_cavity_lowwi_matched_visc.pbs`.
   Pass: A-vs-B centerline `u` relative L2 below `0.1 %`.

2. Re-run `bench/viscoelastic_logfv/analyse_cavity_guo_vs_fd_2d.jl` on
   a fresh post-fix N=64 snapshot.  The old M4 split was `53.5--53.8 %`
   force L2 with max cell `(16, 63)`
   (`BSD_KINETIC_MOMENT_DESIGN_20260516.md:5-19`).  Expected: the
   same-stencil comparison drops sharply.

3. Re-run the canonical non-cavity battery: channel Poiseuille,
   cylinder `Cd`, and contraction 4:1.  The cylinder `Cd` at `Re=1`,
   `Wi=0.1` must stay inside the current ratchet bands in
   `bench/viscoelastic_logfv/VALIDATION_LADDER_AUDIT_20260513.md`.

4. Re-run cavity production with
   `bench/viscoelastic_logfv/run_cavity_oldroydb_vs_rheotool_anygpu.pbs`
   at `u_max=0.005`, `lambda_phys=1.0`.  Expected: partial closure, not
   full agreement.

If Option A does not bring M7b below `0.1 %`, promote Option B and
separate wall-adjacent kinetic-vs-FD force norms from interior norms.

## Risks

- The remaining explicit `(1 - zeta)` polymer body force still travels
  through the polymer tensor-divergence path.  Mixed derivative terms
  such as `d_y(d_x v)` are not identical to the LBM narrow operator.
  A visible residual of roughly `0.05-0.2 %` at `N=64` may remain.

- Option A changes finite-Wi behavior because BSD becomes a divergence
  of rate-of-strain rather than a direct velocity laplacian.  Small
  benchmark shifts should be classified by the canonical ladder before
  being called regressions.

- Option B is smooth-interior validated but wall-adjacent LI-BB behavior
  remains unverified unless moment extraction is fused
  (`BSD_KINETIC_MOMENT_DESIGN_20260516.md:112-124`, `:180-182`).

- Option A adds five persistent `N x N` work arrays if implemented
  literally.  At `N=64--256`, this is acceptable if allocated once
  outside the time loop.

- If M7b remains above `0.1 %`, the next suspect is boundary or
  mixed-derivative discretization, not a Guo-prefactor or sign issue.
