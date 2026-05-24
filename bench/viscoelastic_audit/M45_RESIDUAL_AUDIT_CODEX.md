# M45 residual mesh-Wi audit - Codex

Date: 2026-05-25
Branch: `dev-viscoelastic`
Scope: audit only. No `src/` edits, no jobs, no commits.

Evidence anchor: `tmp/m44_postfix_sweep/21827394.aqua/SUMMARY.csv` has
Wi=1 beta=0.59 falling from Cd=118.101997 at R=30 to 113.234088 at R=60,
with `Cd_s` carrying the drop and `Cd_p` flat (`SUMMARY.csv:14-17`).
The Wi=0.1 beta=0.59 rows are quasi-flat (`SUMMARY.csv:2-5`).

## (α) rT reference R-coverage

NO additional R=40/50/60 rheoTool reference was found for the requested
Wi=1, beta=0.59 case.

The only `bench/rheotool/` directories matching `etaS=0.59`,
`etaP=0.41`, `lambda=1.0` are:

| path | role | Cd evidence |
|---|---:|---:|
| `bench/rheotool/cylinder_wi1.0_shrunk15R` | shrunk 15R/15R rT reference | `Cd.txt:201` gives `120.382717983` at t=20 |
| `bench/rheotool/cylinder_wi1.0` | larger-domain variant, not an R refinement | `Cd.txt:101` gives `120.400592705` at t=10 |

The shrunk case is the correct rT comparator: `constant/constitutiveProperties`
sets `etaS=0.59`, `etaP=0.41`, `lambda=1.0`
(`bench/rheotool/cylinder_wi1.0_shrunk15R/constant/constitutiveProperties:22-24`).
Its `blockMeshDict` has physical cylinder radius 1 and domain vertices
`x=-15..15`, `y=0..2`, with comments naming shrunk `L_up=15R` and
`L_down=15R`
(`bench/rheotool/cylinder_wi1.0_shrunk15R/system/blockMeshDict:20-37`,
`:64-71`). No separate mesh directories encode R=40, 50, or 60.

Implication: hypothesis α remains possible, because the published rT
comparator has no mesh-refinement coverage here. It cannot be promoted from
the current tree. The larger-domain rT variant differs by only about 0.018 Cd
from shrunk15R, so rT domain length alone is not showing a multi-Cd effect.

## (β) Domain-size effect

The Kraken sweep driver interprets `L_up` and `L_down` as radius multiples,
not as absolute lattice lengths:

- `H = 4R`, `Nx = (L_up + L_down)R`, `Ny = H`
  (`bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl:466-468`).
- `nu_total = u_mean * R / Re_R`, with `nu_s = beta*nu_total`,
  `nu_p = (1-beta)*nu_total`, and `lambda = Wi*R/u_mean`
  (`run_cyl_bigsweep_v2_2d.jl:469-473`).
- The cylinder center is `x = L_up*R`, `y = (Ny-1)/2`, and the lattice radius
  is `R` (`run_cyl_bigsweep_v2_2d.jl:480-514`).
- The coupled driver then receives those same `radius`, `H`, `L_up`, and
  `L_down` (`run_cyl_bigsweep_v2_2d.jl:559-561`).

For `L_up=L_down=15`, `u_mean=0.005`, `Re_R=1`:

| R | Nx x Ny | center `(x,y)` | inlet to front surface | rear surface to outlet node | wall clearance | requested `R + L_up*R` | `nu_total` | `s_plus` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 900 x 120 | (450, 59.5) | 420 | 419 | 29.5 | 480 | 0.15 | 1.0526 |
| 40 | 1200 x 160 | (600, 79.5) | 560 | 559 | 39.5 | 640 | 0.20 | 0.9091 |
| 50 | 1500 x 200 | (750, 99.5) | 700 | 699 | 49.5 | 800 | 0.25 | 0.8000 |
| 60 | 1800 x 240 | (900, 119.5) | 840 | 839 | 59.5 | 960 | 0.30 | 0.7143 |

`s_plus = 1/(3nu + 1/2)` per `trt_rates`
(`src/kernels/fused_trt_2d.jl:100-128`).

This does not change nondimensional blockage: continuum `D/H = 2R/4R = 0.5`,
and inlet/front/outlet distances remain about 14 cylinder radii in every row.
So the literal "channel grows physically with R" explanation is not supported
if R is treated as the lattice resolution of the same nondimensional cylinder.

However, pressure-boundary distance in lattice cells does grow strongly
(front clearance 420 -> 840 LU), and the outlet is a fixed-density
Zou-He pressure condition: `default_step_bcspec_2d` sets west
`MaskedZouHeVelocity` and east `MaskedZouHePressure`
(`src/drivers/step_geometry_2d.jl:278-281`), applied each step after the
LI-BB/Guo update (`src/drivers/viscoelastic_logfv_2d.jl:477-481`). Thus a
finite-lattice pressure-boundary imprint could decay with R and make R=30
high relative to R=60. The simultaneous `nu_total`/TRT relaxation change is a
second R-coupled numerical variable.

Verdict on β: true nondimensional domain-size/blockage change is mostly
refuted; lattice-distance plus viscosity/relaxation scaling remains plausible.

## (γ) Residual coupling bug audit

Pressure/readout check:

- The live log-FV readout now computes `rho` as the raw population sum and
  writes velocities without `+F/2`
  (`src/kernels/logconformation_fv_2d.jl:1025-1050`).
- The base 2D forced getter has the same no-half-step convention
  (`src/kernels/macroscopic.jl:60-74`).
- The pressure/VOF getter computes `p = sum(f)/3`; its `+F/2` is only in
  velocity, not pressure (`src/kernels/macroscopic.jl:125-150`), and it is not
  on the cylinder log-FV path.
- `WriteMoments` writes `rho_out = rho` directly (`src/kernels/dsl/bricks.jl:829-835`).
  The two-pass Bouzidi branch even has an explicit cut-link rho recompute by
  re-summing `f_out` (`src/kernels/dsl/bricks.jl:697-724`), though the sweep
  used `wall_bc=halfwayBB`.

No pressure-side `+F/2` or `+G/2` density/readout bug was found in the active
cylinder path.

Operator/coupling check:

- With `embedded_force=0` and `embedded_gradient=0` in the sweep rows
  (`SUMMARY.csv:14-17`), the active polymer-force path is
  `logfv_polymer_force_bc_aware_2d!`, which delegates to
  `fvfd_tensor_divergence_2d!`
  (`src/kernels/logconformation_fv_2d.jl:649-658`).
- That operator is solid-mask and BC aware, but not q_wall cut-cell aware in
  this default mode. Near solids it uses one-sided/central derivatives through
  `_fvfd_solid_bc_derivative_*` (`src/fvfd/operators_2d.jl:13-75`) and then
  forms `div(tau)` (`src/fvfd/operators_2d.jl:724-754`).
- The embedded cut-cell divergence has explicit face fractions, wall lengths,
  and division by cell volume fraction (`src/fvfd/operators_2d.jl:756-857`),
  but that path was off in the residual sweep.
- BSD force uses the same BC-aware compact stencil family and scales as
  `zeta*nu_p*lap(u)` (`src/fvfd/operators_2d.jl:1001-1048`), reached from
  `logfv_bsd_correct_force_bc_aware_2d!`
  (`src/kernels/logconformation_fv_2d.jl:711-719`).

This is the only new gamma-class candidate: a q_wall/staircase mismatch in
the default FVFD polymer-force and BSD-force coupling can feed the solvent
pressure/MEA path at high Wi while leaving wall-integrated `Cd_p` nearly flat.
It is not a confirmed bug. The low-Wi flatness argues against a pure solvent
MEA or base LI-BB error, but high Wi amplifies stress-gradient coupling.

BSD / tau timing check:

The production loop order is current-step advection, current-step velocity
gradient, all polymer substeps, `psixx = latest`, stress reconstruction,
polymer force, BSD correction, constant force, LI-BB/Guo solvent update,
BC rebuild, drag sampling, and finally macroscopic readout
(`src/drivers/viscoelastic_logfv_2d.jl:389-528`). `Cd_p` uses the same latest
`tauxx/tauxy/tauyy` reconstructed before the solvent step
(`src/drivers/viscoelastic_logfv_2d.jl:447-498`). `Cd_bsd` uses the same
current-step velocity gradients used to build the applied BSD force
(`src/drivers/viscoelastic_logfv_2d.jl:467-518`).

No operator was found that uses `tau_p` before the latest polymer substep.

## Verdict

Undetermined, leaning mixed β/γ:

- α is untestable from the available rT tree: no R=40/50/60 reference exists.
- β as a true physical domain-size or blockage change is mostly refuted,
  because the nondimensional geometry is constant. A lattice-distance and
  TRT-relaxation scaling effect remains plausible.
- γ has no pressure half-step smoking gun. The only credible residual
  coupling candidate is the default non-embedded FVFD force/BSD stencil near
  the q_wall cylinder, which can perturb the solvent pressure path at high Wi.

## Recommended next mission (if any)

1. Use existing saved fields to produce per-theta `Cd_s`/pressure and `Cd_p`
   decompositions for R=30/40/50/60 at Wi=1 beta=0.59. Confirm whether the
   drop is front-pole pressure, shoulder, or wake.
2. Run a controlled follow-up when jobs are allowed: R=30 and R=60 with
   `embedded_force=1`, then `embedded_gradient=1`, then both. If `Cd_s`
   moves toward flatness, the gamma FVFD/q_wall candidate is live.
3. Run an L_up/L_down discriminant at fixed R (for example R=60 with 10, 15,
   30 radii) to separate true pressure-boundary distance from mesh refinement.
4. If α must be closed, generate actual rheoTool mesh-refinement cases
   comparable to Kraken R=40/50/60; the current rT artifacts cannot answer it.
