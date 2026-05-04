# Project viscoelastic audit

Date: 2026-05-01.

## Executive status

Superseded status: the practical 2D Oldroyd-B cylinder validation is **not
accepted** as a solver-level validation. The previous `R=30`, `Wi_R=0.1`
agreement is a single-point crossing exposed by the Wi sweep.

The old validation point used:

- `R = 30`, confined cylinder, blockage `D/H = 0.5`.
- `Re_R = U_mean R / ν = 1`, therefore `Re_D = 2`.
- `Wi_R = λ U_mean / R = 0.1`.
- `β = ηs / η0 = 0.59`.
- Log-conformation Oldroyd-B, CNEBB polymer wall condition, LI-BB hydro wall.
- Conformation diffusion setting: `τp,1 = 1`.

This remains a useful numerical record, but it is not sufficient to validate
the solver.

## Final validation numbers

Canonical Kraken run:
`tmp/force_matrix_cnebb_u_tau1_20260429_1130/results/force_scheme_matrix.txt`.

| Case | Kraken `Cd` | Reference | Error | `Cl` |
|---|---:|---:|---:|---:|
| Newtonian vs rheoTool | `132.076371` | `132.362236515` | `-0.216%` | `3.97e-13` |
| Oldroyd-B vs Liu/Yu | `130.739168759` | `130.36` | `+0.291%` | `3.37e-13` |
| Oldroyd-B vs rheoTool | `130.739168759` | `130.428774404` | `+0.238%` | `3.37e-13` |

Interpretation:

- The Newtonian baseline is within `0.22%` of the finite-inertia rheoTool
  reference.
- The viscoelastic drag is within `0.30%` of both Liu/Yu and rheoTool.
- The lift is zero to numerical precision, so the geometry is centered.
- rheoTool and Liu/Yu are mutually consistent at this point:
  `130.428774404` vs `130.36`, i.e. about `0.053%`.

Conclusion: the R30 Oldroyd-B number is close at this one operating point, but
the solver is not validated because the `Cd(Wi)` curve is wrong.

## Reference data

Source files:

- Kraken validation record:
  `bench/viscoelastic_audit/VALIDATION_R30_OLDROYDB.md`.
- rheoTool Newtonian finite-inertia target:
  `bench/rheotool/cylinder_newtonian_re1/RESULTS.md`.
- rheoTool Oldroyd-BLog finite-inertia target:
  `bench/rheotool/cylinder_oldroydb_log_re1_wi01/RESULTS.md`.
- Liu/Yu tabulated values:
  `REFERENCES.md`.

Reference values:

- rheoTool Newtonian: `Cd(t >= 0.8) = 132.362236515`.
- rheoTool Oldroyd-BLog: `Cd_viscous(t >= 0.8) = 130.428774404`.
- Liu/Yu Table 3 at `R=30`, `Wi=0.1`: `Cd = 130.36`.

The rheoTool cases use active inertia (`v·∇v`). At `Re_R = 1`, rheoTool's
viscous normalization is numerically equal to Kraken's inertial drag
normalization `Fx / (0.5 ρ U_mean^2 D)`.

## What remains unresolved

The exact Liu high-Schmidt setup is not validated:

- `Sc = 1e4`.
- `τp,1 ≈ 0.50003`.
- `Λp ≈ 2.5e-7`.

Current audit runs show this configuration over-stresses the cylinder by about
`40%`. This is a scheme-reproduction issue, probably tied to high-Schmidt
conformation boundary/collision coupling near curved walls.

## Bugs found

See `bench/viscoelastic_audit/BUGS_20260501.md`.

Confirmed:

- `source_scaled_mea` is a cancellation/calibration path, not a physical force
  law.
- `compute_polymeric_drag_2d` is not a curved-surface quadrature.
- `hpc/sweep_wi_cylinder.jl` ignored `KRAKEN_GEOMETRY_MODE`.
- The 4:1 contraction log-conformation inlet used `C` where the evolved field
  is `Ψ=log(C)`.
- Step geometries need masked inlet/outlet BCs; the contraction outlet was
  previously rebuilt on solid rows. The fix is modular:
  `StepChannelGeometry2D` plus `run_conformation_step_libb_2d`, not a driver
  per geometry. The audit script now sweeps `contraction,bfs` via the same
  solver path.
- `hpc/bulk_constitutive.jl` now provides the missing constitutive pre-gate:
  exact shear, elongation, and imposed Poiseuille for `direct/logconf` before
  any step or obstacle case.
- The Wi sweep invalidates the previous single-point validation.

High-probability next targets:

- CNEBB is applied directly to `Ψ=log(C)` in log-conformation mode, although
  the boundary condition is derived for `C`.
- The CNEBB equilibrium velocity convention is inconsistent (`u_wall=0` in the
  derivation/comments, local fluid velocity in production).
- The Hermite source prefactor convention still needs one physical default.

## Engineering decision

Do not use the `τp,1 = 1` diffusive conformation configuration as a validated
production path yet.

Do not keep changing Reynolds definitions or Cd normalization for this
benchmark. Those are pinned:

- Liu/Kraken Reynolds for this benchmark uses `R`, not `D`.
- Drag uses `D = 2R` in the denominator.
- Therefore the recorded case has `Re_R = 1` and `Re_D = 2`.

## Recommended next actions

1. Remove `source_scaled_mea` from the validation path.
2. Implement a physical `q_wall`-aware polymer traction quadrature.
3. Isolate CNEBB-on-`Ψ` versus CNEBB-on-`C`.
4. Re-run the full Wi sweep against rheoTool before claiming validation.

## Surgical closure update — 2026-05-03

Closed or guarded:

- `source_scaled_mea` is no longer a validation default. The cylinder LI-BB
  driver defaults to `drag_mode=:post_source_mea` and
  `hermite_source_mode=:liu_direct`; `source_scaled_mea` now requires
  `allow_diagnostic_force_mode=true`.
- Public validation scripts now default to `post_source_mea/liu_direct`.
  Scripts whose purpose is explicitly a force-scheme audit opt in to the old
  cancellation path.
- Unsupported CDE collision windows are guarded. Validation accepts
  `(:trt, tau_plus=1.0)` and the patch-tested high-Schmidt diagnostic windows
  `(:regularized, 0.50001)` / `(:liu_eq26, 0.50001)`. Other combinations need
  `allow_diagnostic_conformation_collision=true`.
- `CNEBBEqGradient` remains diagnostic-only and cannot enter validation flows
  without `allow_diagnostic_polymer_bc=true`.
- `logconf + polymer wall BC` is no longer a silent validation path. It now
  requires `allow_diagnostic_log_wall_bc=true` because current wall BC kernels
  reconstruct scalar `Ψ=log(C)` components, while Yu/CNEBB is derived for the
  conformation tensor `C`.
- Obstacle/step validation launchers now default to direct conformation.
  Log-conformation wall cases are audit-only opt-ins; the only default
  `direct,logconf` sweep left in `hpc/` is `bulk_constitutive`, where no wall
  BC is involved.
- The curved polymer traction quadrature now has exact low-level patch tests:
  wall-link linear/quadratic reconstruction, every tensor component
  `τxx/τxy/τyy`, and both force orientations.
- The analytic cylinder `q_wall` precompute now has a pure geometry canary:
  every D2Q9 cut direction, centered and off-lattice cylinder centers, exact
  wall radius, correct fluid-to-solid link convention, and no missing direction.
- The Hermite source / cut-link MEA local increment now has a full matrix
  canary over all D2Q9 wall directions, `q_wall ∈ {0.3,0.5,0.7}`, stress
  components, and CE on/off modes.
- The constitutive source has exact patch tests for arbitrary incompressible
  stationary gradients in both direct-`C` and log-conformation modes.
- The velocity-gradient stencil used by the conformation source is now tested
  for centered, one-sided, boundary, and fully blocked cases.
- Wall-BC macro/conservation canaries now include an actual off-lattice
  cylinder cut-link matrix, not only synthetic single-link and multi-link
  patches. The arbitrary constant-velocity case is explicitly classified as a
  diagnostic broken path because it drives scalar flux through the cylinder.
- The Cartesian square-obstacle rung is now explicit before the curved-cylinder
  rung. `square_obstacle_channel_geometry_2d` has an exhaustive `q_wall`
  canary showing every obstacle and horizontal channel-wall cut-link is
  exactly halfway (`q=0.5`), the wall-BC matrix is tested on an actual square
  obstacle, and Hermite source stress reconstruction is tested on square
  obstacle fluid-to-solid links only. The latter intentionally excludes
  inlet/outlet channel corners, which are planar-domain corner artefacts, not
  obstacle-wall canaries.
- The same square-obstacle rung exposed a real legacy quadrature bug in
  `compute_polymeric_drag_2d(..., is_solid)`: diagonal D2Q9 neighbours were
  counted as finite surface measure, giving `108` instead of the analytic `36`
  for a linear-stress square patch. The solid-mask fallback now integrates only
  the four axis-aligned faces; the q-wall curved quadrature is unchanged.
- Actual square-obstacle force canaries now also cover q=0.5 MEA aggregation
  and the post-Hermite-source MEA increment over all obstacle cut-links, so
  source/sign/double-counting errors are caught before any curved-wall case.
- The Hermite source stress reconstruction near curved walls now has an
  actual-cylinder canary. This exposed and closed a real pollution path:
  reconstruction samples were allowed to read neighbouring cut-link cells that
  had already been corrupted by wall BCs. The fix skips solid cells and other
  cut-link cells, then extrapolates from the first clean interior samples.
- Local Newtonian-limit Cd sweeps after this fix show a small but real gain on
  direct-`C` cylinder runs. At `β=0.59`, `Wi=0.001`, `R=8/15`, `8000` CPU/F64
  steps, `source_stress_reconstruction=:interior` improves `Cd_VE/Cd_Newt`
  relative to `:raw` by `+6.76e-4` at `R=8` and `+3.34e-3` at `R=15`. This
  confirms the fixed source-reconstruction path affects Cd, but does not close
  the remaining Newtonian-limit deficit by itself.
- Curved-wall recheck after the square-obstacle fixes confirms the same
  integrated signal: exact curved canaries pass, the pure Newtonian control
  (`β=1`) matches at machine precision, but the direct-`C` Newtonian-limit
  cylinder remains low. Current CPU/F64 `β=0.59`, `Wi=0.001`, `R=8/15`,
  `8000`-step ratios are `Cd_VE/Cd_Newt = 0.936125/0.977378`. This matches the
  previous post-source-reconstruction numbers, so the square solid-mask fix
  corrected a legacy diagnostic path but did not close the curved LI-BB/CNEBB
  Cd deficit.
- Patch tests `P19`–`P27` now close the curved-wall analytic chain before Cd:
  affine velocity gradients at actual cylinder cut-links are exact, local
  Oldroyd-B closure and `τp` stress conversion are exact, and Hermite
  source + cut-link MEA matches a link-by-link oracle. The first failing layer
  is the active conformation wall BC under non-zero wall-adjacent velocity:
  `CNEBB`, `CNEBBQAware`, and `YLWBalanceOnly` conserve `Σg=C` but corrupt the
  physical node-centered macro field at every cut-link orientation, while
  `CNEBBEqGradient` preserves the affine stationary patch and suppresses the
  spurious source-force canary.
- The same ladder now prevents a false fix: `CNEBBEqGradient` is exact for the
  one-step equilibrium patch but is unstable on the planar Poiseuille CDE patch
  after repeated collision/streaming. A cut-link-only variant also fails the
  exhaustive single-link `q_wall` matrix even though it is exact on the current
  actual-cylinder static geometry. Both are therefore diagnostic canaries, not
  production BCs.
- Direct-`C` Newtonian-limit cylinder sweeps confirm that these diagnostic
  equilibrium corrections do not close Cd. At `β=0.59`, `Wi=0.001`, `R=8/15`,
  `8000` CPU/F64 steps: `CNEBBEqGradient` changes `Cd_VE/Cd_Newt` from
  `0.936125/0.977378` to `0.941083/0.982676`; the cut-link-only variant gives
  `0.937621/0.980342`. The pure Newtonian `β=1` control remains exact. The
  active-wall macro defect is real, but removing only the equilibrium bias is
  insufficient and can break lower rungs.
- The first production-safe improvement is not a wall-BC change: use the
  already patch-tested quadratic interior reconstruction for the Hermite source
  stress near cut-links. With `source_stress_reconstruction_order=2`,
  `Cd_VE/Cd_Newt` improves to `0.948116/0.980165` at `R=8/15` for the same
  direct-`C`, `β=0.59`, `Wi=0.001`, `8000` CPU/F64 sweep. This is now the
  default for the cylinder scaling launcher and driver. It reduces, but does
  not close, the remaining curved-wall deficit.
- The remaining Newtonian-limit deficit was then closed by the TRT
  conformation magic parameter, not by another BC variant. The patch ladder now
  includes `P18b`, which shows that the Liu/Yu-style `Λp=1e-6` is part of the
  validated planar CDE window while the historical hydrodynamic `Λp=0.25`
  fails that same Poiseuille patch. With `Λp=1e-6` plus quadratic source-stress
  reconstruction, direct-`C` cylinder ratios become `0.962879/0.992932` at
  `R=8/15`, and `1.000741` at `R=30` (`β=0.59`, `Wi=0.001`, CPU/F64). The
  `β=1` Newtonian controls remain exact. Driver and validation launcher
  defaults now use `conformation_magic=1e-6`; `0.25` is a diagnostic opt-in.
- Earlier decompositions explain why force accounting was not the closing
  lever. The explicit physical split was too low (`Cd_split/Cd_Newt = 0.8734`
  at `R=8`, `0.9174` at `R=15`) and `Cd_p/Cd_Newt` remained only
  `0.1080/0.1174` instead of the Newtonian-limit target `1-β = 0.41`, while
  post-source MEA compensated part of the deficit. The closure path is
  therefore: CDE patch ladder → source-stress reconstruction order → TRT
  conformation magic, not a new MEA force law.
- The analytic Poiseuille CDE patch now runs in both orientations:
  horizontal channel with `u_x(y)` and vertical channel with `u_y(x)`. This
  keeps the supported collision/window and wall-BC matrix from silently passing
  only one derivative direction.

Still not a validation claim:

- Post-source MEA on an arbitrary imposed analytic stress field is not a
  physical surface quadrature; this is now treated as a diagnostic of the MEA
  source path, not as a target for `∮τp·n ds`.
- CNEBB applied directly to `Ψ=log(C)` is still a modelling/scheme caveat.
  Existing canaries detect the non-equivalence with C-space BCs, and production
  drivers now require diagnostic opt-in before running this path.
- Any Cd validation claim must therefore be made first on the direct-`C`
  equation ladder. Log-conformation wall Cd runs are useful diagnostics, not
  independent validation against Yu/CNEBB equations.
- High-Schmidt CDE at `tau_plus≈0.5` is not a production validation path for
  TRT. It remains behind diagnostic opt-in until the exact low-level patch
  tests support it.

## 2026-05-04 — Direct-C stability canaries after macro NaNs

New low-level gates added before further Cd validation:

- `conformation_field_diagnostics_2d` is now a reusable host-side diagnostic.
  It reports loss of SPD, non-finite fields, maximum numerical divergence,
  maximum positive strain eigenvalue, and the velocity-gradient tensor at the
  minimum-eigenvalue location.
- The conformation/log-conformation source can be run with
  `conformation_divergence_mode=:trace_free`. This projects the numerical
  velocity gradient to the incompressible trace-free part before evaluating
  the constitutive source, while preserving the old `:numerical` path as a
  diagnostic mode.
- `P15c` closes a boundary canary: a conformation inlet reset can be polluted
  by the collision source if the inlet populations are not reset again after
  collision. Production cylinder CDE now reapplies inlet/outlet conformation
  resets after collision so boundary source artefacts cannot stream into the
  domain on the next step.
- `hpc/cylinder_frozen_velocity_cde_patch.jl` isolates the CDE on a frozen
  Newtonian cylinder velocity field. This separates CDE/BC/gradient failures
  from active polymer feedback through the hydrodynamic source.

Current diagnostic facts:

- CPU/F64 frozen CDE at `R=30`, `Wi=0.1`, `β=0.59`, `15000` CDE steps is stable
  in direct `C` and logconf. The minimum eigenvalue stays positive
  (`min_eig≈0.45–0.55` depending on model/step).
- Metal/F32 frozen CDE at the same point loses SPD in direct `C` around
  `15000` steps. This is therefore not evidence of a bulk equation error in
  the F64 validation path; it is a precision/stability limitation of the local
  Metal/F32 diagnostic path.
- A local CPU/F64 active macro-flow at `R=30`, `Wi=0.1`, `β=0.59`, `15000`
  steps is finite with `Cd≈129.64` and `Cl≈0`. The failed command exit was only
  from printing a non-returned diagnostic field after the run had completed.
- At the SPD-min locations, the raw velocity-gradient source sees local
  `λ·s_max≈0.45–0.67`. A cut-link least-squares gradient check changes these
  values only weakly, so a simple q-aware gradient replacement is not the
  missing fix.
- H100/F64 minimal macro validation was submitted as `20758861.aqua` from an
  isolated synced tree (`tmp/visco_surgical_20260504_1200/repo`) with
  `R=30`, `Wi=0.1`, `β=0.59`, `direct,logconf`, and diagnostics every `10000`
  steps. This is the next gate before making a Cd validation claim.
