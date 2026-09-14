# Equation audit — Liu 2025 / rheoTool / Kraken

Date: 2026-04-29.

Scope: 2D confined Oldroyd-B cylinder validation, `Re_R = 1`, `Wi_R = 0.1`,
`β = 0.59`, blockage `D/H = 0.5`.

Primary references:
- Liu et al. 2025, arXiv `2508.16997`.
- Local rheoTool finite-inertia runs in `bench/rheotool/`.

## Validation targets

| Case | Source | Target |
|---|---|---:|
| Newtonian, `Re_R=1`, `Re_D=2` | `bench/rheotool/cylinder_newtonian_re1/RESULTS.md` | `Cd = 132.362236515` |
| Oldroyd-BLog, `Re_R=1`, `Wi_R=0.1` | `bench/rheotool/cylinder_oldroydb_log_re1_wi01/RESULTS.md` | `Cd = 130.428774404` |
| Oldroyd-B, Liu CNEBB, `R=30`, `Sc=1e4` | Liu Table 3 | `Cd = 130.36` |

Liu normalizes drag with `Fx / (0.5 ρ U_avg^2 D)` and `D = 2R`. For
these cylinder cases Kraken reports both `Re_R = U_avg R / ν0` and
`Re_D = U_avg D / ν0`; the target run is `Re_R=1`, not `Re_D=1`.

## Equation mapping

| Equation block | Reference form | Kraken implementation | Test coverage | Status |
|---|---|---|---|---|
| Oldroyd-B stress closure | `τp = G(C-I)`, `G=νp/λ` | `src/drivers/viscoelastic_spec.jl` | `test/test_viscoelastic_equations.jl` | OK |
| Conservative conformation source | `∂t φ + ∂α(φuα) = ... + Φαβ`; `Φ` includes `Aαβ ∂γuγ` | `src/kernels/conformation_lbm_2d.jl`, `src/kernels/conformation_lbm_3d.jl` | `test/test_viscoelastic_equations.jl` | Fixed: `C div(u)` was missing |
| Log-conformation scalar transport | same conservative LBM form, so source needs `Ψ div(u)` | `src/kernels/logconformation_lbm_2d.jl` | `test/test_viscoelastic_equations.jl`, `test/test_logconformation.jl` | Fixed: `Ψ div(u)` was missing |
| CNEBB wall conservation | reconstruct unknown links, then set rest population to enforce `Σg=φ` | `src/kernels/conformation_lbm_2d.jl`, `src/kernels/conformation_lbm_3d.jl` | `test/test_viscoelastic_equations.jl`, `test/test_conformation_lbm_3d.jl` | Fixed: rest-population rebalance was missing |
| CNEBB domain walls | out-of-domain neighbours are physical walls for the CDE | `src/kernels/conformation_lbm_2d.jl`, `src/kernels/conformation_lbm_3d.jl` | `test/test_viscoelastic_equations.jl` | Fixed: domain edges now trigger CNEBB |
| CNEBB equilibrium velocity | Liu Eq. 39 uses `g_i^eq(x_b,t+Δt)`, i.e. the boundary fluid-node velocity in production runs | `src/kernels/conformation_lbm_2d.jl`, `src/drivers/viscoelastic.jl` | `test/test_viscoelastic_equations.jl`, `test/test_logconformation.jl` | Added velocity-aware 2D CNEBB wrapper |
| CDE discrete source | `Fe_i = w_i S + (1-1/(2τp,1)) w_i e_i·u S / cs²` | `src/kernels/conformation_lbm_2d.jl`, `src/kernels/logconformation_lbm_2d.jl` | `test/test_viscoelastic_equations.jl` | Fixed: first-order Hermite source moment was missing |
| CDE velocity gradient near walls | source `S` needs `∇u`; boundary stencils must not halve the wall shear or wrap inlet/outlet in x | `src/kernels/conformation_lbm_2d.jl`, `src/kernels/logconformation_lbm_2d.jl` | `test/test_viscoelastic_equations.jl` | Fixed: one-sided fluid-side stencil near walls/solids |
| CDE regularized collision | Liu Eq. 26 moment reconstruction from `Bneq` moments | `src/kernels/conformation_lbm_2d.jl` | `test/test_viscoelastic_equations.jl` | Added as optional diagnostic path |
| CDE Eq. 26 source terms | add `Ge_i` density-gradient term and `0.5∂tFe_i` source-history term | `src/kernels/conformation_lbm_2d.jl` | `test/test_viscoelastic_equations.jl` | Added as direct-C diagnostic `:liu_eq26`; does not fix high-Schmidt Cd |
| CDE Eq. 26 `Bneq` definition | diagnostic source-corrected `Bneq`, optional suppression of the zeroth reconstruction moment, and raw-vs-Hermite second moment | `src/kernels/conformation_lbm_2d.jl`, `src/drivers/viscoelastic.jl`, `hpc/force_scheme_matrix.jl` | `test/test_viscoelastic_equations.jl` | Tested: not sufficient; `+0.5Fe` only helps through `B0`, raw second moment has negligible impact |
| CDE macro recovery | diagnostic recovery `C = Σg + αS` after streaming | `src/kernels/conformation_lbm_2d.jl`, `src/drivers/viscoelastic.jl`, `hpc/force_scheme_matrix.jl` | `test/test_viscoelastic_equations.jl` | Tested: `+0.5S` diverges, `-0.5S` overcorrects |
| CNEBB wall `φ` recovery | diagnostic modes `pre_opp`, `post_opp`, and previous-field `φ` at near-wall fluid nodes | `src/kernels/conformation_lbm_2d.jl`, `src/drivers/viscoelastic_spec.jl`, `hpc/force_scheme_matrix.jl` | `test/test_viscoelastic_equations.jl` | `field` reduces high-Schmidt Cd but not enough |
| Straight-wall CDE isolation | fixed-velocity Poiseuille with analytic Oldroyd-B `C`, periodic x, wall CNEBB, `Sc=1e4` | `hpc/poiseuille_highsc_cde_isolation.jl` | local CPU diagnostic | Shows high-Schmidt failure before cylinder drag |
| Hydrodynamic Hermite stress source | collision contains a `-ω T_i` stress term | `src/kernels/collide_viscoelastic_source_2d.jl`, `src/kernels/dsl/bricks.jl` | `test/test_viscoelastic_force_accounting.jl` | OK for moment sign |
| Liu force equation | MEM over post-collision wall links | `compute_drag_libb_liu_eq63_2d` | `test/test_viscoelastic_force_accounting.jl` | Equivalent to Mei reconstruction |

## What changed

Two local bugs were found before `Cd`:

1. The conformation LBM solves the conservative scalar equation, but the
   source implemented only the advective Oldroyd-B right-hand side. The
   missing `φ div(u)` term is zero only for exactly incompressible velocity
   fields. LBM is weakly compressible, so this is a real equation
   transcription bug.
2. The CNEBB kernel computed the conservative wall value `φ` but did not
   rebalance the rest population. The next macro recovery could therefore
   overwrite `φ` with `Σg ≠ φ`, breaking the local conservation property.
3. The CDE source was missing Liu's first-order Hermite source moment
   proportional to `(1-1/(2τp,1)) u S`.
4. CNEBB was not triggered on domain walls because out-of-domain neighbours
   were ignored in the `any_solid` test. This had negligible impact on the
   R30 cylinder Cd but was a real boundary transcription error.
5. The CDE source velocity-gradient stencil used a periodic wrap in x and a
   centered/clamped stencil at walls. A unit wall-shear test now enforces the
   one-sided fluid-side derivative at domain or embedded-solid walls. This is
   the first fix in this sequence that materially changes the R30 Cd.

These fixes are pre-`Cd` unit-level fixes. They do not by themselves prove
the cylinder benchmark is recovered; they remove two equation-level
differences that made exact agreement with Liu/rheoTool impossible.

## Tests run

```text
julia --project=. test/test_viscoelastic_equations.jl
julia --project=. test/test_logconformation.jl
julia --project=. test/test_viscoelastic_force_accounting.jl
julia --project=. test/test_conformation_lbm.jl
julia --project=. test/test_simple_shear_3d.jl
julia --project=. test/test_conformation_lbm_3d.jl
```

All passed on local CPU.

## Next benchmark run

Rerun the centered `R=30` force/source matrix before changing force
accounting again:

- geometry: `KRAKEN_GEOMETRY_MODE=centered_legacy`
- target: Newtonian `Cd = 132.362236515`
- target: Oldroyd-B `Cd ≈ 130.36–130.43`
- expected diagnostic: `Cl ≈ 0` to roundoff

If `Cd` remains near `128.44`, the remaining difference is likely a scheme
difference versus Liu's improved regularized conformation LBM source terms
and/or Liu's wall-optimized `Λp`, not a simple sign error in the equations
covered above.

## R30 benchmark outcomes after fixes

All rows use centered geometry (`Nx=900`, `Ny=120`, `cx=450`, `cy=59.5`),
`Re_R=1`, `Wi_R=0.1`, `β=0.59`, and `Cl≈0`.

| Run | Key parameters | Cd | Reference | Error |
|---|---|---:|---:|---:|
| `tmp/force_matrix_eqfix_20260428_1745` | `τp,1=1`, `Λs=3/16`, log-conf | `128.833065` | `130.36` | `-1.171%` |
| `tmp/force_matrix_fei_20260429_0940` | plus Liu first-order `Fe_i` | `128.832769` | `130.36` | `-1.172%` |
| `tmp/force_matrix_lambdas_20260429_0950` | `Λs=1/4`, `τp,1=1`, log-conf | `128.712484` | `130.36` | `-1.264%` |
| `tmp/force_matrix_liu_params_20260429_0952` | `Sc=1e4`, `τp,1=0.50002655`, `Λp=2.5e-7`, log-conf | `178.639241` | `130.36` | `+37.035%` |
| `tmp/force_matrix_liu_direct_20260429_1010` | same `Sc/Λp`, direct-C simple TRT | `179.631975` | `130.36` | `+37.797%` |
| `tmp/force_matrix_liu_direct_regularized_20260429_1010` | same `Sc/Λp`, direct-C regularized diagnostic | `180.311427` | `130.36` | `+38.318%` |
| `tmp/force_matrix_domain_cnebb_tau1_20260429_1100` | domain-wall CNEBB trigger, `τp,1=1`, log-conf | `128.832769` | `130.36` | `-1.172%` |
| `tmp/force_matrix_domain_cnebb_sc1e4_20260429_1100` | domain-wall CNEBB trigger, `Sc=1e4`, log-conf | `178.385773` | `130.36` | `+36.841%` |
| `tmp/force_matrix_wallgrad_tau1_20260429_1115` | plus wall-aware `∇u`, `τp,1=1`, log-conf | `130.739169` | `130.36` | `+0.291%` |
| `tmp/force_matrix_wallgrad_sc1e4_20260429_1115` | plus wall-aware `∇u`, `Sc=1e4`, log-conf | `182.936749` | `130.36` | `+40.332%` |
| `tmp/force_matrix_cnebb_u_tau1_20260429_1130` | plus velocity-aware CNEBB, `τp,1=1`, log-conf | `130.739169` | `130.36` | `+0.291%` |
| `tmp/force_matrix_cnebb_u_sc1e4_20260429_1130` | plus velocity-aware CNEBB, `Sc=1e4`, log-conf | `182.979541` | `130.36` | `+40.365%` |
| `tmp/force_matrix_liu_eq26_direct_20260429_1210` | direct-C `:liu_eq26`, `Sc=1e4`, `Λp=2.5e-7` | `186.629744` | `130.36` | `+43.165%` |
| `tmp/force_matrix_bneq_mhalf_20260429_1230` | direct-C `:liu_eq26`, `Bneq += -0.5Fe`, `B0` active | `188.050250` | `130.36` | `+44.255%` |
| `tmp/force_matrix_bneq_phalf_20260429_1230` | direct-C `:liu_eq26`, `Bneq += +0.5Fe`, `B0` active | `181.385161` | `130.36` | `+39.142%` |
| `tmp/force_matrix_bneq_b0zero_raw_20260429_1205` | direct-C `:liu_eq26`, `B0=0`, no source correction | `186.629744` | `130.36` | `+43.165%` |
| `tmp/force_matrix_bneq_b0zero_phalf_20260429_1205` | direct-C `:liu_eq26`, `Bneq += +0.5Fe`, `B0=0` | `186.629744` | `130.36` | `+43.165%` |
| `tmp/force_matrix_macro_phalf_20260429_1225` | direct-C `:liu_eq26`, macro recovery `C=Σg+0.5S` | `NaN` | `130.36` | `NaN` |
| `tmp/force_matrix_macro_mhalf_20260429_1225` | direct-C `:liu_eq26`, macro recovery `C=Σg-0.5S` | `79.608737` | `130.36` | `-38.932%` |
| `tmp/force_matrix_bneq_raw_20260429_1240` | direct-C `:liu_eq26`, raw second `Bneq` moment | `186.629744` | `130.36` | `+43.165%` |
| `tmp/force_matrix_bneq_raw_phalf_20260429_1240` | direct-C `:liu_eq26`, raw second `Bneq` moment, `Bneq += +0.5Fe` | `181.384355` | `130.36` | `+39.141%` |
| `tmp/force_matrix_cnebb_postopp_20260429_1300` | direct-C `:liu_eq26`, CNEBB `φ` from post-streaming opposite | `185.869202` | `130.36` | `+42.581%` |
| `tmp/force_matrix_cnebb_field_20260429_1300` | direct-C `:liu_eq26`, CNEBB `φ` from previous macro field | `164.952412` | `130.36` | `+26.536%` |

## Straight-wall CDE isolation

The next pre-drag test is fixed-velocity Poiseuille with analytic Oldroyd-B
conformation as the initial condition:

- `R=30`, `Ny=120`, `Nx=16`, `u_mean=0.005`, `Wi=0.1`, `β=0.59`
- `Sc=1e4`, so `τp,1=0.50002655`
- `Λp=2.5e-7`
- periodic x, wall CNEBB in y
- `120000` CDE steps, no hydrodynamic feedback and no drag computation

| Collision | CNEBB `φ` mode | `Cxy_l2` | `N1_l2` | `min eig(C)` | Conclusion |
|---|---|---:|---:|---:|---|
| TRT | `pre_opp` | `2.7e29` | `6.0e29` | `-1.3e29` | diverges |
| Liu Eq. 26 | `pre_opp` | `0.329` | `0.497` | `0.895` | stable but wrong |
| TRT | `field` | `9.2e-4` | `2.9e-2` | `0.872` | stable and close |
| Liu Eq. 26 | `field` | `1.1e-3` | `1.8e-2` | `0.872` | stable and close |

One-step residual from the exact analytic Poiseuille initial condition
(`steps=1`, same `R/Ny/Nx/Wi/β/Λp`, collisions `trt, regularized, liu_eq26`,
`τp,1∈{1,0.50002655}`):

| CNEBB `φ` mode | `Cxy_l2` | `Cxy_core_l2` | `wall_Cxy_max_abs` | Observation |
|---|---:|---:|---:|---|
| `pre_opp` | `6.08e-4` | `9.48e-10` | `4.08e-4` | wall error appears immediately |
| `field` | `1.08e-16` | `1.06e-16` | `0.0` | Cxy wall residual eliminated |
| `eq_gradient` | `1.12e-16` | `1.12e-16` | `0.0` | same one-step residual as `field` |

The single-link CNEBB unit test now checks Eq. 39 for each D2Q9 wall link and
passes to roundoff, so this is not a local sign/index error in the unknown-link
reconstruction. The error is produced by the wall `φ` recovery applied to a
non-uniform high-Schmidt wall profile.

The `eq_gradient` diagnostic mode reconstructs the wall value as local
equilibrium plus transported non-equilibrium residuals. It passes an exact
linear-equilibrium wall-profile unit test, but it is not a viable fix:
at `Sc=1e4`, `steps=20000`, both TRT and Liu Eq. 26 return `NaN` in the
Poiseuille isolation. The stable candidate remains `field`, but `field` alone
is not enough on the cylinder.

### 2026-04-29 — Curved-wall CNEBB/cut-link isolation

`hpc/cylinder_cnebb_cutlink_isolation.jl` isolates the geometry mismatch on
the R30 cylinder without drag or hydrodynamic feedback:

- solvent wall: LI-BB uses `q_wall` with true cut-link fractions
- polymer wall: CNEBB currently sees only `is_solid`
- R30 grid: `248` near-cylinder fluid cells, `584` cut links
- `q_wall` range: `0.0122` to `1.0`, mean `0.4729`
- `328/584` links have `|q_wall-0.5| > 0.25`

One-step manufactured-profile residuals on cut-link cells:

| Profile | CNEBB `φ` mode | Macro residual `l2` | Macro max | Missing-pop max |
|---|---|---:|---:|---:|
| linear | `pre_opp` | `1.17e-2` | `2.08e-2` | `4.18e-4` |
| linear | `field` | `0.0` | `0.0` | `1.57e-4` |
| linear | `eq_gradient` | `7.5e-17` | `2.2e-16` | `1.57e-4` |
| linear | `cutlink_libb_field` | `0.0` | `0.0` | `1.40e-2` |
| radial quadratic | `pre_opp` | `1.25e-2` | `2.04e-2` | `1.94e-4` |
| radial quadratic | `field` | `0.0` | `0.0` | `2.06e-4` |
| radial quadratic | `eq_gradient` | `9.7e-17` | `2.2e-16` | `2.06e-4` |
| radial quadratic | `cutlink_libb_field` | `0.0` | `0.0` | `1.35e-2` |

This is the first direct evidence that the cylinder blocker is not simply the
straight-wall `φ` recovery. Even when the macro value is pinned (`field`), the
unknown incoming populations at the polymer wall remain inconsistent with the
cut-link geometry used by the solvent. The next candidate fix should therefore
be a q-wall-aware conformation boundary reconstruction, not another Cd force
mode.

A naïve Bouzidi-style q-wall reconstruction with macro rebalance
(`cutlink_libb_field`, implemented only in the diagnostic script) is rejected:
it keeps the macro value exact but increases the missing-population residual by
roughly two orders of magnitude versus `field`.

### 2026-04-29 — Coherent staircase geometry check

To avoid inventing a new wall scheme, `hpc/staircase_coherent_geometry_check.jl`
tests the simpler hypothesis: keep the modern LI-BB/source/drag driver exactly
unchanged, but force every cylinder cut link to `q_wall=0.5`. This gives a
coherent staircase/halfway wall for the solvent while the polymer still uses
the same `is_solid` CNEBB mask. The only intended difference between rows is
therefore `wall_geometry=:cutlink` versus `:staircase`.

Short local R10 check (`Nx=300`, `Ny=40`, `steps=3000`, `avg=600`,
`Re_R=1`, `Wi=0.1`, `β=0.59`, `Sc=1e4`):

| Collision | Wall geometry | Case | `Cd/Cd_newt` | `min eig(C)` | Conclusion |
|---|---|---|---:|---:|---|
| TRT | `cutlink` | `τp,1=1` | `1.024` | `0.814` | healthy |
| TRT | `staircase` | `τp,1=1` | `1.019` | `0.814` | healthy |
| TRT | `cutlink` | `Sc=1e4` | `0.572` | `-0.152` | unphysical |
| TRT | `staircase` | `Sc=1e4` | `0.571` | `-0.151` | unphysical |
| Liu Eq. 26 | `cutlink` | `τp,1=1` | `1.024` | `0.814` | healthy |
| Liu Eq. 26 | `staircase` | `τp,1=1` | `1.018` | `0.814` | healthy |
| Liu Eq. 26 | `cutlink` | `Sc=1e4` | `NaN` | `NaN` | diverges |
| Liu Eq. 26 | `staircase` | `Sc=1e4` | `NaN` | `NaN` | diverges |

Conclusion from this simpler test: the high-Schmidt failure is not caused only
by mixing solvent cut-link geometry with polymer `is_solid` CNEBB. Forcing
the solvent wall to a coherent halfway/staircase geometry leaves the
`Sc=1e4` pathology essentially unchanged. The next root-cause target should
move back to the near-wall high-Schmidt CDE update itself: relaxation/magic,
regularized collision/source ordering, and positivity preservation near walls.

This isolates the blocker before the drag calculation: with the default
`pre_opp` CNEBB recovery, the high-Schmidt CDE does not preserve a simple
straight-wall Oldroyd-B Poiseuille solution. Reusing the previous macro field
as the near-wall conservative `φ` stabilizes the straight-wall test and gives
sub-percent `Cxy` error, but it did not fully fix the cylinder Cd. Therefore
the remaining validation problem is not force accounting; it is the
high-Schmidt conformation boundary update, especially how the CNEBB wall value
is coupled to curved cut links and to the subsequent collision/source update.

Conclusion: the production validation path with Kraken's diffusive
conformation setting (`τp,1=1`) is now numerically close to Liu/rheoTool at
R30 (`+0.29%` versus Liu Table 3, `+0.24%` versus rheoTool). The remaining
blocker is specifically the high-Schmidt Liu configuration: with
`Sc=1e4`, `τp,1≈0.50003`, `Λp=2.5e-7`, Kraken still over-stresses the
cylinder by roughly `40%`. Adding the explicit Eq. 26 `Ge_i` and
`0.5∂tFe_i` terms in direct-C does not fix it; it increases the R30 error to
`+43.165%`. Source-correcting `Bneq` by `+0.5Fe` improves the error only to
`+39.142%`, and suppressing the `B0` reconstruction removes that improvement
entirely. That means the simple transformed-distribution interpretation of
`Bneq` is not the missing high-Schmidt fix. Switching the Eq. 26 second
non-equilibrium moment from Hermite/traceless to raw `Σeαeβgneq` also changes
nothing material at R30. A direct half-source macro recovery is ruled out:
the physically usual `+0.5S` sign diverges, while `-0.5S` moves the drag far
below the target. The remaining likely mismatch is therefore a deeper
high-Schmidt CDE boundary/ordering difference near the cylinder wall, not the
bulk Eq. 26 moment algebra tested so far. This is supported by the CNEBB
diagnostics: `post_opp` barely moves Cd, but using the previous near-wall
macro field for `φ` reduces the error from `+43.165%` to `+26.536%`. It is
still not Liu. The straight-wall Poiseuille isolation makes the next step
concrete: implement a production-quality high-Schmidt wall update that uses a
well-defined wall `φ`/macro state before reconstructing unknown populations,
then rerun the cylinder. Until that passes, further `Cd` force-mode sweeps are
not useful.
