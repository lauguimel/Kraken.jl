# M44-GUO-AUDIT-CODEX verdict

## (A) Convention of collide_guo_field_2d_kernel!

Convention I, integrated. In `src/kernels/collide_guo_2d.jl:97-101`, the field kernel computes the in-collision velocity as `(raw momentum + F/2) / rho` and then applies the standard Guo source with `guo_pref = 1 - omega/2`; the source terms at `src/kernels/collide_guo_2d.jl:103-128` are the Guo 2002 expansion `w_q[((c_q-u).F)/cs^2 + ((c_q.u)(c_q.F))/cs^4]` with factors 3 and 9. Taking the first moment, the BGK part contributes `omega*F/2` and the Guo source contributes `(1 - omega/2)*F`, so the post-collision raw momentum advances by exactly `F` and is the physical next-step velocity. A post-collision readout must therefore use the raw moment with no additional `+F/2`.

This matches the slbm-paper reference: commit `5ec27044` removed `+F/2` from `compute_macroscopic_forced_2d!`, and commit `0bba4f5d` documents `collide_guo_2d!` as integrated and pairs it with `compute_macroscopic_2d!`.

## (B) Bug presence in logfv_compute_macroscopic_forced_field_2d_kernel!

YES. `src/kernels/logconformation_fv_2d.jl:1047-1048` computes:

```julia
ux[i, j] = (f2 - f4 + f6 - f7 - f8 + f9 + fx[i, j] / T(2)) * inv_rho
uy[i, j] = (f3 - f5 + f6 + f7 - f8 - f9 + fy[i, j] / T(2)) * inv_rho
```

The production log-FV Guo call sites pair this getter with Convention-I collisions:

- `src/drivers/viscoelastic_logfv_2d.jl:2223-2224`: `collide_guo_field_2d!` then `logfv_compute_macroscopic_forced_field_2d!`
- `src/drivers/viscoelastic_logfv_2d.jl:2404-2405`: same pair in the coupled Poiseuille path
- `src/drivers/viscoelastic_logfv_2d.jl:2658-2659`: same pair in the advective coupled path
- `src/drivers/viscoelastic_logfv_2d.jl:477-528`: fused LI-BB/TRT Guo-field solvent step, then the same explicit log-FV readout

Given Convention I, the `+fx/2` and `+fy/2` terms in the readout double-count the Guo half-step correction.

## (C) Bug magnitude prediction

For a Convention-I post-collision field, the correct velocity is:

```text
u_correct = j_post / rho
```

The current log-FV getter returns:

```text
u_bug = (j_post + F_total/2) / rho
```

so the instantaneous readout bias is:

```text
Delta ux = fx_total / (2*rho)
Delta uy = fy_total / (2*rho)
F_total = F_polymer + F_solvent
```

This is a local per-readout bias, not an accumulated `N*F/2` acceleration. For the cylinder Wi=1, R=30 case, the far-field scale in the benchmark scripts is `u_mean = 0.005`, `Re_R = 1`, so `nu_total = u_mean*R/Re_R = 0.15`; near the front pole the polymer back-force is concentrated over O(1) lattice cells. A local force scale `|F_total| = O(1e-5)` gives `|Delta u| = O(5e-6)` for `rho ~= 1`, about `1e-3` of the imposed mean speed; if the local force spike reaches `O(1e-4)`, the velocity readout error reaches `O(5e-5)`.

The pressure effect is spatially structured because the extra velocity is proportional to the polymer/solvent force field. It feeds the next log-FV gradient, advection, BSD correction, equilibrium moments, and density/pressure response; the largest pressure contamination should appear where `div(tau_p)` has sharp gradients, especially around the front pole and near-wall force layer.

## (D) Other code paths (G1..GN inventory)

Focused grep was restricted to `src/` and `test/` after excluding pure collision-local reconstructions from the getter list. Collision-local `u = (j + F/2)/rho` is expected inside Guo collisions; the bug appears when a post-collision getter adds another half-force.

| ID | Getter / readout site | `+F/2` lines | Paired collision path found | Collision Convention | Audit verdict |
| --- | --- | --- | --- | --- | --- |
| G1 | `compute_macroscopic_forced_2d_kernel!` in `src/kernels/macroscopic.jl:60-74` | `src/kernels/macroscopic.jl:71-72` | `collide_guo_2d!` in `src/drivers/basic.jl:197-198`, `src/simulation_runner.jl:216-223`, refined closures `src/simulation_runner.jl:1057-1084`; also `collide_rheology_guo_2d!` through `src/simulation_runner.jl:210-223` | I for `collide_guo_2d!`; I-like for `collide_rheology_guo_2d!` because it uses `+F/2` internally and `guo_pref` at `src/kernels/collide_rheology_2d.jl:90-120` | Same mismatch as slbm-paper; needs fix/audit outside the log-FV deliverable. |
| G2 | `compute_macroscopic_forced_3d_kernel!` in `src/kernels/macroscopic.jl:85-105` | `src/kernels/macroscopic.jl:101-103` | `collide_guo_3d!` in `test/test_poiseuille_3d.jl:73-74`; no production `src/` call found in this grep | I by the same structure: `src/kernels/collide_guo_3d.jl:35-45` and field variant `src/kernels/collide_guo_3d.jl:215-222` | Likely double-count in 3D forced readout tests/calls; add a 3D pair test before changing. |
| G3 | `logfv_compute_macroscopic_forced_field_2d_kernel!` in `src/kernels/logconformation_fv_2d.jl:1025-1051` | `src/kernels/logconformation_fv_2d.jl:1047-1048` | `collide_guo_field_2d!` in `src/drivers/viscoelastic_logfv_2d.jl:2223-2224`, `2404-2405`, `2658-2659`; fused Guo-field step followed by explicit readout at `477-528` and `2796-2803` | I | Confirmed bug in the viscoelastic production/audit paths. |
| G4 | `compute_macroscopic_pressure_2d_kernel!` in `src/kernels/macroscopic.jl:123-149` | `src/kernels/macroscopic.jl:147-148` | Pressure/VOF paths in `src/drivers/axisymmetric.jl` pair it with pressure/twophase collisions; `collide_twophase_2d_kernel!` has `+F/2` and `guo_pref` at `src/kernels/vof_2d.jl:623-656` | Likely I for `collide_twophase_2d_kernel!` | Not viscoelastic, but needs a pressure-VOF convention pair test before any edit. |
| G5 | `compute_macroscopic_phasefield_2d_kernel!` in `src/kernels/phasefield_2d.jl:472-497` | `src/kernels/phasefield_2d.jl:495-496` | `collide_pressure_phasefield_mrt_2d_kernel!` in `src/kernels/phasefield_2d.jl:360-443`, used from `src/drivers/rheology.jl` and `src/drivers/axisymmetric.jl` | Moment-space integrated: `jx_star = jx + fx`, `jy_star = jy + fy` at `src/kernels/phasefield_2d.jl:416-424` | Needs separate pressure/phasefield convention audit; likely same readout risk. |
| G6 | `macroscopic_boussinesq` in `src/kernels/fused_thermal_2d.jl:53-63` | `src/kernels/fused_thermal_2d.jl:61` | Fused thermal Boussinesq collision in the same file; non-fused Boussinesq collisions also use standard Guo at `src/kernels/thermal_2d.jl:211-244` and `src/kernels/thermal_3d.jl:304-313` | I-like collision structure | It is an in-step collision helper, not a post-collision production getter. Audit only if its returned `ux/uy` are exposed as final macroscopic fields. |
| G7 | `fused_trt_libb_v2_guo_field_step!` via `CollideTRTDirectGuoField` plus `WriteMoments` | `src/kernels/dsl/bricks.jl:150-156`, written at `src/kernels/dsl/bricks.jl:829-835`; spec at `src/kernels/li_bb_2d_v2.jl:49-53` | `CollideTRTDirectGuoField` in the fused LI-BB V2 Guo-field solvent path | I for the distribution update; `ux/uy` written by `WriteMoments` are collision-local half-step velocities, not a no-correction post-collision raw readout | Needs a dedicated convention contract. In the main log-FV driver it is later overwritten by G3, so the confirmed final-field bug is still G3. |

## (E) Test coverage gap

`dev-viscoelastic` does not contain `test/test_guo_convention_pairs.jl`; `rg --files test` found no equivalent Guo convention pair test. The closest tests are:

- `test/test_viscoelastic_logfv_patch_ladder.jl:1416-1435`, "M5b forced-field macroscopic velocity uses local Guo correction", which checks the log-FV getter alone and currently asserts the `+F/2` behavior from rest equilibrium.
- `test/test_viscoelastic_logfv_patch_ladder.jl:1848-1898`, "M8a modular LI-BB V2 Guo field canaries", which compares fused TRT Guo-field distributions against BGK for a forced field but does not check the macroscopic readout against `gx*N`.
- `test/test_viscoelastic_logfv_patch_ladder.jl:1924-1939`, "M8b BFS hydrodynamic Guo-field pipeline is bounded", which exercises the path but only checks finiteness and density bounds.

Missing coverage: a periodic-box convention-pair test for `collide_guo_field_2d! + logfv_compute_macroscopic_forced_field_2d!`, plus the production fused LI-BB/TRT Guo-field pair. The sentinel should show the broken pair as `gx*N + gx/2` after `N` steps and the corrected pair as `gx*N`.

## (F) Recommended fix

Audit-only: no source edit was applied.

Primary viscoelastic fix:

- Change `src/kernels/logconformation_fv_2d.jl:1047-1048`.
- Replace the current half-force readout with raw post-collision moments:

```julia
ux[i, j] = (f2 - f4 + f6 - f7 - f8 + f9) * inv_rho
uy[i, j] = (f3 - f5 + f6 + f7 - f8 - f9) * inv_rho
```

Required companion test update:

- Replace `test/test_viscoelastic_logfv_patch_ladder.jl:1416-1435` with a pair test, not a standalone getter test. The correct Convention-I production pair is `collide_guo_field_2d!` plus a no-correction readout.
- Add a regression sentinel showing that adding `+F/2` after `collide_guo_field_2d!` produces the `+gx/2` offset.

Additional non-viscoelastic fixes should be done only after separate pair tests:

- `src/kernels/macroscopic.jl:71-72` for 2D uniform forced readout paired with `collide_guo_2d!`.
- `src/kernels/macroscopic.jl:101-103` for 3D uniform forced readout paired with `collide_guo_3d!`.
- Pressure/VOF/phasefield getters listed in G4-G5 need domain-specific convention tests before edits because their density/pressure variables differ from the isothermal single-phase LBM path.

## (G) Refutation criterion

This conclusion is refuted if an executable convention test shows that, for `collide_guo_field_2d!` from rest on a fully periodic box with uniform `gx`, the no-correction raw readout gives `gx*N - gx/2` while `logfv_compute_macroscopic_forced_field_2d!` gives exactly `gx*N`. Equivalently, a one-cell/periodic analytical first-moment measurement after `collide_guo_field_2d!` would refute this verdict if the post-collision raw momentum is lower than physical velocity by `F/2` instead of equal to the physical next-step velocity.
