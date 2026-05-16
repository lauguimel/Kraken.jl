# Cavity pipeline architectural audit (Wi → 0, 2026-05-16)

## Timestep pipeline outline

Line-number drift note: the brief's cavity line anchors were stale for the
current worktree. The wall-gradient helper is currently at
`src/drivers/viscoelastic_logfv_2d.jl:807-857`, and
`run_viscoelastic_logfv_cavity_coupled_2d` is currently at
`src/drivers/viscoelastic_logfv_2d.jl:885-1256`, shifted by an existing
`diagnose_bsd_dual` diagnostic.

- Step 0a (`src/drivers/viscoelastic_logfv_2d.jl:1050-1052`): enter the
  `for step in 1:max_steps` loop, set `completed_steps`, and compute
  `t_phys = step * dt_phys`.
- Step 0b (`src/drivers/viscoelastic_logfv_2d.jl:1053-1055`):
  `_logfv_cavity_update_lid_profile!` refreshes `u_lid_profile` for the
  moving north wall.
- Step 1 (`src/drivers/viscoelastic_logfv_2d.jl:1057-1061`):
  `logfv_cell_velocity_to_faces_bc_aware_2d!` writes `ux_face, uy_face`
  from the current `ux, uy` and `logfv_bc`.
- Step 2 (`src/drivers/viscoelastic_logfv_2d.jl:1063-1070`):
  `logfv_advect_upwind_bc_aware_2d!` writes `psixx_adv, psixy_adv,
  psiyy_adv` from the previous accepted `psixx, psixy, psiyy` using the
  just-built face velocities.
- Step 3a (`src/drivers/viscoelastic_logfv_2d.jl:1072-1075`):
  `fvfd_velocity_gradient_2d!` writes `dudx, dudy, dvdx, dvdy` from
  the current `ux, uy`, using the solid-aware derivative helpers in
  `src/fvfd/operators_2d.jl:947-977`.
- Step 3b (`src/drivers/viscoelastic_logfv_2d.jl:1076-1079`):
  `_logfv_cavity_apply_wall_gradient_correction!` overwrites wall-row
  entries of `dudx, dudy, dvdx, dvdy` using the moving-lid/fixed-wall
  half-cell formula implemented at `src/drivers/viscoelastic_logfv_2d.jl:807-857`.
- Step 4a (`src/drivers/viscoelastic_logfv_2d.jl:1081-1083`): the source
  update starts from the advected `Ψ` aliases `psixx_work, psixy_work,
  psiyy_work = psixx_adv, psixy_adv, psiyy_adv`.
- Step 4b (`src/drivers/viscoelastic_logfv_2d.jl:1083-1093`):
  each polymer substep calls `logfv_step_constitutive_log_2d!`, which reads
  the corrected `dudx, dudy, dvdx, dvdy` and advances `Ψ` through
  `src/kernels/logconformation_fv_2d.jl:417-457`.
- Step 4c (`src/drivers/viscoelastic_logfv_2d.jl:1094-1096`): the accepted
  `psixx, psixy, psiyy` buffers are swapped with the work buffers; the
  old accepted buffers become the next advection scratch buffers.
- Step 5 (`src/drivers/viscoelastic_logfv_2d.jl:1098-1102`):
  `logfv_stress_from_log_2d!` writes `tauxx, tauxy, tauyy` from the updated
  `psixx, psixy, psiyy`; the scalar stress formula is at
  `src/kernels/logconformation_fv_2d.jl:296-303`, and the kernel wrapper is
  at `src/kernels/logconformation_fv_2d.jl:459-494`.
- Step 6 (`src/drivers/viscoelastic_logfv_2d.jl:1104-1108`):
  `logfv_polymer_force_bc_aware_2d!` writes `fx_poly, fy_poly = div(tau_p)`
  using `fvfd_tensor_divergence_2d!`; the wrapper forwards
  `polymer_wall_extrap` at `src/kernels/logconformation_fv_2d.jl:649-658`
  and the FVFD divergence kernel is at `src/fvfd/operators_2d.jl:633-663`.
- Step 7a, kinetic branch (`src/drivers/viscoelastic_logfv_2d.jl:1110-1117`):
  if `bsd_kind === :kinetic`, compute `s_plus_t` and call
  `compute_bsd_force_kinetic_2d!` to write `fx_total, fy_total` from
  `fx_poly, fy_poly` plus a `Π^{neq}`-based BSD correction.
- Step 7b, default FD branch (`src/drivers/viscoelastic_logfv_2d.jl:1118-1123`):
  otherwise call `logfv_bsd_correct_force_bc_aware_2d!`, which writes
  `fx_total, fy_total = fx_poly, fy_poly - ζ*ν_p*lap(u)` via the narrow
  second-derivative operator at `src/fvfd/operators_2d.jl:886-933`.
- Step 7c, optional dual diagnostic (`src/drivers/viscoelastic_logfv_2d.jl:1125-1143`):
  if `diagnose_bsd_dual` is enabled, compute the inactive BSD path into
  `fx_alt, fy_alt` and push a relative L2 diagnostic. This does not change
  the body force consumed by the LBM step.
- Step 8 (`src/drivers/viscoelastic_logfv_2d.jl:1145-1149`):
  `fused_trt_libb_v2_guo_field_step!` consumes the final `fx_total, fy_total`
  and `nu_lbm_t = nu_s + ζ*ν_p`; its entrypoint is
  `src/kernels/li_bb_2d_v2.jl:115-127`.
- Step 9 (`src/drivers/viscoelastic_logfv_2d.jl:1151-1152`):
  `apply_bc_rebuild_2d!` rebuilds populations with Zou-He on the moving
  north wall and halfway bounce-back elsewhere.
- Step 10 (`src/drivers/viscoelastic_logfv_2d.jl:1154-1155`):
  `logfv_compute_macroscopic_forced_field_2d!` rewrites `rho, ux, uy` from
  `f_out` and the same `fx_total, fy_total`; the half-force velocity
  correction is at `src/kernels/logconformation_fv_2d.jl:1024-1062`.
- Step 11a (`src/drivers/viscoelastic_logfv_2d.jl:1157-1182`): optional
  finite diagnostics synchronize and inspect `rho, ux, uy, Ψ, τ_p, fx_poly,
  fy_poly`.
- Step 11b (`src/drivers/viscoelastic_logfv_2d.jl:1184-1197`): kinetic and
  elastic history sampling reads `ux, uy, tauxx, tauyy`.
- Step 11c (`src/drivers/viscoelastic_logfv_2d.jl:1199-1216`): snapshot
  capture copies `ux, uy, Ψ, τ_p` at requested sample steps.
- Step 12 (`src/drivers/viscoelastic_logfv_2d.jl:1218`): swap `f_in, f_out`
  so the next timestep reads the just-updated populations.

## D-buffer lifetime

`dudx`

- Allocation: zero-filled once at `src/drivers/viscoelastic_logfv_2d.jl:985`.
- Base write: Step 3a writes all cells at `src/drivers/viscoelastic_logfv_2d.jl:1073-1075`;
  the FVFD kernel writes zero on solid cells and otherwise calls
  `_fvfd_solid_bc_derivative_x_2d(ux, ...)` at `src/fvfd/operators_2d.jl:947-977`.
- Overwrite: Step 3b overwrites west/east wall columns at
  `src/drivers/viscoelastic_logfv_2d.jl:1076-1079`; the kernel assignments
  are at `src/drivers/viscoelastic_logfv_2d.jl:828-835`.
- Reads: Step 4 reads `dudx` in every polymer substep at
  `src/drivers/viscoelastic_logfv_2d.jl:1084-1088`, and the source kernel
  reads `dudx[i,j]` at `src/kernels/logconformation_fv_2d.jl:426-429`.
- Staleness/provenance: no current same-step stale read was found after
  Step 3b; however, the same array first contains the base solid-aware
  gradient and then the wall-corrected gradient, with no persistent
  `D_base` or `D_source` name.

`dudy`

- Allocation: zero-filled once at `src/drivers/viscoelastic_logfv_2d.jl:986`.
- Base write: Step 3a writes all cells at `src/drivers/viscoelastic_logfv_2d.jl:1073-1075`;
  the FVFD kernel calls `_fvfd_solid_bc_derivative_y_2d(ux, ...)` at
  `src/fvfd/operators_2d.jl:965-967`.
- Overwrite: Step 3b overwrites north/south wall rows at
  `src/drivers/viscoelastic_logfv_2d.jl:1076-1079`; the moving-lid and
  fixed-bottom assignments are at `src/drivers/viscoelastic_logfv_2d.jl:818-825`.
- Reads: Step 4 reads `dudy` at `src/drivers/viscoelastic_logfv_2d.jl:1084-1088`
  and `src/kernels/logconformation_fv_2d.jl:426-429`.
- Staleness/provenance: this is the most important wall-correction carrier
  for the cavity lid. Any same-stencil BSD term inserted before Step 3b would
  see `D_base`; the source ODE in the current source sees `D_source =
  D_base + wall overlay`.

`dvdx`

- Allocation: zero-filled once at `src/drivers/viscoelastic_logfv_2d.jl:987`.
- Base write: Step 3a writes all cells at `src/drivers/viscoelastic_logfv_2d.jl:1073-1075`;
  the FVFD kernel calls `_fvfd_solid_bc_derivative_x_2d(uy, ...)` at
  `src/fvfd/operators_2d.jl:968-970`.
- Overwrite: Step 3b overwrites west/east wall columns at
  `src/drivers/viscoelastic_logfv_2d.jl:1076-1079`; assignments are at
  `src/drivers/viscoelastic_logfv_2d.jl:828-835`.
- Reads: Step 4 reads `dvdx` at `src/drivers/viscoelastic_logfv_2d.jl:1084-1088`
  and `src/kernels/logconformation_fv_2d.jl:426-429`.
- Staleness/provenance: the buffer has two sequential meanings inside a
  timestep, but only the second meaning is consumed by the current source ODE.

`dvdy`

- Allocation: zero-filled once at `src/drivers/viscoelastic_logfv_2d.jl:988`.
- Base write: Step 3a writes all cells at `src/drivers/viscoelastic_logfv_2d.jl:1073-1075`;
  the FVFD kernel calls `_fvfd_solid_bc_derivative_y_2d(uy, ...)` at
  `src/fvfd/operators_2d.jl:971-973`.
- Overwrite: Step 3b overwrites north/south wall rows at
  `src/drivers/viscoelastic_logfv_2d.jl:1076-1079`; assignments are at
  `src/drivers/viscoelastic_logfv_2d.jl:818-825`.
- Reads: Step 4 reads `dvdy` at `src/drivers/viscoelastic_logfv_2d.jl:1084-1088`
  and `src/kernels/logconformation_fv_2d.jl:426-429`.
- Staleness/provenance: as with `dudy`, the corrected wall-row value is
  present before source integration in the current source.

Current-source conclusion: the brief's phrase "wall-gradient correction after
the source ODE has already consumed D" does not match this worktree. The actual
fault is not a same-step stale source read; it is an in-place overwrite that
erases whether downstream code is meant to consume `D_base`, wall-corrected
`D_source`, or a separately captured `D_BSD`.

## τ_p lifetime

`psixx, psixy, psiyy`

- Allocation and initial state: `psixx, psixy, psiyy` are zero-filled at
  `src/drivers/viscoelastic_logfv_2d.jl:971-974`, representing `C = I` and
  `Ψ = log(I) = 0`. Scratch buffers `*_adv` and `*_next` are allocated at
  `src/drivers/viscoelastic_logfv_2d.jl:975-980`.
- Step 2 read/write: advection reads the accepted `psixx, psixy, psiyy` and
  writes `psixx_adv, psixy_adv, psiyy_adv` at
  `src/drivers/viscoelastic_logfv_2d.jl:1063-1070`.
- Step 4 read/write: the source update starts from the advected fields at
  `src/drivers/viscoelastic_logfv_2d.jl:1081-1083`; each substep reads
  `psixx_work, psixy_work, psiyy_work` plus the D buffers and writes
  `psixx_next, psixy_next, psiyy_next` at
  `src/drivers/viscoelastic_logfv_2d.jl:1084-1093`.
- Step 4 overwrite/swap: after subcycling, `psixx, psixy, psiyy` are rebound
  to the work buffers at `src/drivers/viscoelastic_logfv_2d.jl:1094-1096`.
- Later reads: Step 5 reads accepted `Ψ` to reconstruct `τ_p` at
  `src/drivers/viscoelastic_logfv_2d.jl:1098-1102`; diagnostics, snapshots,
  and return values read it at `src/drivers/viscoelastic_logfv_2d.jl:1161-1174`,
  `src/drivers/viscoelastic_logfv_2d.jl:1199-1216`, and
  `src/drivers/viscoelastic_logfv_2d.jl:1246-1248`.
- History property: `Ψ_n` is not an algebraic alias of `D_n`. It is the
  result of repeated advection/source updates from all previous accepted
  states, with Step 4 reading the corrected `D` on the current step only.

`tauxx, tauxy, tauyy`

- Allocation: zero-filled once at `src/drivers/viscoelastic_logfv_2d.jl:982-984`.
- Step 5 write: `logfv_stress_from_log_2d!` writes `tauxx, tauxy, tauyy`
  at `src/drivers/viscoelastic_logfv_2d.jl:1098-1102`. The scalar formula is
  `tau = prefactor * (f*C - I)` with `C = exp(Ψ)` at
  `src/kernels/logconformation_fv_2d.jl:296-303`, and the kernel writes
  output cells at `src/kernels/logconformation_fv_2d.jl:459-494`.
- Step 6 read: `logfv_polymer_force_bc_aware_2d!` reads `tauxx, tauxy, tauyy`
  to produce `fx_poly, fy_poly` at `src/drivers/viscoelastic_logfv_2d.jl:1104-1108`.
- Later reads: finite diagnostics read all three stress buffers at
  `src/drivers/viscoelastic_logfv_2d.jl:1161-1174`; energy sampling reads
  `tauxx, tauyy` at `src/drivers/viscoelastic_logfv_2d.jl:1184-1197`;
  snapshots and return values read all three at
  `src/drivers/viscoelastic_logfv_2d.jl:1199-1216` and
  `src/drivers/viscoelastic_logfv_2d.jl:1249-1251`.
- History property: `τ_p,n` is reconstructed from `Ψ_n`, and `Ψ_n` has
  accumulated the full trajectory through Steps 2 and 4. At small but nonzero
  `Wi`, `τ_p,n` is therefore a lagged/history-bearing stress, not merely
  `2*ν_p*D_n`.
- Comparison to Step 4 D: Step 4 reads the current corrected D buffers at
  `src/drivers/viscoelastic_logfv_2d.jl:1084-1088`. Step 5 then builds `τ_p`
  from the evolved `Ψ`. A same-step `τ_BSD = 2*ζ*ν_p*D_now`, as available in
  `src/kernels/logconformation_fv_2d.jl:678-708`, is instantaneous; it does
  not include the same history unless it is coupled to the exact D capture
  point used by the source ODE or the Wi-zero path is made algebraic.

## F_body assembly

- `fx_poly, fy_poly` are allocated at `src/drivers/viscoelastic_logfv_2d.jl:990-991`
  and written in Step 6 at `src/drivers/viscoelastic_logfv_2d.jl:1104-1108`.
  They are `div(τ_p)`, where `τ_p` came from the history-bearing `Ψ`.
- `fx_total, fy_total` are allocated at `src/drivers/viscoelastic_logfv_2d.jl:992-993`.
  They are the only body-force arrays consumed by the fused LBM step at
  `src/drivers/viscoelastic_logfv_2d.jl:1145-1149` and
  `src/kernels/li_bb_2d_v2.jl:115-127`.
- Default `bsd_kind = :fd`: Step 7 calls
  `logfv_bsd_correct_force_bc_aware_2d!` at
  `src/drivers/viscoelastic_logfv_2d.jl:1118-1123`. That wrapper delegates at
  `src/kernels/logconformation_fv_2d.jl:710-719` to `fvfd_bsd_force_2d!`,
  whose kernel computes `fx_total = fx_poly - ζ*ν_p*lap(ux)` and
  `fy_total = fy_poly - ζ*ν_p*lap(uy)` at `src/fvfd/operators_2d.jl:886-915`.
  Inputs are `fx_poly, fy_poly`, current `ux, uy`, `is_solid`, `bsd_t`,
  `nu_p_t`, `dx, dy`, and `logfv_bc`. This path does not read `dudx, dudy,
  dvdx, dvdy`.
- Kinetic fallback `bsd_kind = :kinetic`: Step 7 calls
  `compute_bsd_force_kinetic_2d!` at
  `src/drivers/viscoelastic_logfv_2d.jl:1110-1117`. The kinetic wrapper reads
  `f_in, rho, ux, uy, is_solid`, allocates `Pi_xx, Pi_xy, Pi_yy`, extracts
  `Π^{neq}` from `f - f_eq` at `src/kernels/bsd_kinetic.jl:12-52`, and writes
  `fx_total, fy_total` through a divergence-like assembly at
  `src/kernels/bsd_kinetic.jl:54-120`. This path reads `Π^{neq}` from the LBM
  populations, not the D buffers.
- Optional dual diagnostic: `diagnose_bsd_dual` allocates `fx_alt, fy_alt`
  at `src/drivers/viscoelastic_logfv_2d.jl:994-997` and writes them only for
  path comparison at `src/drivers/viscoelastic_logfv_2d.jl:1125-1143`.
  The LBM force remains `fx_total, fy_total`.
- Macroscopic coupling detail: after the fused LBM step, Step 10 recomputes
  `ux, uy` using `f_out` and the same `fx_total, fy_total` half-force term at
  `src/drivers/viscoelastic_logfv_2d.jl:1154-1155` and
  `src/kernels/logconformation_fv_2d.jl:1043-1048`. The next timestep's
  Step 3 gradients therefore use post-Guo macroscopic velocities from the
  previous timestep's final body force.

## Wi → 0 consistency check

The design intent is:

```text
τ_p ≈ 2*ν_p*D + O(Wi)
F_body = div(τ_p) - ζ*ν_p*lap(u)
       → (1 - ζ)*ν_p*lap(u)
ν_LBM = ν_s + ζ*ν_p
```

For that cancellation to hold discretely, every occurrence of the Newtonian
polymer contribution must share the same velocity state, wall state, and
discrete operator.

- Step 3 requirement: the D used to advance `Ψ` must be the D that defines
  the Wi-zero Newtonian `τ_p`. Current source writes a base FVFD gradient at
  `src/drivers/viscoelastic_logfv_2d.jl:1073-1075`, then overwrites wall rows
  at `src/drivers/viscoelastic_logfv_2d.jl:1076-1079`. In the current source,
  Step 4 reads the post-correction D, so the same-step source read is not stale.
  Deviation: the buffer does not record whether downstream code is consuming
  base or wall-corrected D, which made M11-style insertion order ambiguous.
- Step 4 requirement: at strict Wi-zero, `Ψ` should collapse to the algebraic
  Newtonian stress associated with the same `D_source`. Current source instead
  advances `Ψ` through a finite-step recurrence at
  `src/kernels/logconformation_fv_2d.jl:164-183` and
  `src/kernels/logconformation_fv_2d.jl:417-457`. Deviation: `τ_p` at Step 5
  is `τ_p(Ψ_history)`, not an explicit `2*ν_p*D_source` field.
- Step 5 requirement: `τ_p` should equal `2*ν_p*D_source` to the expected
  Wi-zero accuracy. Current source builds `τ_p` from `exp(Ψ)` at
  `src/kernels/logconformation_fv_2d.jl:296-303` and
  `src/drivers/viscoelastic_logfv_2d.jl:1098-1102`. Deviation: any finite-Wi
  history or wall-row source history survives into `τ_p` before divergence.
- Step 6 requirement: `div(τ_p)` should apply the same discrete divergence
  that any BSD stress subtraction uses. Current source applies
  `fvfd_tensor_divergence_2d!` with `polymer_wall_extrap` at
  `src/drivers/viscoelastic_logfv_2d.jl:1104-1108`,
  `src/kernels/logconformation_fv_2d.jl:649-658`, and
  `src/fvfd/operators_2d.jl:633-663`.
- Step 7 FD requirement: the BSD subtraction must use the same operator as
  the Newtonian part of Step 6. Current FD-BSD uses
  `_fvfd_solid_bc_second_derivative_{x,y}_2d`, a narrow second derivative, at
  `src/fvfd/operators_2d.jl:77-125` and `src/fvfd/operators_2d.jl:886-915`.
  Deviation: M10's mismatch remains in the default path. `div(τ_p)` is
  generated by first derivative then first derivative, which yields a wide
  two-cell-spaced Laplacian plus discrete mixed terms in the Wi-zero interior,
  while FD-BSD subtracts a one-cell-spaced narrow Laplacian.
- Step 7 wall requirement: polymer divergence and BSD subtraction must use the
  same near-wall extrapolation contract. Current polymer divergence threads
  `polymer_wall_extrap` through `src/fvfd/operators_2d.jl:772-789`, but
  FD-BSD has no corresponding `polymer_wall_extrap` parameter at
  `src/fvfd/operators_2d.jl:917-933`. Deviation: even if bulk stencils were
  repaired, the wall-row cancellation would still have a BC-contract mismatch.
- Step 7 kinetic requirement: if `bsd_kind = :kinetic`, the BSD term should
  represent the same viscous contribution as the lattice collision. It reads
  `Π^{neq}` from `f_in` at `src/kernels/bsd_kinetic.jl:12-52`, so it avoids
  D-buffer ambiguity. Deviation: it still combines `fx_poly = div(τ_p)` from
  the history-bearing FVFD path with a kinetic BSD moment from the current LBM
  populations, so exact Wi-zero cancellation depends on `τ_p` already matching
  the lattice moment at the same time level and wall convention.
- Step 8-10 requirement: the force used for Guo collision and half-force
  macroscopic velocity must be identical. Current source satisfies this by
  passing `fx_total, fy_total` to Step 8 at
  `src/drivers/viscoelastic_logfv_2d.jl:1145-1149` and to Step 10 at
  `src/drivers/viscoelastic_logfv_2d.jl:1154-1155`.

M11 same-stencil fix assessment: building `τ_BSD = 2*ζ*ν_p*D` with
`logfv_bsd_stress_from_gradient_2d!` at `src/kernels/logconformation_fv_2d.jl:678-708`
and routing it through `fvfd_tensor_divergence_2d!` would close the M10 bulk
and wall divergence-stencil mismatch if it also used the same
`polymer_wall_extrap`. It would not by itself close the temporal-history
mismatch, because it subtracts an instantaneous `D_now` field from
`τ_p(Ψ_history)`. It also remains ambiguous unless the source ODE and BSD
construction are explicitly tied to the same named D capture point.

## Architectural fault map

- `src/fvfd/operators_2d.jl:886-915`: default FD-BSD subtracts a narrow
  `lap(ux), lap(uy)` while `src/fvfd/operators_2d.jl:633-663` differentiates
  `τ_p` through tensor divergence. At Wi → 0 this leaves
  `ν_p*(Lap_wide - ζ*Lap_narrow)` instead of the intended
  `(1 - ζ)*ν_p*Lap_same`.
- `src/fvfd/operators_2d.jl:77-125`: the BSD second-derivative helper has no
  `polymer_wall_extrap` choice, while `src/fvfd/operators_2d.jl:13-75` and
  `src/fvfd/operators_2d.jl:772-789` expose linear/quadratic first-derivative
  wall extrapolation to polymer divergence. At Wi → 0 the wall rows retain a
  BC-dependent residual even if the interior cancellation is repaired.
- `src/drivers/viscoelastic_logfv_2d.jl:1073-1079`: the D buffers are written
  once as base FVFD gradients and then overwritten in place by cavity wall
  corrections. Current source applies the overwrite before Step 4, not after,
  but it still collapses two D states into one name. At Wi → 0 an M11-style
  `τ_BSD(D_now)` can silently use the wrong side of the overlay unless the
  capture point is explicit.
- `src/drivers/viscoelastic_logfv_2d.jl:1081-1102`: Step 4 evolves `Ψ` from
  `Ψ_adv` plus current D, then Step 5 reconstructs `τ_p` from that evolved
  history. Any same-step BSD stress built from `D_now` is instantaneous.
  At small Wi the leftover is the part of `div(τ_p(Ψ_history))` not matched by
  `div(2*ζ*ν_p*D_now)`.
- `src/kernels/logconformation_fv_2d.jl:164-183`: the source step composes
  deformation and relaxation over finite substeps. This is correct for the
  log-conformation model, but the cancellation argument needs an algebraic
  Wi-zero stress or a BSD term captured at the same source update. Otherwise
  a finite-history O(Wi) stress is subtracted by a zero-history stabilization
  term.
- `src/drivers/viscoelastic_logfv_2d.jl:1118-1123`: the FD-BSD branch reads
  `ux, uy` directly and never reads `dudx, dudy, dvdx, dvdy`. At Wi → 0 this
  bypasses the wall-corrected D that fed `Ψ`, producing a velocity-Laplacian
  correction that is not sourced from the same D buffer as the polymer stress.
- `src/kernels/bsd_kinetic.jl:103-109`: the kinetic BSD path allocates and
  extracts `Π^{neq}` inside each force assembly call. This is not the default
  fault, but architecturally it means the fallback has a third provenance:
  current `f_in` kinetic moment rather than `D_source` or `τ_p(Ψ_history)`.
  At Wi → 0 it can remove a lattice-viscous component that is not identical
  to the FVFD polymer divergence near walls.
- `src/drivers/viscoelastic_logfv_2d.jl:1154-1155`: macroscopic velocity is
  recomputed after collision with the final force and is the velocity used by
  the next timestep's Step 3. This is a consistent Guo half-step pattern, but
  it means any force assembly inconsistency at step n is immediately embedded
  in `u_{n+1}` and then in `D_{n+1}`, feeding the next `Ψ` history.
- `src/drivers/viscoelastic_logfv_2d.jl:1125-1143`: `diagnose_bsd_dual`
  compares FD and kinetic paths but does not compare either path to the
  same-stencil `τ_BSD` divergence proposed in M10. At Wi → 0 it can quantify
  FD-vs-kinetic disagreement, but not the missing `D_source` vs `D_BSD`
  provenance invariant.

## Recommended refactor

Option 1: introduce separate `D_source` and `D_BSD` buffer ownership.

- Shape: keep the existing `dudx, dudy, dvdx, dvdy` as `D_source`, add
  `dudx_bsd, dudy_bsd, dvdx_bsd, dvdy_bsd` or a compact `D_BSD` tuple, and
  populate it at the same timestep and wall-correction point as `D_source`.
- Files to touch: `src/drivers/viscoelastic_logfv_2d.jl` for buffer allocation
  and pipeline order; likely `src/kernels/logconformation_fv_2d.jl` only if a
  small force-subtract helper is added.
- Estimated diff size: about 45-70 LOC plus four persistent `N x N` buffers,
  or about 70-90 LOC if a named subtraction helper and diagnostic norm are
  added.
- Expected Wi → 0 behaviour: removes the ambiguous D provenance and lets a
  same-stencil `τ_BSD` use exactly the same corrected wall gradient as the
  source ODE.
- Regression risk: medium in cavity, low-to-medium in channel/cylinder if the
  helper is kept cavity-local first. Extra memory is small but GPU allocation
  count must stay outside the timestep loop.

Option 2: defer wall-gradient correction so both source ODE and BSD
construction see post-correction D in one ordered block.

- Shape: compute base D, apply wall correction, immediately call both the
  source ODE and `τ_BSD` construction from that corrected D before any later
  overwrite.
- Files to touch: primarily `src/drivers/viscoelastic_logfv_2d.jl`; no FVFD
  operator changes if the same-stencil divergence is already available.
- Estimated diff size: about 30-55 LOC plus five BSD stress/force buffers
  (`tau_bsd_xx, tau_bsd_xy, tau_bsd_yy, fx_bsd, fy_bsd`) if not already
  present in this driver.
- Expected Wi → 0 behaviour: closes the current source-vs-BSD capture
  ambiguity and the M10 stencil mismatch if the BSD stress is routed through
  `fvfd_tensor_divergence_2d!` with the same `polymer_wall_extrap`.
- Regression risk: medium. It changes finite-Wi stabilization timing in the
  cavity. Channel and cylinder risk depends on whether the same helper is
  generalized beyond the cavity path.

Option 3: compute `τ_BSD = 2*ζ*ν_p*D` at the same source-capture point.

- Shape: directly after Step 3b, while the corrected D buffers are the exact
  inputs to Step 4, call `logfv_bsd_stress_from_gradient_2d!`; later, in Step
  7, compute `F_BSD = div(τ_BSD)` through `fvfd_tensor_divergence_2d!` using
  the same `polymer_wall_extrap`, then write `fx_total = fx_poly - fx_bsd`
  and `fy_total = fy_poly - fy_bsd`.
- Files to touch: `src/drivers/viscoelastic_logfv_2d.jl` for five persistent
  buffers and branch wiring; optionally `src/kernels/logconformation_fv_2d.jl`
  for a tiny `force_difference!` helper if direct elementwise subtraction is
  not already available.
- Estimated diff size: about 50-85 LOC plus five persistent `N x N` buffers.
  If a GPU-safe subtract kernel is missing and must be added, expect the upper
  end of that range.
- Expected Wi → 0 behaviour: closes M10's same-stencil fault and binds BSD
  construction to the same corrected D field read by the source ODE on the
  same step. It does not erase genuine finite-Wi `Ψ_history` effects, but it
  removes the Wi-independent stencil/provenance residual.
- Regression risk: medium. It changes cavity force assembly materially. For
  channel and cylinder, risk is acceptable only if the change is opt-in or
  guarded by the existing BSD path until local FVFD canaries and low-Wi macro
  checks pass.

RECOMMENDED: Option 3.

Option 3 is the minimum-scope architectural refactor that addresses both
layers exposed by this audit: it replaces the default narrow-laplacian BSD
with the same tensor-divergence operator that generates `F_poly`, and it
captures `τ_BSD` at the exact point where the source ODE receives the corrected
D buffers. It should be implemented cavity-local first, with explicit buffer
names (`D_source` or `D_BSD` in comments/API, not another overloaded scratch
array) and with `polymer_wall_extrap` threaded through the BSD divergence. The
expected M16 size is not a tiny one-line fix; budget roughly 50-85 LOC plus
five persistent work arrays and a focused low-Wi matched-viscosity validation.
