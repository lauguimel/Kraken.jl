# Next session prompt — Kraken viscoelastic log-FV cylinder (dev-viscoelastic)

Copy-paste everything below to start a fresh session.

---

Continue work on branch `dev-viscoelastic` of Kraken.jl
(worktree `~/Documents/Recherche/Kraken.jl-viscoelastic`).

## TL;DR — break the ratchet

The log-FV Oldroyd-B cylinder still diverges from the RheoTool reference as
mesh resolution increases. The previous 5+ sessions added **23 "follow-up
slices"** of incremental embedded-geometry coverage (`embedded_advection`,
`embedded_force`, `embedded_gradient`, `embedded_drag`, wall_fraction, normal
convention, wall_distance to centroid, etc.). Each slice passes its own
analytical canary. None has closed the RheoTool drift.

**The single most informative diagnostic listed in
`bench/viscoelastic_logfv/VALIDATION_LADDER_AUDIT_20260513.md:112-114` has
never been done : frozen-flow replay of a RheoTool U field through Kraken's
log-FV polymer pipeline.**

This session : **stop adding embedded slices. Do the replay.** It bisects
the bug space in one experiment.

---

## What is factually established

### Green analytical/Cartesian gates (M0–M5)

- `test/test_fvfd_operators_2d.jl` : 952/952 pass
- `test/test_logfv_frozen_channel_cde.jl` : all 64 checks pass
- `test/test_viscoelastic_logfv_patch_ladder.jl` : 14122/14122 pass
- Frozen channel (Couette/Poiseuille) τ recovery : max error ~4e-9
- Frozen analytical circle shear, Oldroyd-B and FENE-P : τ error ~4e-9
- Curved no-slip tangential shear convergence (R=6/10/14) :
  velocity-gradient error 6.45e-5 → 3.23e-5 → 1.71e-5, τ error O(1e-11)

### RheoTool comparison (Aqua A100 F64, job `21310647.aqua`, full embedded path)

| R  | Wi=0.05 Cd | err RheoTool | Wi=0.1 Cd | err |
|---:|---:|---:|---:|---:|
| 10 | 132.25 | +0.33% | 131.34 | +0.70% |
| 20 | 137.94 | +4.65% | 136.54 | +4.68% |
| 30 | 140.85 | **+6.86%** | 139.27 | **+6.78%** |

RheoTool reference (`bench/rheotool/sweep_wi_results.txt`) :
- Wi=0.05 → Cd = 131.813
- Wi=0.1  → Cd = 130.428

### The two signals the data sends

1. **Error grows with refinement** (anti-convergence). Not a sub-resolved
   scheme. It is an operator whose error amplifies with more cut-cells.
   R=10 has ~30 cut-cells, R=30 has 242, error rises ~20×.
2. **Error is essentially Wi-independent** (~+6.8% at R=30 across Wi
   values). It is NOT the polymer constitutive law, NOT λ-stiffness, NOT
   a Wi-dependent BC. It is a **near-wall geometric/coupling artifact**.

Combined : the bug is in the curved-wall handling of the polymer→momentum
chain (advection / source / wall extrapolation / divergence / drag
quadrature), and is masked at coarse R because few cells are affected.

## What was tried and did NOT close the drift (do not re-litigate)

- Disabling polymer source subcycling explanations (R8 Wi=0.5 canary :
  Cd flat across 8/32/100 substeps).
- `force_boundary_fill=:nearest` (changes fx_total only ~2%).
- Migrating `embedded_advection`, `embedded_force`, `embedded_drag`,
  `embedded_gradient` to the cut-cell geometry one by one.
- Switching `embedded_geometry` from `:qwall` to `:circle`, half-cell
  coordinate-frame shift, centroid-based wall-distance lowering.
- FENE-P vs Oldroyd-B model swap (drift signature identical).

## The diagnostic this session must do — frozen-flow replay

The only way to disambiguate the three remaining bug families :

1. Is the **polymer constitutive/CDE chain** wrong on the cut-cell
   geometry ?
2. Is the **wall traction integration** wrong (drag side) ?
3. Is the **LBM solvent / polymer-to-momentum coupling** wrong ?

Replay protocol :

1. Take a converged RheoTool velocity field `U(x,y)` from
   `bench/rheotool/cylinder_oldroydb_log_re1_wi01/` (the `0.4` / `0.8` /
   final timesteps in the case directory ; OpenFOAM VTK or
   `internalField` ASCII).
2. Resample onto a Kraken Cartesian grid at R=10/20/30 (linear
   interpolation, mark cells inside the cylinder as solid via `q_wall`).
3. **Freeze** that `U`. No LBM update, no momentum equation.
4. Run only the Kraken log-FV pipeline :
   - face-velocity lowering,
   - log-C advection,
   - log-C source (with the same numerical ∇U Kraken would compute),
   - stress reconstruction τ_p = G(C − I) (Oldroyd-B) or FENE-P map,
   - polymer force F_p = ∇·τ_p (for diagnostics, not fed back),
   - polymer drag (wall traction quadrature on the cylinder).
5. Compare to RheoTool the following, on the same grid :
   - `τ_p_xx(x,y)`, `τ_p_xy(x,y)`, `τ_p_yy(x,y)` fields (relative L∞
     and L2 errors, plus side-by-side plots),
   - `‖∇·τ_p‖` field,
   - integrated `Cd_polymer` and `Cd_bsd`,
   - profile at one near-wall slice and one wake slice.

Decision tree :

- **τ fields match within 1%, drag matches** → bug is in LBM solvent or
  the τ_p → f_polymer coupling on the LBM side (Hermite source factor,
  CNEBB, body-force scheme). Next : single-step coupling sanity check.
- **τ fields match within 1%, drag diverges** → bug is in the curved-wall
  traction integration. Next : audit
  `compute_polymeric_drag_2d` + `fvfd_embedded_wall_traction_2d!`,
  check normal-sign convention parity between embedded_gradient (slice
  14 fix) and drag quadrature.
- **τ fields disagree near the wall** → bug is in the polymer CDE pipeline
  on cut cells (advection scheme on cut faces, source from numerical
  ∇U near a curved wall, polymer wall BC). Next : isolate which of
  advection / source / wall closure is responsible by toggling each
  with frozen U.
- **τ fields disagree everywhere** → unlikely given clean analytical
  gates, but if it happens, the FV advection of the log-conformation
  field on a non-uniform-flow Cartesian grid is broken.

Deliverable for this session :

- `bench/viscoelastic_logfv/run_rheotool_frozen_replay_2d.jl`
- Saved fields and a `dashboard.html` per R, side-by-side τ Kraken / τ
  RheoTool / relative error.
- `bench/viscoelastic_logfv/RHEOTOOL_FROZEN_REPLAY_<date>.md` : verdict
  from the decision tree above, plus the NEXT concrete action.

## Two parallel low-effort sanity checks (5–10 min each)

Do these while the replay is running, NOT instead of it :

1. **2D source prefactor parity** for the log-FV path
   (`equations_cross_check.md` Finding 1).
   - `src/kernels/collide_viscoelastic_source_2d.jl:61` uses
     `pre = -ω * 9/2` (no `(1−ω/2)` division).
   - `src/kernels/viscoelastic_3d.jl:26` uses
     `pre = -s_plus * 9/2 / (1 − s_plus/2)`.
   - Verify whether the log-FV cylinder path still goes through the 2D
     direct-source kernel or via a separate regularized path. If through
     the direct source, this is a known unresolved discrepancy. Effect
     is uniform multiplicative on polymer stress, not mesh-divergent —
     so probably NOT the root cause, but worth confirming.

2. **Normal-sign parity between embedded_gradient and drag quadrature**.
   - Slice 14 (commit unknown, see `VALIDATION_LADDER_AUDIT_20260513.md`
     Fourteenth follow-up slice) fixed the circle wall normal to point
     from solid into fluid.
   - Drag is `∫ τ·n dA` on the cylinder. Confirm
     `compute_polymeric_drag_2d` and `fvfd_embedded_wall_traction_2d!`
     use the same normal convention. A flipped sign here would give
     **drag enhancement** (Cd above RheoTool, growing with sample
     count = R) which matches the observed signature.

## What NOT to do

- **Do not add a 24th embedded slice.** The slice-by-slice pattern has
  produced 23 attempts and zero convergence on the macro test. Either
  the diagnostic harness is wrong, or the architecture itself has a
  structural geometry inconsistency that incremental migration cannot
  fix. The replay above is what discriminates.
- **Do not launch another Aqua R-sweep before the replay verdict.** No
  new HPC jobs this session.
- **Do not trust the analytical canaries alone.** They are necessary but
  insufficient — they have been green for 23 slices while the macro
  test drifts.
- **Do not speculate the cause without a test that distinguishes it.**
  Previous sessions had 4 retractions from this pattern.
- **Do not write a 24-th follow-up slice in the audit file.** If you add
  to that file, it must be a verdict slice ("ruled in" or "ruled out"),
  not "more infrastructure added".

## Where everything lives

### Code
- Driver : `src/drivers/viscoelastic_logfv_2d.jl`
  (`run_coupled_viscoelastic_logfv_step_2d`, ~2870 lines)
- Cylinder bench harness :
  `bench/viscoelastic_logfv/logfv_cylinder_cd_convergence.jl`
- Frozen channel reference driver (template for replay) :
  `run_viscoelastic_logfv_frozen_channel_cde_2d`
- Frozen circle (analytical U) :
  `run_viscoelastic_logfv_frozen_circle_shear_cde_2d`
  `run_viscoelastic_logfv_frozen_circle_tangential_shear_cde_2d`
- log-FV kernel : `src/kernels/logconformation_fv_2d.jl`
- Embedded boundary lowering : grep for `FVFDEmbeddedBoundary2D` and
  `fvfd_geometry_from_circle_2d`

### RheoTool reference data
- Case : `bench/rheotool/cylinder_oldroydb_log_re1_wi01/`
- Sweep across Wi : `bench/rheotool/cylinder_wi{0.05,0.1,0.2,0.5,1.0}/`
- Cd values : `bench/rheotool/sweep_wi_results.txt`
- Time-series Cd : `Cd.txt` in each case directory
- Velocity field : OpenFOAM `<time>/U` files in each case
  (rheoTool with mirrorMesh and writeData/writeFields function objects)

### Audit history (READ-ONLY, do not duplicate work)
- `AUDIT_SUMMARY.md` — pre-log-FV LBM-direct path verdict
- `bench/viscoelastic_logfv/VALIDATION_LADDER_AUDIT_20260513.md` — 23
  slices, this is the file to NOT add a 24th slice to
- `bench/viscoelastic_logfv/RHEOTOOL_MESH_DIVERGENCE_AUDIT_20260511.md`
  — geometry-mismatch hypothesis (now partially refuted because
  enabling embedded everywhere did not close the drift)
- `bench/viscoelastic_audit/EQUATION_AUDIT_LIU_RHEOTOOL.md` and
  `bench/equations_cross_check.md` — Hermite prefactor finding
- `DEBUG_PLAN.md` — older M0–M5 plan (LBM-direct path, predates log-FV)

### Memory snapshot of relevant past sessions
- `project_viscoelastic_branch.md`
- `project_viscoelastic_audit.md`
- `feedback_cylinder_benchmark.md`

## Working hypothesis to enter the session with (kill or confirm)

**The polymer wall-traction drag integration is computing a positive bias
that scales with the number of cut-cell wall samples.** This explains :

- Cd above (not below) RheoTool.
- Bias growing with R (more samples).
- Bias Wi-independent at fixed geometry (drag is linear in τ; the bias
  is a quadrature/normal artifact, not a τ-magnitude artifact).
- All analytical canaries passing (they test τ recovery, not the curved
  wall-integral of τ·n).

The replay falsifies this cleanly : if τ matches but drag doesn't, the
hypothesis is confirmed. If τ doesn't match, the hypothesis is wrong and
we move into the polymer CDE pipeline on cut cells.

## Concrete first action this session

```julia
# 1. Read one RheoTool U field at t = 0.8 from
#    bench/rheotool/cylinder_oldroydb_log_re1_wi01/0.8/U  (OpenFOAM ascii)
# 2. Interpolate to a Kraken H=4R=120 by L=20R=600 grid at R=30.
# 3. Save as JLD2 once so we never re-parse OpenFOAM.
# 4. Build run_rheotool_frozen_replay_2d that takes the JLD2 and runs
#    log-FV polymer-only pipeline to steady state.
```

If parsing OpenFOAM ASCII is painful, fall back to writing a small
`writeData` function object in the RheoTool case that dumps `U`, `tau`,
`Cd_pos_neg` on a sampled grid to CSV, re-run the case (cheap : minutes
on Docker), then ingest the CSV.

End of prompt.
