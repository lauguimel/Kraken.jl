# Log-FV validation ladder audit - 2026-05-13

## Reason for this audit

The Oldroyd-B log-FV cylinder still diverges from the RheoTool reference as
mesh resolution increases. The last R30 Weissenberg sweep was useful as a
symptom check, but it was premature as a debugging step: the intermediate
analytical, semi-analytical, and simple-flow gates have not all been closed
quantitatively.

## Commands re-run locally

All three current local gates are green:

| Gate | Command | Result |
| --- | --- | --- |
| FVFD 2D operator library | `julia --project=. test/test_fvfd_operators_2d.jl` | 886/886 pass |
| Log-conformation 3D primitives | `julia --project=. test/test_logconformation_3d.jl` | 48/48 pass |
| Log-FV polymer patch ladder | `julia --project=. test/test_viscoelastic_logfv_patch_ladder.jl` | 14087/14087 pass |

The simple output harness also has historical green runs:

| Output set | Backend | Cases | Status |
| --- | --- | --- | --- |
| `tmp/logfv_simple_validation_outputs/20260511_fvfd_modular_local` | Metal Float32 | Poiseuille, square periodic, square channel, BFS | 4/4 pass |
| `tmp/logfv_simple_validation_outputs/20260511_fvfd_modular_a100` | CUDA Float64 | Poiseuille, square periodic, square channel, BFS | 4/4 pass |

Important limitation: those simple outputs mostly check boundedness, finite
fields, SPD conformation, density/speed ceilings, and one Poiseuille profile
error. They do not yet prove quantitative convergence of `u`, `rho`, `C`,
`tau`, `div(tau)`, wall force, or outlet profiles for square/BFS/RheoTool-like
flows.

## Ladder status

| Stage | Intended role | Current status |
| --- | --- | --- |
| M0 algebra | log/exp SPD identities | Green |
| M1 relaxation | pure Oldroyd-B and FENE-P local relaxation | Green |
| M2 homogeneous source | simple shear/stretching local source | Green |
| M3 advection | constant/affine fields with modular BCs | Green |
| M4 Poiseuille | analytical conformation, stress, force, BSD | Green |
| M5 Couette | analytical conformation, stress, force, BSD | Green |
| M6 square obstacle | first Cartesian obstacle flow | Only bounded/SPD, not a quantitative reference gate |
| M7 coupled square | simple coupled obstacle feedback | Only bounded/near-Newtonian checks |
| M8 BFS | open-x complex simple flow | Operator masks and bounded coupled checks, no RheoTool/profile comparison |
| M9 cylinder | curved-wall macro canary | Used too early for RheoTool convergence diagnosis |

The missing layer is therefore not the low-level analytical portage. The
missing layer is the quantitative semi-analytical/simple-flow isolation between
M5 and the RheoTool cylinder.

## Findings

1. The M0-M5 analytical path is real and currently green.

   The code validates local SPD algebra, exact relaxation, homogeneous shear,
   advection, analytical Couette/Poiseuille conformation, analytical stress,
   polymer force, and BSD force on Cartesian channels.

2. The simple macro cases are currently smoke tests, not bug isolation tests.

   `run_simple_validation_outputs.jl` writes useful VTK/dashboard outputs for
   Poiseuille, square periodic, square channel, and BFS. However, square and BFS
   are accepted by finite/bounded criteria. They do not compare against
   analytical, semi-analytical, or RheoTool/OpenFOAM profiles.

3. The strongest current bug hypothesis remains inconsistent wall geometry in
   the curved-wall log-FV path.

   The cylinder solvent/drag path uses `q_wall` cut-link geometry. The FVFD
   library now lowers `q_wall` into `FVFDEmbeddedBoundary2D`, but production
   coupling still consumes it only for the optional embedded velocity gradient.
   The advection, tensor divergence, BSD correction, and most field BC handling
   still operate through `is_solid` staircase geometry. Drag then goes back to
   `q_wall`.

   That means the coupled cylinder can mix:

   - cut-link geometry for LBM solvent and drag;
   - staircase geometry for polymer advection and force;
   - optional embedded geometry for only `grad(u)`.

   This is consistent with an error that gets worse when the cylinder boundary
   is refined: more cut-link information exists, but the polymer operator path
   does not use it coherently.

4. There is a separate direct-solver 2D/3D stress-source inconsistency.

   Actual source today:

   - `src/kernels/collide_viscoelastic_source_2d.jl`: `pre = -omega * 9/2`
   - `src/kernels/viscoelastic_3d.jl`: `pre = -s_plus * 9/2 / (1 - s_plus/2)`

   `bench/equations_cross_check_kraken_side.md` currently claims the 2D source
   has the division, but the source file does not. This probably does not
   explain the current log-FV R30 sweep by itself, because that path uses
   FV polymer force coupling rather than the old Hermite direct source. It does
   need to be fixed or explicitly justified before claiming direct 2D/3D parity.

5. The FENE-P R30 sweep did not isolate the bug.

   The latest H100 diagnostic compared FENE-P runs to Oldroyd-B RheoTool
   references. It showed that finite extensibility does not remove the drift,
   but it is not a valid model-to-model validation until matching FENE-P
   RheoTool references are generated.

## What was not done yet

The following expected gates are still missing:

- frozen simple-flow replay from RheoTool/OpenFOAM velocity fields into Kraken
  log-FV polymer CDE, without LBM feedback;
- quantitative BFS or square-channel comparison against RheoTool profiles;
- grid convergence of `u`, `rho`, `C`, `tau`, `div(tau)`, and force on simple
  cases before the cylinder;
- cut-wall `q_wall` canaries where every FVFD operator consumes the same
  embedded geometry, not just `grad(u)`;
- a coupled simple-flow RheoTool comparison before returning to the cylinder.

## Recommended next sequence

1. Freeze cylinder RheoTool sweeps as diagnostics only.

2. Promote the existing channel tests into explicit convergence dashboards:
   Couette and Poiseuille, multiple grids, profiles for `u`, `rho`, `C`,
   `tau`, `div(tau)`, and total force against the analytical solution.

3. Add a frozen-flow FVFD replay harness:
   prescribed `u` first, then imported RheoTool/OpenFOAM `U` through
   `fluidfoam`; solve only the polymer CDE and compare `C/tau` profiles.

4. Add `q_wall`-consistent operator canaries:
   the same embedded geometry must feed advection, velocity gradients,
   tensor divergence, BSD, cell volume/area weights, and force diagnostics.

5. Only after those pass, run a coupled square/BFS RheoTool comparison.

6. Only after the simple coupled comparison is clean, return to R30/R35
   cylinder convergence in Oldroyd-B and then FENE-P.

## Immediate conclusion

The current state is not "we validated everything before RheoTool". It is:

- analytical local/cartesian validation is green through M5;
- simple macro outputs exist and are useful, but are too weak;
- the likely unresolved defect is at the geometry/BC interface for curved
  walls, not in generic subcycling;
- the next useful work is a quantitative simple-flow and frozen-flow ladder,
  plus coherent `q_wall` use across all FVFD operators.

## Implementation note

First follow-up slice added after this audit:

- `run_viscoelastic_logfv_frozen_channel_cde_2d` replays analytical Couette and
  Poiseuille velocity fields through FVFD face velocities, FVFD advection, the
  log-C source, `tau_p`, `div(tau_p)`, and BSD diagnostics without LBM feedback.
- `test/test_logfv_frozen_channel_cde.jl` is the local regression canary.
- `bench/viscoelastic_logfv/run_quantitative_simple_ladder.jl` writes
  `summary.csv`, profile CSVs, and `dashboard.html` for the quantitative
  frozen-channel ladder.

Second follow-up slice:

- `FVFDEmbeddedBoundary2D` now carries per-cell west/east/south/north face
  aperture fractions lowered from `q_wall` half-plane geometry.
- `fvfd_cell_velocity_to_faces_embedded_2d!` applies those apertures to
  velocity face lowering, giving advection an explicit embedded-geometry entry
  point instead of only the staircase `is_solid` path.
- `test/test_fvfd_operators_2d.jl` checks the aperture values, the FVFD and
  log-FV wrappers, and a local Metal smoke compiled with lowered device BCs.

Third follow-up slice:

- `fvfd_transfer_field_bc_2d` is now the shared host-to-backend lowering path
  for `FVFDFieldBC2D`. It validates active open sides, converts values to the
  requested floating type, and returns concrete backend vectors.
- The q_wall embedded face-velocity Metal smoke now uses lowered field BCs
  rather than accidental host vectors, closing the immediate CPU/GPU BC
  portability gap found during the aperture canary.

Fourth follow-up slice:

- `fvfd_advect_upwind_embedded_2d!` and
  `fvfd_sym2_advect_upwind_embedded_2d!` now compose embedded face-velocity
  lowering with the regular FVFD upwind kernels through preallocated face
  arrays.
- The new constant-preservation canary verifies that scalar and sym2/log-FV
  fields remain unchanged while q_wall apertures modify the local face fluxes.

Fifth follow-up slice:

- `FVFDEmbeddedBoundary2D` now carries `wall_fraction`, a per-cell embedded
  wall segment length proxy derived from the imbalance of west/east and
  south/north face apertures.
- This is the missing geometric scalar needed before implementing embedded
  tensor divergence and wall traction/drag canaries; it keeps the wall area
  source in the shared lowering layer rather than in a benchmark driver.

Sixth follow-up slice:

- `fvfd_tensor_divergence_embedded_2d!` adds a conservative embedded stress
  divergence path using cut-cell volume fractions, face apertures, and wall
  closure terms.
- The first canary checks a constant stress field through an oblique q_wall
  cut: both FVFD and log-FV wrappers return zero divergence in the cut cell,
  which would fail if Cartesian aperture fluxes leaked through that cell
  without the wall closure.

Seventh follow-up slice:

- `fvfd_geometry_from_halfplane_2d` now builds a coherent analytical
  multi-cell embedded wall from one half-plane.
- The new canary verifies that embedded tensor divergence of a constant stress
  field is zero across all fluid cells of that coherent cut geometry, with a
  local Metal smoke when Metal is available. This is the next required gate
  before promoting embedded divergence into the coupled cylinder path.

Eighth follow-up slice:

- `fvfd_embedded_wall_traction_2d!` adds a lowered embedded-wall traction
  operator for `tau * n` integrated over each cut-cell wall segment.
- The half-plane canary now checks the analytical total traction on CPU and,
  when available, local Metal. This gives a lower-level force-integration gate
  before using polymer drag from a curved macro benchmark as evidence.

Ninth follow-up slice:

- `fvfd_geometry_from_circle_2d` adds a coherent analytical curved-wall
  lowering for the fluid outside a circle. It computes matching face apertures
  on neighboring cells and derives the embedded wall normal from the closed
  local surface vector.
- The new circle canary verifies circumference, zero divergence of a constant
  tensor stress on all fluid cells, and zero total traction under constant
  stress. This is the curved analogue to the half-plane gate and is still
  below any RheoTool/cylinder flow comparison.
- Diagnostic check on the existing direct `q_wall` cylinder lowering is not
  green as a FVFD curved-wall source of truth: on a local R=10 cylinder, the
  q_wall-derived wall length is about 21.14 versus the analytical 62.83 and a
  constant tensor stress leaks O(1) through embedded divergence. This likely
  comes from mixing LBM cut-link geometry, whose nodes live at `(i-1,j-1)`,
  with FVFD cell-centered aperture geometry. The analytical circle lowering is
  therefore the current curved FVFD canary; direct q_wall-to-FVFD promotion
  still needs a dedicated lowering fix.

Tenth follow-up slice:

- The coupled log-FV step-channel driver now has explicit opt-in controls:
  `embedded_geometry=:qwall|:circle`, `embedded_advection=true|false`,
  `embedded_force=true|false`, `embedded_drag=true|false`, and
  `embedded_circle_samples`.
- Defaults are unchanged. A local one-step cylinder smoke with
  `embedded_geometry=:circle`, `embedded_advection=true`,
  `embedded_gradient=true`, `embedded_force=true`, and `embedded_drag=true`
  compiles and runs, exposing `embedded_wall_length` and `embedded_cut_count`
  in the result for diagnostics.
- The smoke is now in the patch ladder as M9c, so future edits must keep the
  coherent circle embedded force path compiling and bounded before any macro
  RheoTool comparison is trusted.

Eleventh follow-up slice:

- `embedded_drag=true` now makes `Cd_p` and `Cd_bsd` use FVFD embedded wall
  traction instead of the legacy q_wall polymer drag quadrature. This keeps
  advection, force, and drag diagnostics on the same FVFD circle geometry when
  all embedded switches are enabled.
- A short local five-step cylinder diagnostic is bounded but not yet a
  validation: legacy q_wall diagnostics gave `Cd≈532.91`, `Cd_p≈13.99`,
  `Cd_bsd≈18.02`, `max|F_p|≈4.7e-5`, while the full embedded circle path gave
  `Cd≈694.11`, `Cd_p≈67.84`, `Cd_bsd≈112.94`, `max|F_p|≈3.05e-2`. This
  confirms the geometry/force choice is dynamically significant and must go
  through simpler curved/frozen-flow canaries before any RheoTool claim.

Twelfth follow-up slice:

- `run_viscoelastic_logfv_frozen_circle_shear_cde_2d` adds an analytical
  imposed-velocity canary on the coherent FVFD circle geometry. It prescribes
  simple shear, initializes the exact Oldroyd-B or FENE-P steady conformation,
  passes it through embedded FVFD log-field advection, the log-C source, and
  stress reconstruction, then reports `u`, `Psi`, `C`, `tau`, and max errors
  against the analytical `tau`.
- The gradient supplied to this canary is analytical by construction. This
  intentionally isolates constitutive/log-FV and embedded advection from the
  embedded no-slip velocity-gradient operator, because global affine shear does
  not satisfy stationary no-slip on an internal circle.
- Local 32x32 R=6 results with `lambda=3`, `shear_rate=0.012`, `dt=0.01`:
  CPU Float64 gives zero embedded-advection error and `max_tau_error≈4e-9`
  for both Oldroyd-B and FENE-P (`L_max=8`); local Metal Float32 gives the
  same qualitative result with `max_tau_error≈4.2e-9`.

Thirteenth follow-up slice:

- `run_viscoelastic_logfv_frozen_channel_cde_2d` now returns the numerical
  velocity-gradient fields `dudx/dudy/dvdx/dvdy` and max errors against the
  analytical Couette/Poiseuille gradients. The local channel canary therefore
  validates `U -> numerical gradient -> log-C source -> tau`, not only
  `U -> tau`.
- `test/test_logfv_frozen_channel_cde.jl` adds a coherent embedded half-plane
  shear canary with a velocity field that is exactly zero on the internal
  plane wall. This exercises the embedded no-slip gradient correction with a
  compatible field, then feeds that numerical gradient into the log-C source
  and checks the recovered Oldroyd-B `tau`.
- Local Metal Float32 smoke: channel Poiseuille gives
  `max_velocity_gradient_error≈1.0e-8`; the half-plane embedded-gradient smoke
  gives `max_dudy_error≈9.3e-9`.

Fourteenth follow-up slice:

- The analytical circle lowering now stores wall normals pointing from the
  solid into the fluid, matching the half-plane convention. Before this fix,
  the circle normal was inward, which made the embedded no-slip gradient
  correction use the wrong normal-derivative sign on curved walls.
- `run_viscoelastic_logfv_frozen_circle_tangential_shear_cde_2d` adds a curved
  no-slip imposed-velocity canary: `u = shear_rate * (r - R) e_theta`, a
  center-mask solid region compatible with the LBM storage model, embedded
  numerical gradients, analytical gradient comparison, and local Oldroyd-B
  source/stress recovery from the numerical gradient.
- Local 64x64 R=10 results with `lambda=2`, `shear_rate=0.006`, `dt=0.001`:
  CPU Float64 gives `max_velocity_gradient_error≈3.23e-5`,
  `max_cut_velocity_gradient_error≈3.18e-5`, and `max_tau_error≈2.83e-11`;
  local Metal Float32 gives the same gradient error and `max_tau_error≈1.2e-8`.

Fifteenth follow-up slice:

- Coupled cylinder results now include `embedded_normal_radial_min`,
  `embedded_normal_radial_mean`, and `embedded_normal_radial_samples` for
  `embedded_geometry=:circle`, and M9c asserts outward radial alignment. This
  turns the curved normal convention into a coupled-driver diagnostic, not only
  a standalone FVFD lowering test.
- Short local five-step diagnostic after the normal fix, R=3/H=16,
  `u_mean=0.005`, `lambda=5`, `nu_s=0.08`, `nu_p=0.02`: legacy q_wall path
  gives `Cd≈539.11`, `Cd_p≈13.30`, `Cd_bsd≈22.36`,
  `max|F_p|≈4.70e-5`; full embedded circle gives `Cd≈538.97`,
  `Cd_p≈43.46`, `Cd_bsd≈55.44`, `max|F_p|≈1.09e-3`,
  `normal_min≈0.9994`. This is still a smoke, not a RheoTool validation, but
  the previous force blow-up from the inward normal is gone.

Sixteenth follow-up slice:

- The curved no-slip tangential-shear canary now has a local mesh-convergence
  gate below RheoTool. With comparable domain/radius ratios, R=6/10/14 gives
  `max_velocity_gradient_error≈6.45e-5`, `3.23e-5`, and `1.71e-5`,
  respectively, while `max_tau_error` remains O(3e-11).
- This confirms that the embedded curved-gradient error decreases under
  refinement in the standalone FVFD/log-C path. Macro cylinder convergence can
  now be debugged against this lower-level trend rather than treated as a
  first diagnostic.

Seventeenth follow-up slice:

- A short coupled embedded-circle smoke across R=3/4/5 with matched
  channel-height scaling remains bounded for five local CPU steps:
  `first_nonfinite_step=0` in all cases, `normal_min>0.999`, wall-length
  relative error falls from about 0.4% to about 0.22%, and
  `max|F_p|` stays O(1e-3) or below.
- The corresponding `Cd` values are not convergence evidence because the runs
  are only five startup steps, but they are useful as a post-normal-fix
  stability sentinel: R=3/4/5 gives `Cd≈538.97/551.46/558.87`.
- Local Metal Float32 smoke of the R=3 embedded-circle coupled path for two
  steps also completes with `first_nonfinite_step=0`, `normal_min≈0.9994`,
  and finite fields. Longer GPU/RheoTool sweeps should still run on Aqua after
  this local sentinel stays green.

Eighteenth follow-up slice:

- `logfv_cylinder_cd_convergence.jl` now accepts explicit embedded-geometry
  controls via environment variables:
  `KRAKEN_LOGFV_EMBEDDED_GEOMETRY`, `KRAKEN_LOGFV_EMBEDDED_ADVECTION`,
  `KRAKEN_LOGFV_EMBEDDED_FORCE`, `KRAKEN_LOGFV_EMBEDDED_DRAG`, and
  `KRAKEN_LOGFV_EMBEDDED_CIRCLE_SAMPLES`. The CSV now records those switches,
  wall length, cut-cell count, and radial normal-alignment diagnostics.
- Added `run_embedded_circle_postfix_a100.pbs`, a short A100 sweep that runs
  Oldroyd-B then FENE-P with the full embedded-circle path, saves fields/VTK,
  and reports MLUPS through the existing harness.
- Submitted Aqua job `21295568.aqua` on 2026-05-13. Initial PBS state was
  queued in `gpu_batch_exec`; expected output root is
  `results/viscoelastic_logfv/embedded_circle_postfix_21295568.aqua/`.

Nineteenth follow-up slice:

- Aqua job `21295568.aqua` exited cleanly but every case failed before doing
  useful work because the synced driver called
  `logfv_cell_velocity_to_faces_embedded_2d!` while the synced kernel/API did
  not yet expose that wrapper. Local `Kraken` now exports the wrapper, and the
  relevant kernel/driver files were resynced to Aqua, but the user asked to run
  the next step locally instead of relaunching the HPC sweep.
- Local Metal Float32 sweeps then found the real coupled embedded-force leak:
  Oldroyd-B and FENE-P both made `rho` non-finite by the first diagnostic
  (`step=20`) for all viscoelastic cases, while Newtonian cases stayed finite.
  A smaller isolation showed `embedded_force=true` alone triggered the failure
  (`step=2`), whereas the same run with embedded advection/gradient but legacy
  force stayed finite.
- Root cause: `embedded_geometry=:circle` was built in the standalone FVFD
  cell-centered coordinate frame, whose control volumes are centered at
  `(i-0.5,j-0.5)`, but the coupled LBM cylinder `q_wall/is_solid` mask is
  node-centered at `(i-1,j-1)`. This half-cell mismatch produced LBM-fluid
  cells with FVFD `cell_fraction=0`, so the embedded tensor-divergence force
  divided by an almost-zero volume and injected a huge polymer force.
- The coupled circle lowering now shifts the analytical FVFD circle center by
  `+0.5dx,+0.5dy` only in the coupled driver, keeping standalone FVFD circle
  tests unchanged. Coupled results now also report
  `embedded_min_fluid_cell_fraction` and
  `embedded_zero_fluid_cell_fraction_count`; M9c runs five steps and asserts no
  zero-volume LBM-fluid cell.
- Post-fix local checks:
  `test/test_fvfd_operators_2d.jl` passes `948/948`,
  `test/test_logfv_frozen_channel_cde.jl` passes all `64` checks, and
  `test/test_viscoelastic_logfv_patch_ladder.jl` passes `14121/14121`.
  Local Metal R=6/10/14, Wi=0.001/0.05/0.1 sweeps for both Oldroyd-B and
  FENE-P finish with `12/12` `ok`, `first_bad_max=0`,
  `embedded_min_fluid_cell_fraction>=0.5107421875`, and zero zero-volume fluid
  cells. These are short startup sweeps, not converged Cd comparisons.

Twentieth follow-up slice:

- Added and submitted `run_embedded_circle_convergence_anygpu.pbs` for the
  first post-fix Aqua convergence sweep without pinning a GPU model. The PBS
  requests `select=1:ncpus=8:ngpus=1:mem=96GB` and deliberately omits
  `gpu_id` so the scheduler can use any available GPU.
- Submitted Aqua job `21310364.aqua` on 2026-05-14. Initial state was queued in
  `gpu_batch_exec`. Defaults: `R=10,20,30`,
  `Wi=0.001,0.05,0.1`, `steps_low_wi=50000`, `steps=100000`,
  `step_cap=100000`, full embedded circle path, Oldroyd-B then FENE-P,
  fields/VTK enabled, output root
  `results/viscoelastic_logfv/embedded_circle_convergence_anygpu_21310364.aqua/`.

Twenty-first follow-up slice:

- Aqua job `21310364.aqua` finished on 2026-05-14 with `Exit_status=0`, but
  the full embedded path (`embedded_gradient=true`) had four near-Newtonian
  failures: R=20/R=30, Wi=0.001 for both Oldroyd-B and FENE-P. All Newtonian,
  Wi=0.05, and Wi=0.1 cases completed. The failed dumps show global NaNs by
  the first 1000-step diagnostic, with first reported field `rho` at `(1,1)`.
  FVFD volume diagnostics stayed clean (`zeroVol=0`), so this is not the
  previous zero-fluid-volume bug.
- Local Metal isolation of R=20/Wi=0.001 showed the actual trigger earlier:
  `embedded_gradient=true` fails within O(10-50) steps near the cylinder,
  while `embedded_advection=true` and `embedded_force=true` without embedded
  gradient remain finite. The legacy gradient path is stable. This points to
  the embedded no-slip velocity-gradient correction as the next lower-level
  bug to close before promoting full embedded gradients.
- Submitted and completed Aqua job `21310394.aqua` with
  `KRAKEN_LOGFV_EMBEDDED_GRADIENT=0` and no `gpu_id` constraint, keeping
  embedded advection/force/drag active. Both Oldroyd-B and FENE-P completed
  all 12 rows with no case errors.
- Gradient-off convergence snapshot against RheoTool references:
  Oldroyd-B: at Wi=0.05, Cd R=10/20/30 is
  `131.6446/136.5961/139.0622` with RheoTool mean error
  `-0.13%/+3.63%/+5.50%`; at Wi=0.1, Cd is
  `130.6687/135.0226/137.3165` with error `+0.18%/+3.52%/+5.28%`.
  FENE-P: at Wi=0.05, Cd is `131.6026/136.5178/138.9648`; at Wi=0.1, Cd is
  `130.5407/134.7787/137.0061`. This is a useful stable comparison, but it is
  not yet the final full-embedded-gradient convergence.

Twenty-second follow-up slice:

- Root cause of the full embedded-gradient failure: the no-slip gradient
  correction used `wall_inv_distance` lowered from Cartesian cell-center to
  wall. For near-tangent circle cut cells in the LBM-coupled R20/R30 geometry,
  this produced distances around `0.00625` at fluid fractions near 0.5
  (`1/d≈160`) and injected an artificial wall-normal velocity gradient. The
  polymer source then drove `C` close to singularity and eventually made the
  LBM fields non-finite.
- The lowering contract has been changed so `wall_distance` means wall to the
  FV representative point of the fluid volume, not wall to the raw Cartesian
  cell center. This is now applied to coherent circles, half-planes, and
  q_wall-derived half-plane cuts; kernels still consume the same lowered
  arrays, so the fix is geometry-lowering generic rather than a circle-only
  branch in the GPU operator.
- Diagnostics after the fix: the R20 shifted-circle near-tangent half-fluid
  cells now have `min_d≈0.229` and `max_inv≈4.36` instead of `max_inv≈160`.
  Local CPU R20/Wi=0.001 with full embedded advection/gradient/force/drag
  stays finite to 500 steps (`first_nonfinite_step=0`, `minC≈0.989`), and
  local Metal Float32 stays finite to 200 steps (`minC≈0.988`).
- Local validation after the generic lowering change:
  `test/test_fvfd_operators_2d.jl` passes `952/952`,
  `test/test_logfv_frozen_channel_cde.jl` passes all embedded/channel
  canaries, and `test/test_viscoelastic_logfv_patch_ladder.jl` passes
  `14122/14122`.
- The corrected files were synced to Aqua and a full embedded-gradient sweep
  was submitted without GPU-model pinning as job `21310647.aqua`. Initial PBS
  state was queued in `gpu_batch_exec`; expected output root is
  `results/viscoelastic_logfv/embedded_circle_convergence_anygpu_21310647.aqua/`.

Twenty-third follow-up slice:

- Aqua job `21310647.aqua` completed successfully on 2026-05-14 on an
  A100-SXM4-40GB with `Exit_status=0`. Both Oldroyd-B and FENE-P reported
  `Done without case errors`; all `24/24` rows are `status=ok`, with saved
  `fields.jls` and VTR snapshots for Newtonian, Wi=0.001, 0.05, and 0.1 at
  R=10/20/30.
- The previous full-embedded-gradient failures are closed: for R20/R30,
  Wi=0.001, both models now have `first_nonfinite_step=0`. Oldroyd-B Cd is
  `138.3776/141.1336` with `minC≈0.9863/0.9837`; FENE-P Cd is
  `138.3776/141.1336` with the same near-Newtonian conformation bounds.
- Full embedded-gradient RheoTool comparison remains mesh-divergent at
  finite Wi. Oldroyd-B Cd at Wi=0.05 is
  `132.2479/137.9420/140.8546` for R=10/20/30, corresponding to RheoTool
  mean errors `+0.33%/+4.65%/+6.86%`. At Wi=0.1, Cd is
  `131.3379/136.5386/139.2704`, errors `+0.70%/+4.68%/+6.78%`.
  FENE-P is similar: Wi=0.05 gives
  `132.2074/137.8679/140.7638`, and Wi=0.1 gives
  `131.2121/136.3006/138.9757`.
- Interpretation: the generic wall-distance/centroid lowering fix closes the
  NaN/non-SPD embedded-gradient bug. It does not yet explain the remaining
  RheoTool mesh-convergence drift; the next target should be a lower-level
  quantitative check of embedded traction/drag and polymer force consistency
  on the saved fields before treating macro Cd as a validation signal.
