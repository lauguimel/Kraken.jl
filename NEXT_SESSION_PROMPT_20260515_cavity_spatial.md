# Next session prompt — Kraken viscoelastic cavity spatial debug

Copy-paste below to start a fresh session.

---

Continue work on branch `dev-viscoelastic` of Kraken.jl
(worktree `~/Documents/Recherche/Kraken.jl-viscoelastic`).

## TL;DR — where we are

**Constitutive validated. Bug is purely spatial/coupling.**

The closed lid-driven cavity Oldroyd-B benchmark against rheoTool's
`Cavity/Oldroyd-BLog` shows Kraken at `0.1%` on the velocity peak but
`18-24% relative L2 error` on the centerline `u(0.5, y)` and the
horizontal `psi_xy(x, 0.75)` profile at `t = 8` for `N = 64`, `De = 1`,
`beta = 0.5`, `bsd_fraction = 0.75`. The constitutive integration itself
matches analytical Oldroyd-B simple shear AND planar extension to
machine precision at production substep cadence. The error must
originate in spatial coupling, not in the per-cell ODE.

This session: stop revisiting the constitutive math. Focus on which of
the five candidate spatial sources actually drives the gap.

## Solid established facts (do NOT re-litigate)

1. **Cavity driver works**: `run_viscoelastic_logfv_cavity_coupled_2d`
   in `src/drivers/viscoelastic_logfv_2d.jl` runs N=64 t=8 stably on
   Aqua A100 in 33 min wall (102400 LBM steps × 1829 polymer substeps).
   Snapshots saved at `t = 8`.

2. **North Zou-He moving lid kernel added this session** in
   `src/kernels/boundary_rebuild.jl::_bc_north_zh_velocity_2d!`. Before
   this, any `ZouHeVelocity` on the north face was silently ignored
   (line 329 TODO). Stationary lid was the prior default.

3. **BSD discordance threshold**: cavity at `bsd_fraction = 1.0`
   crashes (log() throws on non-SPD C at the lid corner). At
   `bsd_fraction in [0, 0.75]` the run is stable and profiles vary by
   less than 2% across that range. We use `bsd_fraction = 0.75`. The
   crash at bsd=1 is the LBM BGK implicit diffusion stencil mismatching
   the FD-central laplacian used in the explicit BSD correction. See
   `bench/viscoelastic_logfv/CAVITY_OLDROYDB_AXIS_ALIGNED_20260515.md`
   for the full sweep table.

4. **Constitutive is correct** (this session, 2026-05-15):
   - Kraken vs analytical Oldroyd-B simple shear, gamma in
     `{0.01, 0.1, 1, 10, 100}`, dt/lambda sweep `{1e-4, 1e-6, 1e-8}`:
     - dt/lambda = 1e-4: err_Linf 5e-5 (linear scaling)
     - dt/lambda = 1e-6: err_Linf 5e-7
     - dt/lambda = 1e-8: err_Linf 5e-9
     - linear convergence err = dt/(2 lambda) * Wi, as expected for
       first-order operator splitting.
   - Production cavity substep dt_sub/lambda_LU = 4.3e-8, so the
     per-step bias is about 2e-8, below rheoTool's reported 1e-7
     relative error.
   - Kraken planar extension (gradU = eps_dot * diag(1, -1)) at
     De in `{0.01, 0.05, 0.1, 0.2, 0.4}` (just below the Oldroyd-B
     blow-up bound De = 0.5) also matches analytical with the same
     linear-in-dt/lambda scaling. No hidden L*C + C*L^T sign bug.
   - rheoTool rheoTestFoam Oldroyd-B at the same gammas reports
     converged extra-stress matching the analytical to ~1e-8.
   - Per-cell Kraken constitutive throughput: ~10 M steps/s/cell on CPU.

5. **The cavity 33 min Aqua wallclock is NOT a constitutive math
   problem**. It is launch-overhead dominated: about 1.9e8 polymer
   substep × cell-ops on N=64, each requiring its own GPU kernel
   dispatch. Reducing the substep count or fusing the substep loop is
   a perf optimisation, not a correctness fix.

## The 5 spatial-coupling candidates for the 18-24% profile gap

Ranked by effort and prior likelihood:

### Candidate 1 — Re mismatch (easy, high prior)

Kraken cavity runs at Re_LU = u_max * N / nu_total = 6.4 (with the
chosen lattice scaling). rheoTool runs at Re_phys = 0.01. The
matching is on De and beta, not Re. At Re ~ 1, inertia adds to the
lid-driven circulation and stiffens the near-lid velocity gradient
relative to the Stokes limit.

**Test**: cavity Aqua sweep at `u_max in {0.005, 0.002, 0.001}` keeping
`N`, `nu_s`, `nu_p`, `lambda_phys` fixed. This lowers Re_LU from 6.4
to 1.3 while keeping De and beta. If the centerline profile L2 drops
significantly at u_max = 0.001, Re mismatch is dominant. Walltime per
case scales as 1 / u_max (smaller u_max needs more steps for same
end_time), so the sweep is roughly 4x the N=64 baseline = `~2 h` total
on A100.

### Candidate 2 — Wall gradient correction artifact (cheap, medium prior)

The closed cavity driver does a half-cell wall gradient correction at
all four walls (see
`_logfv_cavity_apply_wall_gradient_correction_kernel!` in
`src/drivers/viscoelastic_logfv_2d.jl`). At the corner cell `(1, Ny)`
both the north (moving lid) and west (fixed) corrections fire. The
N=64 profile shows a `sign-flip in psi_xy(x, 0.75)` around `x ~ 0.19`
where Kraken is positive (+0.003) but rheoTool is negative (-0.17) —
suspicious local artifact near the top-left corner.

**Test**: local CPU smoke at N=32, end_time=2.0, with the wall
gradient correction stub-replaced by a no-op for the corner cells
only. Compare `psi_xy(x, 0.75)` to the version with full correction.
If the sign-flip disappears with the no-op, the corner treatment is
the issue and needs a softer formulation.

### Candidate 3 — Polymer upwind advection (medium effort, medium prior)

The `logfv_advect_upwind_bc_aware_2d!` is first-order upwind on the
log-conformation field. On a sharp corner-driven flow the upwind
diffusivity is non-negligible at N=64 (Peclet ~ 1 per cell). The
profile shape disagreement may be partly numerical diffusion.

**Test**: import the rheoTool N=127 U field at `t = 8`, freeze it,
and run only the Kraken polymer pipeline (advection + source +
stress + force) on a 64x64 sample of the rheoTool U field. Compare
the Kraken polymer field after one polymer-relaxation timescale to
rheoTool's at the same time. If the upwind diffusion dominates, the
shape is wrong even on a frozen reference U; if the upwind is fine,
the bug is upstream in U itself.

There is already a frozen-replay harness at
`bench/viscoelastic_logfv/run_rheotool_frozen_replay_2d.jl` from the
earlier cylinder pivot. It needs adaptation to the cavity geometry.

### Candidate 4 — Guo body-force stencil consistency (medium-high effort)

`fused_trt_libb_v2_guo_field_step!` applies the polymer force
`f = div(tau)` to the LBM solvent through a Guo body-force term. The
divergence of `tau` is computed by central FD; the Guo body force is
applied through the LBM-native source moment expansion. These two
discretisations do not cancel exactly at the cell-level when `tau`
varies sharply. Similar in spirit to the BSD/LBM discordance, but
applies to every viscoelastic driver, not just cavity.

**Test**: compute `Cd_p` from `tau` field two ways on the saved N=64
field: (a) FD divergence then sum, (b) LBM-Guo accumulator. Compare
the two integrated polymer drag values. If they differ by ~10-20%, the
Guo/FD inconsistency is real and likely contributes to the cavity gap.

### Candidate 5 — BSD truncation residual (low prior at bsd=0.75)

We are operating at `bsd_fraction = 0.75` because `bsd = 1.0` crashes
on the lid corner. RheoTool uses the equivalent of `bsd = 1.0` and
runs fine because its FV-FV split cancels exactly. The 25% BSD
deficit in Kraken could shift the cavity profile, but the previous
sweep at N=32 showed less than 2% sensitivity within
`bsd in [0, 0.75]`, so this is a low-prior contributor at N=64.

The architecturally clean fix is **kinetic-moment BSD**: extract the
rate-of-strain tensor from the LBM non-equilibrium moments
`f_q - f_q^{eq}` (which is exactly what BGK relaxes toward) and use
that tensor in the BSD stress instead of FD-central laplacian. This
would make the BSD/LBM split exact at the discrete level. ~3-4 hours of
code, touches every viscoelastic driver. Deferred until candidates 1-4
have been investigated.

## Strongly recommended first action

**Candidate 1 (Re mismatch sweep) on Aqua**. Cheapest, most
discriminating. If Re mismatch is dominant, the profile L2 should
drop from 18% to single-digit percent at u_max = 0.001.

```bash
# Local: prepare PBS with u_max sweep
# Then on Aqua:
qsub bench/viscoelastic_logfv/run_cavity_oldroydb_vs_rheotool_anygpu.pbs
# After completion, sync results and compute the profile L2 vs u_max.
```

The existing PBS at
`bench/viscoelastic_logfv/run_cavity_oldroydb_vs_rheotool_anygpu.pbs`
already supports `KRAKEN_U_MAX` and `KRAKEN_N_LIST`. Either submit
three separate jobs (one per u_max), or wrap the loop inside the PBS
script.

## What NOT to do

- **Do not re-run 0D constitutive tests.** They are validated against
  analytical and rheoTool to machine precision at production substep
  cadence. Adding more tests is busywork at this point.
- **Do not try `bsd_fraction = 1.0` on cavity.** It crashes by design.
  Use the kinetic-moment BSD route if you want full BSD.
- **Do not start with the kinetic-moment BSD refactor.** It is the
  proper fix but is the wrong order: confirm or rule out candidates
  1-4 first, since they may explain the bulk of the cavity gap with
  less invasive changes.
- **Do not pivot back to the cylinder benchmark.** The cylinder
  embedded-geometry ratchet (23 follow-up slices, see
  `bench/viscoelastic_logfv/VALIDATION_LADDER_AUDIT_20260513.md`) is
  closed. The cavity is the axis-aligned discriminator below the
  curved cylinder; only return to the cylinder after the cavity gap
  is understood.

## Where everything lives

### Code
- Cavity coupled driver:
  `src/drivers/viscoelastic_logfv_2d.jl::run_viscoelastic_logfv_cavity_coupled_2d`
- North Zou-He moving lid:
  `src/kernels/boundary_rebuild.jl::_bc_north_zh_velocity_2d!`
- BC helpers:
  `src/kernels/logconformation_fv_2d.jl::logfv_wallxwally_bcspec_2d`
  `src/fvfd/specs.jl::fvfd_wallxwally_bcspec_2d`
- 0D constitutive harness (already validated):
  `bench/viscoelastic_logfv/run_constitutive_0d_vs_rheotest.jl`
- Cavity comparison harness:
  `bench/viscoelastic_logfv/run_cavity_oldroydb_vs_rheotool.jl`
- Aqua PBS:
  `bench/viscoelastic_logfv/run_cavity_oldroydb_vs_rheotool_anygpu.pbs`

### References and test data
- rheoTool cavity reference (project-local, ~80 MB):
  `bench/rheotool/cavity_oldroydb_log_re001_de1_b05/`
  - sampleDict outputs at every integer t from 1 to 8
  - `kinEner.txt` time series
- rheoTool 0D rheoTestFoam case (project-local):
  `bench/rheotool/rheotest_oldroydb/Report` (5 gammas, converged)
- Aqua N=64 cavity results synced locally:
  `tmp/cavity_aqua_n64/{profile_vertical_x0.5.csv, profile_horizontal_y0.75.csv, fields.jls, summary.csv, kinetic_energy_kraken.csv}`

### Verdict files
- `bench/viscoelastic_logfv/CAVITY_OLDROYDB_AXIS_ALIGNED_20260515.md`
  (cavity setup, BSD sweep, decision tree)
- `bench/viscoelastic_logfv/CONSTITUTIVE_0D_AUDIT_20260515.md`
  (0D shear validation, dt/lambda convergence)

### Git
- Branch: `dev-viscoelastic`
- Last commit on cavity work: `212e26a3` (Metal compat + PBS)
- Latest cavity-validation commit: `f69b2e35` (driver + comparison
  harness + rheoTool reference + first verdict)
- **Not yet committed**: 0D harness extension (planar elongation flow
  support), 0D verdict file `CONSTITUTIVE_0D_AUDIT_20260515.md`,
  rheoTool case `rheotest_oldroydb/`, Aqua results in `tmp/`.

### Aqua state at session end
- Job `21330952.aqua` was killed mid-N96 after N=64 saved. The local
  `tmp/cavity_aqua_n64/` has the data.
- No queued or running jobs.

## Performance reality

The cavity coupled run at N=64 t=8 on Aqua A100 takes 33 min. This is
*launch-overhead bound*, not compute bound. Each LBM step launches
about 5 control kernels plus 1829 polymer substep kernels on a tiny
N x N = 4096 cell grid. The H100/A100 sit waiting on kernel-dispatch
latency. The honest path to speedup is fusion of the substep loop into
a single kernel (or reduction of substep count via a higher-order
constitutive scheme), not a different machine.

For the spatial debug, this means a single N=64 sweep at three u_max
values will take 2-3 hours of HPC walltime. Do not budget less.

## Concrete first commands

```bash
cd ~/Documents/Recherche/Kraken.jl-viscoelastic

# 1. Commit the 0D audit work (so the new session can see it)
git add bench/viscoelastic_logfv/run_constitutive_0d_vs_rheotest.jl \
        bench/viscoelastic_logfv/CONSTITUTIVE_0D_AUDIT_20260515.md \
        bench/rheotool/rheotest_oldroydb \
        bench/viscoelastic_logfv/NEXT_SESSION_PROMPT_20260515_cavity_spatial.md
git commit -m "..."

# 2. Sync repo + N=64 reference results to Aqua
rsync -az --exclude='.git' --exclude='results' --exclude='Manifest.toml' \
        --exclude='tmp' \
    ./ maitreje@aqua.qut.edu.au:~/Kraken.jl-dev-viscoelastic/

# 3. Submit Re-mismatch sweep (three separate jobs)
for u in 0.005 0.002 0.001; do
    ssh maitreje@aqua.qut.edu.au "cd ~/Kraken.jl-dev-viscoelastic && \
        KRAKEN_U_MAX=$u KRAKEN_OUTPUT_DIR=results/viscoelastic_logfv/cavity_remismatch_u${u}_\${PBS_JOBID:-manual} \
        qsub -v KRAKEN_U_MAX=$u \
            bench/viscoelastic_logfv/run_cavity_oldroydb_vs_rheotool_anygpu.pbs"
done

# 4. Once jobs return: sync results, plot u(0.5, y) Kraken vs rheoTool
#    for each u_max, compute relative L2 vs rheoTool. If L2 drops
#    monotonically with u_max, Re mismatch is dominant.
```

End of prompt.
