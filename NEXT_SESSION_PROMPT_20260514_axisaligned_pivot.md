# Next session prompt — log-FV pivot to axis-aligned wall benchmark

Copy-paste below to start a fresh session.

---

Continue work on branch `dev-viscoelastic` of Kraken.jl
(worktree `~/Documents/Recherche/Kraken.jl-viscoelastic`).

## Why we are pivoting (read this once, do not re-litigate)

Five+ sessions and 23 incremental "embedded slices" have not closed the
log-FV cylinder RheoTool drift. The drift signature is :

- error grows with mesh refinement (anti-convergence: +0.33% R=10 → +6.86% R=30) ;
- error is Wi-independent (~+6.8% at R=30 across all Wi) ;
- so it is a **near-wall geometric/coupling artifact**, not a constitutive bug.

The frozen-RheoTool-U replay (commits this session, `bench/viscoelastic_logfv/RHEOTOOL_FROZEN_REPLAY_20260514.md`)
gave a clean but unsatisfying verdict: τ explodes locally
(R=10: τ_xy rel L∞ = 1.18; R=30: 1.3e+11, Cd NaN). Most likely cause is
**mismatch between Kraken's `is_solid` mask and RheoTool's analytic
wall after resample**, producing a fake huge ∇U at near-tangent cut
cells. That makes the replay weaker than expected as a bisection tool.

**The pivot:** drop the curved cylinder for one session. Use an
**axis-aligned wall step benchmark** where Kraken and RheoTool see the
same mesh-aligned geometry by construction. This removes every cut-cell
confound while exercising the same FV polymer operator stack
(`_run_viscoelastic_logfv_step_channel_coupled_2d`).

## The discriminating experiment

### Reference case : RheoTool 4:1 planar contraction Oldroyd-BLog

Tutorial path on this machine :

```text
/Users/guillaume/Documents/Clouds/UGA/Recherche/QUT/Rheotool/container/demo 2/rheoTool/tutorials/rheoFoam/Contraction41/Oldroyd-BLog/
```

Canonical Alves-Pinho-Oliveira 2003 setup. Parameters (from
`constant/constitutiveProperties` and `0/U`):

- Geometry : planar 4:1 contraction, half-heights 4 → 1
  (full heights 8 → 2), `x ∈ [-100, 100]`.
- Inlet : ramped `U_av = 0.25`, ramp time `t_lim = 1`.
- Constitutive : Oldroyd-BLog, β = etaS/(etaS+etaP) = 0.111/1.0 = 0.111,
  λ = 1, ρ = 0.01.
- Wi = λ·U_av/H_out_half = 0.125 ; Re = ρ·U·H_out / η_0 = 0.02 (Stokes).
- Endtime t = 20, deltaT = 2e-4 with maxCo = 0.01 adaptive.

This is **harder than BFS** (strong extensional component at the
throat amplifies polymer stress) and harder than cylinder (Wi = 0.125 vs
0.1, β = 0.111 vs 0.59 — much higher polymer fraction). So if Kraken
matches here, the curved-cylinder drift is isolated to the cut-cell
path.

Run RheoTool first (Docker wrapper at
`/Users/guillaume/Documents/Clouds/UGA/Recherche/QUT/Rheotool/openfoam9-rheotoolv12.sh`)
to refresh `U`, `tau`, and centerline samples to `t=20` (steady-state).
Dump:

- centerline U(x) (extensional acceleration through throat),
- `tau_xx(x, y=0)` along centerline (peak at throat exit),
- vortex corner length `X_R` (where ψ_streamline crosses zero in
  the upstream large-channel corner),
- `tau_xx(y)` at `x = 0` (just after throat).

### Kraken side : add a contraction coupled driver

The geometry primitive `contraction_step_geometry_2d` is already
exported in `src/Kraken.jl:337`. Mirror `run_viscoelastic_logfv_bfs_coupled_2d`
(`src/drivers/viscoelastic_logfv_2d.jl:679`) into a
`run_viscoelastic_logfv_contraction_coupled_2d`. It is ~15 lines of
glue : same `_run_viscoelastic_logfv_step_channel_coupled_2d` core,
swap the geometry call.

Match RheoTool params exactly (or as exactly as a LU lattice allows):

- Build a contraction with `H_in = 16`, `expansion_ratio = 4` (so
  H_out = 4 in lattice units), `L_up = 64`, `L_down = 64` first ;
  refine to `H_in = 32` then `H_in = 64` for convergence.
- LU calibration : pick `u_mean` so Re_LU = 0.02 with the chosen ν,
  and `lambda_lu` so Wi = 0.125.
- Same observables : centerline U(x), τ_xx(x, y=mid), τ_xx(y) at
  throat exit, vortex corner length.

### Comparison

A single dashboard with three Kraken meshes (Ny_out = 4 / 8 / 16) and
one RheoTool reference, plotting :

1. U(x) along centerline.
2. τ_xx(x) along centerline (the big signal).
3. τ_xx(y) at `x = 0`.
4. Streamlines + corner vortex polygon.

Decision tree:

| Outcome | Verdict | Next action |
|---|---|---|
| Kraken matches RheoTool within ~3% on all four observables and converges with mesh | Bug is **strictly in cut-cell path** for cylinder. The 4:1 polymer pipeline is correct. | Audit `compute_polymeric_drag_2d` + `fvfd_embedded_wall_traction_2d!` + cut-cell wall-distance lowering for the curved wall ONLY. The cylinder is a curved-wall problem, not a polymer-pipeline problem. |
| Kraken disagrees on τ_xx peak by >5%, or U(x) overshoots wrong | Bug is in **core FV polymer pipeline** (advection / source / Guo coupling). Cut-cells are a red herring. | Toggle bisection on contraction (advection-only, source-only, force-only); cylinder pause until contraction matches. |
| Kraken diverges (NaN, blow-up at throat) | Bug is upstream: either source subcycling cap (high λ·∇U at throat), or log-conformation stabilization | Tighten `polymer_substeps`, re-test ; if still blows up, the throat extensional stress is exceeding stable Wi for the open-x FV path, and that itself is the bug. |

## Concrete first actions

1. **Re-run the RheoTool tutorial** to `t = 20` and copy out a
   reference snapshot to `bench/rheotool/contraction41_oldroydblog/` so
   it stays version-controlled with the project (Allrun + minimal
   results).

2. **Add `run_viscoelastic_logfv_contraction_coupled_2d`** in
   `src/drivers/viscoelastic_logfv_2d.jl`, mirroring the BFS coupled
   driver but calling `contraction_step_geometry_2d`. Add an export.

3. **Add a quantitative driver**
   `bench/viscoelastic_logfv/run_contraction41_oldroydb_vs_rheotool.jl`
   that runs three Kraken meshes (`H_in = 16, 32, 64`), writes the
   four-panel dashboard, and saves a `summary.csv` with τ_xx peak,
   centerline U-overshoot, vortex length, max polymer substeps.

4. **Verdict file**
   `bench/viscoelastic_logfv/CONTRACTION41_AXIS_ALIGNED_<date>.md` with
   the four-panel comparison and the decision-tree outcome.

5. **If Kraken matches**, also run the same script for the **BFS
   geometry** (which already has a Kraken driver) as a control: if both
   axis-aligned cases pass, the cut-cell pathological cell on the
   curved cylinder is definitively the only bug. If BFS also disagrees,
   the contraction-passing result needs revisiting (probably a Wi/β
   regime difference).

## What NOT to do

- Do not run any cylinder sweep on Aqua this session. Cylinder is
  paused until the axis-aligned verdict is in.
- Do not add a 24th embedded slice to
  `VALIDATION_LADDER_AUDIT_20260513.md`. That document is closed.
- Do not start from the frozen replay : the resample-induced
  ∇U-explosion makes it noisy for this kind of bisection.
- Do not compare against Alves 2003 paper numbers when RheoTool is
  available locally with the same code path — apples-to-apples.

## Where everything lives

### Kraken
- BFS coupled driver to copy : `src/drivers/viscoelastic_logfv_2d.jl:679`
- Contraction geometry primitive : `contraction_step_geometry_2d` (exported)
- Operator core called by all step-channel drivers :
  `_run_viscoelastic_logfv_step_channel_coupled_2d`
  (`src/drivers/viscoelastic_logfv_2d.jl:148`)
- Existing log-FV cylinder bench harness for layout reference :
  `bench/viscoelastic_logfv/logfv_cylinder_cd_convergence.jl`

### RheoTool reference
- Tutorial source :
  `/Users/guillaume/Documents/Clouds/UGA/Recherche/QUT/Rheotool/container/demo 2/rheoTool/tutorials/rheoFoam/Contraction41/Oldroyd-BLog/`
- Docker wrapper :
  `/Users/guillaume/Documents/Clouds/UGA/Recherche/QUT/Rheotool/openfoam9-rheotoolv12.sh`
- Existing cylinder case helpers to copy patterns (Allrun, sampleDict,
  writeData) : `bench/rheotool/cylinder_oldroydb_log_re1_wi01/`

### Audits (READ-ONLY)
- `bench/viscoelastic_logfv/VALIDATION_LADDER_AUDIT_20260513.md` — 23
  slices, do not extend
- `bench/viscoelastic_logfv/RHEOTOOL_FROZEN_REPLAY_20260514.md` — last
  session verdict
- `AUDIT_SUMMARY.md` — pre-log-FV (LBM-direct path)

End of prompt.
