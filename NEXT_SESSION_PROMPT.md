# Next session prompt — Kraken.jl viscoelastic (dev-viscoelastic)

Copy-paste everything below to start a fresh session.

---

Continue work on `dev-viscoelastic` branch of Kraken.jl
(worktree `~/Documents/Recherche/Kraken.jl-viscoelastic`).

## Core question for this session

**Does the 3D viscoelastic sphere driver converge to Lunsmann 1993 reference
when R is refined?** If yes → publishable. If no → real bug to find.

Previous session spent many hours on 2D Poiseuille diagnostics that were
muddled by incorrect scaling (Nx too small, Wi varying with Ny, BGK instead
of TRT, etc). **Do NOT restart those diagnostics.** Go directly to the R-sweep
on sphere 3D.

## What is validated (don't re-test, don't touch)

- **2D Liu cylinder Cd at R=48, Wi=0.1**: 130.78 vs Liu Table 3 130.83 → **0.32% error**
- **2D R-convergence of Cd** (from commit `97c5d35`):
  - R=20: −7.4%, R=30: −3.0%, R=48: +0.35% → order ~3.5 on Cd
- **Driver `run_conformation_cylinder_libb_2d`** (`src/drivers/viscoelastic.jl`):
  TRT + LI-BB V2 + ZouHe inlet/outlet + Hermite source + CNEBB on cylinder
  + inlet/outlet conformation reset. DO NOT CHANGE.
- **Hermite source 3D calibration**: machine-precision match 2D (via
  `hpc/hermite_magnitude_diag.jl`)
- **Simple shear 3D kernels**: identical to 2D at the bit (Mach error 2-6%
  at Wi=0.25, same in both)
- **Test suite**: 16/16 `test_conformation_lbm.jl`, 17/17
  `test_conformation_lbm_3d.jl`, 24/24 contraction, 13/13 sphere 3D
  (but the sphere tolerance is 50%, masks the bug)

## Open bug — 3D sphere Cd deficit

From `hpc/sphere_oldroyd_3d.jl` (job 20147611 on Aqua H100):
```
Wi=0   Cd_Newt = 215.3  (ref)
Wi=0.1 Cd = 192.0  → ratio 0.89 (should be ~1.0 at low Wi)
Wi=0.5 Cd = 144.9  → ratio 0.67
Wi=1.0 Cd = 131.5  → ratio 0.61
```

Cd **DECREASES with Wi**, opposite of Lunsmann 1993 (sphere drag enhancement).

At R=16, the 2D R-convergence analog would give ~10% Cd error (matches the
11% we see). So the deficit **may be just discretization** — but the
monotone-decreasing trend is physically wrong either way.

## Proposed action — decisive test

**Run the sphere at R=16, 32, 48 at Wi=0.1 and see if Cd converges toward
Newtonian** (= ν_total) at low Wi, and **toward Lunsmann-enhancement** at
higher Wi. Use `hpc/sphere_oldroyd_3d.jl` as template, add R sweep.

Expected decisions from the sweep:
- **Monotone Cd → target**, order ~3 like 2D cylinder → driver validated,
  publish at R=48
- **Cd plateau below target** → real bug; drill into CNEBB 3D + LI-BB cut
  link interaction on curved surface (sphere has many q_w at extremes at
  small R, unlike 2D cylinder)

Computational cost per R on H100 (Nx=24R, blockage R/H=0.5):
- R=16: ~3 min (already done = 192.0)
- R=32: ~15 min (factor 8× cells, higher Wi regime so λ scales)
- R=48: ~50 min
- R=64: ~2 hours

A single PBS with all three fits in 4h walltime. Start with R=16/32/48
only (skip 64). If you need R=64 later, submit separately.

## Key gotchas (from `VISCOELASTIC_FINDINGS.md` — READ IT)

1. **MEA drag with Hermite source**: use `compute_drag_mea_2d` in 2D
   (halfway-BB, single cell). In 3D, `compute_drag_libb_3d` is single-cell
   too — NOT the Mei-Bouzidi variant — so it should be OK.
2. **Cd_p (surface integral of τ_p) is DIAGNOSTIC ONLY** when the Hermite
   source is active; Cd_s already captures σ_s + τ_p.
3. **Pre-existing segfault** in `run_sphere_libb_3d` at >30k cells on Mac
   M3 (Metal/CPU). Use H100 for any 3D run with R≥8.
4. **compute_drag_libb_3d** crashes with empty cut-link list (divides by 0
   in GPU partition). Workaround: don't run drivers with no obstacle in 3D.
5. **Julia 1.12 soft-scope**: wrap HPC scripts' main time loop in `let` to
   avoid `f_in, f_out = f_out, f_in` creating new locals.

## Files touched this session (diagnostic detours — DO NOT USE)

These are diagnostic one-offs that were mostly inconclusive or flawed.
Don't depend on them:
- `hpc/poiseuille_2d_analog_diag.jl`, `hpc/duct_visco_diag.jl`,
  `hpc/cylinder_extruded_diag.jl`, `hpc/poiseuille_convergence_diag.jl`,
  `hpc/poiseuille_constwi_convergence.jl`, `hpc/poiseuille_exact_benchmark.jl`,
  `hpc/poiseuille_trt_convergence.jl`, `hpc/hermite_magnitude_diag.jl`,
  `hpc/poiseuille_modular_diag.jl`
- `src/kernels/stream_periodic_3d.jl` (was for a test that didn't help)
- `src/kernels/li_bb_3d_v2.jl` adds `precompute_q_wall_cylinder_extruded_3d`
  and `fused_trt_libb_v2_step_3d_periodic_z!` (for diagnostics, fine to keep)
- `src/drivers/basic.jl` adds `compute_drag_mea_3d` (useful for
  axis-aligned walls, keep)

**Production code unchanged** (`src/drivers/viscoelastic.jl`,
`src/drivers/viscoelastic_3d.jl`, `src/kernels/conformation_lbm_*.jl`,
`src/kernels/viscoelastic_*.jl`). Tests still pass.

## HPC workflow

```bash
rsync -az --exclude='.git' --exclude='results' --exclude='Manifest.toml' \
  ./ maitreje@aqua.qut.edu.au:~/Kraken.jl-dev-viscoelastic/
ssh maitreje@aqua.qut.edu.au 'cd ~/Kraken.jl-dev-viscoelastic && qsub hpc/XXX.pbs'
```

Standard PBS: ncpus=8, gpu_id=H100, mem=32GB, walltime 4-12h.

## Suggested first action

1. **Check whether 20169899 (krk_trtc) and earlier R-sweep jobs have
   finished**: `ssh maitreje@aqua.qut.edu.au 'qstat -u maitreje && ls
   -lt ~/Kraken.jl-dev-viscoelastic/results/'`
2. **Write `hpc/sphere_R_sweep.jl`**: sphere Oldroyd-B at Wi=0.1 for
   R=16, 32, 48. Same setup as `hpc/sphere_oldroyd_3d.jl` (β=0.5, Re=1,
   blockage 0.5, doubly-parabolic inlet). Output Cd(R) + convergence rate.
3. **Submit to Aqua**, wait ~2h for result.
4. **If Cd converges** (order ~2-3 on |Cd − Cd_Newt| or similar metric) →
   driver validated, update `VISCOELASTIC_FINDINGS.md` §1 and §6 to remove
   the "bug confirmed" framing, close the investigation.
5. **If Cd plateaus wrong** → drill in: (a) run the same sphere with
   `polymer_bc=NoPolymerWallBC()` to isolate CNEBB; (b) measure Cd_s and
   Cd_p (polymer stress integral) separately to see which dominates the
   deficit.

## Modular refactor (parked until after the R-sweep verdict)

User wants production viscoelastic drivers decomposed into swappable
bricks:
```
struct ViscoLoop2D
    solvent_step!::Function   # BGKGuoStep | BGKZouHeStep | TRTLIBBStep
    conformation_step!::Function  # PeriodicXWallY | HBBWithReset
    stress_model::AbstractPolymerModel
end
```
so each component can be tested independently. Don't start this until
we know the R-sweep result — we might rebuild and the design might
change depending on what we find.

---

End of prompt. Good luck.
