# Next session prompt — Kraken viscoelastic cavity coupling-bug closure

Copy-paste below to start a fresh session.

---

Continue work on branch `dev-viscoelastic` of Kraken.jl
(worktree `~/Documents/Recherche/Kraken.jl-viscoelastic`).

This session supersedes `NEXT_SESSION_PROMPT_20260515_cavity_spatial.md`.
The 2026-05-16 session ran 15 missions (M1-M15) via the orchestrator
pattern; the bug locus is now precisely localised but the fix is
**gated by a prerequisite refactor**. Read this prompt end-to-end
before launching any mission.

## TL;DR — where we are

**Bug fully localised to the LBM↔polymer coupling layer.** Everything
else is ratched out. The headline numbers:

- **M7b**: at matched total LBM viscosity at `lambda_phys=0.001`
  (Wi≈0), the cavity profile differs by **3.42 % rel L2** between
  `nu_p=0.1, nu_s=0.1` and `nu_p=0, nu_s=0.2` cases — when it should
  be at the **0.014 % Newtonian noise floor** (B-vs-C control). This
  is the cleanest empirical signature: the polymer machinery
  introduces a Wi-INDEPENDENT perturbation that should not exist if
  the BSD/Guo split correctly absorbs `τ_p ≈ 2·ν_p·D` into the LBM
  solvent viscosity.

- **M10**: algebraic root cause = **stencil mismatch**. `div(τ_p)`
  is computed by chaining `fvfd_velocity_gradient_2d!` then
  `fvfd_tensor_divergence_2d!` → wide 3-point laplacian (2dx
  spacing). `fvfd_bsd_force_2d_kernel!` uses a narrow 3-point
  laplacian (dx spacing). The two operators converge to the same
  continuum `∇²u` but differ by O(dx²·∂⁴u) at the discrete level.
  At production cavity gradients they differ by ~O(1) per M14.

- **M12 literature audit**: rheoTool's `stabilization coupling` (iBSD
  mode, used by EVERY rheoTool viscoelastic case) enforces same-
  stencil cancellation at the dictionary level: both `div(τ)` and
  `div((etaP)·grad(U))` declared `Gauss linear` in `fvSchemes`. **Liu
  2025 has NO BSD at all** — direct Hermite-moment injection of the
  stress into the LBM `f_i` (Eq. 22). The wide-vs-narrow trap that
  bites Kraken is structurally absent from both references.

- **M15 architectural audit**: 9 faults identified in the cavity
  pipeline at Wi→0. Top is M10's stencil mismatch. Secondary:
  `τ_p` carries `Ψ_history` while `τ_BSD(D_now)` would be
  instantaneous — different "times" even if same stencil. The
  default FD-BSD path **reads `ux, uy` directly, NOT the
  wall-corrected `D`** the source ODE consumes — provenance
  ambiguity compounds.

- **M9 grid convergence (in progress)**: cavity L2 vs rheoTool falls
  monotonically with N (N=32: 31.4%, N=64: 18.0%, N=96: 12.85%);
  N=128 due ~21:00 AEST. ~30% drop per N-doubling, so the
  18-24 % gap at N=64 is partly a discretization floor against
  rheoTool's N=127 reference.

The 3.4 % Wi-independent bug + the residual finite-Wi gap together
make the 18 % centerline / 24 % psi_xy production gap. Fix the
former; the latter may shrink to single digits once N=128 is in.

## What is ratched OUT (do NOT re-investigate)

| Mission | Concern | Status |
|---------|---------|--------|
| M1 | Re mismatch (u_max sweep at fixed De/β) | Refuted: L2 flat across u_max |
| M3 | Polymer pipeline via frozen cavity replay | Consistent (loose 4 %) |
| M4b | BSD operator (`bsd_fraction` sweep 0..0.75) | BSD *helps* the match, doesn't hurt |
| M6-B | Wall-stencil on τ FD divergence (linear vs quadratic) | Refuted at production |
| M8 | Polymer pipeline analytical Poiseuille | First-order convergent in `dt_poly`; at production `n_substeps=4096`, source error ~4e-6 |
| M13 | Guo body-force injection (inverse test, frozen analytical τ) | Bit-exact (2.7e-20) and second-order convergent in N |

The constitutive ODE, the FV polymer pipeline, the wall velocity-
gradient stencil, the Guo body-force injection, and the BSD-fraction
choice are all ratched. The bug is NOT in any of them.

## The bug locus — single sentence

**The default `:fd` BSD path uses a narrow Laplacian on `(ux, uy)`
while `div(τ_p)` uses a wide Laplacian on `D` (a different field
with different provenance), and the two are not the same discrete
operator on the same input, so the design-intent cancellation
`(1−ζ)·ν_p·∇²u` becomes `ν_p·(Lap_wide − ζ·Lap_narrow)·u + O(Ψ_history)`.**

## What was tried as a fix (do NOT repeat without splitting first)

| Attempt | Method | Outcome |
|---------|--------|---------|
| M5-B | Kinetic `Π^{neq}` BSD kernel, infrastructure default `:fd` | Commited as infra. On smooth t=0: bit-equivalent to FD-BSD (5.85e-16). On dynamic cavity (M14): diverges from FD-BSD by O(1). NOT a fix on its own. |
| M11 | Same-stencil `logfv_bsd_stress_from_gradient_2d!` → `fvfd_tensor_divergence_2d!` route | RED. Produced 64 % A-vs-B (vs 3.4 % bug). Root cause: BSD then used `D_corrected` while `τ_p` carried `Ψ_history` from the source ODE — different "times". Reverted. |
| (intra-session) | Flip `bsd_kind=:kinetic` as default | NaN DomainError; kinetic path can't be the default as-is. Reverted. |
| (intra-session) | `bsd_fraction=0` at low Wi | A0 case NaN-crashes. **BSD is load-bearing for stability**, not optional. |

Three fix attempts all failed for the same structural reason: the
**cavity driver is a 3429-LOC monolith** that mixes geometry, BC,
solver, stencil, and physics in one file. Surgical fixes interact
with neighbouring concerns through shared buffers, shared kwargs,
and shared line context. The fix needs the file split FIRST.

## Critical architectural constraint (this session, going forward)

**Every project using the orchestrator pattern now enforces** (see
`~/.claude/skills/orchestrator/SKILL.md` §Engineering hygiene,
updated 2026-05-16):
- Files target **≤500 LOC (soft)** / **≤700 LOC (hard)**.
- **One file = one concern** (geometry / BC / solver / stencil /
  physics / driver kept separate).
- Symbol-anchored references in briefs (don't trust line numbers).

The Kraken viscoelastic cavity driver violates this at 3429 LOC and
mixes 5 concerns. **The first mission of the next session must be a
SPLIT, not a BSD fix.** Departments that try the BSD fix on the
monolith will repeat M11's failure.

## Next-session mission plan (in order)

### M16 — SPLIT the cavity driver

- **Status**: planned, blocking M17.
- **Goal**: decompose `src/drivers/viscoelastic_logfv_2d.jl` (3429
  LOC) into:
  - `cavity_driver_2d.jl` — `run_viscoelastic_logfv_cavity_coupled_2d`
    only (timestep loop), target ≤700 LOC.
  - `cavity_wall_correction_2d.jl` — wall-gradient correction
    kernels.
  - `cavity_bsd_assembly_2d.jl` — BSD path selection (`:fd`,
    `:kinetic`, and the future Option-3 same-stencil path) +
    `diagnose_bsd_dual` instrumentation.
  - `cavity_init_2d.jl` — buffer allocation and IC setup.
  - `cavity_snapshot_2d.jl` — output / diagnostics writers.
- **Allowed edit zones**: the new files + the existing driver file
  to remove migrated code + `src/Kraken.jl` to update includes.
- **Exit criterion**: `julia --project=. test/runtests.jl` passes
  (no behaviour change). The split is purely refactor; no semantic
  changes. The cavity comparison harness `run_cavity_oldroydb_vs_rheotool.jl`
  reproduces the M1 baseline (centerline L2 = 0.1797, psi_xy L2 =
  0.2441) to machine precision.
- **Notes**: this is a substantive but mechanical task. Department
  should use Codex with the new file-size rules from
  `~/.claude/skills/orchestrator/SKILL.md`. Bias toward extracting
  the wall-correction first (smallest, most isolated), then BSD
  assembly, then init/snapshot, last is the timestep loop itself.

### M17 — Option 3 BSD same-stencil fix (gated on M16)

- **Status**: planned, depends on M16.
- **Goal**: implement the rheoTool-style iBSD pattern in Kraken.
  Per M12, rheoTool uses `fvc::div(τ/ρ) − fvc::div((etaP/ρ)·fvc::grad(U))`
  with same-stencil `Gauss linear` enforced at the `fvSchemes`
  dictionary level. The Kraken equivalent:
  - At the same pipeline step where the source ODE captures
    `D_corrected`, ALSO compute `τ_BSD = 2·ζ·ν_p·D_corrected`
    (using `logfv_bsd_stress_from_gradient_2d!`).
  - Route `τ_BSD` through the SAME `fvfd_tensor_divergence_2d!`
    operator (and the same `polymer_wall_extrap` kwarg) as
    `div(τ_p)`. This guarantees the wide-stencil cancellation
    `F_poly − F_BSD = div(τ_p − τ_BSD) = (1−ζ)·ν_p·∇²_wide·u` at
    the discrete level.
  - Default `bsd_kind=:fd` routes to the FIXED path; the legacy
    narrow-Laplacian path either retires (clean cutover, since it
    was buggy) or moves behind a `bsd_kind=:fd_legacy` flag for
    documentation.
- **Allowed edit zones**: the post-M16 split modules. ~50-85 LOC
  + 5 persistent N×N buffer allocations per M15's estimate.
- **Exit criterion**: the M7b PBS (3-case matched-viscosity) at
  Aqua N=64 t=8 yields A-vs-B rel L2 **< 0.1 %** on centerline u
  (vs the 3.42 % bug signal; well above the 0.014 % noise floor).
- **Risk**: M11's same-stencil attempt destabilised the cavity
  because BSD captured `D` at a different pipeline step than the
  source ODE. M17 must capture both at the SAME step (just after
  the wall-gradient correction). This is what M15 §"Recommended
  refactor" Option 3 prescribes.

### M18 — Validate at production (gated on M17)

- **Status**: planned.
- **Goal**: re-run the cavity Oldroyd-B comparison at production
  parameters (N=64, t=8, De=1, β=0.5) with the M17 fix. Expected
  outcomes:
  - Centerline u L2: 18.0 % → significantly lower (the 3.4 % bug
    component disappears; the remaining gap is the discretization
    floor seen by M9).
  - psi_xy L2: 24.4 % → significantly lower.
- **If gap closes to ~5-8 %**: combined with the M9 grid-convergence
  trajectory (N=128 may already be ~10 %), the cavity benchmark is
  validated. Mandate moves to "done".
- **If gap stays at 12-18 %**: there's a finite-Wi residual not
  captured by M17. M15's other 8 faults become candidates.

### M19 — Synthesis paper / validation note (optional)

After M18 GREEN, write up the cavity validation for the dev-viscoelastic
paper section: ratchet sequence, bug isolation methodology, fix design.

## Artefacts in place (do not re-create)

### Code (committed on dev-viscoelastic HEAD)

- `src/kernels/bsd_kinetic.jl` (M5-B): `compute_pi_neq_2d_kernel!` +
  `compute_bsd_force_kinetic_2d!`. Infrastructure for `Π^{neq}`
  extraction; behaviour-equivalent to FD-BSD on smooth interior.
- `src/kernels/logconformation_fv_2d.jl::logfv_bsd_stress_from_gradient_2d!`
  (lines ~678-708): produces `τ_BSD = 2·ζ·ν_p·D`. Already used by
  M11 (RED) but the M17 plan uses it correctly (captured at the
  right pipeline step).
- `src/drivers/viscoelastic_logfv_2d.jl` kwargs (commit `9d66b2c2`
  series):
  - `skip_top_corners::Bool = false` (M2)
  - `bsd_kind::Symbol = :fd` (M5-B)
  - `polymer_wall_extrap::Symbol = :quadratic` (M6-B)
  - `diagnose_bsd_dual::Bool = false` (M14) — canary for any future
    BSD work; records FD/kinetic divergence per step.
- `src/fvfd/operators_2d.jl::polymer_wall_extrap` kwarg
  (commit `7c790cd8` on `dev/fvfd-core` branch). `dev-viscoelastic`
  reads it from the sibling worktree at
  `~/Documents/Recherche/Kraken.jl-fvfd-core` (untracked on
  `dev-viscoelastic` per project convention).

### Bench scripts (committed)

- `bench/viscoelastic_logfv/run_cavity_oldroydb_vs_rheotool.jl` —
  main cavity comparison harness; consumes `KRAKEN_*` env vars.
- `bench/viscoelastic_logfv/analyse_cavity_remismatch.jl` — generic
  L2 analyser, used by every cavity sweep verdict.
- `bench/viscoelastic_logfv/run_cavity_corner_artifact_2d.jl` (M2)
- `bench/viscoelastic_logfv/run_rheotool_frozen_replay_cavity_2d.jl` (M3)
- `bench/viscoelastic_logfv/analyse_cavity_guo_vs_fd_2d.jl` (M4)
- `bench/viscoelastic_logfv/run_bsd_kinetic_audit_2d.jl` (M5-B)
- `bench/viscoelastic_logfv/run_wall_stencil_audit_2d.jl` (M6-B)
- `bench/viscoelastic_logfv/run_cavity_lowwi_sanity.pbs` (M7)
- `bench/viscoelastic_logfv/run_cavity_lowwi_matched_visc.pbs` (M7b)
- `bench/viscoelastic_logfv/run_poiseuille_polymer_analytical_2d.jl` (M8)
- `bench/viscoelastic_logfv/run_cavity_grid_convergence.pbs` (M9)
- `bench/viscoelastic_logfv/run_bsd_dual_path_diagnostic_2d.jl` (M14)
- `bench/viscoelastic_logfv/run_poiseuille_imposed_stress_2d.jl` (M13)

### Verdict & audit docs (committed)

- `bench/viscoelastic_logfv/CAVITY_REMISMATCH_M1_VERDICT_20260515.md`
- `bench/viscoelastic_logfv/CAVITY_BSD_M4B_VERDICT_20260516.md`
- `bench/viscoelastic_logfv/CAVITY_M6B_CONFIRM_VERDICT_20260516.md`
- `bench/viscoelastic_logfv/CAVITY_LOWWI_M7_VERDICT_20260516.md` (superseded)
- `bench/viscoelastic_logfv/CAVITY_LOWWI_M7B_VERDICT_20260516.md` (SMOKING GUN)
- `bench/viscoelastic_audit/BSD_KINETIC_MOMENT_DESIGN_20260516.md` (M5-A)
- `bench/viscoelastic_audit/WALL_BC_POLYMER_STRESS_AUDIT_20260516.md` (M6-A)
- `bench/viscoelastic_audit/BSD_GUO_WI0_AUDIT_20260516.md` (M10) — **the algebraic root-cause doc**
- `bench/viscoelastic_audit/BSD_LITERATURE_AUDIT_20260516.md` (M12) — **rheoTool & Liu BSD patterns**
- `bench/viscoelastic_audit/CAVITY_PIPELINE_ARCH_AUDIT_20260516.md` (M15) — **9-fault map + Option 3**

### Orchestrator scaffolding

- `.orchestrator/mandate.md` — full mission ledger M1..M15 with statuses.
- `.orchestrator/memory/boss.md`, `department.md`, `engineer.md` —
  filtered project facts; read at session start.

### Aqua data (synced locally under `tmp/`)

- `tmp/cavity_aqua_n64/` — original baseline (pre-sweep)
- `tmp/cavity_remismatch/{u0.001, u0.002, u0.005}/` — M1 Re sweep
- `tmp/cavity_bsd_sweep/{bsd0_0, bsd0_25, bsd0_5, bsd0_75}/` — M4b
- `tmp/cavity_m6b_confirm/{quadratic, linear}/` — M6-B
- `tmp/cavity_lowwi_sanity/{polymer_on, nu_p_zero}/` — M7
- `tmp/cavity_lowwi_matched_visc/{A_polymer_on, B_matched, C_re_ref}/` — M7b
- `tmp/cavity_grid_convergence/` — M9 (incoming; N=32, 64, 96
  present at session-end; N=128 pending Aqua completion ~21:00 AEST)

## What NOT to do

- **Do NOT** retry M11-style same-stencil fix on the monolithic
  driver. It will break the same way. Split first.
- **Do NOT** flip `bsd_kind=:kinetic` as default — it NaN-crashes
  in cavity at production cadence (kinetic path needs more work
  than M5-B's smooth-only validation).
- **Do NOT** set `bsd_fraction=0` to "simplify" — BSD is load-bearing
  for LBM stability; disabling it crashes the polymer_on case.
- **Do NOT** re-run any of the M1, M3, M4b, M6-B, M8, M13 tests
  unless you have a specific new hypothesis. They are ratched.
- **Do NOT** chase line numbers cited in older briefs literally —
  M14's `diagnose_bsd_dual` shifted everything in the driver.
  Always `grep -n` for the symbol first.
- **Do NOT** bundle "split driver" with "fix BSD" in one mission.
  Split is M16; fix is M17.

## Session metadata

- Branch: `dev-viscoelastic`. HEAD at session end: the M14
  `diagnose_bsd_dual` instrumentation is staged in the working tree
  but the M11 and intra-session kinetic-default experiments are
  reverted. Run `git status` first to confirm state.
- 4 commits posted today on dev-viscoelastic: M1 verdict, M5-B
  infra, M6-B verdict + threading, M7b verdict, M10 audit, M12-15
  audits. (~12 commits 2026-05-15..16 combined.)
- Sibling branch `dev/fvfd-core` HEAD `7c790cd8` carries the
  `polymer_wall_extrap` kwarg for the FVFD library.
- Aqua job `21405282.aqua` (M9 grid conv) was still running at
  session end. Check
  `tmp/cavity_grid_convergence/running_l2.txt` for progress, OR
  poll `qstat -fx 21405282.aqua` on Aqua.

End of prompt.
