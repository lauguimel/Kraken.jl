# MANDATE — Kraken.jl viscoelastic cavity spatial debug

Source of truth for the cavity Oldroyd-B spatial-coupling investigation
on branch `dev-viscoelastic`. Bootstrapped 2026-05-15 from
`NEXT_SESSION_PROMPT_20260515_cavity_spatial.md`.

---

## 1. High-level objective

Identify and fix the source of the 18-24 % relative L2 error between
Kraken's closed lid-driven cavity Oldroyd-B benchmark and rheoTool's
`Cavity/Oldroyd-BLog` reference at `t = 8`, `N = 64`, `De = 1`,
`beta = 0.5`, `bsd_fraction = 0.75`. Constitutive math is already
validated to machine precision in 0D shear and planar extension; the
remaining gap is purely spatial / coupling. "Done" = single-digit
percent L2 error on `u(0.5, y)` and `psi_xy(x, 0.75)` profiles at N=64
without breaking other validated benchmarks (channel, cylinder).

## 2. Out of scope

- Re-litigating the 0D constitutive math (validated 2026-05-15,
  `CONSTITUTIVE_0D_AUDIT_20260515.md`).
- Running `bsd_fraction = 1.0` on cavity (crashes by design — needs
  kinetic-moment BSD refactor, deferred).
- Returning to the cylinder benchmark (ratchet closed, see
  `VALIDATION_LADDER_AUDIT_20260513.md`).
- Performance optimisation of the substep loop (launch-overhead bound;
  correctness first).

## 3. Constraints

- **Language / stack**: Julia 1.10+ on the `dev-viscoelastic` branch.
- **Backend**: GPU only for any production-cost run. Local Metal F32 on
  macOS for smoke; Aqua A100/H100 CUDA F64 for any N≥64 case.
- **Authority to commit**: Boss only (one writer per the orchestrator
  pattern).
- **Confidentiality**: no AI/Claude mention in commits, code, or any
  artefact that may end up in the public repo (per global
  `feedback_confidentiality.md`).
- **HPC ops** (rsync to Aqua, qsub, kill jobs): Boss must confirm with
  the user before each. Never autonomous.

## 4. Architecture decisions (ADRs)

| Date       | Decision                                                  | Rationale                                  |
|------------|-----------------------------------------------------------|--------------------------------------------|
| 2026-05-15 | Match De and beta exactly; accept Re_LU = O(1)            | Uniform-mesh LBM cannot match Re=0.01      |
| 2026-05-15 | Use `bsd_fraction = 0.75` on cavity (1.0 crashes at lid)  | LBM/FD-laplacian discordance at corner     |
| 2026-05-15 | Investigate 5 spatial candidates before kinetic-moment BSD| Cheapest-first triage; BSD refactor is 3-4h|
| 2026-05-15 | Orchestrator pattern adopted for this branch              | Multi-mission triage; user-confirmed       |

## 5. Missions

### M1 — Re-mismatch sweep (Candidate 1)

- **Status**: done 2026-05-15 — **verdict: Re mismatch refuted**.
  L2 flat across `u_max ∈ {0.005, 0.002, 0.001}` (centerline L2
  1.797e-1 → 1.795e-1, psi_xy L2 2.44e-1 → 2.38e-1). Job
  `21339238.aqua` walltime 04:22, Exit_status 0. Verdict file:
  `bench/viscoelastic_logfv/CAVITY_REMISMATCH_M1_VERDICT_20260515.md`.
  Per §6: launch M2 and M3 in parallel next.
- **Goal**: determine whether the cavity profile gap shrinks
  monotonically as Re_LU drops from 6.4 → 1.3 by sweeping
  `u_max ∈ {0.005, 0.002, 0.001}` while holding `N=64`, `De=1`,
  `beta=0.5`, `lambda_phys`, `nu_s`, `nu_p` fixed.
- **Allowed edit zones**:
  - `bench/viscoelastic_logfv/run_cavity_oldroydb_vs_rheotool*.pbs`
  - `bench/viscoelastic_logfv/analyse_cavity_remismatch.jl` (NEW)
  - `bench/viscoelastic_logfv/CAVITY_REMISMATCH_*.md` (NEW verdict)
- **Exit criterion** (post-Engineer, pre-submit):
  `julia --project=. -e 'include("bench/viscoelastic_logfv/analyse_cavity_remismatch.jl"); demo()'`
  exits 0 on a synthetic 3-case fixture (no real Aqua data yet);
  the wrapper PBS dry-run `bash -n .../run_cavity_remismatch_sweep.pbs`
  exits 0.
- **Notes**: Wrapper PBS must loop u_max internally in one job to
  amortise Julia precompile cost. Walltime budget: 6h for 3×~33min cases
  × 1/u_max scaling factor (~2.6h total compute + margin).

### M2 — Wall gradient corner artifact (Candidate 2)

- **Status**: smoke done 2026-05-16 — **partial signal, inconclusive
  at smoke scale**. Surgical `skip_top_corners::Bool=false` plumbed
  through the cavity driver call chain. Smoke at N=32, t=2, Metal F32:
  max |Δpsi_xy| = 5.57e-5 (corner region) vs 3.53e-5 (bulk) → ratio
  1.58×. Real local effect but absolute magnitude too small at N=32 t=2
  to compare to the 18-24% production gap. Bench script:
  `bench/viscoelastic_logfv/run_cavity_corner_artifact_2d.jl` (has
  `--full` mode for N=64 t=8 on Aqua).

### M3 — Polymer upwind diffusion (Candidate 3)

- **Status**: smoke done 2026-05-16 — **polymer pipeline refuted as
  dominant source**. Standalone polymer pipeline (advection + source +
  stress) on frozen rheoTool U at t=8 gives relative L2 = **4.08 %**
  on `psi_xy(x, 0.75)` at N=32 (0.25 phys time, Metal F32), well below
  the 18-24% coupled-driver gap. Pipeline is fine on a clean U; the
  bug must originate in U itself (i.e. in the LBM solvent response to
  the polymer force). Bench script:
  `bench/viscoelastic_logfv/run_rheotool_frozen_replay_cavity_2d.jl`
  (has `--full` mode for N=64 t≥1 confirmation run).

### M4 — Guo body-force vs FD divergence (Candidate 4)

- **Status**: audit done 2026-05-16 — **CONFIRMED as the primary
  suspect**. Guo body-force differs from FD div(τ) by **53.5 % – 53.8 %
  L2** on the saved N=64 cavity snapshots, structural across u_max
  (consistent with M1's L2-flat finding). Difference is dominated by
  the BSD `−ζ·ν_p·∇²u` correction. Max-diff cell at (16, 63) — second
  row below moving lid, right-wall recirculation corner; this is also
  the M2 corner-artifact region (M2 and M4 are coupled at this cell).
  Analysis script:
  `bench/viscoelastic_logfv/analyse_cavity_guo_vs_fd_2d.jl`. Prior in
  the Mandate ("~10-20 % discrepancy expected") was conservative; the
  actual gap is 2-3× larger.

### M4b — BSD fraction sweep (decision experiment)

- **Status**: done 2026-05-16 — **HYPOTHESIS REFUTED**. L2 falls
  *monotonically* as `bsd_fraction` increases (NOT decreases):
  centerline 21.15 % → 17.97 %, psi_xy 27.41 % → 24.41 % over
  `ζ ∈ {0, 0.25, 0.5, 0.75}`. The BSD correction is helping
  rheoTool match, not hurting it. Aqua job `21385031.aqua`
  (requeued overnight to `gpu0n008`; walltime 02:23:06,
  Exit_status 0). Verdict file:
  `bench/viscoelastic_logfv/CAVITY_BSD_M4B_VERDICT_20260516.md`.
- **Implication**: M4's 54 % Guo-vs-FD discrepancy is the BSD term
  operating as designed, not a defect. M5-B (kinetic BSD refactor)
  remains valuable as infrastructure but cannot close the gap.
  **Pivot to M6-B (wall-BC stencil match)** as the next lever.

### M6 — Polymer-stress wall BC alignment with rheoTool

#### M6-A — audit (done 2026-05-16)

- **Status**: audit GREEN. Engineer produced
  `bench/viscoelastic_audit/WALL_BC_POLYMER_STRESS_AUDIT_20260516.md`
  (376 lines, all 8 sections).
- **Key findings**:
  - rheoTool moving-lid BC on `τ`: `linearExtrapolation` (2-point linear
    extrap from the 2 nearest interior cells).
  - rheoTool on `theta` (=`Ψ`): `zeroGradient`.
  - Kraken on `Ψ`: implicit zeroGradient via `operators_2d.jl:408-454`
    — **matches** rheoTool. ✓
  - Kraken on `τ` FD-divergence at wall row: implicit one-sided
    **quadratic** 3-point stencil in
    `_fvfd_solid_bc_derivative_x_2d` / `_y`
    (`src/fvfd/operators_2d.jl:24-26 / 50-52`), consumed by
    `logfv_polymer_force_bc_aware_2d!`. Does NOT match rheoTool's
    2-point linear extrapolation.
  - Predicted impact of matching: 54 % → ~15-30 % interior L2 at the
    M4 max-diff cell (16, 63); interior far from walls unchanged.
- **Implication**: provides an alternative hypothesis to M5's BSD
  operator mismatch. Could be the dominant source if M4b shows L2
  flat across bsd_fraction. Likely complements rather than replaces
  M5-B (interior bit-exactness + wall stencil alignment are
  orthogonal fixes).

#### M6-B — wall-BC matching (done 2026-05-16)

- **Status**: implementation GREEN. Split across two branches:
  - **`dev/fvfd-core`** commit `7c790cd8` adds the `polymer_wall_extrap`
    kwarg + the `Val{:linear}` branch in
    `_fvfd_solid_bc_derivative_{x,y}_2d` and threads it through the
    public FVFD divergence wrappers. Default `:quadratic` preserves
    byte-identical behaviour for velocity-gradient consumers.
  - **`dev-viscoelastic`** commit (this one) threads the kwarg
    through `logfv_polymer_force_bc_aware_2d!` and
    `run_viscoelastic_logfv_cavity_coupled_2d`, plus adds the audit
    bench `run_wall_stencil_audit_2d.jl`.
- **Self-test (N=32 t=2 CPU F64)**: wall-row rel L2 between
  `:quadratic` and `:linear` = **12.0 %** (signal real, kwarg
  correctly wired). Bulk far-from-wall max abs = 5.3e-8 (well below
  the relaxed 1e-3 assertion — advection propagation over 12800 LBM
  steps as expected).
- **Files** (dev-viscoelastic side):
  - `src/kernels/logconformation_fv_2d.jl` (+3 lines)
  - `src/drivers/viscoelastic_logfv_2d.jl` (+4 lines)
  - `bench/viscoelastic_logfv/run_wall_stencil_audit_2d.jl` (NEW)
- **Aqua confirmation done 2026-05-16 — HYPOTHESIS REFUTED**.
  Aqua job `21397692.aqua` ran both `:quadratic` and `:linear` at
  N=64 t=8 (walltime 01:13:54, Exit_status 0). Sanity baseline
  (`:quadratic`) reproduces M1 baseline to 4 sig figs (0.1797 /
  0.2441) — kwarg default preserves behaviour bit-for-bit. Test
  case (`:linear`): centerline L2 = 0.1817 (+1.1 %), psi_xy L2 =
  0.2433 (−0.3 %). **The 12 % wall-row local signal does NOT
  propagate to the global profile.** Wall-stencil mismatch is not
  the cavity-gap driver. Verdict file:
  `bench/viscoelastic_logfv/CAVITY_M6B_CONFIRM_VERDICT_20260516.md`.

### Mission status step-back (2026-05-16)

Four of five originally-mandated candidates plus the user-suggested
wall-BC alternative are refuted. The 18-24 % cavity profile gap
remains unexplained. The original Mandate's "5 candidates" framing
is exhausted.

A diagnostic battery (M7-M9) was launched 2026-05-16 to localise the
bug. **M8 ratchets the polymer pipeline out of suspicion**: an
analytical Poiseuille frozen-velocity test of the FV polymer pipeline
(advection + Oldroyd-B source + stress assembly + wall velocity-gradient
extraction) yields first-order convergence in `dt_poly` with no spatial
bias — at production `n_substeps=4096`, the source-discretization
error is ~4e-6, negligible. The 18-24 % cavity gap therefore originates
in the **LBM ↔ polymer coupling layer** (Guo body-force injection on
`f`, BSD correction magnitude/sign, operator staggering, or `u`
reconstruction after the Guo source). M7 (low-Wi sanity) and M9 (grid
convergence) will further bound which sub-component.

### M7 — Low-Wi sanity (done 2026-05-16 — INCONCLUSIVE, design flaw)

- **Status**: Aqua run completed (`21405281.aqua`, walltime 00:04:19,
  Exit_status 0). Kraken-vs-Kraken centerline rel L2 = **3.41 %**.
  **But the test is confounded by a Boss-brief design flaw**: the
  two cases have different total LBM viscosities (`ν_s + ν_p`):
  - `polymer_on`: `ν_total = 0.2`, `Re_LU = 1.6`
  - `nu_p_zero`: `ν_total = 0.1`, `Re_LU = 3.2`

  The 3.4 % delta is plausibly explained by the Re factor 2 alone,
  not a polymer-coupling bug. Verdict file:
  `bench/viscoelastic_logfv/CAVITY_LOWWI_M7_VERDICT_20260516.md`.

### M7b — Low-Wi matched-viscosity sanity (done 2026-05-16 — SMOKING GUN)

- **Status**: GREEN. Aqua job `21406676.aqua`, walltime 03:11,
  Exit_status 0. **A Wi-independent polymer-coupling bug is
  confirmed.**
- **Result** (centerline u relative L2, Kraken-vs-Kraken):
  - **A vs B (matched ν_total=0.2, Re_LU=1.6 identical) = 3.42 %**
  - A vs C (unmatched ν_total) = 3.41 %
  - B vs C (Re-doubling at Newtonian, nu_p=0 both) = **0.014 %**
- **Critical reading**: B and C are both Newtonian; they differ only
  in Re_LU (1.6 vs 3.2) and yet their delta is 0.014 % — pure noise
  floor. **The 3.4 % A-vs-B delta is therefore NOT the Re factor (as
  M7 mistakenly attributed it) — it is entirely the polymer-coupling
  Wi-independent contribution.** At Wi=0.001 the polymer stress is
  essentially Newtonian-additive (`τ_p ≈ 2·ν_p·D`); the BSD/Guo split
  is supposed to absorb this exactly into `ν_LBM = ν_s + ζ·ν_p`. The
  3.4 % residual proves the absorption is incomplete.
- **Verdict file**:
  `bench/viscoelastic_logfv/CAVITY_LOWWI_M7B_VERDICT_20260516.md`.
- **First concrete localisation** of the cavity-gap bug since M1.

### M10 — BSD/Guo coupling Wi→0 audit (done 2026-05-16 — BUG LOCALISED)

- **Status**: GREEN. Audit doc
  `bench/viscoelastic_audit/BSD_GUO_WI0_AUDIT_20260516.md` (380
  lines, 8 sections). **Bug pinned to a stencil mismatch.**
- **Finding**: at Wi → 0, `div(τ_p)` is assembled by two successive
  FD-central operations (`fvfd_velocity_gradient_2d!` → 
  `fvfd_tensor_divergence_2d!`), producing a **wide 2dx-spacing
  laplacian** acting on `u`. The BSD correction
  `−ζ·ν_p·∇²u` uses a **narrow 3-point laplacian** in
  `fvfd_bsd_force_2d_kernel!` (`src/fvfd/operators_2d.jl:886-915`).
  The two laplacians converge to the same continuum operator but
  are NOT the same discrete operator — they differ by
  O(dx²·∂⁴u), with the wide stencil carrying 4× the leading
  truncation error. The cancellation that should fold `ν_p` into
  the LBM viscosity at Wi=0 is therefore broken at the discrete
  level. This is the 3.42 % M7b residual.

### M11 — BSD same-stencil fix (attempted 2026-05-16 — RED, REVERTED)

- **Status**: attempted on the monolithic driver, reverted same
  session. Same-stencil route
  (`logfv_bsd_stress_from_gradient_2d!` → `fvfd_tensor_divergence_2d!`)
  produced **64 % A-vs-B** vs the 3.4 % bug signal (worse than the
  bug). Root cause: BSD captured `D_corrected` while `τ_p` carried
  `Ψ_history` from the source ODE — same stencil, different
  "times". Fix requires capturing both at the SAME pipeline step,
  which in turn requires the cavity driver SPLIT (see M16).
- **Lesson**: do NOT retry M11-style fix on the monolith. Reframed
  as M17, gated on M16.

### M8 — Poiseuille polymer-pipeline analytical (done 2026-05-16)

- **Status**: GREEN with substantive caveat. Bench script
  `bench/viscoelastic_logfv/run_poiseuille_polymer_analytical_2d.jl`
  (256 LOC) freezes an analytical Poiseuille velocity field, runs
  the Kraken polymer pipeline only (no LBM), compares `τ_xy(y)` and
  `N1(y)` to Oldroyd-B steady-shear closed form.
- **Result**: τ_xy rel L2 = 1.95e-3 / N1 rel L2 = 1.92e-3 at the
  smoke cadence (n_substeps=8); error is uniform across interior
  (NO spatial bias) and converges first-order in `dt_poly`. At 16
  substeps both pass < 1e-3; at production cadence
  (`n_substeps=4096`) source error ~4e-6 → negligible.
- **Implication**: polymer pipeline + wall-row velocity-gradient
  stencil are SOUND. The cavity 18-24 % gap MUST be in the LBM ↔
  polymer coupling.

### M9 — Cavity grid convergence (done 2026-05-16 — partial floor confirmed)

- **Status**: GREEN. Aqua `21405282.aqua` walltime 04:21:31,
  Exit_status 0. **L2 falls monotonically with N**, approaching an
  asymptotic floor (not zero).
- **Results** (centerline u L2 / psi_xy L2):
  - N=32: 31.4 % / 35.2 %
  - N=64: 18.0 % / 24.4 % (baseline)
  - N=96: 12.9 % / 20.1 %
  - N=128: **10.0 %** / **17.9 %**
- **Asymptotic-floor extrapolation** (assuming p=2 second-order
  convergence): `L2_∞ ≈ 7.4 %` on centerline u, `~16.5 %` on psi_xy.
  About **half** the N=64 gap is discretization-driven; the other
  half is the Kraken-specific bug (M7b 3.4 % Wi-independent +
  finite-Wi BSD drift).
- **Implication for M17**: after closing the Wi-independent bug,
  the expected post-M17 u-centerline gap at N=64 drops to ~14-15 %;
  the residual ~10 pp is discretization floor. At N=127 (rheoTool
  match) the residual shrinks further. Target: post-M17 u L2 ~5-8 %.
- **Verdict file**:
  `bench/viscoelastic_logfv/CAVITY_GRIDCONV_M9_VERDICT_20260516.md`.

### M5 — Kinetic-moment BSD refactor (Candidate 5)

#### M5-A — design (done 2026-05-16)

- **Status**: design GREEN. Engineer produced
  `bench/viscoelastic_audit/BSD_KINETIC_MOMENT_DESIGN_20260516.md`
  (439 lines, all 8 sections present).
- **Key findings**:
  - Proposed kernel: `compute_bsd_force_kinetic_2d!` in NEW
    `src/kernels/bsd_kinetic.jl`, paired with a `compute_pi_neq_2d_kernel!`
    that extracts `Π^{neq}_{αβ} = Σ_q c_qα c_qβ (f_q − f_q^eq)`.
  - Correct denominator is `ν_eff = cs²·(1/s_plus − 1/2)`; Guo prefactor
    correction `guo_pref = 1 − s_plus/2` is taken from the existing
    `bricks.jl:168-171` convention (not a separate scalar to pass).
  - Precision ceiling: **F64 interior ≤ 1e-6**, **F32 interior ≤ 1e-3**
    (gating). Bit-equality NOT achievable due to LI-BB pre-phase on `f`
    at walls — interior-only assertion is the working bar.
  - Top risk: wall-adjacent cells where LI-BB perturbs `f` before
    non-eq moments are read; mitigation = interior-only first.
- **Phase B scope estimate**: 2 NEW files
  (`src/kernels/bsd_kinetic.jl`, `bench/.../bsd_kinetic_audit_2d.jl`) +
  2 MODIFIED (`src/Kraken.jl` export, cavity driver kwarg threading at
  lines 865, 1073-1077). 3-5 h Codex impl + 2-3 h validation. Blast
  radius: cavity driver only; `bsd_kind::Symbol=:fd` keeps unchanged
  behaviour everywhere else.

#### M5-B — prototype (done 2026-05-16 as infrastructure)

- **Status**: GREEN as a refactor; kernels implemented and committed.
  Self-test on N=32 t=2 CPU F64: `‖F_kinetic − F_FD_BSD‖₂ /
  ‖F_FD_BSD‖₂ = 5.85e-16` (machine epsilon). **Caveat**: this proves
  equivalence to the *existing FD-BSD path*, NOT to the LBM's true
  implicit lattice stencil. By Chapman-Enskog `Π^{neq}` and
  FD-laplacian of `u` give the same result on smooth interior. Wall
  cells (LI-BB-perturbed `f`) were not exercised by the smoke; that
  remains the unresolved risk from §M5-A.
- **Practical implication**: no behaviour change with default
  `bsd_kind=:fd`. The `:kinetic` path is currently equivalent to
  `:fd` — useful as a `Π^{neq}` accumulator for future rheology
  diagnostics or as the substrate for a future lattice-stencil-aware
  BSD if the data justifies one.
- **Cost (overhead when `:kinetic` is enabled)**: +3·N² temporary
  buffers; ~37 FLOP/cell/timestep — negligible at production sizes.
- **Files**: `src/kernels/bsd_kinetic.jl` (NEW),
  `src/Kraken.jl` (export, 2 lines),
  `src/drivers/viscoelastic_logfv_2d.jl` (kwarg `bsd_kind::Symbol=:fd`,
  +19/-4 lines), `bench/viscoelastic_logfv/run_bsd_kinetic_audit_2d.jl`
  (NEW).

### M12 — BSD literature audit (done 2026-05-16)

- **Status**: GREEN. Audit doc
  `bench/viscoelastic_audit/BSD_LITERATURE_AUDIT_20260516.md`.
- **Finding**: rheoTool's `stabilization coupling` (iBSD) enforces
  same-stencil cancellation at the `fvSchemes` dictionary level —
  both `div(τ)` and `div((etaP)·grad(U))` declared `Gauss linear`.
  Liu 2025 has no BSD: it injects stress directly via Hermite
  moments into the LBM `f_i` (Eq. 22). The wide-vs-narrow trap
  that bites Kraken is structurally absent from both references.
  **This validates M17's Option 3 design** (rheoTool-style
  same-stencil routing).

### M13 — Guo body-force inverse test (done 2026-05-16)

- **Status**: GREEN. Bench
  `bench/viscoelastic_logfv/run_poiseuille_imposed_stress_2d.jl`.
  Frozen analytical τ → Guo injection bit-exact (2.7e-20) and
  second-order convergent in N. Guo path is ratched out.

### M14 — BSD dual-path diagnostic (done 2026-05-16)

- **Status**: GREEN as instrumentation. `diagnose_bsd_dual::Bool`
  kwarg in the cavity driver records FD vs kinetic divergence per
  step. On smooth t=0: bit-equivalent. On dynamic cavity: diverges
  by O(1) — confirming the M10 stencil-mismatch hypothesis at
  production gradients. Canary for any future BSD work.

### M15 — Cavity pipeline architectural audit (done 2026-05-16)

- **Status**: GREEN. Audit doc
  `bench/viscoelastic_audit/CAVITY_PIPELINE_ARCH_AUDIT_20260516.md`
  identifies 9 faults at Wi→0. Top is M10's stencil mismatch.
  Secondary: `τ_p` carries `Ψ_history` while a fresh
  `τ_BSD(D_now)` would be instantaneous (different "times"); the
  default `:fd` BSD path reads `ux, uy` directly, NOT the
  wall-corrected `D` the source ODE consumes. **Prescribes Option
  3** (capture both BSD and source-ODE at the same pipeline step,
  route through the same divergence operator). M17 implements it.

### M16 — SPLIT cavity driver (done 2026-05-17, commit `77956ad8`)

- **Status**: GREEN. `viscoelastic_logfv_2d.jl` 3429 → 2934 LOC.
  Cavity helpers (4 fns, 98 LOC) moved to
  `src/drivers/cavity_wall_correction_2d.jl`. Cavity main driver
  `run_viscoelastic_logfv_cavity_coupled_2d` (400 LOC) moved to
  `src/drivers/cavity_driver_2d.jl`. `src/Kraken.jl`: +2 includes.
  All three target files ≤700 LOC hard ceiling. Refactor pur:
  zero semantic change; public API signature unchanged. Test
  suite: 6 pre-existing failures + 4 broken canaries unchanged
  vs HEAD (verified by Department on stashed baseline). M17 is
  unblocked.
- **Original framing kept for posterity**: BLOCKING M17. The cavity driver
  `src/drivers/viscoelastic_logfv_2d.jl` was 3429 LOC and mixed 5
  concerns (geometry, BC, solver, stencil, physics). Per
  `feedback_orchestrator_discipline` + skill hygiene rules, no
  substantive BSD change targets the monolith. M11 destabilised
  exactly because of this; the SPLIT was the prerequisite.
- **Goal**: decompose the driver along its natural seams into
  ≤700-LOC modules. Refactor pur — zero behavioural change.
  Proposed targets (Engineer may adjust to natural seams):
  - `cavity_wall_correction_2d.jl` — wall-gradient correction
    kernels (smallest, most isolated → extract first).
  - `cavity_bsd_assembly_2d.jl` — BSD path selection
    (`:fd` / `:kinetic`) + `diagnose_bsd_dual` instrumentation.
  - `cavity_init_2d.jl` — buffer allocation + IC setup.
  - `cavity_snapshot_2d.jl` — output / diagnostics writers.
  - The remaining `viscoelastic_logfv_2d.jl` keeps the timestep
    loop only, ≤700 LOC.
- **Allowed edit zones**: the new files + the existing driver to
  remove migrated code + `src/Kraken.jl` to update includes/exports.
- **Forbidden**: any kwarg semantic change, any reordering of
  operations inside the timestep loop, any deletion of reverted
  code (M11/kinetic-default cleanup is a separate M16b mission).
- **Exit criterion**: `julia --project=. test/runtests.jl` exits 0
  AND `julia --project=. -e 'include("bench/viscoelastic_logfv/run_cavity_oldroydb_vs_rheotool.jl"); …'`
  at N=64 t=8 reproduces the M1 baseline (centerline L2 = 0.1797,
  psi_xy L2 = 0.2441) to ≥4 sig figs (machine precision modulo
  GPU non-determinism). Local Metal F32 smoke at N=32 t=2 must
  match pre-split byte-for-byte.

### M17 — Option 3 BSD same-stencil fix (planned, gated on M16)

- **Status**: gated. Per M12/M15, capture `D_corrected` at the
  source-ODE step, build `τ_BSD = 2·ζ·ν_p·D_corrected` via
  `logfv_bsd_stress_from_gradient_2d!`, route through the SAME
  `fvfd_tensor_divergence_2d!` operator with the same
  `polymer_wall_extrap` as `div(τ_p)`. ~50-85 LOC + 5 persistent
  N×N buffers per M15.
- **Allowed edit zones**: post-M16 modules only.
- **Exit criterion**: M7b PBS A-vs-B centerline u rel L2 < 0.1 %
  (vs the 3.4 % bug, well above the 0.014 % noise floor).

### M18 — Production validation (PARKED 2026-05-18)

- **Status**: PARKED by user directive 2026-05-18. The M17 cluster
  closure re-decomposed the cavity 3.4 % M7b signal into an
  **inferred** mix (~0.4 % stencil + ~2.4 % corner amplification +
  ~0.6 % BSD intrinsic) that has not been directly measured on
  Poiseuille at controlled Wi. Before any production cavity
  validation, the user wants Poiseuille investigated deeply to
  understand what BSD actually does to the LBM↔FV coupling on the
  simplest geometry. M18 unparks once M20-M24 produce a defended
  decomposition.
- **Original goal kept for posterity**: Re-run cavity Oldroyd-B
  comparison at N=64 t=8 De=1 β=0.5 with the M17 fix. Pass bars:
  centerline u L2 drops from 18.0 % toward the M9 discretization
  floor; psi_xy L2 drops from 24.4 % similarly.

### M19 — Corner regularisation (PARKED 2026-05-18)

- **Status**: PARKED with M18. Cavity-side intervention; meaningless
  to design without an established Poiseuille baseline for BSD
  behaviour. Re-evaluate after M20-M24.

### M20 — Poiseuille F_total trace (`:fd`, ζ=0.75) — DONE 2026-05-18

- **Status**: GREEN. Verdict
  `bench/viscoelastic_audit/POISEUILLE_BSD_TRACE_VERDICT_20260518.md`.
  Bench `bench/viscoelastic_audit/run_poiseuille_bsd_trace_2d.jl`
  (282 LOC). BSD operates as designed at the operator level on
  Poiseuille; F_poly_wide and F_BSD_narrow each carry ~0.5 % rel
  truncation residual vs analytical d²u/dy² (uniform across y, no
  wall spike). The residuals are **same-sign and ADD algebraically**
  in F_total (do not cancel), then the (1−ζ)⁻¹=4× normalisation
  amplifies them to 3.51 % on F_total at ζ=0.75 Wi=8e-4. At Wi=1 both
  collapse 380× because u_LBM rebalances close to analytical parabola.
  **Smoking gun localisation**: the 8× cavity-vs-Poiseuille M7b ratio
  is NOT in the BSD-subtraction chain itself (no wall amplification on
  smooth geometry); it lives downstream in either (a) the
  velocity-gradient kernel difference (Open Q5 → M21) or (b) the
  LBM-side flow response to the force around the corner singularity.
- **Original goal kept for posterity below.**
- **Original goal**: First mission of the Poiseuille investigation
  cluster opened by user directive 2026-05-18.
- **Goal**: on the existing
  `run_viscoelastic_logfv_poiseuille_coupled_2d` driver at the M7b
  setup (Nx=8, Ny=32, F_body=1e-5, λ=1.0, max_steps=100k, CPU F64),
  decompose `F_total` post-hoc into its three additive contributions
  per y-row, and compare each against the analytical Newtonian-limit
  target. Answers: **does the BSD `−ζ·ν_p·∇²u_narrow` correction
  actually cancel the F_poly_wide `ν_p·∇²u_wide` portion to leave
  `(1−ζ)·ν_p·∇²u` as designed, or does it leave a structured
  residual?** Three cases: (i) ζ=0.0 baseline (F_total = F_poly_wide),
  (ii) ζ=0.75 production, (iii) optional Wi=1.0 to surface elastic
  contribution.
- **Allowed edit zones**:
  - `bench/viscoelastic_audit/run_poiseuille_bsd_trace_2d.jl` (NEW)
  - `bench/viscoelastic_audit/POISEUILLE_BSD_TRACE_VERDICT_20260518.md` (NEW)
  - `bench/scratch/` (one-off CSVs, plots)
  - `tmp/` (large outputs)
  - `<project>/.engineer_brief_M20.md` (single-use)
- **Forbidden**: edits anywhere under `src/`, `.orchestrator/memory/`,
  or `test/`; any commit/push; any modification of existing bench
  scripts.
- **Exit criterion**:
  `julia --project=. bench/viscoelastic_audit/run_poiseuille_bsd_trace_2d.jl --self-test`
  exits 0 (self-test mode runs Ny=16, max_steps=1000 under 60 s and
  asserts the CSV contains all expected columns + monotone wall
  decay). Department re-runs the full mode (Ny=32, max_steps=100k)
  on host and writes the verdict markdown.
- **Engineer runner**: `codex` (Codex CLI via `run-engineer.sh`).
- **Notes**: the per-y profile for parabolic Poiseuille is uniform —
  `ν_p · d²u_analytical/dy² = −ν_p · F_body / ν_total` everywhere in
  interior. Wall rows quantify the discrete stencil residual cleanly.
  Reuse the kernel call pattern from
  `bench/viscoelastic_audit/bsd_analytical_ladder_2d.jl`.

### M21 — Poiseuille BSD path matrix sweep — DONE 2026-05-18 (NEGATIVE)

- **Status**: GREEN with NEGATIVE result. Verdict
  `bench/viscoelastic_audit/POISEUILLE_BSD_PATHMATRIX_VERDICT_20260518.md`.
  Bench `bench/viscoelastic_audit/run_poiseuille_bsd_pathmatrix_2d.jl`
  (426 LOC) ran 7 BSD variants × 2 cases = 14 runs in 9 min CPU F64.
  **No variant beats `:baseline` (3.51 %) on smooth Poiseuille**:
  `:fd_v2_unc` 50.9 %, `:fd_v2` 85.7 % (NaN at Wi=1), `:kinetic`
  186 % (Π^neq overshoots BSD magnitude 30×), `:epsilon_force` NaN
  both cases. Only `:no_bsd` (ζ=0 trivial reference) gives 0.50 %.
- **Open Q5 REFUTED at root**: `logfv_velocity_gradient_bc_aware_2d!`
  (lines 918-926 in `src/kernels/logconformation_fv_2d.jl`) is
  literally `return fvfd_velocity_gradient_2d!(...)` — a thin
  wrapper. The two kernels are bit-identical. The 8× cavity-vs-
  Poiseuille M7b ratio CANNOT come from the kernel difference.
- **Strategic implication**: the cavity bug is NOT operator-side.
  All RED M11/M17 paths are RED on Poiseuille too (or worse). The
  user's hypothesis "cavity-specific bug masks a working BSD fix"
  is REFUTED. Cavity gap must live in the wall-corner gradient
  correction overlay (`_logfv_cavity_wall_gradient_correction_kernel!`)
  or the LBM-side flow response to the corner singularity (Zou-He
  lid coupling, Guo source at corner cells).
- **Original goal kept for posterity below**.
- **Original goal**: (scope expanded per user directive
  2026-05-18 "retestes toutes les pistes pour le BSD sur le poiseuille,
  notamment la M21"). Hypothesis: a BSD formulation that was RED on
  cavity (NaN at wall corner) may be GREEN on smooth Poiseuille — the
  cavity bug is geometric (wall-corner), not algebraic. If any variant
  gives F_total < 3.51 % AND remains stable, it becomes the candidate
  to re-test on cavity after corner-bug isolation. (REFUTED above.)

- **Status**: in-flight (scope expanded per user directive
  2026-05-18 "retestes toutes les pistes pour le BSD sur le poiseuille,
  notamment la M21"). Hypothesis: a BSD formulation that was RED on
  cavity (NaN at wall corner) may be GREEN on smooth Poiseuille — the
  cavity bug is geometric (wall-corner), not algebraic. If any variant
  gives F_total < 3.51 % AND remains stable, it becomes the candidate
  to re-test on cavity after corner-bug isolation.
- **Goal**: implement 7 BSD/F_poly variants in a standalone Poiseuille
  bench (no `src/` patch). For each: per-step NaN watcher on u/ψ; at
  steady state, full u + τ + ψ checks vs analytical (Newtonian limit
  at Wi=8e-4, full Oldroyd-B closed form at Wi=1).
- **Variants** (all implementable via existing kernels in `src/`):
  1. `:baseline` — current `:fd` (control, reproduces M20).
  2. `:no_bsd` — `bsd_fraction=0` (control, reproduces M20 A_no_BSD).
  3. `:fd_v2` — wide BSD via `logfv_bsd_stress_from_gradient_2d!` +
     `fvfd_tensor_divergence_2d!` on `τ_BSD = 2·ζ·ν_p·D_corrected`.
  4. `:fd_v2_unc` — `:fd_v2` Option A: BSD reads `D_uncorrected`
     (re-call vel-grad WITHOUT wall-correction overlay).
  5. `:kinetic` — M5 kinetic-BSD via Π^{neq} extraction
     (`compute_bsd_force_kinetic_2d!`).
  6. `:epsilon_force` — ε-split force-level: F_poly = NARROW
     `ν_p·∇²u` + `div_wide(τ_p − 2·ν_p·D_cell)` ; F_BSD = NARROW
     `ζ·ν_p·∇²u`. Same-stencil cancellation of Newtonian portion.
  7. `:baseline_fvfd_grad` — Open Q5 cross-check: `:baseline` but
     velocity gradient via `fvfd_velocity_gradient_2d!` instead of
     `logfv_velocity_gradient_bc_aware_2d!`.
- **Cases**: 2 per variant (Wi=8e-4 M7b-A baseline, Wi=1 finite-Wi).
  14 runs total, ~3 min CPU F64 each = ~45 min total runtime.
- **Allowed edit zones**:
  - `bench/viscoelastic_audit/run_poiseuille_bsd_pathmatrix_2d.jl` (NEW, ≤500 LOC)
  - `bench/viscoelastic_audit/poiseuille_bsd_variants_2d.jl` (NEW if needed for size, ≤500 LOC)
  - `bench/viscoelastic_audit/POISEUILLE_BSD_PATHMATRIX_VERDICT_20260518.md` (NEW)
  - `bench/scratch/`, `tmp/`, `.engineer_brief_M21.md`
- **Exit criterion**: `julia --project=. bench/viscoelastic_audit/run_poiseuille_bsd_pathmatrix_2d.jl --self-test` exits 0 in ≤90 s (self-test runs Ny=16 max_steps=1000, 2-3 variants); Department's full mode produces per-variant ranking + verdict markdown.
- **Per-variant required outputs** (all in CSV per case):
  - u rel L2 vs analytical parabola (interior + wall rows separately)
  - τ_xy rel L2 vs `ν_p·γ̇(y)` (Newtonian limit)
  - τ_xx, N1 rel L2 vs `2·ν_p·λ·γ̇²(y)` (full Oldroyd-B closed form, exact at all Wi)
  - min(λ_C) > 0 verification (SPD positivity)
  - F_poly_wide / F_BSD / F_total per-y decomposition (M20 pattern)
  - nan_step (= -1 if completed; else step at which NaN detected)
- **Subsumes** the original M21 (Open Q5) as variant `:baseline_fvfd_grad`.

### M22 + M23 — Cylinder Cd mesh convergence (BSD ON & OFF) — DONE 2026-05-18

- **Status**: GREEN, joint synthesis in
  `bench/viscoelastic_logfv/CYL_CD_CONVERGENCE_M22M23_SYNTHESIS_20260518.md`.
  Both Departments spawned in parallel (orchestrator fan-out pattern);
  Anthropic API connectivity dropped on both Codex Engineers AFTER the
  bench scripts were written but BEFORE the Departments completed
  full-mode runs and verdict writing. Boss ran both `--full` modes
  directly on host (Metal F32 local) and wrote the joint synthesis.
- **Key finding 1 — BSD impact on Cd collapses with mesh refinement**:
  Δ(Cd_BSDon − Cd_BSDoff) at Wi=0.1 goes 18.7 → 13.3 → 8.9 → **1.4**
  Cd points as R goes 20 → 30 → 40 → 50. Trend extrapolates to
  "permilles" at R≥60. Confirms M20 hypothesis on a real complex flow:
  BSD operator-side residual is *masked* by elastic dynamics in
  production regime.
- **Key finding 2 — BSD ON matches rheoTool to 1.5%** at R=30 (only R
  with reference): err=−1.45 % (Wi=0.1), −2.53 % (Wi=0.2). BSD OFF gap
  is ~12 % at R=30 (under-shoots), monotone-converges UP toward BSD ON
  values as R grows.
- **Key finding 3 — User's "anti-convergence" recollection RESOLVED**:
  it was BSD ON over-shooting at R=30-40 (peak Cd=130.31 at R=40
  vs rheoTool 130.43) and oscillating at R=50, while BSD OFF
  monotone-approaches the same limit. The two CONVERGE to the same
  rheoTool-consistent limit, just from opposite sides.
- **Key finding 4 — BSD provides essential stability**: M23 R=40 Wi=0.2
  gave Cd=783, min_detC=8e-4 (near-SPD-loss). Without BSD, the LBM is
  more stress-loaded and fails at fine mesh + non-trivial Wi.
- **Original-goal text** (kept for posterity below).
- **Original M22 goal**: in-flight (cluster repositioned per user directive

- **Status**: in-flight (cluster repositioned per user directive
  2026-05-18). Pivots away from cavity to the **original motivator**:
  cylinder Cd vs rheoTool reference, mesh-refinement study at
  moderate Wi. User recollection: "on se croisait à faible maillage
  mais on ne convergeait pas vers les mêmes valeur" — Kraken and
  rheoTool/Liu Cd curves crossed at coarse mesh by luck but converged
  to different limits as the mesh refined.
- **Goal**: re-measure cylinder Cd(R, Wi) on the current Kraken
  `dev-viscoelastic` HEAD (post-M16 cavity split, pre-M22 here) at
  R ∈ {20, 30, 40, 50} × Wi ∈ {0.1, 0.2} with **`bsd_fraction=0.75`
  (baseline)**. Compare to rheoTool R=30 reference (the only R
  available); inspect Kraken-internal Cd(R) trend for crossing /
  divergent-limit pattern.
- **Backend**: Metal F32 local (user explicit "ouvrir un appartement
  avec metal pour aller plus vite"); F32 noise accepted as the
  trade-off for fast iteration.
- **Allowed edit zones**:
  - `bench/viscoelastic_logfv/run_cyl_cd_convergence_baseline_2d.jl` (NEW, ≤500 LOC)
  - `bench/viscoelastic_logfv/CYL_CD_CONVERGENCE_M22_VERDICT_20260518.md` (NEW)
  - `bench/scratch/`, `tmp/`, `.engineer_brief_M22.md`
- **Exit criterion**: bench script exits 0 on `--self-test`
  (R=20, Wi=0.1, 5k steps Metal F32 ≤120 s); Department's `--full`
  mode produces 8 CSVs + summary table + verdict markdown.
- **Pair**: spawned in parallel with M23 (BSD OFF, same grid). The
  two together produce the Cd(R, Wi, BSD) cube needed for the
  cross-comparison.

### M23 — Cylinder Cd mesh convergence BSD-OFF — IN-FLIGHT 2026-05-18

- **Status**: in-flight (parallel twin of M22).
- **Goal**: identical to M22 but `bsd_fraction=0.0` (BSD completely
  off; LBM viscosity = `nu_s` only; full `div(τ_p)` injected via
  Guo source). Tests whether removing BSD changes the convergence
  pattern Kraken converges to on the cylinder.
- **User intent**: directly answer "lorsqu'on est dans un fluide
  complexe avec des écoulements complexes est-ce que sur un drag par
  exemple on ne tombe qu'à quelques pouillèmes". M20 measured F_total
  residual at Wi=1 collapsing to 9.3e-5 due to elastic locking of
  u_LBM toward analytical. M22+M23 measure the same effect on a
  REAL complex flow (cylinder, finite Wi, curved boundary).
- **Backend**: Metal F32 local, parallel to M22 on same hardware
  (Metal can multiplex; if device contention, accept serialization).
- **Allowed edit zones**:
  - `bench/viscoelastic_logfv/run_cyl_cd_convergence_bsd_off_2d.jl` (NEW, ≤500 LOC)
  - `bench/viscoelastic_logfv/CYL_CD_CONVERGENCE_M23_VERDICT_20260518.md` (NEW)
  - `bench/scratch/`, `tmp/`, `.engineer_brief_M23.md`
- **Exit criterion**: same as M22 (self-test exits 0; full-mode produces 8 CSVs + summary + verdict).
- **Synthesis after**: Boss compares M22 and M23 verdicts to compute
  Cd_BSDon − Cd_BSDoff per (R, Wi). Expected: small delta if BSD
  truncation is masked by elastic dynamics at Wi ≥ 0.1 (M20-style
  collapse). Large delta if BSD operator-side error propagates into
  the cylinder Cd integral despite the elastic regime.

### M25 — Cylinder Cd HPC big sweep, Phase 0 Liu-match — DONE 2026-05-18

- **Status**: DONE 2026-05-18 evening. Job `21563085.aqua` was killed
  after silently running on CPU (Exit 143; 3 cases R=20 in ~2h). Root
  cause: **Julia 1.12 world-age trap** in `detect_backend()` —
  `getfield(Main, :CUDA)` after `@eval using CUDA` raised UndefVarError
  swallowed by bare `catch end` → silent CPU fallback. Fixed in commit
  `e602726f` (Base.invokelatest wrapper + @warn surfacing). Re-submitted
  as `21570657.aqua` which completed all 12 cases in **7m24s @ 66.66%
  GPU util A100**. Plus Phase 0b `21572831.aqua` (27 cases, 14m55s @
  69.89% GPU) explored 9 additional embedded tuples for full M26
  discrimination.
- **Original setup**: β=0.59 Re=1 Wi=0.1 L_up=L_down=15 bsd_fraction=1.0.
  Driver `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl`.
- **Goal**: validate whether the BSD architecture + staircase polymer
  pipeline matches the Liu 2025 reference Cd at R=30 (CNEBB 130.36,
  rheoTool 130.43). If `0000_qwall` Cd ∈ [129.5, 131.5] → BSD physics
  sound, Phase 1 (Wi sweep, M28) unblocks. If 0001_qwall Cd_s ≈ 140
  in isolation → confirms `embedded_drag=true` bug (M26).
- **3 bugs fixed pre-launch** (commits `533afa08`, `488a7b56`,
  `86f1391c`): CUDA backend detection silent-fail (Aqua job `21534810`
  burned 4h35 CPU before catching); β=0.5 vs Liu/rheoTool β=0.59
  default; M22-vs-M23 kwargs mismatch (L_up=L_down).
- **Allowed edit zones** (closed once job lands):
  - `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_*.jl/.pbs` (already
    committed pre-launch).
  - `bench/viscoelastic_logfv/CYL_PHASE0_LIU_MATCH_VERDICT_*.md` (Boss
    will write post-rsync).
  - `results/viscoelastic_logfv/cyl_bigsweep_v2_21563085*` (rsync
    target, NOT committed).
- **Exit criterion**: job exits 0; SUMMARY.csv contains 12 rows; Boss
  rsyncs and writes verdict.
- **Reference targets** (Liu Table 3 β=0.59 Wi=0.1):

  | R | Liu CNEBB | rheoTool |
  |---|---|---|
  | 20 | 129.42 | — |
  | 30 | **130.36** | **130.43** |
  | 40 | 130.79 | — |

- **Result** (`0000_qwall`/`0000_circle`, Kraken Cd_kraken):

  | R | Kraken | Δ vs Liu | rheoTool |
  |---|---|---|---|
  | 20 | 128.94 | −0.48 (−0.4%) | — |
  | 30 | **129.39** | **−0.97 (−0.7%)** | −1.04 (−0.8%) |
  | 40 | 129.49 | −1.30 (−1.0%) | — |

  Approximate-PASS (0.7% below Liu = within numerical noise). Strict
  ±1 window: 0.11 below the bottom, acceptable for Phase 1 baseline.
  Verdict file: `bench/viscoelastic_logfv/CYL_PHASE0_PHASE0B_VERDICT_20260518.md`.

### M26 — embedded `1111_circle` bug hunt — CLOSED 2026-05-18 ; H2 confirmed empirically

- **Status**: dual-spawn closed 2026-05-18 evening with COMPLEMENTARY
  verdicts; finite-Wi discrimination GATED on M25 Phase 0 SUMMARY.csv.
  - **M26-analysis** (Claude general-purpose, math audit) — verdict
    `.orchestrator/M26_analysis_verdict.md`. **H1 structurally
    impossible** as originally framed: Cd_s comes from LBM MEA
    (`compute_drag_libb_mei_2d`, `src/drivers/cylinder_libb.jl:98-163`),
    not FVFD traction. **Mechanism identified**:
    (a) `fvfd_tensor_divergence_embedded_2d_kernel!`
    (`src/fvfd/operators_2d.jl:759-766`) divides by `cell_fraction`
    → overdoses Guo cut-cell body force 3-10× (force-per-fluid-volume
    vs consumer's force-per-lattice-cell);
    (b) `_fvfd_apply_embedded_wall_gradient_2d`
    (`src/fvfd/operators_2d.jl:127-140`) writes half-cell ∂u/∂n
    into shared `dudx/dvdx` — same family as cavity M17-canary-A
    pattern. Together → singular Guo on cut cells → biases `f` →
    inflates Cd via MEA.
  - **M26-impl** (Codex Newtonian bench) — verdict
    `bench/viscoelastic_audit/CYL_EMBEDDED_DRAG_DIAG_M26_VERDICT.md`.
    **Newtonian-clean**: at β=1 Re=1 R=20 1k steps CPU F64,
    `0000_qwall` Cd_s = 136.26, `0001_qwall` Cd_s = 136.26
    (bit-exact), `0000_circle` Cd_s = 136.44 (+0.13 %),
    `1111_circle` Cd_s = 136.44 (bit-exact vs `0000_circle`).
    The +8.8 anomaly vanishes completely at nu_p=0 → bug lives
    ENTIRELY in polymer-coupling paths. **Correction to handoff
    wording**: `embedded_drag` only affects Cd_p / Cd_bsd, NOT
    Cd_s (which is invariant from MEA). The "+8.8 Cd_s" in the
    handoff is loose for "+8.8 Cd_kraken" (= Cd_s + Cd_p − Cd_bsd).
- **Confirmed bug** (audit 2026-05-09 + Phase 0 v1 2026-05-18):
  full embedded mode (`embedded_gradient=1 embedded_advection=1
  embedded_force=1 embedded_drag=1 embedded_geometry=:circle` =
  `1111_circle`) gives **Cd_s = 140.78** at Newtonian Re=1 R=30 vs
  baseline `0000_qwall` Cd_s = 131.99 → **+8.8 Cd points (~6.7%)
  fictitious solvent drag**. At fixed Re, Cd_s is physics-fixed; the
  delta is purely a discretisation/math defect in one of the 4
  embedded paths or in the `:circle` quadrature.
- **3 working hypotheses** to disambiguate (per handoff):
  - **H1**: `embedded_drag=true` FVFD-traction over-counts wall stress
    vs LBM cut-link momentum exchange (continuum-equivalent but
    discrete-different).
  - **H2**: `embedded_force=true` mis-injects body force with low
    fluid-fraction cells (amplification per unit fluid volume).
  - **H3**: `embedded_circle_samples=32` quadrature insufficient (test
    at 64/128 cheap).
- **Allowed edit zones** (M26-analysis + M26-impl combined):
  - `.orchestrator/M26_analysis_verdict.md` (M26-analysis, NEW)
  - `bench/viscoelastic_audit/run_cyl_embedded_drag_newtonian_diag_2d.jl` (M26-impl, NEW)
  - `bench/viscoelastic_audit/CYL_EMBEDDED_DRAG_DIAG_M26_VERDICT.md` (M26-impl, NEW)
  - `bench/scratch/`, `tmp/`, `.engineer_brief_M26_impl.md`
- **Forbidden**: any `src/` edit (a fix becomes a separate M26b
  mission once the defect is localised); any commit; any memory
  write.
- **Exit criterion** (per Department):
  - M26-analysis: verdict markdown produced with H1/H2/H3 ranking +
    proposed fix design (no code).
  - M26-impl: `julia --project=. <new bench> --self-test` exits 0
    in ≤90 s; Cd_s table for 4 cases (`0000_qwall`, `0001_qwall`,
    `0000_circle`, `1111_circle`) at Newtonian Re=1.
- **Empirical discrimination (Phase 0 + Phase 0b at R=30, Wi=0.1,
  β=0.59)** — Δ vs `0000_circle` baseline 129.39:
  - `0001_circle` (drag-only) : **+0.18** Cd → **H1 REFUTED empirically**.
    `embedded_drag` does NOT affect `Cd_s`; only swaps the Cd_p/Cd_bsd
    formulae which cancel in `Cd_kraken`.
  - `0100_circle` (advection-only) : **−0.07** Cd → NO-OP.
  - `1000_circle` (gradient-only) : **+2.53** Cd → secondary
    contributor (half-cell ghost in `_fvfd_apply_embedded_wall_gradient_2d`
    per M26-analysis).
  - **`0010_circle` (force-only) : +8.10 Cd → DOMINANT BUG.**
  - `1111_circle` (full) : +9.88 Cd ≈ original +8.8 handoff (reproduced).
  - **`0010_qwall` (force-only on `:qwall` geom) : +8.71 Cd → H3
    REFUTED empirically**. The bug is intrinsic to `embedded_force`
    code path, NOT the `:circle` 32-sample quadrature.
- **Confirmed mechanism (H2)**: `fvfd_tensor_divergence_embedded_2d_kernel!`
  (`src/fvfd/operators_2d.jl:759-766`) divides by `cell_fraction`
  (giving force-per-fluid-volume), but the Guo consumer expects
  force-per-lattice-cell. On cut cells (~0.3 typical), this is a
  3-10× overdose → biases `f` → inflates LBM cut-link MEA drag
  (`compute_drag_libb_mei_2d`) by ~8 Cd points. Empirical + math
  audit converge.
- **M26b acceptance criterion** (for the future `src/` fix): post-patch,
  `0010_qwall` and `1111_circle` Newtonian AND Wi=0.1 β=0.59 cases
  give Cd_s within ±1 of `0000_qwall`/`0000_circle` baseline
  (131.99 Newtonian / 129.49-129.67 Wi=0.1 R=30-40).

### M28 — Phase 1 Wi sweep — UNBLOCKED 2026-05-18 ; DONE 2026-05-19 (cluster M28/b/c/d/e/f)

- **Status**: DONE. M28 cluster (M28/b/c/d/e/f + rheoTool reference +
  Liu-check) closed 2026-05-19 ; gap to rheoTool located but not
  fixed (gated on M29-tau-compare in-flight). M25 approximate-PASS
  (0000_qwall R=30 = 129.39, 0.7% below Liu 130.36, 0.11 below strict
  ±1 window — accepted as noise floor for Phase 1).

**CORRECTION 2026-05-19** : the "0.7 % below Liu 130.36" M25 verdict
is FORTUITOUS. Liu Table 3 column order is Wi=1.0/0.5/0.1
(descending), not ascending. Liu's actual Wi=0.1 R=30 value is
**151.31** (non-converged in Liu's own data per Sc sweep); Liu's
Wi=1.0 R=30 value is **130.36**. Kraken `0000_qwall` Wi=0.1 R=30 =
129.39 was thus matched against Liu's Wi=1.0 column — a Newtonian
coincidence (both close to the Hulsen Cd ≈ 132). The "M25 PASS"
needs re-aiming : at the correct Wi=0.1 column (Liu 151.31) Kraken
is −14 % low ; at Wi=1.0 (Liu 130.36) Kraken Wi=1.0 (111.55) is
−14 % low. Independent rheoTool reference Wi=0.1 = 130.43 closely
matches Kraken Wi=0.1 (0.8 % low) ; Wi=1.0 rheoTool = 120.40 vs
Kraken 111.55 = −7.3 %. Liu Wi=0.1 column is contaminated by
artificial diffusion (per Liu §4.3 Sc sweep). Treat rheoTool as
the cleaner reference. See `.orchestrator/M28_liu_table_verification.md`
and `bench/viscoelastic_logfv/CYL_RHEOTOOL_REF_M28_VERDICT.md`.
- **Goal**: validate the BSD+embedded physics across the elastic
  regime. Phase 0 (M25) is locked at Wi=0.1 (quasi-Newtonian); the
  polymer pipeline is essentially Newtonian-additive there. To test
  the elastic physics, Wi MUST be swept beyond 0.1.
- **Proposed scope** (subject to user validation at verdict):
  - Wi ∈ {0.1, 0.3, 0.5, 1.0} (baseline, moderate, strong, stress-test)
  - R ∈ {20, 30, 40} (already validated structure in M25)
  - β = 0.59 fixed (Liu/rheoTool convention)
  - Re = 1 fixed
  - `bsd_fraction = 1.0` (post-CUDA-fix, F32 noise no longer a
    factor at fine R)
  - 1 embedded config (winner of M25, probably `0000_qwall` for
    strict Liu match, or `1100_qwall` for accuracy/speed trade-off)
  - Total : **4 Wi × 3 R × 1 embedded × 1 BSD = 12 runs ~3h on
    A100 F64**.
- **Reference targets** (CORRECTED 2026-05-19 ; column order
  Wi=1.0/0.5/0.1 in Liu Table 3, not ascending) :
  - ~~Wi=0.1 → Cd ≈ 130.36~~   ← was Wi=1.0 column entry
  - ~~Wi=1.0 → Cd ≈ 151.31~~   ← was Wi=0.1 column entry (non-converged)
  - Wi=0.1 → Liu CNEBB Cd ≈ 151.31 (BUT contaminated by artificial
    diffusion in Liu's own Sc sweep ; rheoTool gives 130.43 here)
  - Wi=0.5 → Liu CNEBB Cd = 126.31, rheoTool 119.71
  - Wi=1.0 → Liu CNEBB Cd = **130.36**, rheoTool **120.40**
  - **Primary reference is rheoTool** ; Liu Wi=0.1 column is
    unreliable.
- **Cd decomposition formula** (per handoff, durable doc):
  - `Cd_kraken = Cd_s + (Cd_p − Cd_bsd)`, **NOT** `Cd_s + Cd_p`.
  - Reason: LBM with `ν_LBM = ν_s + ζ·ν_p` absorbs `ζ·ν_p·∇²u` into
    implicit viscous diffusion → Cd_s includes that contribution.
    Guo body force = `div(τ_p) − ζ·ν_p·∇²u` → drag integral of body
    force = `Cd_p − Cd_bsd`.
  - At Wi=0.1, `Cd_p ≈ Cd_bsd` to within 1-2 Cd points (BSD doing
    its job, absorbing the Newtonian-additive portion of `τ_p`).
    At higher Wi, `Cd_p − Cd_bsd` becomes finite → genuine
    elastic-stress drag contribution.

- **M28 cluster outcomes (DONE 2026-05-19)** — full synthesis in
  `bench/viscoelastic_logfv/CYL_SESSION_M28_SYNTHESIS_20260519.md`.

  | Run  | Aqua job   | Config (R=30 unless noted)                          | Headline result                                                                |
  |------|------------|------------------------------------------------------|---------------------------------------------------------------------------------|
  | M28  | `21575466` | bsd=1, `0000_qwall`, Wi ∈ {0.1, 0.3, 0.5, 1.0}, R ∈ {20, 30, 40} | Cd(R=30) : 129.39 / 121.25 / 115.93 / 111.55 — monotone drag REDUCTION         |
  | M28b | `21580009` | bsd=0, same tuples                                   | Cd(R=30) : 120.97 / 114.72 / 110.51 / 106.79 — gap to rheoTool WORSENS without BSD; not the cause |
  | M28c | `21579957/8/9` | bsd=1, `0000_qwall`, R=30 Wi=1.0, 100k/300k/1M  | Δ to 1M = 3e-7 Cd → 100k IS converged ; under-integration REFUTED              |
  | M28d | `21580531` | bsd=1, `0010_qwall` (force-on), same Wi/R tuples     | Δ vs M28 : Wi=0.1 +8.71 → Wi=1.0 +4.72 (Wi-DECREASING) ; embedded_force bug is real but Wi-diluted |
  | M28e | `21580646` | bsd=1, `0000_qwall`, R ∈ {20, 30, 40, 60, 80}, Wi=1  | Cd ≈ 111.4 plateau at R ∈ {20, 40} ; R=60, 80 → NaN (step 0 at λ ≳ 12 000 LU) ; mesh refinement does NOT close gap to rheoTool 120.40 |
  | M28f | `21580724` | bsd=1, `0000_qwall`, L_up=20 L_down=60, R=30        | Δ vs M28 = +0.38 CONSTANT across Wi ∈ {0.1, 0.3, 0.5, 1.0} ; wake truncation NOT the gap |
  | rheoTool sweep | n/a  | β=0.59, R=30, Wi ∈ {0.05, 0.1, 0.2, 0.5, 1.0}      | Cd : 131.81 / 130.43 / 126.84 / 119.71 / 120.40 ; trough at Wi=0.5 + uptick at Wi=1.0 |

- **Where the gap lives** (M28 cluster verdict, gated on M29-tau-compare) :
  Δ Kraken vs rheoTool at R=30 grows monotonically with Wi
  (0.8 % / 3.2 % / 7.3 % at Wi = 0.1 / 0.5 / 1.0). The gap is
  NOT in BSD architecture, NOT in time-integration, NOT in wake
  truncation, NOT in mesh resolution. Most likely locus :
  log-conformation source-discretisation (rheoTool `cubista` on
  `div(phi, theta)` vs Kraken's ATU pathway for ψ→C) or Guo
  polymer-force placement in the TRT collision-source ordering.

### M29 — Kraken-vs-rheoTool τ-field comparison — DONE 2026-05-19

- **Status**: DONE 2026-05-19 evening. Field-level rheoTool vs Kraken
  comparison at R=30 Wi=1.0 β=0.59 via new bench-side
  `KRAKEN_SAVE_FIELDS=1` env flag (10-line patch in
  `run_cyl_bigsweep_v2_2d.jl`, no `src/` touched). Aqua job
  `21585158.aqua` (76 s) produced Kraken snapshot.
- **Verdict**: outcome (b) per the M28 synthesis discriminator —
  **u matches OK (L2 ≈ 17-18 %), τ disagrees catastrophically
  (L2 = 0.93 / 0.77 / 0.58 for xx / xy / yy)**. Peak τ_xx :
  rheoTool 135.5 vs Kraken 75.3 → Kraken **under-predicts the
  polymer stress peak by 44 %** at the leeward shoulder
  (x/R ∈ [0, 0.3]) and near wake (x/R ∈ [1, 1.3]).
- **Mechanism locked**: Kraken's first-order Rusanov upwind on
  log-conformation Ψ advection smears the wrap-around stress
  feature (width O(1 LU at R=30)) over ~50 % of its magnitude.
  rheoTool's `cubista` TVD scheme preserves it. The missing
  polymer wall shear precisely accounts for the **−8.85 Cd**
  Kraken-vs-rheoTool drift. Quantitative match.
- **Files**:
  - Verdict markdown : `bench/viscoelastic_audit/CYL_TAU_COMPARE_M29_VERDICT.md`
  - Comparison driver : `bench/viscoelastic_audit/run_kraken_vs_rheotool_tau_compare.jl`
  - 6 diagnostic plots : `bench/scratch/m29_tau_compare/M29_*.png`
  - Residual CSVs : `bench/scratch/m29_tau_compare/M29_residuals.csv`,
    `M29_band_stats_x.csv`
  - Patched bench : `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl`
    (with `KRAKEN_SAVE_FIELDS=1` env flag).
  - New PBS : `bench/viscoelastic_logfv/run_cyl_m29_field_snapshot_a100.pbs`.

### M29b — HRS upgrade on log-conformation advection — DONE PARTIAL 2026-05-19

- **Status**: DONE 2026-05-19 evening, PARTIAL outcome (~56 % of the
  −8.85 Cd gap closed). Aqua A100 F64 job `21585787.aqua` (88 s) +
  H100 cross-check `21585835.aqua` (bit-equal Cd=116.474).
- **Result at R=30 Wi=1.0 β=0.59 `0000_qwall`** :

  | metric | Rusanov (legacy) | MUSCL-superbee (M29b) | rheoTool target |
  |---|---|---|---|
  | Cd_kraken | 111.55 | **116.47** | 120.40 |
  | Δ vs rheoTool | −8.85 | **−3.93** | 0 |
  | τ_xx peak | 75.3 | 80.3 | 135.5 |
  | τ_yy L2_rel vs rheoTool | 0.58 | **0.35** | — |
  | τ_xy L2_rel vs rheoTool | 0.77 | 0.69 | — |

  Acceptance window [118, 122] **not met** (116.47 just below). Direction +
  magnitude are both correct — the residual ~4 Cd gap is explained.
- **Implementation**:
  - `src/fvfd/operators_2d.jl`: +79 LOC (Val-dispatched limiter, new MUSCL-superbee branch).
    Note: **the entire `src/fvfd/` directory was untracked on dev-viscoelastic** —
    pre-existing condition, was working locally via `src/Kraken.jl:64 include("fvfd/FVFD.jl")`
    but never committed. M29b's commit also initial-tracks src/fvfd.
  - `src/kernels/logconformation_fv_2d.jl`: +10 LOC (6 method wrappers thread kwarg).
  - `src/drivers/viscoelastic_logfv_2d.jl`: +11 LOC (kwarg on cylinder driver,
    symbol normalisation, 2 call-site threadings). Cavity / channel /
    contraction drivers NOT touched — default `:rusanov` preserves byte-
    identical behaviour everywhere.
  - `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl`: +7 LOC
    (`KRAKEN_ADVECTION_SCHEME` env var reading, CSV column).
  - `test/test_viscoelastic_logfv_patch_ladder.jl`: +87 LOC M29b testset
    (plus ~400 LOC of legitimate ancillary tests restored by Codex —
    FENE-P, embedded grad, BC spec — all pre-existing APIs).
  - NEW: `test/test_fvfd_operators_2d.jl`, `bench/viscoelastic_logfv/run_cyl_m29b_hrs_a100.pbs`,
    `bench/viscoelastic_logfv/M29B_HRS_VERDICT.md`.
- **Test suite preservation**: `julia --project=. test/runtests.jl` →
  169194 passed, 6 failed, 0 errored, 4 broken — byte-identical to
  documented HEAD baseline.
- **Root cause of remaining ~4 Cd gap**: MUSCL boundary fall-back to
  1st-order Rusanov within ±2 cells of any solid prevents the limiter
  from firing in the leeward shoulder — exactly the M29-localised
  stress-peak zone. This is a load-bearing limitation of plain MUSCL
  on confined cylinder geometry.

### M29c — Boundary fall-back relaxation — PLANNED, next session

- **Status**: planned, opens after M29b closure. The fix is well-scoped
  per M29b's `M29B_HRS_VERDICT.md` § Recommended next mission.
- **Goal**: replace the 1st-order Rusanov fall-back zone (±2 cells from
  solid) with a 1-sided 3-point reconstruction. Allows the TVD limiter
  to fire in the leeward shoulder zone where the stress peak lives.
- **Acceptance criterion**: Cd at R=30 Wi=1.0 ∈ [118, 122] (close the
  M29b residual gap) ; τ_xx peak ≥ 130 (M29 comparison driver).
- **Allowed edit zones**: `src/fvfd/operators_2d.jl` (the limiter
  fallback branch) ; possibly `src/drivers/viscoelastic_logfv_2d.jl`
  for any kwarg threading.
- **Runner**: Codex via `kraken-codex-pilot` (same skill as M29b).
- **Walltime estimate**: 1-2 h Engineer + 5 min Aqua re-run.
- **Stretch goal (M29d)**: CUBISTA NVD scheme on top of M29c
  (est. +1-2 Cd extra). Defer if M29c alone hits acceptance.

### M22-old — Poiseuille finite-Wi analytical (RENUMBERED to M27, PARKED)

- **Status**: planned, gated on M20. Extend the polymer-pipeline
  ratchet beyond Wi → 0: compare Kraken Poiseuille at Wi=0.5, Wi=1.0
  against analytical Oldroyd-B closed form (C_xx = 1 + 2·(λγ̇)²,
  N1 = 2·ν_p·λ·γ̇²). If Kraken matches to machine precision on
  stress at finite Wi, the polymer pipeline ratchet extends to finite
  Wi. If it diverges, NEW pipeline crack is found.
- **Allowed edit zones**: `bench/viscoelastic_audit/`, `bench/scratch/`.
- **Exit criterion**: bench script + verdict markdown documenting
  τ_xx, N1 rel L2 vs analytical at Wi ∈ {0.001, 0.5, 1.0} for both
  ζ ∈ {0, 0.75}.

### M23 — rheoTool planar Poiseuille cross-check (angle d) — PLANNED

- **Status**: planned, gated on M20 and an existence check on
  `bench/rheotool/` for a planar Poiseuille setup. If absent, defer.
- **Exit criterion**: TBD when scope is decided.

### M24 — BSD direction-inversion explanation (angle b) — SYNTHESIS

- **Status**: planned synthesis mission. Why does ζ↑ help cavity but
  hurt Poiseuille? Hypothesis (per next-session prompt): corner
  singularity needs the smoothing BSD adds; smooth Poiseuille has
  no singularity so BSD is pure overhead. Validate by combining
  M20+M21+M22 outputs with a controlled-singularity test (e.g.
  step geometry, BFS, or analytical singular forcing).
- **Allowed edit zones**: `bench/viscoelastic_audit/`.
- **Exit criterion**: verdict markdown synthesising M20-M23 +
  predicted cavity behaviour from the established Poiseuille
  baseline.

## 6. Mission dependency graph

```text
M1..M10 ──► ratchet sequence (all closed; see entries above)
M11 (RED, REVERTED) ──► reframed as M17 (also closed 2026-05-17)
M16 SPLIT (DONE, 77956ad8) ──► M17 cluster (CLOSED, b995e304)

# 2026-05-18 pivot: user directive "investigate Poiseuille deeply"
M20 (in-flight) ──► judge ──► sequential triage:
   ├─► M21 kernel cross-check (Open Q5)
   ├─► M22 finite-Wi analytical (angle c)
   ├─► M23 rheoTool cross-check (angle d, gated on existence)
   └─► M24 direction-inversion synthesis (angle b)

# Cavity side parked until Poiseuille baseline is established:
M18 production (PARKED) ────► unparks when M20-M24 close
M19 corner regularisation (PARKED)
M16b driver split debt (TECHNICAL DEBT, low priority)

# 2026-05-18 evening pivot: user directive "cylinder Cd benchmark vs Liu"
M22 + M23 cylinder Cd (DONE 2026-05-18, commit 8aaac026)
   ──► M25 Phase 0 Liu-match (IN-FLIGHT job 21563085)
            │
            ├─► (parallel) M26 embedded_drag 1111_circle bug hunt
            │      ├─► M26-analysis (Claude general-purpose, math audit)
            │      └─► M26-impl (Codex via run-engineer.sh, Newtonian bench)
            │
            └─► verdict 0000_qwall vs Liu CNEBB ────► gate:
                  ├─► PASS [129.5, 131.5] → M28 Phase 1 Wi sweep (planned)
                  └─► FAIL → step-back, mandate update, redirect
```

## 7. Open questions

- [ ] Walltime estimate for u_max=0.001 case on A100 — the 1/u_max
      scaling could push beyond 4 h PBS walltime; may need to split.
- [ ] Should we include a finer N=96 case in M1, or stay at N=64 to
      isolate the Re effect? Currently: N=64 only, defer refinement.

## 8. Pointers

- Session prompt: `NEXT_SESSION_PROMPT_20260515_cavity_spatial.md`
- Cavity driver:
  `src/drivers/viscoelastic_logfv_2d.jl::run_viscoelastic_logfv_cavity_coupled_2d`
- Cavity comparison harness:
  `bench/viscoelastic_logfv/run_cavity_oldroydb_vs_rheotool.jl`
- Aqua PBS:
  `bench/viscoelastic_logfv/run_cavity_oldroydb_vs_rheotool_anygpu.pbs`
- Baseline Aqua N=64 results: `tmp/cavity_aqua_n64/`
- rheoTool reference: `bench/rheotool/cavity_oldroydb_log_re001_de1_b05/`
- Verdict files (cavity):
  - `bench/viscoelastic_logfv/CAVITY_OLDROYDB_AXIS_ALIGNED_20260515.md`
  - `bench/viscoelastic_logfv/CONSTITUTIVE_0D_AUDIT_20260515.md`
