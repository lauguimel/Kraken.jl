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

#### M6-B — wall-BC matching (gated on M4b + Boss decision)

- **Status**: pending. Do NOT run in parallel with M5-B; both could
  touch `src/fvfd/operators_2d.jl`. Sequence: M5-B first (current
  background), then M6-B if greenlit.
- **Goal**: add `polymer_wall_extrap::Symbol = :quadratic` kwarg to
  the relevant FD wall helpers, with `:linear` selecting the
  rheoTool-style 2-point extrapolation. Default unchanged.
- **Allowed edit zones** (when greenlit):
  - `src/fvfd/operators_2d.jl` (surgical, ≤ 25 LOC)
  - 2 callers (cited in the audit doc)
- **Exit criterion**: re-run `analyse_cavity_guo_vs_fd_2d.jl` on a
  cavity run with `polymer_wall_extrap=:linear`; expect
  ~54 % → ~15-30 % interior L2.

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

## 6. Mission dependency graph

```text
M1 ──► (decide based on L2 trend)
       ├─► if L2 drops monotonically:  Re mismatch dominant; STOP further candidates, document
       ├─► if L2 flat or non-monotonic: M2 (corner) and M3 (frozen replay) in parallel
       └─► M4 only if M2 and M3 are GREEN but residual gap remains
M5 — fallback if M1-M4 do not close the gap
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
