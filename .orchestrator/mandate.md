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

- **Status**: pending M1 result
- **Goal**: local CPU smoke at N=32, t=2 with corner cells in
  `_logfv_cavity_apply_wall_gradient_correction_kernel!` stub-replaced
  by no-op; check whether the psi_xy sign-flip near (1, Ny) disappears.
- **Allowed edit zones**: cavity driver + new smoke script under
  `bench/viscoelastic_logfv/`.
- **Exit criterion**: TBD after M1 verdict.

### M3 — Polymer upwind diffusion (Candidate 3)

- **Status**: pending M1 result
- **Goal**: adapt existing `run_rheotool_frozen_replay_2d.jl` to cavity
  geometry, freeze rheoTool U at t=8, run only the Kraken polymer
  pipeline; compare with rheoTool polymer at the same time.
- **Allowed edit zones**:
  `bench/viscoelastic_logfv/run_rheotool_frozen_replay_cavity_2d.jl` (NEW).

### M4 — Guo body-force vs FD divergence (Candidate 4)

- **Status**: pending M1-M3 results
- **Goal**: compute integrated polymer drag two ways on the saved N=64
  field, compare. Determine if Guo/FD discordance contributes.

### M5 — Kinetic-moment BSD refactor (Candidate 5)

- **Status**: deferred (3-4 h, invasive)
- **Goal**: extract rate-of-strain from LBM non-equilibrium moments
  instead of FD-central laplacian. Architecturally clean; makes BSD/LBM
  split exact. Only consider if M1-M4 don't close the gap.

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
