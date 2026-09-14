# Next session prompt — Kraken viscoelastic cavity, symptoms snapshot

Copy-paste below to start a fresh session.

---

Continue work on branch `dev-viscoelastic` of Kraken.jl
(worktree `~/Documents/Recherche/Kraken.jl-viscoelastic`).

This prompt is **symptoms-only**. No interpretations, no "dominant
contributor" claims, no closed verdicts. Just the raw measurements
collected on 2026-05-17. Open the orchestrator skill and re-derive
strategy from these symptoms.

## User directive — first move

> "On investigue le poiseuille first et on essai de comprendre ce
> qu'il se passe."

Before touching the cavity again, the next session must **investigate
Poiseuille deeply** to understand what BSD actually does to the
LBM-FV coupling on the simplest possible geometry. Possible angles
(non-exhaustive — Boss decides):

- Trace step-by-step what F_total looks like at each timestep on
  Poiseuille with `:fd, ζ=0.75`: is the BSD compensation actually
  cancelling the LBM `ζ·ν_p·∇²u` portion as designed?
- Why does BSD invert direction (cavity vs Poiseuille — see Open
  Question 2)?
- Run Poiseuille at finite Wi (e.g. Wi=0.5, Wi=1.0) with all the M7b
  cases including kinematic invariants, compare to analytical
  Oldroyd-B at finite Wi (not just Newtonian asymptote — full
  closed form has `C_xx = 1 + 2·(λγ̇)²`, etc.).
- Cross-check against rheoTool channel/Poiseuille (does a planar
  rheoTool case exist in `bench/rheotool/`? if yes, run same setup
  through both, compare).
- Verify the velocity-gradient kernel difference between cavity and
  Poiseuille drivers (Open Question 5) is not the source of the 8×
  cavity-vs-Poiseuille ratio of the M7b signal.

Do not return to cavity-driver edits until Poiseuille is fully
understood.

## Recent HEAD (read these commits' content if needed)

```
b995e304 feat(viscoelastic): M17 cluster closure — Poiseuille analytical + rheoTool cross-check
c20e4e8c feat(viscoelastic): M17 nyquist verification — WIDE F_poly null mode confirmed
f60f5174 feat(viscoelastic): M17 canary A — Option A validated (D_uncorrected for BSD)
a2e6f088 feat(viscoelastic): M17 canary analytical BSD ladder
77956ad8 refactor(viscoelastic): M16 split cavity driver out of monolith
1f968a49 docs(viscoelastic): close M9 — grid convergence partial floor confirmed
```

## Reference docs (start by reading these)

- Mandate: `.orchestrator/mandate.md`
- Boss memory: `.orchestrator/memory/boss.md`
- Department memory: `.orchestrator/memory/department.md`
- Engineer memory: `.orchestrator/memory/engineer.md`
- BSD analytical ladder: `bench/viscoelastic_audit/BSD_ANALYTICAL_LADDER_20260517.md`
- Poiseuille analytical + rheoTool cross-check:
  `bench/viscoelastic_audit/POISEUILLE_ANALYTICAL_BSD_VERDICT_20260517.md`
- Cavity pipeline arch audit (M15):
  `bench/viscoelastic_audit/CAVITY_PIPELINE_ARCH_AUDIT_20260516.md`

Archived RED implementations (for reference, not to retry):
`.orchestrator/red_archives/{M17impl_,M17implv2_,M17implv3_,M17impl_v3_poiseuille_}_20260517/`

## Symptoms — what has been measured

### Polymer pipeline (steady-state Oldroyd-B Poiseuille, CPU F64)

- `tau_xy` rel L2 vs analytical (= ν_p · γ̇) = **5.0e-3** (interior).
- `tau_xx` rel L2 vs analytical (= 2·ν_p·λ·γ̇²) = **8.25e-6**.
- `tau_yy` max abs (analytical = 0) = **5.8e-16** (machine zero).
- `N1` rel L2 vs analytical (= 2·ν_p·λ·γ̇²) = **8.24e-6**.
- min C eigenvalue = 0.99923 (well above 0, SPD-positive).
- Same numbers identical at `bsd_kind=:fd, ζ=0.0` and `ζ=0.75`.

Setup: Nx=8, Ny=32, F_body=1e-5, λ=1.0, max_steps=100k, `:fd` path.

### Velocity field (same Poiseuille setup, M7b-style comparison)

| case | `nu_s` | `nu_p` | ζ | u rel L2 vs analytical |
| --- | --- | --- | --- | --- |
| A | 0.1 | 0.1 | 0.75 | 7.37e-3 |
| A_no_BSD | 0.1 | 0.1 | 0.00 | 5.35e-3 |
| B (matched ν_LBM) | 0.2 | 0.0 | 0.00 | 3.37e-3 |

- A vs B rel L2 = **4.23e-3** (Poiseuille M7b equivalent).
- A_no_BSD vs B rel L2 = 2.57e-3.
- Δ (A_with_BSD − A_no_BSD vs B) = ~1.7e-3.

cf. cavity M7b Aqua F64 N=64: A vs B = **3.42e-2** (`bench/viscoelastic_logfv/CAVITY_LOWWI_M7B_VERDICT_20260516.md`).

### Wi sweep on Poiseuille (matched ν_total, varying λ)

| Wi (≈ λ·γ̇_max) | ζ=0 rel_err_u | ζ=0.75 rel_err_u |
| --- | --- | --- |
| 8e-4 | 5.28e-3 | 6.85e-3 |
| 1e-2 | 5.28e-3 | 6.85e-3 |
| 1e-1 | 1.27e-3 | 2.83e-3 |
| 1.0 | 2.87e-4 | 1.84e-3 |

ζ=0 wins everywhere; advantage grows with Wi (1.3× → 6.4×). No NaN at any Wi or ζ.

### Cavity M4b Aqua F64 N=64 t=8 De=1 β=0.5 (already-recorded)

| ζ | centerline u rel L2 vs rheoTool | psi_xy rel L2 |
| --- | --- | --- |
| 0 | NaN | NaN |
| 0.25 | 21.15 % | 27.41 % |
| 0.5 | ~19 % | ~26 % |
| 0.75 | 17.97 % | 24.41 % |
| 1.0 | crash by design (ADR) | — |

Note: ζ=0 NaN on cavity vs ζ=0 stable on Poiseuille (Wi sweep).

### BSD analytical ladder (Taylor-Green vortex, CPU F64)

L0 — `div(τ_p)` vs `ν_p·∇²U` (periodic, no walls):
- N=32 6.4e-3, N=64 1.6e-3, N=128 4.0e-4. Order ≈ 2.00.

L1 — F_total cancellation vs `(1−ζ)·ν_p·∇²U` (periodic):
- `:fd` N=64 4.0e-3, `:fd_v2` (wide-stencil) N=64 3.2e-3, Δ = 7.2e-3.

L2 — F_total in closed box with TG-matched lid:
- `:fd` interior 4.0e-3, wall 0.10, wall/interior ratio 1.0×.
- `:fd_v2` interior 3.2e-3, wall 303, wall/interior ratio 158×.

L2b — same but BSD reads `D_uncorrected`:
- `:fd_v2_unc` interior 3.2e-3, wall 0.16, wall/interior ratio 1.0×.
- wall drop `:fd_v2` → `:fd_v2_unc` at N=64: 1860×.

L4 — Fourier spectral test of WIDE F_poly vs NARROW Laplacian:
| m | k·dx | WIDE ratio | NARROW ratio |
| --- | --- | --- | --- |
| 1 | π/32 | 0.998 | 0.999 |
| 8 | π/4 | 0.900 | 0.950 |
| 16 | π/2 | 0.637 | 0.811 |
| 32 | π (Nyquist) | **3.4e-15** | 0.405 |

### Cavity dynamic attempts at `:fd_v2_*` (M11, M17-pre v1/v2, M17-impl, M17-impl-v2, M17-impl-v3)

All RED, NaN signature step ~120-3900 in `ux` at wall-adjacent cells.
Static canaries (L0-L4 + L2b) all GREEN. The dynamic-stability
discrepancy between static and dynamic implementations is **not
diagnosed**. Detailed RED reports in `.orchestrator/red_archives/`.

### rheoTool cross-check (cavity Oldroyd-B De=1 β=0.5 N=127, OpenFOAM 9 rheoFoam)

- Original case: `stabilization coupling` (iBSD), run from t=0 to t=8.
- Clone `_no_ibsd/`: `stabilization none`, continued from t=8 state to t=12.

Centerline u(0.5, y) drift trajectory (iBSD-OFF run, vs iBSD-ON
starting state):

| time | rel L2 Ux |
| --- | --- |
| 7.9998 (start) | 0 |
| 8.9994 | 7.4e-3 (overshoot) |
| 10.0001 | 6.7e-3 |
| 11.0007 | 6.3e-3 |
| 11.9997 | 6.0e-3 |

Still slowly decreasing at t=12; asymptote not yet measured.

### Cavity grid convergence (M9 trajectory, rheoTool reference, ζ=0.75)

- N=32: centerline u rel L2 31.4 %.
- N=64: 18.0 %.
- N=96: 12.85 %.
- N=128: pending Aqua rsync from job `21405282.aqua`.

### Test suite identity

`julia --project=. test/runtests.jl` exits 1 with stable signature
**169194 passed, 6 failed, 0 errored, 4 broken** across all commits in
this session. The 6 failures are pre-existing (Pure shear Oldroyd-B
steady state). `Pkg.test()` reports different numbers because
`test/Project.toml` is intentionally minimal (only `Test`). The test
invocation rule is in `.orchestrator/memory/department.md`.

## Open questions

1. **BSD vs rheoTool**: in rheoTool cavity, iBSD ON vs OFF gives
   ~0.6 % steady-state Ux drift. In Kraken cavity, BSD has 18 % gap
   from rheoTool reference at N=64 ζ=0.75. Can we do better than the
   current Kraken BSD against rheoTool's same-stencil iBSD? The
   ~0.6 % rheoTool drift is non-zero — what's its asymptote (continue
   the run to t=20 or t=50)? What does it tell us about the
   "intrinsic" cost of BSD-style coupling?

2. **Cavity ζ-sweep vs Poiseuille ζ-sweep direction**:
   - Cavity (M4b): ζ↑ → rel L2 ↓ (BSD helps).
   - Poiseuille (Wi sweep): ζ↑ → rel L2 ↑ (BSD hurts).
   - The roles invert. Why? Hypothesis (unverified): corner
     singularity on cavity needs the smoothing that BSD adds; on
     Poiseuille no singularity → BSD is pure overhead. Test this
     by checking other singular geometries.

3. **Dynamic instability of `:fd_v2_*` variants**: all 6 attempts
   NaN at wall-adjacent cells despite static GREEN. Static canaries
   missed it. What invariant is being violated dynamically that the
   static tests don't measure?

4. **Cavity benchmark target**: Mandate "Done" = single-digit
   percent rel L2 on `u(0.5, y)` and `psi_xy(x, 0.75)` at N=64.
   Currently 17.97 % / 24.41 % at N=64 ζ=0.75. M9 trajectory predicts
   ~9 % at N=128, ~5 % at N=256. The "Done" target may need
   revision against the rheoTool gold standard (which itself has its
   own discretization error at N=127).

5. **`run_viscoelastic_logfv_poiseuille_coupled_2d`** uses a
   different velocity-gradient kernel than the cavity driver:
   `logfv_velocity_gradient_bc_aware_2d!` vs
   `fvfd_velocity_gradient_2d!`. The 8× cavity-vs-Poiseuille ratio of
   the M7b signal could be partly attributed to this kernel
   difference rather than purely to the corner singularity. Needs a
   controlled cross-check (same kernel both drivers).

## Open missions

- **M18** (production validation): rsync M9 N=128 from Aqua job
  `21405282.aqua`, then submit M18 production cavity at N=128 ζ=0.75
  De=1 β=0.5. HPC op — requires user trigger. Verdict to compare
  against the M9 trajectory extrapolation.
- **M19** (corner regularization, optional): smoother lid profile
  / explicit corner velocity ramp; measure whether cavity gap at
  N=64 closes. Speculative.
- **M16b** (technical debt): split
  `run_viscoelastic_logfv_poiseuille_coupled_2d` and the other
  ~7 drivers out of `src/drivers/viscoelastic_logfv_2d.jl`
  (currently 2934 LOC, far over the 700 LOC hard ceiling). Required
  before any further substantive change to those drivers.
- **test/Project.toml maintenance** (low priority): add the deps
  that `Pkg.test()` needs (KernelAbstractions, Metal, CUDA when
  applicable). Until done, the invocation rule (always use
  `julia --project=. test/runtests.jl`) must be in every brief.

## Session protocol

The user requested: **resume via orchestrator**. Open
`~/.claude/skills/orchestrator/SKILL.md`. Use parallel theory
Departments where appropriate (the 2026-05-17 Claude + Codex parallel
derivation surfaced an important subtle distinction that single-spawn
would have missed — see `boss.md` 2026-05-17 entry).

Agent reliability note: 4 agents stalled in the 2026-05-17 session
(stream watchdog hit 600s no-progress). When that happens, fall back
to direct Boss-execution of mechanical tasks. Avoid `Monitor` tool;
plain Bash with timeout works.

End of prompt.
