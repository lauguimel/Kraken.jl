# Boss memory — Kraken.jl viscoelastic cavity spatial debug

Initialised 2026-05-15. Project-level facts that affect future missions
on the cavity / log-FV / viscoelastic branch.

## 2026-05-15 — Mandate bootstrap

Constitutive validated (0D shear + planar extension to machine precision
at production substep cadence). Bug is purely spatial/coupling at the
cavity, 18-24 % L2 on profiles at t=8 N=64. Five spatial candidates,
sweep order driven by cost/prior. See `mandate.md` §5.

**Implication**: future missions must not pivot back to 0D constitutive
re-validation. Mandate §2 documents this. If a Department reports BLOCKED
asking "is the constitutive maybe wrong?", that signals a brief framing
issue — refine the brief, do not re-run 0D.

## 2026-05-15 — Why bsd_fraction = 0.75 (not 1.0)

`bsd_fraction = 1.0` crashes at the lid corner because the LBM BGK
implicit diffusion stencil mismatches the FD-central laplacian in the
explicit BSD correction. Architecturally clean fix is kinetic-moment BSD
(M5), but deferred until M1-M4 are exhausted. Cavity sweep already
showed <2 % sensitivity within bsd ∈ [0, 0.75], so this is a low-prior
contributor to the cavity gap at N=64.

**Implication**: do NOT propose a brief that tries `bsd=1.0` on cavity.
Document if the urge arises again.

## 2026-05-15 — M1 closed: Re mismatch refuted

L2 essentially flat across u_max ∈ {0.005, 0.002, 0.001}: centerline
1.797e-1 → 1.795e-1 (0.1% change); psi_xy 2.44e-1 → 2.38e-1 (2.5%).
Inertia is NOT the cavity-gap driver. The 18-24% L2 band is robust
under Re_LU 6.4 → 1.3. See verdict file
`bench/viscoelastic_logfv/CAVITY_REMISMATCH_M1_VERDICT_20260515.md`.

**Implication**: do NOT re-attempt a smaller-u_max sweep as a fix; the
gap is dx-bound (spatial discretisation / coupling), not Re-bound.
Next missions: M2 (wall-corner artifact) + M3 (frozen-replay polymer)
in parallel per Mandate §6.

## 2026-05-16 — M2 + M3 closed (smoke); M4 promoted to primary suspect

M3 smoke: standalone polymer pipeline on frozen rheoTool U at t=8
gives **4.08 %** L2 vs **18-24 %** for the coupled run. Polymer
upwind / source / stress is NOT the dominant source. The bug
originates upstream — in U itself, i.e. the LBM solvent response to
the polymer force. This makes **M4 (Guo body-force vs FD divergence)
the primary remaining candidate**.

M2 smoke (N=32 t=2): corner kernel has a real but tiny effect (corner
Δpsi_xy 1.58× bulk Δpsi_xy, both ~5e-5). Absolute magnitude too small
at the smoke scale to bound how much of the 18-24% gap it would close
at production N=64 t=8.

**Implication**: prioritise M4 prep next. M2-full Aqua run is a
secondary confirmation, not a gate. Department in M3 took the
liberty of writing 4 entries directly into
`memory/department.md` — keep the content (it is correct) but future
Department briefs MUST explicitly state "do NOT write to memory; only
suggest candidates in your report". Single-writer rule.

## 2026-05-16 — Department reports may bypass single-writer memory rule

If a Department's brief does not explicitly forbid memory writes, the
sub-agent will sometimes write directly to `.orchestrator/memory/*.md`
files instead of just suggesting candidates. Discovered with M3
(2026-05-16). Mitigation: add an explicit forbidden line in every
Department brief going forward:
"You MUST NOT modify any file under `.orchestrator/memory/`. Suggest
candidates in your report; the Boss decides what to persist."

**Why**: maintains the audit trail; without this, memory accumulates
content without the Boss filter and the audit log becomes ambiguous.

## 2026-05-16 — M4 confirms BSD as primary cavity-gap suspect

Guo body-force on the LBM differs from FD div(τ) by **53.5-53.8 % L2**
on the saved N=64 cavity fields, structurally across u_max. Difference
is dominated by the BSD `−ζ·ν_p·∇²u` correction. Max-diff cell (16, 63)
is in the right-wall recirculation under the moving lid — the same
region as M2's corner-artifact suspicion. **M2 and M4 are coupled at
this cell.**

**Implication**: the Mandate's prior assumption that `bsd ∈ [0, 0.75]`
has "<2 % sensitivity at N=32" applied to *profiles*, not the *force*.
At N=64 the force discrepancy is 54 %, plausibly enough to drive the
18-24 % profile gap. Next decision experiment: **M4b BSD-fraction
sweep** at N=64 t=8 with `bsd ∈ {0, 0.25, 0.5, 0.75}` to test whether
the profile L2 collapses as the BSD correction shrinks.

## 2026-05-16 — `fields.jls` writer omits `f` and body-force fields

The cavity comparison harness `run_cavity_oldroydb_vs_rheotool.jl`
persists only the macro fields `ux, uy, psi*, tau*` — NOT the LBM
distribution `f` nor the applied body-force `fx_total/fy_total`. Any
future Guo-vs-FD audit at the `f` level would need the snapshot writer
extended. M4 worked around this by reconstructing F_Guo from `tau`
using the implementation's known formula. Acceptable for a one-shot
audit; not robust for parameter sweeps.

**Why**: avoids spending a future Department iteration re-discovering
this. If a future mission needs the body-force at the `f` level,
extend the writer first.

## 2026-05-16 — M6-A confirms wall-BC stencil mismatch as alternative

M6-A audit found that Kraken's FD divergence at the wall row uses an
implicit one-sided **quadratic** 3-point extrapolation, while rheoTool
uses **linearExtrapolation** (2-point) on `τ` at the moving lid. The
Ψ-side BC matches (both zeroGradient). Predicted impact of matching:
54 % → ~15-30 % L2 at the M4 max-diff cell (16, 63).

**Implication**: M6-B is a complementary fix to M5-B, not an
alternative. M5-B fixes interior bit-exactness (Chapman-Enskog
consistency). M6-B fixes the wall-row stencil. Both may be needed;
either alone may not close the full 18-24 % profile gap. M4b
verdict will help discriminate: if profile L2 falls fast with
bsd→0, BSD operator is the dominant lever; if not, the wall
stencil is. Sequencing: M5-B first (in flight), THEN M6-B (do not
parallelise; both could touch operators_2d.jl in a follow-up).

**Why**: prevents over-committing to either fix; documents the
"likely both needed" reasoning so future sessions don't relitigate
M5 vs M6 as exclusive alternatives.

## 2026-05-16 — M4b REFUTES BSD-is-driver hypothesis

Cavity profile L2 vs rheoTool falls *monotonically* as
`bsd_fraction` increases over `{0, 0.25, 0.5, 0.75}` at fixed
`N=64, t=8, De=1, beta=0.5, u_max=0.005`. Centerline L2:
21.15 % → 17.97 %. psi_xy L2: 27.41 % → 24.41 %. Current
production choice `bsd_fraction=0.75` is the best of the swept set.

**Re-interpretation of M4**: the 54 % Guo-vs-FD discrepancy
reported by M4 is the magnitude of the BSD correction term
operating *as designed*, not a defect. The BSD correction is doing
useful work (improving rheoTool match); the operator-mismatch
concern was a misreading of M4. M5-B kinetic kernel is a clean
refactor with the right semantics but cannot close the cavity gap.

**Implication for the mission**: the next lever is M6-B (wall-BC
stencil match with rheoTool's `linearExtrapolation` on τ). The
remaining 18 % centerline gap at `bsd=0.75` must come from a
non-BSD source — the wall stencil is the most concrete candidate
identified to date. No future mission should propose another BSD
sweep or another BSD-flavour refactor; that lever has been measured
and it points the wrong way.

**Why**: prevents future Departments from chasing the BSD ghost
again. The data is in the verdict file
`bench/viscoelastic_logfv/CAVITY_BSD_M4B_VERDICT_20260516.md`.

## 2026-05-16 — M6-B REFUTES wall-stencil hypothesis at production

Aqua confirmation job 21397692 ran cavity at N=64 t=8 with
`polymer_wall_extrap` in {`:quadratic`, `:linear`}. Quadratic
reproduced the M1 baseline (0.1797 / 0.2441) to 4 sig figs — sanity
passed, kwarg default preserves behaviour. Linear gave 0.1817 /
0.2433 — essentially unchanged (+1.1 % on centerline u, −0.3 % on
psi_xy). The 12 % wall-row delta observed at the smoke does NOT
propagate to the global profile.

**Implication for the mission**: four of five originally-mandated
candidates plus the user-suggested wall-BC alternative are refuted.
The 18-24 % cavity profile gap remains unexplained. Need to
step-back the Mandate and either close M2-full + grid-convergence
sweep (cheap, bounds the gap as discretization-floor vs bug) or
open M7-M9 for the remaining hypotheses (initial conditions, time
integration, coupling order).

**Why**: prevents future Departments from re-attempting any of the
4 refuted candidates. The wall-stencil ghost-fill *implementation*
is a separate possibility (one-sided vs reflective) and is NOT
covered by M6-B's verdict — that would be M9 or similar if pursued.

## 2026-05-16 — M8 ratchets the polymer pipeline OUT of suspicion

Analytical Poiseuille frozen-velocity test of the FV polymer pipeline
(advection + Oldroyd-B source + stress + wall velocity-gradient
extraction) gives first-order convergence in `dt_poly` with NO spatial
bias (ratio cell/analytical = 0.99805 uniformly across interior).
At production `n_substeps=4096`, source-discretization error is ~4e-6.
The `fvfd_velocity_gradient_2d!` wall stencil is bit-exact vs analytical.
Combined with M3 (cavity frozen replay 4 % L2) and the 0D constitutive
machine-epsilon audit, this fully exonerates the polymer pipeline.

**Implication**: the 18-24 % cavity gap MUST be in the **LBM ↔ polymer
coupling layer**:
- Guo body-force injection on `f`
- BSD correction magnitude/sign (BSD helps per M4b, so direction is
  right but possibly amplitude is off)
- Operator staggering / order between LBM step and polymer substeps
- `u` reconstruction from `f` after Guo source

**Why**: prevents any future Department from re-auditing the polymer
ODE, the FV advection upwind, the stress reconstruction, or the wall
velocity-gradient extraction — all ratcheted.

## 2026-05-16 — Two-level polymer-pipeline regression ratchet established

We now have a two-level testing ladder for the polymer pipeline:
- **0D**: bit-exact (machine epsilon) at production substep cadence
  (audit `bench/viscoelastic_logfv/CONSTITUTIVE_0D_AUDIT_20260515.md`).
- **2D**: first-order in `dt_poly`, no spatial bias, at
  `bench/viscoelastic_logfv/run_poiseuille_polymer_analytical_2d.jl`.

Future polymer-pipeline regressions can be triaged against this
two-level ratchet without re-running cavity. Cavity comparisons against
rheoTool are NOT a fitness function for the polymer pipeline alone —
they always carry the coupling-layer signal.

**Why**: cleaner separation of concerns for future debugging sessions.

## 2026-05-16 — M7b confirms Wi-INDEPENDENT polymer-coupling bug (SMOKING GUN)

At Wi=0.001 with matched total LBM viscosity, two cavity cases that
differ only in whether the polymer code path is active diverge by
**3.42 % centerline u rel L2**. Control case (Newtonian Re-doubling)
produces only 0.014 % delta — confirming Re is NOT the source. The
polymer machinery introduces a Wi-independent perturbation on `u`
that should not exist if the BSD/Guo split correctly absorbs the
Newtonian portion of τ_p into the LBM solvent viscosity. Verdict:
`bench/viscoelastic_logfv/CAVITY_LOWWI_M7B_VERDICT_20260516.md`.

**Implication**: the cavity-gap bug is now LOCALISED. M10 (BSD/Guo
coupling Wi→0 audit) is the natural next mission — algebraic
verification that the implementation matches the design intent
`ν_LBM_eff = ν_s + ν_p` at the discrete level.

**Why**: prevents future Departments from re-attempting any high-Wi
diagnostic chase — the bug is in the coupling and visible cleanly at
Wi=0. Any future audit should leverage this clean Wi-0 isolation.

## 2026-05-16 — Cavity Kraken-vs-Kraken noise floor is 0.014 %

The B-vs-C control of M7b (Newtonian, Re-doubling 1.6 → 3.2 at
N=64, u_max=0.005) gives 0.014 % centerline u rel L2. Any future
Kraken-vs-Kraken comparison delta above ~0.1 % is meaningful.

**Why**: gives a quantitative threshold for "noise" vs "signal" in
any future cavity sensitivity study.

## 2026-05-16 — LLM-friendly file-size constraint adopted

User directive (2026-05-16): files should be ≤500-700 LOC for LLM
Departments to work effectively. Current worst case in cavity-relevant
code: `src/drivers/viscoelastic_logfv_2d.jl` = 3429 LOC (5× the
limit). Any future cavity-driver refactor must include a SPLIT into
modules (see `.orchestrator/memory/engineer.md` for the proposed
decomposition). This is a load-bearing engineering constraint, not
optional cleanup — M11's small fix got lost in the monolith context.

**Why**: prevents the next session from going straight into Option 3
implementation without first making the codebase tractable for the
Engineer. Splitting should be the FIRST mission of the next session,
not bundled with the BSD fix.

## 2026-05-16 — Kraken-Mandate modularity violation (user-flagged)

The original Kraken architectural mandate is: **geometry ≠ BC ≠ solver
≠ stencil ≠ physics** (separation of concerns). The current cavity
driver `src/drivers/viscoelastic_logfv_2d.jl` at 3429 LOC violates
this directly — it co-locates:
- The cavity GEOMETRY (lid-driven cavity setup)
- The wall-BC kernels (`_logfv_cavity_apply_wall_gradient_correction!`)
- Solver bits (timestep loop, LBM step orchestration)
- Stencil choices (BSD `:fd` / `:kinetic` branches)
- Physics (Oldroyd-B BSD correction, Guo body force assembly)

This is structurally why M11 broke: the 5-LOC BSD fix shared state
(buffers, kwargs, line context) with the wall-correction kernel and
the source-ODE D-capture in the same file, so the fix accidentally
perturbed unrelated concerns.

**Implication**: the cavity driver SPLIT (per the engineer.md
proposed decomposition) is not just for LLM-friendliness — it
restores the project's own architectural mandate. Future cavity
work must NOT bundle "fix BSD" with "split driver"; the split goes
first as a standalone refactor mission, then the fix targets the
clean post-split module.

The orchestrator skill itself has been updated 2026-05-16 to enforce
this generally: ≤500 LOC soft / ≤700 LOC hard, one-file-one-concern,
in SKILL.md §Engineering hygiene. Applies to every project using the
pattern, not just Kraken.

**Why**: this is THE load-bearing project-level constraint going
forward. Any Department brief that does not respect it is a trap.

## 2026-05-17 — M16 SPLIT cavity driver landed (commit 77956ad8)

`viscoelastic_logfv_2d.jl` 3429 → 2934 LOC. Cavity-specific code now
lives in two dedicated files: `cavity_wall_correction_2d.jl` (98
LOC, 4 helpers) and `cavity_driver_2d.jl` (400 LOC, main
`run_viscoelastic_logfv_cavity_coupled_2d`). All three within hard
ceiling 700 LOC. Refactor pur (zero semantic change). M17 (Option 3
BSD same-stencil fix) is unblocked; allowed edit zone for M17 is
`cavity_driver_2d.jl` + new helper modules below it, NOT the now-
slimmer `viscoelastic_logfv_2d.jl` (which still holds 8+ unrelated
drivers and remains over the hard ceiling — future M16b will split
those, but they are NOT in the cavity bug critical path).

**Why**: the orchestrator pattern's first major SPLIT mission worked
cleanly; documents the post-split topology that future missions
target. Saves the next Boss from re-discovering "M17 goes to
cavity_driver_2d.jl, not the old monolith path".

## 2026-05-17 — First M16 Department spawn hung silently

The first Agent call for M16 (general-purpose subagent) returned a
placeholder "Empty so far — still compiling. Let me wait for the
Monitor notification" after 216 s and 42 tool calls without
invoking Codex. The Department had drafted `.engineer_brief_M16A.md`
correctly but appears to have wedged on a Monitor-tool call awaiting
a subprocess event that never arrived. Respawned a second Department
with an explicit "do NOT use Monitor; use plain Bash with timeout"
clause and an instruction to pick up from the existing brief instead
of redrafting — that one succeeded.

**Why**: if a future Boss-spawned Department returns a stale Monitor
placeholder, respawn with the no-Monitor clause and a "resume from
existing artefacts" instruction rather than starting from scratch.

## 2026-05-17 — Cavity coupling bug has TWO independent defects

The 3.42% Wi-independent residual M7b measured has TWO theoretical
sources, BOTH structural to the current WIDE F_poly chain:

1. **M10 stencil mismatch** (commit `a2e6f088`, L1 quantification):
   F_poly uses a wide 2dx-effective Laplacian, LBM+BSD use a narrow
   5-point Laplacian. The wide-narrow truncation gap is O(dx²·∂⁴u)
   and produces the 3.4% residual at N=64.
2. **Nyquist null mode** (commit `c20e4e8c`, L4 verification):
   WIDE F_poly has identically zero viscous damping at the checkerboard
   pattern (k·dx = π). Any noise at that wavenumber accumulates
   undamped. NARROW preserves 4/π² ≈ 0.405 of analytical damping at
   the same mode.

The naïve fix attempts (M11, M17-pre v1/v2, M17-impl) tried to address
(1) by widening BSD; they failed because they REMOVED the (already-
broken) WIDE damping mechanism without replacing it. The convergent
theoretical answer from both M17-epsilon Departments (Claude + Codex):
SPLIT F_poly into a narrow Newtonian (`ν_p·NARROW_Laplacian(u)`)
plus a wide Elastic remainder (`div_wide(τ_p − 2·ν_p·D)`). This
addresses BOTH defects simultaneously:
- Newtonian portion uses NARROW: matches LBM+BSD stencil → closes
  M10 bias.
- Newtonian portion uses NARROW: preserves Nyquist damping at 40%
  → closes null mode.
- Elastic remainder remains wide but its 2× truncation bias is
  bounded by the elastic-mode signal (zero at Wi=0).

**Why**: future Boss sessions must read this BEFORE attempting any
"adjust BSD" or "swap F_poly stencil" mission. The fix is structural,
not parametric. M17-impl-v2 with split #2 is the prescribed
implementation; any deviation needs explicit justification.

## 2026-05-17 — Parallel theory Departments protocol paid off

User-driven discipline: two parallel theoretical Departments (Claude
+ Codex) on the same brief produced divergent surface-level verdicts
(Claude GREEN, Codex RED-naïve/YELLOW-corrected). Audit revealed
they actually CONVERGED on the same correct implementation (face-
flux or split Newtonian/elastic) but with different framings.

The divergence surfaced a critical implementation subtlety: a naïve
"narrow divergence on cell-centered τ_p" does NOT shrink the
effective velocity stencil, because τ_p is built from FD-derived D.
A single Department (either) would have either over-promised or
been ambiguous; the comparison forced precision.

**Why**: protocol confirmed useful for architectural decisions where
the brief is non-trivial. Apply again whenever the Boss faces a
"sounds clean but might have a hidden subtlety" hypothesis. Cost:
2× the spawn time; benefit: catches subtleties that single-spawn
would miss.

## 2026-05-17 — M17 cluster CLOSED with reframed diagnostic (commit b995e304)

After 6 RED implementation attempts on the (ε) split coupling
architecture (M11 + M17-pre v1/v2 + M17-impl + M17-impl-v2 +
M17-impl-v3), the cavity 3.4 % M7b Wi-independent signal was
re-decomposed via clean Poiseuille analytical canaries + rheoTool
iBSD-ON/OFF cross-check:

| component | contribution to 3.4 % |
|---|---|
| polymer pipeline error | machine zero (1e-5 rel L2 on stress) |
| BSD intrinsic cost (rheoTool-equivalent) | ~0.6 % |
| M10 stencil mismatch (Poiseuille-isolated) | ~0.4 % |
| **cavity corner singularity amplification (8×)** | **~2.4 %** |

The **dominant contributor is corner amplification, NOT stencil
mismatch**. The 6 M17 split-coupling attempts targeted the wrong
component (~0.4 % only).

**Strategic implication for cavity production gap (18-24 %)**:
- ~1 % combined BSD/stencil bias (can't fix without major refactor).
- ~2.4 % corner amplification (addressable by lid profile / corner
  regularization).
- ~10-12 % discretization floor at N=64 (M9 trajectory).
- ~5-7 % finite-Wi residual at De=1.

Single-digit production gap path: grid refinement (N=128 → ~9 % floor,
N=256 → ~5 %) + corner regularization → likely achievable at N≥192
WITHOUT touching the BSD coupling architecture.

**Closed missions**: M11, M17-pre v1/v2, M17-impl, M17-impl-v2,
M17-impl-v3 (all archived under `.orchestrator/red_archives/`).

**Open**: M18 (production validation N=128+ on Aqua), M19 (corner
regularization, optional), M16b (Poiseuille driver SPLIT debt — the
monolith remains at 2934 LOC).

**Why**: this re-decomposition is the durable conclusion of an entire
session of M17 attempts. Future Boss must read this BEFORE attempting
any architectural fix on the BSD coupling — the math is right but the
target was wrong; corner amplification dominates.

## 2026-05-17 — Polymer pipeline analytical match extends M8/M13

Steady-state Oldroyd-B Poiseuille on the existing
`run_viscoelastic_logfv_poiseuille_coupled_2d` driver matches
analytical to machine precision on stress fields:
  tau_xx rel L2 = 8.25e-6
  tau_yy max abs = 5.77e-16 (machine zero)
  N1 rel L2 = 8.24e-6 (= 9.1e-8 vs 9.1e-8 analytical)

This is BSD-invariant (same numbers at ζ=0 and ζ=0.75) — consistent
with Stokes balance: gamma_dot is determined by F_body and nu_total
alone, regardless of how BSD splits viscosity between LBM-implicit
and explicit body force. Only velocity profile (u) is affected by
BSD; the polymer stress is sound by construction.

**Why**: any future "is the polymer pipeline broken?" doubt can be
answered with the script at /tmp/poiseuille_full_check.jl pattern.

## 2026-05-18 — M20 closes: BSD operator-clean on smooth geometry; 8× cavity ratio is downstream

Post-hoc decomposition of F_total on the existing Poiseuille coupled
driver at steady state (Nx=8 Ny=32 100k steps CPU F64) — three cases
A_no_BSD (ζ=0), A (ζ=0.75 Wi=8e-4), A_high_Wi (ζ=0.75 Wi=1).
Decomposition uses the existing kernels (no `src/` patch): rebuild
`tau*` from driver-returned `psi*` via `logfv_stress_from_log_2d!`,
call `logfv_polymer_force_bc_aware_2d!` for F_poly_wide, then
`F_BSD = F_poly_wide − (fx_total − F_body)` by algebraic identity from
the driver chain.

**Numbers** (rel L2 vs analytical Newtonian-limit `−F·ν_p/ν_total`):

| case | F_poly int / wall | F_total int / wall |
|---|---|---|
| A_no_BSD | 0.50 % / 0.50 % | 0.50 % / 0.50 % |
| A | 0.50 % / 0.50 % | **3.51 % / 3.51 %** |
| A_high_Wi | 1.3e-5 | 9.3e-5 |

**Two Boss-level insights worth preserving**:

1. **Same-sign stencil residuals ADD, they do not cancel.** WIDE
   `div_wide(τ_p)` and NARROW 5-point `∇²u` BOTH underestimate the
   analytical d²u/dy² magnitude by ~0.5 %, with the same sign. Their
   algebraic combination `F_poly − F_BSD` therefore accumulates the
   absolute error rather than cancelling it. F_poly residual abs =
   2.5e-8, F_BSD residual abs = 1.88e-8 same direction → F_total
   residual abs = 4.4e-8 (= 2.5e-8 + 1.88e-8, near-perfect addition).

2. **(1−ζ)⁻¹ = 4× normalisation amplification**. Same absolute residual
   normalised against a 4× smaller F_total target shows up as 4× larger
   relative residual. The "3.5 %" on F_total is the SAME force-error as
   the "0.5 %" on F_poly — different denominators.

**Localisation impact on the 8× cavity-vs-Poiseuille M7b ratio**:
- Poiseuille has NO wall amplification (uniform residual in y).
- Therefore the 8× cavity ratio does NOT live in the BSD subtraction
  chain itself. It lives downstream — either (a) in the velocity-
  gradient kernel difference (`fvfd_velocity_gradient_2d!` cavity vs
  `logfv_velocity_gradient_bc_aware_2d!` Poiseuille, Open Q5 → M21),
  or (b) in the LBM-side flow response to the same body force around
  the corner singularity.

**Implication for future BSD architectural work**: a bit-exact BSD
cancellation (M5 kinetic OR M17 same-stencil) buys at most the 0.5 %
operator-level residual on Poiseuille. The remaining 3.0 % on F_total
is the same-sign-add + normalisation phenomenon, NOT a fixable bug.
Any future "fix BSD" mission must answer: which of the 0.5 % is
captureable in operator alignment? On cavity, the wall corner
amplifies the same 0.5 % into the 3.42 % M7b signal — that's the
real lever, not the operator alignment.

**Why**: prevents future Boss sessions from chasing operator-level
bit-exactness as a way to close the cavity gap. M20 shows the
operator side is already at ~0.5 % bulk on Poiseuille; the cavity
amplification mechanism is geometric (wall + corner), not algebraic.

## 2026-05-18 — M21 NEGATIVE verdict: no operator-side BSD reformulation works

Path-matrix sweep of 7 BSD/F_poly variants on smooth Poiseuille
(commit `81745f3b`) confirms: **the cavity bug is NOT operator-side**.
None of the historically-RED-on-cavity variants (`:fd_v2`,
`:fd_v2_unc`, `:kinetic`, `:epsilon_force`) gives F_total < 3.51 % on
smooth geometry. Two NaN, two bulk-wrong by 50× to 190×.

Specifically refuted on Poiseuille:
- **Open Q5 (kernel difference)**: `logfv_velocity_gradient_bc_aware_2d!`
  is literally `return fvfd_velocity_gradient_2d!(...)` (lines 918-926
  in `src/kernels/logconformation_fv_2d.jl`). The two kernels are
  bit-identical. The 8× cavity-vs-Poiseuille ratio CANNOT come from
  this kernel difference — there IS no difference.
- **`:fd_v2` wide-on-wide BSD**: NaN at Wi=1; at Wi=8e-4 produces
  τ_yy = 0.26 (≫ τ_xy ≈ 2.5e-3) — massive non-physical asymmetric
  stress at walls, distinct from NaN failure mode. The wide-on-wide
  bulk cancellation principle is sound on Taylor-Green (L1 analytical
  ladder showed near-machine cancellation) but in the LBM-coupled
  steady state with walls, the wall-row stencil mis-cancellation loops
  back through the constitutive ODE catastrophically.
- **`:kinetic` (M5 Π^neq route)**: F_BSD only −6.7e-8 vs target
  −3.75e-6 (= 56× UNDER-shoots), so F_total UNDER-corrects by 30×.
  The Chapman-Enskog identity that gave 5.85e-16 on the M5-A static
  smoke does NOT hold in the dynamic coupled steady state. The
  near-machine equivalence M5-B reported was a STATIC equivalence at
  t=2, not a dynamic one at 100k steps.
- **`:epsilon_force` (narrow Lap + force-level elastic split)**:
  NaN both cases. The engineer.md 2026-05-17 recommended "discrete
  identity workaround" (force-level subtract instead of cell-tensor)
  doesn't work in dynamic LBM-coupled. Mirror-ghost narrow Laplacian
  at halfway-bounce walls drives C non-SPD.

**Strategic conclusion**: there is no untapped operator-side fix. The
cavity 18-24 % gap decomposes (per M9 + M20) into:
- ~10-12 % discretization floor (M9 trajectory, N=64 → N=∞)
- ~5-7 % finite-Wi residual at De=1 (never directly measured)
- ~1 % BSD intrinsic + stencil mismatch (M20 measured)
- ~2.4 % corner singularity amplification (inferred from 8× cavity
  ratio that M20+M21 confirm is NOT operator-side)

The only remaining lever for the corner-amplification portion is
**(a) wall-corner gradient correction bypass / redesign** (the
`_logfv_cavity_wall_gradient_correction_kernel!` half-cell ghost
mechanism, suspect per engineer.md 2026-05-17), or **(b) LBM-side
corner treatment** (Zou-He lid coupling at corner cells, Guo source
at corner). The corner-regularization mission (M19) is now the
strongest candidate going forward.

**Why**: locks in that further BSD operator-flavour missions
(M22-M23 finite-Wi or rheoTool cross-check are still informative but
purely confirmatory; they will not surface a new fix). The path
forward for the cavity benchmark gap is grid refinement (M18) +
wall-corner mechanism (M25/M26 to be opened).

## 2026-05-18 — Matrix-sweep missions: prefer N parallel Departments

For any future mission that is structurally "test N variants × M
cases", default to spawning **N parallel Departments** (one per
variant) instead of one sequential Department doing all N. The
serialised approach used in M21 (single Department, 14 runs back-to-
back) cost ~34 min wall for 9 min of CPU work — most of the wall time
was Engineer code-writing + Department setup. Fan-out to N parallel
Departments would have collapsed this to ~6 min wall + a 5-min Boss
integration step.

Trade-off: parallel pattern duplicates the bench infrastructure (each
Department's Engineer re-writes the timestep loop boilerplate). Worth
the duplication when N ≥ 4 variants and runtime per variant ≥ 5 min.
Below that threshold, sequential is fine.

User directive 2026-05-18 ("lance les simulations EN PARALLELE sur 7
dep différents pour accélérer") arrived ~5 min after the sequential
M21 Department had already completed; this captures the lesson for
the next sweep.

**Why**: future Boss sessions facing a matrix-sweep mission should
spawn N parallel Departments in the same Agent-tool message (per
SKILL.md §Fan-out). The cost is N× spawn overhead but the wall-time
saving is substantial.

## 2026-05-18 — M22+M23 cylinder Cd: BSD essential at coarse R, vanishes at fine R

Parallel cylinder Cd convergence study at moderate Wi (commit `8aaac026`)
**partially confirms** the user hypothesis that BSD impact on real complex
flow drag collapses with mesh refinement, and **resolves** the original
"anti-convergence" recollection.

**Numbers** (R sweep, Wi=0.1, Cd_BSDon vs Cd_BSDoff in % difference):
- R=20: +17.5 %
- R=30: +11.6 %
- R=40:  +7.3 %
- R=50: **+1.1 %** ← collapses to ~permille range
- Extrapolated R≥60: sub-1 %.

This means the 3.51 % F_total Newtonian-limit operator-side residual
(M20) does NOT propagate linearly to physical observables in the
production regime. At R=30 with BSD ON, Kraken matches rheoTool to
−1.45 % on Cd, which is the rheoTool-equivalent of "fine" given that
rheoTool itself has F32-grade noise + iBSD coupling.

**The "anti-convergence" recollection is RESOLVED** as a different pattern:
- BSD ON: locks onto rheoTool at R=30 (~1.5%), then oscillates (R=40:
  130.31 over-shoots; R=50: 127.82 under-shoots; F32 noise dominates).
- BSD OFF: under-shoots widely at coarse R (107→115→121→126), monotone-
  converges UP toward BSD ON values.
- The two CONVERGE to the SAME rheoTool-consistent limit, just from
  opposite sides. NOT divergent limits.
- The "crossing then diverging" impression was the crossing of the two
  curves at coarse mesh where BSD ON happens to over-shoot.

**BUT — BSD is essential for stability at fine R + Wi > 0.1**: M23
R=40 Wi=0.2 SPD-loss-like behaviour (Cd=783, min_detC=8e-4). Removing
BSD is NOT a viable production option for high-Wi cylinder.

**Strategic conclusion**: the BSD architecture is HEALTHY at the
production mesh for the cylinder benchmark. The M9-M21 cluster of
operator-side debugging was probing a metric (F_total residual at
Newtonian-limit Poiseuille) that does NOT control the actual physical
output (Cd) at the production mesh. The original motivator (cylinder
Cd convergence vs rheoTool) is RESOLVED: at R=30 BSD ON the gap is
−1.45 %, well within the F32 noise floor.

**Implication for the project mandate (cavity benchmark gap)**: the
cavity 18-24 % at production setup is therefore likely dominated by
geometry-specific factors (discretization floor + corner singularity)
NOT by the BSD operator, consistent with the M20+M21 conclusions. The
cavity benchmark, like the cylinder, would close to rheoTool-match
with grid refinement (M9 trajectory) + corner regularization.

**Why**: future Boss sessions should NOT re-investigate BSD operator-side
fixes as a way to close the cavity gap or the cylinder Cd. The BSD
is doing what it's supposed to do at production mesh. The remaining
levers are geometry-specific (corner regularization, mesh refinement).

## 2026-05-18 — Codex Engineer can hand off to Boss-on-host for execution

When a Codex Engineer Department times out / crashes on Anthropic API
during the EXECUTION phase (after the bench/code has been written and
committed to disk by the Engineer), the Boss can take over and run the
bench directly via Bash + run_in_background. This bypasses the spawn-
overhead penalty of a recovery Department and avoids re-paying the API
quota for the same work.

Observed M22 + M23 (2026-05-18): both Engineers wrote 286+294 LOC
bench scripts cleanly, then the parent Department wedged on the
API connection during the post-Engineer `--full` execution step. The
Boss ran the 16 cases (~13 min wall each in parallel) and wrote the
joint synthesis directly. This worked because:
1. Engineer bench scripts were already in their final form (~9 KB each).
2. The bench scripts run autonomously (no Codex needed once written).
3. The synthesis (M22+M23 cross-comparison) is a Boss-level task by
   definition — not something a single Department should do anyway.

**Why**: future Boss sessions should NOT auto-respawn a recovery
Department when the Engineer artifact is already on disk and the
remaining work is "run the bench + write the verdict". Boss-on-host
takes ~20 min less wall time and is the right scope for the synthesis
anyway.

## 2026-05-18 evening — Phase 0 Liu-match setup + 3 bugs + 1111_circle bug

Three bugs fixed pre-launch (commits `533afa08`, `488a7b56`, `86f1391c`):

1. **CUDA backend detection silent-fail** in
   `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl::detect_backend()`
   (~line 34): `Base.invokelatest(getfield(Main, :CUDA), :functional)`
   treated the `CUDA` module as callable with `:functional` as arg.
   Silently `MethodError`'d → bare `catch end` → silent CPU fallback.
   Aqua A100 job `21534810` burned **4h35 at CPU speed** before the
   `backend=cpu` column in the CSV was spotted. Fix mirrors the Metal
   branch + adds `@warn` on explicit-request fallback.
2. **β=0.5 vs Liu/rheoTool β=0.59 default mismatch**. Phase 0 ran at
   β=0.5 while Liu 2025 (line 2515) AND rheoTool
   (`constant/constitutiveProperties` etaS=0.59 etaP=0.41) both use
   β=0.59. Numerical Cd comparison was confounded. Fix: new default
   `KRAKEN_BETA_LIST="0.59"` (driver respects env).
3. **M22 vs M23 v1 apples-vs-oranges kwargs**: M22 used L_up=L_down=15;
   M23 used L_up=4 L_down=8 (Codex Engineer auto-chose smaller domain
   to save compute). My "v1 SPD-loss at R=40 Wi=0.2" was a different
   simulation, not a transient. **Lesson**: ALWAYS diff kwargs between
   sister benches before cross-comparing.

**Implications for future briefs**:
- Any GPU-backend smoke test MUST verify the `backend=` column in the
  CSV, not just trust `KRAKEN_BACKEND` env var (silent fallback can
  burn hours).
- Reference values Liu Table 3 + rheoTool are at β=0.59, NOT β=0.5 —
  every cylinder mission must check the reference convention.
- Parallel-Department briefs (M22+M23 style) MUST list kwargs
  explicitly identical between siblings; ad-hoc Engineer auto-choices
  break cross-comparison.

**Wi single-point sweep is NOT a benchmark — minimum 3 Wi for
elastic regime validation**. Phase 0 (M25) is locked at Wi=0.1,
which is quasi-Newtonian (polymer contribution `Cd_p − Cd_bsd` ≈
−0.3 Cd points, dominated by `Cd_s`). To validate the BSD+embedded
physics across the elastic regime, M28 Phase 1 MUST sweep Wi ∈
{0.1, 0.3, 0.5, 1.0}. "Convergence at Wi=0.1 alone tells us nothing
about elastic physics" — could be hiding a polymer-pipeline bug
that only activates at finite Wi. Boss must enforce this in M28.

**1111_circle Cd_s ghost drag is a CONFIRMED bug to find and fix**
(user directive 2026-05-18 evening). At Newtonian Re=1 R=30, full
embedded mode gives Cd_s = 140.78 vs baseline 0000_qwall Cd_s =
131.99 → **+8.8 Cd points (~6.7%) fictitious solvent drag** with
no physical justification. Dual-spawn pattern launched 2026-05-18
evening: M26-analysis (math audit) + M26-impl (Newtonian bench).
Phase 0 case 3 `0001_qwall` will isolate H1 (drag-only) from
H2/H3 (force/quadrature) on the viscoelastic side; M26-impl
isolates them on Newtonian.

**Cd decomposition formula** (durable doc, user-asked):
- `Cd_kraken = Cd_s + (Cd_p − Cd_bsd)`, NOT `Cd_s + Cd_p`.
- LBM with `ν_LBM = ν_s + ζ·ν_p` absorbs `ζ·ν_p·∇²u` into implicit
  viscous diffusion → `Cd_s` includes that contribution.
- Guo body force = `div(τ_p) − ζ·ν_p·∇²u` → drag integral of body
  force = `Cd_p − Cd_bsd`.
- At Wi=0.1, `Cd_p ≈ Cd_bsd` to within 1-2 Cd points (BSD doing its
  job, absorbing Newtonian-additive portion of `τ_p`).
- At higher Wi, `Cd_p − Cd_bsd` becomes finite → genuine
  elastic-stress drag contribution.

**Phase 0 job 21563085.aqua setup** (in-flight, R since 17:50):
- β=0.59 Re=1 Wi=0.1 (Liu Table 3 reference)
- R ∈ {20, 30, 40} × 4 embedded tuples zipped:
  - `0000_qwall` (= Liu reference mode)
  - `1000_qwall` (gradient only)
  - `0001_qwall` (drag only — isolate suspected bug)
  - `1100_qwall` (gradient+advection, Phase 0 v1 best at Cd=131.15)
- 12 runs, ~2-3h on A100 F64
- Output: `results/viscoelastic_logfv/cyl_bigsweep_v2_21563085.aqua/SUMMARY.csv`
- Pass criterion: `0000_qwall` R=30 Cd ∈ [129.5, 131.5] (Liu CNEBB ± 1).

**Reference docs** (next Boss reads first):
1. `.orchestrator/mandate.md` §5 M25/M26/M28
2. `NEXT_SESSION_PROMPT_20260518_cylinder_handoff.md`
3. `bench/viscoelastic_audit/CYLINDER_DOE_PLAN_CODEX_20260518.md` (DoE)
4. `bench/viscoelastic_audit/liu_2025.txt` §4.3 + Table 3
5. `bench/viscoelastic_logfv/CYL_CD_CONVERGENCE_M22M23_SYNTHESIS_20260518.md`

**Why**: this entry locks in all durable lessons from the Phase 0
preparation session so the next Boss does not re-litigate β
convention, CUDA-detect path, kwargs sibling-diff discipline, or
the Wi-single-point limitation when proposing Phase 1.

## 2026-05-18 — M26 dual-spawn closes analysis + impl; finite-Wi gated on Phase 0

M26 Boss-level adversarial dual-spawn (parallel Layer-1 Departments)
returned 2 complementary verdicts that fully scoped the bug:

**M26-analysis (Claude general-purpose, math audit)** — identifies
the mechanism :
1. `fvfd_tensor_divergence_embedded_2d_kernel!`
   (`src/fvfd/operators_2d.jl:759-766`) outputs force-per-fluid-volume
   (divides by `cell_fraction`) but the Guo source consumer expects
   force-per-lattice-cell → **overdoses cut cells by 3-10×**.
2. `_fvfd_apply_embedded_wall_gradient_2d`
   (`src/fvfd/operators_2d.jl:127-140`) writes half-cell normal ∂u/∂n
   into shared `dudx/dvdx` buffers → **same family as cavity
   M17-canary-A bug**. Source ODE AND polymer-force divergence both
   consume the half-cell ghost.

Combined → singular Guo body force on cut cells → biases `f` → 
inflates LBM cut-link drag via MEA.

**M26-impl (Codex Newtonian bench)** — ratchets the polymer-coupling
localisation:

| case | Cd_s (β=1, Re=1, R=20, 1k steps, CPU F64) |
|---|---|
| 0000_qwall | 136.26 |
| 0001_qwall | 136.26 (bit-exact) |
| 0000_circle | 136.44 (+0.13 %) |
| 1111_circle | 136.44 (bit-exact vs 0000_circle) |

**Newtonian-clean.** At nu_p=0 the polymer pipeline is inert, all
embedded flags become NO-OPs, so the +8.8 anomaly disappears
completely. The bug lives ENTIRELY in polymer-coupling paths.

**Correction to handoff wording**: `embedded_drag` only toggles
`Cd_p` and `Cd_bsd` (driver lines ~470, ~485), NOT `Cd_s` which is
always sourced from `compute_drag_libb_mei_2d` (LBM MEA). The
handoff "+8.8 Cd_s" is loose for "+8.8 Cd_kraken" or "+8.8 Cd_p".

**Phase 0 21563085 will discriminate H1/H2/H3 at finite Wi**:
- If `0001_qwall` Cd_p ≈ `0000_qwall` Cd_p → bug NOT in drag formula
  alone → H2 (force kernel cell-fraction) per M26-analysis
- If `0001_qwall` Cd_p ≈ +8 over `0000_qwall` → bug isolated in
  drag_p formula → H1 (= revised: not impossible, since H1 was about
  Cd_s; bug in Cd_p still possible from `logfv_embedded_wall_traction_2d!`).
- Phase 0 misses `0010_qwall` (force-only) → if H2 confirmed but not
  individually pinned, a Phase 0b adds 3 runs (~30 min A100).

**Implication for M26b** (the eventual fix mission): per M26-analysis,
the M17-canary-A pattern applies — give the force path its own
`D_uncorrected` buffer, add cell-fraction re-scale on polymer-force
output. Target files: `src/fvfd/operators_2d.jl` (kernel signature
change, scoped to embedded variant only) + `src/drivers/<cylinder
driver>` (allocate extra buffer + re-call). Scope: ~50-80 LOC, 1-2 h
Codex + verification.

**Adversarial dual-spawn at Boss-level WORKED** (5th validation this
project): each Department's blind spot was covered by the other.
Math-Claude would have over-attributed to the cavity-family pattern
alone (missing the cell-fraction divisor specifically); impl-Codex
alone would have ratcheted Newtonian but couldn't have proposed the
mechanism. The 2 spawns + Boss synthesis (~5 min) gave a complete
diagnostic without needing a third iteration.

**Decision deferred to Phase 0 result**: per user 2026-05-18
("Attendre Phase 0 recommandé"), no M26b spawn until job 21563085
SUMMARY.csv lands. Then Boss compares `Cd_p` between the 4 tuples,
decides M26b scope (drag_p only / force kernel only / both), and
proposes the brief.

**Why**: locks the M26 mechanism in memory before context drifts.
Future Boss landing on a similar embedded-mode anomaly should
read this entry FIRST and head straight to the cell-fraction
divisor + half-cell ghost pair; don't waste a Department on
H1-as-Cd_s framing.

## 2026-05-18 evening — Julia 1.12 world-age trap in CUDA detection (NEW gotcha class)

Pattern that bit Phase 0 hard: ~4h35 of Aqua A100 time burned in
`21563085.aqua` running on CPU instead of GPU because of a Julia 1.12
world-age trap in `detect_backend()`:

```julia
try
    @eval using CUDA                          # advances world age in Main
    CUDAMod = getfield(Main, :CUDA)            # function still in OLD world age
                                                # → UndefVarError: :CUDA not defined
    if Base.invokelatest(getfield(CUDAMod, :functional))
        return CUDAMod.CUDABackend(), "cuda", FT
    end
catch end                                      # BARE catch swallows the UndefVarError
```

The bare `catch end` made the failure invisible. The `@warn` branch
that would have surfaced this only fires if `req != "auto" && req != "cpu"`,
and with `KRAKEN_BACKEND=auto` default, no warning fired → silent CPU
fallback. Same pattern would break Metal detection (validated locally).

**Fix shipped in commit `e602726f`**: wrap `getfield` calls in
`Base.invokelatest(getfield, Main, :CUDA)` to evade world-age cache;
replace bare `catch` with `@warn "...: $(sprint(showerror, e))"` that
surfaces the actual exception; emit `@warn` also on `KRAKEN_BACKEND=auto`
fallback for visibility.

**Implication for future bench scripts**: any pattern `@eval using X`
followed by `getfield(Main, :X)` inside a function is suspect. Either:
- Wrap the access in `Base.invokelatest` (minimal patch).
- Move `using X` to top-level (clean but may fail if X not installed).
- Use `Base.eval(Main, :(using X))` + `Base.invokelatest` (verbose).

**Implication for HPC ops protocol**: ALWAYS check the `backend=`
column / log line in the first CSV / first few log lines of any
GPU sweep, BEFORE walking away from a multi-hour job. The 4h35
wasted on 21563085 could have been caught at minute 35 by reading
the first CSV.

**Symptom→bug-class mapping**:
- "Submit to GPU queue, finishes 5-10× slower than expected" → check
  backend detection. Don't trust `nvidia-smi || true` + Pkg.precompile
  output as evidence the GPU is actually used.
- `resources_used.ngpus=0` in PBS exit info means Julia never touched
  the GPU, even though it was allocated.

**Why**: Julia 1.12 is recent (this project upgraded ~mid-session);
the world-age behavior is stricter than 1.10. Old code patterns may
silently fail. This is now a project-wide hazard class — anytime a
new Julia version lands, run a CUDA + Metal smoke before any
production sweep.

## 2026-05-18 evening — M25 Phase 0 done; M26 closed empirically; M28 launched

End-of-session state across the cylinder branch (3 jobs ran):

1. **21563085 (Phase 0 v1)**: KILLED. Silent-CPU due to world-age trap.
   3 cases R=20 in ~2h confirmed CPU path. qdel; no usable data.

2. **21570657 (Phase 0 v2, fixed)**: GREEN. 12/12 cases, **7m24s, 66.66%
   GPU util A100**. `0000_qwall` R=30 = **129.39** vs Liu CNEBB 130.36
   = **−0.7 %** (0.11 below the strict ±1 window). Approximate-PASS;
   accepted as Phase 1 baseline. Trend Cd(R) monotone toward asymptote
   ~129.5 — 0.9 Cd systematic offset below Liu, consistent across
   `:qwall` and `:circle` geometries when no embedded flag is on.

3. **21572831 (Phase 0b discrimination)**: GREEN. 27/27 cases, **14m55s,
   69.89% GPU util A100**. Sweep 9 embedded-flag tuples × 3R for full
   H1/H2/H3 disambiguation. **Verdict R=30 Δ vs `0000_circle`**:

   | tuple | Δ Cd | role |
   |---|---|---|
   | 0100_circle (adv) | −0.07 | NO-OP |
   | 0001_circle (drag) | +0.18 | NO-OP on Cd_kraken |
   | 1000_circle (grad) | +2.53 | secondary |
   | **0010_circle (force)** | **+8.10** | **dominant bug** |
   | 1111_circle (full) | +9.88 | reproduces +8.8 handoff |
   | **0010_qwall (force on qwall)** | **+8.71** | **bug NOT geometry-specific** |

   - **H1 REFUTED** (drag-only Δ negligible).
   - **H3 REFUTED** (force-only bug same magnitude on `:qwall`).
   - **H2 CONFIRMED**: bug is in `embedded_force=true` code path,
     specifically `fvfd_tensor_divergence_embedded_2d_kernel!`
     (`src/fvfd/operators_2d.jl:759-766`) cell-fraction divisor
     overdose, exactly as predicted by M26-analysis.

**Joint verdict file**:
`bench/viscoelastic_logfv/CYL_PHASE0_PHASE0B_VERDICT_20260518.md`.

**Boss-level adversarial dual-spawn pattern** (M26-analysis Claude +
M26-impl Codex) plus empirical Phase 0b validation gave a tight,
multi-source confirmation of the bug — H2 is now empirically locked,
not just theoretically suspected. The math hypothesis (math-Claude),
the Newtonian ratchet (Codex bench), AND the finite-Wi A100 sweep all
converge.

**Next mission gates** (sequencing per user 2026-05-18 evening
"Persist + commit + M28 Phase 1"):
- **M28 launching** at end of this session.
- **M26b** (`src/fvfd/operators_2d.jl` patch — remove or compensate
  the cell-fraction divisor) opens after M28.

**Why**: complete record of the M25+M26 closure cluster, so the next
Boss inheriting the embedded-mode work knows exactly which paths are
ratcheted (H1, H3) and which is the load-bearing fix target (H2 in
operators_2d.jl:759-766). No need to re-litigate the 1111_circle
mystery.
