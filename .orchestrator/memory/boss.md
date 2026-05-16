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
