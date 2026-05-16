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
