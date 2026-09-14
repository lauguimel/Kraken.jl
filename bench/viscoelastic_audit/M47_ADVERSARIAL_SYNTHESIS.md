# M47 — Viscoelastic R-sweep + Bouzidi-FL trace_C synthesis

Date: 2026-05-26  ·  Branch: `dev-viscoelastic`  ·  Status: post-M44 (Guo
half-step fix shipped, cluster M28-M42 CLOSED)  ·  Shared brief for
**adversarial Codex + Claude Code**.

---

## 0. Mission (read this first)

You are one of two independent reviewers (the other is Codex if you are
Claude, and Claude Code if you are Codex). You will NOT see the other's
report until the Boss compares them.

**Your task**:

1. Identify the SINGLE most likely root cause for either:
   - **A** — the R=60 Wi=1 temporal drift (Cd dropping monotonically
     113.234 → 112.130 → 109.424 across 100k → 200k → 400k, drift
     accelerating); AND/OR
   - **B** — the Bouzidi-FL polymer-chain trace_C blow-up (1938 →
     1.4×10⁷ at β=1.0 R=30 → R=60 Newtonian, polymer should be
     dormant since ν_p=0).
2. Design the SMALLEST possible patch test (CPU or Metal F32, **wall
   time <60s**) that discriminates between your top hypothesis and the
   next-most-likely runner-up.
3. Rank your top 3 hypotheses with evidence pro/contra each.

**Hard constraints on your verdict**:

- **WRITE-FIRST discipline** (per `[[feedback_codex_write_first_discipline]]`):
  drop an initial sketch (your top hypothesis + 1-paragraph mechanism
  + patch-test concept) to disk **within 5 minutes** before opening
  more than 4 source files. The sketch goes to `bench/viscoelastic_audit/
  M47_ADVERSARIAL_<YOU>.md` and is *append-only* — you may add to it
  but not overwrite earlier sections.
- **Residual proxy** (per `[[feedback_residual_proxy_required]]`): any
  discriminator instrumentation must subtract a known-zero baseline
  (analytic feq, symmetric pair, pre-event value). Logging raw kernel
  output is 7+ decades above the signal.
- **Like-for-like** (per `[[feedback_benchmark_loyalty]]`): if your
  patch test compares two configs, they must be on the same backend,
  precision, and kernel family.
- **Code-path provenance** (per `[[feedback_code_path_provenance]]`):
  any hypothesis citing a kernel must include empirical evidence that
  the kernel is actually exercised by the failing case (instrument or
  static trace; grep is not sufficient).
- **NO Aqua submission**. Patch test runs on CPU or local Metal F32 only.

---

## 1. What works ✓ (Newtonian path, all numbers Aqua A100 F64)

### 1a. K Newtonian via halfwayBB + qwall + β=1.0 placeholder

Driver: `run_viscoelastic_logfv_cyl_coupled_2d` with `β=1.0` (zero
polymer fraction → F_poly=0 on LBM side). Job `21861929.aqua`, 4 cases.

| R  | Cd     | Cd_s   | trace_C_max | NaN | vs rT 132.37 |
|---:|-------:|-------:|------------:|----:|-------------:|
| 30 | 132.076 (matches M41 anchor) | 132.076 | 209 | false | −0.22 % |
| 40 | 132.380 | 132.380 | 227 | false | +0.01 % |
| 50 | 132.558 | 132.558 | 237 | false | +0.14 % |
| 60 | 132.675 | 132.675 | 321 | false | +0.23 % |

**Cd INCREASES monotonically (+0.60 from R=30→60)**. Boundary-layer
resolution improving with R as expected. trace_C grows modestly (209
→ 321, 1.5×). Healthy.

### 1b. K Newtonian via Bouzidi-FL + qwall + β=1.0 placeholder

Same driver, BC swap to `bouzidi_fl_twopass`.

| R  | Cd     | Cd_s   | trace_C_max | NaN | vs rT 132.37 |
|---:|-------:|-------:|------------:|----:|-------------:|
| 30 | 132.637 (matches M41) | 132.637 | 1938 | false | +0.20 % |
| 40 | 133.537 (matches M41) | 133.537 | 3358 | false | +0.88 % |
| 50 | 134.313 | 134.313 | 9610 | false | +1.47 % |
| 60 | 135.436 (matches M41) | 135.436 | **1.4×10⁷** | false | +2.32 % |

**Cd INCREASES monotonically (+2.80 from R=30→60). trace_C_max grows
~7200× (1938 → 14 000 000) over the same R range**, despite β=1.0
(polymer should be dormant, no F_poly back-reaction). Cd_kraken NOT
contaminated (= Cd_s, no polymer drag), so the C-tensor pathology is
internal to the polymer chain only — but the chain is reading something
from the Bouzidi-FL wall side that grows pathologically with R.

### 1c. Channel-class V&V (L1)

- `test/test_viscoelastic.jl` pure shear γ̇·y imposed, evolve_stress_2d!
  50 000 steps → τ_xy/τ_xx match analytic Oldroyd-B steady at rtol=1e-3.
- `test/test_logfv_frozen_channel_cde.jl` Couette + Poiseuille frozen
  velocity, log-FV CDE driver → atol 1e-12 on gradients.
- `bench/viscoelastic_validation/L1_poiseuille_oldb/` Wi sweep
  {0.001..1.0} all PASS via `compare.jl` (RESULTS_20260523.md).

### 1d. M44 Guo half-step fix at R=30 Wi=1 (commit 9fd92ab0)

| Case | pre-fix | post-fix | rT ref | post-fix gap |
|------|--------:|---------:|-------:|-------------:|
| R=30 Wi=1 β=0.59 100k | 111.09 | **118.10** | 120.38 | −1.89 % |
| R=30 Wi=1 β=0.59 400k | n/a    | **118.099** | 120.38 | −1.90 % (= 100k ±0.003) |

R=30 Wi=1 anchor is **temporally rock-solid**. M28-M42 cluster
empirically closed.

---

## 2. What fails ✗ (the two anomalies)

### 2A. R=60 Wi=1 halfwayBB temporal drift (M46-B, job `21862685.aqua`)

Same driver as M44 anchor: viscoelastic Wi=1 β=0.59 muscl_superbee
halfwayBB qwall, varying max_steps.

| Case | Cd | Δ vs 100k | Cd_s | Cd_p | trace_C_max | flow-through |
|------|---:|----------:|-----:|-----:|------------:|-------------:|
| R=30 @ 400k | **118.099** | =100k ±0.003 | 118.891 | 13.99 | ~230 | 2.22 |
| R=60 @ 100k | 113.234 | (baseline) | 113.037 | 13.97 | ~230 | 0.28 |
| R=60 @ 200k | 112.130 | **−1.10** | 111.587 | 14.48 | ~230 | 0.56 |
| R=60 @ 400k | **109.424** | **−3.81** | 108.187 | 15.30 | ~230 | 1.11 |

Net Cd drops because Cd_s drops faster than Cd_p grows. trace_C stays
bounded ~230 (no polymer blow-up). 0 NaN. Drift is **accelerating**:
−1.10 then −2.71 per 200k window.

Naive interpretation: "wake still developing, need more steps". But
under that hypothesis we'd expect Cd to approach a steady state
asymptotically, not to *accelerate* downward. The acceleration is the
hard part.

### 2B. Bouzidi-FL polymer-chain explosion at β=1.0

Section 1b: trace_C_max grows from 1938 (R=30) to 1.4×10⁷ (R=60),
~7200× over a 2× resolution change, while halfwayBB grows only 1.5×.

Per cell, Bouzidi-FL writes more "wall ghost" populations than halfwayBB
(Bouzidi-FL is 2-pass and supports fractional wall distance q_w; halfwayBB
collapses to one pass when q_w=0.5 everywhere). The polymer C-tensor
field is read by the Hermite source kernel from cells adjacent to the
cut links; if Bouzidi-FL writes something into a buffer the polymer
chain interprets as a moment, the pathology would scale with cell-count
along the wall (which scales linearly with R).

Cd is unaffected here because β=1.0 → ν_p=0 → F_poly=0 → the inflated
C never injects back to LBM. **But at β<1 the same mechanism would
inject inflated τ_p into the solvent and could explain the R=60 Wi=1
drift in 2A.**

---

## 3. Production code paths (file:line + mode/backend)

### 3a. The M44 fix — viscoelastic_logfv_2d driver path

Production driver for cylinder Wi=1 sweep:

```
src/drivers/viscoelastic_logfv_2d.jl
  run_viscoelastic_logfv_cyl_coupled_2d
   → CPU/CUDA/Metal dispatch on backend
   → call chain:
     1. collide_guo_field_2d!           (Convention-I: integrates half-step)
     2. logfv_compute_macroscopic_forced_field_2d_kernel!  (LINE 1047-1050: post-fix readout)
     3. log-FV polymer chain:
        - logfv_evolve_psi_2d! (CDE advection on Ψ=log C)
        - logfv_hermite_source_2d! (compute τ_p from C)
        - logfv_polymer_force_bc_aware_2d! (F_poly = div(τ_p), feeds next collide)
```

The M44 fix removed `+F/2` from `_kernel!` at line 1047-1050; the
companion fix in `src/kernels/macroscopic.jl:71-75` (G1, base 2D
non-viscoelastic) is collateral.

### 3b. Wall BC paths

Both BC kernels read/write the LBM `f`-field and produce a polymer-chain
input via the per-cell ρ, ux, uy moments.

- **halfwayBB**: simple bounce-back at the half-link. One pass. Used at
  q_w=0.5 (uniform half-step).
- **Bouzidi-FL twopass**: 2-pass Bouzidi (Filippova-Hänel rescaling) for
  arbitrary q_w ∈ (0, 1). Pass 1 streams the cut populations; pass 2
  applies the wall correction. Used for cylinder cut links.

The pre-computed cut-link table is built by `precompute_q_wall_cylinder`
(`src/kernels/...polymeric_drag_geometry.jl`) — tested at exact closed
form in `test/test_polymeric_drag_geometry.jl`.

Note: per M44 audit, default FVFD path
(`logfv_polymer_force_bc_aware_2d!`) is NOT q_wall-aware unless
`embedded_force=1` is set in the spec. The cylinder sweep used
`embedded_force=0` (M45 γ candidate, NOT confirmed; deferred).

### 3c. Codex G1-G7 inventory (M44 audit)

7 getters identified with potential `+F/2` pattern:

| ID | File | Function | Fixed? |
|----|------|----------|--------|
| G1 | `src/kernels/macroscopic.jl:71-75` | `compute_macroscopic_forced_2d_kernel!` (base 2D) | ✓ M44 |
| G2 | `src/kernels/macroscopic.jl` (3D variant) | `compute_macroscopic_forced_3d_kernel!` | ✗ |
| G3 | `src/kernels/logconformation_fv_2d.jl:1047-1050` | `logfv_compute_macroscopic_forced_field_2d_kernel!` | ✓ M44 |
| G4 | `src/kernels/macroscopic.jl` | `compute_macroscopic_pressure_2d_kernel!` (VOF) | ✗ |
| G5 | `src/kernels/macroscopic.jl` | `compute_macroscopic_phasefield_2d_kernel!` | ✗ |
| G6 | `src/kernels/macroscopic.jl` | `macroscopic_boussinesq` (thermal) | ✗ |
| G7 | `src/kernels/...li_bb_v2.jl` | `WriteMoments` (fused LI-BB Guo-field) | ✗ (downstream of G3) |

Codex M44 audit claimed G2/G4/G5/G6/G7 "not on cylinder path". Worth
re-checking under the lens of "is any of them dispatched in viscoelastic
cylinder F64?".

---

## 4. V&V infrastructure inventory

See `bench/viscoelastic_validation/INVENTORY.md` (230 lines, 2026-05-23)
for full detail. Quick map:

| Level | Path | Status | What |
|-------|------|--------|------|
| L0 | `test/test_logconformation*.jl`, `test_viscoelastic_equations.jl`, `test_fvfd_operators_2d.jl` | ✓ HIGH (atol 1e-12) | Algebraic identities |
| L1 frozen | `test/test_viscoelastic.jl` (pure shear), `test/test_logfv_frozen_channel_cde.jl` (Couette+Poiseuille) | ✓ HIGH | Imposed U, constitutive only |
| L1 inverse | `test/test_viscoelastic_force_accounting.jl` | ✓ HIGH | Prescribed τ_p, check 2nd-moment |
| L1 local patches | `test/test_viscoelastic_patch_tests.jl` (656 LOC) | ✓ CNEBB only | LBM patches at wall |
| L2 body-force Poiseuille | `bench/viscoelastic_logfv/run_poiseuille_polymer_analytical_2d.jl` (Waters & King) | ✓ in bench, NOT promoted to test | Full pipeline, analytic ref |
| 4-roll mill | `bench/viscoelastic_extension_vv/A_extensional/run_4roll_mill_sweep.jl` + `analytic_4roll_mill.jl` | ✓ extensional | Pure extension, analytic |
| L3/L4 cavity, BFS, contraction, cylinder | `bench/viscoelastic_validation/L3*/L4*/STUB.md` | ✗ STUB | not implemented |

**Gaps directly relevant to M47**:

1. **No curved-wall + polymer + frozen-U patch test**. Would directly
   discriminate 2B mechanism (cylinder R=6 closed box, frozen analytic
   Stokes-around-cyl, polymer C=I at t=0, watch trace_C(t) for both
   halfwayBB and Bouzidi-FL).
2. **No rotation-by-π/8 invariance test**. Channel tilted by π/8 with
   frozen tilted Poiseuille — exposes lattice anisotropy bugs in the
   polymer chain.
3. **No Bouzidi-FL + polymer combo unit test**. Patch tests use CNEBB.
4. **No GPU vs CPU bit-comparison smoke** for viscoelastic_logfv driver.

---

## 5. Recent context (compact)

- **M28-M42 cluster** (10+ RED missions on advection limiter, 8 days,
  2026-05-16 → 2026-05-24) **CLOSED by M44**: ported `slbm-paper`
  commit `5ec27044` (Guo half-step +F/2 double count). Adversarial
  Codex+Claude on M43 was CONCORDANT-HIGH on the wrong framing
  (advection band) — see `[[feedback_code_path_provenance]]` and
  `[[feedback_port_sister_branch_fixes]]`.
- **M45 residual audit** (B per-θ + C Codex α/β/γ): found mixed β
  (lattice-distance + TRT scaling) + γ (FVFD non-q_wall-aware stencil),
  no Guo-class double-count. **Partially invalidated** by M46-B finding
  that R=60 snapshot used was non-converged.
- **M46 Newt sweep** (1a + 1b above) + **M46-B temporal probe** (2A)
  reframed the M44 R≥40 "Cd decreases with R" pattern as temporal
  under-sampling, NOT a mesh effect. R=30 still solid.
- **User directive 2026-05-26**: ABSOLUMENT use small tests (micro-
  canaries CPU/Metal <30-60s) before any Aqua submission. See
  `[[feedback_small_tests_first]]`.

---

## 6. Suspect ranking (current best-guess, you should challenge this)

| # | Hypothesis | Pro | Contra |
|---|------------|-----|--------|
| **H1** | Bouzidi-FL writes contaminate cells the polymer C-chain reads from (via ghost or boundary buffer) → C blows up. The contamination scales with wall-cell count = O(R). | trace_C 7200× over R=30→60 at β=1.0; halfwayBB stays bounded under same conditions; mechanism explains both 2A (at β<1 the inflated C injects F_poly back to LBM → solvent drag drops monotonically) and 2B at once. | Cd at β=1.0 is unaffected (Cd_kraken=Cd_s), suggesting C-side pathology stays internal; would need to show that at β=0.59 the back-reaction actually appears. |
| **H2** | R=60 Wi=1 drift is genuine slow polymer transient: wake polymer stretches over many flow-throughs at high R. | Cd_p increases monotonically (13.97→15.30) over time at R=60; wake length matches expectation. | Drift is *accelerating* not decelerating; trace_C is constant ~230 (no buildup); at R=30 wake establishes by ~0.6 flow-through. |
| **H3** | Codex G2/G4/G5 unfixed `+F/2` pattern in another getter fires on viscoelastic cylinder path at high R. | M44 G3 fix gave 78 % closure at R=30 but did not fully close → there could be a residual analogous bug; G2/G4/G5 untested. | M44 Codex audit explicitly checked dispatch table at R=30 Wi=1 and found none of G2-G7 on cylinder path; need to re-check at R=60. |
| **H4** | halfwayBB at q_w=0.5 has its own R-dependent bias at high resolution that grows over time. Newt R=30 shows −0.22 % vs rT; viscoelastic amplifies it because polymer responds to wall-region velocity. | Newt halfwayBB shows residual; viscoelastic shows much bigger drift. | Newt residual is small and grows *less* with R (+0.6 Cd); viscoelastic drift is large and grows *more* with R (−4.85 Cd_s drop). Sign/scale mismatch. |
| **H5** | Domain-size + outlet ZouHe effect: lattice-distance to outlet grows with R; outlet boundary reflects back stress waves that cycle through. | Lattice distance 420→840 LU; ZouHe pressure BC at outlet not flux-conservative. | Should affect Newt too but Newt is monotone-convergent in the right direction. |
| **H6** | TRT relaxation s_plus scaling (1.05 → 0.71 across R=30→60 from M45 C audit) approaches 0.5 and amplifies numerical artefacts at higher R. | Scaling is known; numerical artefacts at low τ are well-documented. | Would affect Newt and viscoelastic similarly; Newt is fine. |

H1 is the working hypothesis; H2 is the null hypothesis (no bug, just
wake transient). Your verdict should either confirm one and design a
discriminator, or propose H7+ I missed.

---

## 7. Required reading (your context window is finite)

In order of importance:

1. **This file** — already loaded.
2. `bench/viscoelastic_audit/M46_NEWT_AND_TCONV_VERDICT.md` (200 lines)
   — full numbers for 1b and 2A.
3. `bench/viscoelastic_audit/M44_GUO_FIX_VERDICT.md` (the closed cluster)
4. `bench/viscoelastic_audit/M44_GUO_AUDIT_CODEX.md` (G1-G7 inventory)
5. **Source files** (read the kernels, not just the names):
   - `src/kernels/logconformation_fv_2d.jl` (the polymer chain — 1500+ LOC,
     read AROUND the M44 fix at 1047-1050 first; then the Hermite source,
     the BC-aware force, the wall handling)
   - `src/fvfd/operators_2d.jl` (FVFD wall stencils, q_wall awareness)
   - `src/drivers/viscoelastic_logfv_2d.jl` (the driver call chain
     `run_viscoelastic_logfv_cyl_coupled_2d`)
   - Search for "bouzidi_fl" or "bouzidi" in `src/kernels/` for the
     2-pass Bouzidi-FL kernel
6. `bench/viscoelastic_validation/INVENTORY.md` — V&V inventory if you
   need to know what tests already exist.

**Do NOT** read: `.engineer_brief_*`, `.orchestrator/memory/*`,
`bench/scratch/`, anything in `/tmp/` or `tmp/`. Stay in `src/`, `test/`,
`bench/viscoelastic_*/`, and `bench/viscoelastic_validation/`.

---

## 8. Deliverable spec

Write **append-only** to `bench/viscoelastic_audit/M47_ADVERSARIAL_<YOU>.md`
where `<YOU>` is `CODEX` or `CLAUDE` (you know which one you are).

**Required sections** (in order):

```md
# M47 — Adversarial verdict (<YOU>)
Date: 2026-05-26  ·  Engine: <Codex|Claude Code>

## 0. WRITE-FIRST sketch (drop within 5 min, before deep dive)
- Top hypothesis (one sentence)
- 1-paragraph mechanism
- Smallest patch test concept (CPU/Metal <60s)

## 1. Top 3 hypotheses ranked
| # | Hypothesis | Mechanism (≤3 sentences) | Discriminating evidence required |
| 1 |
| 2 |
| 3 |

## 2. Discriminator patch test design
- Geometry, grid size, BC, run length (must be CPU or Metal F32, <60s)
- What field/quantity is logged
- What baseline is subtracted (residual proxy)
- Predicted output under H1, H2, H3
- Pass/fail criterion (concrete threshold)
- Estimated LOC + dev cost
- Filepath where it lives (e.g. `bench/viscoelastic_validation/patch_tests/PT_X.jl`)

## 3. Code-path provenance evidence
- For each hypothesis citing a kernel, the empirical entry proof
  (trace, log, or static dispatch evidence — NOT grep alone)
- Files read + key lines

## 4. Skills you actually used
- List skills loaded + which proved useful + which were noise

## 5. Confidence (LOW/MED/HIGH) on your top hypothesis + 1-sentence why
```

**Word budget**: ≤1500 words total. Lists/tables count cheaper than
prose. Concision beats completeness.

---

## 9. Anti-patterns the Boss will reject

- **Confirmation bias**: copying the H1-H6 ranking from §6 without
  independent reasoning. Re-derive from §1-§3 evidence.
- **Hand-waving**: "the BC is probably buggy somewhere" without naming
  a kernel and a mechanism.
- **Patch test that needs Aqua**: if it requires F64 or Wi-sweep, it
  fails the constraint.
- **Patch test that compares apples to oranges**: CPU scalar vs GPU
  custom, two different BCs at two different precisions, etc.
- **Discriminator without baseline subtraction**: see
  `[[feedback_residual_proxy_required]]`.
- **Recommending fix code without a discriminator first**: the patch
  test must localize the bug BEFORE we agree on the fix.

---

## 10. What you may invent if needed

If §6 H1-H6 are all wrong, propose H7. Same format. Pro/contra/evidence.

If the discriminator patch test you design doesn't exist in
`bench/viscoelastic_validation/`, that's fine — design it from scratch.
Specify the filename and the (≤50 LOC) skeleton.

End of synthesis.
