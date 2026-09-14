# Debug plan — Kraken viscoelastic confined Oldroyd-B

Session 2026-04-22, branch `dev-viscoelastic`. Companion to
`bench/viscoelastic_audit/REFERENCES_STUDY.md`.

## Objective

Resolve the ~7% gap between **Kraken's Cd_visco/Cd_Newt = 0.901** at
β=0.5, blockage 0.5, Wi=0.1, R=30 and the **literature expectation
~0.96** (Kumar 2024 arxiv 2403.05904 Table 2 : 126.96/132.09 = 0.961
at De=0.1 ; Liu 2025 arxiv 2508.16997 : 130.36 at Wi=0.1 β=0.59).

Decompose the gap into testable sub-hypotheses and eliminate them in
order of cost. Each milestone has a **kill criterion** : pass → advance,
fail → stop, fix, re-enter.

---

## Hypotheses to eliminate (ranked by prior likelihood)

| H | Description | Status | Milestone |
|---|-------------|--------|-----------|
| H0 | Kraken Cd_Newt = 142.87 is biased +8% (reference wrong, visco ratio artificially low) | Suspected from audit | **M0** |
| H1 | D3Q19 conformation lattice misses moments that D3Q27 captures | Suspected (Liu uses D3Q27) | **M1** |
| H2 | Hermite source prefactor or CNEBB 3D sign/factor error | Plausible, not inspected | **M1** |
| H3 | Polymer-coupling structural bias (Cd_visco systematically wrong) | Plausible | **M2** |
| H4 | Kraken physically correct, literature interpretation wrong | Least likely | **M3** |
| H5 | λ-stiffness at Wi < 0.05 (separate, already partly diagnosed) | Secondary | **M4** |

---

## Milestone 0 — Cd_Newt convergence + normalisation audit

**Effort** : 0.5 day, local (Metal F32 OK).
**Cost** : 1-2 local runs.

### Tasks
1. `run_cylinder_libb_2d` at R ∈ {30, 60, 120} with identical setup :
   B=0.5, Re=1 (ν_total = u_mean·R/Re), Poiseuille inlet, avg_window
   at least 5× (R²/ν_total) = 5·R²/0.6 steps.
2. Log Cd_Newt(R=30), Cd_Newt(R=60), Cd_Newt(R=120).
3. Read the Cd formula in `compute_drag_libb` and cross-check against
   Liu Eq. 64 : `Cd = F_x / (0.5·ρ·U_avg²·D)`. Check u_ref convention.
4. Read `run_cylinder_libb_2d` inlet handling — is the parabolic
   profile fully developed by x=cx=15R ?

### Pass criterion
Cd_Newt(R=120) ∈ [130, 134]  (within 1-2% of canonical ~132
    from Hulsen / Dou-Phan-Thien 1999 / Kumar 2024 Table 2).

### Fail criterion
Cd_Newt(R=120) > 135 or < 128 → **stop here**. The Newtonian
    reference is broken and must be fixed before any visco audit.
    Likely causes :
    (a) Cd normalisation inconsistent with Liu
    (b) Inlet parabolic not fully developed
    (c) Outlet reflection or domain-length effect
    (d) LI-BB drag integration error on curved cylinder

### Reference
- `bench/viscoelastic_audit/step5_3d_diagnostic/5d_cylinder_wi_sweep.jl`
  already runs Cd_Newt at R=30 → extend to R=60, 120.
- Canonical Cd_Newt = 132 : Kumar 2024 Table 2 at De=0.025 = 132.09.
- Also Dou & Phan-Thien 1999 = 131.5 (cited by Kumar 2024).

### Deliverable
`bench/viscoelastic_audit/step6_cdnewt_convergence.jl` + results file.
Decision logged in `AUDIT_SUMMARY.md`.

---

## Milestone 1 — Equation cross-check (Kraken vs Liu 2025)

**Effort** : 1-2 days, local, paper-reading + `grep`.
**Cost** : zero HPC.

### Tasks
1. Download Liu 2025 PDF locally : `arxiv.org/pdf/2508.16997`.
2. Extract Eqs. 25 (Hermite source, full prefactor), 30 (g_eq), 36
   (TRT Hermite source), 38-46 (CNEBB — 2D AND 3D), 62 (Poiseuille
   analytic), 64 (Cd). Annotate lattice choice (D2Q9 / D3Q27).
3. Diff line-by-line against Kraken :
   - `src/multiphysics/viscoelastic.jl` → `apply_polymer_hermite_source_*`
   - `src/multiphysics/viscoelastic.jl` → `feq_conformation_*`
   - `src/multiphysics/viscoelastic.jl` → `apply_cnebb_*` (2D, 3D)
   - `src/multiphysics/viscoelastic.jl` → `reset_conformation_inlet_*`
   - `src/drivers/cylinder_libb.jl` → drag integration
4. **Critical**: verify D3Q19 vs D3Q27. Compute by hand :
   Σ w_i e_iα e_iβ, Σ w_i e_iα e_iβ e_iγ, for Kraken's D3Q19.
   If any 2nd-order or 3rd-order moment required by the conformation
   equation is wrong in D3Q19, **this is the bug**.
5. Write `bench/equations_cross_check.md` (as required by the prompt)
   : each equation side-by-side Kraken / Liu with page/line refs.

### Pass criterion
Every equation matches Liu to the last prefactor AND D3Q19 preserves
    the needed C moments.

### Fail criterion
Any factor-of-2 / sign / missing-moment found → **stop and fix**.
    Re-run M0 test on 2D cylinder, the 2D kernel may need the same
    fix.

### Reference
- Liu 2025 arxiv 2508.16997 (PDF).
- Pimenta & Alves 2017 JNNFM 239:85-104 (UCD source-term form, for
  an independent cross-check if Liu is ambiguous).
- Fattal & Kupferman 2004 (log-conformation, future).

### Deliverable
`bench/equations_cross_check.md`.

---

## Milestone 2 — β sweep vs Kumar 2024

**Effort** : 1 day HPC (Aqua H100 F64).
**Cost** : ~4 runs × 30 min each on H100 = 2 hours GPU.

### Tasks
1. At fixed R=30, Wi=0.1, B=0.5, Re=1, sweep β ∈ {0.3, 0.5, 0.7, 0.9}.
2. For each β run both viscoelastic AND Newtonian (ν=ν_total) → ratio
   per β.
3. Compare trend Kraken vs Kumar 2024 Figure (need the figure — get
   the PDF `arxiv.org/pdf/2403.05904`).
4. Also plot ratio vs β from Kraken's own 2D audit data if β-sweep
   wasn't done, this is a new sweep.

### Pass criterion
Kraken's ratio-vs-β slope matches Kumar's within 2% AND constant
    offset is stable (pure calibration). → H3 eliminated.

### Fail criterion
Slope disagreement > 5% → **structural polymer coupling bias**.
    The τ_p → f coupling or the conformation → τ_p mapping is
    wrong. Re-enter M1 with this as a lead.

### Reference
- Kumar et al. 2024 arxiv 2403.05904 — full β sweep at B=0.5 Re=0.01.
- Note : Kumar uses Re=0.01 (creeping), Kraken uses Re=1. Inertia at
  Re=1 shifts the Newtonian baseline by ~1% ; the visco/Newt ratio
  should be less affected.

### Deliverable
`bench/viscoelastic_audit/step7_beta_sweep.jl` + Aqua job + results
plot.

---

## Milestone 3 — Third-party cross-check via rheoTool

**Effort** : 2-3 days if M0-M2 haven't closed the gap.
**Cost** : local install + 1 day for setup + 1 run.

### Tasks
1. Clone rheoTool :
   ```bash
   git clone --depth=1 https://github.com/fppimenta/rheoTool /tmp/rheotool
   ```
2. Confirm whether a confined cylinder tutorial exists :
   ```bash
   find /tmp/rheotool -type d -name '*cylinder*' -o -name '*Cylinder*'
   ```
3. If yes : adapt mesh to B=0.5, run with Oldroyd-B at β=0.5 Wi=0.1
   Re=1. Extract Cd.
4. If no : find the Alves, Oliveira, Pinho 2001 benchmark mesh (may
   be shared as supplementary material or reproducible from the paper
   description).
5. Run `foam-extend` Oldroyd-B tutorial as fallback (confirmed path :
   `openfoam-extend-foam-extend-3.1/tutorials/viscoelastic/viscoelasticFluidFoam/Oldroyd-B`).

### Pass criterion
rheoTool Cd_visco / Cd_Newt at β=0.5 B=0.5 Wi=0.1 Re=1 ∈ [0.88, 0.91]
    → **Kraken is validated**. The "0.89 vs 0.96 literature" gap is
    literature interpretation, not Kraken bug. Kumar 2024 is either
    at different β or Re.

### Fail criterion
rheoTool gives ratio ∈ [0.94, 0.98] → **Kraken bug confirmed**.
    Return to M1/M2 findings as bug lead.

### Reference
- rheoTool : https://github.com/fppimenta/rheoTool
- Pimenta & Alves 2017 JNNFM 239:85-104 (rheoTool paper).
- Alves, Oliveira, Pinho 2001 JNNFM 97:207 (confined cylinder
  benchmark — the reference mesh).

### Deliverable
`bench/viscoelastic_audit/step8_rheotool_crosscheck/` directory with
OpenFOAM case files and results summary.

---

## Milestone 4 — λ-stiffness diagnostic (secondary)

**Effort** : 1 day HPC.
**Cost** : 3-4 runs.

Independently of the M0-M3 chain, clarify the "+10% at Wi=0.001"
anomaly (AUDIT_SUMMARY §step 5 corrections).

### Tasks
1. Stiffness test per the NEXT_SESSION_PROMPT §Q2 :
   at fixed Wi=0.1, scan R ∈ {15, 30, 60, 120}. λ = Wi·R/u_mean scales
   with R. If bias is λ-driven (numerical stiffness), bias decreases
   as R grows. If bias changes at fixed λ(R=30)=150 → discretisation,
   not stiffness.
2. Complementary : at fixed R=30, scan Wi ∈ {0.001, 0.003, 0.01,
   0.03, 0.1}. Plot ratio vs Wi. If +10% at Wi=0.001 is a polymer-
   relaxation-time numerical artefact, it should die for λ > a few
   cells (Wi ≳ 0.02 at R=30).

### Pass criterion
Bias at Wi=0.001 decreases monotonically with R. → λ-stiffness
    confirmed. Note as known numerical limitation, restrict
    publishable regime to Wi ≥ 0.05.

### Fail criterion
Bias persists at large R or has non-monotone behaviour → something
    else is going on, re-diagnose.

### Deliverable
`bench/viscoelastic_audit/step9_stiffness.jl`.

---

## Milestone 5 — Log-conformation 3D (stretch, post-paper)

Only if Wi > 0.5 is a target publication case. The 2D log-conf is
already validated. The 3D version requires 3×3 symmetric eigen-
decomposition (Cardano + Jacobi). `LogConfOldroydB` explicitly
rejected by `run_conformation_sphere_libb_3d` today.

### Tasks
1. Implement `logconf_decompose_3x3_kernel!` (GPU-friendly Jacobi).
2. Port the Fattal-Kupferman 2004 3D equations.
3. Validate on sphere Wi=1 with small radius.

### Reference
- Fattal, Kupferman 2004 JNNFM 123:281-285.
- Hulsen, Fattal, Kupferman 2005.
- Afonso, Alves, Pinho 2011.

---

## Decision gate (end of M0–M3)

After M3, document the answer to :

> **Is Kraken publication-ready for confined Oldroyd-B cylinder at
> β ∈ [0.3, 0.9], Wi ∈ [0.05, 0.3], Re=1, blockage 0.5 ?**

Three possible outcomes :

1. **Yes, validated** — Kraken ratio 0.89 matches rheoTool at this
   setup. Publish. Literature 0.96 was misinterpreted (different β or
   Re convention). Update `REFERENCES.md` with the rheoTool-based
   reference.

2. **Yes, with caveat** — Kraken matches Kumar 2024 trend vs β after
   fixing the Newtonian reference normalisation (M0). Publishable
   with a correction factor documented.

3. **No, bug present** — M1-M2 identified a specific equation or
   coupling bug. Fix, re-run M0-M2, re-evaluate. Depending on severity,
   either scope down publication to 2D (where audit found it clean)
   or delay 3D sphere publication.

---

## Timeline & resources (optimistic)

| Day | Task | Compute |
|-----|------|---------|
| 0 | M0 : Cd_Newt convergence | local Metal F32, 1 h |
| 1 | M1 : equations cross-check (reading + code diff) | zero |
| 2 | M1 : finish + write `bench/equations_cross_check.md` | zero |
| 3 | M2 : β sweep, submit Aqua jobs | Aqua H100, 2 h |
| 4 | M2 : analyse vs Kumar 2024 | local |
| 5-7 | M3 : rheoTool setup + run (if needed) | local (workstation OpenFOAM) |
| 8 | Gate decision + AUDIT_SUMMARY.md update | — |
| 9-10 | M4 : stiffness diagnostic (parallel to M3) | Aqua |

Total : ~2 weeks to a firm decision, of which ~1 day GPU.

---

## What NOT to do (hard-earned from previous sessions)

- **Do not run a new HPC sweep before M0 is done**. The Cd_Newt
  reference is the denominator of every ratio ; a biased denominator
  poisons every subsequent measurement.
- **Do not speculate a cause without a test that distinguishes it**
  (4 retractions in the previous session came from this).
- **Do not compare to Lunsmann 1993** for the confined sphere.
  Unbounded vs confined is a different problem.
- **Do not assume Kraken is correct because G=0 gives ratio=1.0000
  exact**. That test eliminates gross coupling bugs but says nothing
  about the polymer stress coefficient structure.
- **Do not use Float32 (Metal local) for precision-sensitive
  convergence runs**. Known 70% shift vs F64 Aqua. All benchmark runs
  from here must be F64 on Aqua.

---

## Risk register

| Risk | Mitigation |
|------|-----------|
| Liu 2025 PDF Eqs. 44-46 not extractable | Use rheoTool (Pimenta-Alves 2017) as alternate canonical ref |
| rheoTool cylinder tutorial does not exist | Adapt Alves 2001 mesh manually, or fall back to foam-extend Oldroyd-B tutorial |
| Aqua queue busy | Run M2 on Metal F32 as lower-precision preview, confirm on Aqua later |
| Bug turns out to be in 3D streaming / CNEBB corners | Scope down publication to 2D cylinder (fully audited) |

---

## Sources

All citations collected in `bench/viscoelastic_audit/REFERENCES_STUDY.md`.
Primary numerical references :
- Kumar et al. 2024 arxiv 2403.05904 — β=0.1..0.9 B=0.5 Re=0.01.
- Liu et al. 2025 arxiv 2508.16997 — β=0.59 B=0.5 Re=1.
- Alves, Oliveira, Pinho 2001 JNNFM 97:207 — FVM B=0.5 Oldroyd-B.
- Fan, Kahrilas, Tanner 1999 — FEM B=0.5 Oldroyd-B.
- Dou & Phan-Thien 1999 — Cd_Newt = 131.5 reference.
- Pimenta & Alves 2017 JNNFM 239:85 — rheoTool paper.
