# Next session prompt — Kraken.jl viscoelastic (dev-viscoelastic)

Copy-paste everything below to start a fresh session.

---

Continue work on `dev-viscoelastic` branch of Kraken.jl
(worktree `~/Documents/Recherche/Kraken.jl-viscoelastic`).

## Where we stand (2026-04-22)

A full audit ran last session. Prior claims (cylinder 0.32% error,
3D sphere 10% deficit diagnosed as HWNP or as 3D-specific bug) were
retracted — **read `AUDIT_SUMMARY.md` for the full record of what was
ruled out and the flip-flops I made**.

**What is factually established by data** :

1. Kraken gives **the same Cd_visco/Cd_Newt ratio ~0.89 at Wi=0.1**
   across 2D cylinder (β=0.5 and β=0.59) and 3D sphere (β=0.5), and
   across Float64 (Aqua) and Float32 (Metal local).

   | Benchmark | Wi=0.001 | Wi=0.01 | Wi=0.1 |
   |-----------|----------|---------|--------|
   | 2D cyl β=0.5 R=30 F64 | 1.103 | 1.076 | 0.901 |
   | 2D cyl β=0.5 R=30 F32 | 1.173 | 1.067 | 0.847 |
   | 3D sphère β=0.5 R=16 F64 | — | 1.081 | 0.892 |

2. **Bug hypotheses ruled out by targeted tests** :
   - Not Float32 (F64 reproduces).
   - Not 3D-specific (2D shows same pattern at β=0.5).
   - Not a Hermite-source or UCD formula error (verified by hand,
     2D vs 3D kernels bit-consistent).
   - Not a coupling-loop bug (G=0 gives ratio=1.000 exact).
   - Not CNEBB causing the deficit (removing CNEBB makes it worse).

**What is NOT established** (and what I pretended to establish last
session — explicit self-retractions) :

- The **cause** of the +10% at Wi=0.001 is unknown. I speculated
  "λ-stiffness" but did NOT run a test that isolates λ from Wi. A
  real stiffness signature would be monotone bias decreasing with
  larger λ; the data shows bias flipping sign between Wi=0.01 and
  Wi=0.1, which is NOT the stiffness pattern. Do not trust that
  narrative.
- Whether ratio 0.89 at Wi=0.1 is **physically correct** is unknown.
  I claimed "probably physical for confined geometry" but I did not
  find any reference to support this.
- Bug-free status of large chunks of the code I did NOT inspect :
  the 3D streaming on g, the 3D CNEBB implementation details, the
  feq_3d used for conformation equilibrium (does it use the right
  moments?), the inlet/outlet reset in 3D. Absence of evidence is
  not evidence of absence.

## Core question for this session

**Two open questions, both requiring actual tests — not speculation** :

### Q1 : Is ratio 0.89 at Wi=0.1 β=0.5 blockage 0.5 physically correct ?

Last session's comparison to Lunsmann 1993 was a category error
(Lunsmann is unbounded, Kraken is confined). Need a PROPER reference
for confined Oldroyd-B sphere. Candidates to check :

- Alves, Oliveira, Pinho 2001 "The flow of viscoelastic fluids past a
  cylinder" — confined cylinder Oldroyd-B Cd vs Wi. Should directly
  compare.
- Owens & Phillips 2002 book, ch. 7 — tables for confined geometries.
- Phan-Thien 1984, Zheng et al. for ducted sphere.
- Liu 2025 arxiv 2508.16997 Table 3 at β=0.59 R=30 gives Cd at
  Wi=0.1,0.5,1.0 = {130.36, 126.31, 151.31} — note the non-monotone
  shape (dips at mid-Wi, rises high-Wi). Kraken's β=0.5 result
  qualitatively matches the low-Wi dip.

**Concrete action** : find one reference that gives
Cd_visco/Cd_Newt at β=0.5 blockage 0.5 Wi=0.1, compare to Kraken's
0.901. If it matches within ~5% → Kraken validated for this benchmark.
If off by >10% → real issue, dig into code.

### Q2 : What actually causes the +10% bias at Wi=0.001 ?

My "λ-stiffness" speculation is NOT consistent with the data (sign
flip between Wi=0.01 and Wi=0.1 is not a stiffness signature). The
real cause is unknown. Targeted diagnostics to try :

- **Stiffness test** : at fixed Wi=0.1 (known "regime OK"), scan R ∈
  {15, 30, 60, 120} to vary λ (= Wi·R/u_mean). If bias is λ-driven,
  larger R should keep λ > threshold and bias should be stable. If
  bias changes with R at fixed Wi, it's discretization, not λ.
- **Artifact-decomposition** : at Wi=0.001 where +10% appears, dump
  Cxy(x,y) and C_xx(x,y) profiles. Compare to analytic Poiseuille
  Oldroyd-B. If the fields are vastly off from analytic at places
  where τ_p is small (far from cylinder), something WAY upstream is
  wrong, not just the drag integration.
- **β sweep at Wi=0.001** : if +10% bias scales with β (more polymer
  contribution → larger bias), it's a polymer-stress bug. If it's
  insensitive to β, it's flow-side.

### Code paths NOT yet inspected (bug hypotheses still open)

- **3D streaming for g** (`stream_3d!`) — how does it handle
  boundaries, does it see is_solid, interaction with CNEBB.
- **3D CNEBB implementation details** — specifically how the 19-
  population version groups the wall-adjacent populations.
- **`feq_3d` used for conformation equilibrium** — is the Mach
  expansion correct? g_q^eq in 3D should match C tensor moments; any
  mismatch would give a systematic drift.
- **`reset_conformation_inlet_3d!` / `reset_conformation_outlet_3d!`**

Any of these could contain a factor-of-2 or sign error that produces
the systematic bias without affecting G=0 tests (where τ_p never
enters f).

### Equations: Kraken vs canonical (NOT CROSS-CHECKED LAST SESSION)

Last session I verified 2D-vs-3D internal consistency by eye. I did
NOT compare against canonical references. This is a hand-wave and
should be the FIRST thing done now.

**Canonical references to check term-by-term** :

- **Liu et al. arxiv 2508.16997 Eq. 25** (Hermite source, the exact
  prefactor with / without (1−s/2) depending on TRT vs BGK)
- **Liu Eqs. 38-39** (CNEBB at curved walls — 2D then 3D port)
- **Pimenta & Alves 2017** J. Non-Newt. Fluid Mech. — the rheoTool
  paper, which documents the UCD source terms and boundary conditions
  for confined cylinder Oldroyd-B at canonical accuracy.
- **Fattal & Kupferman 2004** — log-conformation formulation (for
  future 3D log-conf extension).

**Open-source codes to transpose / compare against** :

- **rheoTool** (https://github.com/fppimenta/rheoTool) — OpenFOAM
  Oldroyd-B solver by the Alves group. Gold standard for confined
  viscoelastic benchmarks. Implements Pimenta-Alves 2017 equations
  directly. Good path : run rheoTool's cylinder benchmark at β=0.5
  blockage 0.5 Wi=0.1 → get a canonical Cd_visco/Cd_Newt ratio →
  compare to Kraken's 0.901. If rheoTool gives 0.9, Kraken is
  validated. If rheoTool gives something else, Kraken bug is real.
- **Palabos viscoelastic plugin** (C++ LBM, less canonical).
- **Liu et al. supplementary material** — check arxiv 2508.16997 for
  linked GitHub or supplement. If they released code, transposing
  to Julia for bit-for-bit comparison is the cleanest path.

Concrete test: write `bench/equations_cross_check.md` that lists each
equation used in Kraken viscoelastic 3D side-by-side with its citation
in Liu 2025 or Pimenta-Alves 2017. Any mismatch is a lead.

## What NOT to do

- Do not speculate a cause without a test that distinguishes it. The
  previous session had 4 retractions because I kept narrating instead
  of testing.
- Do not invoke HWNP at Wi ≤ 0.1 — that regime is never HWNP.
- Do not assume Kraken is correct OR incorrect without a reference.
  Both possibilities are open.
- Do not assume the "+10% at Wi=0.001" is the same phenomenon as the
  "-10% at Wi=0.1". They may have entirely different causes.

## Session artifacts (read before acting)

- `AUDIT_SUMMARY.md` — single source of truth
- `REFERENCES.md` — Liu Table 3 canonical values
- `bench/viscoelastic_audit/` — 5 modular diagnostic scripts + results
  - `step1_bgk_guo.jl` — canal baseline (bulk order 1.5, wall bias)
  - `step1b_profile_dump.jl` — shows wall bias is localized
  - `step1c_analytic_wall.jl` — proves scheme is correct with good BC
  - `step2_trt_hermite.jl` — TRT ≡ BGK on flat wall
  - `step5_3d_diagnostic/` — 5a/b/c/d sphère + 2D cyl
- `~/.claude/projects/.../memory/project_viscoelastic_audit.md` —
  ongoing notes

## Aqua jobs left queued

Check with `ssh maitreje@aqua.qut.edu.au qstat -u maitreje` — the
sphere + cylinder R-convergence jobs may still have data in
`~/Kraken.jl-dev-viscoelastic/results/` from last session.

## HPC workflow

```bash
rsync -az --exclude='.git' --exclude='results' --exclude='Manifest.toml' \
  ./ maitreje@aqua.qut.edu.au:~/Kraken.jl-dev-viscoelastic/
ssh maitreje@aqua.qut.edu.au 'cd ~/Kraken.jl-dev-viscoelastic && qsub hpc/XXX.pbs'
```

Standard PBS: ncpus=8, gpu_id=H100, mem=32GB, walltime 4-12h.
GPU queue was idle after wp_mesh_6 jobs cleared.

## Key gotchas (persist from prior session)

1. **3D sphere R=48 needs >80 GB** — don't try it on H100. Stick to
   R≤32 or implement memory-efficient variant.
2. **Julia 1.12 soft-scope** — wrap HPC-script `for` loops assigning
   `f, f_buf = f_buf, f` in `let … end`.
3. **Inlet C profile** : for `:uniform` in 3D, C_inlet = I (identity).
   For `:parabolic_y`, full analytic profile. Check match in
   `run_conformation_sphere_libb_3d`.
4. **compute_drag_libb_3d** IS halfway-BB single-cell (NOT Mei-
   Bouzidi), despite what `VISCOELASTIC_FINDINGS §1` claimed.
5. **Prior sphere tests passed 13/13 with 50% tolerance** — that
   tolerance masked nothing because there is probably nothing to mask,
   just means the tests aren't tight enough to be useful.

---

End of prompt. Good luck. Focus on the physical reference question
before anything else.
