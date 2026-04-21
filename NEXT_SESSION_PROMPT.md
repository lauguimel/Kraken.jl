# Next session prompt — Kraken.jl viscoelastic (dev-viscoelastic)

Copy-paste everything below to start a fresh session.

---

Continue work on `dev-viscoelastic` branch of Kraken.jl
(worktree `~/Documents/Recherche/Kraken.jl-viscoelastic`).

## Where we stand (2026-04-22)

A full audit ran last session and **invalidated most prior claims**.
Read `AUDIT_SUMMARY.md` at repo root FIRST — it is the single source
of truth. Key facts :

**Kraken is self-consistent** across 2D cylinder (β=0.5 and β=0.59)
and 3D sphere (β=0.5), and across Float64 (Aqua CUDA) and Float32
(Metal local). At Wi=0.1 they all give Cd_visco/Cd_Newt ≈ **0.89–0.90**.

| Benchmark | Wi=0.001 | Wi=0.01 | Wi=0.1 |
|-----------|----------|---------|--------|
| 2D cyl β=0.5 R=30 F64 | 1.103 | 1.076 | 0.901 |
| 2D cyl β=0.5 R=30 F32 | 1.173 | 1.067 | 0.847 |
| 3D sphère β=0.5 R=16 F64 | — | 1.081 | 0.892 |

Ruled out this session :
- Not a 3D-specific bug (2D shows same pattern).
- Not a Float32 precision issue (F64 Aqua reproduces).
- Not a Hermite-source or UCD formula error (verified by hand 2D vs 3D).
- Not a coupling-loop bug (G=0 gives ratio=1.000 exact).
- Not CNEBB causing the deficit (removing CNEBB makes it worse).
- Not HWNP (I incorrectly claimed HWNP at Wi=0.03 ; retracted).

**Two real findings (not bugs)** :
1. λ < ~50 lattice units → TRT-LBM conformation is stiff → numerical
   artifact giving +5–17% bias (visible at Wi=0.001 with λ=1.5). NOT
   physical, but not a code bug either.
2. Ratio ≈ 0.89 at Wi=0.1 β=0.5 blockage 0.5 is reproduced across
   2D/3D/Float32/Float64. Kraken is self-consistent. Whether this is
   the CORRECT physical answer is the open question.

## Core question for this session

**Is ratio 0.89 (drag reduction of 10%) at Wi=0.1 β=0.5 blockage 0.5
the physically correct answer, or is there a bug we haven't found ?**

The prior "expected enhancement from Lunsmann 1993" comparison was a
category error : Lunsmann is for UNBOUNDED sphere, but the Kraken
benchmark is heavily confined (blockage 0.5). For confined geometry,
drag reduction by Oldroyd-B can be physically correct.

## Suggested actions

1. **Literature dive** for confined cylinder/sphere Oldroyd-B Cd :
   - Alves, Oliveira, Pinho, J. Non-Newtonian Fluid Mech. (multiple
     papers on 2D cylinder between parallel plates, especially the
     2001 paper cited often : "The flow of viscoelastic fluids past
     a cylinder: finite-volume high-resolution methods")
   - Owens & Phillips book "Computational Rheology" (Imperial College
     Press 2002) ch. 7 on confined geometries
   - Fan, Tanner, Phan-Thien papers on ducted sphere (Phan-Thien 1984,
     Zheng et al.)
   - **Liu et al. 2025 (arxiv 2508.16997)** itself — Table 3 reports
     Cd at Wi=0.1,0.5,1.0 for β=0.59 R=30 = {130.36, 126.31, 151.31}.
     Note the Wi=0.5 value (126.31) is LOWER than Wi=0.1 (130.36),
     confirming confined geometry can give non-monotone Cd(Wi) with a
     dip near Wi~0.5. Consistent with Kraken's pattern.

   Find one reference that gives ratio Cd_visco/Cd_Newt (or absolute Cd
   plus a Newtonian reference) at β=0.5 blockage 0.5 so we can compare
   directly to 128.70/142.87=0.901. If it matches → Kraken validated.

2. **Verify λ-stiffness explanation** by running 2D cylinder at larger
   R with the same Wi to check if the +10% artifact at Wi=0.001
   disappears when λ grows. E.g. R=120 Wi=0.001 gives λ=6 (still
   stiff). R=1000 Wi=0.001 gives λ=50 (should be clean). Useful
   sanity check but low priority.

3. **Alternative sphere 3D reference** : Alves 2003 did "4:1:1
   contraction" with a plug, which is a different geometry. For a
   sphere in a CUBIC duct with blockage 0.5, Owens-Phillips book
   section 7.4 has tables. Phan-Thien has Cd vs Wi for PTT at β=0.5.

4. **Once reference is found**, report Kraken vs reference at R=16,32
   and settle the validity question.

## What NOT to do

- Do not propose "log-conformation 3D" as THE fix until we confirm
  there is a bug to fix. The sphere data at Wi=0.1 is CONSISTENT with
  Kraken's 2D result and with Liu's own Wi-sweep pattern. It may not
  be wrong.
- Do not invoke HWNP as an explanation unless Wi > ~0.5 and the scheme
  actually blows up (NaN / oscillations). The Wi=0.1 regime is NOT
  HWNP.
- Do not reverse-engineer a narrative to explain away a result before
  the raw data clearly demands it. Last session I did this twice and
  had to retract both times.

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
