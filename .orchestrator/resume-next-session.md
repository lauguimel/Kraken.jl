# RESUME BRIEF — KRK-SHIP-002 remediation (next session)

_Written end of 2026-06-01 session (Boss context full). Read this FIRST, then boss.md latest + mandate §4 ADRs. Pedagogical user updates required ([[feedback_pedagogical_checkpoints]] — explain in words, no acronym dumps)._

## Where we are
- **Remediation branch = `dev/v0.3-campaign`** (worktree `Kraken.jl-v0.3-campaign`). RC label `dev/v0.2-multiphysics` is a snapshot at `3d49db834` — it will be RE-CUT from the remediated branch at release time.
- **Docs phase M9–M14 DONE** (tri-track docs shipped; commits e35288cff..3d49db834). VE log-FV solver assembled + precompiles on isolated `feat/ve-logfv-on-v03` (`f290ca17e`) — **fold into the release is USER-GATED** (9000 LOC, needs a functional drag-smoke + full-suite-with-VE green first).
- **Engineering-hygiene remediation STARTED.** 8 oversized files in the AMR / mesh-refinement family split into ≤700-line files, behavior-preserving. Commits: `73dfac420`(SPLIT-1), `3819bead1`, `bc70b61ea`, `4c00416b1`, `7196821e0`, `b6bcccac9`, `b95b83382`, `321c00af2`(SPLIT-8). **Full test suite GREEN after SPLIT-8: 34598 pass / 7 pre-existing fails / 0 errored / 0 new.**

## USER SCOPE DECISION (2026-06-01) — READ CAREFULLY
- **OUT OF SCOPE for v0.2 (deferred to a future release):** `src/kernels/vof_2d.jl` (807, two-phase volume-of-fluid), `src/curvilinear/slbm.jl` (1036, curvilinear/body-fitted), `src/drivers/axisymmetric.jl` (1498, axisymmetric). **Do NOT split these now** — their physics (multiphase / curvilinear / axisymmetric) is not part of the v0.2 MVP. They may stay oversized OR be excluded from the release branch; that's a release-cut decision, not a remediation task now.
- **IN SCOPE — the remaining oversized files to fix for v0.2:** `src/io/kraken_parser.jl` (2314), `src/simulation_runner.jl` (2714), `src/kernels/boundary_rebuild.jl` (1240).

## NEXT-SESSION PLAN — "delegation maximale" (user's words)
Two src tracks (sequential on dev/v0.3-campaign — they both edit Kraken.jl includes) + parallel non-src tracks.

### Src Track A — `.krk` reader: split `kraken_parser.jl` (2314) + add symbolic options
- Split into `src/io/krk/` (create the folder). HIGH-risk: it's the front door for every `.krk` file.
- ALSO add the missing feature: `Define`/`Physics` blocks are numeric-only today → enable SYMBOLIC options (wall-treatment, advection-scheme, formulation, collision) from `.krk`. This was flagged in the audit as a real gap (formerly "K1").
- Because this ADDS a feature (not a pure relocation), gate on the FULL test suite (`julia --project=<wt> <wt>/test/runtests.jl` → 34598/7/0, plus a new test for the symbolic options), NOT the fast load-check.
- Consider loading `kraken-trace` (runtime code-path provenance) before forming hypotheses about the parser flow.

### Src Track B — simulation pilot: split `simulation_runner.jl` (2714) + `boundary_rebuild.jl` (1240)
- Create `src/bc/` (boundary-conditions folder); relocate the boundary-handling + inflow/outflow + the bc kernels there; carve out IO emitters and per-driver `_run_*` functions. HIGHEST-risk file (central dispatch hub).
- Pure-relocation parts → load-check OK; any non-byte-identical extraction → FULL test suite.
- May want USER INPUT on folder names / how finely to carve. Load `kraken-trace` + `kraken-codebase-map`.

### Parallel non-src tracks (start these too — max delegation; they don't touch src on this branch)
1. **Efficiency benchmark** (formerly E1): on the Aqua cluster, measure MLUPS for a cavity / Taylor-Green BGK D2Q9 case at N=1024/2048, CUDA Float64, and report BOTH (a) the % of peak memory bandwidth achieved (roofline) AND (b) a comparison vs a published single-GPU LBM number (waLBerla/Palabos/OpenLB). Needs `pbs` + `hpc-watch`. Gotcha: use `gpu_mem=60gb` in the PBS select (40GB A100 nodes OOM the big F64 case); `set -o pipefail` + `exit ${PIPESTATUS[0]}` so a Julia crash fails the job.
2. **Reference-solver backfill** (formerly P1, literal): create `benchmarks/results/rheotool_compare/<module>/` with REAL RheoTool/OpenFOAM runs per in-scope module (Newtonian, thermal, viscoelastic) — CSVs of both solvers at matching probes + L1/L2/Linf error norms + a comparison plot. Use `sim-rheotool`/`sim-openfoam`. Heavy; user chose this over the lighter "document accepted substitutes" option.
3. **Docs showing-off**: rework the human docs to be impressive — remove the bibliography references from the index page, use the nice icon at large size, regenerate prettier GIFs, improve every PNG output and tutorial, and refresh the LLM/agent docs. Likely uses the `sci-style` skill + may need GPU runs to regenerate showcase GIFs.

## Proven SPLIT recipe (reuse for any pure relocation)
Codex does a STATIC byte-identical relocation (the Codex `--ephemeral` sandbox has NO working julia — lockfile bug), self-validating that the concatenation of the new files is byte-identical to the original + symbol counts match. THEN the Boss gates from the normal-env Bash: `julia --project=<wt> -e 'using Kraken; println("LOAD_OK")'` (~8s, confirms include order) chained to `git add src/<dir> src/Kraken.jl && git commit`. A FULL `runtests.jl`-direct checkpoint every ~6-8 splits. Brief template: `/tmp/s_amr_split8_brief.md` (swap the target filename + concern hint).

## Hard gotchas
- Run the suite via `julia --project=<wt> <wt>/test/runtests.jl` DIRECT — NOT `Pkg.test()` (its sandbox lacks KernelAbstractions → false "not found" + BoundsError cascade).
- Fresh worktree needs `Pkg.instantiate()` first.
- NEVER push (public repo). The release goes through a SCRUBBED `release/v0.2` (confidentiality audit of the 200+ commits, `.orchestrator/` EXCLUDED, no AI mentions) at the very end, user-confirmed. NEVER commit `.orchestrator/` to a public branch.
- Decisions LOCKED (mandate §4): full-ideal tier, BOTH efficiency references (roofline + published), axisymmetric scope-cut, literal RheoTool backfill.
