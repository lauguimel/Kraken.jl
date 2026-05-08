# Branch Contract

## Branch

- Name: `dev-viscoelastic`
- Objective: build a robust GPU-oriented viscoelastic backend for Kraken, starting in 2D and porting to 3D only after the 2D ladder is green.
- Non-goals: do not use cylinder drag or Liu Wi=1 as the primary debugging target; do not tune nonphysical parameters to fit one macro benchmark.

## Architecture

- Production path: LBM solvent coupled to a cell-centered log-conformation FV/FD polymer CDE solver.
- Benchmark-only path: direct-conformation population LBM, including Liu Eq26 variants.
- Diagnostic path: direct-C regularized/TRT, log-conformation population LBM, and frozen-flow CDE probes.

## Allowed Edit Zones

- `docs/design/viscoelastic_logfv_gpu_design.md`
- `docs/agent/branch_contract.md`
- `src/kernels/logconformation_fv_2d.jl`
- `src/kernels/polymer_stencils_2d.jl`
- `src/drivers/viscoelastic_logfv_2d.jl`
- `test/test_viscoelastic_logfv_patch_ladder.jl`
- `test/test_viscoelastic_logfv_gpu_smoke.jl`
- `bench/viscoelastic_logfv/`

Keep edits to legacy Liu/direct-C files narrowly scoped and only when a lower-level canary proves the need.

## Forbidden Shortcuts

- No macro benchmark debugging before lower analytical and patch canaries.
- No long CPU production simulations.
- No dynamic least-squares solve, allocation, host callback, or runtime dispatch-heavy logic inside GPU kernels.
- No silent SPD projection or eigenvalue floor in production code.
- No fitting of `magic`, BSD fraction, stress caps, or other nonphysical controls to a single benchmark.
- No staging unrelated dirty files.
- No push without explicit confirmation.

## Validation Ladder

1. M0 algebra: 2x2 symmetric `exp`, `log`, SPD, repeated-eigenvalue behavior.
2. M1 pure Oldroyd-B relaxation with exact analytical solution.
3. M2 homogeneous simple shear/stretching local source.
4. M3 advection of constant and affine `Psi`.
5. M4 Poiseuille analytical conformation and Newtonian limit.
6. M5 Couette analytical conformation and Newtonian limit.
7. M6 frozen-flow square obstacle CDE.
8. M7 coupled square obstacle, low Wi and beta sweep.
9. M8 BFS coupled Newtonian and viscoelastic checks.
10. M9 Liu cylinder Wi 0.1 and 0.5.
11. M10 Liu/high-Wi cylinder only after M0-M9.

Main early validation command:

```bash
julia --project=. test/test_viscoelastic_logfv_patch_ladder.jl
```

Legacy audit ladder:

```bash
julia --project=. test/test_viscoelastic_equation_patch_ladder.jl
```

## GPU/HPC Rules

- Use local CPU/Metal only for small canaries and smoke tests.
- Use A100/H100 for long Float64 validation.
- Use SoA arrays and fixed compact stencils.
- Precompute near-wall/cut-cell coefficients CPU-side.
- Report memory footprint and cells/sec before making performance claims.

## Commit Rules

- Commit after a focused green canary, a frozen diagnostic canary, or a durable design/contract update.
- Stage only intentional files.
- Use factual commit messages.
- Never mention AI tooling in commits.

## Known Traps

- A better `Cd` does not prove the polymer operator is correct.
- Log-conformation preserves SPD algebraically but can still blow up through advection/source/coupling errors.
- Low beta requires momentum stabilization such as BSD/iBSD/DEVSS-like terms; log-conformation alone is not enough.
- The existing 3D conformation path is not a robust high-Wi production backend.
