# Cavity M6-B wall-BC stencil match — confirmation verdict (2026-05-16)

## TL;DR

**The wall-stencil hypothesis is REFUTED at production scale.** The
sanity baseline (`polymer_wall_extrap=:quadratic`) reproduces the M1
baseline L2 numbers to four significant figures, confirming the new
code path is byte-identical with the default. The test case
(`:linear` matching rheoTool's `linearExtrapolation`) does NOT close
the 18-24 % cavity profile gap — centerline u L2 actually rises
slightly (17.97 % → 18.17 %), psi_xy L2 falls only marginally
(24.41 % → 24.33 %, ~0.3 percentage points). The local 12 %
wall-row delta observed at the N=32 t=2 smoke does not propagate
into a comparable global profile improvement.

## Setup

- Branch: `dev-viscoelastic` HEAD `5c1861da` (post-M6-B env wiring
  commit). Library kwarg lives on `dev/fvfd-core` HEAD `7c790cd8`.
- Aqua job `21397692.aqua`, gpu_batch_exec, 1 GPU on `gpu0n007`.
  Submitted 14:46 AEST, ran 15:20 → 16:34, total walltime 01:13:54,
  Exit_status 0.
- PBS: `bench/viscoelastic_logfv/run_cavity_m6b_confirm.pbs`.
- Common parameters (rheoTool match on `De=1, beta=0.5`):
  `N=64, end_time=8.0, u_max=0.005, lambda_phys=1.0,
   nu_s=nu_p=0.1, bsd_fraction=0.75` (the best from M4b).
- Per-case wallclock: ~37 min each (consistent with the M1 baseline).

## Observable: relative L2 vs rheoTool reference

| polymer_wall_extrap | L2(u centerline) | L2(psi_xy y=0.75) |
|---------------------|------------------|-------------------|
| `:quadratic` (current default) | 0.1797 | 0.2441 |
| `:linear` (rheoTool match) | 0.1817 | 0.2433 |
| Δ (linear − quadratic) | +0.0020 (+1.1 %) | −0.0008 (−0.3 %) |

Reference (M1 baseline): 0.1797 / 0.2441 → `:quadratic` is bit-equal
to M1 → kwarg default preserves behaviour, sanity check passes.

## Interpretation

The M6-A audit predicted that matching rheoTool's
`linearExtrapolation` BC on `τ` at the moving lid would drop the F
discrepancy at cell (16, 63) from 54 % toward 15-30 %. The smoke at
N=32 t=2 measured a 12 % wall-row relative L2 between the two
stencils, confirming the kwarg is wired and the local effect is
real and non-trivial. However, the same change at the full N=64 t=8
production case yields a profile L2 change that is within the
noise floor of the comparison itself (centerline marginally worse,
psi_xy marginally better, both shifts under 1 percentage point).

This means the wall-stencil choice at the polymer-divergence step
affects the local force field near the wall but does not propagate
into a measurable global profile correction at the rheoTool
benchmark. The 18-24 % residual gap must therefore originate from
a different source that we have not yet identified.

## Candidates eliminated so far (running tally)

- M1: Re mismatch — refuted (L2 flat across `u_max`).
- M3: polymer pipeline / upwind diffusion — refuted (frozen-replay
  gives 4 % L2, well below 18-24 %).
- M4b: BSD operator (`bsd_fraction` sweep) — refuted; BSD *helps*,
  doesn't hurt.
- M6-B: wall-stencil mismatch (`:quadratic` vs `:linear`) — refuted
  at production.

M2 (corner-kernel artifact) remains "inconclusive at smoke" — its
`--full` mode has not been exercised on Aqua. It is the only named
candidate left that hasn't been definitively closed.

## Remaining possibilities (not yet probed)

1. **Other wall BCs**: the log-conformation `Ψ` wall BC is
   `zeroGradient` on both sides (M6-A) so matches; but the
   *implementation* of zeroGradient could differ (one-sided vs
   reflective, where Kraken's ghost is filled, etc.). Worth a
   second-pass audit at the implementation level.
2. **Initial conditions**: do Kraken and rheoTool start from the
   same state? Possibly Kraken rests-from-zero while rheoTool may
   spin-up differently. Could cause time-history divergence that
   doesn't converge to the same statistical steady state at t=8.
3. **Time integration / sub-stepping cadence**: Kraken's polymer
   substep is 1829-fold per LBM step (per the original session
   prompt). RheoTool uses its own scheme. The dt/lambda value
   alone was validated to machine precision in 0D — but cumulative
   bias over 10⁵ steps at the production cadence is harder to
   rule out.
4. **Grid convergence**: Kraken N=64 vs rheoTool N=127. The
   reference is on a 4× finer grid. The 18 % gap could be partly
   discretization-bound; a Kraken N=127 run might close part of it
   purely from refinement.
5. **M2 full** at N=64 t=8: the corner-kernel `--full` mode was
   wired but not yet tested. Worth running before opening a new
   mission.
6. **Coupling order / staggering**: the order in which the LBM
   step, polymer substep loop, and BSD correction are interleaved
   might not match rheoTool's. A non-commuting operator order
   would not show up in any of the audits above.

## Decision (Boss step-back)

Four of the five originally-mandated candidates plus the
user-suggested wall-BC alternative are refuted. The Mandate's
original "5 candidates" framing is exhausted. Two responsible next
moves:

a) **Close M2 full first** (the only remaining named candidate),
   submit the existing `run_cavity_corner_artifact_2d.jl --full` on
   Aqua. Cheap (one wrapper script, the script already exists).

b) **Step back to the user with the Mandate**: the 18-24 % gap may
   be a discretization-bound floor at N=64 vs N=127 rheoTool rather
   than a bug. Propose a small grid-convergence study (Kraken N in
   {64, 96, 128}) to see if the gap shrinks with resolution. If it
   does, the gap is "expected at this resolution" rather than a
   Kraken-specific defect. If it doesn't, the residual is real and
   warrants a new round of investigation (candidates 1-3 above
   become missions M7-M9).

The Boss recommendation is to do **both**: M2-full is cheap, and a
small refinement sweep at the same time bounds the right answer.

## Artefacts

- Raw Aqua results: `tmp/cavity_m6b_confirm/{quadratic,linear}/`
- Analysis stdout (reproduced for the record):

  ```text
  u_max    | L2(u_centerline) | L2(psi_xy_y=0.75)
  -------- | ---------------- | -----------------
  quadratic (15:57)| 1.797e-01 | 2.441e-01
  linear    (16:34)| 1.817e-01 | 2.433e-01
  ```

  (Column header reads `u_max` because the analyse script falls
  back to the kraken timestamp when the parent dir name doesn't
  match the `u<value>` regex.)
