# M58 Analytic Chain Verdict

## Summary
FO is GREEN across the full imposed-field analytic chain: all field, rotation, gate, and bin aggregates remain at machine precision or zero. M55b is GREEN only away from the embedded wall. It is RED for every cut-cell Gate 1 gradient aggregate on the requested non-no-slip fields, and Gates 2-3 inherit that gradient error algebraically. Gate 4 is also RED for M55b, including AXIS_INTERIOR aggregates, because the divergence stencil reads the corrupted cut-cell stresses. The systematic pattern is not a random rotation-only defect: the M55b helper hard-codes the embedded wall value as zero, while these imposed analytic fields are generally nonzero on the cylinder wall.

## Methodology
The patch test uses the L1 R=4 cylinder mesh with `Nx=Ny=32`, `cx=cy=16.5`, and `Kraken.precompute_q_wall_cylinder` followed by `fvfd_embedded_boundary_from_qwall_2d`, so `embedded.wall_q` is populated and M55b fires. It imposes `F_shear`, `F_rotation`, and `F_poiseuille` at rotations `0`, `pi/4`, `pi/2`, and `3pi/4`, then compares gradient, Oldroyd-B source at `C=I`, stress after `dt=0.01*lambda`, and polymer divergence force against analytic fields. The CPU run wrote `scratch/M58/M58_per_cell_errors.csv` and `scratch/M58/M58_aggregate_matrix.csv`. Run note: `git stash pop` failed with `could not write index` in this linked worktree, but the M55b files were already active in the filesystem and `src/fvfd/qw_wall_gradient_2d.jl` existed.

## Gate 1 (gradient): aggregate matrix
Values are max_abs_err. Bold entries are RED under `max_abs_err > 0.1 * ||grad u||_inf`. This R=4 mesh produced only `CUT_Q_0.5_0.7` and `CUT_Q_0.7_0.9` bins.

| field | rot | FO axis | FO q.5-.7 | FO q.7-.9 | M55 axis | M55 q.5-.7 | M55 q.7-.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F_shear | 0 | 0 | 0 | 0 | 0 | **10.1** | **6.44** |
| F_shear | pi/4 | 4.44e-15 | 1.11e-15 | 1.78e-15 | 4.44e-15 | **7.59** | **4.25** |
| F_shear | pi/2 | 2.66e-15 | 6.05e-16 | 1.61e-16 | 2.66e-15 | **10.1** | **6.44** |
| F_shear | 3pi/4 | 4.44e-15 | 1.11e-15 | 1.78e-15 | 4.44e-15 | **7.59** | **4.25** |
| F_rotation | 0 | 0 | 0 | 0 | 0 | **10.1** | **6.44** |
| F_rotation | pi/4 | 7.11e-15 | 1.78e-15 | 2.44e-15 | 7.11e-15 | **10.1** | **6.44** |
| F_rotation | pi/2 | 1.11e-16 | 1.11e-16 | 0 | 1.11e-16 | **10.1** | **6.44** |
| F_rotation | 3pi/4 | 7.11e-15 | 1.78e-15 | 2.44e-15 | 7.11e-15 | **10.1** | **6.44** |
| F_poiseuille | 0 | 0 | 0 | 0 | 0 | **2.50** | **1.56** |
| F_poiseuille | pi/4 | 9.10e-15 | 5.00e-16 | 6.66e-16 | 9.10e-15 | **1.87** | **0.959** |
| F_poiseuille | pi/2 | 5.33e-15 | 3.14e-16 | 1.55e-16 | 5.33e-15 | **2.50** | **1.56** |
| F_poiseuille | 3pi/4 | 9.10e-15 | 5.00e-16 | 6.66e-16 | 9.10e-15 | **1.87** | **0.959** |

## Gate 2-4: aggregate
Gate 2 source: FO remains GREEN everywhere. M55b has 24 RED aggregate rows, exactly the cut-cell rows that were RED in Gate 1; the worst source error is `17.895` for `F_rotation`, `rot=0`, `CUT_Q_0.5_0.7`. This is not a separate source-closure break; it is the algebraic result of feeding the bad M55b gradient into the Oldroyd-B source at `C=I`.

Gate 3 stress: FO remains GREEN everywhere. M55b again has 24 RED cut-cell aggregate rows, with worst stress error `0.3579` for `F_rotation`, `rot=0`, `CUT_Q_0.5_0.7`. This scales exactly as `G*dt` from the Gate 2 error.

Gate 4 divergence: FO remains GREEN everywhere. M55b has 36 RED aggregate rows: every field and rotation is RED in both cut bins, and AXIS_INTERIOR also becomes RED because the divergence operator sees neighboring corrupted stress. Worst observed M55b force error is `0.2412` for `F_shear`, `rot=0`, `CUT_Q_0.5_0.7`.

## Direction asymmetry pattern (rotation analysis)
M55b Gate 1 max cut-cell errors by rotation:

| field | 0 | pi/4 | pi/2 | 3pi/4 | pattern |
|---|---:|---:|---:|---:|---|
| F_shear | 10.1 | 7.59 | 10.1 | 7.59 | pi/2 periodic; axis-aligned worse |
| F_rotation | 10.1 | 10.1 | 10.1 | 10.1 | rotation-invariant failure |
| F_poiseuille | 2.50 | 1.87 | 2.50 | 1.87 | pi/2 periodic; axis-aligned worse |

The observed asymmetry is pi/2 periodic for shear and Poiseuille, not a pi/4 D2Q9-link periodicity. Rigid rotation fails uniformly because the imposed tangential wall velocity is nonzero everywhere on the cylinder.

## M55b vs FO comparison
FO wins every `(field, rotation)` pair on Gate 1: its max errors are zero or `O(1e-15)`, while M55b max cut-cell errors range from `0.959` to `10.1`. M55b never improves on FO for these imposed fields. The failure regime is specifically cut cells with populated `wall_q`; off-wall M55b and FO are tied at machine precision.

## Localisation
Smallest failing combination: `F_shear`, `rot=0`, Gate 1, M55b, `CUT_Q_0.5_0.7`. At cell `(17,13)`, `dudy` is `11.060175126340535` versus analytic `1.0`, so `abs_err=10.060175126340535`; the cell has max populated `q_w=0.677` and normal approximately `(0,-1)`. Because the requested fields violate the helper's zero-wall-value assumption, this verdict localizes a BC-assumption incompatibility in M55b rather than proving a sign/rotation algebra bug inside the bilinear q-wall formula.

## Next-mission candidates
- Add an M58b zero-wall manufactured-field variant, for example simple affine/quadratic fields multiplied by `(r^2 - R^2)`, to test M55b under its actual no-slip wall contract.
- Add an explicit wall-value parameter to the embedded gradient helper if arbitrary imposed analytic fields are meant to be supported.
- Re-run this matrix on a no-slip analytic field family before using M55b as evidence for or against the M48 NaN.
- If M55b remains RED on zero-wall manufactured fields, instrument `_fvfd_qw_wall_equation_2d` at the smallest failing cell `(17,13)` and compare target normal derivative to closed form.
