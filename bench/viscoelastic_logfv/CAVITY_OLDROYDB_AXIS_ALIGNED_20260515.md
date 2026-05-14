# Cavity Oldroyd-B axis-aligned validation — 2026-05-15

## Objective

Compare the closed Kraken log-FV polymer pipeline against the rheoTool
`Cavity/Oldroyd-BLog` tutorial (user guide section 5.1.4). The cavity has
all-axis-aligned walls and a closed domain, isolating the core polymer
pipeline from cut-cell and open-boundary confounds that complicate the
cylinder and contraction benchmarks.

## What was implemented this session

1. **`logfv_wallxwally_bcspec_2d`** — new closed-domain BC spec
   (`src/fvfd/specs.jl`, `src/kernels/logconformation_fv_2d.jl`).
2. **`_bc_north_zh_velocity_2d!`** — new Zou-He moving-lid kernel and
   dispatch through `apply_bc_rebuild_2d!`
   (`src/kernels/boundary_rebuild.jl`).
   Prior to this session, `ZouHeVelocity` placed on the north face was
   silently ignored (the comment at line 329 explicitly said "South /
   North ZouHe variants can be added later"). Any prior cavity attempt
   would have run with a stationary lid.
3. **`run_viscoelastic_logfv_cavity_coupled_2d`** — new coupled driver
   in `src/drivers/viscoelastic_logfv_2d.jl` with:
   - smooth-start lid profile `U_lid(x, t) = 8 * u_max * (1 + tanh(8 (t − 0.5))) * x^2 (1 − x)^2`,
   - half-cell wall velocity gradient correction on all 4 walls
     (mandatory because the standard solid-aware gradient kernel uses
     only interior cells and would miss the moving lid Dirichlet),
   - closed-domain initialisation (`u = 0`, `rho = 1`, `psi = 0`).
4. **rheoTool reference** — project-local copy of the tutorial at
   `bench/rheotool/cavity_oldroydb_log_re001_de1_b05/` (tutorial
   `writeInterval`, `endTime`, `adjustTimeStep`, `maxDeltaT` tuned for
   tractable wall-clock; reference run completed in ~81 min wall on
   Docker, snapshots from `t = 1` to `t ≈ 8` plus sampled centerline
   and horizontal profiles and `kinEner.txt`).
5. **Comparison harness**
   `bench/viscoelastic_logfv/run_cavity_oldroydb_vs_rheotool.jl` that
   parses rheoTool `.xy` profile files, runs the Kraken cavity driver
   at matching parameters, interpolates Kraken fields onto rheoTool
   sample lines, and writes CSV summaries.

## BSD discordance — new finding

Default Kraken driver setting is `bsd_fraction = 1.0` (full
both-side-discretization split: LBM solvent uses augmented viscosity
`nu_s + nu_p`; explicit polymer force subtracts `nu_p * laplacian(u)`
to compensate). In a continuous formulation the split cancels exactly.

In Kraken's LBM the implicit BGK-LBM diffusion stencil and the
explicit FD central laplacian used in the BSD correction are not
bit-equivalent. The discordance is `O(h^2)` in smooth regions but
amplifies at velocity discontinuities — exactly what the moving lid
introduces at the top corners of the cavity.

The cavity at `N = 32`, `De = 1`, `u_max = 0.005`, `t = 2` BSD sweep
(2026-05-14):

| bsd_fraction | completed | first_nonfinite | max|ux| | max|psixx| | max|tauxy| |
|--------------|-----------|-----------------|---------|------------|------------|
| 0.00 | 12800/12800 | none | 4.97e-3 | 4.21 | 5.04e-5 |
| 0.25 | 12800/12800 | none | 4.97e-3 | 4.20 | 5.00e-5 |
| 0.50 | 12800/12800 | none | 4.97e-3 | 4.19 | 4.98e-5 |
| 0.75 | 12800/12800 | none | 4.97e-3 | 4.18 | 4.95e-5 |
| 1.00 | THROW | log(−1.35e+67) | — | — | — |

Within the stable range the polymer profile is essentially insensitive
to `bsd_fraction` (variations of `<2%` on `max|psixx|` and `max|tauxy|`).
The jump between `bsd = 0.75` and `bsd = 1.0` is a discrete-stability
threshold, not a slow drift.

This is a Kraken-architecture issue, not a problem with BSD/coupling
as such. RheoTool uses the same coupling stabilization on the same
case at `bsd = 1` equivalent without trouble because rheoTool is FV
throughout (the implicit and explicit halves share the discrete
laplacian stencil).

## Implication for the cylinder RheoTool drift

The cylinder benchmark uses `bsd_fraction = 1.0` by default. The
mesh-divergent `+6.86%` drift in `Cd` at `R = 30` against rheoTool is
Wi-independent, growing with refinement — a signature compatible with
the same LBM/FD discordance, accumulating at the curved wall instead
of crashing at the lid.

A direct cylinder bsd-sensitivity test is the cleanest next step:
re-run `R = 30`, `Wi = 0.1`, `beta = 0.59` with `bsd_fraction in {0.25,
0.5, 0.75, 1.0}` and observe whether the drift decreases. This is a
separate session.

## Proper fix (deferred)

The architecturally clean fix is **kinetic-moment BSD**: replace the
FD central laplacian in the explicit half with the rate-of-strain
tensor extracted from the LBM non-equilibrium moments
`f_q − f_q^{eq}`. That tensor is, by Chapman-Enskog construction,
exactly what BGK is relaxing toward, so the cancellation becomes exact
at the discrete level (modulo BC details).

Implementation:
1. New kernel `extract_strain_rate_from_neq_2d!(D, f, rho, omega)` that
   computes `D_{ab} = -(1 / (2 rho cs^2)) * sum_q omega * (f_q − f_q^{eq}) * c_{qa} * c_{qb}`
   for each fluid cell.
2. Replace the FD laplacian in `logfv_bsd_correct_force_*` with
   `-2 * bsd * nu_p * div(D_LBM)` (using the same `fvfd_tensor_divergence_2d!`
   that already handles polymer stress, so the divergence stencil
   matches the rest of the polymer pipeline).

This change touches every step-channel-style coupled driver. It is
deferred until the cavity comparison gives a concrete success/fail
verdict, and until the cylinder bsd-sensitivity sweep confirms or
rules out the discordance hypothesis as the dominant source of the
cylinder drift.

## Validation smokes (this session)

| Configuration | Status |
|---|---|
| Newtonian cavity (nu_p=0): 12800 steps | finite, max\|ux\|=u_max, vortex secondary motion present |
| Visco De=0.01 + bsd_fraction=0: 8000 steps | finite, max\|psixx\|≈0.06 |
| Visco De=1 + bsd_fraction=0: 8000 steps | finite, max\|psixx\|≈3 (C_xx ≈ 20, full stretching) |
| Visco De=1 sweep bsd ∈ {0, 0.25, 0.5, 0.75, 1.0}: 12800 steps each | bsd<=0.75 finite, bsd=1.0 crash |

## Next concrete steps

1. **Production comparison run**: `run_viscoelastic_logfv_cavity_coupled_2d`
   at `N = 64`, `end_time = 8`, `bsd_fraction = 0.75` (closest stable
   to rheoTool default), CPU.
2. **Compare profiles** `u(x = 0.5, y)` and `psi_xy(x, y = 0.75)` against
   the rheoTool snapshots at `t ≈ 8`. The harness produces CSVs ready
   for plotting.
3. Optionally refine `N = 128` if `N = 64` is in the right ballpark
   but with quantitative discrepancies.
4. Write a follow-up verdict file with the actual relative L2 errors
   on profiles and the conclusion on the cavity validation.
5. Then: cylinder bsd-sensitivity sweep (separate session, HPC).
