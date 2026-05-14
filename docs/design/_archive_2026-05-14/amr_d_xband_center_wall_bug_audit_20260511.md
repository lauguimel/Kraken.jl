# AMR-D center xband wall-touch bug audit

Date: 2026-05-11
Branch: `slbm-paper`

## Scope

This audit targets the center-only nested `xband` case:

```text
benchmarks/krk/amr_d_convergence_2d/poiseuille_xband_center_only_nested4_debug.krk
Refine xband { region = [6, 0, 10, 12], ratio = 16 }
wall_xband_closure = 0
```

The goal is not to loosen macro tolerances. The required fix is an operator fix
that lets a vertical band touch the south/north walls without adding top/bottom
wall caps to the active refinement topology.

## Observations

The center-only topology builds as:

```text
xband_L1: i=6:11,  j=1:12
xband_L2: i=12:21, j=1:24
xband_L3: i=24:41, j=1:48
xband:    i=49:80, j=1:96
```

Active leaf ranges:

```text
L0: i=1:16,   j=1:12
L1: i=11:22,  j=1:24
L2: i=23:42,  j=1:48
L3: i=47:82,  j=1:96
L4: i=97:160, j=1:192
```

A one-step static leaf-equivalent route stream is exact for uniform streamwise
equilibrium:

```text
center-only static max_abs_rho_dev = 2.22e-16
center-only static mass_delta      = 5.68e-14
```

The buffered subcycled scheduler is not exact:

```text
center-only buffered max_abs_rho_dev = 1.6666666667e-5
worst active cell                    = L4 (97, 16)
mass_delta                           = 1.14e-13
```

The same probe is exact for:

```text
wall-closed xband buffered max_abs_rho_dev = 0
yband buffered max_abs_rho_dev             = 0
```

The defect at L4 `(97, 16)` is a single-population imbalance:

```text
q6 delta_rho =  0.0
q7 delta_rho = +1.6666666667e-5
rho defect   = +1.6666666667e-5
```

On a full fine Cartesian grid after 16 fine steps, the same wall signal is:

```text
q6 delta_rho = -1.6666666667e-5
q7 delta_rho = +1.6666666667e-5
rho defect   =  0.0
```

So the AMR scheduler is not creating the `q7` wall reflection by mistake. It is
missing the compensating `q6` diagonal population that should enter the fine
band from the coarse side.

The rho error is antisymmetric in x and sums to roundoff at each wall cone. This
matches the dashboard observation: rho is greater than 1 upstream of the band and
less than 1 downstream, with near-perfect left/right cancellation. A one-step
probe on the center-only case gives:

```text
L1 wall cone: west +9.375890e-7, east -9.375890e-7, sum 4.44e-16
L2 wall cone: west +5.846024e-6, east -5.846024e-6, sum 3.33e-16
L3 wall cone: west +O(1.4e-5), east -O(1.4e-5), sum 3.77e-15
L4 wall cone: west positive, east negative, sum 3.33e-14
```

Therefore the fix must not be a scalar density correction. The missing piece is
an antisymmetric diagonal population correction at wall/interface corners.

## Root Cause

The failing packet is the NE diagonal `q6` injected into L4 `(97, 16)` from the
adjacent L3 source `(48, 8)`:

```text
SPLIT_CORNER q6 weight=0.25
src = L3 (48, 8)
dst = L4 (97, 16)
```

That packet has already interacted with the south wall in the fine-equivalent
Cartesian evolution before it crosses the lateral coarse/fine interface. The
current subcycled coarse-to-fine path cannot represent that phase:

- `sync_down` deposits C2F packets from parent `owned`/restricted state before
  the child interval.
- `conservative_tree_subcycle_deposit_coarse_to_fine_route_2d!` computes one
  scalar parent packet and distributes it uniformly across child substeps.
- The parent state is cell-averaged at the parent level; it has no subcell or
  per-substep memory of the diagonal wall reflection.
- The static route table is exact because it samples all active leaves at the
  finest level in a single transport pass. The subcycled scheduler loses that
  leaf-equivalent phase information at the C2F boundary.

Relevant code:

```text
src/refinement/conservative_tree_subcycling_2d.jl
  sync_down source selection:                lines 1928-1972
  scalar C2F packet deposit:                 lines 1058-1095
  parent restriction stores only aggregates: src/refinement/conservative_tree_subcycle_buffers_2d.jl lines 236-292
```

The wall-closed/H topology works because it actively refines the wall-origin
causal cone. The missing diagonal wall phase never has to cross from coarse to
fine at the side of the center band.

## Rejected Fixes

These are not sufficient:

```text
coarse_to_fine_predictor_weight = 0.0..1.0
```

No effect on the canary because the predictor hook currently predicts collision,
not transport.

```text
coarse_to_fine_prolongation = :limited_linear
```

Still fails:

```text
flat           max_abs_rho_dev = 1.6666666667e-5
limited_linear max_abs_rho_dev = 1.6687528292e-5
```

```text
route_sampling = :subcycled_hybrid or :level_native
```

Worse or still incomplete:

```text
leaf_equivalent max_abs_rho_dev = 1.6666666667e-5
subcycled_hybrid max_abs_rho_dev = 1.5629687969e-1
level_native     max_abs_rho_dev = 6.9506946528e-3
```

A naive one-step parent transport predictor also overcorrects because it streams
the whole parent level, not just the phase-resolved C2F boundary samples.

## Implemented Solution

The validated patch uses a precomputed wall-corner diagonal balance list in the
subcycle route bank:

```text
src/refinement/conservative_tree_subcycling_2d.jl
  ConservativeTreeWallCornerBalancePair2D
  _conservative_tree_wall_corner_balance_pairs_2d
  conservative_tree_apply_wall_corner_balance_2d!
```

For each level, it records only rows inside the wall causal cone:

```text
south cone: j <= 2^level, using q7/q6
north cone: j > Ny_level - 2^level, using q8/q9
```

Full-width rows are skipped, which leaves the yband and wall-closed controls
unchanged. Rows with an x-normal wall/interface corner are paired west/east
inside the precomputed active-row envelope. After a buffered subcycled step,
the patch transfers only the antisymmetric mass difference between the paired
diagonal populations:

```text
south: F[left, q7] -= delta; F[right, q6] += delta
north: F[left, q8] -= delta; F[right, q9] += delta
delta = (mass(left) - mass(right)) / 2
```

This is not a scalar rho clamp: it is a conservative diagonal population
closure on the wall/interface corner pairs. It preserves global mass and removes
the artificial west/east dipole without changing the center-only topology.

The C2F phase-resolved ledger shadow remains a possible future refinement, but
the direct C2F experiments moved the defect instead of closing it. The accepted
CPU fix is therefore the compact post-subcycle wall-corner balance above.

## Acceptance Tests

The bug is closed only when all of these pass:

1. Center-only xband analytical uniform equilibrium:

```text
wall_xband_closure = 0
max_abs_rho_dev <= 1e-14
max_x_jump      <= 1e-14
```

2. Center-only xband diagonal perturbation:

```text
epsilon = 1e-6
max_abs_rho <= 1e-18
max_x_jump  <= 1e-18
```

3. Existing controls remain green:

```text
yband nested4
wall-closed xband nested4
rest-state nested bands
```

4. Macro validation at a matched physical final time:

```text
t_final_leaf_steps == t_final_reference_leaf_steps
r2_ux_active_vs_reference >= 0.99
r2_rho_active_vs_reference >= 0.99
r2_ux_active_vs_analytic >= 0.95
r2_rho_active_vs_analytic >= 0.999
rho_level_boundary_max_abs_dev <= 5e-5
rho_level_boundary_max_jump <= 5e-5
```

Validated local dashboard:

```text
benchmarks/results/quicklook/amr_d_xband_center_only_wall_corner_canary_tfinal400_cpu_f64_20260511
steps = 400
t_final_leaf_steps = 6400
t_final_reference_leaf_steps = 6400
r2_ux_active_vs_reference = 0.9940348411748451
r2_rho_active_vs_reference = 0.9999999999904717
r2_ux_active_vs_analytic = 0.9940192361156314
r2_rho_active_vs_analytic = 0.9999999999904717
rho_level_boundary_max_jump = 2.4431228227594914e-5
rho_level_boundary_max_abs_dev = 2.7674794950804937e-5
validation_status = validated
```

5. Required branch canaries:

```bash
julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_gpu_pack_2d.jl")'
julia --project=. -e 'using Test; using Kraken; include("test/test_amr_d_krk_validation_2d.jl")'
```
