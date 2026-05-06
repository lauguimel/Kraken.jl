# AMR-D KRK Validation Matrix

Date: 2026-05-06

## Scope

This patch makes AMR-D `.krk` cases auditable before they are launched.

New helper APIs:

- `conservative_tree_amr_d_case_from_krk_2d`;
- `conservative_tree_amr_d_boundary_policy_2d`;
- `conservative_tree_amr_d_geometry_2d`;
- `conservative_tree_amr_d_support_matrix_2d`;
- `run_conservative_tree_amr_d_case_from_krk_2d`.

The intent is to separate three states:

- parse/spec supported;
- runtime supported;
- intentionally unsupported.

That distinction matters for IBB/LIBB and nested obstacle cases. They must not
look green just because the DSL parses.

## Current AMR-D 2D Status

Runtime supported:

- one-level Poiseuille and Couette route-native cases;
- one-level BFS open-channel route-native case;
- one-level square and cylinder solid-mask route-native cases;
- nested Poiseuille/Couette channel subcycling up to `max_level = 4`.

DSL/spec supported but runtime pending:

- nested square/cylinder/BFS obstacle cases;
- nested open-channel cases;
- long-channel cylinder lift show-off case.

Not AMR-D route-native D features yet:

- IBB;
- LIBB.

IBB/LIBB exist elsewhere in Kraken, but AMR-D route-native D currently uses a
halfway bounce-back solid mask. The support matrix marks IBB/LIBB as unsupported
in AMR-D so this cannot be hidden by a parser-only test.

## KRK Canaries

Nested channel canaries:

```text
benchmarks/krk/amr_d_convergence_2d/poiseuille_nested4_channel.krk
benchmarks/krk/amr_d_convergence_2d/couette_nested4_channel.krk
```

Both use:

```krk
Refine channel_core { region = [7.5, 4.5, 8.5, 5.5], ratio = 16 }
```

The static spec helper expands `ratio = 16` into four adjacent ratio-2 levels
with a one-level nesting difference between neighbors.

Show-off target:

```text
benchmarks/krk/amr_d_showoff_2d/cylinder_lift_re100_long_channel.krk
```

This file defines a long-channel cylinder lift target with open inlet/outlet and
a nested wake refinement. It is deliberately classified as runtime pending until
nested obstacle plus open-channel surgical tests pass.

## Surgical Test Command

```bash
julia --project=. -e 'using Test; include("test/test_amr_d_krk_validation_2d.jl")'
```

The test verifies:

- every convergence `.krk` is classified;
- nested channel `.krk` files run for a two-step smoke;
- nested cylinder is static-spec green but runtime red;
- show-off lift case is parse/spec green but runtime red;
- IBB/LIBB are explicitly unsupported in AMR-D D.

## Quicklook Output From KRK

The quicklook helper turns one or several `.krk` files into data and figures:

```bash
julia --project=. benchmarks/amr_d_quicklook_from_krk_2d.jl
```

Useful overrides:

```bash
KRK_AMR_D_QUICKLOOK_CASES=poiseuille_nested4_channel.krk,cylinder_scale1.krk \
KRK_AMR_D_QUICKLOOK_STEPS_OVERRIDE=20 \
KRK_AMR_D_QUICKLOOK_OUTDIR=benchmarks/results/quicklook/manual_check \
julia --project=. benchmarks/amr_d_quicklook_from_krk_2d.jl
```

For each case, the helper writes:

- `status.csv`: parsed flow, BC policy, wall model, max level, runtime support;
- `mesh_static.csv/png`: static tree mesh directly from the `.krk`;
- `mesh_amr_d.csv/png`: actual runtime mesh when the case is executable;
- `fields_amr_d.csv/png`: `rho`, `ux`, `uy`, `|u|`, level and solid mask;
- `profiles_amr_d.csv/png`: mean profile, centerline and vertical probe;
- `fields_compare.png`: AMR-D vs reference fields and field differences;
- `profiles_compare.png`: AMR-D vs reference profiles, plus analytic profile
  when available;
- `debug_dashboard.png`: one-page debug board with classic Cartesian mesh/ux/rho
  on top, AMR-D mesh/ux/rho in the middle, and profiles versus analytic on the
  bottom;
- `values.csv`: final mass drift, max raw mass residual before correction,
  mean/min/max values, profile errors and field errors;
- `summary.csv`: all generated artifact paths.

For nested channel cases, the reference method is `cartesian_classic`: a dense
D2Q9 Cartesian array at the finest leaf-equivalent resolution, without AMR tree,
AMR route table or AMR projection. For one-level obstacle cases, the reference
is the existing leaf-oracle route.

Runtime-pending cases, such as the long-channel cylinder lift show-off, still
emit `status.csv` and `mesh_static.csv/png`. They do not emit physical fields
until the corresponding AMR-D runtime gate is closed.

The `*_nested4_debug.krk` files are local visual debug cases. They use stronger
forcing or wall speed than the tiny scheduler canaries so `ux`, `rho`, field
differences and profiles are visible in quicklook plots. The
`cylinder_lift_nested4_probe.krk` case is an off-centre cylinder target: it must
show the nested mesh and cylinder mask now, but runtime fields remain pending
until nested open-channel obstacle routing is implemented.

Focused smoke:

```bash
julia --project=. -e 'using Test; include("test/test_amr_d_quicklook_krk_2d.jl")'
```
