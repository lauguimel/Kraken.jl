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
