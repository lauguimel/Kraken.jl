# AMR D Publication Validation Plan

Date: 2026-05-05

## Scope

The publication-facing D stream is a static fixed-patch AMR feature:

- 2D D2Q9;
- ratio-2 refinement;
- one fixed patch known before the run;
- no dynamic regrid during the time loop;
- no multi-level claim;
- no GPU claim;
- no AD claim.

This keeps the feature honest and publishable. The pool/topology is fixed for
the whole simulation, so validation can focus on transport, collision,
boundaries and obstacle accuracy instead of runtime allocation or adaptation.

## Strategy

The current route-native implementation is conservative and preserves rest
states, but obstacle accuracy is sensitive to a coarse/fine interface too close
to the body. For publication D, the obstacle validation therefore uses an
interface-buffered patch:

```text
base obstacle domain: Nx=24, Ny=14
default patch:         8:17 x 4:11
publication patch:     3:22 x 1:14
```

The publication patch makes the obstacle and near wake live in a fine,
wall-to-wall band. The coarse/fine interfaces are moved upstream/downstream
instead of cutting through the obstacle boundary layer, while retaining two
active coarse columns on each periodic-x side in the base case.

This is not a trick: it is the expected static-AMR usage pattern for body
flows. A body-fitted or locally refined solver should not place its strongest
resolution transition directly on the quantity being measured.

## Validation Ladder

Required local validation sequence:

1. subcycling ledger canaries remain green;
2. route-native topology/streaming canaries remain green;
3. open-boundary/BFS surgical canaries remain green;
4. Couette route-native profile gates remain green;
5. Poiseuille route-native profile gates remain green;
6. square obstacle conserves fluid mass to roundoff;
7. cylinder interface-buffered route-native Cd is within 10% of the leaf
   oracle on short local canaries;
8. aqua ladder for square/cylinder scales 1 and 2 with
   `patch_strategy=:interface_buffered`.

Local baseline recorded in
`benchmarks/results/amr_obstacle_convergence_2d_local_interface_buffered_20260505.csv`:

- cylinder scale 1: route-native Cd/oracle Cd = `1.060x`;
- cylinder scale 2: route-native Cd/oracle Cd = `1.052x`;
- square scales 1 and 2: finite positive streamwise velocity and roundoff mass
  drift;
- all route-native obstacle rows: relative mass drift below `1e-12`.

## Limits

The default compact patch is still useful as a stress test. Its cylinder Cd
gap documents the remaining near-interface transport error. That error is the
motivation for future subcycling integration.

Do not claim that D has solved arbitrary coarse/fine obstacle placement until
the subcycling time integrator is wired and the compact-patch ladder also
passes.

## Commands

Local surgical checks:

```bash
julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_subcycling_2d.jl"); include("test/test_conservative_tree_obstacle_interface_2d.jl"); include("test/test_conservative_tree_open_boundary_2d.jl"); include("test/test_conservative_tree_streaming_2d.jl")'
```

Publication obstacle ladder:

```bash
KRK_AMR_CONV_PATCH_STRATEGY=interface_buffered \
KRK_AMR_CONV_BASE_STEPS=1200 \
KRK_AMR_CONV_SCALES=1,2 \
julia --project=. benchmarks/amr_obstacle_convergence_2d.jl
```

Aqua:

```bash
KRK_AMR_CONV_PATCH_STRATEGY=interface_buffered \
qsub hpc/amr_obstacle_convergence_2d_aqua.pbs
```
