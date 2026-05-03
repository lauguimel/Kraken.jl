# AMR Route-Native Progress Ledger

Date: 2026-05-03

This document tracks the autonomous AMR work against the first eight milestones
from `docs/design/amr_complete_project_plan.md`. It is intentionally strict:
completed means backed by surgical patch tests and, where relevant, a short
macro-flow.

## Current Validation Commands

```bash
julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_streaming_2d.jl")'
julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_topology_2d.jl"); include("test/test_conservative_tree_2d.jl")'
```

## 1. Streaming Composite Natif 2D

Status: done for one ratio-2 patch.

Implemented:

- same-level active-cell routes;
- coarse-to-fine face and corner splits;
- fine-to-coarse face and corner coalesces;
- route topology and packed route representation;
- periodic-x, wall-y, moving-wall-y and solid-mask route variants.

Surgical tests:

- packet canaries for direct, split and coalesce routes;
- orientation-wise conservation;
- route-native streaming against leaf-grid oracle canaries;
- periodic and wall boundary conservation.

Remaining before "complete AMR 2D":

- multi-patch routes;
- open inlet/outlet routes.

## 2. Collision Locale Active 2D

Status: done for BGK and Guo on active cells.

Implemented:

- coarse active cells collide with volume `1`;
- fine active cells collide with volume `0.25`;
- coarse cells covered by the patch are excluded;
- Guo forcing works in route-native Poiseuille.

Validation:

- route-native Couette runner;
- route-native Poiseuille runner;
- `validate_conservative_tree_route_native_phase_p_2d` compares route-native
  Couette and Poiseuille to the Phase-P oracle runners.

## 3. Boundaries Natives 2D

Status: partially done.

Done:

- periodic x;
- static wall y;
- moving wall y for Couette;
- integrated cell-level Zou-He west velocity and east pressure closures;
- composite active-cell Zou-He west/east application for coarse and fine
  boundary cells;
- one-step open-x route smoke with wall-y and composite Zou-He closures;
- short route-native open-channel smoke with bounded mass drift;
- bounce-back solid mask;
- square obstacle route-native smoke;
- vertical facing step route-native smoke.

Not done yet:

- longer open-channel/BFS stability with Zou-He route-native boundaries;
- BFS route-native validation. The current D stream has Poiseuille, Couette,
  square obstacle and VFS only.

Next surgical patch:

- extend the open-channel smoke toward an obstacle-free BFS precursor, then
  add the step only after mass/momentum behaviour stays bounded.

## 4. Multi-Patch Statique 2D

Status: ownership tables started; route topology still pending.

Implemented:

- `ConservativeTreePatchSet2D` groups several disjoint ratio-2 patches;
- parent-cell and leaf-cell owner tables;
- active coarse mask and active volume accounting;
- overlap and domain rejection;
- parser-adjacent `.krk` helpers converting base-grid `Refine` blocks to
  conservative-tree patch ranges.

Validated by:

- disjoint patch ownership canaries;
- parent/leaf owner lookup canaries;
- active volume and active coarse mask canaries;
- `.krk` `Refine` helper canaries for valid, unsupported ratio, nested parent
  and 3D-refine rejection.

Required next sequence:

1. Add route tests for coarse-to-fine and fine-to-coarse near two patches.
2. Add patch-patch adjacency tests.
3. Add route-native streaming over disjoint patch sets.
4. Only then run macro-flows with two refined bands.

Exit gate:

- no double ownership;
- active mass/volume matches oracle;
- all route weights sum to one by orientation.
- route topology covers patch-patch adjacency without packet loss.

## 5. Adaptation Dynamique CPU 2D

Status: first conservative prescribed regrid path done.

Implemented:

- direct conservative patch regrid;
- mask-driven patch adaptation for solid proximity;
- pure solid-mask patch indicator;
- pure scalar-threshold patch indicator;
- pure gradient-magnitude indicator field;
- composite AMR leaf velocity field for indicator input;
- velocity-gradient patch decision with direct conservative regrid;
- short route-native Poiseuille macro-flow driven by velocity-gradient regrid;
- mask-adaptive VFS route-native macro-flow around the step;
- range-level hysteresis primitive for grow/shrink decisions;
- prescribed adaptive Poiseuille route-native canary.

Validated by:

- direct regrid equals leaf-oracle regrid for grow, shrink and shift;
- population sums are conserved through regrid;
- indicator and hysteresis functions are tested without mutating transport
  state;
- composite velocity extraction feeds the gradient selector on a surgical
  local-speed patch;
- velocity-gradient adaptation regrids that patch while conserving population
  sums;
- gradient-driven adaptive Poiseuille regrids conservatively over a short
  route-native run;
- mask-driven VFS regrids around the step while keeping fluid-mass drift
  bounded;
- adaptive Poiseuille keeps mass drift bounded.

Not done yet:

- error indicators;
- integration of physical error indicators with hysteresis;
- 2:1 balancing for multiple patches;
- adaptive square obstacle/VFS macro-flow.

Next surgical patch:

- add route-native open-boundary patch tests before attempting BFS in this D
  stream.

## 6. Sous-Cycling Temporel 2D

Status: not started.

Required before implementation:

- define the conservative ledger for one coarse step and two fine steps;
- add packet tests for half-step interface transfers;
- prove that no packet is streamed twice or dropped on a full cycle.

Exit gate:

- cycle-level mass conservation per orientation;
- Couette and Poiseuille short runs remain close to the non-subcycled path.

## 7. GPU 2D

Status: not started.

Prerequisites:

- packed route arrays must be the only topology input in the hot loop;
- boundary route types must be explicit;
- no hash lookup or allocation in route streaming kernels.

First target:

- CPU/GPU parity for one small route packet canary, then one small
  route-native Poiseuille run.

## 8. Topologie Et Primitives 3D

Status: primitive, active-topology and interior-streaming canaries started.

Existing 3D refinement utilities are not yet the complete conservative-tree
D3Q19 route topology. The first accepted 3D slice is deliberately small:

- D3Q19 directions and opposites;
- coarse volume `1`, fine volume `1/8`;
- 8-child coalesce and uniform explode primitives;
- split/coalesce weights for face and edge crossings;
- explicit empty corner route set, because D3Q19 has no body-diagonal
  populations;
- topology canaries only, no 3D macro-flow until these pass.

Validated canaries:

- `test/test_conservative_tree_3d.jl`
- `test/test_conservative_tree_topology_3d.jl`
- `test/test_conservative_tree_streaming_3d.jl`
- D3Q19 integer accessors match the lattice constants;
- 8-child coalesce preserves every oriented population, mass and momentum;
- uniform explode followed by coalesce conserves the parent;
- integrated D3Q19 equilibrium preserves volume-weighted mass and momentum;
- fixed ratio-2 3D patch allocation uses active fine state and parent shadow
  ledger arrays with D3Q19 layout;
- active 3D topology stores inactive coarse ledger cells under the patch and
  active fine children with volume `1/8`;
- route tables classify D3Q19 links as direct, coarse-to-fine, fine-to-coarse
  or boundary;
- face packets split over four boundary-adjacent children and coalesce from
  the same child set;
- edge packets split over two edge-adjacent children and coalesce from the
  same child set;
- every logical 3D link has route weights summing to one;
- route-native 3D interior streaming moves direct packets, splits
  coarse-to-fine face/edge packets, coalesces fine-to-coarse face/edge packets
  by accumulation, and conserves active D3Q19 population sums when boundary
  sources are zeroed;
- periodic-x 3D transport wraps coarse boundary packets, wraps coarse packets
  into a fine patch touching the periodic seam, and wraps fine boundary packets
  back to active coarse cells;
- periodic-x plus stationary wall-y/z 3D transport bounces coarse and fine wall
  packets, keeps periodic x wrapping, and gives y/z walls priority at corner
  exits;
- integrated D3Q19 BGK collision conserves mass and momentum, `omega=1`
  projects to equilibrium at conserved moments, and integrated D3Q19 Guo
  collision conserves mass while driving momentum in the force direction;
- a tiny fixed-patch 3D transport+BGK loop with periodic x and stationary
  wall-y/z conserves active mass;
- all corner transfer calls reject, documenting the empty D3Q19 corner route
  set.

Exit gate:

- orientation-wise conservation for every D3Q19 route primitive;
- active mass and momentum agree before and after projection/restriction;
- 3D route-native macro-flow remains pending; no 3D AMR flow claim yet.

## Publication-P Milestone Gate

Currently publishable inside the D stream only as:

```text
2D, single fixed patch, route-native conservative AMR:
Couette + Poiseuille + square obstacle + VFS
```

The wording must not claim:

- BFS route-native, because it is not implemented in this D stream;
- open boundaries, because Zou-He route-native is pending;
- multi-patch route-native transport or adaptation, because only ownership and
  `.krk` setup helpers are started;
- subcycling;
- GPU AMR;
- 3D AMR.

Next commits should continue in this order:

1. production-grade adaptation plan helpers and DSL-facing guards;
2. route tests over multi-patch ownership tables;
3. subcycling ledger and packet canaries;
4. GPU packing parity canaries;
5. route-native open-boundary patch tests before any BFS macro-flow;
6. BFS route-native macro-flow only after those open-boundary tests.
