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
- bounce-back solid mask;
- square obstacle route-native smoke;
- vertical facing step route-native smoke.

Not done yet:

- Zou-He inlet/outlet on route-native AMR;
- BFS route-native validation. The current D stream has Poiseuille, Couette,
  square obstacle and VFS only.

Next surgical patch:

- impose west/east moments on active composite cells;
- add one-cell inlet/outlet patch tests before any BFS macro-flow.

## 4. Multi-Patch Statique 2D

Status: not started in code.

Required next sequence:

1. Generalize ownership from one patch to a patch list.
2. Add active-cell lookup tests for disjoint patches.
3. Add route tests for coarse-to-fine and fine-to-coarse near two patches.
4. Add patch-patch adjacency tests.
5. Only then run macro-flows with two refined bands.

Exit gate:

- no double ownership;
- active mass/volume matches oracle;
- all route weights sum to one by orientation.

## 5. Adaptation Dynamique CPU 2D

Status: first conservative prescribed regrid path done.

Implemented:

- direct conservative patch regrid;
- mask-driven patch adaptation for solid proximity;
- prescribed adaptive Poiseuille route-native canary.

Validated by:

- direct regrid equals leaf-oracle regrid for grow, shrink and shift;
- population sums are conserved through regrid;
- adaptive Poiseuille keeps mass drift bounded.

Not done yet:

- error indicators;
- hysteresis;
- 2:1 balancing for multiple patches;
- adaptive square obstacle/VFS macro-flow.

Next surgical patch:

- introduce a pure indicator function that returns a target patch range without
  mutating state, then test it independently from transport.

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

Status: pending.

Existing 3D refinement utilities are not yet the complete conservative-tree
D3Q19 route topology. The first acceptable 3D milestone is deliberately small:

- D3Q19 directions and opposites;
- coarse volume `1`, fine volume `1/8`;
- 8-child coalesce and uniform explode primitives;
- split/coalesce weights for face, edge and corner crossings;
- topology canaries only, no 3D macro-flow until these pass.

Exit gate:

- orientation-wise conservation for every D3Q19 route primitive;
- active mass and momentum agree before and after projection/restriction.

## Publication-P Milestone Gate

Currently publishable inside the D stream only as:

```text
2D, single fixed patch, route-native conservative AMR:
Couette + Poiseuille + square obstacle + VFS
```

The wording must not claim:

- BFS route-native, because it is not implemented in this D stream;
- open boundaries, because Zou-He route-native is pending;
- multi-patch AMR;
- subcycling;
- GPU AMR;
- 3D AMR.

Next commits should continue in this order:

1. route-native open-boundary patch tests;
2. BFS route-native macro-flow after those patch tests;
3. pure indicator and hysteresis tests for dynamic 2D adaptation;
4. multi-patch ownership tests;
5. D3Q19 primitive tests.
