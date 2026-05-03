# AMR Route-Native Progress Ledger

Date: 2026-05-03

This document tracks the autonomous AMR work against the first eight milestones
from `docs/design/amr_complete_project_plan.md`. It is intentionally strict:
completed means backed by surgical patch tests and, where relevant, a short
macro-flow.

## Current Validation Commands

```bash
julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_streaming_2d.jl")'
julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_open_boundary_2d.jl")'
julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_gpu_pack_2d.jl")'
julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_subcycling_2d.jl")'
julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_adaptation_2d.jl"); include("test/test_conservative_tree_multipatch_2d.jl")'
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
- surgical open-boundary canaries for fine inlet/outlet faces when a patch
  touches west/east boundaries;
- short inlet-spanning open-channel smoke with bounded finite drift;
- short route-native open-channel smoke with bounded mass drift;
- bounce-back solid mask;
- square obstacle route-native smoke;
- vertical facing step route-native smoke.

Not done yet:

- longer open-channel/BFS stability with Zou-He route-native boundaries;
- long inlet-spanning open-channel stability. A 40-step probe still shows large
  mass drift, so this remains a BFS blocker;
- BFS route-native validation. The current D stream has Poiseuille, Couette,
  square obstacle and VFS only.

Next surgical patch:

- debug the long inlet-spanning open-channel drift at the boundary packet level,
  then extend toward an obstacle-free BFS precursor only after mass/momentum
  behaviour stays bounded.

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
- prescribed adaptive Poiseuille route-native canary;
- production-facing adaptation policy/proposal/plan layer;
- parent-grid indicator plans with padding, min-size, growth limits and
  hysteresis;
- `.krk` `Refine` blocks converted to named adaptation proposals;
- plan application helper that calls the direct conservative regrid path only
  when the selected patch range changes.

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
- adaptive Poiseuille keeps mass drift bounded;
- policy-level tests cover domain clamp, min-size expansion, max-growth
  limiting, near-shrink hysteresis, indicator-driven plans, `.krk` proposals
  and conservative regrid through a plan.

Not done yet:

- error indicators;
- integration of physical error indicators with hysteresis;
- 2:1 balancing for multiple patches;
- adaptive square obstacle/VFS macro-flow.

Next surgical patch:

- add physical error indicators on top of the policy/proposal/plan layer, then
  keep BFS blocked behind route-native open-boundary patch tests.

## 6. Sous-Cycling Temporel 2D

Status: conservative packet ledger started; no time integrator yet.

Implemented:

- `ConservativeTreeSubcycleLedger2D` for one coarse step and two fine
  half-steps;
- coarse-to-fine face and corner packet deposits split in time;
- fine-to-coarse face and corner packet accumulation by half-step;
- orientation and total packet sums for cycle-level canaries;
- reset helper for repeated canary cycles.

Validated by:

- a face packet is consumed once across two half-steps;
- a corner packet is split only in time, not spatially duplicated;
- fine-to-coarse half-step packets accumulate to the expected orientation sum;
- a symmetric full-cycle ledger preserves expected orientation and total sums;
- unsupported ratio, invalid face/corner direction, wrong block shape and bad
  substep are rejected.

Still required before implementation:

- wire the ledger into route-native coarse/fine streaming;
- define the collision ordering for coarse step and two fine steps;
- compare subcycled and non-subcycled Couette/Poiseuille short runs.

Exit gate:

- cycle-level mass conservation per orientation;
- Couette and Poiseuille short runs remain close to the non-subcycled path.

## 7. GPU 2D

Status: GPU-ready route packing and CPU parity canaries started; no device
kernel yet.

Implemented:

- `ConservativeTreeGPURoutePack2D` structure-of-arrays route pack;
- primitive route arrays for source/destination packed cell, `q`, route kind
  and weight;
- primitive block and cell metadata arrays;
- packed route weight-sum canary helper;
- CPU replay path driven only by the packed arrays for interior routes.

Validated by:

- primitive array types are `Int32`, `UInt8` and typed route weights;
- packed route source/destination ids match logical topology routes;
- route categories match direct/interface/boundary topology categories;
- every packed source/orientation route sum remains one;
- packed CPU replay matches logical route-native interior streaming exactly.

Still required before implementation:

- transfer this pack to CUDA/Metal arrays;
- write the no-allocation device kernel for interior routes;
- add boundary route kernels after open-boundary hardening;
- compare CPU/GPU route-native Poiseuille on a small fixed patch.

First target:

- device parity for one small packed route packet canary, then one small
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
- robust open boundaries, because only one-step/short smokes are covered and
  long inlet-spanning drift remains unresolved;
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
5. debug long inlet-spanning open-channel drift with packet-level canaries;
6. route-native BFS macro-flow only after those open-boundary tests.
