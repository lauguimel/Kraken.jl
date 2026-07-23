# AMR-E Basilisk-Like LBM

Date: 2026-05-06

Status: saved as a future research track. Do not mix with AMR-D implementation
until AMR-D reaches its publication milestone.

## Purpose

AMR-E is the alternative path inspired by recent Basilisk-style adaptive LBM.
It is useful to keep because it may be a better long-term architecture for
dynamic multi-level 2D/3D AMR, but it is not the current publication path.

AMR-D remains:

- route-native;
- collide-stream;
- subcycled across levels;
- fixed/pool-based for the publication feature.

AMR-E would be:

- non-uniform-grid native;
- single global time-step, or at least no classical recursive collide-stream
  subcycling;
- based on a finite-difference / finite-volume transport discretization of the
  population advection equation;
- closer to the Cheng-Wachs/Basilisk strategy than to Filippova-Hanel
  collide-stream refinement.

## Core Idea

Classical LBM streaming uses the exact lattice shift:

```text
f_i(x + c_i dt, t + dt) = f_i*(x, t)
```

That exact shift becomes awkward on a non-uniform quadtree/octree because
neighbor cells can have different sizes and, with subcycling, different local
times.

AMR-E rewrites streaming as advection of each population:

```text
partial_t f_i + c_i . grad(f_i) = collision_i + forcing_i
```

The advection term is then discretized on the adaptive grid, for example with a
Lax-Wendroff-like stencil. The tree geometry is still Basilisk-like, but the
LBM step is no longer a pure exact shift along lattice links.

## Why It Matters

The current AMR-D nested blocker comes from an intermediate level having two
time roles:

- child of `L0`, receiving reflux/prolongation;
- parent of `L2`, feeding finer ghost/interface states.

With a single global time-step, all levels are synchronized at the same physical
time. That removes much of the recursive `sync_down / fine substep / sync_up`
machinery. The price is a different numerical transport operator.

## What AMR-E Reuses

AMR-E can reuse from AMR-D/v0.4:

- static and adaptive quadtree/octree ownership;
- pool-based leaf storage;
- 2:1 balance and nesting constraints;
- `.krk` refinement DSL;
- adaptation criteria;
- restriction/prolongation concepts;
- output and benchmark infrastructure.

AMR-E should not reuse directly:

- route-native exact streaming tables;
- AMR-D subcycle ledgers;
- the current bounce-back/MEA obstacle code without revalidation;
- any macro-flow result obtained from collide-stream assumptions.

## Numerical Risks

AMR-E is not just an implementation refactor.

Main risks:

- Lax-Wendroff transport adds dispersion/diffusion errors absent from exact
  lattice streaming;
- CFL and stability limits must be explicit;
- conservation must be proven for integrated populations on non-uniform leaves;
- obstacle and open-boundary closures need a new derivation;
- GPU kernels become irregular stencil kernels, not simple route scatter;
- comparison against standard Cartesian LBM is mandatory before macro claims.

## Validation Ladder

If AMR-E is opened later, use this surgical order:

1. Scalar advection of one population on a 1D non-uniform mesh.
2. D2Q9 rest state on a 2D quadtree.
3. Uniform velocity advection on a fixed quadtree.
4. Taylor-Green or periodic shear on uniform Cartesian vs non-uniform tree.
5. Couette with a refinement interface crossing the shear.
6. Poiseuille with one refinement interface, then nested interfaces.
7. Moving/adaptive refinement without obstacles.
8. Square/cylinder obstacle with a rederived wall model.
9. 3D sphere only after 2D stability and accuracy are established.

## Decision Rule

AMR-E should become active only if one of these is true:

- AMR-D subcycling remains too complex after the buffer-contract rewrite;
- dynamic AMR becomes more important than route-native publication simplicity;
- 3D multi-level fixed AMR exposes a structural limitation in the subcycled
  collide-stream route.

Until then:

- finish AMR-D;
- keep AMR-E as a documented research option;
- do not let AMR-E change AMR-D tests, APIs, or milestone gates.

## Literature Anchors

- Filippova-Hanel grid-refined LBGK: DOI `10.1006/jcph.1998.6089`.
- Dupuis-Chopard grid refinement: DOI `10.1103/PhysRevE.67.066707`.
- Lagrava et al. multi-domain LBM: DOI `10.1016/j.jcp.2012.03.015`.
- Guzik et al. cell-centered AMR-LBM: DOI `10.1016/j.jcp.2013.11.037`.
- Cheng-Wachs Basilisk-like adaptive LBM: DOI `10.1016/j.jcp.2022.111669`.
