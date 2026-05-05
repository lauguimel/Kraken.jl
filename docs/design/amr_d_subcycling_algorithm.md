# AMR-D Subcycling Algorithm

Date: 2026-05-06

This document freezes the algorithmic correction for AMR-D before more macro
flows are attempted. The current one-level transport canary is useful, but the
nested failure shows that the runtime must stop treating one mutable population
matrix as state, ghost source, reflux target, and restriction buffer at the same
time.

## Literature Constraints

LBM grid refinement is not just finite-volume reflux with scalar fluxes. The
interface exchanges full lattice populations, including equilibrium and
non-equilibrium parts.

Relevant families:

- Filippova-Hanel style LBGK refinement:
  DOI `10.1006/jcph.1998.6089`.
  This is the classical local grid-refinement route: rescale non-equilibrium
  populations across levels and treat coarse/fine transfers as a reconstruction
  problem, not as blind packet copy.
- Dupuis-Chopard / Palabos multi-domain refinement:
  DOI `10.1103/PhysRevE.67.066707` and Lagrava et al.
  DOI `10.1016/j.jcp.2012.03.015`.
  These emphasize transition reconstruction, fine-to-coarse filtering/decimation,
  and stability at interfaces.
- Guzik-Weisgraber-Colella-Alder cell-centered AMR-LBM:
  DOI `10.1016/j.jcp.2013.11.037`.
  This is closest in spirit to AMR-D because it uses cell-centered adaptive
  refinement and constrained conservation. The important lesson is that
  space-time interpolation and constraints are needed to preserve LBM accuracy.
- Basilisk/Cheng-Wachs:
  DOI `10.1016/j.jcp.2022.111669`.
  This route avoids classical collide-stream subcycling by using a Lax-Wendroff
  streaming operator on non-uniform quad/octree grids with one global time scale.
  That is elegant, but it is a different algorithm from AMR-D's route-native
  subcycled collide-stream design.

## Chosen AMR-D Contract

AMR-D remains a fixed, route-native, cell-centered tree for the publication
feature. It can be Basilisk-like in geometry ownership, but not in time
integration:

- cells are leaves of a quadtree/octree;
- populations are integrated per cell volume;
- adjacent active leaves differ by at most one level;
- subcycling uses `dt_L = 2 * dt_(L+1)`;
- each level pair exchanges through explicit buffers;
- no level reads a parent row that has been partially modified by a child
  reflux at the same logical time.

## Required Buffers

For each level `L`, the runtime owns four distinct population buffers:

- `owned[L]`: committed physical populations for active rows at level `L`;
- `ghost_from_coarse[L]`: prolongated/reconstructed boundary populations coming
  from `L-1`;
- `reflux_to_coarse[L]`: packets accumulated by level `L+1` and waiting to be
  applied to level `L`;
- `restrict_to_parent[L]`: conservative child-to-parent sums used when a fine
  level has completed a coarse interval.

The old pattern `Fstate += pending; stream(Fstate); sync_down(Fstate)` is not a
valid multi-level algorithm. It lets an intermediate level act as both a child
receiving reflux and a parent feeding a finer level from a half-closed state.

## Recursive Step

For a parent level interval `[t0, t1]`:

1. Prepare child ghost:
   reconstruct `ghost_from_coarse[L+1]` from the committed parent state at
   `t0` and, for dynamic flows, from a time-consistent interpolation.
2. Advance child:
   execute the `ratio` child substeps using `owned[L+1]` plus child ghost
   buffers. Do not write into `owned[L]`.
3. Accumulate fine exits:
   child-to-parent exits accumulate into `reflux_to_coarse[L]` or route-level
   reflux registers.
4. Synchronize:
   after the child interval completes, restrict the child state into
   `restrict_to_parent[L]` and apply reflux/conservation constraints.
5. Commit parent:
   only after synchronization may `owned[L]` be updated and exposed as a source
   for lower levels or for macro diagnostics.

## Reconstruction Rule

The first implementation can use exact integrated restriction/prolongation at
rest:

- fine-to-coarse restriction is a sum of integrated child populations;
- coarse-to-fine rest prolongation divides integrated parent populations by
  four in 2D.

For physical flows, this is not enough. The next reconstruction layer must use:

```text
f = f_eq(rho, u) + alpha(level_from, level_to, tau) * f_neq
```

where `alpha` encodes the Filippova-Hanel / multi-domain non-equilibrium
rescaling. Without this, Couette/Poiseuille crossing an AMR boundary can pass a
mass canary while still producing a velocity or stress artifact.

## Gates Before Macro Flows

Implementation must progress in this order:

1. Buffer role tests:
   owned/ghost/reflux/restriction are disjoint and resettable.
2. Exact restriction/prolongation tests:
   rest-state integrated populations are conserved on `L0/L1` and
   `L0/L1/L2`.
3. Recursive scheduler tests:
   a same-tick child `sync_down` cannot sample a parent row after child reflux
   has been partially applied.
4. Transport-only canaries:
   rest and uniform velocity on `Lmax = 1, 2, 3`.
5. Shear canaries:
   Couette crossing a single interface, then nested interfaces.
6. Forced channel canaries:
   Poiseuille crossing one interface, then nested interfaces.
7. Macro-flow gates:
   BFS, square obstacle, cylinder convergence, then long aqua runs.

The current nested rest canary remains broken until the buffer contract is wired
into the actual transport integrator. Do not mask it by tuning route weights.

## Implementation Patch Log

Patch `2026-05-06-buffer-contract`:

- added `ConservativeTreeSubcycleBufferBank2D`;
- added per-level `owned`, `ghost_from_coarse`, `reflux_to_coarse`, and
  `restrict_to_parent` dense CPU buffers;
- added reset/store/restore/reflux/restriction/prolongation helpers;
- added surgical tests proving that the buffers are disjoint and conservative;
- intentionally did not change the production transport skeleton yet.
