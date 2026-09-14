# Conservative Fixed-Patch Refinement for LBM Using Integrated Populations

## Abstract Draft

We present a conservative fixed-patch refinement prototype for the D2Q9
lattice Boltzmann method. The method stores integrated populations
`F_i = f_i V` on a composite coarse/fine state: coarse cells remain active
outside the patch, fine cells are active inside the patch, and coarse parents
covered by fine cells are retained only as inactive ledgers. Projection to a
uniform leaf grid and restriction back to the composite state are conservative
per population orientation. A limited linear parent-to-leaf reconstruction is
used for composite projection because uniform conservative explosion preserves
global moments but destroys local velocity gradients and non-equilibrium
stress. The method is validated on Couette flow, Poiseuille refinement bands,
a square obstacle enclosed by the refined patch, and a backward-facing-step
velocity comparison.

## Method Draft

The conservative variable is the integrated population

```text
F_i = f_i V_cell
```

where `f_i` is the usual LBM density population and `V_cell` is the cell
volume. Collision is applied in density-population form by converting locally
from `F_i` to `f_i`, applying BGK or Guo forcing, and storing the result again
as `F_i`.

The composite state has three ownership classes:

- active coarse cells outside the refined patch;
- active fine children inside the refined patch;
- inactive coarse parent ledgers under the refined patch.

Diagnostics over the composite state count only active cells. The inactive
coarse ledgers are used for conservative aggregation and reconstruction, but
not as fluid degrees of freedom.

The validation path is intentionally explicit:

```text
composite state
  -> conservative projection to uniform leaf grid
  -> leaf-grid collision/streaming/boundary update
  -> conservative restriction to composite state
```

This is not yet native AMR streaming. It is a controlled oracle that isolates
the conservation and reconstruction behavior before introducing route-driven
composite streaming.

## Reconstruction Point

A uniform parent-to-four-children explosion is conservative:

```text
F_child = F_parent / 4
```

but it is not physically transparent. It erases local velocity gradients and
non-equilibrium stress information inside a parent block. This is visible in
the dedicated canary `uniform projection loses local velocity and stress
moments`.

The current composite projection therefore uses a limited linear
population-wise reconstruction. It preserves the parent integrated population
while allowing local variation across the four children. This is the key
reason the Poiseuille vertical and horizontal refinement bands remain
transparent at the current error level.

## Validation Draft

The reproducible command is:

```bash
julia --project=. scripts/figures/plot_voie_d_phase_p.jl
```

This command regenerates the paper figures and enforces the Phase P numerical
gate.

Current validation cases:

```text
Couette:
  central fixed patch, analytic profile

Poiseuille:
  vertical refinement band at x=L/2, analytic profile
  horizontal refinement band at y=L/2, analytic profile

Square obstacle:
  obstacle enclosed by refined patch
  drag/lift compared with a coarse Cartesian reference

BFS/VFS:
  open-channel backward-facing-step velocity field
  compared with a Cartesian leaf-grid reference
```

Current generated metrics are stored in:

```text
paper/data/voie_d_phase_p_summary.md
```

## Results Snapshot

Latest gate values:

```text
Couette L2:                     3.95e-4
Poiseuille vertical band L2:    1.198e-3
Poiseuille horizontal band L2:  1.126e-3
Square obstacle Fx ratio:       0.986428
BFS/VFS ux mean delta:          1.30e-4
```

These results support the limited claim that a conservative fixed patch can
remain transparent for canonical shear/pressure-driven flows and can enclose
a simple obstacle without large drag distortion.

## Limitations Draft

The current method does not claim:

- dynamic AMR adaptation;
- native `dx`-local composite streaming;
- temporal subcycling;
- optimized GPU AMR kernels;
- 3D octree support;
- cylinder validation for Phase P.

The next technical step is route-driven native composite streaming without
collision, compared against the current leaf-grid oracle.

