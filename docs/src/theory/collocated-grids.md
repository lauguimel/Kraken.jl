# Collocated Grids

## What is this?

In CFD, you have to decide **where on the grid** to store each variable (velocity, pressure). There are two main approaches: staggered grids (different variables at different locations) and collocated grids (everything at the same location). Kraken uses collocated grids because they are simpler to implement and much better suited for GPU computing.

## Staggered grids (MAC method)

The original approach, introduced by Harlow and Welch (1965), stores variables at different locations:

- Pressure ``p``: at **cell centers**
- Horizontal velocity ``u``: at **vertical cell faces** (between cells in ``x``)
- Vertical velocity ``v``: at **horizontal cell faces** (between cells in ``y``)

This arrangement is called the **Marker-And-Cell (MAC)** grid. Its key advantage: the pressure gradient and velocity divergence are naturally computed at the correct locations using central differences, which gives strong pressure-velocity coupling and avoids the checkerboard problem (see below).

The drawback: different variables live on different grids with different indexing, which makes the code more complex. On GPUs, this means non-coalesced memory accesses (neighboring velocity components are not contiguous in memory), reducing performance.

## Collocated grids

A collocated grid stores **all variables at cell centers**:

- ``u_{i,j}``, ``v_{i,j}``, ``p_{i,j}`` are all defined at the same point ``(x_i, y_j)``

This simplifies everything:
- One indexing system for all variables
- Contiguous memory layout: ``u[i,j]`` and ``v[i,j]`` are stored together, perfect for GPU memory coalescing
- Boundary conditions are applied uniformly
- Extending to new variables (temperature, concentration) requires no new grid

## The checkerboard problem

Collocated grids have one well-known issue. When computing ``\nabla p`` with central differences, the pressure at point ``(i,j)`` only "sees" points ``(i \pm 1, j)`` and ``(i, j \pm 1)``, skipping itself. This means two independent pressure fields — one on even grid points, one on odd — can coexist without influencing each other:

```math
\frac{\partial p}{\partial x}\bigg|_i \approx \frac{p_{i+1} - p_{i-1}}{2h}
```

A pressure field like ``p_{i,j} = (-1)^{i+j}`` (checkerboard pattern) has zero gradient everywhere in the discrete sense, even though the true gradient is enormous. This decoupled mode is unphysical and can corrupt the solution.

On staggered grids, this problem does not arise because the pressure gradient naturally couples adjacent cells.

## Rhie-Chow interpolation

The standard fix for collocated grids is **Rhie-Chow interpolation** (1983). The idea: when computing the mass flux at a cell face (needed for the divergence constraint), use a special interpolation that includes the pressure gradient. This re-couples the pressure field and eliminates checkerboard modes.

At a cell face ``i+1/2``, the face velocity is:

```math
u_{i+1/2} = \frac{u_i + u_{i+1}}{2} - c_d\left[\left(\frac{\partial p}{\partial x}\right)_{i+1/2} - \overline{\left(\frac{\partial p}{\partial x}\right)}_{i+1/2}\right]
```

where the first term is simple linear interpolation, and the correction term is the difference between the face pressure gradient (compact stencil, ``(p_{i+1} - p_i)/h``) and the interpolated cell-center pressure gradients. The coefficient ``c_d`` involves the cell volume and momentum equation coefficients.

The key insight: the compact face gradient ``(p_{i+1} - p_i)/h`` couples adjacent cells, while the interpolated cell-center gradients ``(p_{i+1} - p_{i-1})/2h`` skip cells. Their difference adds just enough coupling to suppress checkerboard modes.

## Why collocated for GPU?

The choice is driven by memory access patterns:

| Aspect | Staggered | Collocated |
|--------|-----------|------------|
| Memory layout | 3 separate arrays with offsets | Aligned arrays, same indexing |
| GPU coalescing | Poor (offset access patterns) | Excellent (contiguous) |
| Code complexity | High (3 index systems) | Low (1 index system) |
| Pressure coupling | Natural | Needs Rhie-Chow |
| Used by | OpenFOAM (partially), academic | Fluent, WaterLily.jl, Kraken |

Modern GPU-oriented CFD codes (WaterLily.jl, Fluent, many LBM codes) overwhelmingly choose collocated grids. The slight complexity of Rhie-Chow is far outweighed by the GPU performance gains and code simplicity.

## Implementation in Kraken.jl

Kraken stores all fields (``u``, ``v``, ``p``) as 2D arrays with the same dimensions, including a layer of ghost cells for boundary conditions (see [Boundary Conditions](@ref)). All operators ([`laplacian!`](@ref), [`gradient!`](@ref), [`divergence!`](@ref), [`advect!`](@ref)) use the same ``(i,j)`` indexing.

The pressure-velocity coupling in the [projection method](@ref "Projection Method") currently uses a simple approach. Full Rhie-Chow interpolation will be added alongside the PISO algorithm in a future release.

## References

- C.M. Rhie and W.L. Chow, "Numerical study of the turbulent flow past an airfoil with trailing edge separation," *AIAA Journal*, 21:1525-1532, 1983.
- F.H. Harlow and J.E. Welch, "Numerical calculation of time-dependent viscous incompressible flow of fluid with free surface," *Phys. Fluids*, 8:2182-2189, 1965.
- J.H. Ferziger, M. Perić, and R.L. Street, *Computational Methods for Fluid Dynamics*, 4th ed., Springer, 2020, Ch. 7.
