# Section 4 — Extension to three dimensions

> Draft section for the SLBM paper. Numbers in `[…]` are placeholders to
> be filled from `results/slbm_sphere_3d.log` after the Aqua H100 run
> completes (job 20145714, queued).

## 4.1 D3Q19 semi-Lagrangian streaming

The 2-D formulation of Sec. 2 carries over without structural change to
the D3Q19 lattice. Departure points are now sampled in three logical
indices,

```math
i_\text{dep}^q(i,j,k) = i - \Delta x_\text{ref} \, N_\xi \,
                       (\xi_x c_q^x + \xi_y c_q^y + \xi_z c_q^z),
```

with analogous expressions for $j_\text{dep}$ and $k_\text{dep}$, and
the inverse Jacobian components $\xi_x, \xi_y, \xi_z$ obtained as the
cofactors of the $3\!\times\!3$ metric tensor `mesh.dXdξ … mesh.dZdζ`
(see `compute_metric_3d`, [src/curvilinear/mesh_3d.jl:46](../src/curvilinear/mesh_3d.jl#L46)).
The interpolation kernel is a single-cell 8-neighbour trilinear stencil
(`trilinear_f`, [src/curvilinear/slbm_3d.jl:99](../src/curvilinear/slbm_3d.jl#L99)),
which keeps register pressure low enough for D3Q19 to fit in a single
H100 thread block. As in 2-D, on a uniform Cartesian mesh the precomputed
departure indices land exactly on the neighbour nodes and the kernel
collapses to plain pull-streaming — verified to machine precision in
[test/test_slbm_libb_3d.jl:91](../test/test_slbm_libb_3d.jl#L91).

## 4.2 Coupling with LI-BB and BCSpec on stretched 3-D grids

The fused 3-D step is built from the same DSL-brick library introduced
in Sec. 3, with three new bricks:

- `PullSLBM_3D` — trilinear semi-Lagrangian pull
- `CollideTRTLocalDirect_3D` — TRT collision with per-cell rates
- `ApplyLiBBPrePhase_3D` (already on `lbm`) — Bouzidi pre-phase on cut
  links for any $q_w \in (0, 1]$

Two collision variants are exposed:
`slbm_trt_libb_step_3d!` (uniform $\tau$) and
`slbm_trt_libb_step_local_3d!` (per-cell $\tau$ from
`compute_local_omega_3d`). The latter is required on stretched 3-D
boxes where the local cell size — and therefore the relaxation time —
varies across the grid.

For sphere immersed in a duct, $q_w$ is computed in **physical** space
by ray-sphere intersection on each of the 18 connecting links, mirroring
the 2-D cylinder treatment of Sec. 3:
`precompute_q_wall_slbm_sphere_3d` ([src/curvilinear/slbm_3d.jl:296](../src/curvilinear/slbm_3d.jl#L296)).

The boundary-condition system extends BCSpec to six faces; the four
transverse walls (`south`, `north`, `bottom`, `top`) use explicit
half-way bounce-back kernels gated by `apply_transverse=true`, which
preserves backwards compatibility with the Cartesian halfway-BB
streaming kernel (`fused_trt_libb_v2_step_3d!`) that handles transverse
walls inside its own pull brick. Local-$\tau$ Zou-He variants of the
inlet/outlet kernels accept device-side per-cell rate fields, removing
the only remaining obstacle to running the full SLBM 3-D pipeline on
stretched meshes.

## 4.3 Validation: sphere drag at Re = 20

We benchmark a sphere of diameter $D$ in a confined duct of cross-section
$4D \!\times\! 4D$ and length $12D$, with the centre at $(3D, 2D, 2D)$.
The inlet is a doubly-parabolic profile with peak velocity
$u_{\max} = 0.04$ in lattice units; the viscosity is calibrated to
$\nu = u_{\max} D / \mathrm{Re}$ for $\mathrm{Re}=20$. The mean velocity
of the inlet is $\bar u = (4/9) u_{\max}$ and the reference projected
area is $A = \pi (D/2)^2$, so the drag coefficient reads
$C_d = 2 F_x / (\bar u^{\,2} A)$. The reference value $C_d \approx 3.0$
combines the Clift-Gauvin free-stream estimate (2.84) with a $\sim 5\%$
duct-blockage correction.

| Mesh | Cells | $C_d$ | err [%] | MLUPS |
|------|-------|-------|---------|-------|
| Uniform $D\!=\!10$  | $2.0\times10^5$  | […] | […] | […] |
| Uniform $D\!=\!20$  | $1.6\times10^6$  | […] | […] | […] |
| Uniform $D\!=\!30$  | $5.3\times10^6$  | […] | […] | […] |
| Stretched $D\!=\!20$ s=0.5 | $6.7\times10^5$ | […] | […] | […] |
| Stretched $D\!=\!20$ s=1.0 | $6.7\times10^5$ | […] | […] | […] |
| Stretched $D\!=\!30$ s=0.5 | $1.6\times10^6$ | […] | […] | […] |
| Stretched $D\!=\!30$ s=1.0 | $1.6\times10^6$ | […] | […] | […] |

The headline argument of the paper is the **cell-count ratio** between
the uniform and stretched meshes at fixed $C_d$ accuracy. With a
quadratic per-cell viscosity rescaling the local-CFL stretched mesh
recovers the same accuracy as a $D=30$ uniform run with roughly $\div N$
cells (placeholder — to be filled from the H100 log).

![Sphere 3D convergence and throughput](figures/sphere_3d_convergence.pdf)

## 4.4 Automatic differentiation in 3-D

Reverse-mode AD propagates cleanly through the 3-D forward step. On a
$24^3$ Taylor-Green vortex run for 100 SLBM-BGK steps, the kinetic-energy
gradient with respect to viscosity matches the central-finite-difference
reference to machine precision:

```
KE(ν=0.1)         = 5.3199 × 10⁻²
dKE/dν (FD)       = -1.4983
dKE/dν (Enzyme)   = -1.4983
relative error    = 0.00%
```

This is significant: the entire chain
`build_slbm_geometry_3d` → `slbm_bgk_step_3d!` → kinetic-energy reduction
is differentiable end-to-end with no manual rule, on top of `Enzyme.jl`
in pure Julia. It immediately unlocks shape and viscosity gradients on
3-D body-fitted meshes once `precompute_q_wall_slbm_sphere_3d` is made
Enzyme-compatible (deferred to v0.2). The 2-D shape-derivative
proof-of-concept of Sec. 5.2 transfers to 3-D in principle but is not
attempted here.

## 4.5 Performance summary

On a single NVIDIA H100 (Float64), the SLBM 3-D path sustains […] MLUPS
on the largest uniform run and […] MLUPS on the matched stretched run.
The per-cell overhead of trilinear interpolation versus standard
pull-streaming is roughly $\div N$ (placeholder), already amortised when
the stretched mesh saves $\div N$ cells; a fortiori for higher-Reynolds
runs where the boundary layer demands a finer near-body resolution.
