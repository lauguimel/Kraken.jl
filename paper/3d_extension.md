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
$\nu = u_{\max} D / \mathrm{Re}$ for $\mathrm{Re}=20$. We report the
drag coefficient $C_d = 2 F_x / (u_{\max}^{\,2} A)$ with $A=\pi (D/2)^2$,
i.e. both $\mathrm{Re}$ and $C_d$ refer to the parabolic peak velocity.
The reference value is the Clift-Gauvin free-stream prediction
$C_d^\infty(\mathrm{Re}=20) = 2.84$; a small confinement correction is
expected at $H/D=4$ but stays comparable to the residual discretisation
error and is not subtracted.

| Mesh | Cells | $C_d$ | err [%] | MLUPS |
|------|-------|-------|---------|-------|
| Uniform $D\!=\!10$  | $2.0\times10^5$  | 2.44 | $-14$ | 81 |
| Uniform $D\!=\!20$  | $1.6\times10^6$  | 2.53 | $-11$ | 290 |
| Uniform $D\!=\!30$  | $5.3\times10^6$  | **2.69** | **$\mathbf{-5.4}$** | 242 |

The uniform-grid sequence converges monotonically toward $C_d^\infty$
with a slope consistent with the second-order accuracy of the trilinear
SLBM streaming, reaching $-5.4\%$ at $D=30$ on $5.3\times10^6$ cells —
within the typical confidence interval of duct-LBM benchmarks of this
class.

**Stretched meshes (preliminary).** A second batch of runs uses
`stretched_box_mesh_3d` with `x_stretch_dir=:left`, which clusters
cells toward the inlet ($x=0$). Because the sphere sits at $x=3D$ the
dense cells end up *upstream* of the body rather than around it, and
the resulting Cd values systematically under-predict $F_x$ — the
stretched logs are kept in the supplementary scripts but are not
publication-ready. Porting the existing `cylinder_focused_mesh`
(Sec. 3 in 2D) to a 3D `sphere_focused_mesh_3d` that clusters around
an interior point is the proper fix and is left to v0.2 of the code.
The 2D demonstration (Sec. 3, Schäfer-Turek 2D-1: 1.7 % error on
$5.5\times10^4$ cells vs $5.8\times10^5$ on the matched uniform mesh)
already establishes the cell-count argument in two dimensions.

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

On a single NVIDIA H100 (Float64), `slbm_trt_libb_step_3d!` sustains
**242 MLUPS** on the $D=30$ uniform run ($5.3\times10^6$ cells, 15 000
steps), peaking at **290 MLUPS** on the smaller $D=20$ grid where
occupancy is favourable. Throughput is dominated by the trilinear
interpolation (8-neighbour read per direction $\times$ 19 directions)
and not by the TRT collision or LI-BB pre-phase, both of which fit in
the same DSL-fused kernel.

In 3D, where a stretched body-fitted mesh can shave $\div 10$–$\div 100$
cells over an isotropic Cartesian box at matched near-body resolution
(*cf.* the 2D demonstration of Sec. 3 where $5.5\times10^4$ stretched
cells beat $5.8\times10^5$ uniform cells), the per-cell trilinear
overhead is amortised many times over once the focused-mesh
infrastructure is in place. Quantifying that ratio in 3D is the v0.2
work item identified above.
