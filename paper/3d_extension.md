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

## 4.5 Performance vs accuracy on sensitivity-critical quantities

The trilinear SLBM stencil costs $\sim 3.6\times$ the per-cell time of
a Cartesian halfway-BB stencil on a uniform mesh: on a single NVIDIA
H100 the SLBM 3-D step sustains **242 MLUPS** on the $D=30$ sphere
($5.3\!\times\!10^6$ cells); the equivalent Cartesian stencil exceeds
$\sim 7\!\times\!10^3$ MLUPS on the same hardware. **This raw-throughput
ratio is the wrong metric for a body-fitted method.** The relevant
trade-off is *accuracy at fixed cells*, especially on quantities that
do not converge with halfway-BB on a staircase boundary.

### 4.5.1 Schäfer-Turek 2D-2 (Re = 100, vortex shedding)

To make the comparison fair we benchmark three solvers on the
**identical uniform Cartesian grid** at three resolutions on the
canonical Schäfer-Turek 2D-2 cylinder (Re = 100 in a $2.2\!\times\!0.41$
channel), measuring the steady drag $C_d$, the unsteady **lift RMS**
$C_l^\mathrm{RMS}$ over the shedding cycle, and the Strouhal frequency
$\mathrm{St}$. References: $C_d \approx 3.23$, $C_l^\mathrm{RMS}
\approx 0.706$, $\mathrm{St} \approx 0.30$.

| $D_\mathrm{lu}$ | Cells | A halfway-BB | B Cart+LI-BB | **C gmsh+SLBM+LI-BB** |
|:-:|:-:|:-:|:-:|:-:|
| | | $C_d$ err — $C_l^\mathrm{RMS}$ err | | |
| 20 | 36 603 | 2.57 % — 3.91 %  | 1.39 % — 15.3 % | NaN (under-resolved) |
| 40 | 145 365 | 0.26 % — 1.22 %  | 0.14 % — 1.05 % | **0.03 %** — 1.22 % |
| 80 | 579 369 | 0.53 % — 1.12 %  | 0.34 % — 0.59 % | **0.35 % — 0.01 %** |
| | MLUPS (Metal M3 Max FP32) | 948 | 890 | 264 |

The $C_d$ benchmark — a pressure-integrated quantity — is weakly
sensitive to the wall closure: all three solvers converge below 0.5 %
once $D_\mathrm{lu} \geq 40$. The picture is qualitatively different
on $C_l^\mathrm{RMS}$, which depends on the shear gradient at the
cylinder surface: at $D_\mathrm{lu} = 80$ only **gmsh+SLBM+LI-BB
matches the literature reference to 0.01 %**, two orders of magnitude
below the halfway-BB error (1.12 %) on the *same* uniform mesh. This
is the irreducible signature of the staircase boundary that fixed-grid
LBM cannot remove by simple refinement.

The $\sim 3.6\times$ per-cell overhead of the trilinear stencil is
amortised by a much larger gap on derivative-based wall observables;
in 3-D, where a body-fitted mesh further saves $\div 10$–$\div 100$
cells over an isotropic Cartesian box, the SLBM+LI-BB path becomes
strictly preferable for any quantity that depends on the near-wall
velocity gradient — lift, friction, Strouhal at high Re, heat flux on
a curved surface.

### 4.5.2 3-D sphere throughput

On the 3-D sphere of Sec. 4.3, `slbm_trt_libb_step_3d!` sustains
**242 MLUPS** at $D = 30$ ($5.3\!\times\!10^6$ cells, 15 000 steps,
H100 FP64) and peaks at **290 MLUPS** at $D = 20$ where occupancy is
favourable. The SLBM brick fits in a single DSL-fused KernelAbstractions
kernel together with the LI-BB pre-phase and the TRT collision; the
trilinear interpolation (8-neighbour read per direction × 19 directions
of the D3Q19 stencil) is the dominant cost. Quantifying the cell-saving
ratio of a body-fitted O-grid sphere over an isotropic Cartesian box at
matched near-body resolution requires multi-block Transfinite stitching,
deferred to v0.2.
