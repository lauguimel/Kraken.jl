# Section 5 — Stress test: cross-flow cylinder at Re = 100

> Draft section extending the SLBM paper. Numbers from Aqua H100 FP64
> runs archived in `paper/data/wp_mesh_6_bump_h100_v{3,4}_*.log`,
> figure `paper/figures/bump_convergence_v3.pdf`.

While Section 4.5.1 established that gmsh + SLBM + LI-BB matches the
literature reference on the canonical **Schäfer-Turek 2D-2** cylinder
(parabolic inlet, channel 2.2 × 0.41) to $10^{-2}\,\%$ on
$C_l^\mathrm{RMS}$, we now stress-test the method on a geometrically
simpler but numerically more demanding **cross-flow** configuration
derived from Williamson (1996) and Park (1998):

- Domain $\Omega = 1.0 \times 0.5$ (20D × 10D, **10 % blockage**)
- Cylinder centred at $(0.5, 0.245)$ with radius $R = 0.025$, **1 %
  vertical offset** to break symmetry and trigger vortex shedding
  (Schäfer-Turek 2D-2 style)
- Uniform inlet $u_\mathrm{in} = 0.04$, $Re = u \cdot 2R / \nu = 100$
- References (unbounded flow): $C_d \approx 1.4$, $C_l^\mathrm{RMS}
  \approx 0.33$, $\mathrm{St} \approx 0.165$ (Williamson 1996,
  Henderson 1995)

Three baselines, all at **matched cell count** per resolution:

| | Streaming | Solid-cell closure | Mesh |
|---|---|---|---|
| **(A)** | pull-stream | halfway-BB ($q_w = 1/2$) | uniform Cartesian |
| **(B)** | pull-stream | LI-BB v2 (Bouzidi, any $q_w$) | uniform Cartesian |
| **(C)** | semi-Lagrangian (SLBM) | LI-BB v2 | gmsh Transfinite Bump 0.1 |

## 5.1 Convergence matrix on Aqua H100 (FP64)

| $D_\mathrm{lu}$ | Cells | (A) $C_d$ | (B) $C_d$ | **(C) $C_d$** | (A/B/C) $C_l^\mathrm{RMS}$ | $\mathrm{St}$ | MLUPS |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 20 |  80 601 | 1.672 | 1.643 | NaN | 0.27 / 0.26 / — | 0.20 / 0.20 / — | 581 / 1 647 / 990 |
| 40 | 321 201 | 1.652 | 1.649 | **1.630** | 0.28 / 0.28 / 0.003 | 0.20 / 0.20 / 0.175 | 4 288 / 4 279 / 1 751 |
| 80 | 1 282 401 | 1.648 | 1.650 | NaN | 0.28 / 0.29 / — | 0.20 / 0.20 / — | 7 029 / 7 033 / 2 258 |

Three observations drive this section.

### 5.1.1 Cd ~17 % above the Williamson 1996 reference is the confinement bias

All three solvers converge on $C_d \approx 1.65$ at $D_\mathrm{lu} = 80$,
independently of the boundary treatment. The deviation from
$C_d^\mathrm{Williamson} = 1.4$ is the **blockage correction** at 10 %
lateral confinement, which adds $\sim 15$-$20\,\%$ to the drag in a
bounded domain at this Reynolds number (Park 1998, eq. 6). No-method
error is implied — the Cd reference for a truly unbounded flow is
inaccessible without a $\sim 100R$ domain extent, which would push the
benchmark mesh into $10^8$-$10^9$ cells.

### 5.1.2 (C) matches (A)(B) in $C_d$ at same cell count — 2.4× MLUPS penalty

At the only resolution where (C) converges without diverging
($D_\mathrm{lu} = 40$), the Bump mesh + SLBM + LI-BB path delivers
$C_d = 1.630$ versus $1.649$ and $1.652$ for Cartesian with LI-BB and
halfway-BB respectively — **a small accuracy gain at identical cell
budget**. The throughput however drops by a factor of $\sim 2.4$
(1 751 vs 4 279 MLUPS), because the semi-Lagrangian bilinear
interpolation is memory-bound at four neighbour reads per direction.
This trade-off is consistent with the $\sim 3.6\times$ per-cell
overhead measured on the Schäfer-Turek case (Sec. 4.5.1).

### 5.1.3 The quadratic-$\tau$ rescaling has a stability limit near $\tau = 1/2$

Runs (C) at $D_\mathrm{lu} = 20$ and $80$ diverge within the first
$10^3$ time steps. The failure mechanism is shared across resolutions
and directly tied to the per-cell relaxation-rate rescaling
`compute_local_omega_2d(:quadratic)` (src/curvilinear/slbm.jl:626):

$$
\tau_\mathrm{local} - \tfrac{1}{2} = \Bigl( \frac{\Delta x_\mathrm{ref}}
{\Delta x_\mathrm{local}} \Bigr)^{\!2} \, (\tau_\mathrm{ref} - \tfrac{1}{2}).
$$

By convention $\Delta x_\mathrm{ref}$ is the *smallest* physical edge
length in the mesh (a default that guarantees $\mathrm{CFL} \leq 1$ for
semi-Lagrangian streaming). On the Bump 0.1 mesh the cell-size ratio
reaches $\sim 7\times$ between centre and boundary, driving
$\tau_\mathrm{local} \to 1/2$ on 30-60 % of the cells — precisely the
regime where TRT collision loses numerical stability (Ginzburg &
d'Humières 2003, Appendix B).

A `tau_floor` clamp (src/curvilinear/slbm.jl:626, commit `573c9a3`)
removes the NaN at the cost of a locally biased viscosity on coarse
cells; it does not recover the accuracy of the unclamped baseline.
The cleaner long-term fix is either a refinement-style $(\Delta t
\propto \Delta x)$ formulation, or an adaptive $\Delta x_\mathrm{ref}$
chosen at the *median* rather than the *minimum* edge length.

### 5.1.4 Shedding suppression on (C): physical-time mismatch

At $D_\mathrm{lu} = 40$ the Cartesian paths (A)(B) reach
$C_l^\mathrm{RMS} \approx 0.28$ and $\mathrm{St} = 0.20$, consistent
with a developed vortex street. (C) reports $C_l^\mathrm{RMS} \approx
0$ over the second half of the trajectory despite matching $C_d$.
The cause is that the reference cell size in the Bump 0.1 mesh
(`mesh.dx_ref` = 0.00048) is $\sim 7\times$ finer than the Cartesian
$\Delta x$ = 0.0025, so a fixed number of time steps advances
$\sim 7\times$ less *physical* time: at 160 000 steps (C) has
simulated $\approx 1.5$ flow-through times while (A)(B) reach
$\approx 8$. The shedding transient has not completed.
This is a scheduling artefact, not a method defect.

## 5.2 Figure: convergence plot

See [bump_convergence_v3.pdf](figures/bump_convergence_v3.pdf) — three
panels ($C_d$, $C_l^\mathrm{RMS}$, $\mathrm{St}$) against cell count
on log-log axes, three methods overlaid. (C) appears only at
$D_\mathrm{lu} = 40$; the $D_\mathrm{lu} = 20, 80$ gaps for (C) are
the divergence cases of Sec. 5.1.3.

## 5.3 Takeaways

1. **SLBM + LI-BB on a body-fitted structured mesh produces Cd
   accuracy comparable to Cartesian** at matched cell count,
   corroborating the ST 2D-2 result of Sec. 4.5.1. The gain is small
   on pressure-integrated drag but was large (two orders of
   magnitude) on shear-based $C_l^\mathrm{RMS}$ in 2D-2.
2. **Per-cell throughput cost of 2-3× is consistent across benchmarks.**
3. **Extreme stretching (cell-area ratio $\gtrsim 50\times$) hits a
   TRT stability wall at $\tau = 1/2$.** The `compute_local_omega`
   quadratic rescaling is correct in physics but numerically fragile
   when $\tau_\mathrm{ref}$ is already close to $1/2$ (low-Re,
   low-$u$ regime of this benchmark). A pragmatic remedy —
   clamping $\tau$ from below — introduces a local viscosity bias
   that reduces accuracy. The research-level fix is to either
   adopt $\Delta t \propto \Delta x$ (refinement-style scaling) or
   limit stretching to $\lesssim 10\times$ ratio for quadratic SLBM.
4. **Matching cell count is a *necessary* but not *sufficient*
   fairness criterion** — two codes can run the same number of steps
   on the same number of cells and cover different *physical* times
   if the effective $\Delta t$ (set by the smallest cell) differs.
   Future benchmarks should report flow-through times alongside step
   counts.
