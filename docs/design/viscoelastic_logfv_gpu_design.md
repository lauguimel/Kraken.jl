# Viscoelastic Log-FV/FD GPU Design

*Status: 2026-05-08. Design document for a production-oriented Kraken
viscoelastic backend. The goal is a robust high-Weissenberg, low-beta
solver, not another Liu-cylinder-only patch.*

## 1. Decision

Build a new polymer backend:

```text
LBM solvent + cell-centered log-conformation FV/FD polymer solver
```

The existing population-LBM conformation modes remain useful, but they
must be treated as benchmark/audit modes:

```text
direct-C Liu Eq26        -> paper reproduction / audit
direct-C TRT/regularized -> diagnostic
log-FV/FD cell-centered  -> production candidate
```

The production path must not depend on a Liu-style polymer magic
parameter to remain stable.

## 2. Why this pivot

The direct-conformation population solver has two structural issues for
production high-Wi work:

1. It does not preserve positive definiteness of `C` by construction.
2. Wall treatment is expressed in missing populations rather than in
   physical conformation or stress variables.

At low viscosity ratio `beta < 0.1`, the polymer stress dominates:

```text
tau_p = (1 - beta) / Wi * (C - I)
```

Small errors in `C` therefore become large body-force errors. A robust
production backend must enforce or preserve SPD, use physical wall
operations, and keep the GPU kernels simple.

## 3. Governing equations

Use the Oldroyd-B conformation equation as the first target:

```text
D C / Dt = C * grad(u) + grad(u)' * C - (C - I) / lambda
```

with

```text
D / Dt = d/dt + u . grad
```

The polymer stress is

```text
tau_p = eta_p / lambda * (C - I)
```

or, in common nondimensional form,

```text
tau_p = (1 - beta) / Wi * (C - I)
```

The solver variable is

```text
Psi = log(C)
```

with `Psi` symmetric. Reconstruction is local:

```text
C = exp(Psi)
```

In 2D:

```text
Psi = [psi_xx psi_xy
       psi_xy psi_yy]
```

In 3D:

```text
Psi = [psi_xx psi_xy psi_xz
       psi_xy psi_yy psi_yz
       psi_xz psi_yz psi_zz]
```

3D comes only after the 2D patch ladder is green.

## 4. Time step pipeline

Baseline coupled step:

```text
1. Read solvent velocity u^n from LBM fields.
2. Compute grad(u^n).
3. Advect Psi^n with a cell-centered FV/FD advective operator.
4. Apply local stretching/rotation/relaxation source.
5. Reconstruct C^(n+1) = exp(Psi^(n+1)).
6. Compute tau_p^(n+1).
7. Compute Fp = div(tau_p).
8. Apply BSD correction if enabled.
9. Advance the solvent LBM with integrated forcing.
```

For the first implementation, use first-order operator splitting:

```text
advect Psi -> local source -> stress/force -> solvent step
```

Then add Strang splitting only if a canary shows splitting error is the
dominant defect:

```text
half source -> advection -> half source
```

## 5. Log-conformation source update

Two source kernels should be kept separate.

### Source mode A: exact local C update, then log

This is the recommended first production path for 2D because it is easy
to patch-test.

Given fixed local velocity gradient `L = grad(u)` during a substep:

```text
dC/dt = C * L + L' * C - (C - I) / lambda
```

Write

```text
M = L - I / (2lambda)
Q = I / lambda
```

then

```text
dC/dt = C * M + M' * C + Q
```

The homogeneous part is solved with a small matrix exponential. The
forcing integral can be handled by a fixed 2x2 closed form or by a
fixed-order quadrature validated against analytical cases. After the
local update:

```text
Psi_new = log(C_new)
```

This preserves SPD if the local update produces SPD. If a numerical
roundoff floor is needed, it must be explicit, logged, and tested as a
diagnostic path, not silently enabled.

### Source mode B: Fattal-Kupferman split

This is the literature-standard log-conformation source formulation. It
decomposes `grad(u)` in the eigenframe of `C` and evolves `Psi` directly.
It should be implemented after mode A is green, or earlier only if mode A
fails a low-level canary.

The implementation must be local, allocation-free, and use fixed 2x2
closed forms first. A 3x3 version is deferred.

## 6. Advection of Psi

The CDE uses an advective material derivative:

```text
d_t Psi + u . grad(Psi) = source
```

The discrete operator must preserve a constant `Psi` exactly, even if
the weakly-compressible LBM velocity has small numerical divergence.

Use a flux-difference form with a divergence correction:

```text
adv(Psi)_cell =
    (F_e - F_w) / dx + (F_n - F_s) / dy
    - Psi_cell * div_u_discrete
```

where face fluxes use the same face velocities as `div_u_discrete`.

Initial scheme order:

```text
P0: first-order upwind
P1: MUSCL with MC or van Leer limiter
P2: CUBISTA/WENO only after all low-level canaries are green
```

Do not start with CUBISTA. Start robust, then increase accuracy.

## 7. Low-beta stabilization: BSD in an LBM solvent

For `beta < 0.1`, using the physical solvent viscosity directly in LBM
can push the LBM relaxation too close to the stability limit. The solvent
LBM should support a Both-Sides-Diffusion style stabilization.

Let

```text
nu_total = nu_s + nu_p
nu_s     = beta * nu_total
nu_p     = (1 - beta) * nu_total
zeta     = BSD fraction, 0 <= zeta <= 1
```

Run the LBM collision with an effective viscosity

```text
nu_lbm = nu_s + zeta * nu_p
```

and add the compensating force

```text
F_total = div(tau_p) - zeta * nu_p * laplacian(u)
```

At the continuum level, the added diffusion cancels:

```text
nu_lbm * laplacian(u) + F_total
  = nu_s * laplacian(u) + div(tau_p)
```

This makes the momentum operator more elliptic while preserving the
target equation in the consistency limit.

Recommended defaults:

```text
beta >= 0.3  -> zeta = 0.0 initially
0.1 <= beta < 0.3 -> zeta = 0.5 diagnostic sweep
beta < 0.1  -> zeta = 1.0 as the production candidate
```

`zeta` is not a physical fitting knob. Every benchmark must report it,
and convergence must be checked with at least two `zeta` values before
claiming benchmark agreement.

Potential next stabilizations:

```text
iBSD
DEVSS-like velocity-gradient stabilization
velocity-stress coupling
```

Do not add these before plain BSD has analytical and coarse-flow
canaries.

## 8. Boundary handling

The polymer backend is cell-centered. It should not use missing
populations at walls.

Wall rules:

- No advective wall flux for impermeable walls: `u_n = 0`.
- Tangential wall velocity enters only through `grad(u)` stencils.
- For `Psi`, use zero-normal-gradient or physically prescribed wall
  extrapolation depending on the benchmark.
- For inlets, prescribe `Psi` or `C` from the known inflow state.
- For outlets, use convective/zero-gradient treatment and verify with a
  patch test before macro flows.

Near-wall gradients must use precomputed coefficients:

```text
dfdx[cell] = sum_k a[cell,k] * f[node[cell,k]]
            + sum_m aw[cell,m] * f_wall[wall_slot[cell,m]]
```

GPU kernels must not solve a least-squares problem or dynamically build
a stencil.

## 9. GPU data layout

Use structure-of-arrays. No `SMatrix` arrays in device memory.

2D fields:

```text
psi_xx, psi_xy, psi_yy
Cxx, Cxy, Cyy
tauxx, tauxy, tauyy
Fx, Fy
dudx, dudy, dvdx, dvdy
```

Optional scratch fields:

```text
psi_xx_tmp, psi_xy_tmp, psi_yy_tmp
limiter scratch or residual scratch
```

3D fields later:

```text
psi_xx, psi_xy, psi_xz, psi_yy, psi_yz, psi_zz
Cxx, Cxy, Cxz, Cyy, Cyz, Czz
```

Kernel rules:

- no allocation;
- no scalar indexing fallback;
- no runtime dispatch on abstract types;
- fixed loop bounds;
- branch-light wall handling;
- explicit Float32/Float64 behavior;
- backend compatible with CUDA first, then Metal smoke tests.

## 9.1 AMR and SLBM-paper compatibility

The first macro driver may run on a single uniform Cartesian grid, but
the backend must not be designed as a single-grid dead end. The
`SLBM-paper` branch is developing AMR-like infrastructure, so the
log-FV polymer path must stay patch-local. The intended AMR model is
Basilisk-style quadtree refinement in 2D: each refined parent cell maps
to four child cells at the next level.

Required constraints:

```text
all kernels operate on one rectangular patch at a time
dx, dy are explicit kernel arguments
level id and refinement ratio are explicit at wrapper/driver level
interior update excludes halo/ghost cells unless a kernel says otherwise
boundary, halo, prolongation, and restriction are separate operators
no hidden global Nx/Ny assumptions in polymer state structs
no dependence on LBM population storage for polymer state
```

For early uniform-grid canaries, this means:

```text
uniform grid = one patch with no refinement
physical boundaries = patch boundary operators
future AMR interfaces = quadtree patch exchange/prolong/restrict wrappers
```

Refinement-sensitive quantities:

- `Psi` must be prolongated/restricted in log space, not by silently
  reconstructing/clipping `C`.
- Parent-to-child prolongation is 1-to-4 per level in 2D. Child-to-parent
  restriction should preserve the cell average of `Psi` or of a documented
  conservative proxy; do not average stresses and call that the state.
- `tau_p` and `Fp = div(tau_p)` should be recomputed from `Psi` on each
  level after exchange when possible.
- Flux-form advection must eventually use conservative coarse/fine flux
  correction.
- BSD `laplacian(u)` must use level-local spacing and corrected coarse/fine
  boundary values.

Do not implement AMR in the first log-FV pass. Keep the interfaces shaped
so AMR can be added without rewriting the polymer kernels.

## 10. Kernel decomposition

Start with clear kernels before fusion:

```text
compute_velocity_gradient_2d!
advect_logconf_upwind_2d!
source_logconf_local_2d!
exp_logconf_stress_2d!
polymer_force_2d!
bsd_correction_force_2d!
lbm_fluid_step_with_polymer_force_2d!
```

After correctness:

```text
fuse exp_logconf_stress_2d! + polymer_force_2d! if memory-bound
fuse BSD correction into polymer_force_2d! if stencil reuse is clean
keep advection separate until limiter behavior is stable
```

Do not fuse before the analytical patch ladder can isolate each operator.

## 11. Proposed files

New implementation files:

```text
src/kernels/logconformation_fv_2d.jl
src/kernels/polymer_stencils_2d.jl
src/drivers/viscoelastic_logfv_2d.jl
```

Deferred:

```text
src/kernels/logconformation_fv_3d.jl
src/kernels/polymer_stencils_3d.jl
src/drivers/viscoelastic_logfv_3d.jl
```

Tests:

```text
test/test_viscoelastic_logfv_patch_ladder.jl
test/test_viscoelastic_logfv_gpu_smoke.jl
```

Benchmarks:

```text
bench/viscoelastic_logfv/poiseuille.jl
bench/viscoelastic_logfv/couette.jl
bench/viscoelastic_logfv/square_obstacle.jl
bench/viscoelastic_logfv/bfs.jl
bench/viscoelastic_logfv/liu_cylinder.jl
```

Documentation:

```text
docs/design/viscoelastic_logfv_gpu_design.md
docs/agent/branch_contract.md
docs/src/theory/15_viscoelastic.md      # update only after backend is real
```

## 12. Julia API and DSL shape

Internal Julia config:

```julia
struct LogFvPolymerSpec{T}
    model::Symbol              # :oldroydb first
    advection::Symbol          # :upwind, :muscl, :cubista
    source_update::Symbol      # :exact_c_then_log, :fattal_kupferman
    beta::T
    Wi::T
    bsd_fraction::T
    substeps::Int
    gradient_stencil::Symbol   # :bulk, :wall_aware, :precomputed
end
```

Driver keyword sketch:

```julia
run_viscoelastic_logfv_2d(;
    backend=CUDABackend(),
    polymer_model=:oldroydb,
    polymer_solver=:logfv,
    polymer_advection=:upwind,
    polymer_source=:exact_c_then_log,
    beta=0.1,
    Wi=1.0,
    bsd_fraction=1.0,
    polymer_substeps=1,
)
```

`.krk` shape, after the driver is stable:

```text
[physics]
mode = "viscoelastic"

[viscoelastic]
model = "oldroydb"
solver = "logfv"
beta = 0.1
Wi = 1.0
advection = "upwind"
source = "exact_c_then_log"
bsd_fraction = 1.0
polymer_substeps = 1
```

Until the backend is validated, this DSL must be documented as
development-only.

## 13. Patch-test ladder

No macro benchmark is allowed before these pass.

### M0: algebra

- `log(exp(Psi)) == Psi` for symmetric 2x2 matrices.
- `exp(Psi)` is SPD.
- eigenvectors/eigenvalues stable near repeated eigenvalues.
- symmetry is preserved.

### M1: pure relaxation

Set `u = 0`. Exact solution:

```text
C(t) = I + (C0 - I) * exp(-t / lambda)
```

Then compare both `C` and `Psi = log(C)`.

### M2: homogeneous simple shear

Use constant velocity gradient:

```text
grad(u) = [0 gamma_dot
           0 0]
```

Compare the local source kernel against an analytical or high-precision
reference ODE solve for one cell.

### M3: advection

Fixed velocity, no source:

- constant `Psi` must remain bit/exact or roundoff-exact;
- affine `Psi` should converge with the expected order;
- wall-impermeable patch must preserve constant `Psi`.

### M4: Poiseuille analytical

Start with prescribed steady velocity. Validate local `C` against the
known Oldroyd-B shear solution:

```text
Cxy = lambda * du/dy
Cxx = 1 + 2 * lambda^2 * (du/dy)^2
Cyy = 1
```

Then couple to the solvent and verify the Newtonian limit as `Wi -> 0`.

### M5: Couette analytical

Same checks as Poiseuille with constant shear.

### M6: frozen-flow square obstacle

Use a fixed Newtonian velocity field and solve only the polymer CDE. This
isolates advection/source/wall effects without force feedback.

### M7: coupled square obstacle

Verify:

- Newtonian limit;
- `beta` sweep;
- `Wi` sweep;
- BSD `zeta` sweep.

### M8: BFS

Use BFS before cylinder because geometry is Cartesian and easier to
localize.

### M9: cylinder Liu low Wi

Only after M0-M8. Validate Wi 0.1 and 0.5 first.

### M10: cylinder high Wi

Wi 1, 2, 3, 5, 10 only after low-Wi and low-beta canaries are green.

## 14. Acceptance gates

The backend is not production-ready until:

```text
1. M0-M5 pass on CPU and GPU.
2. M6-M8 pass on coarse grids with beta = 0.9, 0.5, 0.1, 0.01.
3. No silent SPD projection is used.
4. BSD zeta dependence is documented and converges with refinement.
5. Direct-C Liu mode remains available as benchmark/audit.
6. Metal local smoke tests pass.
7. A100/H100 Float64 validation passes for at least one macro flow.
```

## 15. Performance plan

Expected memory advantage in 2D:

```text
direct-C population LBM: 3 tensor components * 9 populations = 27 fields
log-FV polymer:          3 Psi fields + derived stress/force fields
```

The log-FV path spends more flops per cell but far less memory bandwidth
on polymer transport. On modern GPUs this is a reasonable trade.

Report:

```text
cells/sec for polymer step
MLUPS for coupled solvent+polymer
memory footprint per cell
number of kernels per time step
Float32 vs Float64 behavior
CUDA vs Metal smoke behavior
```

Do not compare Metal debug runs to A100/H100 production runs.

## 16. Development sequence

### Phase 0: branch contract

Create `docs/agent/branch_contract.md` with:

```text
production path = log-FV/FD polymer
benchmark path = Liu Eq26 direct-C
validation ladder = M0-M10 from this document
```

### Phase 1: local algebra and source

Implement only test helpers first:

```text
_logsym2x2
_expsym2x2
_oldroydb_relax_exact
_oldroydb_local_source_reference
```

Then promote stable helpers to `src/`.

### Phase 2: advection operator

Implement upwind advection of `Psi` with constant/affine canaries.

### Phase 3: GPU kernels

Port M0-M3 kernels to `KernelAbstractions.jl` and validate on CPU, Metal,
and CUDA where available.

### Phase 4: force and BSD

Implement `div(tau_p)`, `laplacian(u)`, and BSD correction. Validate with
manufactured fields before coupling to LBM.

### Phase 5: coupled coarse flows

Run Poiseuille, Couette, square, BFS. No cylinder yet.

### Phase 6: benchmark flows

Run Liu cylinder low Wi, then high Wi. Record exact parameters and
failure diagnostics.

### Phase 7: 3D

Port only after 2D M0-M10 are stable.

## 17. Known traps

- A good cylinder `Cd` does not prove the polymer operator is correct.
- A stable result obtained by tuning `bsd_fraction` on one case is not a
  production result.
- A stress cap or eigenvalue floor can be a diagnostic, but not a silent
  production fix.
- Log-conformation preserves SPD algebraically, but it does not prevent
  physical or numerical stress blow-up.
- Low beta weakens momentum ellipticity; log-conformation alone is not
  enough.
- Near-wall gradient quality matters more than benchmark drag suggests.
- 3D must not inherit an unvalidated 2D shortcut.

## 18. Literature anchors

- Fattal and Kupferman, 2005, second-order finite-difference
  log-conformation scheme for high-Weissenberg viscoelastic flows:
  <https://doi.org/10.1016/j.jnnfm.2004.12.003>
- Hulsen, Fattal, and Kupferman, 2005, cylinder simulations with matrix
  logarithms and DEVSS/DG context:
  <https://doi.org/10.1016/j.jnnfm.2005.01.002>
- Afonso, Oliveira, Pinho, and Alves, 2009, log-conformation in a
  finite-volume framework around confined cylinders:
  <https://doi.org/10.1016/j.jnnfm.2008.09.007>
- Habla et al., 2014, OpenFOAM collocated FVM log-conformation
  implementation with 3D validation:
  <https://doi.org/10.1016/j.jnnfm.2014.08.005>
- rheoTool, OpenFOAM viscoelastic toolbox and practical FVM reference:
  <https://github.com/fppimenta/rheoTool>
- Ke and Wang, 2024, low-viscosity-ratio log-conformation stabilization
  comparison including BSD, DEVSS-omega, and velocity-stress coupling:
  <https://doi.org/10.3390/math12030430>
- iBSD, improved Both-Sides-Diffusion stabilization:
  <https://doi.org/10.1016/j.jnnfm.2017.09.008>
