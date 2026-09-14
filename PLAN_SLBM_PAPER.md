# Plan v0.1 paper — SLBM + LI-BB + AD on GPU

Branch: `slbm-paper` (from `lbm`)
Target: JCP (Journal of Computational Physics)
Timeline: 14 weeks (~3.5 months)

## Paper title (draft)

> Semi-Lagrangian Lattice Boltzmann on body-fitted stretched grids
> with Ginzburg-exact interpolated bounce-back and automatic
> differentiation on GPU

## 3 novel contributions

1. **SLBM + LI-BB** — no publication combines semi-Lagrangian streaming
   with interpolated bounce-back on curvilinear meshes
2. **SLBM 3D GPU** — first published D3Q19 semi-Lagrangian LBM on GPU
3. **AD through SLBM** — shape derivatives via Enzyme.jl on stretched
   grids (differentiable body-fitted LBM)

## Measured performance baseline

- SLBM kernel: **~20x slower per cell** than standard LBM (interpolation)
- 2D stretched: div 100 cells, div 20 per-cell → **5x net speedup**
- 3D stretched: div 1000 cells, div 20 per-cell → **~50x net speedup**
- The paper argument is 3D-centric: geometry beats compute cost

---

## What EXISTS (on branch lbm)

| Module | Files | Tests | GPU |
|--------|-------|-------|-----|
| SLBM 2D BGK bilinear + biquadratic | src/curvilinear/slbm.jl (592L) | 3 files, 823L | KA ok |
| SLBM 2D MRT | slbm.jl | ok | KA ok |
| SLBM 2D moving-wall (Ladd) | slbm.jl | ok | KA ok |
| Curvilinear mesh + ForwardDiff metric | mesh.jl + generators.jl (365L) | test_curvilinear_mesh.jl | CPU |
| Polar + stretched_box + cartesian mesh | generators.jl | ok | — |
| LI-BB V2 2D Ginzburg-exact | li_bb_2d_v2.jl + DSL bricks | 18k+ tests | KA ok |
| LI-BB V2 3D D3Q19 | li_bb_3d_v2.jl | ok | KA ok |
| BCSpec modular 2D/3D | boundary_rebuild.jl | ok | KA ok |
| GPU drag cut-link | drag_gpu.jl | ok | KA ok |
| STL -> q_wall 2D/3D sub-cell | stl_libb.jl | ok | CPU precompute |

## What NEEDS BUILDING — 8 work packages

### WP1 — SLBM + LI-BB 2D (the core coupling)
**Goal**: Compute q_wall in physical space via the mesh Jacobian and
inject it into the SLBM streaming kernel.

Steps:
- [ ] For each cut link (i,j,q), map the link endpoints to physical
      coords using CurvilinearMesh.(X,Y) and the metric
- [ ] Intersect the physical-space link with the body surface (analytic
      or STL) to get the physical distance d_phys
- [ ] Convert d_phys to computational-space fraction q_w = d_phys / link_phys_length
- [ ] Store q_wall[i,j,q] array (same layout as existing LI-BB V2)
- [ ] Modify slbm_bgk_step_kernel! to use the LI-BB pre-phase brick
      (ApplyLiBBPrePhase equivalent) before collision for cut links
- [ ] Validate on Taylor-Couette (analytic q_w on O-grid)

Effort: 1-2 weeks
Depends on: nothing (parallel start)
Test: Taylor-Couette L2 < 0.5% on O-grid

### WP2 — SLBM + BCSpec 2D (inlet/outlet on curvilinear mesh)
**Goal**: Apply Zou-He velocity/pressure BCs on mesh boundaries where
face normals are not axis-aligned.

Steps:
- [ ] Map BCSpec face selection (west/east/south/north) to mesh boundary
      rows (j=1, j=Ny, i=1, i=Nx in computational space)
- [ ] Transform the prescribed velocity profile from physical to
      computational coordinates
- [ ] Apply the existing boundary_rebuild kernel on the SLBM f arrays
      (the collision is local, so the same TRT-collide-local works)
- [ ] Validate: Poiseuille on stretched_box with parabolic inlet

Effort: 1 week
Depends on: WP1 (shared mesh infrastructure)
Test: Poiseuille L_inf < 1% on stretched mesh

### WP3 — SLBM + TRT 2D (upgrade from BGK)
**Goal**: Replace BGK collision in SLBM kernel with TRT (CollideTRTDirect),
needed for Ginzburg-exact LI-BB coupling.

Steps:
- [ ] Add slbm_trt_step_kernel! with s_plus/s_minus parameters
- [ ] Use trt_rates(nu; Lambda=3/16) for the rate computation
- [ ] Validate: same Poiseuille test, verify nu_eff / nu_in ~ 1.0

Effort: 3-5 days
Depends on: nothing
Test: Poiseuille nu_eff error < 5%

### WP4 — SLBM 3D D3Q19 (the big payoff)
**Goal**: Port the 2D SLBM to 3D with trilinear interpolation and 3D
metric tensors.

Steps:
- [ ] CurvilinearMesh3D struct with 3D Jacobian (3x3 metric tensor)
- [ ] build_mesh_3d with ForwardDiff on (xi,eta,zeta) -> (X,Y,Z)
- [ ] Stretched box 3D generator (tanh stretching per axis)
- [ ] build_slbm_geometry_3d: departure points (i_dep,j_dep,k_dep) for
      19 directions using inverse Jacobian
- [ ] trilinear_f interpolation helper (8 neighbors)
- [ ] slbm_bgk_step_3d! kernel (fused stream+collide)
- [ ] transfer_slbm_geometry_3d for GPU
- [ ] Validate: 3D Poiseuille (pipe), SLBM cartesian == standard LBM

Effort: 2-3 weeks
Depends on: nothing (parallel start)
Test: Poiseuille 3D L_inf < 1%, bit-exact on uniform cartesian

### WP5 — SLBM 3D + LI-BB (curved walls in 3D)
**Goal**: Combine WP4 (3D SLBM) with LI-BB V2 for curved geometries.

Steps:
- [ ] q_wall computation in 3D physical space (extend WP1 to 3D metric)
- [ ] STL sub-cell intersection in physical coordinates
- [ ] LI-BB pre-phase in the 3D SLBM kernel
- [ ] GPU drag on SLBM 3D (reuse cut-link list)
- [ ] Validate: sphere drag on stretched O-grid, compare to Clift et al.

Effort: 1-2 weeks
Depends on: WP4 + WP1
Test: Sphere Cd within 5% of Clift at Re=20

### WP6 — AD proof-of-concept (GO/NO-GO)
**Goal**: Differentiate slbm_bgk_step! with Enzyme.jl on CPU.
Compute dCd/d_nu on a simple Poiseuille and compare to finite differences.

Steps:
- [ ] Add Enzyme.jl as optional dependency
- [ ] Write a minimal forward pass: 100 steps SLBM BGK -> compute Cd
- [ ] Call Enzyme.autodiff(Reverse, forward_pass, ...) for d_Cd/d_nu
- [ ] Compare to (Cd(nu+eps) - Cd(nu-eps)) / (2*eps)
- [ ] If it works: document which Enzyme version, any workarounds

Effort: 1 week
Depends on: nothing (parallel start)
Test: AD gradient vs finite diff relative error < 1e-4
**THIS IS THE GO/NO-GO FOR THE AD ANGLE (week 3)**

### WP7 — AD on GPU (Enzyme + KernelAbstractions)
**Goal**: Same as WP6 but on CUDA backend.

Steps:
- [ ] Test Enzyme.autodiff on a CUDA KA kernel (minimal: just BGK)
- [ ] If direct doesn't work: try EnzymeCore rules for the interpolation
- [ ] Benchmark: AD overhead vs forward-only

Effort: 1-2 weeks
Depends on: WP6 passing
Test: GPU AD gradient matches CPU AD gradient

### WP8 — AD shape derivatives (the flagship result)
**Goal**: Differentiate Cd with respect to mesh node positions.
This gives shape sensitivities: how does moving the surface change the drag?

Steps:
- [ ] Parameterize the mesh mapping (e.g., cylinder radius R as a scalar)
- [ ] Compute d_Cd/d_R through the full chain:
      mesh(R) -> Jacobian -> departure points -> SLBM steps -> Cd
- [ ] Validate vs finite differences on R
- [ ] Demonstrate: simple shape optimization (find R that minimizes Cd)
      using gradient descent on the AD-computed sensitivity
- [ ] Extension: mesh node perturbation (not just scalar R)

Effort: 2-3 weeks
Depends on: WP7 + WP1
Test: d_Cd/d_R vs finite diff < 1e-3

---

## Benchmarks for the paper

### Validation benchmarks (6 figures)

| # | Case | Proves | Reference | WP |
|---|------|--------|-----------|-----|
| B1 | Poiseuille on stretched grid | SLBM convergence, exact on cartesian | analytic | WP1 |
| B2 | Taylor-Couette on O-grid + LI-BB | SLBM + LI-BB on curved geometry | analytic | WP1 |
| B3 | Schafer-Turek 2D-1 on O-grid | Cd <1% with 10x fewer cells | ST 1996 | WP1+2+3 |
| B4 | 3D sphere on stretched grid | First SLBM 3D GPU, 50x speedup | Clift | WP4+5 |
| B5 | dCd/d_nu on Poiseuille SLBM | AD validation (Enzyme vs fin. diff.) | exact | WP6 |
| B6 | dCd/d_shape on cylinder O-grid | Shape derivative (flagship) | fin. diff. | WP8 |

### Performance figures (3 figures)

| # | Figure | Message |
|---|--------|---------|
| F1 | MLUPS vs N: SLBM stretched vs uniform (Metal + H100) | 2D: 5x, 3D: 50x net speedup |
| F2 | L2 convergence: bilinear O(dx^2) vs biquadratic O(dx^3) | Interpolation order matters on distorted grids |
| F3 | Cd convergence: SLBM+LI-BB vs LBM+LI-BB (same phys. resolution) | Stretched mesh converges with fewer cells |

---

## Test matrix (CI / pre-merge validation)

For each SLBM kernel (2D BGK, 2D TRT, 2D MRT, 3D BGK):

| Backend | Precision | Required |
|---------|-----------|----------|
| CPU | Float64 | yes |
| CPU | Float32 | yes |
| Metal | Float32 | yes (local M3 Max) |
| CUDA | Float64 | yes (Aqua H100) |
| CUDA | Float32 | yes (Aqua H100) |

Test cases per backend x precision:

| Test | 2D | 3D | Threshold |
|------|----|----|-----------|
| Poiseuille stretched L_inf | x | x | < 1% |
| Couette stretched L_inf | x | x | < 0.1% |
| Taylor-Green decay (SLBM vs std) | x | — | bit-exact on cartesian |
| Taylor-Couette O-grid + LI-BB | x | — | < 0.5% |
| Mass conservation (1000 steps) | x | x | < 1e-10 |
| SLBM cartesian == LBM standard | x | x | .== (bit-exact) |
| AD gradient vs finite diff | x | — | rel err < 1e-4 |

---

## Paper structure (JCP, ~15 pages)

1. Introduction (1.5p) — LBM limited by uniform grids; SLBM exists but
   no IBB, no 3D GPU, no AD
2. Semi-Lagrangian LBM formulation (2p) — streaming interpolation,
   bilinear/biquadratic, CurvilinearMesh, metric
3. Coupling with LI-BB V2 (2p) — q_wall in physical space, Ginzburg
   TRT magic parameter, sub-cell accuracy on curved walls
4. Extension to 3D (1.5p) — D3Q19 trilinear, 3D metric tensor,
   stretched grid generators
5. Automatic differentiation (2p) — Enzyme.jl, differentiation through
   Jacobian -> departure points -> interpolation -> collision,
   shape derivatives
6. Validation (3p) — B1-B6 benchmarks with convergence plots
7. Performance (1.5p) — F1-F3, MLUPS scaling, 50x 3D argument
8. Demonstration: shape optimization (1p) — gradient descent on
   cylinder shape using AD-computed sensitivities
9. Conclusion (0.5p) — open-source MIT, code availability, roadmap

---

## Timeline

| Week | Front 1: SLBM+LI-BB 2D | Front 2: 3D | Front 3: AD |
|------|-------------------------|-------------|-------------|
| S1-S2 | **WP1**: q_wall physical space | **WP4** start: D3Q19 SLBM | **WP6**: Enzyme PoC CPU |
| S3 | **WP2**: BCSpec + **WP3**: TRT | WP4 continues | WP6 result → **GO/NO-GO** |
| S4-S5 | **B2** Taylor-Couette, **B3** ST 2D-1 | **WP5**: LI-BB 3D | **WP7**: Enzyme GPU |
| S6-S7 | Perf benchmarks F1-F3 | **B4**: sphere 3D | **WP8**: shape derivatives |
| S8-S9 | Test matrix (all backends) | Test matrix 3D | **B5-B6**: AD validation |
| S10-S14 | Paper writing |||

## Merge criteria (slbm-paper -> lbm)

- [ ] All WP1-WP5 tests pass on CPU + Metal
- [ ] B1-B4 benchmarks produce publication-quality figures
- [ ] AD (WP6-WP8) either fully integrated or cleanly scoped out
- [ ] No regression on existing 18k+ tests
- [ ] CUDA validation on Aqua H100 for key benchmarks
