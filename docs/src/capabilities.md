# Capabilities matrix

This page is the single-source-of-truth for **what Kraken.jl can do today**,
what is partially implemented, and what is out of scope for v0.1.0. Every
entry links to the theory page that derives it, the example that exercises
it, and the API or .krk reference that documents how to call it.

If a row says **in v0.1.0** it is part of the shipped feature set. Rows
marked **code present, v0.2.0** exist in the codebase (and in some cases are
fully tested) but are deliberately excluded from the v0.1.0 publication
scope — they will be documented and validated in a later release.

Legend: ✓ = works and tested · ~ = implemented, partial validation ·
— = not applicable · ✗ = not yet

---

## 1. Core LBM

| Capability | 2D | 3D | GPU | .krk | Theory | Example | API |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| D2Q9 / D3Q19 lattice | ✓ | ✓ | ✓ | ✓ | [01](theory/01_lbm_fundamentals.md), [02](theory/02_d2q9_lattice.md), [06](theory/06_from_2d_to_3d.md) | [Poiseuille 2D](examples/01_poiseuille_2d.md), [Cavity 3D](examples/05_cavity_3d.md) | [lattice](api/lattice.md) |
| BGK collision | ✓ | ✓ | ✓ | ✓ | [03](theory/03_bgk_collision.md) | [Cavity 2D](examples/04_cavity_2d.md) | [collision](api/collision.md) |
| MRT collision | ✓ | ✗ | ✓ | ✓ | [12](theory/12_mrt.md) | — | [collision](api/collision.md) |
| Streaming (periodic/wall) | ✓ | ✓ | ✓ | ✓ | [04](theory/04_streaming.md) | [Couette](examples/02_couette_2d.md) | [streaming](api/streaming.md) |
| Macroscopic recovery (ρ, u) | ✓ | ✓ | ✓ | — | [01](theory/01_lbm_fundamentals.md) | — | [macroscopic](api/macroscopic.md) |
| Body force (Guo forcing) | ✓ | ✓ | ✓ | ✓ | [07](theory/07_body_forces.md) | [Poiseuille 2D](examples/01_poiseuille_2d.md) | [collision](api/collision.md) |
| Axisymmetric (z, r) | ✓ | — | ✓ | ✓ | [09](theory/09_axisymmetric.md) | [Hagen–Poiseuille](examples/09_hagen_poiseuille.md) | [drivers](api/drivers.md) |

**Limitations:** MRT is D2Q9 only in v0.1.0. D3Q27 not implemented.

## 2. Boundary conditions

| BC type | 2D | 3D | Spatial f(x,y,z) | Time f(t) | .krk syntax |
|---|:-:|:-:|:-:|:-:|---|
| Wall (bounce-back) | ✓ | ✓ | — | — | `Boundary north wall` |
| Velocity (Zou-He) | ✓ | ✓ | ✓ | ✓ | `Boundary west velocity(ux=4*U*y*(H-y)/H^2)` |
| Pressure/density outlet | ✓ | ✓ | ~ | ✓ | `Boundary east pressure(rho=1.0)` |
| Periodic | ✓ | ✓ | — | — | `Boundary x periodic` |
| Fixed temperature (Dirichlet) | ✓ | ✓ | ✗ | ✗ | `Boundary west wall(T=1.0)` |
| Adiabatic (zero-flux) | ✓ | ✓ | — | — | default when no `T=` specified |
| Outflow / Neumann / symmetry | ~ | ~ | — | — | recognized, partial |

- Theory: [05 BCs](theory/05_boundary_conditions.md), [19 spatial BCs](theory/19_spatial_bcs.md)
- .krk: [BC types](krk/bc_types.md), [Expressions](krk/expressions.md)
- Expression whitelist: arithmetic, `sin/cos/tan/exp/log/sqrt/abs/clamp`,
  variables `x, y, z, t, Lx, Ly, Lz, Nx, Ny, Nz, dx, dy, dz` + `Define`d vars.

**Limitations:** thermal BCs are scalar only in v0.1.0 (no `T = f(x,y,t)`).
Outflow/Neumann/symmetry are recognized by the parser but kernels are
minimal — use with caution.

## 3. Thermal

| Capability | 2D | 3D | Refined | GPU | .krk | Theory / Example |
|---|:-:|:-:|:-:|:-:|:-:|---|
| Passive scalar transport (DDF) | ✓ | ✓ | ✓ | ✓ | `Module thermal` | [08 thermal DDF](theory/08_thermal_ddf.md) |
| Boussinesq buoyancy | ✓ | ✓ | ✓ | ✓ | auto when hot/cold BCs set | [Rayleigh-Bénard](examples/08_rayleigh_benard.md) |
| Heat conduction (pure) | ✓ | ✓ | — | ✓ | ✓ | [Heat conduction](examples/07_heat_conduction.md) |
| Natural convection (cavity) | ✓ | ✓ | ✓ | ✓ | ✓ | driver `run_natural_convection_2d/3d` |
| ν(T) Arrhenius (kernel) | ✓ | ✗ | ✗ | ✓ | ✗ | code: `collide_boussinesq_vt_2d!` |
| ν(T) modified Arrhenius (Rc) | ✓ | ✗ | ✗ | ✓ | ✗ | driver via `Rc` keyword |
| α(T), κ(T) variable diffusivity | ✗ | ✗ | — | — | ✗ | v0.2.0 |
| Validation vs De Vahl Davis (Ra=1e3) | ✓ 1.4% | — | ✓ 1.4% | ✓ | — | Nu = 1.118 ref |
| Validation vs Fusegi (3D Ra=1e3) | — | ~ 5% | pending AQUA | ✓ | — | Nu = 1.085 ref |

**Limitations in v0.1.0:**
- ν(T) is wired at the **driver** level (`run_natural_convection_2d(; Rc=5)`)
  but **not yet dispatched from .krk**. Adding `Rc = 5` in `Physics {}` does
  not activate the variable-viscosity kernel. Workaround: call the Julia
  API directly. Fix planned for v0.2.0.
- No custom `ν(x, y, z, t, γ̇, T)` user-defined function inlined into
  kernels — this requires a parser extension (planned v0.2.0).
- Thermal BCs are scalar only; `T = sin(π*y)*t/T_max` is not supported yet.

## 4. Grid refinement

| Capability | 2D | 3D | Thermal | GPU | .krk | Theory / API |
|---|:-:|:-:|:-:|:-:|:-:|---|
| Filippova–Hanel patch-based refinement | ✓ | ✓ | ✓ | ✓ | `Refine name { region, ratio }` | [18 grid refinement](theory/18_grid_refinement.md) |
| Temporal sub-cycling (ratio 2, 4, …) | ✓ | ✓ | ✓ | ✓ | auto | [refinement](api/refinement.md) |
| Ghost-layer bilinear/trilinear prolongation | ✓ | ✓ | ✓ | ✓ | — | [refinement](api/refinement.md) |
| Block-average restriction (ratio²/ratio³) | ✓ | ✓ | ✓ | ✓ | — | [refinement](api/refinement.md) |
| Automatic τ rescaling on fine grid | ✓ | ✓ | ✓ | ✓ | — | sanity check reports τ_fine |
| Full-domain wall patches | ✓ | ~ | ~ | ✓ | ✓ | AQUA H100 validation pending |
| Unified .krk dispatch (no driver) | ✓ | ✓ | ✓ | ✓ | ✓ | [Refine block](krk/directives.md) |
| Multi-level (patch inside patch) | ✓ | ✓ | ~ | ~ | ✓ | parser supports, partially tested |
| Geometry on fine patches | ✓ | ✓ | — | ✓ | ✓ | `_apply_patch_geometry!` |

**Key design:** refinement has **no dedicated driver**. A `Refine {}` block
in any .krk file routes automatically to `_run_refined` (2D) or
`_run_refined_3d` (3D), which in turn dispatch the thermal branch when
`Module thermal` is active.

**Limitations:** 3D thermal full-domain wall patches are under AQUA
validation (the `stencil_clamped` guard was removed in 21ae88b because it
forced α=0 at domain edges — which gave stable but wrong Nu values in 3D).
2D full-domain works fine.

## 5. Geometry and obstacles

| Capability | Status | .krk syntax | Notes |
|---|:-:|---|---|
| Implicit condition (circle, box, sphere) | ✓ | `Obstacle cyl { (x-cx)^2 + (y-cy)^2 <= R^2 }` | any boolean expression |
| STL import (binary + ASCII) | ✓ | `Obstacle body stl(file="geom.stl", scale=0.5)` | auto-detect format |
| 3D voxelization (Möller–Trumbore ray casting) | ✓ | — | requires watertight mesh |
| 2D voxelization (slice at z_slice) | ✓ | `stl(file=..., z_slice=0.0)` | — |
| Named primitives `circle()`, `box()`, `sphere()` | ✗ | planned v0.2.0 | ergonomic sugar for common shapes |
| Multiple obstacles | ✓ | repeat `Obstacle` block | — |
| Fluid region (inverse logic) | ✓ | `Fluid inside { x < L/2 }` | — |

- Theory: — (voxelization is implementation detail, not LBM theory)
- API: [io](api/io.md) — `read_stl`, `voxelize_2d`, `voxelize_3d`

**Limitation:** no named primitives yet. Users must spell out
`(x-cx)^2 + (y-cy)^2 <= R^2`. Planned for v0.2.0:
```
Obstacle cyl circle(center=[0.5, 0.5], radius=0.1)
Obstacle box rectangle(xmin=0.2, xmax=0.8, ymin=0.3, ymax=0.7)
Obstacle sph sphere(center=[0.5, 0.5, 0.5], radius=0.1)
```

## 6. Sanity checks (parameter validation)

Running `sanity_check(setup)` — and the check that `run_simulation` does
automatically — covers the following:

| Check | Level | What it verifies |
|---|---|---|
| Relaxation τ = 3ν + 0.5 | error/warn | τ < 0.5 unstable; τ < 0.51 marginal; τ > 10 over-diffusive |
| Compressibility | error/warn | U_ref (Ma = U/c_s); Ma > 0.1 flags compressibility error |
| Spatial resolution | warn | N < 10, N/Re < 1 (boundary layer under-resolved) |
| Thermal τ_α = 3α + 0.5 | error/warn | same bounds as flow τ |
| Prandtl number | warn | Pr < 0.1 or > 10 extreme for SRT thermal |
| Rheology τ_min | warn | from `nu_min` in GNF models |
| **Fine-grid τ (refinement)** | warn | Filippova-Hanel rescaled τ on each patch |
| **Fine-grid τ_α (refinement)** | warn | thermal τ on fine grids (added in 3f05d79) |
| **Fine-grid N/Re** | warn | boundary layer resolution on refined patch |
| Two-phase density/viscosity ratio | warn | code present, v0.2.0 |

- .krk: [Sanity checks](krk/sanity.md)
- Parameter summary is printed at run-time with 2D/3D tag.

**Limitations / planned:**
- No Ra-vs-N thermal BL resolution check (planned — ~ Ra^{1/4} minimum N).
- No CFL check for prescribed-velocity advection (VOF is v0.2.0 anyway).
- Custom user checks not yet exposed in .krk.

## 7. KRK DSL

| Feature | Status | Reference |
|---|:-:|---|
| `.krk` extension + VSCode highlighting | ✓ | [overview](krk/overview.md) |
| Core directives (Simulation, Domain, Physics, Run, Output) | ✓ | [directives](krk/directives.md) |
| Boundary / Initial / Obstacle / Fluid / Velocity blocks | ✓ | [directives](krk/directives.md) |
| Modules: `thermal`, `axisymmetric` | ✓ | [modules](krk/modules.md) |
| Rheology block (GNF + VE parsing) | ✓ | code present, v0.2.0 scope |
| Refine block (2D and 3D) | ✓ | [directives](krk/directives.md) |
| Define user variables | ✓ | [expressions](krk/expressions.md) |
| Expressions (math + `x, y, z, t`) | ✓ | [expressions](krk/expressions.md) |
| **Greek letters** (ν/nu, ρ/rho, σ/sigma, τ/tau …) | ✓ | [aliases](krk/aliases.md) |
| ASCII aliases (`nu` → `ν`) | ✓ | [aliases](krk/aliases.md) |
| **CLI wrapper** (`krk sim.krk`) | ✓ | installed bin: `bin/krk` |
| Preset templates (cavity_2d, etc.) | ✓ | [presets](krk/presets.md) |
| Setup helpers (`Setup reynolds = 100`) | ✓ | [helpers](krk/helpers.md) |
| Sweep (parametric studies) | ✓ | [directives](krk/directives.md) |

**Limitations:**
- No `include` directive for reusable .krk fragments.
- No conditional blocks (`if Re > 100 { ... }`).
- Sweeps are Cartesian product; no Latin hypercube.

## 8. GPU support

| Backend | Status | File type | Notes |
|---|:-:|---|---|
| CPU (KernelAbstractions) | ✓ | Float32/Float64 | reference |
| Metal (Apple Silicon) | ✓ | Float32 recommended | M3 Max validated |
| CUDA (NVIDIA) | ✓ | Float32/Float64 | H100 + A100 validated |
| ROCm (AMD) | ✗ | — | KA supports it, not tested |
| oneAPI (Intel) | ✗ | — | — |

- All kernels use `KernelAbstractions.@kernel` → single source, multi-backend.
- **Fix a82957c:** `unsafe_trunc` instead of `trunc(Int, ...)` in all
  refinement and dualgrid kernels (required for Metal; allocated on GPU).

**Performance reference (H100):** 7675 MLUPS BGK D2Q9 at N=1024, 24k MLUPS
with AA+Float32 kernels (see [MLUPS CPU vs GPU](benchmarks/mlups_cpu_gpu.md)).

## 9. Rheology (v0.2.0 scope)

The following are **implemented in code and tested** but excluded from v0.1.0
publication scope. They will be documented in v0.2.0.

| Model | 2D | 3D | GPU | Thermal coupling |
|---|:-:|:-:|:-:|:-:|
| Newtonian | ✓ | ✓ | ✓ | Arrhenius / WLF |
| Power-law | ✓ | ✗ | ✓ | Arrhenius / WLF |
| Carreau-Yasuda | ✓ | ✗ | ✓ | Arrhenius / WLF |
| Cross | ✓ | ✗ | ✓ | Arrhenius / WLF |
| Bingham (Papanastasiou) | ✓ | ✗ | ✓ | Arrhenius / WLF |
| Herschel–Bulkley | ✓ | ✗ | ✓ | Arrhenius / WLF |
| Oldroyd-B (log-conf) | ✓ | ✗ | ✓ | Arrhenius / WLF |
| FENE-P | ✓ | ✗ | ✓ | Arrhenius / WLF |
| Saramito (EVP) | ✓ | ✗ | ✓ | Arrhenius / WLF |

Effective viscosity dispatches at compile-time (zero-cost abstraction
via Julia's JIT). See `src/rheology/` (models, viscosity, strain_rate).

## 10. Output and diagnostics

| Output | Status | .krk syntax |
|---|:-:|---|
| VTK (+ PVD time series) | ✓ | `Output vtk every 1000 [rho, ux, uy]` |
| PNG snapshots (CairoMakie) | ✓ | `Output png every 500 [uy]` |
| GIF animations | ✓ | `Output gif every 100 [ux] fps=15` |
| Diagnostics CSV | ✓ | `Diagnostics every 100 [step, drag, lift]` |
| ParaView-ready layout | ✓ | — (VTK pvd files) |

- API: [io](api/io.md), [postprocess](api/postprocess.md)
- Helper: `open_paraview(result)` opens the latest output in ParaView.

---

## Out-of-scope for v0.1.0

These features exist in the codebase (tests often pass) but are
deliberately not documented in v0.1.0:

- **Multiphase (VOF + PLIC)** — see `src/kernels/vof_2d.jl`,
  `src/kernels/pressure_vof_2d.jl` · v0.2.0
- **Phase-field (Allen-Cahn)** — `src/kernels/phasefield_2d.jl` · v0.2.0
- **Shan–Chen spinodal** — `src/drivers/multiphase.jl` · v0.2.0
- **Species transport** — `src/kernels/species_2d.jl` · v0.2.0
- **Viscoelastic cylinder** — `src/drivers/viscoelastic.jl` · v0.2.0
- **3D rheology** — kernels 2D only · v0.2.0 or later
- **3D MRT** — D3Q19 BGK only · v0.2.0 or later

---

## Quick decision helper

| I want to… | I need… |
|---|---|
| lid-driven cavity 2D | BGK + wall BCs + velocity BC. [Example](examples/04_cavity_2d.md) |
| flow past a cylinder | BGK + inlet parabolic BC + obstacle (`Obstacle { ... }`). [Example](examples/06_cylinder_2d.md) |
| natural convection in a cavity | `Module thermal` + two `wall(T=...)` BCs. [Example](examples/08_rayleigh_benard.md) |
| refine near a wall | add `Refine name { region=[...], ratio=2 }` block. [20](examples/20_grid_refinement_cavity.md) |
| run on H100/A100 | `backend = CUDABackend(); run_simulation(...; backend=backend)` |
| vary the viscosity with temperature | call `run_natural_convection_2d(; Rc=10)` directly (not via .krk yet) |
| use an STL geometry | `Obstacle body stl(file="geom.stl")` |
| sweep parameters | `Sweep Re = [100, 200, 400]` |
| run from the terminal | `krk sim.krk` (CLI wrapper) |
