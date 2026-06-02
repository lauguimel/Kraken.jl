# Curvilinear LBM for Kraken.jl v0.2 — literature survey and recommendation

*Research document answering `research_prompt_curvilinear_lbm.md`. Input to
the design doc. Opinionated — one recommended path, with a conservative
fallback.*

---

## 1. Formulation comparison

There are really three families of "curvilinear LBM", and confusing them
has caused most of the bad press the approach has. The split below is
the one I think matters for implementation.

### Family A — Coordinate-transformed LBM (CT-LBM / FD-LBM in generalised coordinates)

Transform the discrete BGK equation from physical `(x, y)` to computational
`(ξ, η)`, producing Jacobian-weighted advection in the streaming step. The
stream step becomes a finite-difference discretisation of
`∂_t f_i + c̃_i·∇_{ξη} f_i = Ω_i`, where the transformed velocities
`c̃_i = J^{-1} c_i` depend on the metric. Grid is logically uniform
`N_ξ × N_η`.

- **Mei & Shyy, 1998, *J. Comput. Phys.* 143, 426-448.**
  Seminal paper. BGK + second-order upwind FD streaming. Validated on
  lid-driven cavity and flow past a NACA airfoil at Re ≤ 10⁴ on a C-grid.
  Known to require small CFL and decent orthogonality.
- **Imamura, Suzuki, Nakamura, Yoshida, 2005, *J. Comput. Phys.* 207,
  747-779.** Extension of Mei-Shyy with MUSCL-TVD flux for the advection;
  airfoil Re = 5000 transonic. Showed the FD streaming introduces
  dissipation that shrinks the effective viscosity range.
- **Budinski, 2014, *Int. J. Numer. Methods Fluids* 75, 417-444.**
  MRT + generalised coordinates; power-law fluids; natural-convection
  cavity and Taylor-Couette with eccentricity. MRT is essential — BGK
  goes unstable at moderate Re on stretched meshes.
- **Hejranfar & Ezzatneshan, 2014, *Comput. Fluids* 91, 12-30.**
  Compressible FD-LBM in generalised coordinates with MUSCL-Hancock.

Pros: single logically-uniform grid, local stencil, straightforward to
put in one kernel. Cons: you lose the exact Lagrangian streaming that
makes LBM appealing; effective numerical viscosity depends on mesh and
CFL; BGK is not enough — MRT/TRT at minimum.

### Family B — Interpolation-supplemented LBM (ISLBM) and semi-Lagrangian LBM (SLBM)

Keep Lagrangian streaming `f_i(x + c_i Δt, t+Δt) ← f_i(x,t) + Ω_i`, but
the departure point `x - c_i Δt` (or arrival point `x + c_i Δt`) no
longer lands on a mesh node. Interpolate from neighbours.

- **He, Luo, Dembo, 1996, *J. Stat. Phys.* 87, 115-136.** ISLBM on
  non-uniform lattices; second-order interpolation (Lagrange). Validated
  on Poiseuille and Taylor-Couette.
- **Filippova & Hänel, 1998, *J. Comput. Phys.* 147, 219-228.**
  Grid refinement by rescaling; not strictly curvilinear but the
  interpolation philosophy is the same and we already ship this.
- **Krämer, Küllmer, Reith, Foysi, Steiner, 2017, *Phys. Rev. E* 95,
  023305, "Semi-Lagrangian off-lattice Boltzmann method for weakly
  compressible flows".** This is the paper that revived the approach
  for modern hardware. SLBM on arbitrary body-fitted meshes, quadratic
  interpolation, decoupled Mach and mesh spacing (CFL ≠ 1 allowed). GPU
  implementation reported.
- **Di Ilio, Chiappini, Ubertini, Bella, Succi, 2018, *Phys. Rev. E* 97,
  053307.** SLBM on moving body-fitted grids (rotating airfoil, flapping
  wing). Single kernel per timestep in their CUDA code.
- **Wilde, Krämer, Reith, Foysi, 2020, *Comput. Fluids* 204, 104519.**
  SLBM with D3Q19/D3Q27 on curvilinear meshes; flow over spheres and
  airfoils.

Pros: keeps the Boltzmann-physics character of streaming; CFL and mesh
spacing are decoupled; stability is best-in-class for curved grids.
Cons: interpolation stencil (typically 2×2 or 4×4 bilinear/bicubic per
distribution) is heavier than plain copy-and-stream; you need to store
interpolation weights.

### Family C — Finite-volume and finite-element LBM (FV-LBM, DG-LBM, SE-LBM)

Integrate the discrete BGK over arbitrary cells; fluxes are reconstructed
from neighbours.

- **Nannelli & Succi, 1992, *J. Stat. Phys.* 68, 401-407.** Founding
  FV-LBM paper.
- **Peng, Xi, Duncan, Chou, 1999, *Phys. Rev. E* 59, 4675-4682.** FV-LBM
  on unstructured meshes.
- **Patil & Lakshmisha, 2009, *J. Comput. Phys.* 228, 5262-5279.** FV-LBM
  with flux-limiters.
- **Min & Lee, 2011, *J. Comput. Phys.* 230, 245-259.** Spectral-element
  discontinuous-Galerkin LBM. High order, expensive.
- **Shrestha, Biswas, Manna, 2015,** FV-LBM on collocated body-fitted
  grids (cited for completeness).

Pros: native unstructured support; conservative. Cons: heavy per-cell
work, multiple neighbour lookups, not a good match for single-kernel GPU
with KernelAbstractions; and we *don't want* unstructured in v0.2.

### Open-source implementations

| Code | Curvilinear? | Notes |
|---|---|---|
| Palabos (C++) | No. Handles curved walls on Cartesian via off-lattice BCs (Guo-Zheng-Shi, Bouzidi-Firdaouss-Lallemand). | Closest to what we already do with ghost fluid. |
| OpenLB (C++) | No. Cartesian AMR (cuboid decomposition). | No body-fitted. |
| waLBerla (C++) | No. Block-structured Cartesian. | Fastest Cartesian code; no curvilinear path. |
| Musubi / APES (Fortran) | Octree Cartesian with XDG cut-cells. | Not curvilinear in the generalised-coords sense. |
| STLBM, lbmpy, TCLB | Cartesian. | Same story. |
| Sailfish-CFD | Cartesian. | GPU, unmaintained. |

**Bottom line for Section 1:** there is *no widely-used open-source LBM
code with a native curvilinear / body-fitted grid path*. Everything
either off-lattice-BCs the curvature onto a Cartesian mesh, or cut-cells
it. A clean, GPU-native curvilinear LBM is a real gap — and a
differentiator worth putting in the v0.2 paper.

---

## 2. Recommendation for Kraken.jl

**Go with semi-Lagrangian LBM (Family B) on a single logically-structured
`N_ξ × N_η` grid, with regularised or MRT collision. Reference
implementation: Krämer et al. 2017 + Di Ilio et al. 2018.**

Rationale, explicit against the next-best alternative (Mei-Shyy FD-LBM):

| Criterion | SLBM (recommended) | Mei-Shyy FD-LBM |
|---|---|---|
| Single-kernel GPU | ✓ One kernel: interpolate-collide-replace. Stencil is 2×2 per distribution (bilinear); trivially a KA.jl `@kernel`. | ✓ But the FD stream needs an upwind-biased stencil of the same width. Similar cost. |
| Accuracy on stretched grids | Second-order in space *independent* of mesh distortion (Krämer 2017 §5). | Second-order only on orthogonal meshes; drops to first-order on skewed cells (Mei-Shyy §4, §6). |
| Stability at Re 100-400 | Stable; Wilde 2020 ran Re = 20 000 on a sphere mesh. | Marginal with BGK; needs MRT + TVD flux (Imamura 2005). |
| Implementation time | ~3 weeks (stream replaced by interp; collision unchanged). | ~4-5 weeks (FD fluxes, limiters, BC reformulation). |
| Fit with existing Kraken code | Current ghost-fluid grid refinement *already* does interpolation-based streaming across patches (Filippova-Hänel). Curvilinear SLBM is the same machinery applied per-cell. | Requires new FD infrastructure we don't have. |
| CFL decoupling | ✓ CFL ≠ 1 allowed. Lets us keep Δt_max on the coarse part of the mesh. | ✗ Δt tied to min cell size via acoustic CFL of the FD scheme. |
| Spurious modes / dissipation | Interpolation is purely dispersive; energy is conserved up to rounding. | FD introduces numerical viscosity ~u·Δx·(1-CFL), must be subtracted from physical ν. |

The only argument for Mei-Shyy is "it's the classical paper everyone
cites". In 2026 that's not enough. Krämer 2017 and Wilde 2020 have
already shown SLBM beats FD-LBM on every metric we care about.

**Secondary choice: MRT collision, not BGK.** Curvilinear grids excite
ghost modes that BGK cannot damp. MRT (or the newer cumulant / central-
moment collisions — Geier, Schönherr, Pasquali, Krafczyk 2015 and Coreixas
et al. 2017) fixes this for very modest extra cost. We already have MRT
in Kraken, so reuse it.

**Tertiary choice: quadratic (biquadratic) interpolation, not linear.**
Linear interpolation degrades to O(Δx) in practice (Krämer §5.2).
Biquadratic (3×3 stencil) is second-order on distorted meshes and fits in
shared memory on both H100 and M3 Max. This is where almost all the
implementation risk lives — get the stencil indexing right once, reuse
forever.

---

## 3. Key pitfalls & design requirements

Grouped by category, each item with the reference that warned us.

### Mesh quality

- **Aspect ratio ≤ 10 in the log-law region, ≤ 100 anywhere.** Beyond
  that, the regularised collision's trace-free hypothesis starts to
  break (Dorschner, Chikatamarla, Karlin, 2016, *J. Comput. Phys.* 315,
  434-457).
- **Jacobian positivity everywhere.** A cell with `det J ≤ 0` is a mesh
  fold — SLBM silently produces NaNs downstream. Assert at mesh build
  time (Mei-Shyy 1998, §3; Imamura 2005, Appendix A).
- **Orthogonality at walls.** Wall-normal departure points must land
  cleanly on the interior, not spill across the wall. Budinski 2014
  reports 30 % error on `Cd` when orthogonality degrades below 60°.
  Enforce this in the mesh generator for O-grids.
- **Skewness on concave corners.** Sharp re-entrant corners (e.g.
  step flow) need local smoothing; the Jacobian derivative becomes
  discontinuous and the interpolation stencil straddles it. Wilde 2020,
  §6.3.

### Stability thresholds

- **Ma < 0.15 for incompressible regime**, same as Cartesian LBM. SLBM
  actually tolerates higher Ma than Cartesian (Krämer 2017, Fig. 7) but
  v0.2 isn't the place to push this.
- **Mesh-spacing Reynolds `Re_h = u Δx / ν < 5`** empirically on
  stretched grids (Budinski 2014, §4.2). Tighter than Cartesian
  (`Re_h < 10`). Caused by anisotropic viscous propagation — MRT helps
  but doesn't eliminate it.
- **CFL_max ≈ 1.5** for SLBM with biquadratic interpolation (Wilde 2020,
  Table 2). Going over loses monotonicity.

### Boundary conditions — these *all* change from Cartesian

- **Bounce-back on curved walls.** Halfway bounce-back on a curvilinear
  mesh requires the wall to coincide with the computational face
  `η = η_wall`. Use Ladd-Verberg (Ladd 1994, *J. Fluid Mech.* 271,
  285-309) or Yu-Mei-Luo-Shyy interpolated bounce-back (2003, *Prog.
  Aerosp. Sci.* 39, 329-367). For O-grids the wall is always at
  `η = 0`, so this simplifies.
- **Zou-He for curvilinear inlets.** The classical Zou-He (Zou & He,
  1997, *Phys. Fluids* 9, 1591-1598) assumes the inlet is axis-aligned.
  In computational space the inlet *is* axis-aligned (`ξ = 0`), so Zou-
  He works — but only if you specify `u_ξ, u_η` in computational space,
  not `u_x, u_y`. Document this loudly.
- **Non-reflecting pressure outlets.** Izquierdo-Fueyo 2008, Phys. Rev.
  E 78, 046707; or the characteristic outlet of Heubes-Bartel-Ehrhardt
  2014. Plain `∂p/∂n = 0` reflects on stretched grids.
- **Periodic in θ for O-grids.** Trivial in computational space —
  `f(ξ, η=0) = f(ξ, η=N_η)`. Make sure the interpolation stencil reads
  across the seam.

### Numerical issues

- **Pressure-velocity coupling.** Classical LBM computes `p = ρ c_s²`
  where `ρ = Σ f_i`. On stretched grids this picks up a spurious
  component proportional to the Jacobian gradient (Krämer 2017, §5.4).
  Wilde 2020 proposes a Gauss-Hermite projection step; for v0.2, use a
  regularised collision which kills it by construction.
- **Spurious Galilean-invariance breaking.** D2Q9 has it anyway;
  curvilinear makes it worse when the mesh advects with the flow. Not
  an issue for us (fixed meshes in v0.2).
- **Grid-induced anisotropy.** The effective stress tensor rotates with
  the mesh. For Newtonian it's invisible; for viscoelastic (future lbm
  branch merge) it matters. Note for v0.3.

### Design requirements that fall out

1. Store the metric `(J, ∂x/∂ξ, ∂x/∂η, ∂y/∂ξ, ∂y/∂η)` per cell —
   5 `FT` per cell. One-shot, computed at mesh build.
2. Store interpolation weights per `(cell, q)` — `9 × 9 = 81` `FT` per
   cell *if* we precompute. Or recompute on the fly — cheap for D2Q9,
   saves memory. Decide in Week 2 after benchmarking.
3. Regularised or MRT collision *required*, not optional.
4. Mesh generator must emit: `(X, Y, J, dXdξ, dXdη, dYdξ, dYdη, wall_mask)`.

---

## 4. Validation roadmap

Ordered cheapest → hardest. Each case has an analytical or published
reference, an expected order, and a nominal target error.

### 4.1 Taylor-Couette (analytical) — Week 2

Concentric cylinders, `R_i` inner rotating at `Ω_i`, `R_o` outer fixed.
Analytical `u_θ(r) = A r + B/r`. Polar O-grid, `Re = Ω_i R_i (R_o - R_i) / ν`.

- Grid: `n_r = 32, 64, 128`, `n_θ = 64, 128, 256`, uniform in θ, uniform in r.
- Expected: second order in `r`, machine precision in `θ` (exact
  translational symmetry). L2 error in `u_θ` at `n_r = 64`: target
  ≤ 1×10⁻³ at `Re = 10`.
- Validates: metric, streaming on a uniform polar grid, periodic BC in
  θ, Ladd bounce-back on the curved walls.
- Reference: Ladd 1994 for LBM Taylor-Couette precedent.

### 4.2 Taylor-Couette stretched (analytical) — Week 2

Same case, *stretched* in `r` with `r_stretch = tanh(2.0)` (fine near
inner wall). Isolates the effect of mesh distortion.

- Target: same 1×10⁻³ error. If the stretched case is worse than
  the uniform case by more than 2×, interpolation is the culprit —
  debug before moving on.

### 4.3 Poiseuille on stretched grid (analytical) — Week 2

Channel flow with wall-normal `tanh` stretching, periodic in streamwise.
Logically rectangular mesh, no curvature but strong anisotropy.

- `Re = 100, Re_h,max = 2`.
- Analytical: `u(y) = (G/2ν) y (H-y)`, parabolic.
- Target: L2 error ≤ 5×10⁻⁴. Convergence rate 2 confirmed between
  `N_y = 32` and `N_y = 64`.
- Validates: Zou-He inlet in computational space (if not periodic),
  outlet BC, stretched-grid stability.

### 4.4 Natural convection cavity, Ra = 10⁴–10⁶ — Week 3

Body-fitted rectangular mesh with wall-normal stretching; DDF thermal
LBM. Matches our existing `run_natconv_refined_2d`.

- Reference: de Vahl Davis 1983, *Int. J. Numer. Methods Fluids* 3,
  249-264.
- Target: Nu on hot wall within 1 % of de Vahl Davis at Ra = 10⁶
  with `64 × 64` stretched cells. (Current Cartesian refined run
  needs ~`128 × 128`. This is the payoff case.)
- Validates: coupling of curvilinear fluid and thermal DDFs, buoyancy
  body force in computational coordinates.

### 4.5 Flow past cylinder, Re = 20 — Week 3 (THE target)

Schäfer & Turek 1996 benchmark 2D-1, steady laminar.

- Reference: Schäfer & Turek 1996, "Benchmark Computations of Laminar
  Flow Around a Cylinder", in *Flow Simulation with High-Performance
  Computers II*, ed. E.H. Hirschel, Notes on Numerical Fluid Mechanics
  52, 547-566. `Cd = 5.5795, Cl = 0.01061, ΔP = 0.11752`.
- O-grid, `n_r = 96, n_θ = 256`, `r_stretch ≈ 1.1` geometric near wall,
  `R_out = 8D`, far-field uniform inlet Zou-He, outlet equilibrium
  extrapolation.
- Target: `Cd` within 1 % of reference = `5.524 ≤ Cd ≤ 5.636`.
- Validates: the full stack. This is the figure of the v0.2 paper.

### 4.6 Flow past cylinder, Re = 100, vortex shedding — Week 4

Same Schäfer-Turek 2D-2 case, unsteady. `St = 0.3 ± 0.005`, `Cd_max ≈ 3.23`.

- Target: `St` within 2 %, `Cd_max` within 3 %.
- Validates: time-dependent stability of the curvilinear path at a
  Reynolds number that *we've had to fight for* in the uniform-grid
  version (Hulsen K=132 memory).

### 4.7 Flow past cylinder, Re = 200-400 — stretch goal, Week 4

Not for the paper, but if the Re=100 case works, push the envelope to
see where stability fails. Gives us honest limits to publish.

---

## 5. Mesh generation strategy

### Recommendation: parametric generators only for v0.2. Defer Gmsh to v0.3.

Reasoning: every validation case in §4 can be expressed with one of two
parametric generators. Gmsh adds a dependency, a file format (`.msh4`),
a non-trivial parser, and zero benefit for Taylor-Couette / Poiseuille /
Schäfer-Turek. Ship two generators, nail the cylinder paper, then add
Gmsh when someone actually needs a NACA airfoil.

### Generator 1: `polar` (O-grid around a point)

```
cx, cy          # centre
r_inner         # inner radius (body surface)
r_outer         # outer radius (far-field)
n_r, n_theta    # resolution
r_stretch       # tanh parameter, 0 = uniform, large = concentrated at wall
theta_offset    # default 0, starts at +x axis
```

Enough for cylinder, Taylor-Couette, and any circular body. Emits `X, Y, J,
metric` on a `(n_r+1) × n_theta` grid. Periodic in `θ`.

### Generator 2: `stretched_box` (orthogonal rectangular)

```
x_min, x_max, n_x, x_stretch, x_stretch_dir   # (:none, :left, :right, :both)
y_min, y_max, n_y, y_stretch, y_stretch_dir
stretch_fn     # :tanh (default), :geometric, :linear
```

Covers Poiseuille, natural-convection cavity, channel flow. No curvature,
but the *metric* path is exercised because `dX/dξ ≠ const`.

### Why not C-grid / H-grid in v0.2?

- C-grid needs a cut and a special interpolation stencil at the trailing
  edge; airfoils are v0.3.
- H-grid is trivial (it's just two boxes) — if a user wants it they can
  use two `stretched_box` patches with our existing refinement glue.

### Why not Gmsh?

- `.msh4` parsing is 500+ lines even for simplicial meshes, more for
  quadrilateral (the only curvilinear-LBM-compatible topology).
- Gmsh quadrilateral meshing via `Mesh.RecombineAll = 1` is notoriously
  flaky on non-trivial geometries.
- Any user who wants Gmsh already has a `.msh → (X, Y)` script. We can
  write a thin adapter in v0.3 that reads an already-structured block.

### Mesh validation at build time

Before returning the mesh, assert:

1. `det J > 0` everywhere (no folds).
2. Minimum orthogonality angle ≥ 45° (Budinski's warning).
3. Maximum aspect ratio ≤ 100 (Dorschner's warning).
4. For O-grids: `n_theta` ≥ some minimum for `r_outer / r_inner` to
   prevent radial rays from diverging too fast.

Emit warnings (not errors) with the offending cell indices. Mesh
debugging on first contact will save us a week.

---

## 6. API surface for the .krk DSL

Three alternatives considered. Recommendation first, alternatives second.

### Recommended: first-class `Mesh` block with `type` keyword

```
Mesh {
    type = polar
    center = [2.0, 2.0]
    r_inner = 0.5
    r_outer = 5.0
    n_r = 96
    n_theta = 256
    r_stretch = 2.0
}
```

or

```
Mesh {
    type = stretched_box
    x = [0.0, 22.0]
    y = [0.0, 4.1]
    n_x = 512
    n_y = 128
    y_stretch = 2.0
    y_stretch_dir = both
}
```

Pros:

- One block name (`Mesh`) for all topologies. Matches the existing
  single-block convention (`Domain`, `Body`, `Refine`).
- `type = polar | stretched_box | cartesian` is a clean discriminator and
  extends to `gmsh` in v0.3 without an API break.
- `cartesian` becomes an explicit opt-in — self-documenting.

Cons:

- Users migrating from v0.1 `.krk` files (which have no `Mesh` block)
  need to add one. Mitigation: if `Mesh` is absent, default to
  `Mesh { type = cartesian, ... }` built from the existing `Domain`
  block. No breaking change.

### Alternative A: subtyped block name (`Mesh polar { ... }`)

The prompt's original example. Maps well to the parser but:

- Two-word block names aren't used anywhere else in the DSL
  (`Domain`, `Body`, `Refine`, `Lattice` are all single-word).
- Introducing a two-word pattern means the parser must disambiguate
  `Block word { }` vs. `Block word word { }`. Small complication now,
  accumulates.

I'd avoid it. The type-as-keyword pattern wins on DSL consistency.

### Alternative B: function-based mapping

```
Mesh {
    X = (xi, eta) -> cx + r_inner*exp(eta*ln(r_outer/r_inner)) * cos(2π*xi)
    Y = (xi, eta) -> cy + r_inner*exp(eta*ln(r_outer/r_inner)) * sin(2π*xi)
    n_xi = 256; n_eta = 96
}
```

Pros: ultimate flexibility, researcher-friendly.
Cons: users write analytical mappings wrong; Jacobian computation by
AD is heavy; poor error messages; most users don't need it.

**Decision: ship (Recommended) for v0.2. Keep (Alternative B) as an
undocumented escape hatch `type = custom` for power users.** Document
it properly in v0.3 once the footguns are known.

---

## 7. Open questions to resolve before implementation

1. **Store interpolation weights or recompute on the fly?**
   81 `FT` per cell at `N_ξ × N_η = 96 × 256` = 2 MB. Trivial on H100
   but ~10 % of M3 Max shared memory for larger grids. Decision matrix:
   recompute on Metal (bandwidth-bound anyway), cache on CUDA (compute-
   bound). Benchmark in Week 2.

2. **Which regularised collision?** Coreixas recursive regularisation
   (2017), Latt-Chopard projected regularisation (2006), or MRT? MRT is
   already in the codebase, so the zero-cost option is to use it first
   and only add regularised collision if MRT instabilities surface on
   stretched meshes. I'd start with MRT, switch only if needed.

3. **How do we couple curvilinear fluid with the existing patch-based
   refinement?** Near-term answer: we don't. v0.2 curvilinear is a
   standalone topology. A curvilinear patch inside a Cartesian parent,
   or vice versa, is a v0.3 problem. Confirm the user agrees before
   locking the API.

4. **Metric derivatives: analytical or finite-difference?**
   For `polar` and `stretched_box` both are available — analytical is
   simpler to implement but pushes the mesh generator toward
   hand-derived formulas. Finite-difference (central, 4-th order) works
   for any parametric generator but costs a 5-point stencil at build
   time. Recommended: FD for genericity, cross-check against analytical
   for `polar` and `stretched_box` in the unit tests.

5. **Do we commit to writing the Schäfer-Turek Re=100 unsteady case
   before v0.2 ships, or is Re=20 steady enough for the paper?**
   My vote: Re=100 unsteady is the real selling point. Re=20 alone
   is 1999-grade validation. The `St` match on a single-kernel GPU
   curvilinear code is the paper's headline figure.

---

## Summary in one paragraph

SLBM (Krämer 2017, Di Ilio 2018, Wilde 2020) on a logically-uniform
`N_ξ × N_η` grid with MRT collision, biquadratic interpolation, a
parametric `polar` + `stretched_box` mesh generator, and a `Mesh { type = … }`
block in the .krk DSL. Validate: Taylor-Couette → stretched Poiseuille →
natural-convection cavity → Schäfer-Turek 2D-1 (Re=20) → Schäfer-Turek
2D-2 (Re=100, unsteady). Three to four weeks if the interpolation indexing
is reused from our existing Filippova-Hänel refinement machinery. The
differentiator for the paper: there is no other open-source GPU-portable
LBM code with a native curvilinear path.
