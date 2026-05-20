# M32 Phase 1 — rheoTool ↔ Kraken setup audit verdict

**Date:** 2026-05-21
**Department:** M32-Phase1-setup-audit
**Mandate:** Compare every physical and numerical parameter between the rheoTool
reference cases (`cylinder_wi{0.1,0.5,1.0}`) and the Kraken production sweep
(`bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl`) **before** any further
Phase 2-3 matrix testing.

---

## 1. Side-by-side parameter table

Notes on conventions:

* rheoTool length scale: R = 1 (cylinder radius, from `blockMeshDict` arcs at
  unit radius).
* rheoTool velocity scale: U_mean = 1 (set by `codedFixedValue` `Umean = 1.0`
  in `0/U`, file
  `bench/rheotool/cylinder_wi1.0/0/U:32`).
* Kraken length scale: 1 lattice unit. Radius `R` is the kwarg (30, 50, 80).
* Kraken velocity scale: `u_mean = 0.005` LU (env `KRAKEN_U_MEAN`,
  `run_cyl_bigsweep_v2_2d.jl:118`).
* Reynolds: rheoTool Re_R = ρ·Umean·R/(etaS + etaP) = 1·1·1/(0.59+0.41) = **1.0**.
  Kraken Re_R = u_mean·R/(ν_s + ν_p) with ν_total set via Re_R env var.

| Parameter | rheoTool (all 3 Wi) | Kraken production | Match? | Citation |
|---|---|---|---|---|
| Cylinder radius R | 1.0 m (length unit) | R ∈ {30, 50, 80} LU | both = 1 in dimensionless units | `bench/rheotool/cylinder_wi1.0/system/blockMeshDict:18-37` arcs at radius 1; `run_cyl_bigsweep_v2_2d.jl:99,239` |
| Domain L_up / R | **20.0** | **15.0** | **NO (HARD)** | `blockMeshDict:20` `(-20 0 0)`; `run_cyl_bigsweep_v2_2d.jl:101` `KRAKEN_L_UP_LIST=15.0` |
| Domain L_down / R | **60.0** | **15.0** | **NO (HARD)** | `blockMeshDict:30` `(60 0 0)`; `run_cyl_bigsweep_v2_2d.jl:102` `KRAKEN_L_DOWN_LIST=15.0` |
| Channel half-width / R | 2.0 (post-mirror full channel y∈[-2, 2]) | 2.0 (H = 4R, half = 2R) | yes | `0/U:38` `halfHeight=2.0` + `mirrorMeshDict:19-22` mirror about y=0; `run_cyl_bigsweep_v2_2d.jl:239` `H = 4*R` |
| Blockage R/halfH | 0.5 | 0.5 | yes | as above |
| Inlet BC (U) | **parabolic Poiseuille**, Umean=1, halfHeight=2 | **parabolic Poiseuille**, mean=u_mean (lattice) | yes | `0/U:24-43`; `viscoelastic_logfv_2d.jl:273` (`parabolic_face_profile_2d`) + `step_geometry_2d.jl:214-231` |
| Inlet BC (p) | zeroGradient | implicit via ZouHe velocity | equivalent | `0/p:23-25`; `viscoelastic_logfv_2d.jl:276,377` (`default_step_bcspec_2d` → `MaskedZouHeVelocity`) |
| Inlet BC (tau) | fixedValue 0 (NOT analytical Oldroyd-B inlet) | psixx=psixy=psiyy=0 (zero stress, log conf = 0 → C=I) | yes — both impose stress-free fluid at inlet | `0/tau:23-26`; `viscoelastic_logfv_2d.jl:295-303` allocate zeros, never overwritten west boundary |
| Outlet BC (U) | zeroGradient | implicit via MaskedZouHePressure | ~equivalent | `0/U:46-49`; `step_geometry_2d.jl:281` |
| Outlet BC (p) | fixedValue 0 | ρ_out = 1 (Zou-He pressure) | yes | `0/p:27-31`; `viscoelastic_logfv_2d.jl:276` (`default_step_bcspec_2d(geom, u_profile, one(T))`) |
| Outlet BC (tau) | zeroGradient | psi extrapolated (open BC) | yes | `0/tau:27-30`; `viscoelastic_logfv_2d.jl:377` (`fvfd_openx_wally_bcspec_2d`: west=:open, east=:open) |
| Top/bottom wall BC (U) | fixedValue (0,0,0) — no-slip | halfway-BB no-slip | yes | `0/U:51-56`; `viscoelastic_logfv_2d.jl:476-478` + `specs.jl:91` (south=:wall, north=:wall) |
| Top/bottom wall BC (tau) | **linearExtrapolation** | extrapolated via `:wall` FVFD BC | likely equivalent | `0/tau:32-37` |
| Cylinder BC (U) | fixedValue (0,0,0) — no-slip on conformal mesh | halfway-BB on rasterised q_wall (cut-link) | **DIFFERENT MESH** (SOFT) | `0/U:51-56`; `viscoelastic_logfv_2d.jl:823` (`precompute_q_wall_cylinder`) |
| Cylinder BC (tau) | linearExtrapolation | extrapolated by `:open` outlet on advection | match in spirit | `0/tau:32-37` |
| Density ρ | 1 | 1 (LBM convention) | yes | `constant/constitutiveProperties:21`; `run_cyl_bigsweep_v2_2d.jl` (implicit LBM ρ=1) |
| Solvent viscosity etaS | **0.59** Pa·s | ν_s = β·ν_total with β=0.59 | yes | `constant/constitutiveProperties:22`; `run_cyl_bigsweep_v2_2d.jl:96,244` |
| Polymer viscosity etaP | **0.41** Pa·s | ν_p = (1-β)·ν_total with β=0.59 | yes | `constant/constitutiveProperties:23`; `run_cyl_bigsweep_v2_2d.jl:245` |
| β = etaS / (etaS+etaP) | 0.59 | 0.59 | yes | both |
| Relaxation time λ | 0.1, 0.5, 1.0 s | λ = Wi · R / u_mean (lattice units) | dimensionless Wi matches | `constant/constitutiveProperties:24`; `run_cyl_bigsweep_v2_2d.jl:246` |
| Weissenberg Wi = λ·Umean/R | 0.1, 0.5, 1.0 | scan in {0.1, 0.3, 0.5} default; env-extensible | partial overlap | `sweep_wi_cylinder.sh` envvar `lambda`; `run_cyl_bigsweep_v2_2d.jl:97` |
| Reynolds Re_R = ρ·U·R/(etaS+etaP) | **1.0** | 0.1 OR 1.0 (env var sweeps both) | partial — must select Re=1 | `0/U` Umean=1, constitutive eta=1; `run_cyl_bigsweep_v2_2d.jl:98,243` |
| Polymer model | **Oldroyd-B (log formulation)** `Oldroyd-BLog` | `:oldroydb` (log-conformation) | yes | `constant/constitutiveProperties:19`; `run_cyl_bigsweep_v2_2d.jl:281` |
| Stabilization | **coupling** (no BSD) | bsd_fraction = 1.0 (BSD ON) | **NO (HARD/SOFT)** | `constant/constitutiveProperties:26`; `run_cyl_bigsweep_v2_2d.jl:13,100` |
| Advection scheme (U) | CUBISTA (`GaussDefCmpw cubista`) | LBM (D2Q9 TRT) | **DIFFERENT METHOD** (SOFT) | `system/fvSchemes:35`; `viscoelastic_logfv_2d.jl:475-478` (`fused_trt_libb_v2_guo_field_step!`) |
| Advection scheme (τ) | CUBISTA (`GaussDefCmpw cubista`) | Rusanov upwind (option `:muscl_superbee` available) | **DIFFERENT METHOD** (SOFT) | `system/fvSchemes:37`; `run_cyl_bigsweep_v2_2d.jl:110` default `rusanov`; `viscoelastic_logfv_2d.jl:405-415` |
| Time integration | implicit Euler (`Euler` ddtScheme), SIMPLE/PISO coupled | explicit LBM + explicit polymer substeps | **DIFFERENT** (SOFT) | `system/fvSchemes:18`; LBM/substep loop `viscoelastic_logfv_2d.jl:387-555` |
| Timestep Δt | **1e-2** (wi0.5, wi1.0), **2e-2** (wi0.1) | LBM dt = 1 LU | DIFFERENT but dimensionless τ_p_LBM vs λ_rT compatible | `controlDict:26`; LBM convention |
| Convective Co_max | 0.01 (`maxCo`) | LBM Mach ≈ u_mean/cs ≈ 0.0087 | comparable | `controlDict:50` |
| Total simulated time | endTime = 6 (wi0.1), 10 (wi0.5, wi1.0) → 6·Umean/R = 6, 10 advective times | max_steps · dt_LBM / (R · 1) = 100k / R lattice times = 100k / (R / u_mean) · u_mean = 100k · u_mean / R = 16.67 (R=30), 10 (R=50), 6.25 (R=80) advective times | comparable but Kraken shorter at R=80 | `controlDict:24`; `run_cyl_bigsweep_v2_2d.jl:15` |
| Time-averaging window for Cd | rT: instantaneous Cd_last (or last 20 samples ≈ last 0.2·endTime); Kraken: last 20% of max_steps | comparable convention | yes-ish | `summarize_cd.sh:11-44`; `run_cyl_bigsweep_v2_2d.jl:120,248` |
| Mesh near-wall resolution | **0.005R** perpendicular near cylinder (block 1 j-size 0.00489 = 0.0049R; ~40 cells around quarter-arc) | **1/R = 0.033R (R=30), 0.02R (R=50), 0.0125R (R=80)** | rT is ~3-6× finer than Kraken R=80 | `log.blockMesh:59-79` block sizes; Kraken LBM grid: 1 LU = 1/R in radius |
| Mesh type | structured conformal blockMesh (4 O-grid quarters + downstream block, arcs ≡ cylinder) | uniform Cartesian LBM grid, cylinder rasterised on q_wall | **DIFFERENT** (HARD — staircase vs body-fitted) | `blockMeshDict:18-176`; q_wall halfway-BB |
| Total cells (post-mirror) | 24,894 cells | (L_up+L_down)·R · H = (15+15)·R · 4R = 120 R² → R=30: 108k, R=50: 300k, R=80: 768k | Kraken much higher cell count but uniform | `log.mirrorMesh:28` |
| Cd normalisation | **Cd = Σ_patch [(τ + 2·etaS·D(U) - p·ρ·I)·dA]_x / (etaS + etaP)** = viscous Cd (per unit depth, per unit characteristic stress (etaS+etaP)·1 = unit), no factor 2/U²D | Kraken `result.Cd` returned by driver | **POTENTIALLY DIFFERENT** — must inspect Kraken normalisation | `controlDict:83-87`; `viscoelastic_logfv_2d.jl:858-883` (drag computed via `compute_drag_libb_mei_2d` + tau on q_wall) |

---

## 2. Mismatch classification

### HARD (different physics — must reconcile or document interpretation)

H1. **Domain length** — rheoTool L_up=20R + L_down=60R = 80R total streamwise;
    Kraken L_up=L_down=15R = 30R total. **At Wi=1 the polymer wake takes
    O(20-40R) downstream to relax;** Kraken's 15R downstream may force the
    outlet ZouHe-pressure to absorb non-zero stress, biasing Cd.
    *Citations: `blockMeshDict:20,30`; `run_cyl_bigsweep_v2_2d.jl:101-102`.*

H2. **Stabilisation method** — rheoTool uses Pimentel-style `coupling`
    stabilisation (block-coupled p/U solve with τ explicit, no BSD); Kraken
    uses **both-symmetric-discretisation (BSD)** with `bsd_fraction=1.0`,
    which adds the full polymer viscosity into the Newtonian Laplacian and
    subtracts it as an explicit source. BSD changes the effective LBM
    viscosity from ν_s to ν_s + ν_p (= ν_total), so Re_LBM = u·R/ν_total = 1
    while the polymer is reintroduced as a source. **rT does NOT do this.**
    This is the canonical bsd-vs-no-bsd discrepancy the M30/M31 sweep was
    documenting.
    *Citations: `constitutiveProperties:26` `stabilization coupling`;
    `run_cyl_bigsweep_v2_2d.jl:13,100` `KRAKEN_BSD_LIST=1.0`.*

H3. **Cylinder boundary representation** — rheoTool: body-fitted O-grid with
    arc edges (analytic cylinder surface, conformal cell faces); Kraken:
    staircased LBM grid with cut-link q_wall halfway-BB on a uniform
    Cartesian mesh. **At R=30 the staircase introduces O(1/R) drag error
    independent of Wi**, which compounds the polymer drag mismatch.
    *Citations: `blockMeshDict:18-37,76-97` arcs; `viscoelastic_logfv_2d.jl:823`.*

### SOFT (different numerics — may explain some gap)

S1. **Mesh resolution near wall** — rT 0.005R vs Kraken 1/R LU
    (0.013R at R=80, 0.033R at R=30). rT is ~3-6× finer in the boundary
    layer.

S2. **Advection scheme for τ** — rT CUBISTA (TVD bounded), Kraken Rusanov
    (low-order upwind). The M29 audit (memory `project_kernel_dsl`,
    M29b commit `42d2177a`) already identified this as a wake-stress gap;
    MUSCL-superbee is the v2 mitigation.

S3. **Advection scheme for U** — rT CUBISTA, Kraken D2Q9 TRT-LBM. Two
    fundamentally different fluid solvers.

S4. **Time integration** — rT implicit (SIMPLE/PISO) at Co=0.01 with Δt=1e-2;
    Kraken explicit LBM at Ma≈0.0087. Both pseudo-steady — should converge
    to same fixed point if physics matches.

### CONVENTION (different post-processing — must reconcile before any number comparison)

C1. **Cd normalisation** — rT's `outputCd` function object computes
    `Fx_patch = ∫_cyl (τ + 2·etaS·sym(∇U) - p·ρ·I)·n_x dA / (etaS + etaP)`.
    With ρ=1, Umean=1, R=1, depth=1: this is **dimensionless drag per unit
    span normalised by (etaS+etaP)·Umean = 1**, i.e. exactly the Hulsen/Liu
    convention K = Fx/(η_total · U). It is **NOT** the classical Cd =
    2Fx/(ρU²D). The factor 2 is absent and the normalisation uses dynamic
    viscosity not (ρU²D)/2. This matches what `sweep_wi_results.txt` reports
    (Cd_last ≈ 120-131 ≈ Hulsen K~132). **Kraken's `Cd_kraken` must use the
    same K convention** — verify in driver before comparing.

C2. **rheoTool Wi=1.0 NOT TIME-CONVERGED** — Cd_mean_t08 = 116.99 vs Cd_last
    = 120.40, a **3.4 unit drift in the last 2 seconds**. The reference value
    itself is uncertain by ~3%, which is the entire Phase 2-3 gap budget.
    Wi=0.1 and Wi=0.5 are stable (Cd flat at 130.43 and 119.71 respectively).
    *Citations: `bench/rheotool/sweep_wi_results.txt:3-7`;
    `cylinder_wi1.0/Cd.txt` lines 9.6-10.*

C3. **Time-averaging window** — rT `summarize_cd.sh` averages the last 20
    samples (last ~0.2 of endTime, comparable to Kraken's `avg_window_frac
    = 0.2`). OK on convention.

---

## 3. Recommendation for canonical setup

**Option A (preferred): shrink rheoTool domain to match Kraken (L_up=15R,
L_down=15R).**

* Rationale: Kraken's domain is fixed by lattice cost (R=80 already
  ~768k cells at L=30R; doubling to 80R would blow up to ~2M cells per
  case, multiplying campaign cost ×3+). rT is cheap to rerun (~2-3h per
  case on a workstation) once `blockMeshDict` is shortened.
* Action: copy `cylinder_wi{0.1,0.5,1.0}` to `cylinder_wi*_short`, edit
  `system/blockMeshDict` vertices x=-20 → x=-15 and x=60 → x=15, keep
  block i-counts proportional (or accept slightly different cell-density),
  rerun. Document in M32 Phase 2 as the canonical reference.

**Option B (rejected): extend Kraken to L_down=60R.** Tripling the
cell-count for a parameter we don't trust upstream of M32 is wasteful.

**Option C (acceptable fallback if Phase 2 unblocks A): keep both domains
mismatched, but explicitly fit the wake-decay tail in both codes to extract
the "infinite-domain" Cd by Richardson-style extrapolation in L_down.**
Document the residual outlet-effect as a known systematic.

**Independent of A/B/C — three additional gates:**

G1. **Re-run rT Wi=1.0 with endTime=20 to verify time-convergence**
    before any cross-code comparison. The current 3-unit drift between
    t=8 and t=10 is unacceptable as a reference.

G2. **Verify Kraken's `Cd_kraken` normalisation matches rT's
    `Fx_patch / (etaS+etaP)` exactly** (no factor 2, no ρ U² D). Inspect
    `compute_drag_libb_mei_2d` and `compute_polymeric_drag_2d` return
    convention.

G3. **Run a no-polymer Newtonian sanity check** at the same geometry on
    both codes (rT `cylinder_newtonian_re1` already exists in
    `bench/rheotool/`) to isolate geometry/mesh/stabilisation effects
    from polymer effects. If Newtonian Cd at Re=1 differs by >2% between
    rT and Kraken with the **same canonical domain**, no viscoelastic
    comparison is meaningful.

---

## 4. Files cited (all absolute paths)

* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/bench/rheotool/cylinder_wi1.0/system/blockMeshDict`
* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/bench/rheotool/cylinder_wi1.0/system/controlDict`
* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/bench/rheotool/cylinder_wi1.0/system/fvSchemes`
* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/bench/rheotool/cylinder_wi1.0/system/fvSolution`
* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/bench/rheotool/cylinder_wi1.0/system/mirrorMeshDict`
* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/bench/rheotool/cylinder_wi1.0/0/U`
* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/bench/rheotool/cylinder_wi1.0/0/p`
* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/bench/rheotool/cylinder_wi1.0/0/tau`
* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/bench/rheotool/cylinder_wi1.0/constant/constitutiveProperties`
* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/bench/rheotool/cylinder_wi1.0/log.blockMesh` (cell sizes)
* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/bench/rheotool/cylinder_wi1.0/log.mirrorMesh` (cell counts)
* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/bench/rheotool/cylinder_wi1.0/Cd.txt`
* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/bench/rheotool/cylinder_wi{0.1,0.5}/system/controlDict`
* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/bench/rheotool/cylinder_wi{0.1,0.5}/constant/constitutiveProperties`
* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/bench/rheotool/sweep_wi_results.txt`
* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl`
* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/src/drivers/viscoelastic_logfv_2d.jl` (lines 173-555, 806-883)
* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/src/drivers/step_geometry_2d.jl` (lines 214-282)
* `/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/src/fvfd/specs.jl` (lines 88-95)
