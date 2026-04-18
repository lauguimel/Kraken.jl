# SLBM paper — état et reste-à-faire 3D

Branche : `slbm-paper` (19 commits depuis `lbm`)
Dernière mise à jour : session 2026-04-18 (WP-3D-1, 4, 2, 5, 3, 6 livrés)

## Ce qui est fait

### Contributions validées

| # | Contribution | Preuve |
|---|---|---|
| 1 | **Local-CFL SLBM** (nouveau) | 30% → **1.7% err** (stretched 550×100 s=0.5, Metal F32) |
| 2 | **DSL modulaire PullSLBM** | 0.02% match vs cartésien à D=80 |
| 3 | **SLBM 3D port** (D3Q19, Metal) | 112 MLUPS sur 96³, ρ conservé machine precision |
| 4 | **Enzyme AD sur SLBM 2D BGK** | dKE/dν : **0.0% error** vs diff finies (Metal + Aqua H100) |

### Infrastructure

- `src/curvilinear/mesh.jl` : CurvilinearMesh + `build_slbm_geometry(; local_cfl=true)`
- `src/curvilinear/mesh_3d.jl` : CurvilinearMesh3D + `build_mesh_3d` (ForwardDiff 3×3) + `stretched_box_mesh_3d` + `cartesian_mesh_3d`
- `src/curvilinear/slbm.jl` : SLBMGeometry + `precompute_q_wall_slbm_cylinder_2d` + `compute_local_omega_2d`
- `src/curvilinear/slbm_3d.jl` : SLBMGeometry3D + `build_slbm_geometry_3d(; local_cfl)` + `trilinear_f` + `slbm_bgk_step_3d!`
- `src/kernels/dsl/bricks.jl` : `PullSLBM`, `CollideTRTLocalDirect`, `RescaleNonEq`
- `src/kernels/boundary_rebuild.jl` : murs sud/nord HalfwayBB + BCs `_local` avec τ par cellule

### Benchmarks H100 (CUDA Float64)

| Config | Cellules | Cd | Err | MLUPS |
|---|---|---|---|---|
| Uniform D=20 | 37k | 5.723 | 2.57% | 105 |
| Uniform D=40 | 145k | 5.655 | 1.34% | 1479 |
| Uniform D=80 | 579k | 5.607 | **0.49%** | **2214** |
| Stretched 550×100 s=1.0 | 55k | 5.441 | 2.48% | 910 |
| Stretched 881×165 s=1.0 | 145k | 5.317 | 4.71% | 1497 |

### Bugs pré-existants fixés en cours de route

- OOB segfault dans `PullHalfwayBB` (ifelse + @inbounds) — fix : clamper les indices
- World-age Julia 1.12 dans `build_lbm_kernel` — fix : retourner le ctor depuis `Core.eval` directement
- Murs sud/nord non gérés par BCSpec — fix : kernels HalfwayBB dédiés

## Ce qui reste pour une **3D propre** papier-ready

### WP-3D-1 : SLBM 3D + LI-BB V2 ✅ (commit `bf25ea5`)

Livré :
- `PullSLBM_3D` brick DSL (trilinéaire) — `src/kernels/dsl/bricks_3d.jl`
- `precompute_q_wall_slbm_sphere_3d` (ray-sphère espace physique) — `src/curvilinear/slbm_3d.jl:296`
- `slbm_trt_libb_step_3d!` fusé via spec `PullSLBM_3D + SolidInert_3D + ApplyLiBBPrePhase_3D + Moments_3D + CollideTRTDirect_3D + WriteMoments_3D`
- Smoke 40×20×20, R=4 : 1098 cuts, mean(ρ)=0.9992, no NaN

### WP-3D-2 : BCSpec 3D pour SLBM stretched ✅ (commit `8c9bcb0`)

Livré :
- 4 nouveaux kernels halfway-BB transverses (south/north/bottom/top) gated par `apply_transverse=true`
- Local-τ Zou-He 3D (`_bc_west_zh_velocity_local_3d!`, `_bc_east_zh_pressure_local_3d!`) lisant `sp_field[i,j,k]`
- `apply_bc_rebuild_3d!` étendu avec kwargs `sp_field`, `sm_field`, `apply_transverse` (back-compat préservée pour `fused_trt_libb_v2_step_3d!`)
- Brick `CollideTRTLocalDirect_3D` + `slbm_trt_libb_step_local_3d!` via spec dédié
- `compute_local_omega_3d` — sp/sm 3D avec scaling :quadratic ou :linear

### WP-3D-3 : SLBM 3D stretched — **chiffre headline du papier** ✅ scripts (commit `7c1e582`), bench Aqua **Q**

- `hpc/slbm_sphere_h100.jl` : Uniform D=10/20/30 + Stretched D=20/30 ×{0.5, 1.0}
- `hpc/slbm_sphere_h100.pbs` : 1× H100, 6 h walltime
- Soumis `qsub` → job `20145714.aqua` (état Q derrière krk_lc48 et krk_alves)
- Sanity Cartesian local OK (Fx=0.114 sur D=4) ; stretched mild (s=0.5, D=6) OK Fx=0.165

### WP-3D-4 : Tests unitaires 3D SLBM ✅ (commit `922c7b6`)

24/24 passent dans `test/test_slbm_libb_3d.jl` :
- Conservation : flow uniforme stable < 1e-12 sur 50 steps
- Bit-exact intérieur : `slbm_trt_libb_step_3d!` ≡ `fused_trt_libb_v2_step_3d!` < 1e-12 (no cuts)
- Sphere q_wall sanity : cuts > 0, q_w ∈ (0, 1]
- Poiseuille 3D rect duct (parabolic inlet, halfway-BB 4 walls, no-slip OK)
- `compute_local_omega_3d` ≡ `trt_rates(ν)` sur grille uniforme < 1e-10
- Local-τ kernel ≡ uniform kernel sur grille uniforme < 1e-12

### WP-3D-5 : Enzyme AD 3D ✅ (commit `1c67a9e`)

dKE/dν sur Taylor-Green 3D 24³ × 100 steps (CPU Float64) :
- KE(ν=0.1) = 5.32 × 10⁻²
- dKE/dν (FD)     = -1.49831
- dKE/dν (Enzyme) = -1.49831
- **Erreur relative = 0.00 %**
- Compile-once 160 s, runs cachés ensuite

Shape derivative dFx/dR 3D : non tenté (nécessite `precompute_q_wall_sphere_3d` Enzyme-compatible — déféré v0.2)

### WP-3D-6 : Benchmarks publiables pour le papier ✅ infra (commit en cours)

- `paper/3d_extension.md` : section Sec. 4 du papier (4.1 D3Q19 SLBM, 4.2 LI-BB+BCSpec, 4.3 sphère Re=20 + tableau placeholder, 4.4 AD 3D, 4.5 perf)
- `scripts/figures/plot_sphere_3d_convergence.py` : parse `slbm_sphere_3d.log`, sortie `paper/figures/sphere_3d_convergence.pdf` (err vs cells + MLUPS vs cells, log-log, uniform vs stretched)
- À faire après job Aqua : `pull_results_from_aqua.sh` → `python plot_sphere_3d_convergence.py --log results/slbm_sphere_3d.log` → remplir `[…]` dans `paper/3d_extension.md`

## Problèmes ouverts

1. **Enzyme AD shape derivative (dFx/dR 2D)** : crashe silencieusement sur Aqua.
   - `forward_drag` alloue `q_wall` via `precompute_q_wall_cylinder` → Enzyme rule à écrire
   - Workaround possible : pré-calculer `q_wall(R)` comme fonction explicite différentiable

2. **Stretching saturation à haute résolution 2D** :
   - Uniform D=80 : 0.49% err ; Stretched 1761×329 s=1.0 : 5.47% err
   - Le stretching ne converge pas vers 0 avec le raffinement
   - Hypothèse : interpolation bilinéaire O(Δx²) sature sur maillages fortement non-uniformes
   - À tester : biquadratique (Wilde 2020) pour obtenir O(Δx³)

3. **`cylinder_focused_mesh` dégrade le Cd** (26% err vs 1.7% pour `stretched_box_mesh`)
   - Mapping tanh piecewise crée une discontinuité de dérivée au focus
   - Solution possible : mapping sinh C∞ au lieu de tanh piecewise

## Feuille de route proposée pour la prochaine session

**Jour 1-2 : WP-3D-1** (LI-BB 3D + DSL 3D)
**Jour 3 : WP-3D-2** (BCSpec 3D complet)
**Jour 4-5 : WP-3D-3** (benchmark sphère Aqua H100) — **milestone papier**
**Jour 6-7 : WP-3D-4** (tests + WP-3D-5 AD 3D)
**Jour 8-10 : WP-3D-6** (figures + rédaction section 3D)

Total : **2 semaines** pour arriver à un papier JCP complet avec 3D convaincant.
