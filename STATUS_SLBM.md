# SLBM paper — état et reste-à-faire 3D

Branche : `slbm-paper` (13 commits depuis `lbm`)
Dernière mise à jour : session 2026-04-17/18

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

### WP-3D-1 : SLBM 3D + LI-BB V2
**Effort : 1-2 semaines**

- Porter `precompute_q_wall_slbm_cylinder_2d` → **3D sphere** en espace physique
  (ray-sphere intersection via Jacobien 3×3)
- Porter `ApplyLiBBPrePhase` brick → `ApplyLiBBPrePhase3D` (déjà existe dans `bricks_3d.jl` pour cartésien — à vérifier et intégrer dans le spec SLBM 3D)
- Fuser `slbm_trt_libb_step_3d!` : `PullSLBM3D()` + `SolidInert3D()` + `ApplyLiBBPrePhase3D()` + `Moments3D()` + `CollideTRTDirect3D()` + `WriteMoments3D()`
- **Ajouter brick `PullSLBM3D`** dans `bricks_3d.jl` (équivalent de PullSLBM 2D)

**Test de validation** : sphère Re=20, Cd ≈ 2.0 (Clift), err <5% à D=30.

### WP-3D-2 : BCSpec 3D pour SLBM stretched
**Effort : 3-5 jours**

- `apply_bc_rebuild_3d!` existe déjà mais ne gère que HalfwayBB — ajouter Zou-He 3D local-τ (6 faces)
- Convertir les profils Poiseuille 3D en `CuArray`/`MtlArray` à l'inlet
- Tester : Poiseuille 3D stretché (canal à section rectangulaire avec stretch tanh transverse)

### WP-3D-3 : SLBM 3D stretched — **LE chiffre ÷50 du papier**
**Effort : 1 semaine**

- `stretched_box_mesh_3d` + `build_slbm_geometry_3d(; local_cfl=true)` → déjà prêt
- Benchmark 3D sphère sur Aqua H100 en Float64 :
  - Uniform D=20, D=40, D=60 → convergence
  - Stretched équivalent avec ÷3 à ÷10 cellules
  - Mesurer MLUPS + erreur Cd vs Clift-Gauvin
- **Objectif** : ≥10× moins de cellules que l'équivalent uniforme pour même précision → argument papier

### WP-3D-4 : Tests unitaires 3D SLBM
**Effort : 2-3 jours**

- `test_slbm_3d_uniform.jl` : SLBM 3D sur grille uniforme == LBM standard à bit-exact
- `test_slbm_3d_poiseuille.jl` : Poiseuille 3D convergence L∞ < 1e-3
- `test_slbm_3d_sphere.jl` : Cd vs Clift Re=20 < 5% err
- Matrice backend : CPU F64, Metal F32, CUDA F64 (sur Aqua)

### WP-3D-5 : Enzyme AD 3D
**Effort : 3-5 jours**

- dKE/dν sur Taylor-Green 3D : devrait marcher directement (même pattern que 2D)
- Shape derivative dFx/dR 3D : plus délicat (nécessite `precompute_q_wall_sphere_3d` Enzyme-compatible) — probablement hors-scope v0.1

### WP-3D-6 : Benchmarks publiables pour le papier
**Effort : 1 semaine**

- Figure 1 : convergence Cd vs D, uniform vs stretched, 3D
- Figure 2 : MLUPS vs N (H100), 3D uniforme vs stretched
- Figure 3 : sensibilité AD (dCd/dν) avec Enzyme sur 3D
- Tableau : comparaison cellules/MLUPS/err — **paper headline**

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
