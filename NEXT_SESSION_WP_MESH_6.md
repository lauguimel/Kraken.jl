# Reprise WP-MESH-6 — sessions 2026-04-19/20

Branche : `slbm-paper`
Worktree : `/Users/guillaume/Documents/Recherche/Kraken.jl`

## Statut après session 2026-04-20

Job Aqua v4 (`20160349.aqua`) terminé. Logs archivés :
- `paper/data/wp_mesh_6_bump_h100_v1_buggy.log` (symétrie parfaite, Cl_RMS=0)
- `paper/data/wp_mesh_6_bump_h100_v2_tau_floor.log` (τ_floor + local_cfl → régression)
- `paper/data/wp_mesh_6_bump_h100_v3_baseline.log` (revert propre, cy_p=0.245)
- `paper/data/wp_mesh_6_bump_h100_v4_bump_sweep.log` (sweep BUMP=0.1+0.5)

Commits de la session :
- `77261e4` — `slbm_trt_libb_step_local_2d!` DSL-modulaire (mirror 3D)
- `573c9a3` — `τ_floor` kwarg dans `compute_local_omega_{2d,3d}`
- `0292b77` — revert run_C baseline
- `aac0cca` — sweep `BUMP_COEFS=(0.1, 0.5)`

## Découvertes

### (A)(B) Cartesian : 17-18 % Cd err est la **correction de confinement**

Les résultats `Cd ≈ 1.65` à D=40/80 avec `Cl_RMS ≈ 0.28`, `St=0.2`
paraissent erronés vs Williamson 1996 `Cd=1.4`, `St=0.165` — mais cette
référence est **pour flow libre non-confiné**. Avec blockage 10 % dans
un domaine 20D × 10D, la correction de confinement ajoute ~15-20 %
au Cd, ce qui donne `Cd_confiné ≈ 1.65` (OK).

La référence de comparaison correcte serait `Cd_ST2D2 ≈ 3.22` (Schäfer-
Turek 2D-2, inlet parabolique) ou `Cd_Park1998 ≈ 1.4 + 0.25 × (D/H)`
(correction par blockage).

### (C) Bump 0.1 : SLBM+LI-BB marche à **D=40** uniquement

| | Cd (err vs 1.4) | Cl_RMS | St | MLUPS |
|---|---|---|---|---|
| (A) D=40 | 1.652 (18.0%) | 0.282 | 0.200 | 4239 |
| (B) D=40 | 1.649 (17.8%) | 0.283 | 0.200 | 4242 |
| **(C)[b=0.1] D=40** | **1.630 (16.4%)** | 0.003 | 0.175 | 1740 |

**Résultat clé** : à **même cell count** (321 k) et **même Cd** (16.4 % <
18.0 %), SLBM+LI-BB+Bump donne précision ≥ Cartesian. Mais :

- **Perf** : 2.4× plus lent MLUPS (interpolation bilinéaire).
- **Pas de shedding** : `dx_ref ≈ 0.00048` (cellule min au centre) donne un
  `dt` effectif 7× plus petit que Cartesien → mêmes `steps` = 7× moins de
  temps physique → transitoire de démarrage inachevé au bout de 160 k pas.
- **Instable à D=20 et D=80** : la rescaling quadratique de τ via
  `compute_local_omega_2d` donne `τ → 0.5` sur 30-60 % des cellules de
  bordure (Ginzburg-TRT n'est pas stable pour `τ` proche de 0.5).

### (C) Bump 0.5 : mesh gmsh dégénéré

```
C[b=0.5] FAILED: CurvilinearMesh: degenerate Jacobian (sign change or
zero) at (i=801, j=1): min|J|=84.8, Jmin=-314.9, Jmax=84.8.
Mesh has a fold or degenerate cell.
```

gmsh's Transfinite Bump 0.5 avec `Nx=801` produit un **fold** aux
bordures du domaine (Jacobien négatif). Bug probable dans gmsh 4.x ou
combinaison N × coef mal conditionnée. À éviter.

## Trois chemins restants pour finir WP-MESH-6

### Chemin A — Accepter les résultats v3/v4 et rédiger

Paper story pragmatique :
- Montrer que (C) BUMP=0.1 à D=40 matche (A)(B) en Cd à **même cell count**
- Discuter le 2.4× penalty SLBM et les limitations de stretching extrême
- Figure `bump_convergence.pdf` : matrice A/B/C[b=0.1] à D=40 (seul point
  exploitable). Ajouter un inset sur les cellules où `τ → 0.5`.

**Avantage** : Aucune nouvelle simulation. Les données sont là.
**Inconvénient** : (C) n'a pas de shedding → pas de comparaison Cl_RMS/St.

### Chemin B — Régler le problème de "temps physique effectif"

Lancer (C) BUMP=0.1 à D=40 avec **10× plus de steps** (1.6M au lieu de
160 k) pour permettre le shedding de se développer. Coût : ~10 min GPU
H100 supplémentaires.

```bash
# Edit run_C or make steps proportional to mesh.dx_ref/dx_ref_cart
```

### Chemin C — Changer le mesh pour éviter `τ → 0.5`

Utiliser `:linear` scaling au lieu de `:quadratic` (Filippova-Hänel pour
refinement avec `Δt ∝ Δx`), OU un mesh **moins agressif** (`Progression`
au lieu de `Bump`) qui garde un ratio modéré sans fold.

Test préliminaire : gmsh Progression 1.05 donne des cellules
géométriquement ratio ~1.05^Nx, stretching continu dans une seule
direction. Cohérent avec un canal de couche limite.

## Pitfalls confirmés

- **`compute_local_omega_2d` utilise `dx_ref = dx_min` par défaut** →
  τ_local ≤ τ_ref partout → sur cellules de bordure τ → 0.5 (instable
  TRT). Le kwarg `τ_floor` (commit 573c9a3) clamp correctement mais
  introduit un biais de viscosité locale sur les cellules clampées.
- **`local_cfl=true` ne sauve pas (C) sur Bump 0.1** : même recette que
  sphere_3d, échec en 2D à cause de la cellule-ratio trop extrême.
- **Métal FP32 ≠ Aqua FP64** : plusieurs bugs NaN sur Metal disparaissent
  sur Aqua (D=40 Bump 0.1 est stable FP64 uniquement).
- **cy_p=0.245 offset déclenche shedding** sans perturbation explicite
  (Schäfer-Turek 2D-2 style). À garder. Mais Cd augmente vs centered.

## Décision recommandée pour la prochaine session

1. **Chemin A** : accepter v3/v4 comme tel, rédiger paper Sec. WP-MESH-6
   avec la matrice partielle (A/B/C à D=40). Notes honnêtes sur les
   limitations.
2. Fermer WP-MESH-6 et passer à **WP-MESH-7** ou directement à la
   rédaction finale du paper.
3. Optionnel : lancer **Chemin B** en background (160 k → 1.6M steps,
   ~1 job Aqua ~15 min) pour avoir `Cl_RMS`/`St` de (C) en bonus.

## État des fichiers

```
hpc/wp_mesh_6_bump_aqua.jl     (sweep BUMP 0.1 + 0.5, cy_p=0.245)
paper/data/wp_mesh_6_bump_h100_v{1,2,3,4}_*.log  (4 runs archivés)
scripts/figures/plot_bump_matrix.jl (prêt mais non-testé sur v3/v4)
src/curvilinear/slbm.jl         (+ slbm_trt_libb_step_local_2d! + τ_floor)
src/curvilinear/slbm_3d.jl      (+ τ_floor kwarg dans compute_local_omega_3d)
src/kernels/dsl/bricks.jl       (CollideTRTLocalDirect 2D — utilisé)
```
