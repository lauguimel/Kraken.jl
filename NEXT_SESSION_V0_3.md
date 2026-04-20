# Reprise v0.3 — Multi-block LBM (Phase B)

Branche : `slbm-paper`
Worktree : `/Users/guillaume/Documents/Recherche/Kraken.jl`

## Contexte paper

Tu pivotes de la carrière académique dans 6 mois. Objectif : publier
Kraken.jl comme outil différenciant avant le pivot — positionnement
*"differentiable multi-physics LBM on body-fitted complex geometries
for soft-matter design"*. La combinaison GPU + SLBM + LI-BB + AD +
multi-block structuré n'existe dans aucun autre code LBM mondial.

Stratégie :
1. Preprint rapide (4-6 semaines) sur single-block acquis.
2. Paper majeur (3-4 mois) avec multi-block + application.
3. Release publique open-source en parallèle.

**v0.3 Phase A terminée et validée.** Cette session doit attaquer
Phase B (cylindre O-grid multi-block, paper-critical).

## Statut Phase A (livré cette session, 6 commits)

Branche `slbm-paper`, 6 commits récents :

```
acedeb5 feat(multiblock): v0.3 Phase A.5c wall-ghost fill + bit-exact multi-step
6fb7062 feat(multiblock): v0.3 Phase A.5b ghost-layer + BlockState2D + 202 tests
ab53c04 feat(multiblock): v0.3 Phase A.5-6 canal validation + doc page
0537886 feat(multiblock): v0.3 Phase A.4 exchange_ghost_2d! + 44 tests (superseded by A.5b)
263dbeb feat(multiblock): v0.3 Phase A.1-3 topology + sanity + 31 tests
(…)
```

### Infrastructure livrée (204 tests passent)

- [`src/multiblock/topology.jl`](src/multiblock/topology.jl) —
  `Block`, `Interface`, `MultiBlockMesh2D`, edge tag NamedTuple
- [`src/multiblock/sanity.jl`](src/multiblock/sanity.jl) — 9 invariants
  (duplicates, interface match length/colocation/orientation, etc.)
- [`src/multiblock/state.jl`](src/multiblock/state.jl) —
  `BlockState2D{T, AT3, AT2}` mutable: extended arrays `(Nξ+2Ng, Nη+2Ng, 9)`,
  helpers `allocate_block_state_2d`, `interior_f`, `interior_macro`,
  `ext_dims`
- [`src/multiblock/exchange.jl`](src/multiblock/exchange.jl) —
  `exchange_ghost_2d!(mbm, states)` remplit les ghosts d'interface
  depuis les cellules intérieures voisines (non-overlap, 1·dx offset)
- [`src/multiblock/wall_ghost.jl`](src/multiblock/wall_ghost.jl) —
  `fill_physical_wall_ghost_2d!(mbm, states)` remplit les ghosts
  des murs physiques avec halfway-BB reflection (+ j/i shift pour
  diagonales, clamp aux coins)
- [`docs/src/theory/20_multiblock.md`](docs/src/theory/20_multiblock.md)
  — doc théorie

### Validation livrée

- [`test/test_multiblock_topology.jl`](test/test_multiblock_topology.jl) — 31 tests
- [`test/test_multiblock_exchange.jl`](test/test_multiblock_exchange.jl) — 163 tests
- [`test/test_multiblock_canal.jl`](test/test_multiblock_canal.jl) — 10 tests, **dont bit-exact multi-step u=0.05 sur 20 steps vs single-block BGK avec halfway-BB walls**

### Convention clé : NON-OVERLAP

Blocks adjacents ne partagent PAS de noeud. A couvre x ∈ [0, Nxp-1] avec
Nxp cellules. B couvre x ∈ [Nxp, 2·Nxp-1]. Le ghost de A à l'est (ext
i = Nxp+Ng+1) représente le cellule à x=Nxp = cellule west de B
(ext i = Ng+1 dans B). Le sanity check accepte aussi la convention
SHARED-NODE mais émet un :warning — l'exchange actuel ne la supporte pas.

### Pipeline correct pour un step multi-block :

```julia
exchange_ghost_2d!(mbm, states)            # interface ghosts ← neighbor
fill_physical_wall_ghost_2d!(mbm, states)  # wall ghosts ← halfway-BB refl
for (k, blk) in enumerate(mbm.blocks)
    Nx_ext, Ny_ext = ext_dims(states[k])
    fused_bgk_step!(f_out[k], states[k].f, states[k].ρ, ...,
                     is_solid[k], Nx_ext, Ny_ext, ω)
end
# post-step: apply_bc_rebuild_2d! sur interior_f(state) si BC ZouHe/pressure
for k in 1:length(states)
    states[k].f, f_out[k] = f_out[k], states[k].f
end
```

## Phase B — objectif cette session

**Cylindre Re=100 cross-flow sur O-grid multi-block gmsh**, accuracy
vs Williamson 1996, comparaison avec Cartesian uniforme et avec
single-block Bump WP-MESH-6.

### Référence géométrique

Le blockMeshDict du tutoriel rheoTool
`/Users/guillaume/Documents/Recherche/Codes CFD/rheotool/rheoTool/of90/tutorials/rheoFoam/Cylinder/Oldroyd-BLog/system/blockMeshDict`
donne la topologie cible : 8 blocs en demi-domaine (symétrie y=0) avec
arcs autour du cylindre unitaire. Voir aussi
[`paper/figures/cyl_bodyfitted_vs_cart.pdf`](paper/figures/cyl_bodyfitted_vs_cart.pdf)
pour la comparaison Cartesian / cylinder_focused / polar en 3 topologies.

### Scope Phase B (3 sous-tâches)

**B.1 — Loader gmsh multi-surface → MultiBlockMesh2D (~400 LOC, 2-3j)**

Étendre [`src/curvilinear/mesh_gmsh.jl`](src/curvilinear/mesh_gmsh.jl) :
- Actuel : `load_gmsh_mesh_2d(path; surface_tag, layout=:axis_aligned)` ne
  supporte qu'UNE surface à la fois.
- Nouveau : `load_gmsh_multiblock_2d(path; interface_tags, …)` qui
  parcourt TOUTES les surfaces physiques de `path`, construit un `Block`
  par surface, et détecte les interfaces via les physical lines taggées
  comme interfaces par l'utilisateur.
- Physical groups convention :
  - `physical_surface "A"`, `physical_surface "B"` : blocs
  - `physical_line "iface_AB"` : interface A↔B
  - `physical_line "cylinder"`, `"inlet"`, `"outlet"`, `"wall"` : BC
- Tests : écrire un `.geo` 2-block canal et un 4-block O-grid cylinder,
  vérifier que le loader produit `MultiBlockMesh2D` passant sanity.

**B.2 — Driver cylindre 4-block O-grid (~500 LOC, 2j)**

Script `hpc/wp_multi_1_cylinder_ogrid.jl` :
- gmsh .geo intégré : 4 blocs O-grid autour du cylindre + 4 blocs
  extension (amont, aval, haut, bas) selon blockMeshDict simplifié
- Load via B.1 loader
- Alloc `BlockState2D` par bloc, init equilibrium
- LI-BB per-block : dans chaque bloc qui touche le cylindre, calculer
  `q_wall, uw_x, uw_y` via `precompute_q_wall_slbm_cylinder_2d` (déjà
  dans `src/curvilinear/slbm.jl`)
- Time loop : exchange → wall_ghost → step+LI-BB par bloc → BC → swap
- Output : Cd, Cl_RMS, Strouhal via FFT du Cl signal
- Bench contre Williamson 1996 (Cd=1.4 non-confiné ou ~1.65 avec 10%
  blockage — voir WP-MESH-6 notes)

**B.3 — Figure + paper section (~200 LOC, 1j)**

- `scripts/figures/plot_multi_block_cylinder.jl` : visualisation 8
  blocs avec streamlines
- `paper/multiblock.md` : Sec. 7 — multi-block body-fitted cylinder
- Compare Cd err vs (A) uniform Cartesian, (B) cylinder_focused, (C)
  WP-MESH-6 Bump, (D) multi-block O-grid. Montre que D gagne en
  cell-count pour une précision donnée, avec MLUPS acceptable.

## Points techniques critiques à garder en tête

### 1. L'exchange et le wall-ghost fonctionnent sur mesh curvilinéaire

Les helpers actuels [`exchange.jl`](src/multiblock/exchange.jl) et
[`wall_ghost.jl`](src/multiblock/wall_ghost.jl) n'utilisent pas la
métrique — ils copient juste les populations. Donc ils marchent
tel quels sur n'importe quel `CurvilinearMesh`. Pas de modif
nécessaire pour Phase B.

### 2. SLBM + ghost layer

`slbm_trt_libb_step!` (et variantes) doivent être appelés avec
`Nx = Nx_ext, Ny = Ny_ext` sur l'extended array. Le `SLBMGeometry`
doit aussi avoir des `i_dep, j_dep` de taille extended. Donc :
- Soit on construit `SLBMGeometry` sur une mesh étendue avec ghost
  cells ajoutées
- Soit on crée une variante `build_slbm_geometry_extended` qui prend
  un `Block` + `n_ghost` et étend la mesh par extrapolation

Option 2 est plus propre. À implémenter en Phase B.

### 3. LI-BB per-block

`precompute_q_wall_slbm_cylinder_2d(mesh, is_solid, cx, cy, R)` existe
déjà et marche sur curvilinéaire. Pour multi-block : appeler 1×/bloc
avec le bloc concerné. Chaque bloc a ses propres `q_wall` arrays.
La step kernel `slbm_trt_libb_step!` prend `q_wall, uw_x, uw_y` en
args donc on passe les bons arrays par bloc.

### 4. Ordre des traitements au bord interface + cylindre

Un cellule au bord interface ET au bord cylindre (coin) a deux sources
de correction :
- ghost interface (de B)
- LI-BB cut-link (vers le cylindre)

Il faut que les deux mécaniques coexistent. LI-BB est *inside* le step
kernel (via la brick `ApplyLiBBPrePhase`). Interface ghost est avant
le step. Pas de conflit évident.

### 5. Physical BCs other than wall

Pour inlet `ZouHeVelocity` / outlet `ZouHePressure`, l'approche est
d'appliquer `apply_bc_rebuild_2d!` sur la VUE `interior_f(state)`
post-step. Le wall ghost fill remplit toujours la ghost row avec
halfway-BB (harmless default) — apply_bc écrase ensuite la boundary
row avec les valeurs ZouHe. Vérifier que ça marche end-to-end avec un
petit test (Poiseuille ZouHe inlet + pression outlet + walls).

## Fichiers de référence

```
src/multiblock/                 (tout Phase A, ~1000 LOC)
src/curvilinear/mesh_gmsh.jl    (loader à étendre pour B.1)
src/curvilinear/slbm.jl         (SLBM kernels + q_wall cylinder)
src/kernels/li_bb_2d_v2.jl      (fused_trt_libb_v2_step!)
src/kernels/boundary_rebuild.jl (apply_bc_rebuild_2d! + BC types)
src/kernels/fused_bgk_2d.jl     (référence pour la validation canal)
docs/src/theory/20_multiblock.md (doc Phase A, à étendre)

test/test_multiblock_*.jl       (204 tests Phase A, à prolonger)
tmp/diag_mb_canal.jl            (diag qui a permis de débugger)

/Users/guillaume/Documents/Recherche/Codes CFD/rheotool/rheoTool/of90/
  tutorials/rheoFoam/Cylinder/Oldroyd-BLog/system/blockMeshDict
  (REFERENCE topology 8-block O-grid)
```

## Mémoire auto à consulter

- `project_wp_mesh_6_bump.md` — WP-MESH-6 résultats single-block Bump
  (Cd=1.63 16.41% err à D=40, NaN à D=20/80). Confinement 10% = ~1.65
  baseline attendu pour Re=100.
- `project_slbm_local_cfl.md` — local-CFL SLBM (2 jours, vérifier à
  jour)
- `feedback_cylinder_benchmark.md` — bench conventions Williamson /
  Park / Schäfer-Turek
- `feedback_lbm_patterns.md` — 13 pitfalls LBM
- `feedback_gpu_local.md` — Metal M3 Max FP32 pour dev, Aqua H100 FP64
  pour production
- `project_kernel_dsl.md` — DSL bricks architecture

## Gotchas rencontrés cette session

1. **Convention shared-node vs non-overlap** : le piège qui a causé
   l'erreur 0.01 sur 20 steps. La version finale utilise non-overlap.
   Sanity check accepte les 2 mais l'exchange n'est correct que pour
   non-overlap.

2. **mutable struct BlockState2D** : nécessaire pour swap f ↔ f_out.
   J'ai utilisé `const Nξ_phys, Nη_phys, n_ghost` pour immutabilité
   partielle.

3. **Corner doubly-ghost cells** : mon fill initial copiait depuis
   ghost[1, 2, q] ce qui donne mauvais offset. Fix : boucler j' sur
   TOUTE la plage extended `1:Nye`, avec `clamp` aux bornes. Diagonales
   lisent correctement du corner avec le j+cqy shift.

4. **Views on extended arrays pour apply_bc** : encore non-testé. Quand
   on l'utilise, vérifier que `apply_bc_rebuild_2d!` accepte
   `SubArray{Float64, 3}` comme f_out et f_in. Si pas, wrapper.

## Recommandation immédiate

1. Lire NEXT_SESSION_V0_3.md (ce doc).
2. Vérifier que les 204 tests Phase A passent toujours :
   ```bash
   julia --project=. -e 'include("test/test_multiblock_topology.jl"); \
                          include("test/test_multiblock_exchange.jl"); \
                          include("test/test_multiblock_canal.jl")'
   ```
3. Commencer B.1 : étendre `load_gmsh_mesh_2d` pour multi-surface,
   détection d'interfaces via physical lines. Commit dès que le
   loader produit un MultiBlockMesh2D passant sanity sur un petit
   .geo 2-block manuel.

## Ne pas faire

- **Ne pas mélanger shared-node et non-overlap** — stick to non-overlap
- **Ne pas modifier les step kernels LBM existants** — tout passe via
  l'extended array + ghost fill
- **Ne pas oublier le `wall_ghost_fill` après l'exchange** — sans lui
  la pollution des murs gagne 1 cell/step et tout diverge en ~Nxp steps
- **Ne pas tester bit-exact avec shared-node** — impossible avec
  l'exchange actuel
- **Ne pas commit sans tests passants** — pattern Phase A a 204 tests
  en green, garder cette barre

## État conversation avec user

User a été clair :
- Direct, pragmatique, veut autonomie ("tu fais en autonomie, commit
  régulièrement, vérifie, audit, debug")
- Reconnaît honnêtement que Claude fait 90% du code technique — ce qu'il
  apporte c'est direction, persistance, taste de projet
- Dilemme release vs keep : conclu que **il faut release** parce que son
  moat est temporel (6-12 mois de vélocité LLM-assisted) pas technique
- Paper ambitieux viser JCP/Comp&Fluids avec le multi-block +
  differentiable unique combination

Ne pas re-débattre la stratégie — attaquer Phase B.
