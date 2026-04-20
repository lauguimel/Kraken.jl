# Reprise v0.3 — approach (3) Multi-block O-grid Re=100 (Phase B.2+)

Branche : `slbm-paper`.

## Contexte paper — état 2026-04-21

La matrice comparative WP-MESH-6 vise 4 approches sur **même nombre de
cellules** pour le cylindre Re=100 (Williamson 1996, Schäfer-Turek 2D-2) :

| # | Approche | Mesh | Status |
|---|---|---|---|
| (A) | Cartésien uniforme + halfway-BB | `cartesian_mesh` | ✅ Aqua H100, D=40 Cd=1.652 |
| (B) | Cartésien uniforme + LI-BB v2 | `cartesian_mesh` | ✅ Aqua H100, D=40 Cd=1.649 |
| (C) | gmsh Bump (clustering sur parois rect) + SLBM + LI-BB | gmsh Transfinite Bump | ✅ Aqua D=40 Cd=1.630 ; NaN D=20/80 |
| **(D)** | **cylinder_focused_mesh (clustering sur cylindre) + SLBM + LI-BB** | `cylinder_focused_mesh` | **🟡 driver committé `2facd3d`, à submit sur Aqua** |
| **(E)** | **Multi-block O-grid + extensions rect + SLBM + LI-BB per block** | gmsh multi-surface | **🔴 infrastructure B.2.1 ✓, reste B.2.2 + B.2.3 + driver** |

## Livré cette session (3 commits)

1. **`9d08e74`** — v0.3 Phase B.1 gmsh multi-surface loader + 44 tests
   - `load_gmsh_multiblock_2d(path; layout=:auto)` in
     [src/multiblock/mesh_gmsh_multiblock.jl](src/multiblock/mesh_gmsh_multiblock.jl)
   - Canal 2-block + O-grid 4-block cylindre validés
   - Sanity colocation check flip-aware
2. **`2facd3d`** — approche (D) cylinder_focused single-block
   - Driver production CUDA : [hpc/wp_mesh_6_focus_aqua.jl](hpc/wp_mesh_6_focus_aqua.jl)
   - PBS : [hpc/wp_mesh_6_focus_aqua.pbs](hpc/wp_mesh_6_focus_aqua.pbs)
   - Smoke Metal : [tmp/smoke_focus_local_metal.jl](tmp/smoke_focus_local_metal.jl)
     (F32 instable à D=20/40 — même symptôme que bump smoke, instabilité
     F32 pas F64, à valider sur Aqua)
   - Pipeline : `cylinder_focused_mesh` + `local_cfl=true` +
     `compute_local_omega_2d(:quadratic)` + `slbm_trt_libb_step_local_2d!`
   - Sweep strength ∈ (1.0, 2.0) sur 3 résolutions D ∈ (20, 40, 80)
3. **`4451b7f`** — v0.3 Phase B.2.1 shared-node ghost exchange + 363 tests
   - `exchange_ghost_shared_node_2d!` in
     [src/multiblock/exchange.jl](src/multiblock/exchange.jl)
   - Lit depuis `interior[ng + 1 + k]` au lieu de `interior[ng + k]`
     (skip la colonne dupliquée à l'interface partagée)
   - Tests : [test/test_multiblock_exchange_shared.jl](test/test_multiblock_exchange_shared.jl)

**Total multiblock v0.3 : 611 tests verts** (31 + 163 + 10 + 44 + 363).

## À faire en début de prochaine session

### Priorité 1 — Submit (D) sur Aqua pour résultats paper

```bash
cd ~/Documents/Recherche/Kraken.jl
./hpc/sync_to_aqua.sh         # rsync branche slbm-paper
ssh maitreje@aqua.qut.edu.au
cd Kraken.jl && qsub hpc/wp_mesh_6_focus_aqua.pbs
```

Attendre ~6h, puis `./hpc/pull_results_from_aqua.sh`.

**Résultat attendu** : Cd ≈ 1.64 à D=40 pour strength=1 et strength=2,
Cl_RMS ≈ 0.28, MLUPS meilleur que Bump (par meilleure condition du τ).

### Priorité 2 — B.2.2 block reorientation OU .geo canonique

Le topological walker donne des orientations ξ/η arbitraires par bloc.
Pour O-grid 4-block, la sanity check flag `InterfaceOrientationTrivial`
sur 2 des 4 interfaces. 2 options :

**Option A — reorientation pass explicite (recommandé)** :
`reorient_block!(mbm, bid; flip_ξ=false, flip_η=false)` qui applique la
transformation à la mesh + boundary_tags. BFS sur interfaces pour
auto-détecter la chaîne de flips nécessaires.

**Option B — .geo avec Transfinite Surface dans un ordre canonique** :
Chaque bloc défini avec Transfinite Surface corners `{BL, BR, TR, TL}`
où BL = coin inner-south (cylindre, début angulaire), BR = inner-end
(fin angulaire), TR = outer-end (far-field, fin), TL = outer-start.
Ça force ξ = angulaire, η = radial, et tous les interfaces deviennent
canoniques east↔west. Plus simple à implémenter côté loader :
extraire l'ordre transfinite depuis gmsh.

### Priorité 3 — B.2.3 SLBM geometry sur extended array

Pour que `slbm_trt_libb_step_local_2d!` tourne par bloc sur les arrays
étendus `(Nξ+2Ng, Nη+2Ng, 9)`, il faut une `SLBMGeometry` de taille
extended.

**Approche** : dans [src/multiblock/](src/multiblock/), écrire
`build_block_slbm_geometry_extended(block; n_ghost=1)` qui :
1. Étend la mesh `(Nξ, Nη)` en `(Nξ+2Ng, Nη+2Ng)` par extrapolation
   linéaire des coins (préserve les vecteurs d'edge aux bords intérieurs)
2. Construit `CurvilinearMesh(X_ext, Y_ext; FT)` — fit les splines
3. Appelle `build_slbm_geometry(mesh_ext; local_cfl=true)`
4. Retourne la geom + la mesh_ext pour que `compute_local_omega_2d`
   puisse aussi être étendu

### Priorité 4 — Driver approach (E) O-grid + extensions

Écrire [hpc/wp_mesh_6_multi_aqua.jl](hpc/wp_mesh_6_multi_aqua.jl) :

1. `.geo` intégré : 4 blocs O-grid (R_in=0.025 → R_out=0.1) + 4 blocs
   extension rectangulaires (4 cardinaux). 8 blocs au total, 12
   interfaces.
2. Load via `load_gmsh_multiblock_2d` + sanity + réorientation
3. Per-block :
   - O-grid blocks : SLBM extended + LI-BB via
     `precompute_q_wall_slbm_cylinder_2d` + local-τ
   - Rectangle extension blocks : standard Cartesian +
     `fused_trt_libb_v2_step!` avec `q_wall=0` (no cylinder)
4. Time loop :
   ```julia
   exchange_ghost_shared_node_2d!(mbm, states)  # interfaces
   fill_physical_wall_ghost_2d!(mbm, states)    # walls rectangle
   for blk in mbm.blocks
       if blk_is_ogrid(blk)
           slbm_trt_libb_step_local_2d!(…, extended)
       else
           fused_trt_libb_v2_step!(…, extended)
       end
   end
   # per-block BC apply (inlet on ext_W, outlet on ext_E, walls N/S)
   ```
5. Aggregate drag/lift across the 4 O-grid blocks touching cylinder

### Priorité 5 — Paper matrix update

Mettre à jour [paper/wp_mesh_6.md](paper/wp_mesh_6.md) avec lignes (D)
et (E). Killer chiffre attendu : (E) atteint Cd + Cl_RMS de référence
à budget cellules < (A) et (B).

## Handoff B.2 cheat-sheet

```
Pipeline approach (E) step :
  exchange_ghost_shared_node_2d!(mbm, states)   # NEW B.2.1 ✓
  fill_physical_wall_ghost_2d!(mbm, states)     # Phase A.5c ✓
  for (k, blk) in enumerate(mbm.blocks)
      Nx_ext, Ny_ext = ext_dims(states[k])
      if is_cylinder_block[k]
          slbm_trt_libb_step_local_2d!(…, geom_ext[k], sp[k], sm[k])  # needs B.2.3
      else
          fused_trt_libb_v2_step!(…, Nx_ext, Ny_ext, ν)
      end
      apply_bc_rebuild_2d!(interior_f(f_out[k]), interior_f(state[k].f),
                             bcspec[k], ν, blk.mesh.Nξ, blk.mesh.Nη)
  end
  swap(states[k].f, f_out[k]) for each k
```

Points critiques :
- **Orientation ξ/η** : doit être cohérente OU exchange extended pour
  handler tous les pairs. Voir Priorité 2.
- **Interface + LI-BB corner cells** : coexistent sans interaction.
  LI-BB est in-kernel (brick `ApplyLiBBPrePhase`); interface ghost
  est pre-step.
- **Drag sum** : `compute_drag_libb_mei_2d` par bloc qui touche
  le cylindre, sum Fx/Fy puis normaliser par `u² × D`.

## Mémoire utile

- [project_multiblock_v03.md](~/.claude/projects/-Users-guillaume-Documents-Recherche-Kraken-jl/memory/project_multiblock_v03.md)
  — état phase A + B.1 + B.2.1
- [project_slbm_local_cfl.md] — local-CFL 1.7% err
- [project_wp_mesh_6_bump.md] — résultats (C) et instabilité τ→0.5
- [project_kernel_dsl.md] — LI-BB V2 + GPU drag patterns
