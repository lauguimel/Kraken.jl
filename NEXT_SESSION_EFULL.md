# Reprise E-full — 8-block O-grid cylinder Re=100

Branche : `slbm-paper`

## Ce qui MARCHE (validé cette session)

### Multi-block Cartésien (E3) — bit-exact
- **Cd=1.644/1.649/1.650** à D=20/40/80 sur Aqua H100 FP64
- Matche (A)(B) Cartésien à 1e-3
- 2 bugs trouvés et fixés via ladder décomposée (Poiseuille → cylindre) :
  1. `ebf0867` : BC corner skipping (south/north BB ignorait les coins interface×wall)
  2. `b51f52b` : FP drift dans cx_local per bloc (global q_wall → slice)
- **Validation bit-exact** 1-block = 2-blocks = 3-blocks sur 6 niveaux

### Pipeline E-full — wired end-to-end
- `load_gmsh_multiblock_2d` → `autoreorient_blocks` (avec transpose J>0)
  → `extend_mesh_2d` (skip_validate + dx_ref preservé)
  → `build_block_slbm_geometry_extended` → `slbm_trt_libb_step_local_2d!`
  → `exchange_ghost_shared_node_2d!` + `fill_physical_wall_ghost_2d!`
  → `apply_bc_rebuild_2d!` (avec east ZouHeVelocity + west ZouHePressure ajoutés)
  → `compute_drag_libb_mei_2d` (guard ndrange=0)
- Les 3 résolutions D=20/40/80 tournent sans crash sur Aqua (5→94 MLUPS)
- Metal F32 smoke 50 steps : pipeline wire OK

### Maillage corrigé
- Progression inversée : `target_ratio^(+1/(N-1))` au lieu de `^(-1/(N-1))`
- Avant : cellules fines au MUR RECTANGLE (AR=117). Après : cellules fines au CYLINDRE (AR=4.6)
- Plots : `tmp/ogrid_mesh_fixed.png`, `tmp/ogrid_zoom_fixed.png`

## Ce qui NE MARCHE PAS

### Cd=NaN à toutes les résolutions et tous les Re
- Re=100 : NaN
- Re=20 : NaN aussi
- Avec local_cfl=true ET false : NaN
- Avec progression corrigée ET ancienne : NaN
- Avec inlet uniforme ET parabolique : NaN
- Avec ZouHeVelocity ET ZouHePressure sur inlet : NaN

### Symptôme précis
- Step 1 : u_max intérieur ~0.04 (OK pour ring_0/7 outlet), ~0.35 pour rings 2-6
- Step 10 : ring_3/4 (inlet) u_max ~0.5, ρ_min ~0.45
- Step 20 : ring_3/4 u_max ~0.5, ρ_max ~1.15
- Step 50 : explosion ρ ∈ [-1e24, +1e24]
- ring_0/7 (outlet) restent STABLES jusqu'à step 20-30

### Ce que ça exclut
- **PAS un problème de maillage** : AR max 4.6, Jacobien positif partout, pas de fold
- **PAS un problème de Re** : diverge aussi à Re=20
- **PAS un problème de local-CFL** : diverge avec local_cfl=false aussi
- **PAS un problème de progression** : diverge avec toute progression
- **PAS un problème de type d'inlet** : velocity et pressure inlet divergent

### Ce qui N'A PAS été vérifié / testé
1. **Est-ce que le SLBM step seul (sans multi-block) fonctionne sur un ring block ?**
   → Prendre UN SEUL ring block (ex: ring_0), pas de multi-block, juste SLBM step. Stable ?

2. **Est-ce que l'exchange_ghost_shared_node fonctionne correctement sur cette topologie ?**
   → Tester avec BGK (pas SLBM) sur la topologie 8-block : populations échangées bit-exact ?
   → L'exchange shared-node sur des blocs transposés a-t-elle été testée ?

3. **Est-ce que le fill_physical_wall_ghost interagit correctement avec les edges courbées (west=cylinder) ?**
   → wall_ghost pour west=cylinder avec le kernel _fill_wall_vertical : les populations BB reflétées depuis la courbe sont-elles cohérentes ?

4. **Est-ce que les ring blocks SANS cylinder (ring_4 solid=0, ring_0 solid=8) divergent différemment ?**
   → ring_4 n'a AUCUNE cellule solide ni cut-link (solid=0, cut_links=0).
     Si ring_4 diverge quand même, le problème n'est pas LI-BB mais exchange/BC.

5. **Est-ce que u_max ~0.35 au step 1 (sur les rings 2-6) est un artefact de ghost cells ou d'intérieur ?**
   → Les ρ au step 1 vont jusqu'à 0.18 minimum (certains blocks). C'est très bas pour un init à ρ=1.
   → D'où viennent ces ρ faibles ? Cells initiales ou ghost contaminé ?

6. **Est-ce que le kernel SLBM gère correctement les cellules sur l'ARC du cylindre (west edge des rings) ?**
   → Au i=1 (west, arc), les cellules sont SUR le cylindre (is_solid=true pour certaines).
     Le kernel fait BB. Mais les DEPARTURE POINTS du SLBM à i=2 (première cellule fluide)
     vont-ils chercher des infos dans le ghost i=0 qui est INSIDE le cylindre ?

## Fichiers clés

| Fichier | Description |
|---|---|
| `src/multiblock/reorient.jl` | autoreorient avec transpose J>0 |
| `src/multiblock/mesh_extend.jl` | extend_mesh_2d avec skip_validate + dx_ref + curved_edges (non utilisé) |
| `src/kernels/boundary_rebuild.jl` | BC corner fix + east ZouHeVelocity + west ZouHePressure |
| `src/kernels/drag_gpu.jl` | Guard ndrange=(0,) |
| `src/curvilinear/mesh.jl` | skip_validate kwarg |
| `hpc/wp_mesh_6_ogrid_aqua.jl` | Driver Aqua E-full |
| `tmp/gen_ogrid_rect_8block.jl` | Générateur .geo 8-block |
| `tmp/smoke_ogrid_local_metal.jl` | Smoke Metal (pipeline wire OK) |
| `tmp/diag_efull_local.jl` | Diagnostic local CPU (montre divergence step-by-step) |
| `tmp/plot_ogrid_mesh.jl` | Plot maillage complet |
| `tmp/plot_ogrid_zoom.jl` | Plot zoom cylindre |
| `tmp/poiseuille_multiblock_ladder.jl` | Ladder validation Niveaux 0-3 |
| `tmp/level5_cylinder_libb.jl` | Level 5 cylindre + LI-BB 1-block vs 3-block |
| `tmp/level6_e3_local.jl` | Level 6 E3 setup local CPU |

## Suggestion d'approche pour la prochaine session

**Ne PAS assumer la cause.** Décomposer en cas élémentaires :

1. **Single ring block SLBM** : prendre ring_0 seul (west=cylinder, east=outlet,
   south=wall, north=wall — simuler les 2 interfaces comme walls). Faire tourner
   le SLBM step seul (pas de multi-block). Si ça marche → le SLBM sur O-grid est OK.

2. **2 ring blocks SLBM** : ring_0 + ring_1 avec une seule interface. Même test.
   Si ça diverge → l'exchange_ghost_shared_node sur blocs transposés a un bug.

3. **8 ring blocks BGK Cartésien** : remplacer slbm_trt_libb_step_local_2d par
   fused_bgk_step (ignore la métrique curvilinéaire, traite comme Cartésien).
   Si ça stabilise → le SLBM kernel a un problème spécifique sur cette géométrie.

4. **8 ring blocks SANS cylinder** : enlever is_solid et q_wall (pas de cylindre,
   juste un domaine annulaire vide). Si ça diverge → le problème est dans exchange/BC,
   pas dans LI-BB.

Chaque test isole UN composant. Le premier qui casse identifie la cause.

## Commits cette session (branche `slbm-paper`)

```
ebf0867 fix(multiblock): halfway-BB south/north must cover interface-wall corners
b51f52b fix(hpc): compute cylinder q_wall globally and slice per block
147afc5 feat(slbm): add :none scaling to compute_local_omega_2d for diagnostics
5e12b48 docs(paper): E3 multi-block Aqua results + document both fixes
37d4d33 feat(multiblock): unblock (E-full) 8-block O-grid pipeline
25e21ef fix(drag_gpu): guard against ndrange=(0,) when no cut links
5e5ff24 fix(E-full): preserve dx_ref on extended mesh + adaptive radial progression
1f622d1 fix(E-full): reverse radial Progression exponent
```

Total : 7707 tests verts.
