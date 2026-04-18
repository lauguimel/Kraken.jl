# Viscoelastic LBM — gotchas et bugs identifiés

Document de transfert pour réutiliser ces enseignements sur d'autres
branches (notamment toute branche qui réimplémente du LI-BB + Hermite
source ou du log-conformation).

Branche source : `dev-viscoelastic` (Kraken.jl).
Référence article : Liu et al., arxiv 2508.16997 (TRT-LBM Oldroyd-B).

---

## 1. Bug MEA + source Hermite — double-comptage de drag (10–15 %)

### Symptôme
Sur le cylindre confiné Liu (Re=1, Wi=0.001, β=0.59, R=20), le `Cd_s`
calculé par `compute_drag_libb_mei_2d` (formule Mei avec interpolation
Bouzidi sur les liens coupés) sortait **+10 à +15 % au-dessus** de la
limite Newtonienne attendue dès que la source Hermite injectait `τ_p`
non-trivial dans `f`. Le bug se manifestait aussi à très bas Wi où
`τ_p ≈ 2·ν_p·S` est censé être indistinguable d'une viscosité
Newtonienne supplémentaire.

### Cause racine
`compute_drag_libb_mei_2d` reconstruit le drag via Mei (Yu, Mei, Shyy &
Luo 2003) en mélangeant les populations post-collision de **plusieurs
cellules voisines** via les coefficients d'interpolation Bouzidi
`q_w ∈ (0,1]`. Quand `apply_hermite_source_2d!` ajoute
`T_q ∝ -s_plus · w_q · H_{q,αβ} · τ_{αβ}` à `f_out` **après** la
collision fusionnée, chaque cellule reçoit une perturbation locale
différente (parce que `τ_p(x)` varie spatialement).

Mei mélange alors des populations qui ne sont **pas physiquement
cohérentes** entre elles : la cellule `(i,j)` voit `τ_p[i,j]` injecté,
la cellule `(i±1, j)` voit `τ_p[i±1, j]`, et l'interpolation Mei
suppose implicitement que ces deux cellules portent la même physique
sous-jacente. Résultat : un terme spurieux d'ordre `s_plus · ∇τ_p`
apparaît dans le drag intégré.

### Solution validée
Utiliser **`compute_drag_mea_2d` (halfway-BB MEA standard)** dans tous
les drivers viscoélastiques, même quand le solveur de fluide utilise
LI-BB V2 pour le streaming sur paroi courbe. Le halfway-BB MEA n'utilise
qu'**une seule cellule** par lien coupé (la cellule fluide adjacente),
donc l'injection locale de `τ_p` est cohérente.

Coût : on perd la sub-cell accuracy de Mei sur la paroi courbe, mais
le gain en cohérence avec la source Hermite domine. La validation finale
sur le cylindre Liu donne **0.35 % d'erreur sur Cd à R=48, Wi=0.1**
(vs Liu Table 3, 130.83 → 131.29).

### Fichiers / commits
- Fix : commit `03b863a` (`fix(viscoelastic): use std halfway-BB MEA
  instead of Mei for cylinder`)
- Diagnostic isolant le bug : `hpc/debug_isolate.jl` (sweep 6 combinaisons
  drag formula × kernel × source on/off)
- Driver corrigé : `src/drivers/viscoelastic.jl`
  `run_conformation_cylinder_libb_2d` lignes ~538-551 — voir le commentaire
  inline qui documente le piège.

### Règle pour toute future branche
> Tout driver qui combine `apply_hermite_source_*d!` (post-collision
> injection de `τ_p` dans `f`) avec un calcul de drag DOIT utiliser le
> halfway-BB MEA, JAMAIS Mei-with-Bouzidi.

### Confirmation 3D (Aqua H100, 2026-04-19, job 20147611)
**Le même bug a été reproduit en 3D.** Run `hpc/sphere_oldroyd_3d.jl` à
384×64² sur sphère R=16 (blockage 0.25), β=0.5, Re=1 :

| Wi  | Cd_visco | Cd_Newt | Erreur  | Attendu (Lunsmann 1993) |
|-----|----------|---------|---------|-------------------------|
| 0   | —        | 215.3   | (réf)   | —                       |
| 0.1 | 192.0    | 215.3   | **−10.8 %** | ~ +2 à +5 % (enhancement) |
| 0.5 | 144.9    | 215.3   | −32.7 % | ~ +10 à +15 %           |
| 1.0 | 131.5    | 215.3   | −38.9 % | ~ +15 à +25 %           |

Le Cd **décroît** avec Wi au lieu d'augmenter (drag enhancement
attendu). À Wi=0.1 le fluide est quasi-Newtonien (ν_total identique),
donc Cd_visco devrait matcher Cd_Newt à ~1 %. L'écart de −10.8 % à
Wi=0.1 signale une **inversion de signe ou un facteur de magnitude
incorrect** dans l'intégration du drag avec la source Hermite 3D.

Indice additionnel : sur grille locale 40³ (blockage 0.4, parabolic_y
inlet), le même code donne Cd_visco **supérieur** à Cd_Newt de +30 %
— **direction opposée** à celle observée sur Aqua. Un bug
magnitude-only donnerait un écart constant-signe ; l'inversion selon
blockage/inlet pointe vers un mélange incohérent de cellules voisines
par `compute_drag_libb_3d` (Mei-Bouzidi 3D), exactement le mode de
défaillance du 2D.

### Action à faire (prochaine session)
1. Écrire `compute_drag_mea_3d(f, is_solid, Nx, Ny, Nz)` : halfway-BB
   MEA D3Q19 standard (pas de q_wall), somme sur les liens
   fluide→solide uniquement dans la cellule fluide adjacente.
2. Remplacer `compute_drag_libb_3d` par `compute_drag_mea_3d` dans
   `run_conformation_sphere_libb_3d`
   (`src/drivers/viscoelastic_3d.jl:~216`).
3. Relancer le job Aqua `hpc/sphere_oldroyd_3d.jl`. Attendu : Cd_visco
   matche Cd_Newt à <5 % à Wi=0.1 et croît ensuite avec Wi jusqu'au
   plateau Lunsmann vers Wi=1.
4. Vérifier aussi que le Newtonian-only `run_sphere_libb_3d` donne la
   même valeur avec les deux formules — la Mei 3D ne doit pas être
   systématiquement biaisée sans source Hermite.

---

## 2. Cd_p (intégrale de surface du stress polymère) double-compte aussi

### Symptôme
Pendant la mise au point, l'addition `Cd_total = Cd_s + Cd_p` donnait
des valeurs aberrantes (+50 à +100 % vs Newtonien à bas Wi).

### Cause
La source Hermite injecte directement `τ_p` dans les populations `f`,
donc le MEA sur `f` capture **toute la contrainte effective**
(σ_solvent + τ_p) en une seule passe. L'intégrale séparée
`compute_polymeric_drag_2d(τ_p · n)` re-comptabilise la même contribution
polymère.

### Solution
- `Cd = Cd_s` (MEA halfway-BB sur f post-source)
- `Cd_p` est conservé comme **diagnostic seul**, jamais ajouté au total.
- Documenté dans la docstring de `run_conformation_cylinder_libb_2d`
  et `run_conformation_sphere_libb_3d`.

### Règle pour toute future branche
> Si la source Hermite est utilisée → `Cd = Cd_s` uniquement.
> Si on évolue `τ_p` séparément avec `compute_polymeric_force_2d!`
> (formulation Guo/force-volumique), alors il faut au contraire ajouter
> `Cd_s + Cd_p` (les deux contributions sont disjointes). Voir
> `run_viscoelastic_cylinder_2d` (formulation `:stress`) pour le contre-
> exemple.

---

## 3. HWNP (High-Weissenberg Number Problem) résiduel à haut Wi

### Symptôme
Même avec la formulation log-conformation (Fattal-Kupferman 2004,
`LogConfOldroydB`), à R=48 sur le cylindre Liu :

| Wi  | Cd_logconf | Cd_Liu  | Erreur  |
|-----|------------|---------|---------|
| 0.1 | 130.78     | 130.36  | +0.32 % |
| 0.5 | 100.24     | 126.31  | −20.6 % |
| 1.0 |  91.51     | 151.31  | −39.5 % |

Le scaling avec R est lent à haut Wi : à R=30 on avait −26 % / −44 %,
à R=48 on a −20.6 % / −39.5 %. La résolution seule ne résout pas le
problème.

### Cause probable
La log-conformation **stabilise** le schéma (pas de NaN, pas
d'eigenvalues négatives), mais ne **régularise** pas les couches
limites de stress polymère, qui restent sous-résolues. À Wi=1, λ=2400,
u=0.02 → distance d'établissement ~ λ·u = 48 cellules, comparable au
diamètre du cylindre. Le sillage de stress n'a pas la place de se
développer correctement.

### Pistes pour avancer
1. **Résolution adaptative (refinement patches)** autour du sillage
   polymère. La branche `refinement-patches-dev` a l'infrastructure ;
   il faudrait étendre les exchange kernels au tenseur de conformation.
2. **SUPG / Galerkin stabilisé** sur le terme advectif de C : le Liu
   2025 mentionne un Crank-Nicolson semi-implicite qu'on n'a pas
   implémenté.
3. **Diffusion artificielle locale** (κ_artif = q_wall · κ_0) près des
   parois pour amortir les oscillations du polymer wake. Hack mais
   souvent suffisant pour passer Wi=1.
4. **Vérifier que `tau_plus` n'est pas trop bas** (artificial diffusion
   intrinsèque trop faible). Tester `tau_plus = 1.5 - 2.0` à haut Wi.

---

## 4. Corruption de C aux frontières inlet/outlet

### Symptôme
Sans intervention, `stream_2d!` applique du bounce-back aux frontières
ouest/est sur les populations `g_*` du tenseur de conformation, ce qui
corrompt `C` à l'inlet (devrait être analytique pour Poiseuille
Oldroyd-B) et à l'outlet (devrait être zero-gradient).

### Solution
- `reset_conformation_inlet_2d!(g, C_inlet, u_profile, Ny)` : force
  `g[1, j, q] = g^eq(C_inlet[j], u_profile[j])` pour tout q.
- `reset_conformation_outlet_2d!(g, Nx, Ny)` : zero-gradient
  `g[Nx, j, q] = g[Nx-1, j, q]`.
- Profil C analytique pour Poiseuille Oldroyd-B (Liu Eq 62) :
  - `C_xy(y) = λ · ∂u/∂y`
  - `C_xx(y) = 1 + 2·(λ·∂u/∂y)²`
  - `C_yy = 1`

Ports 3D existent : `reset_conformation_inlet_3d!`,
`reset_conformation_outlet_3d!` (commit `64701d1`). Pour le 3D le
profil C analytique généralisé n'est implémenté que pour
`inlet=:parabolic_y` (cisaillement y uniquement).

---

## 5. Bug pré-existant — `run_sphere_libb_3d` segfault à 60×30×30

### Symptôme
Sur Macbook M3 Max (Metal/CPU), `run_sphere_libb_3d` segfault au-delà
d'environ Nx·Ny·Nz > 30 000 cellules avec radius ≥ 4. Reproductible :
- 40×20×20 r=3 → OK
- 60×30×30 r=4 → SIGSEGV
- 60×30×30 r=6 → SIGSEGV

Le segfault se produit même en mode Newtonien pur (sans la couche
viscoélastique) → bug dans le driver 3D ou dans `fused_trt_libb_v2_step_3d!`,
**pas** dans le port viscoélastique 3D que j'ai ajouté.

### Status
**Non résolu**. Le port 3D viscoélastique se contourne en testant
sur grille ≤ 40³ localement et en validant à pleine taille (384×64²) sur
H100 Aqua. À investiguer dans une session dédiée — c'est probablement
un overrun d'index dans un des kernels 3D ou un problème d'allocation
mémoire CPU.

### Reproduction minimale
```julia
using Kraken
run_sphere_libb_3d(; Nx=60, Ny=30, Nz=30, radius=4,
                     u_in=0.04, ν=0.08, inlet=:uniform,
                     max_steps=200, avg_window=50)
# → SIGSEGV après ~10s
```

---

## 6. Newtonian-limit dans le port 3D viscoélastique — MAJ Aqua

### Symptôme initial (local CPU)
Le test `test_conformation_sphere_3d.jl` "low-Wi Newtonian consistency"
à 40×20×20, r=4, blockage R/H=0.4, inlet parabolic_y :
- Cd_Newtonien (`run_sphere_libb_3d` à ν_total) = 41.22
- Cd_viscoelastique (`run_conformation_sphere_libb_3d` à Wi=0.013) = 53.77
- Écart relatif = **+30.5 %** (visco > Newt)

### Confirmation Aqua H100 (384×64², 2026-04-19)
Le même driver à résolution de production (blockage 0.25, inlet
uniform) donne le résultat **inverse** :
- Wi=0.1 : Cd_visco = 192.0, Cd_Newt = 215.3 → **−10.8 %** (visco < Newt)
- Wi=1.0 : Cd_visco = 131.5 → −38.9 %, monotonement décroissant

Le changement de signe entre les deux configurations confirme le
diagnostic du §1 : `compute_drag_libb_3d` mélange des cellules
voisines via Bouzidi, et cette opération n'est plus cohérente quand la
source Hermite injecte un `τ_p(x)` variable. L'orientation de
l'erreur dépend de la géométrie locale du gradient, d'où le signe
inverse entre 40³ parabolic_y et 384×64² uniform.

### Diagnostic et fix prévu
Voir §1 "Action à faire" : écrire `compute_drag_mea_3d` (halfway-BB
standard) et remplacer `compute_drag_libb_3d` dans
`run_conformation_sphere_libb_3d`. Attendu :
- Cd_visco(Wi=0.1) match Cd_Newt à <5 %
- Cd_visco(Wi↑) croît (drag enhancement, cf. Lunsmann 1993)
- Test CPU 40³ passe à <10 % au lieu de +30 %

Ne PAS conclure à un bug dans la source Hermite 3D elle-même ni dans
la calibration du préfacteur `9/2 / (1 − s/2)` sans avoir d'abord
isolé le drag integration. Les kernels D3Q19 (TRT 6 composantes,
CNEBB, Hermite source) ont passé leurs tests unitaires (17/17) et
conservent C = I au repos à la précision machine — le bug n'est
probablement PAS dans ces kernels.

---

## 7. Post-processing Alves — Re/précision/visibilité

### Symptôme Aqua H100 (job 20145695, 2026-04-19)
Run `hpc/alves_contraction.jl` 4:1 planaire, H_out=20, β_c=4, β=0.59,
u_out_mean=5e-4, ν_total=1 → Re=0.01. Les trois Wi (0.5, 1, 2) ont
tous complété sans diverger, mais le post-processing affiche :

```
Wi     X_R_south   X_R_north   X_R/H   N1_max
0.50   0.00        0.00        0.000   0.0000
1.00   0.00        0.00        0.000   0.0000
2.00   0.00        0.00        0.000   0.0000
```

### Causes (pas un bug de simulation)
1. **Re = 0.01 → pas de vortex inertiel.** Le salient-corner vortex
   d'Alves 2003 n'apparaît qu'à Re ≳ 0.2 ou en présence de
   l'élasticité suffisante pour générer un corner stress. À Re=0.01 +
   Wi=2, le produit De·Re reste trop faible pour la grille et le
   temps simulé.
2. **N1 réellement non nul mais invisible en `%-12.4f`.** Estimation
   d'ordre de grandeur au centreline :
   - `u_out_max = 1.5·u_out_mean = 7.5e-4`
   - `∂u/∂y ∼ u_out_max/(H_out/2) = 7.5e-5`
   - `C_xy_steady = λ·γ̇ = 40000·7.5e-5 = 3`
   - `C_xx_steady = 1 + 2·(λγ̇)² = 19`
   - `τ_p_xx = G·(C_xx - 1) = (ν_p/λ)·18 = (0.41/40000)·18 = 1.8e-4`
   - `N1 = τ_xx - τ_yy ≈ 1.8e-4`

   Le format `"%-12.4f"` affiche 4 décimales fixes → 1.8e-4 apparaît
   comme `0.0002`. Mais le script imprime `0.0000`, ce qui laisse
   deux possibilités : (a) le N1 réel est encore plus petit (< 5e-5,
   ce qui arriverait si le polymer n'a pas eu le temps de se
   développer), ou (b) les fields ont été retournés par le try/catch
   sans être remplis (r.Nx=0 path).

### Actions à faire
1. Passer `@printf "%-12.4e"` (scientifique) pour les N1 et X_R.
2. Augmenter `u_out_mean` à `1e-2` (Re ≈ 0.4, dans la zone où Alves
   2003 rapporte les vortex). Mettre `max_steps = 200_000` fixe pour
   éviter les max_steps à 14e6 qui augmentent le coût inutilement.
3. Ajouter un dump VTK/PNG des champs à la fin du run (ux, τ_p_xx,
   streamlines) pour visualiser ce qu'on a réellement simulé. Le
   driver retourne déjà `C_xx`, `tau_p_xx`, etc. — il manque juste
   l'écriture disque.
4. Insérer un check NaN+max values explicite après chaque Wi :
   ```
   println("  max|ux|=", maximum(abs, r.ux),
           "  max|tau_p_xx|=", maximum(abs, r.tau_p_xx))
   ```
   pour distinguer "simulation a tourné proprement mais signaux trop
   faibles" de "simulation a diverged et rendu zeros".

---

## 8. Patterns réutilisables pour d'autres modèles polymères

### Architecture modulaire validée
- `AbstractPolymerModel` : `OldroydB`, `LogConfOldroydB`. Ajouter
  `FENE_P`, `Giesekus`, `PTT` via dispatch sur `update_polymer_stress!`
  (et `collide_conformation_2d!` pour les modèles avec terme source
  non-linéaire).
- `AbstractPolymerWallBC` : `CNEBB`, `NoPolymerWallBC`. Le dispatch
  `apply_polymer_wall_bc!` route sur 2D/3D par dimensionnalité du
  array `g` (3-d → D2Q9, 4-d → D3Q19).
- `BCSpec{2,3}D` : per-face BC dispatch (HalfwayBB, ZouHeVelocity,
  ZouHePressure). Géométries axis-aligned (channel, contraction)
  utilisent HalfwayBB sur N/S ; géométries courbes (cylindre, sphère)
  utilisent les `q_wall` precomputés.
- Backend : tous les kernels GPU/CPU via `KernelAbstractions.allocate(backend, ...)`.
  Validé sur CUDA H100 et CPU.

### Fichiers à ne pas réécrire
- `src/kernels/conformation_lbm_2d.jl` (TRT 3 composantes 2D)
- `src/kernels/conformation_lbm_3d.jl` (TRT 6 composantes 3D, port 64701d1)
- `src/kernels/viscoelastic_2d.jl` (Hermite source 2D, log-conf eigen 2×2)
- `src/kernels/viscoelastic_3d.jl` (Hermite source 3D, port 64701d1)
- `src/kernels/li_bb_2d_v2.jl` (TRT + LI-BB V2 fused, validé Couette/Poiseuille)
- `src/kernels/boundary_rebuild.jl` (BCSpec dispatch, modulaire)

### Ce qui reste à porter en 3D
1. **Log-conformation 3D** — nécessite eigen-decomposition 3×3 symétrique
   (ex: méthode Cardano + Jacobi pour les vecteurs propres). Le
   `LogConfOldroydB` 3D est explicitement refusé dans
   `run_conformation_sphere_libb_3d` pour éviter une divergence silencieuse.
2. **STL → q_wall 3D pour géométries arbitraires** — l'infrastructure
   `precompute_q_wall_from_stl_3d` existe (commit `8c411a` avant cette
   branche) mais n'a pas été testée avec la couche viscoélastique.

---

## Tests à garder verts

```bash
julia --project=. -e 'using Test; include("test/test_conformation_lbm.jl")'        # 16 tests
julia --project=. -e 'using Test; include("test/test_conformation_lbm_3d.jl")'     # 17 tests (port 3D)
julia --project=. -e 'using Test; include("test/test_conformation_cylinder.jl")'   # 8 tests + GPU
julia --project=. -e 'using Test; include("test/test_conformation_sphere_3d.jl")'  # 13 tests (port 3D)
julia --project=. -e 'using Test; include("test/test_viscoelastic_coupling.jl")'   # 7 tests
julia --project=. -e 'using Test; include("test/test_logconformation.jl")'         # 8 tests (1 fail pré-existant — cylinder Cd convergence à 3000 steps insuffisant)
julia --project=. -e 'using Test; include("test/test_contraction_libb.jl")'        # 24 tests (port contraction)
```

Le seul échec connu et non-bloquant est le test "Low-Wi consistency:
direct-C ≈ log-conformation" dans `test_logconformation.jl` ligne 54 —
échec pré-existant (commit `03b863a` antérieur à cette session) lié à
un budget de pas de temps insuffisant (3000 steps) pour atteindre
l'état stationnaire du cylindre. Non-bloquant pour la validation Liu.
