# Audit — Wi sweep cylinder, divergence Kraken vs rheoTool

Date : 2026-04-30. Branche `dev-viscoelastic`.

## TL;DR

Le sweep Wi sur le cylindre 2D Oldroyd-B (R=30, β=0.59, Re_R=1) montre
que la "validation" précédente à Wi=0.1 (+0.24% vs rheoTool) était une
**coïncidence à un point** : les courbes Cd(Wi) Kraken et rheoTool ont
des **pentes très différentes** et se croisent par hasard à Wi=0.1.

Le coupable principal identifié est le facteur CE
`source_scale = 1/(1-s_plus/2) ≈ 2.883` appliqué à l'amplitude du
Hermite source dans la **dynamique** (kernel
`apply_hermite_source_2d!` avec `ce_correction=true`). Ce facteur
sur-couple l'effet du polymère sur le champ de vitesse d'environ ×3,
puis le drag measurement applique l'inverse `0.347` pour "récupérer"
la bonne valeur — mais seulement à **un** point de fonctionnement.

## Données mesurées

Backend : Aqua H100 Float64. Sweep avec `:ce_corrected` + `:source_scaled_mea`
(le mode "validé" à Wi=0.1).

| Wi   | λ_lat | Cd_Kraken | Cd_rheoTool | ΔCd_K vs Newt | ΔCd_RT vs Newt | Ratio Wi |
|------|-------|-----------|-------------|---------------|----------------|----------|
| 0.05 |  300  | **149.76** | 131.81      | **+17.7**     | -0.5           | **35×** |
| 0.10 |  600  | 130.74    | 130.43      | -1.3          | -1.9           | 0.7×     |
| 0.20 | 1200  | **111.74** | 126.83      | **-20.3**     | -5.5           | **3.7×** |
| 0.50 | 3000  | 93.19     | (in flight) | -38.9         | (~ -8)         | ~5×      |
| 1.00 | 6000  | 85.49     | (in flight) | -46.6         | (~ -13)        | ~3.6×    |

Cd_Newt(Kraken) = 132.08, Cd_Newt(rheoTool) = 132.36.

**Symptôme clé** : à Wi=0.05, Kraken donne une **augmentation** du drag
(+17.7) alors que la physique donne une faible **réduction** (-0.5).
Le sens de l'effet polymère est **inversé** à bas Wi.

## Décomposition du drag (Kraken)

| Wi   | Cd_s   | Cd_p  | Cd_split | Cd_mea_post | Cd_scaled (mode validé) |
|------|--------|-------|----------|-------------|-------------------------|
| 0.05 | 138.15 | 26.89 | 165.05   | 171.60      | **149.76**              |
| 0.10 | 122.47 | 19.09 | 141.56   | 146.31      | **130.74**              |
| 0.20 | 106.43 | 12.20 | 118.63   | 121.73      | **111.74**              |
| 0.50 |  90.44 |  6.31 |  96.75   |  98.37      | **93.19**               |
| 1.00 |  83.81 |  3.86 |  87.67   |  88.67      | **85.49**               |

`Cd_s` = drag MEA mesuré **avant** injection Hermite (donne le drag
solvant sur le champ de vitesse converged). À Wi=0.05, `Cd_s = 138 >
Cd_Newt = 132` : le polymère **modifie** le champ de vitesse de
manière à **augmenter** le drag solvant. Ce n'est pas physique à si
bas Wi — la rétroaction polymère devrait être négligeable.

Cela prouve que le couplage τ_p → u dans la dynamique est trop fort.

## Cause racine (hypothèse)

### Le code

`src/kernels/collide_viscoelastic_source_2d.jl:114` :

```julia
pre = -s_plus * T(9.0/2.0) * source_scale
```

avec `source_scale = 1/(1-s_plus/2) ≈ 2.883` quand `ce_correction=true`.

`src/drivers/viscoelastic.jl:680` (driver actif) :

```julia
apply_hermite_source_2d!(f_out, is_solid, s_plus_s, ...;
                          ce_correction = hermite_source_mode === :ce_corrected)
```

→ Le driver passe `ce_correction=true` par défaut.

### Pourquoi le facteur 2.883 sur-couple

Le Hermite source injecte la contribution stress polymère dans les
populations f. L'amplitude `pre` contrôle directement combien de
stress polymère apparaît dans le champ macroscopique. Avec
`source_scale=2.883`, le schéma applique **3× le stress polymère
réel** dans la dynamique, ce qui :

1. **Sur-estime la viscosité effective** à bas Wi (où τ_p est dominé
   par τ_p_xy ∝ ν_p·γ̇). Effet : Cd augmente au lieu de baisser.
2. **Sur-estime l'effet de N1** à plus haut Wi (où τ_p_xx ∝ ν_p·λ·γ̇²
   devient dominant). Effet : Cd baisse beaucoup trop.
3. La transition se fait autour de Wi ≈ 0.1 (par hasard) — c'est
   exactement là où le mode de drag `:source_scaled_mea` divise par
   2.883 donne approximativement la bonne valeur.

### Incohérence interne du code

Le projet contient **deux préfacteurs différents** pour le même
schéma :

| Lieu | Code | Effet |
|---|---|---|
| `fused_trt_libb_v2_hermite_step!` (in-collision, 2D) | `source_scale=1` par défaut | PAS de facteur CE |
| `apply_hermite_source_2d!` (post-collision, 2D) | `ce_correction=true` par défaut | facteur CE = 2.883 |
| `viscoelastic_3d.jl:26` (3D) | `pre = -s_plus × 9/2 / (1 − s_plus/2)` | facteur CE forcé |

Le `equations_cross_check.md` (2026-04-22) avait déjà identifié cette
incohérence comme "HIGH severity" sans la résoudre.

Liu 2025 (Eq. 25) utilise `pre = -ω × 9/2` (sans facteur CE) — mais
dans un schéma régularisé où le facteur (1-ω/2) est implicitement
absorbé. Pour un Hermite source post-collision dans un TRT standard,
le facteur correct n'est pas évident a priori.

## Test diagnostique

Job Aqua **20544456** (A100) : sweep Wi × 2 modes.

| Mode | hermite_source_mode | drag_mode | Pre-factor | source_force_scale |
|---|---|---|---|---|
| **CE_scaled** (actuel) | `:ce_corrected` | `:source_scaled_mea` | -s_plus × 9/2 × 2.883 | 0.347 |
| **Liu_raw** (test) | `:liu_direct` | `:post_source_mea` | -s_plus × 9/2 | 1.0 |

### Prédictions

Si l'hypothèse est correcte, **Liu_raw** devrait donner :

- Cd_s ≈ Cd_Newt à bas Wi (pas de feedback polymère anormal)
- Pente Cd(Wi) plate, suivant rheoTool
- Cd(Wi=0.1) légèrement différent de 130.74 (le "miracle" disparaît)

Si **Liu_raw** donne aussi des résultats faux mais dans une autre
direction, le bug est ailleurs (probablement dans le terme source du
tenseur de conformation, ou dans la coupling C → τ_p).

## Conséquences pour les validations antérieures

1. La validation `+0.30%` à Wi=0.1 R=30 (`VALIDATION_R30_OLDROYDB.md`)
   reste numériquement vraie pour ce point, mais **n'a pas la valeur
   d'une validation du solver** : c'est un point de croisement.

2. Le sweep Wi montre que le solver actuel n'est **pas utilisable
   tel quel** au-delà du voisinage de Wi=0.1. Pour intégrer le
   viscoélastique dans la lib, il faut résoudre ce problème de
   couplage.

3. L'audit 3D précédent (sphère, ratio 0.89 à Wi=0.1) doit être
   re-exécuté après correction — la cohérence 2D/3D obtenue précédemment
   pourrait elle aussi être un artefact.

## Action immédiate

1. **Attendre résultats job 20544456** (sweep × 2 modes).
2. Si `Liu_raw` donne le bon comportement Cd(Wi) :
   - Changer le défaut du driver à `hermite_source_mode=:liu_direct`
     et `drag_mode=:post_source_mea`.
   - Re-valider Newtonian + Wi=0.1 contre rheoTool avec le nouveau défaut.
   - Re-valider la cohérence 2D/3D à Wi=0.1.
3. Si `Liu_raw` ne corrige pas la pente, investiguer :
   - Le terme source dans `collide_conformation_2d_kernel!` (gradient
     de vitesse, divergence, terme upper-convected).
   - Le couplage `update_polymer_stress!` (formule G·(C−I)).
   - Le streaming de g aux frontières (CNEBB ou stream periodic).

---

## UPDATE 2026-04-30 16:42 — Résultats complets (job 20547171)

3 modes testés × 5 Wi avec dumps de profils.

### Tableau de comparaison final

| Wi   | Mode A (CE)  | Mode B (Liu post) | Mode C (Liu int) | B: Cd_split | rheoTool |
|------|--------------|-------------------|------------------|-------------|----------|
| 0.05 | 149.76 (+14%) | 110.82 (-16%)    | 110.81           | 125.70 (-5%) | 131.81  |
| 0.10 | 130.74 (+0.2%) | 101.94 (-22%)   | 101.93           | 112.59 (-14%) | 130.43 |
| 0.20 | 111.74 (-12%) | 93.28 (-26%)     | 93.27            | 100.14 (-21%) | 126.83 |
| 0.50 |  93.19 (-22%) | 85.10 (-29%)     | 85.10            |  88.68 (-26%) | 119.29 |
| 1.00 |  85.49 (-27%) | 81.72 (-30%)     | 81.72            |  83.91 (-28%) | 117.00 |

### Découvertes invalidant l'hypothèse initiale

1. **post-collision = in-collision** : modes B et C donnent des résultats
   bit-near-identiques (différence < 5e-3 sur Cd). Le débat sur le timing
   de la source était une fausse piste.

2. **Le facteur CE n'est PAS le bug principal** : modes A et B ont des
   champs de vitesse quasi-identiques (Δu_max < 2%) et des τ_p quasi-
   identiques (Δτ < 3%). La différence de 35% sur Cd vient ENTIÈREMENT
   de la mesure du drag.

3. **Le vrai problème est la pente** : peu importe le mode, Kraken
   prédit une réduction de drag monotone et non bornée
   (132 → 85 sur Wi ∈ [0, 1]), alors que rheoTool donne une réduction
   modérée et bornée (132 → 117).

4. **τ_p est correct localement** : profil τ_xy à x=cx column est
   ν_p × γ̇_local comme attendu. La conformation tensor ne semble pas
   être le coupable direct.

### Hypothèses restantes (à tester séparément)

#### H1 — Évolution de C dans le bulk
La dynamique de C contient le terme upper-convected ∇u·C + C·∇uᵀ et
−(C−I)/λ. Un facteur 2 ou un sign flip dans ces termes ne se voit pas
en Poiseuille (C atteint vite l'équilibre) mais affecterait la couche
limite près du cylindre. Test : vérifier C_xx(y) loin du cylindre vs
analytique 1+2(λγ̇)².

#### H2 — Diffusion numérique sur C
Kraken utilise τ_plus=1 → Sc = 3 ν_s / κ_p ≈ 200. Liu utilise τ_plus≈0.5
(Sc ~ 1e4). L'excès de diffusion sur C écrase les pics près du cylindre,
réduisant τ_p_peak et donc la rétroaction polymère sur le flow. Test :
sweep τ_plus ∈ {0.503, 0.51, 0.6, 1.0} à Wi=0.1.

#### H3 — CNEBB biaise C aux parois courbes
Le CNEBB applique une forme de bounce-back conservative pour C aux
frontières solides. Pour un cylindre courbe, la version utilise la
vitesse hydrodynamique locale comme vitesse de paroi (devrait être
zéro en no-slip strict). Cela peut introduire un biais. Test : remplacer
CNEBB par un BB simple sur g et comparer.

#### H4 — Le schéma régularisé Liu
Liu utilise un schéma LBM régularisé pour C, où les populations sont
reconstruites depuis les moments à chaque pas. Kraken utilise TRT standard.
Pour les régimes raides (Wi modéré), la différence pourrait être
significative. Test : implémenter une variante régularisée et comparer.

### Conséquences

- La validation +0.30% à Wi=0.1 R=30 (`VALIDATION_R30_OLDROYDB.md`)
  reste numériquement vraie pour ce point mais **n'est pas une
  validation du solver**.
- Le solver actuel ne reproduit **aucun** point hors Wi=0.1 R=30 β=0.59
  Re_R=1 dans la marge < 5%.
- L'audit 3D précédent (sphère ratio 0.89 à Wi=0.1) doit être considéré
  comme également invalide.
- **Recommandation** : ne pas merger le viscoélastique dans la lib
  publique tant que le bug racine n'est pas identifié et corrigé.

### Données disponibles

- Logs Aqua : `results/sweep_wi_cylinder_20260430_1602.txt` (2 modes),
  `results/sweep_wi_cylinder_20260430_1642.txt` (3 modes)
- Dumps de profils + state : `results/sweep_wi_fields_20260430_1642/`
  (téléchargé localement dans `tmp/sweep_wi_fields/`)
- RheoTool référence : `bench/rheotool/sweep_wi_results.txt`
- Cas rheoTool : `bench/rheotool/cylinder_wi*/`

## Liens

- Job Aqua initial (sweep CE_scaled seul) : `20531344` →
  `results/sweep_wi_cylinder_20260430_1111.txt`
- Job Aqua A/B test : `20544456` (queue A100)
- Cas rheoTool référence : `bench/rheotool/cylinder_wi*` (Wi=0.05, 0.1,
  0.2 terminés ; 0.5, 1.0 en flight)
- Cross-check existant : `bench/equations_cross_check.md` (Finding 1)
- Driver concerné : `src/drivers/viscoelastic.jl:454-810`
- Kernel source : `src/kernels/collide_viscoelastic_source_2d.jl`
