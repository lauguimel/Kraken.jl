# Reprise WP-MESH-6 — debug bug C local-CFL

Branche : `slbm-paper` (44 commits depuis `lbm`)
Worktree : `/Users/guillaume/Documents/Recherche/Kraken.jl`

## Contexte rapide

Le pipeline d'import gmsh est livré (WP-MESH-1..5, voir
`STATUS_SLBM.md` et `project_gmsh_workflow.md` dans la mémoire auto).
Le but de WP-MESH-6 est de démontrer le **vrai** gain SLBM+LI-BB
sur un mesh **structuré non-régulier** raffiné dans la couche limite,
contre un mesh Cartesien uniforme avec **exactement le même nombre
de cellules totales**.

Setup (commits `7b321fe`, `095aa75`, `4afa9d6`, et patch non-commité
`cy_p = 0.245`) :

- **Cylinder cross-flow Re=100 (Williamson 1996)** :
  domaine 1.0 × 0.5, cylindre R=0.025, blockage 10 %
- Référence : Cd = 1.4, Cl_RMS = 0.33, St = 0.165
- 3 résolutions D_lu = 20 (80 k cells), 40 (321 k), 80 (1.28 M)
- 3 baselines à **même cell count** :
  - (A) `cartesian_mesh` + `fused_trt_libb_v2_step!` halfway-BB (q_w = 0.5)
  - (B) `cartesian_mesh` + `fused_trt_libb_v2_step!` LI-BB
  - (C) **gmsh single-block Transfinite Bump 0.1** + `slbm_trt_libb_step!`
        → cellules denses au centre (cylindre), 93× plus grosses aux bords

## Bug bloquant à résoudre

Aqua job `20155686` (H100 FP64) montre :

| | Cd err | Cl_RMS err |
|---|---|---|
| (A) D=20 | 5.5 % | 100 % |
| (B) D=20 | 4.2 % | 100 % |
| **(C) D=20** | **NaN** ⚠ | NaN |
| (A) D=40 | 2.7 % | 100 % |
| (B) D=40 | 2.4 % | 100 % |
| **(C) D=40** | **12.9 %** ⚠ | 100 % |

Deux problèmes :

### Bug 1 — pas de shedding (Cl_RMS = 100 % d'erreur)

Symétrie parfaite cylindre/inlet → flow stationnaire forcé. Patch
**non-commité** dans `hpc/wp_mesh_6_bump_aqua.jl` : `cy_p = 0.245`
au lieu de `0.25` (asymétrie 10 % du diamètre, style Schäfer-Turek
2D-2). À commit + tester.

### Bug 2 — (C) SLBM diverge ou biaise (NaN à D=20, Cd 12.9 % à D=40)

**Hypothèse principale** : le mesh Bump 0.1 a un ratio de taille de
cellule de 93× entre centre et bords. Le SLBM est appelé avec un
unique τ global (`slbm_trt_libb_step!` accepte `ν::Real`, recalcule
`s_plus, s_minus = trt_rates(ν)` une fois). Sur les cellules les
plus petites (centre), τ_eff ≈ 0.5 → instabilité numérique. Sur les
plus grosses (bords), ν_eff trop élevée → mauvais Cd.

**Solution attendue** : utiliser `compute_local_omega_2d` (existe
déjà à `src/curvilinear/slbm.jl:626`) pour calculer `sp[i,j]`,
`sm[i,j]` rescalés pour préserver une viscosité physique constante,
puis appeler une variante locale du step.

**Bloquant** : `slbm_trt_libb_step_local_2d!` n'existe pas encore.
La version 3D est livrée à `src/curvilinear/slbm_3d.jl` (commit
`8c9bcb0`). Mirror direct nécessaire en 2D.

## TODO ordonné

### 0. Récupérer le log final

```bash
ssh aqua "tail -50 ~/Kraken.jl/wp_mesh_6_bump.log"
scp aqua:~/Kraken.jl/wp_mesh_6_bump.log paper/data/wp_mesh_6_bump_h100_v1_buggy.log
```

Garder pour archive et benchmarker contre le run corrigé.

### 1. Implémenter `slbm_trt_libb_step_local_2d!` (mirror 3D)

Cible : `src/curvilinear/slbm.jl`.

```julia
const _SLBM_TRT_LIBB_LOCAL_SPEC_2D = LBMSpec(
    PullSLBM(), SolidInert(),
    ApplyLiBBPrePhase(),
    Moments(), CollideTRTLocalDirect(),
    WriteMoments();
    stencil = :D2Q9,
)

function slbm_trt_libb_step_local_2d!(f_out, f_in, ρ, ux, uy, is_solid,
                                       q_wall, uw_link_x, uw_link_y,
                                       geom::SLBMGeometry,
                                       sp_field, sm_field)
    backend = KernelAbstractions.get_backend(f_in)
    kernel! = build_lbm_kernel(backend, _SLBM_TRT_LIBB_LOCAL_SPEC_2D)
    kernel!(f_out, ρ, ux, uy, f_in, is_solid,
            q_wall, uw_link_x, uw_link_y,
            geom.i_dep, geom.j_dep,
            geom.Nξ, geom.Nη,
            sp_field, sm_field,
            geom.periodic_ξ, geom.periodic_η;
            ndrange=(geom.Nξ, geom.Nη))
end
```

Brick `CollideTRTLocalDirect` 2D existe déjà
(`src/kernels/dsl/bricks.jl:177`). Export dans `src/Kraken.jl`.

### 2. Patcher `run_C` du driver

Dans `hpc/wp_mesh_6_bump_aqua.jl`, fonction `run_C`, après le
`build_slbm_geometry(mesh)` :

```julia
sp_h, sm_h = compute_local_omega_2d(mesh; ν=s.ν, scaling=:quadratic)
sp = CuArray(T.(sp_h))
sm = CuArray(T.(sm_h))
```

Et remplacer la ligne du step :
```julia
slbm_trt_libb_step_local_2d!(fb, fa, ρ, ux, uy, is_solid, q_wall,
                              uw_x, uw_y, geom, sp, sm)
```

### 3. Commit + sync + resub

```bash
git add hpc/wp_mesh_6_bump_aqua.jl src/curvilinear/slbm.jl src/Kraken.jl
git commit -m "feat(slbm): slbm_trt_libb_step_local_2d! + WP-MESH-6 cy=0.245 + run_C local-tau"
bash hpc/sync_to_aqua.sh --apply
ssh aqua "cd ~/Kraken.jl && qsub hpc/wp_mesh_6_bump_aqua.pbs"
```

Monitor le job avec :
```julia
# Use the Monitor tool with this command
prev=""; while true; do
  state=$(ssh -o ConnectTimeout=15 aqua "qstat -x <JOB_ID>.aqua 2>/dev/null | tail -1 | awk '{print \$5}'" 2>/dev/null)
  if [ -n "$state" ] && [ "$state" != "$prev" ]; then
    echo "$(date +%H:%M:%S) state=$state"; prev=$state
  fi
  if [ "$state" = "F" ]; then
    ssh aqua "tail -50 ~/Kraken.jl/wp_mesh_6_bump.log"; exit 0
  fi
  sleep 90
done
```

### 4. Si (C) converge → figure + paper

Plotter prêt :
```bash
scp aqua:~/Kraken.jl/wp_mesh_6_bump.log paper/data/wp_mesh_6_bump_h100.log
julia --project=docs scripts/figures/plot_bump_matrix.jl \
    --log paper/data/wp_mesh_6_bump_h100.log \
    --out paper/figures/bump_convergence.pdf
```

Update `paper/3d_extension.md` Sec. 4.5 ou créer Sec. 6 avec la
matrice Bump (en plus de la ST 2D-2 actuelle).

### 5. Si (C) diverge encore

Plan B : réduire `BUMP_COEF` de 0.1 à 0.3 (stretching plus modéré,
ratio cell-area ~10× au lieu de 93×). C'est plus représentatif d'un
vrai O-grid CFD.

Plan C : revoir le test de stabilité TRT — peut-être que `Λ = 3/16`
n'est pas optimal pour SLBM avec local-CFL, tester `Λ = 1/4` ou
`Λ = 1/12`.

## Pitfalls à garder en tête (mémoire auto)

- **Drag formula** : `compute_drag_libb_2d` (MEA simple) sur-estime de
  50% en LI-BB sub-cell. Toujours `compute_drag_libb_mei_2d`
  (`feedback_cylinder_benchmark.md`).
- **Wall-classification artefact** : sur Cartesien uniforme avec
  R = N×dx pile, le test `d²≤R²` integer-exact classe les nodes
  polaires en solid → asymétrie staircase. Cause Cl_RMS faussement
  bas en (B). Bug nature, pas un gain.
- **Aqua Manifest.toml gitignored** : le PBS doit faire
  `Pkg.add(["Interpolations", "Gmsh"])` AVANT `Pkg.instantiate()`.
  Déjà géré dans `hpc/wp_mesh_6_bump_aqua.pbs`.
- **Toujours GPU local** sur Mac M3 Max via `MetalBackend()` Float32.
  Run CPU = 30× plus lent.
- **Conventional commits**, en anglais. Pas de mention Claude/AI/LLM
  dans les commits/PRs/issues. Ne jamais push sans confirmation.

## Etat des fichiers (au moment de la sauvegarde)

```
hpc/wp_mesh_6_bump_aqua.jl      (cy=0.245 patch en working tree, NON COMMITÉ)
hpc/wp_mesh_6_bump_aqua.pbs     (commited)
scripts/figures/plot_bump_matrix.jl (commited, prêt à plotter)
paper/figures/bump_mesh_preview.{pdf,png} (commited, mesh validé visuellement)
src/curvilinear/slbm.jl         (compute_local_omega_2d existe, slbm_trt_libb_step_local_2d à ajouter)
src/curvilinear/slbm_3d.jl      (référence : slbm_trt_libb_step_local_3d existe)
src/kernels/dsl/bricks.jl       (CollideTRTLocalDirect existe ligne 177)
```

## Job Aqua actif au moment de la sauvegarde

`20155686.aqua` — encore en R, devrait finir vers 11:25.
Résultats partiels documentés ci-dessus.
