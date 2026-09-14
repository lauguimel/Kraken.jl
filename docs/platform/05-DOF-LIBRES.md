# L'AXE DES DEGRÉS DE LIBERTÉ LIBRES — du paramètre fixe au modèle appris

> Le cœur conceptuel de la plateforme. Ce qui la rend GÉNÉRALE n'est pas une physique ni une
> méthode : c'est la capacité de transformer **n'importe quelle quantité figée** de `R(u,p)`
> (paramètre, champ, force, terme source, **ou modèle constitutif entier**) en un **degré de
> liberté libre** — calibré depuis des données, voire **remplacé par une closure hybride
> IA-modèle**. Tout ça via le **même** `fit`/`loss`/`ParameterSpace`/adjoint.

---

## Les 3 axes orthogonaux

Une simulation = `residual(problem, method, u, p ; closures)`. Trois axes **indépendants** :

1. **Méthode** — LBM, FV/FD, VoF, futur. Fournit `R`.
2. **Physique** — Newtonien, thermique, VE, dépôt de poussière, réaction-diffusion
   (déforestation), … = quels termes composent `R`.
3. **DOF libres** — *ce que tu inverses/apprends* = une **sélection parmi les entrées de `R`**.
   Pilotée par `ParameterSpace` (DOF de forme fixe) + `AbstractClosure` (DOF fonctionnels/appris).

Le moteur différentiable (résidu + adjoint + `fit`) est **agnostique à l'axe 3**. Sa seule
exigence : `R` différentiable par rapport à la quantité libérée.

## La loi d'ingénierie

> **Le coût n'est jamais dans `fit` (générique). Il est dans : rendre `R` différentiable
> vis-à-vis du DOF qu'on veut libérer.**

- Tout ce qui est une **constante** dans `R` (viscosité, force, coefficient, terme source) peut
  devenir un **`p` libre**.
- Toute **fonction figée** (Oldroyd-B / FENE-P / PTT) peut devenir une **`closure` apprise**
  `evaluate(c, inputs, θ)` injectée dans `R`, `θ` = poids NN inférés (ex. depuis PIV).
- `fit` consulte `capabilities()` → gradient adjoint si `SteadyAdjoint`, sinon `FiniteDiff`.

---

## L'échelle des DOF libres (coût honnête)

| DOF libéré | Exprimé comme | VJP requis | Statut |
|------------|---------------|-----------|--------|
| **Géométrie / angle** | `p={radius,angle}`, dérivée de forme | ∂R/∂(forme) | ✅ **câblé** (`ad_api.jl`, cas spécial) |
| **Paramètre scalaire** (ν, force, Wi, Ra) | `p={ν}` ∈ ParameterSpace | ∂R/∂ν (Enzyme) | ◐ **petite extension** : dé-`Const`-er + chemin non-fusionné |
| **Champ** (ν(x), source/puits s(x), taux dépôt, forme d'onde) | `p={ν_i}` sur cellules/base | ∂R/∂ν(x) | ◐ **même moteur**, haute-dim + régularisation |
| **Closure grey-box** (modèle physique + correction NN) | `evaluate(c,inputs,θ)` dans `R` | ∂R/∂θ | ✗ **vrai travail** : `AbstractClosure` + NN-dans-résidu (Phase 3) |
| **Closure black-box** (terme/constitutif entièrement appris) | `evaluate` *remplace* un terme | ∂R/∂θ | ✗ même hook + identifiabilité données |

Lignes 2-5 = **même machinerie**. La géométrie est câblée en premier par accident historique
(le travail QoI portait sur la forme), pas parce que c'est le cas naturel ou le plus utile.

**Caveat kernel (BILAN §A2/§E) :** certains chemins de production sont fusionnés et **non
Enzyme-diff** (ex. LI-BB fusionné). Libérer un paramètre sur ces chemins exige une **variante
non-fusionnée** différentiable (le chemin steady `ad_step!` l'est déjà). Le coût par DOF =
« rendre le terme concerné de `R` différentiable », pas réécrire le moteur.

---

## Les exemples cibles, posés sur l'échelle

| Cas utilisateur | Physique (axe 2) | DOF libéré (axe 3) | Rung |
|-----------------|------------------|--------------------|------|
| Dépôt de poussière + optimiser forme/angle | advection-diffusion + puits de dépôt | géométrie *ou* params dépôt | 1-2 |
| Optimiser une forme d'onde / contrôle (DOD) | la physique du procédé | champ de contrôle (espace/temps) | 3 |
| Déforestation : source→nucléation-diffusion, retrouver localement | réaction-diffusion + champ source | **champ source s(x) inféré du motif** | 3 |
| Propriétés mécaniques d'un fluide | la rhéologie | params constitutifs (scalaire/champ) | 2-3 |
| **VE, PIV, Oldroyd-B/FENE-P échouent → PTT-modifié-IA** | VE log-conf | **closure θ inférée de la PIV** (grey→black) | 4-5 = **étoile polaire** |

Le dernier cas est la démonstration phare du moat JEI : quand les modèles classiques échouent,
on **infère** (partie ou totalité d') le modèle constitutif depuis les données — c'est
`closure` (Phase 3) + adjoint-paramètre (Phase 4) combinés, derrière exactement le même contrat.

---

## Ce que doit garantir l'architecture (pour que l'axe 3 reste ouvert)

1. `p` peut être **n'importe quel sous-ensemble** des entrées de `R` (scalaire, champ, poids NN) —
   c'est le rôle de `ParameterSpace` (named↔flat, log-scale, bornes, fixed/free).
2. Un **point d'injection unique** pour les DOF fonctionnels : `AbstractClosure`/`evaluate` **dans
   le résidu** (jamais dispersé dans les kernels).
3. La VJP `(∂R/∂p)ᵀv`, `(∂R/∂θ)ᵀv` provient de l'adjoint existant, **étendue entrée par entrée**.
4. `fit` reste **identique** quel que soit le DOF ; seule change la parametrization + la
   régularisation (champs/closures haute-dim).
5. `capabilities()` déclare, **par méthode et par DOF**, si l'adjoint est disponible — l'appelant
   (humain ou agent NL) sait alors ce qui est inférable avant de lancer.

> Conséquence pour la tête de pont (corrige `01-PLAN-DE-MATCH.md`) : le premier milestone n'est
> **pas** « inversion géométrique » mais **« libérer une quantité fixe (scalaire d'abord, puis
> champ) et l'inférer depuis des données »**, la géométrie venant en prime. L'objectif est de
> prouver que **« libérer ν », « libérer un champ source » et « remplacer le modèle par une IA »
> sont le même geste** sur des entrées différentes de `R`.
