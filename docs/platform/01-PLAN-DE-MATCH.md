# PLAN DE MATCH — séquencé, par seams, anti-dette

> Croissance par **coutures stables** autour du LBM qui marche. Chaque phase : objectif,
> milestone **vérifiable**, risque, et **garde anti-dette**. On ne remplit que ce qu'un cas
> réel exige (principe #6/#8 du brief).
>
> **Principe directeur (ta correction « rester général ») :** le contrat est neutre vis-à-vis
> méthode ET physique. Le VE n'est **pas** la tête de pont — il est le *cas dur de validation*.
> La tête de pont est la **sensibilité/inversion géométrique steady**, démontrée d'abord sur le
> **Newtonien**, puis « gratuitement » thermique + VE (même adjoint). Cf. `00-BILAN.md` §A2.

---

## Reframe de la tête de pont (vs. milestone 1 du brief)

**La cible générale est l'AXE DES DOF LIBRES** (voir `05-DOF-LIBRES.md`) : libérer n'importe quelle
quantité figée de `R` (paramètre, champ, force, terme source, ou modèle constitutif → closure IA)
et l'inférer depuis des données, via le **même** `fit`. La géométrie n'est PAS la cible — c'est
juste le seul DOF câblé aujourd'hui.

Le brief disait : *« calibrer un paramètre matériau VE en steady avec l'AD existant »*.
**Impossible tel quel** : l'AD existant ne différencie que la géométrie (BILAN §A2). Mais c'est une
limite de câblage, pas de moteur. Échelle de tête de pont (coût croissant, même machinerie) :

- **Rung 0 — géométrie (gratuit, déjà là) :** inverse de forme/angle, marche déjà sur
  Newtonien/thermique/VE. Sert de **test de parité** du contrat, pas de finalité.
- **Rung 1 — paramètre scalaire (tête de pont recommandée) :** libérer ν / une force / Wi et
  l'inférer depuis des données. Petite extension (dé-`Const`-er + chemin non-fusionné), mais c'est
  le **premier vrai geste « quantité fixe → DOF libre »** que tu décris. Plus représentatif que la
  géométrie.
- **Rung 2 — champ :** ν(x), un champ source/puits (déforestation), une forme d'onde. Même moteur,
  haute-dim + régularisation.
- **Rung 3-4 — closure apprise (étoile polaire) :** PTT-modifié-IA inféré depuis la PIV.
  Phase 3 (closure) + Phase 4 (adjoint-paramètre). Vrai travail, même contrat.

Le but du plan : prouver que **rung 1, 2 et 3 sont le même `fit`** sur des entrées différentes de
`R`. On choisit le **premier rung concret selon où il y a de la vraie donnée** (à trancher).

---

## Phase 0 — Poser le contrat (squelette, non-cassant)

**Objectif :** introduire les types et verbes stables **autour** du solveur, sans toucher aux kernels.

- `AbstractProblem`, `AbstractMethod`, `AbstractSolution`, `AbstractObservable`, `AbstractClosure`,
  `capabilities(m)::Set{Capability}`.
- `solve(problem, method, p)` = wrapper mince qui appelle `run_simulation` et **emballe** le
  NamedTuple existant dans une `LBMSolution <: AbstractSolution` (zéro régression).
- `sample(sol, field, query)` réécrit `extract_line`/`probe` derrière le type Solution.
- `LBM <: AbstractMethod` ne fait au début qu'**encapsuler** le dispatch actuel.

**Milestone vérifiable :** un cas LBM existant tourne via `solve(...)` et rend exactement les
mêmes nombres que `run_simulation` (test d'égalité bit-à-bit). `capabilities(LBM())` renvoie au
moins `{ForwardSolve, GPUExecution, SteadyAdjoint}`.

**Risque :** sur-design d'abstractions spéculatives.
**Garde anti-dette :** **le contrat ne se généralise QUE quand un 2e implémenteur le force.**
En Phase 0 il y a un seul `AbstractMethod` (LBM) ; on n'invente pas d'IR.

---

## Phase 1 — Observables & prédiction (la couche comparable-aux-données)

**Objectif :** `observe(sol, obs)::Prediction` défini **via `sample()`**, jamais via le stockage
interne de la méthode (principe brief). Quelques observables réels : `DragCoefficient`,
`FieldProbe`, `LineProfile`, `NusseltNumber`.

**Milestone :** `predict(problem, method, p) = observe(solve(...), obs)` reproduit les QoI déjà
calculées par `steady_shape_sensitivity` (`value`) à tolérance machine.

**Risque :** observables qui fuient l'implémentation (accès direct aux arrays).
**Garde :** revue — tout `observe` passe par `sample`. Lint/skill (cf. hygiène).

---

## Phase 2 — Exposer le seam résidu + calibration steady GÉNÉRALE (tête de pont)

**Objectif :** exposer `residual(problem, method, u, p)` et, si `SteadyAdjoint ∈ capabilities`,
les produits `(∂R/∂u)ᵀv` et `(∂R/∂p)ᵀv` **à partir de l'adjoint LBM existant** (`R(u)=u−G(u)`
est déjà calculé, BILAN §A1). Puis la pile calibration **agnostique** :
`ParameterSpace` (named↔flat, log-scale, bornes, fixed/free), `loss`, `fit`.

**Milestone :** `fit(problem, LBM(), data, p0; pspace)` doit résoudre **deux DOF différents avec le
même code** : (a) **parité** — un inverse géométrique (déjà câblé) ; (b) **tête de pont réelle** —
libérer un **paramètre scalaire** (ν / force) et l'inférer depuis des données, gradient AD (pas
FiniteDiff). Que `pspace` contienne `{radius}` ou `{ν}` ne change que la parametrization, pas `fit`.
Puis montrer que le même `fit` accepte un **champ** `{ν_i}` (rung 2). Benchmark obligatoire par
DOF : cas canonique + solution de référence + gradient de réf. Cf. `05-DOF-LIBRES.md`.

**Risque :** confondre « résidu de point fixe » et « résidu PDE » ; chemins fusionnés non
Enzyme-diff (un DOF scalaire peut exiger une variante non-fusionnée, BILAN §A2) ; le VE casse (annulation 20×).
**Garde :** `residual` est défini *par méthode* (LBM = `u−G(u)`), pas comme une PDE symbolique.
Le VE garde son `fwd_tol=1e-13` et **n'est pas** sur le chemin critique du milestone (cas dur, pas bloquant).

---

## Phase 3 — Closures (injection unique dans le résidu)

**Objectif :** `evaluate(c::AbstractClosure, inputs, θ)` — point d'injection **unique** d'un terme
appris **ou** analytique, **dans le résidu**. Première closure = **analytique** (ex. correction
constitutive ou de fermeture), même API que le futur NN.

**Milestone :** une closure analytique branchée modifie `R` et le gradient passe à travers
(`∂R/∂θ`), validé vs FiniteDiff sur un cas jouet **général** (pas forcément VE).

**Risque :** une API closure trop liée au VE.
**Garde :** la 1re closure doit être démontrée sur ≥2 contextes (ex. un terme source générique)
pour rester neutre.

---

## Phase 4 — Moteurs orthogonaux (SciML) + paramètres matériau

**Objectif :** (a) brancher le **temporel** derrière le même contrat via
`DifferentialEquations + SciMLSensitivity + Optimization` (ne PAS réinventer SciML, principe #3) ;
(b) **Voie M** : étendre l'adjoint steady aux **paramètres matériau** → vraie calibration
rhéologique/physique (l'actif JEI).

**Milestone :** un problème temporel simple résolu+calibré via `solve/observe/fit` sur backend SciML ;
et un `∂(QoI)/∂(param matériau)` validé vs FiniteDiff.

**Risque :** double surface (Kraken vs SciML).
**Garde :** SciML est **derrière** le contrat, pas exposé ; le `.krk`/JSON ne connaît que `method`.

---

## Phase 5 — Transient-AD isolé (spike, jamais bloquant)

**Objectif :** prototype d'adjoint transitoire sur **un** cas, via checkpointing + route array/XLA
(façon XLB/Reactant), en s'attaquant aux bloqueurs de `00-BILAN.md` §E (sous-cyclage, ping-pong).

**Milestone :** gradient transitoire validé vs FiniteDiff sur un cas minimal **sans sous-cyclage**.
**Risque :** tar pit. **Garde :** **la plateforme ne dépend jamais de cette phase** (principe #4).
Time-boxé ; si échec, on reste steady + SciML.

---

## Vue d'ensemble (dépendances)

```
P0 contrat ──> P1 observables ──> P2 résidu+calibration steady (TÊTE DE PONT, Voie G)
                                        │
                                        ├──> P3 closures (analytique d'abord)
                                        ├──> P4 SciML (temporel) + Voie M (params matériau)
                                        └──> P5 transient-AD (spike isolé, non bloquant)
```

Chaque phase est **livrable seule** et n'ouvre la suivante que sur milestone vert. Les méthodes
forward déjà présentes (VoF/diphasique, FV/FD VE 2D/3D) entrent **derrière le contrat en P0/P1
comme `AbstractMethod` forward**, sans dette, sans être rendues différentiables prématurément.
