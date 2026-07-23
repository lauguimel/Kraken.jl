# EXPLORATION DES AXES — options, pas une voie

> Explorer & cadrer. **Aucun de ces axes ne se code maintenant.** On vérifie seulement que le
> contrat (`00`/`01`/`02`) laisse une porte propre pour chacun. Frontière clé (principe #7) :
> le **JSON couvre l'enveloppe problème-inverse** (data, observables, paramètres, loss, résultat) ;
> le **modèle forward reste par-méthode** (`"method":"lbm","config":{...}` = texte `.krk`).

---

## ADR-01 — Formats de données & sortie (VERROUILLÉ 2026-06-09)

**Décision :** deux substrats, deux jobs distincts.
- **Viz + archivage des champs → VTKHDF** (HDF5 sous le capot, **lu nativement par ParaView** ≥5.12 :
  séries temporelles en **un seul fichier**, ImageData pour la grille cartésienne LBM, OverlappingAMR
  pour l'AMR). Écrit via `WriteVTK.jl` (VTKHDF) ou `HDF5.jl`.
- **Datasets calibration / entraînement IA / cloud → Zarr** (chunké, compressé, écriture parallèle,
  interop Python/xarray/PyTorch). Via `Zarr.jl`.

**Raisons :** ne pas confondre viz (format que ParaView lit) et stockage (n-D chunké pour ML).
HDF5/VTKHDF et Zarr couvrent chacun le meilleur de son rôle ; on n'écrit **aucun writer maison**.

**Conséquences :**
- **Métadonnées vs données lourdes séparées** : l'artefact de run = **JSON** (enveloppe contrat,
  léger, diffable) **pointant** vers les champs lourds VTKHDF/Zarr (cf. Axe 3).
- ⚠️ **Zarr n'est pas lu nativement par ParaView** → réservé stockage/ML, jamais le chemin viz primaire.
- **NetCDF-CF différé** : à activer (via `NCDatasets.jl`, lu par ParaView + xarray) **seulement si**
  la verticale déforestation/géo le réclame ; sinon Zarr suffit.
- Pas de CSV pour les champs (seulement, à la rigueur, des scalaires/séries 0D).

**Implémentation :** writers sous `src/io/` (VTKHDF + Zarr), derrière `solve`/l'artefact de run.

---

## Axe 1 — Ingestion de données expérimentales

**Point d'ancrage contrat :** `data` consommé par `loss(problem, method, p, data)` et comparé via
`observe(sol, obs)::Prediction`. La donnée n'entre **jamais** dans la méthode ; elle entre dans
la *loss*, comparée à des `Prediction` issues de `sample`.

**Options (orthogonales) :**
- **O1.a — `Observation` typée minimale** : `struct Observation{O<:AbstractObservable}` =
  `(observable, value, weight, uncertainty)`. La loss somme `Σ wᵢ ‖observe(sol,obsᵢ)−valueᵢ‖²/σᵢ²`.
  *Faible dette, recommandé pour démarrer.*
- **O1.b — Dataset tabulaire** (CSV/Arrow) → liste d'`Observation`. Un adaptateur, pas un format maison.
- **O1.c — Champs denses** (PIV, thermographie) : `observable = FieldProbe` sur une région ;
  rééchantillonnage via `sample`. Plus lourd ; ne le faire que si un cas réel l'exige.

**Garde anti-dette :** pas de « base de données expérimentale » maison. Adaptateurs vers
`Observation`, point.

---

## Axe 2 — Ingestion de modèles IA pré-entraînés (closures, surrogates)

**Point de branchement unique : `AbstractClosure` + `evaluate(c, inputs, θ)`** (Phase 3). Un NN
pré-entraîné ailleurs est *une closure parmi d'autres*, **même API** que la closure analytique.

**Options pour le format d'échange :**
- **O2.a — ONNX via ONNXRunTime.jl / Lux import** : standard inter-frameworks, robuste, mais
  poids figés (inférence). *Recommandé pour « modèle entraîné ailleurs, utilisé ici ».*
- **O2.b — Lux.jl/Flux natif** : si on veut **co-entraîner** la closure *dans* la boucle `fit`
  (la closure devient des `θ` calibrables → réutilise `ParameterSpace`). Plus puissant, plus couplé.
- **O2.c — Frontière surrogate** : un surrogate qui *remplace* `solve` (pas une closure dans `R`)
  = une `AbstractMethod` à part avec `capabilities={ForwardSolve}` (et éventuellement
  `SteadyAdjoint` si différentiable). Garde les deux notions distinctes : **closure = terme dans
  le résidu ; surrogate = méthode**.

**Garde anti-dette :** un seul point d'injection (`evaluate` dans `residual`). On ne disperse pas
les hooks NN dans les kernels. Grey-box d'abord (closure corrective sur un modèle physique), pas
black-box.

---

## Axe 3 — Boucle de retour utilisateur

**Idée :** chaque run produit un artefact reproductible (problème JSON + méthode/config + résultat
+ métriques) ; les retours (annotations, corrections, nouvelles données) ré-alimentent `data` et
`ParameterSpace`.

**Options :**
- **O3.a — Artefact run immuable** : `(problem.json, method_config, prediction, residual_norm,
  capabilities_used)` écrit à chaque `solve`/`fit`. Base de la reproductibilité ET du futur NL.
- **O3.b — Registre de calibrations** : versionne `p` calibrés par (problème, données) → historique
  réutilisable, point de départ `p0`.
- **O3.c — Boucle active** : la loss/uncertainty suggère la prochaine mesure (design d'expérience).
  *Différé — n'a de sens qu'avec données réelles récurrentes.*

**Garde :** l'artefact run est le **même** objet que le contrat JSON (axe 4) → pas de format parallèle.

---

## Axe 4 — Langage naturel → contrat JSON + API (l'axe que tu veux creuser)

**Thèse :** une couche NL/agent ne pilote PAS le solveur ; elle **émet et répare un JSON** validé
contre le contrat. Le solveur reste déterministe. La boucle d'auto-correction se ferme sur
**`capabilities()` + erreurs structurées**, pas sur du texte libre.

### Le contrat machine (à spécifier, pas à coder)

```jsonc
{
  "problem":   { "domain": {...}, "boundaries": {...} },     // enveloppe
  "method":    "lbm",                                          // par-méthode
  "config":    { "krk": "<texte .krk ou AST sérialisé>" },     // forward = .krk (principe #7)
  "inverse":   {                                               // enveloppe agnostique
     "parameters": [ {"name":"radius","bounds":[..],"scale":"log","free":true} ],
     "observables":[ {"type":"DragCoefficient","value":1.2,"weight":1.0} ],
     "loss":"weighted_l2",
     "data": "ref:dataset_id" },
  "request":   "solve" | "fit" | "predict"
}
```

Le forward (`config`) reste **opaque et par-méthode** ; l'enveloppe `inverse` est **commune**.
C'est la ligne qui empêche un IR universel par accident.

### La boucle NL (auto-correction)

```
NL utilisateur ──(LLM)──> JSON candidat
       │
       ▼  validate(json) contre schema + capabilities(method)
   ┌── erreurs STRUCTURÉES (code, champ, attendu, suggestion) ──┐
   │                                                            │
   └────────── LLM répare le JSON (re-prompt ciblé) ◄───────────┘
       │ (valide)
       ▼ solve/fit/predict déterministe ──> artefact run (axe 3)
```

**Pré-requis côté code (issus de `00-BILAN.md` §C/§D) — manquants aujourd'hui :**
1. `capabilities(method)` introspectable (sinon le LLM ne sait pas ce qui est faisable).
2. **Erreurs structurées** (code + champ + valeur attendue + suggestion) — aujourd'hui ce sont
   des `ArgumentError` en texte libre. C'est *le* chantier habilitant pour l'auto-correction.
3. Sérialiseur JSON de l'enveloppe inverse (trivial) + stratégie `config` (réutiliser le **texte
   `.krk`** comme blob, déjà source-stringifiable).

**Options d'architecture NL :**
- **O4.a — JSON-only, LLM externe** (recommandé pour commencer) : la plateforme n'expose qu'un
  schéma + un validateur + des erreurs structurées. **Aucun code LLM dans Kraken.** N'importe quel
  agent (Claude, etc.) émet le JSON. Découplage total, zéro dette IA dans le solveur.
- **O4.b — Génération contrainte par schéma** (JSON-schema/grammar) : réduit les allers-retours
  de réparation ; le schéma EST le contrat.
- **O4.c — Tool/function-calling** : exposer `solve/fit/predict/capabilities` comme outils ;
  l'agent compose. Plus interactif, plus de surface à maintenir.
- **O4.d — DSL `.krk` comme cible NL directe** : le LLM écrit du `.krk` (déjà conçu pour la
  lisibilité) plutôt que du JSON pour le forward, JSON pour l'inverse. Tire parti de l'existant.

**Garde anti-dette :** le contrat JSON/`.krk` doit être **utilisable à la main sans LLM**. Si un
humain ne peut pas écrire/lire le JSON, l'IA non plus de façon fiable. La couche NL est un
*générateur de contrat*, jamais un chemin d'exécution privilégié.

---

## Ce qui doit exister AVANT tout axe (récap)

| Pré-requis | Axe servi | Statut |
|------------|-----------|--------|
| `capabilities()` introspectable | 2, 4 | absent (`00-BILAN.md` §C) |
| Erreurs structurées (code/champ/suggestion) | 4 | absent (bare exceptions) |
| `AbstractClosure`/`evaluate` | 2 | absent (Phase 3) |
| Artefact run reproductible (JSON) | 3, 4 | absent (zéro JSON) |
| `Observation`/`loss`/`ParameterSpace` | 1, 3 | absent (Phase 2) |

Aucun axe ne se construit avant que `00-02` (le contrat) soient posés.
