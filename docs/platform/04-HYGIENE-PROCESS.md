# HYGIÈNE DÉPÔT, PROCESS & RED-TEAM DU PLAN

> État git constaté le 2026-06-09 (audit lecture seule). Inclut une stratégie worktrees,
> branches/bugs, nettoyage, skills, et une **critique adversariale du plan lui-même**.

---

## 1. État git constaté

- **15 worktrees** d'un même dépôt `Kraken.jl` + **5 dépôts séparés** (`Kraken-AMR.jl`,
  `kraken-foam`, `kraken-research`, `kraken-sim`, `Kraken.jl.backup.mirror`) + 1 worktree
  externe (`Deforestation/kraken`).
- **Désync worktrees (à réparer en premier, non destructif) :** tout a été déplacé sous
  `kraken/` sans mettre à jour le lien git → les 15 worktrees liés sont marqués `prunable`
  (faux positif). Les commandes `git -C <worktree>` échouent ; la lecture de fichiers marche.
  **Fix :** `git -C .../kraken/Kraken.jl worktree repair <tous-les-nouveaux-chemins>` puis
  re-vérifier `git worktree list` (plus de `prunable`).
- **Branches 100 % mergées** (0 commit d'avance, contenu déjà dans le trunk `dev/v0.3-campaign`)
  → retirables : `feat/units-on-v03` (Kraken.jl-geometry-stl), `feat/krk-meanvelocityforce`
  (Kraken.jl-krk-force).
- **Detached anonyme** : `Kraken.jl-battery` (HEAD `9353d5e10`, smoke-test FVFD log-conf) → nommer ou jeter.
- **Non-buildable** : `dev-viscoelastic` HEAD (includes non trackés) → à réparer avant tout merge.
- **Dépôts dormants** depuis mars 2026 : `kraken-foam`, `kraken-research`, `kraken-sim` (commits
  d'init seulement) → archiver ou clarifier le rôle.

---

## 2. Stratégie worktrees (anti « 50 worktrees »)

**Règle :** un worktree = une **mission active** vivant ≤ semaines, pas un dossier permanent.

| Persistant (garder) | Éphémère (créer→détruire) |
|---------------------|---------------------------|
| `Kraken.jl` (trunk `dev/v0.3-campaign`) | un worktree par feature/bug **en cours**, supprimé au merge |
| `Kraken-release-v0.2` (release publique) | worktrees dual-spawn orchestrateur (`*_codex`/`*_claude`) |
| `Kraken.jl-ad-steady` tant que l'AD évolue (sinon retirer : déjà sur trunk, §A6 du BILAN) | probes temporaires (`/tmp/...`) |

**Conventions :**
- Nommage : `Kraken.jl-<topic-court>` ; branche `feat|fix|dev/<topic>`.
- **Création** : seulement quand le travail est parallèle au trunk ET dure > 1 session.
- **Destruction** : à chaque merge → `git worktree remove` + `git branch -d` (refuse si non mergé).
- **Plafond cible** : ≤ 6 worktrees vivants. Au-delà, audit (`git-audit` skill) + élagage.
- **Audit périodique** : `bash ~/.claude/scripts/git-audit/branch-audit.sh --repo=.../Kraken.jl`.

---

## 3. Branches & bugs (isoler sans polluer le trunk)

- **Trunk = intégration**, jamais le terrain de debug. Un bug → branche `fix/<bug>` depuis le
  point de divergence concerné, worktree dédié, **canary reproduisant le bug** d'abord (test
  rouge), puis fix (test vert), puis merge.
- **Remontée** : `fix/*` → trunk ; backport vers `release/*` **seulement** par cherry-pick
  explicite documenté (jamais merge release←dev en masse).
- **Aucun fix sur `release/*`** directement ; release ne reçoit que des cherry-picks validés.
- Chaque fix arrive avec son **test de non-régression** (le canary devient permanent).

---

## 4. Nettoyage (code mort, README menteurs, incohérences)

| Cible | Action |
|-------|--------|
| `docs/src/architecture.md` (décrit l'ancien layout LBM-only) | réécrire selon le contrat (`02` §refactor 3) |
| `simulation_runner.jl` ~1843 LOC (dispatch par matching de chaîne `setup.name`) | extraire un **registre de méthodes typé** (après contrat posé) |
| `io/kraken_parser.jl` ~2010 LOC mal placé | reloger sous `io/krk/` |
| `bench/scratch/rheotool_*` (faux positifs « FVFD solver ») | confirmer que c'est de la réf RheoTool, documenter, ne pas confondre |
| ~132 clés `refs.bib` pendantes | nettoyer avant tout tag release |
| `dev-viscoelastic` non-buildable | réparer les includes ou marquer la branche archive |

**Règle anti-dette :** tout nettoyage > déplacement de fichier passe par une PR avec test de parité.

---

## 5. Skills — inventaire & propositions

**Existants utiles à la plateforme :** `kraken-architect` (stratégie/worktrees), `kraken-codebase-map`
(carte code), `kraken-trace` (provenance runtime), `kraken-codex-pilot` (exécution Julia),
`kraken-doc` (docs), `git-audit` (audit branches), `orchestrator`. Côté Codex :
`kraken-fvfd-operator-library`, `kraken-branch-governor`, `kraken-resource-integrator`,
`kraken-amr-canary` (seule discipline de validation, mais **spécifique AMR**).

**GAP majeur (le brief le demande explicitement) :** aucun skill n'encode **le contrat stable +
le protocole de benchmark obligatoire** pour ajouter une méthode/physique. Proposition :

- **Skill proposé `kraken-platform-contract`** : « tout nouvel `AbstractMethod`/physique doit
  implémenter `solve/sample/observe/capabilities`(+`residual` si SteadyAdjoint), passer le test
  de parité, et livrer son benchmark (cas canonique + réf analytique + ordre de convergence +
  gradient de réf). » → encode `02` §invariants. **Codex** (s'exécute pendant le dev).
- **Skill proposé `kraken-inverse-contract`** (optionnel) : schéma JSON enveloppe-inverse +
  erreurs structurées + validation `capabilities`, pour l'axe NL.

> Création de skill = **jamais silencieuse** : à valider avec toi avant d'écrire le fichier.

---

## 6. RED-TEAM du plan (critique adversariale)

Le plan (`01`/`02`) attaqué de front — où peut-il créer de la dette ?

1. **Sur-abstraction spéculative.** Risque : poser `AbstractMethod`/`capabilities`/`residual`
   comme un grand framework avant qu'un 2e implémenteur ne le force → un IR maison déguisé.
   *Mitigation :* Phase 0 = **un seul** `AbstractMethod` (LBM), wrappers minces ; on ne généralise
   un verbe que quand une 2e méthode l'exige (PR 6+). Le test de **parité** interdit la dérive.

2. **Le résidu qui ment.** `R(u)=u−G(u)` est un résidu de **point fixe**, pas un résidu PDE. Le
   présenter comme « la primitive PDE partagée » rejouerait l'illusion du brief (BILAN §A1).
   *Mitigation :* `residual` est documenté **par-méthode** ; on n'écrit jamais d'assembleur PDE
   symbolique (principe #2).

3. **Le piège matériau-AD.** Promettre la calibration matériau « avec l'AD existant » est faux
   (BILAN §A2). Si le narratif JEI s'appuie dessus, c'est un trou.
   *Mitigation :* tête de pont = inverse **géométrique** (Voie G) ; calibration matériau = Phase 4
   reconnue comme vrai dev d'adjoint.

4. **VoF/transient sirène.** La capture d'interface est séduisante mais transitoire et non
   différentiable → tentation d'y investir tôt et de se retrouver bloqué sur le transient-AD.
   *Mitigation :* VoF entre **derrière le contrat comme forward**, pas dans la boucle différentiable.
   Transient-AD = Phase 5 time-boxée, **jamais bloquante** (principe #4).

5. **Double surface SciML.** Réimplémenter du temporel/ODE alors que SciML existe = dette pure.
   *Mitigation :* SciML **derrière** le contrat, invisible au `.krk`/JSON (principe #3).

6. **Deux parsers (`.krk` vs JSON).** Risque de divergence sémantique.
   *Mitigation :* `.krk` reste **canonique** ; le JSON ne couvre que l'enveloppe inverse, le
   forward = texte `.krk` (principe #7).

7. **Couche NL prématurée.** Coder l'agent avant d'avoir `capabilities()` + erreurs structurées =
   château sur sable.
   *Mitigation :* axe 4 explicitement **non codé** ; pré-requis listés (`03` §récap) d'abord.

8. **Worktree sprawl récurrent.** Sans discipline, on re-créera 15 worktrees.
   *Mitigation :* §2 (plafond ≤6, destruction au merge, audit périodique).

**Verdict red-team :** le plan tient *si* l'ordre est respecté (contrat → parité → un 2e
implémenteur → généralisation) et *si* la tête de pont reste **géométrique et générale** (pas VE,
pas matériau). Les deux plus gros risques de dette sont l'**over-engineering du contrat** (#1) et
le **trou narratif matériau-AD** (#3).
