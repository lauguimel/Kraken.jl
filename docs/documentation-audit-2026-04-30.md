# Audit documentation - 2026-04-30

Objectif: rendre la documentation propre, humaine, pedagogique, exacte, centree sur les fichiers `.krk`, complete cote API Julia, et exploitable par un agent LLM sous une forme compacte.

## Synthese

La documentation n'est pas encore publiable en l'etat. Le probleme principal n'est pas le volume, mais l'ecart entre les promesses documentees et ce que le code de cette branche expose reellement.

Les pages actuelles donnent deja une bonne base: structure Documenter/VitePress, pages `.krk`, exemples, benchmarks, theorie, reference API. Mais plusieurs pages annoncent des capacites absentes ou non reliees dans le code courant: MRT, axisymetrique, grid refinement, rheologie avancee, VOF, phase-field, Shan-Chen, especes, viscoelasticite. Le site final masque aussi des problemes concrets: icone absente, GIFs non publies, liens locaux casses, liens GitHub mal formes, controles Documenter trop permissifs.

La direction recommandee est de repositionner la documentation autour d'un principe simple:

> Un utilisateur de Kraken commence par un fichier `.krk`; Julia est ensuite l'API avancee, l'extension et l'automatisation.

## Constats critiques

### 1. Derive entre documentation et code

La page `docs/src/capabilities.md` est le plus gros risque d'exactitude. Elle se presente comme source de verite, mais annonce des fonctions ou modules qui ne sont pas presents dans `src/` sur cette branche.

Exemples a corriger ou retirer avant publication:

- MRT et collision axisymetrique annoncees, avec liens vers des pages absentes.
- Grid refinement annonce comme capacite documentee et benchmarkee, mais les pages `theory/18_grid_refinement.md`, `api/refinement.md` et `benchmarks/refinement_showcase.md` n'existent pas.
- Rheologie, VOF, phase-field, Shan-Chen, especes et viscoelasticite sont listees comme implementees ou testees, alors que les sources correspondantes ne sont pas presentes dans `src/`.
- Certaines pages API mentionnent des symboles non exportes ou absents, par exemple des fonctions axisymetriques, MRT, refined drivers ou fused/persistent kernels.

Decision recommandee: separer strictement trois statuts:

- `Stable`: present dans `src/`, exporte si public, teste, documente.
- `Experimental`: present dans le code mais API ou validation non stabilisee.
- `Planned`: non documente comme fonctionnalite utilisable.

### 2. Liens locaux casses

Audit local des liens Markdown: 17 liens internes cassent actuellement. Les plus visibles concernent:

- `theory/12_mrt.md`
- `theory/09_axisymmetric.md`
- `theory/18_grid_refinement.md`
- `api/refinement.md`
- `examples/09_hagen_poiseuille.md`
- `benchmarks/refinement_showcase.md`
- `assets/cavity.krk` depuis `_helpers/_test_helpers.md`

Ces liens sont d'autant plus dangereux que `ignoreDeadLinks = true` cote VitePress, `warnonly = true` et `checkdocs = :none` cote Documenter laissent passer des erreurs de publication.

### 3. Icone et GIFs

Les assets existent dans le depot:

- `docs/src/assets/icon.png`
- `docs/src/assets/showcases/rayleigh_benard_ra1e5.gif`
- `docs/src/assets/showcases/cavity_re1000.gif`
- `docs/src/assets/showcases/vonkarman_re200.gif`
- `docs/src/assets/showcases/taylor_green_decay.gif`

Mais ils ne sont pas presents dans le build VitePress final `docs/build/1`. Les GIFs sont references dans `docs/src/index.md` avec des chemins absolus `/assets/showcases/...`, ce qui suppose des assets publics VitePress. Ils sont copies dans l'intermediaire Documenter, pas dans le site final.

Actions recommandees:

- Ajouter les assets publics sous un repertoire VitePress public, par exemple `docs/src/public/assets/...`, ou configurer explicitement `publicDir`.
- Remplacer les placeholders de `docs/src/.vitepress/config.mts` pour `favicon` et `themeConfig.logo`.
- Verifier que le build final contient bien `icon.png` et les GIFs.

### 4. Benchmarks et materiel

La documentation melange Apple M2, Apple M3 Max, H100 et A100.

Constats:

- `benchmarks/hardware.toml` contient `apple_m3max`, `aqua_h100`, `aqua_a100`.
- `docs/src/benchmarks/hardware.md` et `benchmarks/README.md` parlent encore de `apple_m2`.
- Les resultats disponibles incluent un CSV CPU `apple_m2`, un CSV Metal `apple_m3max`, et des fichiers de convergence H100.
- Les chiffres H100 affiches dans `docs/src/benchmarks/performance.md` ne sont pas rattaches de facon evidente a un CSV de provenance dans `benchmarks/results/`.

Position recommandee: ne pas faire du M2 ou M3 Max un axe narratif public. Pour une documentation propre, utiliser:

- H100 comme benchmark GPU principal.
- CPU comme baseline suffisante et reproductible.
- M3 Max seulement dans les resultats bruts ou historiques, si utile.

Chaque chiffre publie doit pointer vers:

- hardware id,
- commande lancee,
- commit ou version,
- CSV source,
- date,
- precision,
- taille de grille,
- backend,
- options de benchmark.

### 5. `.krk` comme concept central

Les fichiers `.krk` existent et couvrent deja les cas utiles:

- `examples/cavity.krk`
- `examples/couette.krk`
- `examples/rayleigh_benard.krk`
- `examples/poiseuille.krk`
- `examples/cylinder.krk`
- `examples/taylor_green.krk`
- `examples/cavity_3d.krk`
- `examples/heat_conduction.krk`

La documentation devrait inverser son approche:

- D'abord: ecrire, lire et modifier un `.krk`.
- Ensuite: lancer le `.krk`.
- Ensuite: comprendre ce que Kraken construit en Julia.
- Enfin: utiliser l'API Julia pour automatiser, etendre ou integrer.

Aujourd'hui, les pages `.krk` existent, mais elles ne structurent pas encore toute la pedagogie.

### 6. Couverture API Julia incomplete

`src/Kraken.jl` exporte environ 121 symboles. Un audit textuel simple montre qu'au moins une vingtaine de symboles publics ne sont pas clairement couverts dans les pages API ou `.krk`.

Exemples de symboles a verifier et documenter:

- `run_natural_convection_3d`
- `open_paraview`
- `has_variable`
- `is_time_dependent`
- `is_spatial`
- `DomainSetup`
- `PhysicsSetup`
- `GeometryRegion`
- `InitialSetup`
- `OutputSetup`
- `DiagnosticsSetup`
- `SanityIssue`
- `LBMParams`
- `lbm_params`
- `lbm_params_table`
- plusieurs fonctions thermiques 3D et boundary kernels 3D.

Action recommandee: generer une page "Julia API completeness" depuis les exports reels, puis imposer que chaque export soit soit documente, soit marque interne/non-public et retire de l'export.

### 7. Pas de version LLM compacte

Aucun fichier de type `llms.txt`, `llms-full.txt`, `agent-context.md` ou equivalent n'a ete trouve.

Contenu recommande pour une version agent:

- identite du projet et version cible,
- workflow minimal `.krk`,
- grammaire/schemas `.krk`,
- fonctions publiques Julia par categorie,
- exemples `.krk` compacts,
- backends supportes,
- limitations connues,
- claims de performance autorises avec provenance,
- liens vers pages longues,
- regles pour ne pas halluciner des capacites non presentes.

Format recommande:

- `docs/src/llms.md` pour une page humaine.
- `docs/src/llms.txt` ou `docs/src/agent-context.md` pour le contexte compresse.
- Une generation automatique depuis les sources doc si possible.

### 8. Pages helper exposees

`docs/src/_helpers/_test_helpers.md` est construit dans le site final alors qu'il ressemble a une page de support/generation. Il ne devrait pas etre indexable dans la documentation publique.

Action recommandee: exclure les helpers du build public ou les deplacer hors `docs/src`.

### 9. Liens GitHub mal formes

Le build contient des liens d'edition du type:

```text
https://https://github.com/lauguimel/Kraken.jl/edit/release/v0.1.0/...
```

Action recommandee: corriger la valeur `repo` ou la configuration VitePress/Documenter responsable de la double prefixation.

## Evaluation par objectif

| Objectif | Etat | Commentaire |
| --- | --- | --- |
| Propre | Partiel | Structure solide, mais liens casses, helpers exposes, assets absents. |
| Humaine | Partiel | Ton global lisible, mais trop de promesses techniques non hierarchisees. |
| Pedagogique | Partiel | Bon potentiel avec les exemples, mais le parcours `.krk` n'est pas encore le fil conducteur. |
| Accurate | Non | Plusieurs claims ne correspondent pas au code courant ou aux fichiers disponibles. |
| Resultats/references | Fragile | Chiffres a relier a des CSV, commits, hardware ids et commandes. |
| Basee sur `.krk` | Partiel | Les fichiers existent, mais la narration reste trop API/feature-first. |
| Complete API Julia | Non | Exports non couverts, API docs parfois en avance sur le code. |
| Toute doc Julia utile | Non | Pas de carte des docs Julia/officielles externes et packages dependants. |
| Version LLM | Non | Aucun artefact compact agent-ready trouve. |

## Architecture cible proposee

### Parcours humain

1. `Home`
   - Positionnement simple: Kraken execute des simulations LBM depuis des fichiers `.krk`.
   - Trois entrees: lancer un `.krk`, comprendre le format, utiliser Julia.

2. `Getting started`
   - Installer.
   - Lancer `examples/cavity.krk`.
   - Lire les sorties.
   - Modifier un parametre.
   - Passer CPU/GPU.

3. `.krk Reference`
   - Syntaxe.
   - Blocs obligatoires.
   - Blocs optionnels.
   - Types et valeurs.
   - Expressions.
   - Diagnostics.
   - Outputs.
   - Erreurs courantes.

4. `Examples`
   - Une page par `.krk`.
   - Chaque page: but physique, fichier complet, commande, sorties, reference analytique ou numerique.

5. `Julia API`
   - Reference generee depuis les exports.
   - Un statut par symbole: public stable, experimental, internal.

6. `Theory`
   - Seulement les modeles actuellement supportes.
   - Les modeles futurs vont dans une roadmap separee, pas dans la reference utilisateur.

7. `Benchmarks`
   - Chiffres publies uniquement si reproductibles depuis un fichier de resultats.
   - H100 + CPU baseline comme histoire principale.

8. `LLM / agent context`
   - Version compacte.
   - Version complete.
   - Regles anti-hallucination et limites connues.

### Parcours agent

Un agent doit pouvoir repondre a:

- Comment lancer une simulation depuis `.krk`?
- Quels champs `.krk` sont valides?
- Quelle fonction Julia correspond a ce workflow?
- Quels exports publics existent?
- Quelles capacites ne doivent pas etre affirmees?
- Quels benchmarks peuvent etre cites?
- Ou trouver les references longues?

## Plan de correction recommande

### P0 - Bloquants publication

- Corriger ou retirer les claims absents de `capabilities.md`, `concepts_index.md`, `index.md`, `api/*`.
- Supprimer ou archiver `docs/COVERAGE.md` s'il ne reflete plus cette branche.
- Corriger les 17 liens locaux casses.
- Publier correctement `icon.png` et les GIFs dans le build VitePress final.
- Corriger les liens GitHub `https://https://...`.
- Remplacer `ignoreDeadLinks = true`, `warnonly = true`, `checkdocs = :none` par des controles stricts, au moins en CI.
- Retirer `apple_m2` de la narration publique ou le reclasser en resultat historique.
- Rattacher chaque chiffre benchmark a sa source.

### P1 - Recentrage produit/documentation

- Recrire `index.md` autour du workflow `.krk`.
- Recrire `getting_started.md` comme tutoriel `.krk` en premier.
- Ajouter une reference `.krk` complete et testable.
- Generer une page de couverture des exports Julia.
- Ajouter une page de liens vers les docs Julia et packages officiels utiles.
- Ajouter `llms.txt` ou `agent-context.md`.

### P2 - Qualite editoriale

- Uniformiser le ton: moins de claims marketing, plus de faits reproductibles.
- Ajouter des encadres "supported / experimental / planned".
- Standardiser les pages exemples: physique, `.krk`, commande, resultat, verification.
- Garder les GIFs, mais les relier a des exemples reproductibles.
- Ajouter les references scientifiques uniquement la ou elles soutiennent directement un modele documente.

## Decisions a prendre

1. Scope de la v0.1 publique
   - Option stricte recommandee: documenter uniquement ce que `src/` expose et ce que les exemples `.krk` couvrent.

2. Benchmarks
   - Option recommandee: H100 + CPU baseline. M3 Max reste un artefact de benchmark local, pas un argument central.

3. Roadmap
   - Les capacites futures restent visibles uniquement dans une page roadmap, clairement separee de la documentation d'utilisation.

4. Exports Julia
   - Soit tout export est public et documente.
   - Soit les exports non stabilises sont retires ou marques experimental.

## Verification effectuee

Commandes/audits locaux effectues:

- inspection de `docs/src`, `docs/build`, `docs/make.jl`, `docs/src/.vitepress/config.mts`;
- recherche des assets icon/GIF dans source, intermediaire et build final;
- audit simple des liens Markdown locaux;
- comparaison textuelle entre exports `src/Kraken.jl` et pages API/`.krk`;
- inspection des resultats `benchmarks/results` et de `benchmarks/hardware.toml`.

Cet audit n'est pas une validation scientifique des equations ou des resultats numeriques. Il identifie les ecarts structurels, editoriaux et de provenance qui bloquent une documentation fiable.
