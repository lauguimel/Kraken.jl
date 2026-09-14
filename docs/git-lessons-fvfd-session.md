# Git pédagogie — leçons tirées de la session `src/fvfd/`

Date : 2026-05-19. Branche : `dev-viscoelastic`.

Ce document utilise un cas concret de cette session pour illustrer
quelques notions git qu'on ne croise pas dans le workflow quotidien
(commit / push / pull / amend / branch).

## 1. Contexte

Pendant la session, j'ai dû modifier `src/fvfd/operators_2d.jl` pour
deux missions distinctes (M26b puis M29b). Quand est venu le moment de
commiter, on a découvert que **tout le dossier `src/fvfd/` n'était pas
suivi par git** — alors que `src/Kraken.jl:64` l'incluait.

Concrètement : le code marchait sur ma machine, mais un `git clone`
frais aurait donné un repo qui ne compile pas. C'est un piège classique
qu'on appelle un fichier *orphelin* (orphan-but-used).

## 2. Les 4 états possibles d'un fichier sous git

Avant tout, le rappel des 4 états. C'est la base, mais c'est rarement
explicité de cette façon :

| État        | Description                                                 | Quand `git status` le montre              |
|-------------|-------------------------------------------------------------|-------------------------------------------|
| **untracked** | Le fichier existe sur disque mais git n'en a jamais entendu parler. | `??` dans `git status --short`            |
| **tracked, unmodified** | Suivi par git, identique à la dernière version commitée. | Invisible dans `git status`               |
| **tracked, modified**   | Suivi, mais différent de la dernière version commitée. | ` M` (espace + M) dans `git status --short` |
| **staged**   | Modifications mises dans l'index, prêtes à être commitées. | `M ` (M + espace) dans `git status --short` |

Le piège qu'on a rencontré : un fichier `untracked` qui n'est **pas non
plus ignoré** par `.gitignore`. Il est juste... oublié. Personne ne l'a
jamais `git add`-é, donc il reste invisible aux opérations git, mais
visible sur le disque.

## 3. Trois commandes pour inspecter l'état d'un dossier suspect

Ces commandes sont sous-utilisées par les beginners mais essentielles
pour diagnostiquer des situations comme la nôtre.

### `git ls-files <path>` — liste les fichiers que git suit pour de vrai

```bash
$ git ls-files src/fvfd/
(rien — donc git ne suit aucun fichier dans src/fvfd/)
```

Si ça renvoie vide alors que le dossier existe sur disque, c'est le
signal que tous les fichiers du dossier sont `untracked`.

### `git check-ignore -v <file>` — est-ce qu'un `.gitignore` masque le fichier ?

```bash
$ git check-ignore -v src/fvfd/operators_2d.jl
(rien — donc le fichier n'est PAS ignoré)
```

Si ça renvoie quelque chose comme `.gitignore:5:*.tmp src/fvfd/operators_2d.jl`,
c'est qu'une règle d'ignorance s'applique. Si ça renvoie vide, le
fichier est juste oublié — pas ignoré.

### `git log --diff-filter=AM -- <path>` — historique des ajouts/modifs

```bash
$ git log --diff-filter=AM --oneline -- 'src/fvfd/'
(rien — donc le dossier n'a JAMAIS été ajouté ni modifié dans aucun commit)
```

Si l'historique est vide, le dossier n'a jamais existé du point de vue de
git. C'était notre cas — `src/fvfd/` était orphelin depuis sa création.

## 4. Pourquoi c'est dangereux

Un fichier orphelin a trois inconvénients silencieux :

1. **Un clone frais casse**. Si un collègue clone le repo, il n'a pas
   `src/fvfd/`, donc `using Kraken` plante avec un `LoadError`.
2. **CI cassée**. Les pipelines CI partent toujours d'un clone propre.
   Tu ne le vois pas car tu travailles sur ta copie locale qui a les
   fichiers.
3. **Branches confusantes**. Si tu changes de branche avec `git
   checkout`, les fichiers untracked restent en place. Mais si tu fais
   un `git clean -fd`, ils disparaissent — sans avertissement.

## 5. Comment c'est arrivé chez nous (probablement)

Je n'ai pas l'historique précis, mais le pattern le plus fréquent :

1. Quelqu'un crée `src/fvfd/` en local pendant une session.
2. Modifie d'autres fichiers, fait `git add` ciblé (par exemple
   `git add src/drivers/`).
3. `git commit` — réussit (les fichiers stagés sont commités), mais
   `src/fvfd/` n'a jamais été stagé donc reste `untracked`.
4. La session se termine. Le code marche localement.
5. Sessions suivantes : tout le monde travaille avec le code local, le
   problème reste invisible.

**Leçon** : préférez `git status` à `git diff` après un commit pour
vérifier qu'il ne reste rien d'untracked d'important.

## 6. Comment on a réparé pendant cette session

Le commit `42d2177a` (M29b) a fait deux choses en même temps :

1. Ajouté les **4 fichiers `src/fvfd/`** (orphelins de longue date) :
   - `FVFD.jl`
   - `lowering_2d.jl`
   - `operators_2d.jl`
   - `specs.jl`
2. Ajouté **la modification M29b** (MUSCL-superbee scheme) qui se
   trouve... dans `src/fvfd/operators_2d.jl` justement.

Concrètement la commande était :

```bash
git add src/fvfd/ src/drivers/... test/... bench/... .orchestrator/...
git commit -m "feat(viscoelastic): M29b MUSCL-superbee + initial src/fvfd tracking"
git push
```

### Pourquoi bundler les deux dans un seul commit ?

- **Solution propre** : 2 commits (un "chore: track src/fvfd directory"
  puis un "feat: M29b MUSCL-superbee"). Mais le diff du commit 2 serait
  incomplet (le M29b patch lui-même est dans `src/fvfd/operators_2d.jl`
  qui est nouveau-tracké, donc apparait dans le commit 1, pas le 2).
- **Solution pragmatique** : 1 commit qui bundle tout, avec un message
  qui explique les deux choses. On a choisi celle-là.

Note dans le message de commit qui décrit les deux aspects :

```
feat(viscoelastic): M29b MUSCL-superbee on Psi-advection + initial src/fvfd tracking

PARTIAL fix to the M29-localised Kraken-vs-rheoTool gap [...]

Bundled: initial tracking of src/fvfd/ directory (4 files, ~2000 LOC:
FVFD.jl, lowering_2d.jl, operators_2d.jl, specs.jl). The directory was
referenced by src/Kraken.jl:64 but had never been committed - pre-
existing condition. Local builds worked; fresh clones would have failed.
```

C'est honnête : le lecteur futur du commit voit clairement les deux
choses qu'il contient.

## 7. Style de commit qu'on a utilisé : conventional commits

Tous mes commits de la session suivent le format
[Conventional Commits](https://www.conventionalcommits.org/) :

```
<type>(<scope>): <résumé court en anglais>

<corps en plusieurs paragraphes optionnel>
```

Les types qu'on a utilisés cette session :
- `feat:` — nouvelle fonctionnalité (M28 synthesis, M29 comparison, M29b scheme)
- `fix:` — correction de bug (world-age trap dans `detect_backend`)

Le scope `(viscoelastic)` indique qu'on travaille sur la branche
viscoélastique du projet (pas le solver Newtonien principal).

Exemples des 5 commits de la session (en ordre chronologique) :

```bash
$ git log --oneline -5
42d2177a feat(viscoelastic): M29b MUSCL-superbee on Psi-advection + initial src/fvfd tracking
94f4b82d feat(viscoelastic): M29 tau-field comparison locates gap to Rusanov upwind on log-conf advection
2945b198 feat(viscoelastic): cylinder Cd M28 cluster synthesis + Liu/rheoTool cross-validation
d708da57 feat(viscoelastic): cylinder Cd Phase 0+0b verdicts + M26 bug localisation
e602726f fix(viscoelastic): world-age trap in CUDA detection causes silent CPU fallback
```

Le 1er mot du résumé est toujours un verbe à l'impératif anglais
(`Add`, `Fix`, `Update`, etc.) — pas une phrase narrative.

## 8. La règle "git add ciblé"

Tu as peut-être l'habitude de `git add -A` ou `git add .`. Pendant cette
session, je n'ai JAMAIS utilisé ces formes. À la place :

```bash
git add src/fvfd/operators_2d.jl src/drivers/viscoelastic_logfv_2d.jl test/... bench/...
```

Pourquoi ? Trois raisons :
1. **Sécurité** : éviter de commiter accidentellement des secrets
   (`.env`, `credentials.json`) ou des gros binaires (`output/*.vtr`).
2. **Lisibilité** : chaque commit ne contient que ce qui appartient à
   son sujet. Pas de diff parasite "j'ai aussi modifié ce fichier
   par hasard".
3. **Précision** : tu vois exactement ce que tu commit. Si tu te trompes,
   `git diff --cached --stat` te le dit avant `git commit`.

Le `working tree` avait par exemple `M output/cylinder_*.vtr` (gros
fichiers VTK régénérés par mes runs locaux) que je n'ai jamais commités
— ils sont régénérables et le repo n'en a pas besoin.

## 9. Trois étapes pré-commit que je fais systématiquement

```bash
# 1. Voir ce qui est sur le point d'être commité
$ git diff --cached --stat

# 2. Voir ce qui reste dans le working tree (pour vérifier qu'on n'oublie rien)
$ git status --short

# 3. Voir l'historique récent (pour respecter le style de message)
$ git log -3 --oneline
```

Si l'étape 1 montre un fichier suspect (par exemple `M tmp/big_output.bin`),
je l'enlève avec `git reset HEAD tmp/big_output.bin` avant de commiter.

## 10. La règle "push avec confirmation"

Toute la session, j'ai utilisé `git push` SEULEMENT après confirmation
explicite. Pourquoi ?

- `git push` peut publier des secrets si l'étape 8 a foiré.
- `git push` peut casser la branche pour les collègues s'il y a un
  `force-push`.
- `git push` rend les commits visibles publiquement (l'historique est
  permanent sur le remote, même si on `git reset` localement).

Si jamais tu pousses un commit que tu regrettes, la procédure de
récupération est :

```bash
# (a) Si personne d'autre n'a encore pull
git reset --hard HEAD~1     # défait le commit local
git push --force-with-lease # publie le rollback (DANGEREUX)

# (b) Si quelqu'un d'autre a peut-être pull
git revert <bad-commit-sha>  # crée un commit qui annule le bad
git push                      # publie l'annulation (SAFE)
```

`git revert` est toujours plus sûr que `git reset --hard` + `push --force`
sur une branche partagée.

## 11. 5 commandes à retenir pour le pattern "fichier orphelin"

| Commande                                            | Question qu'elle répond                          |
|-----------------------------------------------------|--------------------------------------------------|
| `git ls-files <path>`                               | Quels fichiers de `<path>` sont vraiment suivis ? |
| `git check-ignore -v <file>`                        | Est-ce que `<file>` est filtré par un `.gitignore` ? |
| `git log --diff-filter=AM -- <path>`                | Quel est l'historique des ajouts/modifs sur `<path>` ? |
| `git status --short`                                | Quels fichiers sont untracked / modified / staged ? |
| `find src/ -name '*.jl' \| xargs git ls-files --error-unmatch` | Y a-t-il des `.jl` qui ne sont PAS suivis ? |

La 5ème est utile en audit : si la commande affiche une erreur pour un
fichier, c'est qu'il est orphelin. À faire de temps en temps sur un repo
en activité pour détecter ce genre de piège tôt.

## 12. Récap de la session côté git

5 commits créés et poussés sur `dev-viscoelastic` :

```
42d2177a  feat: M29b MUSCL-superbee + initial src/fvfd tracking
94f4b82d  feat: M29 tau-field comparison locates gap to Rusanov upwind
2945b198  feat: M28 cluster synthesis + Liu/rheoTool cross-validation
d708da57  feat: Phase 0+0b verdicts + M26 bug localisation
e602726f  fix: world-age trap in CUDA detection
```

Pattern récurrent :
1. Diagnostic d'un problème (M26 bug, M28 gap, etc.)
2. Tests, mesures, verdicts → fichiers `*_VERDICT.md`
3. Mise à jour mémoire (`.orchestrator/memory/*.md`) + mandate
4. `git add` ciblé sur les fichiers du sujet
5. `git commit -m "<type>(<scope>): <résumé>` avec corps de message
6. `git push` après confirmation

Et le commit M29b a accidentellement résolu une dette pré-existante
(`src/fvfd/` orphelin) en bonus.

---

**Pour aller plus loin** :
- [Pro Git book (gratuit, en français disponible)](https://git-scm.com/book/fr/v2)
- `man gitignore` (très clair sur les règles d'ignorance)
- `git help status` (l'option `--ignored` peut être utile pour voir aussi
  les fichiers ignorés)
