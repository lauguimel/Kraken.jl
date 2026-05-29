# Resume prompts — Kraken.jl

Copy-paste these into a fresh Claude Code session opened in
`~/Documents/Recherche/Kraken.jl/`. The SessionStart hook will fire and
surface the relevant Kraken skills automatically — these prompts state
the specific intent of the session.

---

## C. Dispatch M1 du ship-plan (vraie première mission de code)

```
Nous sommes dans Kraken.jl. KRK-SHIP-001 est défini dans
.orchestrator/ship-plan.md (5 sections + §6 overlay Boss).

Tâche :
1. Charge les skills `orchestrator` et `kraken-architect`.
2. Lis .orchestrator/mandate.md (mandat + ADRs) et
   .orchestrator/memory/boss.md (état actuel).
3. Lis le §3 Mission graph du ship-plan et identifie M1
   (la première mission dispatchable sans dépendance).
4. Cross-check avec §6 Locked decisions (overrident toute divergence
   §3). Si M1 est M2 (retirements), bascule sur le prompt A à la place.
5. Pour la mission identifiée, draft un Department brief en remplissant
   ~/.claude/skills/orchestrator/department_brief_template.md. Inputs,
   allowed edit zones, forbidden actions, exit criterion concret,
   runner recommandé, format de rapport.
6. Avant de spawner le Department, montre-moi le brief complet.
   Je valide ou corrige.
7. Une fois validé : spawn le Department via Agent (background si la
   mission est longue, foreground si courte).

NE fais AUCUN edit dans src/ ou test/. Toute exécution passe par
Department → Engineer.
```

---

## D. Generic resume ("où on en est, qu'est-ce qu'on fait")

```
Nous sommes dans Kraken.jl. Reprends le contexte de KRK-SHIP-001.

Tâche (read-only audit) :
1. Lis .orchestrator/mandate.md (focus §3 Constraints, §4 ADRs récents,
   §5 Branch map, §6 Modular architecture target, §8 Open questions).
2. Lis .orchestrator/memory/boss.md.
3. Lis .orchestrator/ship-plan.md §3 (Mission graph) et §6 (overlay).
4. Run l'audit branches courant :
   bash ~/.claude/scripts/git-audit/branch-audit.sh --repo=. --with-github
5. Compare l'état actuel (audit) avec ce que le mandate §5 prédit :
   y a-t-il du drift ? Branches nouvelles non documentées ? Branches
   supprimées entre temps ?

Output attendu : un récap ≤250 mots structuré comme :
- État de KRK-SHIP-001 (combien de missions M1..M15 done, combien restent)
- Drift mandate ↔ git (lignes à corriger dans mandate)
- Prochaine mission dispatchable + dépendances
- Open questions §8 qui bloquent

NE lance aucune mission, ne touche à rien. Juste l'audit + recap.
Charge `kraken-architect` et `git-audit`.
```

---

## E. Si Codex doit prendre le relais (Julia code-heavy)

```
Nous sommes dans Kraken.jl. La mission M<N> du ship-plan
(.orchestrator/ship-plan.md §3) est code-heavy Julia (kernels FVFD,
collision, streaming, viscoelastic). Elle doit être exécutée par Codex,
pas par un subagent Claude.

Tâche :
1. Charge `kraken-codex-pilot` (Claude-side) + `orchestrator`.
2. Read .orchestrator/mandate.md + memory/boss.md + memory/engineer.md
   + le slice de ship-plan.md pertinent pour M<N>.
3. Draft le Engineer brief Codex via
   ~/.claude/skills/orchestrator/engineer_brief_template.md.
   Inclut les skills Codex à charger :
   `kraken-branch-governor`, `kraken-fvfd-operator-library` (si FVFD),
   `kraken-resource-integrator` (si fixtures/benchmarks),
   `kraken-port-fidelity` (si port C++→Julia).
4. Montre-moi le brief avant de lancer Codex.
5. Une fois validé : lance via
   `bash ~/.claude/skills/orchestrator/run-engineer.sh <repo> M<N>
        <brief_path>`.
6. À la fin, review le diff Codex, re-vérifie le exit criterion, et
   reporte-moi GREEN/YELLOW/RED. Pas de commit sans mon OK.

NE lance pas le Department-style spawn. C'est Codex direct (Layer 2).
```

---

## Maintenance — quand mettre à jour ces prompts

Ce fichier devrait être édité quand :
- Une mission est complète → retire le prompt qui la dispatche.
- Le ship-plan évolue (nouvelle mission, scope cut) → ajoute ou modifie
  le prompt correspondant.
- Un workflow récurrent émerge entre 2-3 sessions → propose-le comme
  skill via le mécanisme `orchestrator §"Proposing new skills"`.

Le fichier vit avec le ship-plan et le mandate. Quand KRK-SHIP-001 ferme
et que KRK-SHIP-002 ouvre, archive ces prompts dans une section
`## Archive — KRK-SHIP-001` et écris les nouveaux pour ship-2.
