# Resume prompts — Kraken.jl

Copy-paste these into a fresh Claude Code session opened in
`~/Documents/Recherche/Kraken.jl/`. The SessionStart hook will fire and
surface the relevant Kraken skills automatically — these prompts state
the specific intent of the session.

---

## Status (2026-05-29)

KRK-SHIP-001 progress: **M1 ✅** (module audit — mandate §6 filled),
**M2 ✅** (`lbm` + `refinement-patches-dev` retired; `dev/v0.2-architecture`
kept), **M3 ✅** (units spec frozen — `docs/spec/units-v1.md` on
`dev/units-module`). Phase A + Phase-B spec done.

**Next dispatchable: M4** — implement `src/units/` Phase 1 (Newtonian + VE)
from the frozen spec. Code-heavy Julia → **Codex** mission (use prompt **E**
with `M<N> = M4`), runs on the `dev/units-module` worktree
(`/Users/guillaume/Documents/Recherche/Kraken.jl-units`). M4 required reading:
`docs/spec/units-v1.md` AND `dev/v0.2-architecture:src/runtime_specs.jl`
(prior art). **Canonical mandate lives on `slbm-paper`** — read it from
`/Users/guillaume/Documents/Recherche/Kraken.jl/.orchestrator/`, never from a
sibling worktree's stale `.orchestrator` copy (3 divergent copies exist).

Prompts A (M2), B (cherry-pick), C (M1 dispatch) are **DONE** and have been
retired. D (generic resume) and E (Codex relay) remain — both reusable.

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
