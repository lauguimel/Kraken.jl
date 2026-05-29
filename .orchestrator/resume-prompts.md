# Resume prompts — Kraken.jl

Copy-paste these into a fresh Claude Code session opened in
`~/Documents/Recherche/Kraken.jl/`. The SessionStart hook will fire and
surface the relevant Kraken skills automatically — these prompts state
the specific intent of the session.

---

## Status (2026-05-29, updated end-of-session)

KRK-SHIP-001 progress: **M1 ✅** (module audit — mandate §6 filled),
**M2 ✅** (`lbm` + `refinement-patches-dev` retired; `dev/v0.2-architecture`
kept), **M3 ✅** (units spec frozen, `docs/spec/units-v1.md`),
**M4 ✅** (units Phase 1 Newt+VE — 165 tests), **M5 ✅** (units Thermal-Boussinesq
Phase 2 — 277 tests, zero-edit contract §7 PROVEN), **M6 ✅** (Cartesian cavity
benchmark vs Ghia 1982 + icoFoam — **Phase C OPENED**; icoFoam <0.5%, Kraken BGK
2-3% rel; 3D Zou-He top-BC bug fixed `9674e1c4c`→main `a5ce7c6f0`, deliverables
`a524f7ded`). **Phase A + B COMPLETE; Phase C started.** All on `dev/units-module`
(`/Users/guillaume/Documents/Recherche/Kraken.jl-units`), local commits, no push.

**Parallel track KRK-GEO** (STL/complex geometry, separate from ship-1): M-GEO-1+2
GREEN on `feat/geometry-stl` (off `dev/v0.3-campaign`) — STL via `.krk` + mesh-field
regression fix. See boss.md.

**Next dispatchable**:
- **M7** RheoTool thermal Rayleigh-Bénard (buoyantBoussinesqPimpleFoam, Ra 1e3-1e5) — dep M5 ✓.
  Reuses the M6 icoFoam-Docker recipe (**`cd /case` inside `bash -c`** — the 2412
  image entrypoint forces CWD to /root) + `bench/scratch/run_cavity_bench.jl`+`plot_cavity_bench.jl`.
- **M8** RheoTool viscoelastic cylinder (rheoFoam Oldroyd-B, Wi 0.1/0.5/1.0) — dep M4 ✓; Aqua.
- **M9–M13** tri-track docs; **driver-integration** (deferred Cd/Nu repro, merge debt).
**M6 RESOLVED — cavity is <1%**: the 2-3% was a **half-cell coordinate bug** (Zou-He
lid on-node vs hand-coded `(j-0.5)/N`), NOT BGK/MRT/resolution. Fixed via
`axis_node_coords(N;lo,hi)` (`src/axis_coords.jl`, `78ed276c4`→main `9b9a97aa8`,
corrected M6 `04188e067`): Kraken-vs-Ghia rel-L2 **0.47/0.41/1.05%**. Couette control =
machine-zero ⇒ solver exact; TRT confirmed irrelevant (≡BGK), port reverted. **PENDING:
cherry-pick the helper to lineage branches** (slbm-paper/dev-visco/v0.3/amr/docs —
deferred, dirty/active worktrees; batch w/ KRK-GEO's `fc9a4d7eb` or via M14). Convention:
[[feedback_wall_aware_coords]]. Residual open item: (b) BGK marginal-ω instability (16³, ω≈1.89).

**For ANY units follow-up**: the module lives on `dev/units-module`; 277 tests
green via `julia --project=. -e 'using Pkg; Pkg.test(; test_args=["units"])'`.
**Canonical mandate lives on `slbm-paper`** — read it from
`/Users/guillaume/Documents/Recherche/Kraken.jl/.orchestrator/`, never from a
sibling worktree's stale `.orchestrator` copy (3 divergent copies — consolidation
is a deferred mission).

Prompts A (M2), B (cherry-pick), C (M1) are **DONE** and retired. D (generic
resume), E (Codex relay, used for M4+M5) remain — both reusable. M6/M7/M8 are
`claude-subagent` Validate missions (NOT Codex) — use prompt D to re-orient, then
draft a Department brief per the orchestrator skill with `sim-rheotool` loaded.

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
