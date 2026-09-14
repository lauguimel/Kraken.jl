# Department memory — Kraken.jl

## 2026-05-28 — Pre-flight for ship-plan / strategic missions

Before drafting any strategic plan on this repo, run BOTH:

1. `bash ~/.claude/scripts/git-audit/branch-audit.sh --repo=/Users/guillaume/Documents/Recherche/Kraken.jl --with-github` — branch + worktree state in one shot
2. A light `ls src/` across 2-3 candidate worktrees (e.g. `slbm-paper`, `dev-viscoelastic`, `dev/fvfd-core`) to ground the modular architecture audit.

Do NOT deep-read source files in a strategic mission — that's the Engineer's job downstream. Stay at the directory/module level.

## 2026-05-28 — Pre-existing engineer briefs may already exist in worktrees

`Kraken.jl-viscoelastic/.engineer_brief_M62b_IMPL.md` was already drafted before the orchestrator pattern formalisation. It IS the LU/real-units design. When the Boss surfaces an apparent gap, check first whether an existing brief covers it in any worktree.

Pattern: `find /Users/guillaume/Documents/Recherche/Kraken*/. -maxdepth 2 -name '.engineer_brief*.md'` before assuming a brief must be drafted from scratch.

## 2026-05-28 — Scope cuts are ADRs in mandate §4 — respect them

Mandate §4 lists scope cuts (axisym, backend, multiphase, AMR, STL deferred for ship-1). A brief that touches any of these without an ADR override is malformed. Refuse and ask the Boss to either widen the mandate or rescope the brief.
