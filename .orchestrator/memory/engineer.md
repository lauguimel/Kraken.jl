# Engineer memory — Kraken.jl

## 2026-05-28 — `src/units/` is the locked name for the LU↔real conversion module

The M62b implementation brief originally called it `src/sim_planner/`. The Boss locked `src/units/` for consistency with `mandate.md §6`. If you receive a brief that still references `sim_planner`, rename `s/sim_planner/units/g` everywhere — directory, exports, tests, `.krk` parser hooks, doc files.

## 2026-05-28 — `.krk` is the user-facing canonical interface

Anything a Julia user can do must also be doable from a `.krk` file. If a feature needs a Julia-only API for now, mark it with a TODO and an `.orchestrator/mandate.md §8` open question — don't ship a Julia-only feature silently.

## 2026-05-28 — Tri-track docs are mandatory for every public module

For each new public module:
- `docs/src/users/<feature>.md` — human-facing, `.krk` examples + benchmark plots.
- `docs/src/api/<module>.md` — Julia API (auto-generated from docstrings).
- `docs/agent/<module>-implication.md` — LLM-friendly module map (what depends on what, where the kernel fires, what tests cover what).

If only one of the three exists at PR time, the brief is not complete.

## 2026-05-28 — Match RheoTool benchmarks EXACTLY (numerical, not qualitative)

Per mandate non-negotiable: every multiphysics module must reproduce a RheoTool reference case numerically (1% tolerance unless otherwise specified per mission). Qualitative ≈ fail.

Use the `sim-rheotool` skill (Claude side) to set up the RheoTool case to compare against. The benchmark mission consumes BOTH solvers' outputs and produces a comparison artefact (CSV + plot in `docs/src/users/benchmarks/`).
