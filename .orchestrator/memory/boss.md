# Boss memory — Kraken.jl

## 2026-05-28 — Bootstrap of orchestrator pattern on Kraken.jl

`.orchestrator/` created with mandate, ship-plan, and memory layers. First strategic mission (KRK-SHIP-001) drafted by Department.

## 2026-05-28 — KRK-SHIP-001 — ship plan adopted (Boussinesq variant)

The ship plan in `.orchestrator/ship-plan.md` lists 15 missions M1–M15 covering: branch cleanup, `src/units/` (M62b port), viscoelastic finalisation, thermal Boussinesq, RheoTool benchmarks per axis, tri-track docs, integration into `dev/v0.2-multiphysics`.

Estimated minimum: **9–10 sessions**. Comfortable: 14–16.

**Implication for future missions**: every Department brief on this project must point at `.orchestrator/ship-plan.md` as the mission catalogue, and at `mandate.md §4 ADRs` for scope decisions already taken. Do not re-debate frozen scope (thermal split, units name, merge target).

## 2026-05-28 — `dev-viscoelastic` is the de-facto multiphysics worktree

Department audit revealed that `dev-viscoelastic` (236 commits ahead, 863 files diff) carries the thermal + viscoelastic integration — NOT `slbm-paper` as the initial mandate §5 implied. Mandate §5 will be updated as this branch evolves.

**Implication**: any multiphysics-touching mission targets `dev-viscoelastic` as source, merges into `dev/v0.2-multiphysics` as target. `slbm-paper` is paper-only after this point.

## 2026-05-28 — Thermal scope split between ship-1 (Boussinesq) and ship-2 (full)

User initially asked for full thermal in ship-1, Department warned scope creep, Boss escalated, user accepted split. Ship-2 plan (KRK-SHIP-002) to be drafted only after ship-1 reaches MVP — do not pre-plan.

**Implication**: when ship-1 closes and the user asks "what's next on thermal?", load this entry — the answer is a new ship plan for full thermal + dissipation, matching `rheoHeatFoam/channel/PTTLog` (Br=-25, WLF/Arrhenius).

## 2026-05-28 — Scope cuts ADR'd for ship-1

Axisymmetric, backend module factoring, multiphase, AMR, STL — all deferred. Do not let any mission scope them in. If a brief touches any of these, refuse and re-scope.
