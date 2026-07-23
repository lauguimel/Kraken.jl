---
module: llm-implication-map
path: docs/spec/
owner_concern: doc-format
status: frozen
last_verified: 2026-05-31
depends_on: []
---

# LLM module-implication map — format specification v1

**Spec ID**: KRK-LLM-MAP-001
**Status**: FROZEN (M9, 2026-05-31)
**Implements**: mandate §3 "doc-as-code" non-negotiable; ship-plan §2.2 Track C.
**Linter**: `scripts/lint-implication-map.sh` (pure bash + grep, no external deps).
**Reference example**: `docs/agent/units-implication.md`.

---

## 0. Purpose

Every public Kraken module ships THREE doc surfaces (ship-plan §2.2): Track A
(human/GitHub-Pages narrative), Track B (Documenter.jl API docstrings), Track C
(this — the LLM module-implication map). The implication map is the
**machine-navigable** surface: a future LLM session (or a human in a hurry) reads
one file per module and learns what the module exposes, what state it touches,
what will explode, and where to look first when it does — *without re-deriving it
from source*.

The format below is the canonical contract. It is deliberately minimal: a YAML
frontmatter block (machine-parsable metadata) + a markdown body with **exactly
six mandatory `##` section headers**. The linter (`scripts/lint-implication-map.sh`)
enforces presence and non-emptiness of all six plus the frontmatter.

> **Ship rule (ADR, mandate §4)**: A public module is **shipped only when its
> implication map exists in `docs/agent/<module>-implication.md` AND lints
> clean** (`scripts/lint-implication-map.sh <file>` exits 0). No map → module
> not shipped. No exceptions for "small" modules.

---

## 1. File location and naming

- One file per module: `docs/agent/<module>-implication.md`.
- `<module>` matches the §2.2 tri-track checklist slug: `io-krk`, `geometry`,
  `lbm`, `physics-newtonian`, `physics-viscoelastic`, `physics-thermal`,
  `units`, `bc`, `backend`.
- The file is plain UTF-8 markdown with a leading YAML frontmatter block.

---

## 2. YAML frontmatter

The file MUST begin (byte 0) with a frontmatter block delimited by a line
containing only `---`, then `key: value` lines, then a closing `---` line.

### 2.1 JSON-Schema for the frontmatter

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://kraken.jl/spec/llm-implication-map-v1/frontmatter",
  "title": "LLM module-implication map frontmatter",
  "type": "object",
  "required": ["module", "path", "owner_concern", "status", "last_verified"],
  "additionalProperties": true,
  "properties": {
    "module": {
      "type": "string",
      "description": "Module slug, must match the docs/agent/<module>-implication.md filename stem.",
      "pattern": "^[a-z0-9]+(-[a-z0-9]+)*$"
    },
    "path": {
      "type": "string",
      "description": "Repo-relative path of the module's source directory or facade file, e.g. src/units/."
    },
    "owner_concern": {
      "type": "string",
      "description": "The single concern this module owns (one file = one concern, mandate §3). E.g. 'lu-nondim-conversion', 'krk-parsing', 'wall-bc'.",
      "enum": [
        "lu-nondim-conversion",
        "krk-parsing",
        "geometry",
        "lbm-operator",
        "constitutive",
        "boundary-condition",
        "backend-dispatch",
        "io-output",
        "doc-format"
      ]
    },
    "status": {
      "type": "string",
      "description": "Maturity of the documented module.",
      "enum": ["draft", "implemented", "frozen", "deprecated"]
    },
    "last_verified": {
      "type": "string",
      "description": "ISO date (YYYY-MM-DD) the map was last checked against the source it describes.",
      "format": "date"
    },
    "depends_on": {
      "type": "array",
      "description": "Module slugs this module reads from at compile/run time (the read-only dependency graph). Mirrors the '## Reads from' section.",
      "items": {
        "type": "string",
        "pattern": "^[a-z0-9]+(-[a-z0-9]+)*$"
      },
      "default": []
    }
  }
}
```

### 2.2 Authoring rules for the frontmatter

- `module` MUST equal the filename stem (`docs/agent/units-implication.md` →
  `units`). The linter does not check this equality (it is policy, not lint),
  but reviewers must.
- `owner_concern` is the ONE concern. If you cannot pick one, the underlying
  source file likely violates "one file = one concern" (mandate §3) — flag a
  SPLIT, do not invent a compound concern.
- `depends_on` is the machine mirror of `## Reads from`. Keep them in sync; the
  array is what a graph tool consumes, the section is what a human reads.
- `last_verified` must be bumped whenever the map is re-checked against source.

---

## 3. The six mandatory sections

Each section is a level-2 markdown header (`## <exact title>`) with a non-empty
body. The titles are **exact and ordered** (the linter matches them literally):

| # | Header | Semantics (2–3 lines) |
|---|--------|------------------------|
| 1 | `## Public surface` | Every public symbol the module exposes (functions, types, the entry points other code/`.krk` calls) with a 1-line description each. This is the API contract an LLM may call. List exported names AND the `Module.name`-accessible names that are de-facto public. |
| 2 | `## Reads from` | Which OTHER modules' state/types this module consumes read-only (its dependencies). Name the sibling module + the concept/type it reads. If it reads nothing, say "Nothing — pure / leaf module" explicitly. Mirrors `depends_on`. |
| 3 | `## Writes to` | What the module mutates or produces: return values, mutated arguments, global registries, files written. Be explicit about "mutates nothing" when the module is pure/compile-time. This is the blast-radius surface. |
| 4 | `## Backend constraints` | GPU-safety: does it allocate inside hot kernels? does it run only at compile time? is it backend-agnostic / KernelAbstractions-clean? Any per-step cost? Any Float32 caveat? |
| 5 | `## Failure modes` | Known antipatterns, past bugs, footguns, and the rabbit-holes this module exists to prevent. CITE the mission/postmortem IDs (e.g. M48, M59–M61) so the reader can pull the receipt. This is the highest-value section for an LLM. |
| 6 | `## Touch order` | For a NEW bug suspected in this module, the ordered list of files to inspect first → last, with a one-line "look here for X" per file. Saves the "edited dead code by inference" anti-pattern. |

### 3.1 Authoring rules for the body

- The six headers MUST all be present, spelled exactly, each with a non-empty
  body (the linter rejects an empty section). Order should follow the table but
  the linter does not enforce order — reviewers should.
- Cite **symbols, not line numbers** (mandate §3: line numbers rot). E.g.
  "`compile` in `Units.jl`", never "line 188".
- `## Failure modes` MUST cite at least one concrete receipt (mission ID, ADR
  date, or test name) when the module has any history. A module with genuinely
  no known footguns states "No known failure modes as of <date>".
- `## Touch order` lists actual filenames in the module, ordered by
  investigate-first probability for that module's typical bug class.
- Extra `##` sections beyond the six are allowed (e.g. `## Worked example`,
  `## Cross-references`) and ignored by the linter.

---

## 4. Worked mini-example

A complete (tiny) map for a hypothetical `bc-periodic` module:

```markdown
---
module: bc-periodic
path: src/bc/periodic.jl
owner_concern: boundary-condition
status: implemented
last_verified: 2026-05-31
depends_on:
  - lbm
---

# bc-periodic — implication map

## Public surface
- `apply_periodic_x!(f)` — wraps the x-edge populations in place.
- `PeriodicX <: AbstractBC` — the BC tag dispatched by the runner.

## Reads from
- `lbm` — the D2Q9/D3Q19 lattice opposite-direction table `OPP`.

## Writes to
- Mutates the distribution array `f` in place at the x-edge halo cells.
  Produces no return value, touches no global registry.

## Backend constraints
- GPU-safe: KernelAbstractions kernel, no dynamic allocation, fixed stencil.
  Runs every timestep; cost is O(Ny) edge cells, negligible vs collide.

## Failure modes
- Corner double-application with halfway-BB walls (see multiblock corner BC bug,
  ebf0867): periodic-x at a wall corner can bounce a population twice.

## Touch order
1. `src/bc/periodic.jl` — the kernel itself; check the OPP indexing first.
2. `src/lbm/stream.jl` — verify the halo width matches the wrap stride.
```

---

## 5. Build verification (the linter)

`scripts/lint-implication-map.sh <path.md>` is pure bash + grep (no jq, no
python). It exits `0` iff ALL of:

1. The file starts with a YAML frontmatter block delimited by `---` (an opening
   `---` on the first line and a matching closing `---`).
2. All six mandatory section headers are present (exact `## <title>` match).
3. Each of the six section bodies is non-empty (at least one non-blank,
   non-comment content line before the next `##` header or EOF).

On failure it prints `FAIL: <reason>` naming the first missing/empty item and
exits non-zero. On success it prints `PASS: <file>`.

CI (M10) runs the linter over every `docs/agent/*-implication.md`; any non-zero
exit fails the docs job.

---

## 6. Scaling note (the ~9 checklist modules)

The format scales to the full ship-plan §2.2 checklist
(`io-krk`, `geometry`, `lbm`, `physics-newtonian`, `physics-viscoelastic`,
`physics-thermal`, `units`, `bc`, `backend`) without change: the six sections
are concern-agnostic, and `owner_concern` already enumerates the nine concerns.
Modules that share a concern family (the three `physics-*`) differ only in their
`## Public surface` / `## Failure modes` content — the skeleton is identical, so
maps can be produced from a template and lint-gated uniformly in one CI loop.

---

_End of LLM-implication-map format specification v1._
