# Engineer memory — Kraken.jl viscoelastic cavity spatial debug

Codebase do/don't and stack-specific traps. Read this before writing any
code. Initialised 2026-05-15.

## 2026-05-15 — GPU-only for non-trivial runs

Anything beyond a 50-line smoke test must run on GPU. CPU runs on the
log-FV cavity are pointless: substep cadence and BSD stencil are tuned
for GPU memory layout. Local: `KRAKEN_BACKEND=metal` (Float32, macOS
M-series). HPC: `KRAKEN_BACKEND=cuda` (Float64, Aqua A100/H100).

Tests under `test/` are allowed on CPU (<30 s budget).

**Why**: CPU log-FV cavity at N=64 takes hours and reveals nothing the
GPU version doesn't.

## 2026-05-15 — `tmp/` is .gitignored — do not commit results

Any CSV / .jls / fields snapshot from Aqua belongs under `tmp/` or
`results/`. Do not stage these. The verdict markdown that summarises
them goes under `bench/viscoelastic_logfv/` and IS committed.

**Why**: prior sessions cluttered the worktree with multi-MB binaries.

## 2026-05-15 — Confidentiality rules

- Never write `Claude`, `AI`, `LLM`, `Anthropic`, `assistant`,
  `Co-Authored-By` in source, comments, commits, scripts, or markdown.
- Conventional-commits message style (`feat:`, `fix:`, `refactor:`,
  `docs:`, `chore:`). No emojis.

**Why**: this repo will be made public; history will be audited.

## 2026-05-15 — Self-tests must isolate to `mktempdir()`

Bench-side self-tests (`--self-test` modes) must build all fixtures
inside `mktempdir() do dir ... end`. Never depend on `tmp/cavity_aqua_n64/`
or any other path under `tmp/` — that directory is `.gitignored` and
absent on a fresh clone or CI.

**Why**: ensures the self-test runs from a clean checkout; cleanup
guaranteed even on assertion failure. Found during M1.
