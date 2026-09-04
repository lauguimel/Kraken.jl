# AGENTS.md — working rules for Kraken.jl

This file is read automatically by coding agents (Codex, Claude Code) at the
repository root. It defines how work is done here. Follow it literally.

Maintainer: Guillaume Maîtrejean (@lauguimel).

---

## 1. Ownership — read this before editing anything

| Directory | Owner | Contributors may |
|---|---|---|
| `src/` | maintainer only | **do not edit** |
| `ext/` | maintainer only | **do not edit** |
| `test/analytical/`, `test/reference/`, `test/platform/` | shared | add new files freely |
| `benchmarks/`, `benchmarks/results/` | shared | add new files freely |
| `docs/src/users/benchmarks/` | shared | add new pages |
| `docs/src/theory/`, `docs/agent/`, `docs/spec/` | maintainer only | propose in an issue |

**If you find a bug in `src/`: do not fix it.** Write a test that fails, place
it under `test/analytical/`, and open a GitHub issue describing the symptom and
pointing at the test. A failing test is the deliverable. A patch to `src/`
submitted by a contributor will be closed.

Reason: `src/` carries the platform contract, the automatic-differentiation
seam, and the GPU kernels. A local fix that works on one case routinely breaks
three others that are not visible from where the bug appeared.

---

## 2. Branches and submissions

- Never commit to `main`. Never commit to `dev/platform`. Both are protected.
- Integration branch is `dev/platform`. Everything targets it.
- One branch per unit of work, named after what it does:
  - validation case: `val/<case-id>` — e.g. `val/TH-001R`
  - bug report with a failing test: `bug/<short-name>`
- Open a pull request into `dev/platform`. The maintainer reviews and merges.
- Commit messages in English, conventional style: `test:`, `fix:`, `docs:`,
  `feat:`, `chore:`.
- Never mention an AI assistant, a model name, or a co-author trailer in a
  commit message, pull request, issue, or comment.

---

## 3. Running the test suite

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project test/runtests.jl
```

Environment gates (all default to off):

| Variable | Effect |
|---|---|
| `KRAKEN_TEST_HEAVY=true` | adds the long validations (Ghia cavities, multigrid up to 512²) |
| `KRAKEN_INCNS_ONLY=true` | runs only the incompressible Navier–Stokes and solver-services tier |
| `KRAKEN_AD_ONLY=true` | runs only the automatic-differentiation tier |

A pull request is not reviewable until `julia --project test/runtests.jl`
passes locally on the branch. Say so explicitly in the pull request body.

---

## 4. Adding a validation case

This is the main way to contribute. The shape is fixed:

1. **One file per case**, in `test/analytical/`, named after the case
   identifier: `test/analytical/<case-id>.jl` (e.g. `AXI-001.jl`). Keep your
   own identifier scheme — it becomes the project's.
2. **Register it** by adding an `include(...)` line in `test/runtests.jl`,
   inside the matching `@testset`.
3. **Frozen gates.** Every quantitative claim is asserted against a threshold
   that is written down in the file and does not move. State next to each gate
   where the number comes from (analytical solution, published reference with a
   citation, or a measured baseline with the date and the machine). A gate that
   is loosened to make a test pass is a regression, not a fix.
4. **Reference data** (published values, tables digitised from a paper,
   reference fields produced by another code) goes in `test/reference/`, as
   plain CSV with a header line, plus a short `README.md` in the same directory
   stating the provenance: who produced it, with what code, what parameters,
   what date.
5. **Figures** go in `benchmarks/results/<topic>/`, with a `README.md` giving
   the command that regenerates them.
6. **Both precisions, both backends** where the case allows it: CPU Float64,
   CPU Float32, GPU Float64, GPU Float32. Parity between CPU and GPU is itself
   a gate.

---

## 5. GPU

- Long runs are GPU runs. Use `CUDABackend()`. Do not benchmark on CPU.
- CPU is acceptable for short unit tests (under about thirty seconds).
- On the QUT Aqua cluster, always export `JULIA_CUDA_USE_COMPAT=false` before
  launching Julia — the compute nodes need the system CUDA driver. Request GPUs
  with `gpu_id=H100` or `gpu_id=A100`. The scheduler is PBS Pro, not Slurm;
  containers are Apptainer, not Docker.

---

## 6. Configuration files (`.krk`)

Kraken simulations can be driven by a `.krk` configuration file parsed by
`src/io/kraken_parser.jl`, run with
`run_simulation("path/to/case.krk")`. When a validation case can be expressed
as a `.krk` file, prefer that over a hand-written Julia driver: it is what
users will actually run, so it is what should be tested.

Fixture naming convention, which carries meaning:

- `<flow>_<layout>.krk` — production configuration, default settings.
- `<flow>_<layout>_debug.krk` — sets a non-default internal mode. A failure
  here may be intentional behaviour of that mode, not a bug.

Do not change the semantics of an existing `.krk` keyword. That is a breaking
change for every user script and needs a decision from the maintainer first.

---

## 7. Conventions

- Code, comments, docstrings, commit messages, documentation: **English**.
- Docstrings: Google style adapted to Julia.
- Equations in documentation: LaTeX inside ` ```math ` blocks.
- Every bibliographic reference carries a DOI and lives in `docs/refs.bib`.
- Documentation pages are written with Literate.jl: a `.jl` file whose comments
  starting with `# ` become Markdown.

---

## 8. Things that waste days here — verify before assuming

- **Check which code path actually runs** before forming a hypothesis about a
  kernel bug. Several past investigations correctly analysed a function that
  the failing case never calls. Add a log line at the entry of every candidate
  kernel, run the failing case for five steps, and read the trace first.
- **Check the backend.** The same logical operation lives in different files
  for CPU and for GPU. State the backend explicitly in any bug report.
- **Check the mode.** A `.krk` fixture may select a strict internal mode whose
  observable behaviour differs from the default on purpose.

---

## 9. Asking

If a task requires touching an owned directory, or changes a `.krk` keyword, or
loosens a frozen gate: stop and open an issue instead. Describe what you wanted
to do and why the current structure blocks it.
