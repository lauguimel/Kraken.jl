# Contributing to Kraken.jl

## Quick Start

```bash
# Install / update dependencies
julia --project -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project test/runtests.jl

# Build docs locally
julia --project=docs docs/make.jl
```

## Project Structure

```
src/
  Kraken.jl              # Main module (exports)
  simulation.jl          # High-level simulation drivers
  lattice/               # D2Q9, D3Q19 lattice definitions
  kernels/               # GPU kernels (stream, collide, BCs, thermal, multiphase)
  rheology/              # Non-Newtonian rheology models
  refinement/            # Patch-based grid refinement
  io/                    # VTK output, .krk config parser
test/
  runtests.jl            # Test entry point
docs/
  make.jl                # Documenter + Literate.jl build
  refs.bib               # BibTeX references (DOI-verified)
  src/
    theory/              # Literate theory pages (.jl → .md)
    examples/            # Literate validation examples (.jl → .md)
    benchmarks/          # Literate benchmark pages (.jl → .md)
```

## Conventions

- **Code and comments**: English
- **Commit style**: conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`)
- **Docstrings**: Google-style adapted for Julia
- **Documentation**: literate (Literate.jl) — `.jl` files with `# ` markdown comments
- **Equations**: LaTeX in ` ```math ` blocks
- **References**: all with DOI in `docs/refs.bib`
- **Examples**: self-contained, runnable, with validation against analytical/reference solutions

## Adding New Physics

1. Create kernel(s) in `src/kernels/`
2. Export functions in `src/Kraken.jl`
3. Add a simulation driver in `src/simulation.jl` (or appropriate `src/drivers/` file)
4. Write tests in `test/`
5. Add a theory page in `docs/src/theory/`
6. Add a validation example in `docs/src/examples/`
