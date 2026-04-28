# Contributing to Kraken.jl

Contributions are welcome! This guide covers the basics.

## Quick Start

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project -e 'using Pkg; Pkg.test()'
julia --project=docs docs/make.jl      # build docs locally
```

## Project Structure

```
src/
  Kraken.jl              # Module definition and exports
  lattice/               # D2Q9, D3Q19 lattice definitions
  kernels/               # GPU-portable collision, streaming, BC kernels
  drivers/               # High-level simulation drivers (basic, thermal)
  io/                    # VTK writer, .krk parser, diagnostics
  simulation_runner.jl   # Generic .krk → simulation dispatcher
  postprocess.jl         # Post-processing helpers
examples/                # .krk configuration files
docs/
  src/tutorials/         # Progressive tutorials (Literate.jl)
  src/examples/          # Validated example pages
  src/theory/            # LBM theory pages
test/                    # Test suite (379 tests)
vscode-krk/             # VS Code extension for .krk files
bin/                     # CLI wrapper
```

## Conventions

- **Code and comments**: English
- **Commit style**: conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`)
- **Documentation**: Literate.jl — `.jl` files with `# ` markdown comments
- **Equations**: LaTeX in ` ```math ` blocks
- **References**: all with DOI in `docs/refs.bib`
- **Array layout**: always `f[i, j, q]`

## Adding a New Example

1. Create a `.krk` file in `examples/`
2. Create a Literate.jl page in `docs/src/examples/` or `docs/src/tutorials/`
3. Add CairoMakie plot generation so figures are produced when running
4. Add the page to `docs/make.jl`
5. Add a test in `test/test_krk_examples.jl`

## Adding New Physics

1. Create kernel(s) in `src/kernels/`
2. Export functions in `src/Kraken.jl`
3. Wire the `.krk` dispatch in `src/simulation_runner.jl`
4. Write tests in `test/`
5. Add a theory page + validated example in `docs/`
