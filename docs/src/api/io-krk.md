# KRK I/O API

The KRK I/O API documents expression parsing, `.krk` setup loading and sanity checks, the generic simulation runner, and the Units-side bridge for parsing lattice-unit planning blocks.

```@autodocs
Modules = [Kraken]
Pages = [
    "io/expression.jl",
    "io/kraken_parser.jl",
    "simulation_runner.jl",
]
Order = [:constant, :type, :function]
```

```@autodocs
Modules = [Kraken.Units]
Pages = [
    "units/krk_binding.jl",
]
Order = [:constant, :type, :function]
```
