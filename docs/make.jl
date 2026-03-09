using Documenter
using Kraken

makedocs(;
    sitename = "Kraken.jl",
    modules = [Kraken],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://lauguimel.github.io/Kraken.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Theory" => [
            "theory/governing-equations.md",
            "theory/finite-differences.md",
            "theory/projection-method.md",
            "theory/spatial-schemes.md",
            "theory/time-integration.md",
            "theory/linear-solvers.md",
            "theory/collocated-grids.md",
            "theory/boundary-conditions.md",
        ],
        "Benchmarks" => [
            "benchmarks/overview.md",
        ],
        "Tutorials" => [
            "tutorials/getting-started.md",
        ],
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/lauguimel/Kraken.jl.git",
    devbranch = "main",
)
