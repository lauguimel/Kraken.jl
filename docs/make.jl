using Documenter
using Kraken

makedocs(;
    sitename = "Kraken.jl",
    modules = [Kraken],
    pages = [
        "Home" => "index.md",
    ],
)
