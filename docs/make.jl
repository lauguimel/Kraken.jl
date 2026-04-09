using Documenter
using DocumenterCitations
using Literate
using PlutoStaticHTML
using Kraken

# --- Process Literate.jl files ---

const DOCS_SRC = joinpath(@__DIR__, "src")

const LITERATE_DIRS = [
    "theory",
    "examples",
    "benchmarks",
    "tutorials",
]

for dir in LITERATE_DIRS
    src_dir = joinpath(DOCS_SRC, dir)
    out_dir = joinpath(DOCS_SRC, dir)
    for file in sort(readdir(src_dir))
        endswith(file, ".jl") || continue
        Literate.markdown(
            joinpath(src_dir, file), out_dir;
            documenter = true,
            credit = false,
            # Non-executable code blocks: show code without running simulations
            # To enable execution (for CI with GPU), change to:
            #   codefence = nothing  (default, generates @example blocks)
            codefence = "```julia" => "```",
        )
    end
end

# --- Process Pluto notebooks (interactive tutorials with WGLMakie) ---

const TUTORIAL_DIR = joinpath(DOCS_SRC, "tutorials")

# Pluto notebooks (if any): only process files that start with Pluto header
if isdir(TUTORIAL_DIR)
    pluto_files = filter(readdir(TUTORIAL_DIR)) do f
        endswith(f, ".jl") || return false
        first_line = readline(joinpath(TUTORIAL_DIR, f))
        return startswith(first_line, "### A Pluto.jl notebook ###")
    end
    if !isempty(pluto_files)
        @info "Building Pluto notebooks → tutorials/" pluto_files
        bopts = BuildOptions(TUTORIAL_DIR;
            output_format = documenter_output,
            use_distributed = false,
        )
        build_notebooks(bopts, sort(pluto_files))
    end
end

# --- Bibliography ---

bib = CitationBibliography(joinpath(@__DIR__, "refs.bib"); style=:numeric)

# --- Build documentation ---

# Note: on Julia 1.12+ with Metal.jl, makedocs may segfault due to libgit2.
# Workaround: run from a temp directory with remotes=nothing, or use CI.

makedocs(;
    sitename = "Kraken.jl",
    modules = [Kraken],
    plugins = [bib],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://lauguimel.github.io/Kraken.jl",
        edit_link = "lbm",
        repolink = "https://github.com/lauguimel/Kraken.jl",
        mathengine = Documenter.MathJax3(),
        size_threshold = nothing,
    ),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        # v0.1.0 scope: single-phase LBM (2D/3D), thermal, grid refinement,
        # spatial BCs, .krk DSL. Out-of-scope pages (phasefield, VOF/PLIC,
        # rheology, viscoelastic, Shan-Chen, species) are excluded here.
        # TODO(Phase 4): re-add theory 12 (MRT), 18 (grid refinement),
        # 19 (spatial BCs) and example 20 (grid refinement cavity) once the
        # corresponding .jl source files are authored.
        "Theory" => [
            "theory/01_lbm_fundamentals.md",
            "theory/02_d2q9_lattice.md",
            "theory/03_bgk_collision.md",
            "theory/04_streaming.md",
            "theory/05_boundary_conditions.md",
            "theory/06_from_2d_to_3d.md",
            "theory/07_body_forces.md",
            "theory/08_thermal_ddf.md",
            "theory/09_axisymmetric.md",
            "theory/10_limitations.md",
        ],
        "Examples" => [
            "examples/01_poiseuille_2d.md",
            "examples/02_couette_2d.md",
            "examples/03_taylor_green_2d.md",
            "examples/04_cavity_2d.md",
            "examples/05_cavity_3d.md",
            "examples/06_cylinder_2d.md",
            "examples/07_heat_conduction.md",
            "examples/08_rayleigh_benard.md",
            "examples/09_hagen_poiseuille.md",
            "examples/10_krk_config.md",
        ],
        "Tutorials" => [
            "tutorials/02_couette_2d.md",
        ],
        "Benchmarks" => [
            "benchmarks/mlups_cpu_gpu.md",
            "benchmarks/mesh_convergence.md",
            "benchmarks/comparison_openfoam.md",
        ],
        "API Reference" => "api.md",
    ],
    remotes = nothing,
    warnonly = true,
    checkdocs = :none,
)

deploydocs(
    repo = "github.com/lauguimel/Kraken.jl.git",
    devbranch = "main",
)
