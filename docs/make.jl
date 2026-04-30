using Documenter
using DocumenterCitations
using DocumenterVitepress
using Literate
using NodeJS_20_jll: node, npm
using PlutoStaticHTML
using Kraken

# --- Living-documentation helpers (Phase 4.1A) ---
# Loaded into Main so Literate.jl preprocessing and @example blocks
# can call extract_function / krk_download / api_page_data directly.
include(joinpath(@__DIR__, "src", "_helpers", "source_extract.jl"))
include(joinpath(@__DIR__, "src", "_helpers", "krk_download.jl"))
include(joinpath(@__DIR__, "src", "_helpers", "api_extract.jl"))

# --- Process Literate.jl files ---

const DOCS_SRC = joinpath(@__DIR__, "src")

const LITERATE_DIRS = [
    "theory",
    "examples",
    "benchmarks",
    "tutorials",
]

# Out-of-scope sources (not listed in `pages`) are skipped so Vitepress does
# not try to build orphan pages that may reference missing assets.
const LITERATE_EXCLUDE = Set{String}([
    # Replaced by curated, CSV-backed benchmark pages in this branch.
    "benchmarks/mesh_convergence.jl",
    "benchmarks/mlups_cpu_gpu.jl",
])

# --- Living-doc preprocessing: expand @@EXTRACT path symbol@@ markers ---
# Markers in Literate sources are replaced at build-time with a fenced
# Julia code block extracted from the real source file via
# `extract_function`. This keeps theory pages in sync with the code.
const _SRC_ROOT = joinpath(@__DIR__, "..")

# The `@@EXTRACT path symbol@@` marker sits inside a Literate `# ...`
# comment block. It is expanded at build time into a fenced julia code
# block that still lives in the Literate comment region (each emitted
# line is prefixed with `# `), so Literate renders it as raw markdown.
function literate_preprocess(content::AbstractString)
    re = r"(?m)^#\s*@@EXTRACT\s+(\S+)\s+(\S+?)@@\s*$"
    return replace(content, re => function (m)
        mm = match(re, m)
        relpath = String(mm.captures[1])
        symname = Symbol(String(mm.captures[2]))
        filepath = joinpath(_SRC_ROOT, relpath)
        try
            r = extract_function(filepath, symname)
            # Emit as a Literate markdown block: each line prefixed with "# "
            lines = split(r.full_text, '\n')
            return join(("# " * l for l in lines), '\n')
        catch err
            @warn "EXTRACT marker expansion failed" relpath symname err
            return "# `extract failed: $(relpath) :$(symname)`"
        end
    end)
end

for dir in LITERATE_DIRS
    src_dir = joinpath(DOCS_SRC, dir)
    out_dir = joinpath(DOCS_SRC, dir)
    for file in sort(readdir(src_dir))
        endswith(file, ".jl") || continue
        joinpath(dir, file) in LITERATE_EXCLUDE && continue
        Literate.markdown(
            joinpath(src_dir, file), out_dir;
            documenter = true,
            credit = false,
            preprocess = literate_preprocess,
            # Non-executable code blocks: show code without running simulations
            # To enable execution (for CI with GPU), change to:
            #   codefence = nothing  (default, generates @example blocks)
            codefence = "```julia" => "```",
        )
    end
end

# `_helpers/` contains build helpers, not public documentation pages.

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
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/lauguimel/Kraken.jl",
        devurl = "dev",
        devbranch = "release/v0.1.0",
        deploy_url = "lauguimel.github.io/Kraken.jl",
        build_vitepress = false,
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "Installation" => "installation.md",
            "Quick start" => "getting_started.md",
            "Concepts" => "concepts_index.md",
            "Capabilities" => "capabilities.md",
            "Integration roadmap" => "integration_roadmap.md",
            "LLM / agent context" => "llms.md",
        ],
        "Theory" => [
            "LBM fundamentals" => "theory/01_lbm_fundamentals.md",
            "D2Q9 lattice" => "theory/02_d2q9_lattice.md",
            "BGK collision" => "theory/03_bgk_collision.md",
            "Streaming" => "theory/04_streaming.md",
            "Boundary conditions" => "theory/05_boundary_conditions.md",
            "From 2D to 3D" => "theory/06_from_2d_to_3d.md",
            "Body forces" => "theory/07_body_forces.md",
            "Thermal DDF" => "theory/08_thermal_ddf.md",
            "Limitations" => "theory/10_limitations.md",
            "Spatial BCs" => "theory/19_spatial_bcs.md",
        ],
        "Tutorials" => [
            "First Steps" => [
                "Hello KRK" => "tutorials/first_steps/01_hello_krk.md",
                "Build a KRK" => "tutorials/first_steps/02_build_a_krk.md",
                "Cookbook" => "tutorials/first_steps/03_cookbook.md",
            ],
            "Simulations" => [
                "Your first simulation" => "tutorials/01_first_simulation.md",
                "Body forces" => "tutorials/02_body_forces.md",
                "Obstacles" => "tutorials/03_obstacles.md",
                "Thermal flows" => "tutorials/04_thermal.md",
            ],
            "Newtonian Flows" => [
                "Poiseuille 2D" => "examples/01_poiseuille_2d.md",
                "Couette 2D" => "examples/02_couette_2d.md",
                "Taylor-Green 2D" => "examples/03_taylor_green_2d.md",
                "Lid-driven cavity 2D" => "examples/04_cavity_2d.md",
                "Lid-driven cavity 3D" => "examples/05_cavity_3d.md",
                "Cylinder 2D" => "examples/06_cylinder_2d.md",
            ],
            "Thermal Flows" => [
                "Heat conduction" => "examples/07_heat_conduction.md",
                "Rayleigh-Bénard" => "examples/08_rayleigh_benard.md",
            ],
            "KRK Walk-through" => [
                ".krk config reference" => "examples/10_krk_config.md",
            ],
        ],
        "Benchmarks" => [
            "Performance" => "benchmarks/performance.md",
            "Accuracy" => "benchmarks/accuracy.md",
            "External comparison" => "benchmarks/external.md",
            "Hardware" => "benchmarks/hardware.md",
        ],
        "Reference" => [
            ".krk DSL" => [
                "Overview" => "krk/overview.md",
                "Directives" => "krk/directives.md",
                "BC types" => "krk/bc_types.md",
                "Modules" => "krk/modules.md",
                "Presets" => "krk/presets.md",
                "Helpers" => "krk/helpers.md",
                "Expressions" => "krk/expressions.md",
                "Sanity" => "krk/sanity.md",
                "Errors" => "krk/errors.md",
                "Aliases" => "krk/aliases.md",
            ],
            "Julia API" => [
                "Public API inventory" => "api/public_api.md",
                "Lattice" => "api/lattice.md",
                "Collision" => "api/collision.md",
                "Streaming" => "api/streaming.md",
                "Boundary" => "api/boundary.md",
                "Macroscopic" => "api/macroscopic.md",
                "Drivers" => "api/drivers.md",
                "IO" => "api/io.md",
                "Postprocess" => "api/postprocess.md",
                "Config" => "api/config.md",
            ],
            "Julia ecosystem docs" => "julia_docs.md",
        ],
    ],
    remotes = nothing,
    warnonly = false,
    checkdocs = :none,
)

# --- Prune orphan pages, then invoke Vitepress build manually ---
# DocumenterVitepress copies all of `docs/src/` into `build/.documenter/`, so
# out-of-scope .md files (v0.1.0 excludes phasefield, VOF, rheology, etc.)
# reach Vitepress and may fail on missing assets. Drop them before building.
let vp_input = joinpath(@__DIR__, "build", ".documenter")
    rm(joinpath(vp_input, "_helpers"); recursive=true, force=true)
    # DocumenterVitepress rewrites `../assets/...` links inside `examples/`
    # to `assets/...`. Mirror downloadable .krk files at that rewritten path
    # so VitePress dead-link checks stay strict.
    krk_src = joinpath(vp_input, "assets", "krk")
    krk_dst = joinpath(vp_input, "examples", "assets", "krk")
    if isdir(krk_src)
        mkpath(krk_dst)
        for file in readdir(krk_src; join=true)
            endswith(file, ".krk") && cp(file, joinpath(krk_dst, basename(file)); force=true)
        end
    end
    for rel in LITERATE_EXCLUDE
        for ext in (".jl", ".md")
            p = joinpath(vp_input, replace(rel, r"\.jl$" => ext))
            isfile(p) && rm(p)
        end
    end

    cd(@__DIR__) do
        tmpl_pkg = joinpath(dirname(pathof(DocumenterVitepress)), "..", "template", "package.json")
        pkg_json = joinpath(@__DIR__, "package.json")
        cleanup_pkg = !isfile(pkg_json)
        cleanup_pkg && cp(tmpl_pkg, pkg_json)
        try
            node(; adjust_PATH = true, adjust_LIBPATH = true) do _
                run(`$(npm) install`)
                run(`$(npm) run env -- vitepress build $(vp_input)`)
            end
        finally
            if cleanup_pkg
                rm(pkg_json; force = true)
                rm(joinpath(@__DIR__, "package-lock.json"); force = true)
            end
        end
    end
end

DocumenterVitepress.deploydocs(;
    repo = "github.com/lauguimel/Kraken.jl.git",
    target = joinpath(@__DIR__, "build"),
    devbranch = "release/v0.1.0",
    branch = "gh-pages",
    push_preview = true,
)
