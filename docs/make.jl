using Documenter
using DocumenterCitations
using Literate
using PlutoStaticHTML
using Kraken

const DOCS_LINKCHECK = lowercase(get(ENV, "DOCUMENTER_LINKCHECK", "false")) in ("1", "true", "yes")

function lint_implication_maps()
    agent_dir = joinpath(@__DIR__, "agent")
    linter = joinpath(@__DIR__, "..", "scripts", "lint-implication-map.sh")
    isfile(linter) || error("Track-C linter not found: $(linter)")
    isdir(agent_dir) || error("Track-C agent docs directory not found: $(agent_dir)")

    maps = sort(filter(name -> endswith(name, "-implication.md"), readdir(agent_dir; join = true)))
    isempty(maps) && error("No Track-C implication maps found under $(agent_dir)")

    for map in maps
        @info "Linting Track-C implication map" map
        try
            run(`bash $linter $map`)
        catch err
            error("Track-C implication map lint failed: $(map)\n$(sprint(showerror, err))")
        end
    end
end

lint_implication_maps()

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

# --- Phase 4.1A proof-of-concept: build _helpers/_test_helpers.jl ---
# Only the `_test_helpers.jl` file in docs/src/_helpers/ is a Literate page;
# the other .jl files there are plain Julia helper modules loaded above.
let helpers_dir = joinpath(DOCS_SRC, "_helpers"),
    test_file  = joinpath(helpers_dir, "_test_helpers.jl")
    if isfile(test_file)
        Literate.markdown(
            test_file, helpers_dir;
            documenter = true,
            credit = false,
            execute = true,
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
        "Getting started" => "getting_started.md",
        "Concepts" => "concepts_index.md",
        "Capabilities" => "capabilities.md",
        "Users" => [
            "KRK reference" => "users/krk-reference.md",
            "Axis tutorials" => [
                "Cartesian cavity" => "users/tutorials/cartesian-cavity.md",
                "Thermal natural convection" => "users/tutorials/thermal-natural-convection.md",
                "Sphere drag 3D" => "users/tutorials/sphere-drag-3d.md",
                "Viscoelastic cylinder" => "users/tutorials/viscoelastic-cylinder.md",
            ],
            "Benchmarks" => [
                "Validation matrix" => "users/benchmarks/validation-matrix.md",
                "Cartesian cavity" => "users/benchmarks/cartesian-cavity.md",
                "Thermal natural convection" => "users/benchmarks/thermal-natural-convection.md",
                "Sphere drag 3D" => "users/benchmarks/sphere-drag-3d.md",
                "Viscoelastic cylinder (Oldroyd-B)" => "users/benchmarks/viscoelastic-cylinder.md",
                "GPU certification" => "users/benchmarks/gpu-certification.md",
            ],
        ],
        "API" => [
            "Units" => "api/units.md",
            "Geometry" => "api/geometry.md",
            "LBM" => "api/lbm.md",
            "Physics: Newtonian" => "api/physics-newtonian.md",
            "Physics: Viscoelastic" => "api/physics-viscoelastic.md",
            "Physics: Thermal" => "api/physics-thermal.md",
            "Boundary conditions" => "api/bc.md",
            "Backend" => "api/backend.md",
            "KRK I/O" => "api/io-krk.md",
        ],
        # v0.2.0 scope: single-phase LBM (2D/3D), thermal, grid refinement,
        # spatial BCs, .krk DSL. Out-of-scope pages (phasefield, VOF/PLIC,
        # rheology, viscoelastic, Shan-Chen, species) are excluded here.
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
            "theory/12_mrt.md",
            "theory/18_grid_refinement.md",
            "theory/19_spatial_bcs.md",
        ],
        "Examples-tutorials" => [
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
            "examples/20_grid_refinement_cavity.md",
        ],
        "Benchmarks" => [
            "benchmarks/performance.md",
            "benchmarks/accuracy.md",
            "benchmarks/external.md",
            "benchmarks/hardware.md",
        ],
        ".krk DSL reference" => [
            "krk/overview.md",
            "krk/directives.md",
            "krk/bc_types.md",
            "krk/modules.md",
            "krk/presets.md",
            "krk/helpers.md",
            "krk/expressions.md",
            "krk/sanity.md",
            "krk/errors.md",
            "krk/aliases.md",
        ],
        "Julia API reference" => [
            "api/lattice.md",
            "api/collision.md",
            "api/streaming.md",
            "api/boundary.md",
            "api/macroscopic.md",
            "api/drivers.md",
            "api/refinement.md",
            "api/io.md",
            "api/postprocess.md",
            "api/config.md",
        ],
    ],
    remotes = nothing,
    linkcheck = DOCS_LINKCHECK,
    warnonly = DOCS_LINKCHECK ? Documenter.except(:linkcheck) : true,
    checkdocs = :none,
)

deploydocs(
    repo = "github.com/lauguimel/Kraken.jl.git",
    devbranch = "main",
)
