# --- Parser for .krk simulation config files ---

"""
    DomainSetup

Domain geometry: physical extents and grid resolution.
"""
struct DomainSetup
    Lx::Float64
    Ly::Float64
    Lz::Float64
    Nx::Int
    Ny::Int
    Nz::Int
end

"""
    PhysicsSetup

Physical parameters (nu, Pr, Ra, etc.) and optional body force expressions.
"""
struct PhysicsSetup
    params::Dict{Symbol, Float64}
    body_force::Dict{Symbol, KrakenExpr}  # :Fx, :Fy, :Fz
end

"""
    STLSource

Reference to an STL file with optional transform parameters.
"""
struct STLSource
    file::String
    scale::Float64
    translate::NTuple{3, Float64}
    z_slice::Float64   # z-plane for 2D cross-section (default 0.0)
end

"""
    GeometryRegion

A geometry region defining solid obstacles or fluid zones.
Either via a condition expression OR an STL file (mutually exclusive).
"""
struct GeometryRegion
    name::String
    kind::Symbol                        # :obstacle or :fluid
    condition::Union{KrakenExpr, Nothing}  # (x, y [,z]) -> Bool (nothing if STL)
    stl::Union{STLSource, Nothing}         # STL file source (nothing if condition)
    bc_type::Symbol                     # :wall (default)
    bc_values::Dict{Symbol, KrakenExpr}
end

"""
    BoundarySetup

Boundary condition for one face of the domain.
"""
struct BoundarySetup
    face::Symbol          # :north, :south, :east, :west (2D)
    type::Symbol          # :wall, :velocity, :pressure, :periodic
    values::Dict{Symbol, KrakenExpr}
end

"""
    InitialSetup

Initial condition expressions for fields.
"""
struct InitialSetup
    fields::Dict{Symbol, KrakenExpr}
end

"""
    OutputSetup

Output configuration: format, interval, and field names.
"""
struct OutputSetup
    format::Symbol
    interval::Int
    fields::Vector{Symbol}
    directory::String
end

"""
    DiagnosticsSetup

Diagnostics logging configuration.
"""
struct DiagnosticsSetup
    interval::Int
    columns::Vector{Symbol}
end

"""
    RefineSetup

Refinement patch specification from a .krk `Refine` block.
"""
struct RefineSetup
    name::String
    region::NTuple{4, Float64}   # (x_min, y_min, x_max, y_max)
    ratio::Int                   # refinement ratio (default 2)
    parent::String               # parent patch name ("" = base grid)
end

"""
    RheologySetup

Rheology model specification for a single phase.
The `model` symbol selects the type (`:newtonian`, `:power_law`, `:carreau`,
`:cross`, `:bingham`, `:herschel_bulkley`, `:oldroyd_b`, `:fene_p`, `:saramito`).
The `params` dict holds the model parameters (e.g., `K`, `n`, `nu_min`, `nu_max`).
"""
struct RheologySetup
    phase::Symbol                    # :liquid, :gas, or :default
    model::Symbol                    # :newtonian, :power_law, :carreau, etc.
    params::Dict{Symbol, Float64}
end

"""
    SimulationSetup

Top-level simulation configuration parsed from a .krk file.
"""
struct SimulationSetup
    name::String
    lattice::Symbol
    domain::DomainSetup
    physics::PhysicsSetup
    user_vars::Dict{Symbol, Float64}
    regions::Vector{GeometryRegion}
    boundaries::Vector{BoundarySetup}
    initial::Union{InitialSetup, Nothing}
    modules::Vector{Symbol}
    max_steps::Int
    output::Union{OutputSetup, Nothing}
    diagnostics::Union{DiagnosticsSetup, Nothing}
    refinements::Vector{RefineSetup}
    velocity_field::Union{InitialSetup, Nothing}  # prescribed velocity expressions (ux, uy)
    rheology::Vector{RheologySetup}                # per-phase rheology models
end

# --- Tokenization: strip comments, join multi-line blocks ---

"""
    _preprocess_lines(text::String) -> Vector{String}

Strip comments, remove blank lines, and join multi-line `{ ... }` blocks
into single statements.
"""
function _preprocess_lines(text::String)
    lines = String[]
    raw_lines = split(text, '\n')

    buffer = ""
    brace_depth = 0

    for raw in raw_lines
        # Strip comments (but not inside strings — simple heuristic)
        stripped = _strip_comment(String(raw))
        trimmed = strip(stripped)
        isempty(trimmed) && brace_depth == 0 && continue

        # Count braces
        opens = count(==( '{'), trimmed)
        closes = count(==('}'), trimmed)

        if brace_depth == 0 && opens == 0
            # Simple single-line statement
            push!(lines, String(trimmed))
        else
            # Accumulate multi-line block
            buffer *= " " * String(trimmed)
            brace_depth += opens - closes
            if brace_depth <= 0
                push!(lines, strip(buffer))
                buffer = ""
                brace_depth = 0
            end
        end
    end

    if brace_depth > 0
        throw(ArgumentError("Unclosed brace in .krk file"))
    end

    return lines
end

function _strip_comment(line::String)
    # Simple: strip everything after # that's not preceded by backslash
    idx = findfirst('#', line)
    idx === nothing && return line
    return line[1:prevind(line, idx)]
end

# --- Main parser ---

"""
    load_kraken(filename::String; kwargs...) -> SimulationSetup

Parse a `.krk` file and return a `SimulationSetup` struct.
Keyword arguments override `Define` defaults for parametric studies.

# Example
```julia
setup = load_kraken("examples/cavity.krk")
setup = load_kraken("examples/cavity.krk"; Re=400, N=256)
```
"""
function load_kraken(filename::String; kwargs...)
    text = read(filename, String)
    return parse_kraken(text; kwargs...)
end

"""
    parse_kraken(text::String; kwargs...) -> SimulationSetup

Parse .krk format text into a SimulationSetup struct.
Keyword arguments override `Define` defaults for parametric studies.

# Example
```julia
setup = parse_kraken(text; Re=400, N=256)
```
"""
function parse_kraken(text::String; kwargs...)
    setups = _parse_kraken_internal(text; kwargs...)
    length(setups) == 1 || throw(ArgumentError(
        "parse_kraken: got $(length(setups)) setups (Sweep directive present?). " *
        "Use parse_kraken_sweep for sweeps."))
    return setups[1]
end

"""
    parse_kraken_sweep(text::String; kwargs...) -> Vector{SimulationSetup}

Parse a .krk file containing zero or more `Sweep param = [a, b, c]` directives
and return one `SimulationSetup` per combination of sweep values. If no sweep
is present, returns a single-element vector.
"""
function parse_kraken_sweep(text::String; kwargs...)
    return _parse_kraken_internal(text; kwargs...)
end

"""
    load_kraken_sweep(filename::String; kwargs...) -> Vector{SimulationSetup}

File version of [`parse_kraken_sweep`](@ref).
"""
function load_kraken_sweep(filename::String; kwargs...)
    return parse_kraken_sweep(read(filename, String); kwargs...)
end

function _parse_kraken_internal(text::String; kwargs...)
    lines = _preprocess_lines(text)

    # --- Pre-pass: Preset expansion ---
    expanded = String[]
    for line in lines
        if _first_word(line) == "Preset"
            append!(expanded, _expand_preset(line))
        else
            push!(expanded, line)
        end
    end
    lines = expanded

    # --- Sweep pre-pass: collect sweeps and expand combinations ---
    sweeps = Pair{Symbol, Vector{Float64}}[]
    non_sweep_lines = String[]
    for line in lines
        if _first_word(line) == "Sweep"
            push!(sweeps, _parse_sweep(line))
        else
            push!(non_sweep_lines, line)
        end
    end
    lines = non_sweep_lines

    if !isempty(sweeps)
        setups = SimulationSetup[]
        # Cartesian product
        counters = ones(Int, length(sweeps))
        sizes = [length(s.second) for s in sweeps]
        while true
            sweep_kwargs = Dict{Symbol, Any}(kwargs)
            for (i, sw) in enumerate(sweeps)
                sweep_kwargs[sw.first] = sw.second[counters[i]]
            end
            append!(setups, _parse_kraken_internal_single(lines; sweep_kwargs...))
            # advance
            k = length(counters)
            while k > 0
                counters[k] += 1
                if counters[k] > sizes[k]
                    counters[k] = 1
                    k -= 1
                else
                    break
                end
            end
            k == 0 && break
        end
        return setups
    end

    return _parse_kraken_internal_single(lines; kwargs...)
end

function _parse_kraken_internal_single(lines::Vector{String}; kwargs...)
    # --- First pass: collect Define defaults ---
    user_vars = Dict{Symbol, Float64}()
    for line in lines
        _first_word(line) == "Define" || continue
        k, v = _parse_define(line)
        user_vars[k] = v
    end

    # Override with kwargs (highest priority)
    for (k, v) in kwargs
        user_vars[k] = Float64(v)
    end

    # --- Second pass: parse everything ---
    name = ""
    lattice = :D2Q9
    domain = nothing
    physics_params = Dict{Symbol, Float64}()
    body_force = Dict{Symbol, KrakenExpr}()
    regions = GeometryRegion[]
    boundaries = BoundarySetup[]
    initial = nothing
    velocity_field = nothing
    modules = Symbol[]
    # Pre-scan modules so boundary parsing can honour module-specific aliases
    # (e.g. `Module axisymmetric` enables `Boundary z/wall/axis ...`).
    for line in lines
        _first_word(line) == "Module" || continue
        push!(modules, _parse_module(line))
    end
    is_axisym = :axisymmetric in modules
    max_steps = 0
    output = nothing
    diagnostics = nothing
    refinements = RefineSetup[]
    rheology_setups = RheologySetup[]
    setup_helpers = Dict{Symbol, Float64}()  # reynolds, rayleigh, prandtl, L_ref, U_ref

    for line in lines
        keyword = _first_word(line)

        if keyword == "Simulation"
            name, lattice = _parse_simulation(line)
        elseif keyword == "Domain"
            domain = _parse_domain(line, user_vars)
        elseif keyword == "Physics"
            merge!(physics_params, _parse_physics(line, user_vars))
        elseif keyword == "Define"
            # Already processed in first pass
        elseif keyword == "Obstacle"
            push!(regions, _parse_obstacle(line, user_vars))
        elseif keyword == "Fluid"
            push!(regions, _parse_fluid(line, user_vars))
        elseif keyword == "Boundary"
            append!(boundaries, _parse_boundary(line, user_vars; is_axisym=is_axisym))
        elseif keyword == "Refine"
            push!(refinements, _parse_refine(line, user_vars))
        elseif keyword == "Initial"
            initial = _parse_initial(line, user_vars)
        elseif keyword == "Velocity"
            velocity_field = _parse_initial(line, user_vars)  # same { ux = ... uy = ... } syntax
        elseif keyword == "Module"
            # Already collected in the pre-scan above

        elseif keyword == "Run"
            max_steps = _parse_run(line)
        elseif keyword == "Output"
            output = _parse_output(line)
        elseif keyword == "Diagnostics"
            diagnostics = _parse_diagnostics(line)
        elseif keyword == "Rheology"
            push!(rheology_setups, _parse_rheology(line, user_vars))
        elseif keyword == "Setup"
            merge!(setup_helpers, _parse_setup(line, user_vars))
        else
            known = ("Simulation", "Domain", "Physics", "Define", "Obstacle",
                     "Fluid", "Boundary", "Refine", "Initial", "Velocity",
                     "Module", "Run", "Output", "Diagnostics", "Rheology",
                     "Setup", "Preset", "Sweep")
            suggestion = _suggest_name(keyword, known)
            msg = "Unknown keyword '$keyword' in .krk file"
            if suggestion !== nothing
                msg *= " (did you mean: $suggestion?)"
            end
            throw(ArgumentError(msg))
        end
    end

    domain === nothing && throw(ArgumentError("Missing 'Domain' in .krk file"))
    isempty(name) && throw(ArgumentError("Missing 'Simulation' in .krk file"))
    max_steps == 0 && throw(ArgumentError("Missing 'Run' in .krk file"))

    # --- Apply kwargs overrides to Physics and Domain ---
    param_kwargs = Dict{Symbol,Float64}(k => Float64(v) for (k, v) in kwargs)

    # Override Physics params with matching kwargs
    for (k, v) in param_kwargs
        if haskey(physics_params, k)
            physics_params[k] = v
        end
    end

    # Override Domain with matching kwargs (Nx, Ny, Nz, Lx, Ly, Lz)
    if domain !== nothing
        Nx = haskey(param_kwargs, :Nx) ? round(Int, param_kwargs[:Nx]) : domain.Nx
        Ny = haskey(param_kwargs, :Ny) ? round(Int, param_kwargs[:Ny]) : domain.Ny
        Nz = haskey(param_kwargs, :Nz) ? round(Int, param_kwargs[:Nz]) : domain.Nz
        Lx = get(param_kwargs, :Lx, domain.Lx)
        Ly = get(param_kwargs, :Ly, domain.Ly)
        Lz = get(param_kwargs, :Lz, domain.Lz)
        domain = DomainSetup(Lx, Ly, Lz, Nx, Ny, Nz)
    end

    # Override max_steps with kwarg
    if haskey(param_kwargs, :max_steps)
        max_steps = round(Int, param_kwargs[:max_steps])
    end

    # Parse body force if present in physics_params
    for sym in (:Fx, :Fy, :Fz)
        if haskey(physics_params, sym)
            body_force[sym] = parse_kraken_expr(string(physics_params[sym]), user_vars)
            delete!(physics_params, sym)
        end
    end

    # --- Apply Setup helpers (Reynolds, Rayleigh) ---
    _apply_setup_helpers!(physics_params, setup_helpers, domain, boundaries)

    physics = PhysicsSetup(physics_params, body_force)

    setup = SimulationSetup(name, lattice, domain, physics, user_vars,
                            regions, boundaries, initial, modules,
                            max_steps, output, diagnostics, refinements,
                            velocity_field, rheology_setups)

    # --- Validate face names against lattice dimensionality ---
    _validate_faces_vs_lattice(setup)

    # --- Run sanity checks (warnings only) ---
    sanity_check(setup)

    return [setup]
end

# --- Individual statement parsers ---

function _first_word(line::String)
    m = match(r"^(\w+)", line)
    m === nothing && throw(ArgumentError("Cannot parse line: $line"))
    return m.captures[1]
end

"""Parse: Simulation <name> <lattice>"""
function _parse_simulation(line::String)
    tokens = split(line)
    length(tokens) < 3 && throw(ArgumentError("Simulation needs name and lattice: $line"))
    name = String(tokens[2])
    lattice = Symbol(tokens[3])
    lattice in (:D2Q9, :D3Q19) || throw(ArgumentError("Unknown lattice '$lattice'. Use D2Q9 or D3Q19"))
    return name, lattice
end

"""Parse: Domain L = <Lx> x <Ly> N = <Nx> x <Ny>  (values can be expressions/variables)"""
function _parse_domain(line::String, user_vars::Dict{Symbol,Float64}=Dict{Symbol,Float64}())
    # Extract L = ... x ... and N = ... x ...  (accept variable names and numbers)
    lm = match(r"L\s*=\s*([\w.eE+-]+)\s*x\s*([\w.eE+-]+)(?:\s*x\s*([\w.eE+-]+))?", line)
    nm = match(r"N\s*=\s*([\w.eE+-]+)\s*x\s*([\w.eE+-]+)(?:\s*x\s*([\w.eE+-]+))?", line)

    lm === nothing && throw(ArgumentError("Cannot parse Domain L = ... : $line"))
    nm === nothing && throw(ArgumentError("Cannot parse Domain N = ... : $line"))

    Lx = _eval_domain_value(lm.captures[1], user_vars)
    Ly = _eval_domain_value(lm.captures[2], user_vars)
    Lz = lm.captures[3] !== nothing ? _eval_domain_value(lm.captures[3], user_vars) : 1.0

    Nx = round(Int, _eval_domain_value(nm.captures[1], user_vars))
    Ny = round(Int, _eval_domain_value(nm.captures[2], user_vars))
    Nz = nm.captures[3] !== nothing ? round(Int, _eval_domain_value(nm.captures[3], user_vars)) : 1

    return DomainSetup(Lx, Ly, Lz, Nx, Ny, Nz)
end

"""Evaluate a domain value: either a number literal or a user variable."""
function _eval_domain_value(s::AbstractString, user_vars::Dict{Symbol,Float64})
    val = tryparse(Float64, s)
    val !== nothing && return val
    sym = Symbol(s)
    haskey(user_vars, sym) && return user_vars[sym]
    throw(ArgumentError("Unknown variable '$s' in Domain. Define it with 'Define $s = ...' or pass as kwarg."))
end

"""Parse: Physics <key> = <value> ...  (values can be expressions with user vars)"""
function _parse_physics(line::String, user_vars::Dict{Symbol,Float64}=Dict{Symbol,Float64}())
    params = Dict{Symbol, Float64}()
    # Match all key = value pairs (value can be a number or an expression)
    for m in eachmatch(r"(\w+)\s*=\s*([\w.eE+\-*/()]+)", line)
        key = Symbol(m.captures[1])
        val_str = strip(String(m.captures[2]))
        # Try literal first, then evaluate as expression with user vars
        val = tryparse(Float64, val_str)
        if val === nothing
            expr = parse_kraken_expr(val_str, user_vars)
            val = Float64(evaluate(expr))
        end
        params[key] = val
    end
    return params
end

"""Parse: Define <VAR> = <expression>"""
function _parse_define(line::String)
    m = match(r"^Define\s+(\w+)\s*=\s*(.+)$", line)
    m === nothing && throw(ArgumentError("Cannot parse Define: $line"))
    key = Symbol(m.captures[1])
    val = parse(Float64, strip(m.captures[2]))
    return key, val
end

"""Parse: Obstacle <name> [wall(...)] { <condition> }"""
function _parse_obstacle(line::String, user_vars::Dict{Symbol,Float64})
    return _parse_geometry_region(line, :obstacle, user_vars)
end

"""Parse: Fluid <name> { <condition> }"""
function _parse_fluid(line::String, user_vars::Dict{Symbol,Float64})
    return _parse_geometry_region(line, :fluid, user_vars)
end

"""
Parse: Refine <name> { region = [x0, y0, x1, y1], ratio = 2, parent = <name> }
"""
function _parse_refine(line::String, user_vars::Dict{Symbol,Float64})
    # Extract name (second word)
    after_kw = strip(replace(line, r"^\w+" => ""))
    name_m = match(r"^(\w+)", after_kw)
    name_m === nothing && throw(ArgumentError("Missing name in Refine: $line"))
    name = String(name_m.captures[1])

    # Extract block content inside { ... }
    brace_m = match(r"\{(.+)\}", line)
    brace_m === nothing && throw(ArgumentError("Missing { ... } block in Refine: $line"))
    content = strip(String(brace_m.captures[1]))

    # Parse region = [x0, y0, x1, y1]
    region_m = match(r"region\s*=\s*\[([^\]]+)\]", content)
    region_m === nothing && throw(ArgumentError("Missing 'region = [x0, y0, x1, y1]' in Refine: $line"))
    coords = [_eval_number(strip(s), user_vars) for s in split(region_m.captures[1], ",")]
    length(coords) == 4 || throw(ArgumentError("Refine region must have 4 values: $line"))
    region = (coords[1], coords[2], coords[3], coords[4])

    # Parse ratio (default 2)
    ratio_m = match(r"ratio\s*=\s*(\d+)", content)
    ratio = ratio_m !== nothing ? parse(Int, ratio_m.captures[1]) : 2

    # Parse parent (default "" = base grid)
    parent_m = match(r"parent\s*=\s*(\w+)", content)
    parent = parent_m !== nothing ? String(parent_m.captures[1]) : ""

    return RefineSetup(name, region, ratio, parent)
end

"""Evaluate a number string, substituting user variables."""
function _eval_number(s::AbstractString, user_vars::Dict{Symbol,Float64})
    # Try direct parse first
    v = tryparse(Float64, s)
    v !== nothing && return v
    # Try user variable
    sym = Symbol(s)
    haskey(user_vars, sym) && return user_vars[sym]
    throw(ArgumentError("Cannot evaluate '$s' as a number"))
end

function _parse_geometry_region(line::String, kind::Symbol, user_vars::Dict{Symbol,Float64})
    # Extract name (second word after keyword)
    after_kw = strip(replace(line, r"^\w+" => ""))
    name_m = match(r"^(\w+)", after_kw)
    name_m === nothing && throw(ArgumentError("Missing name in: $line"))
    name = String(name_m.captures[1])

    # Check for wall(...) with properties
    bc_type = :wall
    bc_values = Dict{Symbol, KrakenExpr}()
    wall_m = match(r"wall\(([^)]+)\)", line)
    if wall_m !== nothing
        for param_m in eachmatch(r"(\w+)\s*=\s*([^,)]+)", wall_m.captures[1])
            k = Symbol(param_m.captures[1])
            v = strip(String(param_m.captures[2]))
            bc_values[k] = parse_kraken_expr(v, user_vars)
        end
    end

    # Check for STL source: stl(file = "...", scale = ..., ...)
    stl_m = match(r"stl\(([^)]+)\)", line)
    if stl_m !== nothing
        stl_source = _parse_stl_params(stl_m.captures[1])
        return GeometryRegion(name, kind, nothing, stl_source, bc_type, bc_values)
    end

    # Otherwise: condition expression in { ... }
    brace_m = match(r"\{(.+)\}", line)
    brace_m === nothing && throw(ArgumentError("Missing { condition } or stl(...) in: $line"))
    condition_str = strip(String(brace_m.captures[1]))
    condition = parse_kraken_expr(condition_str, user_vars)
    return GeometryRegion(name, kind, condition, nothing, bc_type, bc_values)
end

"""
    _resolve_axisym_face(face) -> String

Alias resolution for `Module axisymmetric` cases. The axisymmetric solver
lives on a 2D `(z, r)` mesh where `z` is the streamwise axis (mapped to the
`x` direction of the underlying D2Q9 grid) and `r` is the radial coordinate
(mapped to `y`). Hence the user-facing aliases:

- `z`     → `x`      (streamwise axis, usually periodic)
- `wall`  → `north`  (outer radial wall at r = R, i.e. y = Ly)
- `axis`  → `south`  (axis of symmetry at r = 0,  i.e. y = 0)
"""
_resolve_axisym_face(face::AbstractString) =
    face == "z"    ? "x"     :
    face == "wall" ? "north" :
    face == "axis" ? "south" : String(face)

"""Parse: Boundary <face> <type>(<params>) or Boundary <axis> periodic"""
function _parse_boundary(line::String, user_vars::Dict{Symbol,Float64};
                         is_axisym::Bool=false)
    # Remove keyword
    rest = strip(replace(line, r"^Boundary\s+" => ""))

    # In axisymmetric mode, rewrite the first token using the (z, r) aliases
    # before the generic face/type parsing runs.
    if is_axisym
        tok_m = match(r"^(\S+)(\s.*)?$", rest)
        if tok_m !== nothing
            first_tok = String(tok_m.captures[1])
            resolved = _resolve_axisym_face(first_tok)
            if resolved != first_tok
                tail = tok_m.captures[2] === nothing ? "" : String(tok_m.captures[2])
                rest = resolved * tail
            end
        end
    end

    # Check for axis periodic shorthand: "Boundary x periodic"
    axis_m = match(r"^(x|y|z)\s+periodic$", rest)
    if axis_m !== nothing
        axis = axis_m.captures[1]
        if axis == "x"
            return [BoundarySetup(:west, :periodic, Dict{Symbol,KrakenExpr}()),
                    BoundarySetup(:east, :periodic, Dict{Symbol,KrakenExpr}())]
        elseif axis == "y"
            return [BoundarySetup(:south, :periodic, Dict{Symbol,KrakenExpr}()),
                    BoundarySetup(:north, :periodic, Dict{Symbol,KrakenExpr}())]
        end
    end

    # Parse face name
    face_m = match(r"^(\w+)\s+", rest)
    face_m === nothing && throw(ArgumentError("Cannot parse Boundary face: $line"))
    face = Symbol(face_m.captures[1])
    # Allowed faces:
    #   2D (D2Q9):  west (x=0), east (x=Lx), south (y=0), north (y=Ly)
    #   3D (D3Q19): + bottom (z=0), top (z=Lz); front/back kept as legacy aliases
    # Lattice-specific validation (2D rejects top/bottom) happens post-parse
    # in `_validate_faces_vs_lattice` once the lattice symbol is known.
    face in (:north, :south, :east, :west, :front, :back, :top, :bottom) ||
        throw(ArgumentError("Unknown boundary face '$face'. " *
              "Use north/south/east/west (2D) or +top/bottom (3D)"))

    after_face = strip(rest[face_m.offset + length(face_m.match):end])

    values = Dict{Symbol, KrakenExpr}()

    # `symmetry` is accepted by the parser (emitted on the axis face in
    # axisymmetric cases); the axisymmetric kernel enforces the axis
    # condition internally, so non-axisym runners may treat it as a no-op.
    known_bc_types = (:wall, :velocity, :pressure, :periodic, :outflow,
                      :neumann, :symmetry)

    # Check for type(params) format — find matching parentheses
    type_m = match(r"^(\w+)\(", after_face)
    if type_m !== nothing
        bc_type = Symbol(type_m.captures[1])
        if bc_type ∉ known_bc_types
            sug = _suggest_name(String(type_m.captures[1]), known_bc_types)
            msg = "Unknown boundary type '$bc_type'"
            sug !== nothing && (msg *= " (did you mean: $sug?)")
            throw(ArgumentError(msg))
        end
        paren_start = length(type_m.match)
        params_str = _extract_balanced_parens(after_face, paren_start)
        for kv in _split_params(params_str)
            kv_m = match(r"^\s*(\w+)\s*=\s*(.+)$", kv)
            kv_m === nothing && continue
            k = Symbol(kv_m.captures[1])
            v = strip(String(kv_m.captures[2]))
            values[k] = parse_kraken_expr(v, user_vars)
        end
        # Check for additional params after closing paren (e.g. thermal BC)
        close_idx = paren_start + length(params_str) + 1
        if close_idx < length(after_face)
            extra = strip(after_face[nextind(after_face, close_idx):end])
            _parse_kv_pairs!(values, extra, user_vars)
        end
        return [BoundarySetup(face, bc_type, values)]
    end

    # Simple type without parentheses: "wall" or "wall T = 1.0"
    simple_m = match(r"^(\w+)(.*)", after_face)
    if simple_m !== nothing
        bc_type = Symbol(simple_m.captures[1])
        if bc_type ∉ known_bc_types
            sug = _suggest_name(String(simple_m.captures[1]), known_bc_types)
            msg = "Unknown boundary type '$bc_type'"
            sug !== nothing && (msg *= " (did you mean: $sug?)")
            throw(ArgumentError(msg))
        end
        extra = strip(String(simple_m.captures[2]))
        if !isempty(extra)
            _parse_kv_pairs!(values, extra, user_vars)
        end
        return [BoundarySetup(face, bc_type, values)]
    end

    throw(ArgumentError("Cannot parse Boundary: $line"))
end

"""Extract content between balanced parentheses starting at position `start`."""
function _extract_balanced_parens(s::AbstractString, start::Int)
    depth = 1
    i = nextind(s, start)
    while i <= lastindex(s) && depth > 0
        c = s[i]
        if c == '('
            depth += 1
        elseif c == ')'
            depth -= 1
        end
        depth > 0 && (i = nextind(s, i))
    end
    return s[nextind(s, start):prevind(s, i)]
end

"""Split parameter string by top-level commas (respecting nested parens)."""
function _split_params(s::AbstractString)
    parts = String[]
    current = IOBuffer()
    depth = 0
    for c in s
        if c == '('
            depth += 1
            write(current, c)
        elseif c == ')'
            depth -= 1
            write(current, c)
        elseif c == ',' && depth == 0
            push!(parts, strip(String(take!(current))))
        else
            write(current, c)
        end
    end
    rest = strip(String(take!(current)))
    isempty(rest) || push!(parts, rest)
    return parts
end

"""Parse key = value pairs from a string and add to dict."""
function _parse_kv_pairs!(values::Dict{Symbol,KrakenExpr}, s::AbstractString,
                          user_vars::Dict{Symbol,Float64})
    for pm in eachmatch(r"(\w+)\s*=\s*(\S+)", s)
        k = Symbol(pm.captures[1])
        v = strip(String(pm.captures[2]))
        values[k] = parse_kraken_expr(v, user_vars)
    end
end

"""Parse: Initial { field = expr ... }"""
function _parse_initial(line::String, user_vars::Dict{Symbol,Float64})
    brace_m = match(r"\{(.+)\}", line)
    brace_m === nothing && throw(ArgumentError("Missing { ... } in Initial: $line"))
    content = strip(String(brace_m.captures[1]))

    fields = Dict{Symbol, KrakenExpr}()
    for m in eachmatch(r"(\w+)\s*=\s*([^=]+?)(?=\s+\w+\s*=|$)", content)
        k = Symbol(m.captures[1])
        v = strip(String(m.captures[2]))
        fields[k] = parse_kraken_expr(v, user_vars)
    end

    return InitialSetup(fields)
end

"""Parse: Module <name>"""
function _parse_module(line::String)
    tokens = split(line)
    length(tokens) < 2 && throw(ArgumentError("Module needs a name: $line"))
    return Symbol(tokens[2])
end

"""
Parse: Rheology [phase] <model> { key = value ... }
Examples:
    Rheology power_law { K = 0.1  n = 0.5 }
    Rheology liquid  power_law   { K = 0.1  n = 0.5  nu_min = 1e-6 }
    Rheology gas     newtonian   { nu = 0.01 }
"""
function _parse_rheology(line::String, user_vars::Dict{Symbol,Float64}=Dict{Symbol,Float64}())
    # Extract brace block if present
    params = Dict{Symbol, Float64}()
    brace_m = match(r"\{(.+)\}", line)
    if brace_m !== nothing
        for m in eachmatch(r"(\w+)\s*=\s*([\w.eE+\-*/()]+)", brace_m.captures[1])
            key = Symbol(m.captures[1])
            val_str = strip(String(m.captures[2]))
            val = tryparse(Float64, val_str)
            if val === nothing
                expr = parse_kraken_expr(val_str, user_vars)
                val = Float64(evaluate(expr))
            end
            params[key] = val
        end
    end

    # Parse tokens before the brace: "Rheology [phase] model"
    pre_brace = brace_m !== nothing ? strip(line[1:brace_m.offset-1]) : strip(line)
    tokens = split(pre_brace)

    known_phases = (:liquid, :gas, :default)
    known_models = (:newtonian, :power_law, :carreau, :cross, :bingham,
                    :herschel_bulkley, :oldroyd_b, :fene_p, :saramito)

    if length(tokens) >= 3
        phase = Symbol(tokens[2])
        model = Symbol(tokens[3])
        if phase ∉ known_phases
            # Maybe no phase specified, tokens[2] is the model
            model = Symbol(tokens[2])
            phase = :default
        end
    elseif length(tokens) == 2
        phase = :default
        model = Symbol(tokens[2])
    else
        throw(ArgumentError("Rheology needs at least a model name: $line"))
    end

    model ∉ known_models && throw(ArgumentError("Unknown rheology model '$model'. Known: $known_models"))

    return RheologySetup(phase, model, params)
end

"""
    build_rheology_model(rs::RheologySetup; T=Float64) → AbstractRheology

Instantiate a concrete rheology model from a parsed `RheologySetup`.
"""
function build_rheology_model(rs::RheologySetup; FT=Float64)
    p = rs.params
    g = (k, default) -> FT(get(p, k, default))

    # Thermal coupling
    thermal = if haskey(p, :E_a)
        ArrheniusCoupling(g(:T_ref, 1.0), g(:E_a, 0.0))
    elseif haskey(p, :C1)
        WLFCoupling(g(:T_ref, 1.0), g(:C1, 8.86), g(:C2, 101.6))
    else
        IsothermalCoupling()
    end

    if rs.model == :newtonian
        return Newtonian(g(:nu, 0.1); thermal=thermal)
    elseif rs.model == :power_law
        return PowerLaw(g(:K, 0.1), g(:n, 1.0);
                        nu_min=g(:nu_min, 1e-6), nu_max=g(:nu_max, 10.0), thermal=thermal)
    elseif rs.model == :carreau
        return CarreauYasuda(g(:eta_0, 1.0), g(:eta_inf, 0.01), g(:lambda, 1.0),
                             g(:a, 2.0), g(:n, 0.5); thermal=thermal)
    elseif rs.model == :cross
        return Cross(g(:eta_0, 1.0), g(:eta_inf, 0.01), g(:K, 1.0), g(:m, 1.0);
                     thermal=thermal)
    elseif rs.model == :bingham
        return Bingham(g(:tau_y, 0.1), g(:mu_p, 0.05);
                       m_reg=g(:m_reg, 1000.0), thermal=thermal)
    elseif rs.model == :herschel_bulkley
        return HerschelBulkley(g(:tau_y, 0.1), g(:K, 0.1), g(:n, 0.5);
                               m_reg=g(:m_reg, 1000.0), thermal=thermal)
    elseif rs.model == :oldroyd_b
        form = haskey(p, :formulation) && p[:formulation] == 0.0 ? StressFormulation() : LogConfFormulation()
        return OldroydB(g(:nu_s, 0.1), g(:nu_p, 0.05), g(:lambda, 1.0);
                        formulation=form, thermal=thermal)
    elseif rs.model == :fene_p
        form = haskey(p, :formulation) && p[:formulation] == 0.0 ? StressFormulation() : LogConfFormulation()
        return FENEP(g(:nu_s, 0.1), g(:nu_p, 0.05), g(:lambda, 1.0), g(:L_max, 100.0);
                     formulation=form, thermal=thermal)
    elseif rs.model == :saramito
        form = haskey(p, :formulation) && p[:formulation] == 0.0 ? StressFormulation() : LogConfFormulation()
        return Saramito(g(:nu_s, 0.1), g(:nu_p, 0.05), g(:lambda, 1.0), g(:tau_y, 0.01);
                        n=g(:n, 1.0), m_reg=g(:m_reg, 1000.0),
                        formulation=form, thermal=thermal)
    else
        throw(ArgumentError("Unimplemented rheology model: $(rs.model)"))
    end
end

"""Parse: Run <N> steps"""
function _parse_run(line::String)
    m = match(r"(\d+)", line)
    m === nothing && throw(ArgumentError("Cannot parse Run: $line"))
    return parse(Int, m.captures[1])
end

"""Parse: Output <format> every <N> [field1, field2, ...]"""
function _parse_output(line::String)
    # Format
    fmt_m = match(r"^Output\s+(\w+)", line)
    fmt_m === nothing && throw(ArgumentError("Cannot parse Output format: $line"))
    format = Symbol(fmt_m.captures[1])

    # Interval
    int_m = match(r"every\s+(\d+)", line)
    int_m === nothing && throw(ArgumentError("Cannot parse Output interval: $line"))
    interval = parse(Int, int_m.captures[1])

    # Fields: [field1, field2, ...]
    fields_m = match(r"\[([^\]]+)\]", line)
    fields = Symbol[]
    if fields_m !== nothing
        for f in split(fields_m.captures[1], r"[,\s]+")
            s = strip(f)
            isempty(s) || push!(fields, Symbol(s))
        end
    end

    # Directory (optional, default "output/")
    dir_m = match(r"\[([^\]]*)\].*\[([^\]]*)\]", line)
    # Try: Output vtk every 1000 [/path/to/dir] [fields]
    # Or simpler: directory is just "output/" by default
    directory = "output/"

    return OutputSetup(format, interval, fields, directory)
end

"""Parse: Diagnostics every <N> [col1, col2, ...]"""
function _parse_diagnostics(line::String)
    int_m = match(r"every\s+(\d+)", line)
    int_m === nothing && throw(ArgumentError("Cannot parse Diagnostics interval: $line"))
    interval = parse(Int, int_m.captures[1])

    fields_m = match(r"\[([^\]]+)\]", line)
    columns = Symbol[]
    if fields_m !== nothing
        for f in split(fields_m.captures[1], r"[,\s]+")
            s = strip(f)
            isempty(s) || push!(columns, Symbol(s))
        end
    end

    return DiagnosticsSetup(interval, columns)
end

"""Parse STL parameters from stl(file = "...", scale = ..., translate = [...], z_slice = ...)"""
function _parse_stl_params(params_str::AbstractString)
    # Extract file path (quoted string)
    file_m = match(r"""file\s*=\s*"([^"]+)""", params_str)
    file_m === nothing && throw(ArgumentError(
        "STL source requires file parameter: stl(file = \"path.stl\")"))
    file = String(file_m.captures[1])

    # Optional: scale
    scale = 1.0
    scale_m = match(r"scale\s*=\s*([\d.eE+-]+)", params_str)
    scale_m !== nothing && (scale = parse(Float64, scale_m.captures[1]))

    # Optional: translate = [x, y, z]
    translate = (0.0, 0.0, 0.0)
    tr_m = match(r"translate\s*=\s*\[\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\]", params_str)
    if tr_m !== nothing
        translate = (parse(Float64, tr_m.captures[1]),
                     parse(Float64, tr_m.captures[2]),
                     parse(Float64, tr_m.captures[3]))
    end

    # Optional: z_slice (for 2D cross-section)
    z_slice = 0.0
    zs_m = match(r"z_slice\s*=\s*([\d.eE+-]+)", params_str)
    zs_m !== nothing && (z_slice = parse(Float64, zs_m.captures[1]))

    return STLSource(file, scale, translate, z_slice)
end

# =============================================================================
# Phase 2 helpers: Setup directive, Presets, Sanity, Sweeps, spell-correction
# =============================================================================
#
# These helpers extend the .krk DSL with convenience directives:
#   - `Setup reynolds = 1000 [L_ref = ...] [U_ref = ...]`
#   - `Setup rayleigh = 1e5 prandtl = 0.71`
#   - `Preset <name>`  (cavity_2d, poiseuille_2d, couette_2d, taylor_green_2d,
#                        rayleigh_benard_2d)
#   - `Sweep param = [a, b, c]` (expands into multiple SimulationSetups)
# Sanity checks (tau, Mach) run at parse time.
# Unknown identifiers get Levenshtein-based "did you mean?" suggestions.

"""
    _levenshtein(a::AbstractString, b::AbstractString) -> Int

Compute Levenshtein edit distance between two strings.
"""
function _levenshtein(a::AbstractString, b::AbstractString)
    m, n = length(a), length(b)
    m == 0 && return n
    n == 0 && return m
    av = collect(a)
    bv = collect(b)
    prev = collect(0:n)
    curr = zeros(Int, n + 1)
    for i in 1:m
        curr[1] = i
        for j in 1:n
            cost = av[i] == bv[j] ? 0 : 1
            curr[j+1] = min(curr[j] + 1, prev[j+1] + 1, prev[j] + cost)
        end
        prev, curr = curr, prev
    end
    return prev[n + 1]
end

"""
    _suggest_name(name, candidates) -> Union{String, Nothing}

Return the closest candidate by Levenshtein distance if within threshold
(distance ≤ max(2, length(name) ÷ 3)), otherwise `nothing`.
"""
function _suggest_name(name::AbstractString, candidates)
    best = nothing
    best_d = typemax(Int)
    for c in candidates
        d = _levenshtein(lowercase(String(name)), lowercase(String(c)))
        if d < best_d
            best_d = d
            best = String(c)
        end
    end
    threshold = max(2, length(name) ÷ 3)
    return best_d <= threshold ? best : nothing
end

"""
    _parse_setup(line, user_vars) -> Dict{Symbol,Float64}

Parse a `Setup key = value ...` directive. Known keys:
`reynolds`, `rayleigh`, `prandtl`, `L_ref`, `U_ref`.
"""
function _parse_setup(line::String, user_vars::Dict{Symbol,Float64})
    out = Dict{Symbol, Float64}()
    known = (:reynolds, :rayleigh, :prandtl, :L_ref, :U_ref)
    for m in eachmatch(r"(\w+)\s*=\s*([\w.eE+\-*/()]+)", line)
        m.captures[1] == "Setup" && continue
        key = Symbol(m.captures[1])
        if key ∉ known
            sug = _suggest_name(String(m.captures[1]), known)
            msg = "Unknown Setup key '$key'"
            sug !== nothing && (msg *= " (did you mean: $sug?)")
            throw(ArgumentError(msg))
        end
        val_str = strip(String(m.captures[2]))
        val = tryparse(Float64, val_str)
        if val === nothing
            val = Float64(evaluate(parse_kraken_expr(val_str, user_vars)))
        end
        out[key] = val
    end
    return out
end

"""
    _apply_setup_helpers!(physics_params, helpers, domain, boundaries)

Mutate `physics_params` to add auto-computed `nu`, `alpha`, `gbeta_DT` from
`reynolds`/`rayleigh`/`prandtl` helpers. Errors if conflicts exist.
"""
function _apply_setup_helpers!(physics_params::Dict{Symbol,Float64},
                               helpers::Dict{Symbol,Float64},
                               domain,
                               boundaries::Vector{BoundarySetup})
    isempty(helpers) && return

    # Determine L_ref: explicit > domain min(Nx, Ny) in lattice units
    L_ref = get(helpers, :L_ref, Float64(min(domain.Nx, domain.Ny)))

    # Determine U_ref: explicit > probe a velocity BC > default 0.1
    U_ref = get(helpers, :U_ref, _probe_U_ref(boundaries))

    if haskey(helpers, :reynolds)
        Re = helpers[:reynolds]
        if haskey(physics_params, :nu)
            throw(ArgumentError(
                "Setup reynolds conflicts with Physics nu (both specified). " *
                "Remove one: either use `Setup reynolds = $Re` or `Physics nu = ...`."))
        end
        ν = U_ref * L_ref / Re
        physics_params[:nu] = ν
        physics_params[:Re] = Re
    end

    if haskey(helpers, :rayleigh)
        Ra = helpers[:rayleigh]
        Pr = get(helpers, :prandtl, get(physics_params, :Pr, 0.71))
        # Standard scaling: ν = U_ref * L_ref / sqrt(Ra/Pr), α = ν/Pr,
        #                   gβΔT = Ra * ν * α / L_ref^3
        # Using U_ref = sqrt(gβΔT L) gives ν = sqrt(Pr/Ra) * U_ref * L
        ν_ra = sqrt(Pr / Ra) * U_ref * L_ref
        α_ra = ν_ra / Pr
        gβΔT = Ra * ν_ra * α_ra / L_ref^3
        if haskey(physics_params, :nu)
            throw(ArgumentError(
                "Setup rayleigh conflicts with Physics nu (both specified)."))
        end
        physics_params[:nu] = ν_ra
        physics_params[:alpha] = α_ra
        physics_params[:gbeta_DT] = gβΔT
        physics_params[:Ra] = Ra
        physics_params[:Pr] = Pr
    end

    return
end

"""
    _validate_faces_vs_lattice(setup)

Reject 3D-only face names (`:top`, `:bottom`) when the lattice is D2Q9.
Allowed:
- D2Q9:  `:north`, `:south`, `:east`, `:west` (+ `:front`/`:back` legacy aliases)
- D3Q19: all of the above + `:top`, `:bottom`

Axisymmetric cases (`Module axisymmetric`) express boundaries with the
(z, r) aliases `z`/`wall`/`axis` in the user-facing .krk text; those are
rewritten to `x`/`north`/`south` in `_parse_boundary` before this check
runs, so by the time we get here only standard face symbols remain.
"""
function _validate_faces_vs_lattice(setup::SimulationSetup)
    d2_faces = (:north, :south, :east, :west, :front, :back)
    if setup.lattice == :D2Q9
        for b in setup.boundaries
            if !(b.face in d2_faces)
                throw(ArgumentError(
                    "Boundary face ':$(b.face)' is not valid for D2Q9. " *
                    "2D face names are: north/south/east/west."))
            end
        end
    end
    return nothing
end

"""Scan boundary conditions for a velocity BC and return its magnitude, or 0.1."""
function _probe_U_ref(boundaries::Vector{BoundarySetup})
    for b in boundaries
        if b.type == :velocity
            ux = 0.0; uy = 0.0
            if haskey(b.values, :ux)
                try; ux = Float64(evaluate(b.values[:ux])); catch; end
            end
            if haskey(b.values, :uy)
                try; uy = Float64(evaluate(b.values[:uy])); catch; end
            end
            mag = sqrt(ux^2 + uy^2)
            mag > 0 && return mag
        end
    end
    return 0.1
end

"""
    sanity_check(setup::SimulationSetup) -> Nothing

Validate LBM stability parameters for a parsed setup.
Emits `@warn` for soft issues and throws `ErrorException` for tau < 0.5.

Checks:
- `tau < 0.5`  → error (unstable)
- `tau < 0.51` → warn (marginally stable)
- `U_ref > 0.1` → warn (compressibility; standard LBM bound)
- `U_ref > 0.3` → warn (CFL risk)

Note: the compressibility bound is expressed on `U_ref` (lattice units)
rather than on `Ma = U_ref / cs`, because the textbook bound U_ref ≤ 0.1
maps to Ma ≤ 0.1*sqrt(3) ≈ 0.173, not Ma ≤ 0.1.
"""
function sanity_check(setup::SimulationSetup)
    ν = get(setup.physics.params, :nu, NaN)
    if !isnan(ν)
        τ = 3ν + 0.5
        if τ < 0.5
            error("Sanity check failed: tau = $τ < 0.5 (unstable). " *
                  "To fix: increase ν (current $ν), or use Setup reynolds with " *
                  "a larger L_ref / smaller Re.")
        elseif τ < 0.51
            @warn "tau = $τ is very close to 0.5 (marginally stable). " *
                  "To fix: increase ν from $ν, or reduce Re / increase grid N " *
                  "from $(setup.domain.Nx)."
        end
    end

    U_ref = _probe_U_ref(setup.boundaries)
    # Standard LBM compressibility bound is U_ref ≤ 0.1 in lattice units
    # (textbook, e.g. Krüger et al. 2017). In Mach-number space this is
    # Ma = U_ref/cs ≤ 0.1*sqrt(3) ≈ 0.173 — NOT Ma ≤ 0.1. We therefore
    # compare U_ref against 0.1 directly to avoid a false-positive that
    # fired on every default case where U_ref fell back to 0.1 exactly.
    if U_ref > 0.1
        cs = 1.0 / sqrt(3.0)
        Ma = U_ref / cs
        @warn "Lattice velocity U_ref = $U_ref exceeds 0.1 " *
              "(Mach number Ma = $Ma, compressibility errors likely). " *
              "To fix: decrease U_ref to ≤ 0.1, " *
              "or increase grid N to reduce lattice velocity."
    end
    if U_ref > 0.3
        @warn "Lattice velocity U_ref = $U_ref > 0.3 (CFL risk). " *
              "To fix: decrease U_ref, or increase N from $(setup.domain.Nx)."
    end
    return nothing
end

"""
    _parse_sweep(line) -> Pair{Symbol, Vector{Float64}}

Parse `Sweep param = [a, b, c]` into a (name, values) pair.
"""
function _parse_sweep(line::String)
    m = match(r"^Sweep\s+(\w+)\s*=\s*\[([^\]]+)\]", line)
    m === nothing && throw(ArgumentError("Cannot parse Sweep: $line"))
    key = Symbol(m.captures[1])
    vals = Float64[]
    for s in split(m.captures[2], ",")
        t = strip(s)
        isempty(t) && continue
        push!(vals, parse(Float64, t))
    end
    return key => vals
end

"""
    _expand_preset(line) -> Vector{String}

Expand a `Preset <name>` directive into a list of .krk lines.
Known presets: `cavity_2d`, `poiseuille_2d`, `couette_2d`,
`taylor_green_2d`, `rayleigh_benard_2d`.
"""
function _expand_preset(line::String)
    tokens = split(line)
    length(tokens) >= 2 || throw(ArgumentError("Preset needs a name: $line"))
    name = tokens[2]
    known = ("cavity_2d", "poiseuille_2d", "couette_2d",
             "taylor_green_2d", "rayleigh_benard_2d")
    if name ∉ known
        sug = _suggest_name(name, known)
        msg = "Unknown Preset '$name'"
        sug !== nothing && (msg *= " (did you mean: $sug?)")
        throw(ArgumentError(msg))
    end
    return _preset_lines(name)
end

function _preset_lines(name::AbstractString)
    if name == "cavity_2d"
        return [
            "Simulation cavity_2d D2Q9",
            "Domain L = 1.0 x 1.0  N = 128 x 128",
            "Physics nu = 0.01",
            "Boundary north velocity(ux = 0.1, uy = 0)",
            "Boundary south wall",
            "Boundary east wall",
            "Boundary west wall",
            "Run 10000 steps",
        ]
    elseif name == "poiseuille_2d"
        return [
            "Simulation poiseuille_2d D2Q9",
            "Domain L = 4.0 x 1.0  N = 64 x 32",
            "Physics nu = 0.1 Fx = 1e-5",
            "Boundary x periodic",
            "Boundary south wall",
            "Boundary north wall",
            "Run 10000 steps",
        ]
    elseif name == "couette_2d"
        return [
            "Simulation couette_2d D2Q9",
            "Domain L = 1.0 x 1.0  N = 32 x 64",
            "Physics nu = 0.1",
            "Boundary x periodic",
            "Boundary south wall",
            "Boundary north velocity(ux = 0.05, uy = 0)",
            "Run 5000 steps",
        ]
    elseif name == "taylor_green_2d"
        return [
            "Simulation taylor_green_2d D2Q9",
            "Domain L = 1.0 x 1.0  N = 64 x 64",
            "Physics nu = 0.01",
            "Boundary x periodic",
            "Boundary y periodic",
            "Initial { ux = 0.05*sin(2*pi*x)*cos(2*pi*y) uy = -0.05*cos(2*pi*x)*sin(2*pi*y) }",
            "Run 5000 steps",
        ]
    elseif name == "rayleigh_benard_2d"
        return [
            "Simulation rayleigh_benard_2d D2Q9",
            "Domain L = 2.0 x 1.0  N = 128 x 64",
            "Physics nu = 0.02 Pr = 0.71 Ra = 1e5",
            "Module thermal",
            "Boundary x periodic",
            "Boundary south wall T = 1.0",
            "Boundary north wall T = 0.0",
            "Run 20000 steps",
        ]
    end
    return String[]
end

