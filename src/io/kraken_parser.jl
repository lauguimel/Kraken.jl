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
    fps::Int             # frames per second (used by :gif format, default 10)
end

# Convenience constructor without fps (backward compat)
OutputSetup(format, interval, fields, directory) =
    OutputSetup(format, interval, fields, directory, 10)

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
    outputs::Vector{OutputSetup}
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
    outputs = OutputSetup[]
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
            push!(outputs, _parse_output(line))
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
                            max_steps, outputs, diagnostics, refinements,
                            velocity_field, rheology_setups)

    # --- Validate face names against lattice dimensionality ---
    _validate_faces_vs_lattice(setup)

    # --- Run sanity checks (no summary at parse time — printed at run time) ---
    sanity_check(setup; verbose=false)

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

"""Parse: Output <format> every <N> [field1, field2, ...] [fps=<N>]"""
function _parse_output(line::String)
    # Format
    fmt_m = match(r"^Output\s+(\w+)", line)
    fmt_m === nothing && throw(ArgumentError("Cannot parse Output format: $line"))
    format = Symbol(fmt_m.captures[1])
    format in (:vtk, :png, :gif) || throw(ArgumentError(
        "Unknown Output format '$format'. Use vtk, png, or gif."))

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

    # fps parameter (optional, for gif format)
    fps = 10
    fps_m = match(r"fps\s*=\s*(\d+)", line)
    if fps_m !== nothing
        fps = parse(Int, fps_m.captures[1])
    end

    # Directory (optional, default "output/")
    directory = "output/"

    return OutputSetup(format, interval, fields, directory, fps)
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

# ══════════════════════════════════════════════════════════════════════
#  LBM parameter calculator / advisor
# ══════════════════════════════════════════════════════════════════════

"""
    LBMParams

Computed lattice-Boltzmann parameters with feasibility assessment.
Returned by [`lbm_params`](@ref).
"""
struct LBMParams
    # --- Inputs (lattice units) ---
    Re::Float64
    N::Int
    U_ref::Float64
    # --- Derived ---
    nu::Float64       # lattice viscosity
    tau::Float64      # relaxation time
    omega::Float64    # relaxation rate
    Ma::Float64       # Mach number
    # --- Quality assessment ---
    feasible::Bool           # all hard constraints satisfied
    regime::Symbol           # :optimal, :acceptable, :marginal, :diffusive, :infeasible
    warnings::Vector{String} # human-readable diagnostics
    # --- Recommendations ---
    recommended_N::Int           # best N for this Re at default U_ref
    recommended_U_ref::Float64   # best U_ref for this Re and N
end

function Base.show(io::IO, p::LBMParams)
    status = p.feasible ? "✓ feasible" : "✗ INFEASIBLE"
    println(io, "LBMParams ($status, regime = $(p.regime))")
    println(io, "  Re = $(p.Re), N = $(p.N), U_ref = $(round(p.U_ref, digits=6))")
    println(io, "  ν  = $(round(p.nu, digits=6)), τ = $(round(p.tau, digits=4)), ω = $(round(p.omega, digits=6))")
    println(io, "  Ma = $(round(p.Ma, digits=4))")
    if !isempty(p.warnings)
        println(io, "  Diagnostics:")
        for w in p.warnings
            println(io, "    ⚠ ", w)
        end
    end
    if !p.feasible || p.regime in (:diffusive, :marginal)
        # Compute τ for each recommendation
        τ_rec_N = 3.0 * 0.01 * p.recommended_N / p.Re + 0.5
        τ_rec_U = 3.0 * p.recommended_U_ref * p.N / p.Re + 0.5
        println(io, "  Recommendations:")
        println(io, "    → N = $(p.recommended_N) at U_ref = 0.01 → τ = $(round(τ_rec_N, digits=3))")
        println(io, "    → U_ref = $(round(p.recommended_U_ref, digits=6)) at N = $(p.N) → τ = $(round(τ_rec_U, digits=3))")
        # Low-Re advisory
        if p.Re < 0.1
            println(io, "  Note: Re = $(p.Re) < 0.1 — Stokes regime.")
            println(io, "    LBM is poorly suited for very low Re (τ grows as 1/Re).")
            println(io, "    Best achievable τ at N=10: $(round(3.0 * 0.058 * 10 / p.Re + 0.5, digits=1))")
        end
    end
end

# τ regime thresholds
const _TAU_UNSTABLE   = 0.5
const _TAU_MARGINAL   = 0.51
const _TAU_OPTIMAL_HI = 2.0
const _TAU_ACCEPT_HI  = 10.0
const _TAU_ABSURD     = 100.0

# Target τ values for recommendations (best → fallback)
const _TAU_TARGETS = [1.0, 1.5, 2.0, 5.0, 10.0]

"""Recommend best N for given (Re, U_ref), trying τ targets from ideal to fallback."""
function _recommend_N(Re, U_ref, N_min)
    for τ_t in _TAU_TARGETS
        N_rec = ceil(Int, Re * (τ_t - 0.5) / (3.0 * U_ref))
        N_rec >= N_min && return N_rec
    end
    return N_min  # best effort
end

"""Recommend best U_ref for given (Re, N), trying τ targets from ideal to fallback."""
function _recommend_U(Re, N, U_min, U_max)
    for τ_t in _TAU_TARGETS
        U_rec = Re * (τ_t - 0.5) / (3.0 * N)
        U_min ≤ U_rec ≤ U_max && return U_rec
    end
    # If all targets give U outside bounds, clamp to nearest feasible
    U_for_tau10 = Re * (_TAU_TARGETS[end] - 0.5) / (3.0 * N)
    return clamp(U_for_tau10, U_min, U_max)
end

"""
    lbm_params(; Re, N, U_ref=0.01, L_ref=N)

Compute all LBM lattice parameters from physical inputs and assess feasibility.

Returns an [`LBMParams`](@ref) with derived quantities, regime classification,
diagnostics, and concrete recommendations.

# Regimes (based on τ = 3ν + 0.5, where ν = U_ref × L_ref / Re)

| Regime       | τ range       | Meaning                                    |
|:-------------|:--------------|:-------------------------------------------|
| `:optimal`   | 0.51 – 2.0    | BGK accurate, best precision               |
| `:acceptable`| 2.0 – 10.0    | OK with MRT, some numerical diffusion       |
| `:marginal`  | 0.5 – 0.51    | Nearly unstable, MRT mandatory              |
| `:diffusive` | 10.0 – 100.0  | Collision quasi-inactive, MRT mandatory     |
| `:infeasible`| < 0.5 or > 100| Cannot run — parameters must change         |

# Additional constraints checked
- **Mach number**: Ma = U_ref × √3 ≤ 0.17 (compressibility)
- **CFL**: U_ref ≤ 0.3
- **Float32 precision**: U_ref ≥ 1e-5
- **Resolution**: N ≥ 10

# Examples
```julia
julia> lbm_params(Re=100, N=128)         # standard case
julia> lbm_params(Re=0.01, N=64)         # your problematic case
julia> lbm_params(Re=0.01, N=64, U_ref=0.001)  # with adjusted velocity
```
"""
function lbm_params(; Re::Real, N::Integer, U_ref::Real=0.01, L_ref::Real=N)
    Re = Float64(Re)
    N = Int(N)
    U_ref = Float64(U_ref)
    L_ref = Float64(L_ref)

    # --- Derived quantities ---
    ν   = U_ref * L_ref / Re
    τ   = 3ν + 0.5
    ω   = 1.0 / τ
    cs  = 1.0 / sqrt(3.0)
    Ma  = U_ref / cs

    warnings = String[]
    feasible = true

    # --- Regime classification ---
    regime = if τ < _TAU_UNSTABLE
        feasible = false
        push!(warnings, "τ = $(round(τ, digits=4)) < 0.5 — UNSTABLE (negative effective viscosity)")
        :infeasible
    elseif τ < _TAU_MARGINAL
        push!(warnings, "τ = $(round(τ, digits=4)) ≈ 0.5 — marginally stable, MRT mandatory")
        :marginal
    elseif τ ≤ _TAU_OPTIMAL_HI
        :optimal
    elseif τ ≤ _TAU_ACCEPT_HI
        push!(warnings, "τ = $(round(τ, digits=2)) — numerical diffusion significant, MRT recommended")
        :acceptable
    elseif τ ≤ _TAU_ABSURD
        push!(warnings, "τ = $(round(τ, digits=1)) — collision quasi-inactive (ω = $(round(ω, digits=5))), " *
                        "only $(round(ω*100, digits=1))% relaxation per step")
        :diffusive
    else
        feasible = false
        push!(warnings, "τ = $(round(τ, digits=0)) — absurd, advective physics dead " *
                        "(ω = $(round(ω, digits=6)))")
        :infeasible
    end

    # --- Mach / compressibility ---
    if Ma > 0.3 * sqrt(3.0)
        feasible = false
        push!(warnings, "Ma = $(round(Ma, digits=3)) > 0.52 — CFL violation, will diverge")
    elseif Ma > 0.1 * sqrt(3.0)
        push!(warnings, "Ma = $(round(Ma, digits=3)) > 0.17 — compressibility errors > 1%")
    end

    # --- Float32 precision ---
    if U_ref < 1e-5 && U_ref > 0
        push!(warnings, "U_ref = $(U_ref) — round-off dominates in Float32 " *
                        "(relative precision ~ $(round(eps(Float32)/U_ref, digits=0)))")
    elseif U_ref < 1e-3 && U_ref > 0
        push!(warnings, "U_ref = $(U_ref) — use Float64 for best accuracy")
    end

    # --- Resolution ---
    if N < 10
        push!(warnings, "N = $N — insufficient spatial resolution")
    end
    if Re > 0 && N / Re < 1.0
        push!(warnings, "N/Re = $(round(N/Re, digits=2)) < 1 — boundary layer under-resolved")
    end

    # --- Recommendations ---
    # Goal: find (U_ref, N) that gives τ in optimal range [0.55, 2.0]
    # Constraints: Ma ≤ 0.1 (U ≤ 0.058), U ≥ 1e-5, N ≥ 10
    #
    # τ = 3·U·N/Re + 0.5  ⟹  U·N = Re·(τ-0.5)/3
    #
    # Strategy: target τ = 1.0 (ideal). If that requires U < 1e-5 or N < 10,
    # relax τ target upward until feasible, capping at τ = 10 (MRT acceptable).
    U_max = 0.058  # Ma ≈ 0.1
    U_min = 1e-5
    N_min_rec = 10

    # For recommended_N: fix U = min(0.01, U_max) and solve N
    rec_U_fixed = min(0.01, U_max)
    rec_N = _recommend_N(Re, rec_U_fixed, N_min_rec)

    # For recommended_U: fix N and solve U
    rec_U_for_N = _recommend_U(Re, N, U_min, U_max)

    return LBMParams(Re, N, U_ref, ν, τ, ω, Ma, feasible, regime,
                     warnings, rec_N, rec_U_for_N)
end

"""
    lbm_params(setup::SimulationSetup)

Extract parameters from a parsed `.krk` setup and compute [`LBMParams`](@ref).
"""
function lbm_params(setup::SimulationSetup)
    params = setup.physics.params
    ν = get(params, :nu, NaN)
    isnan(ν) && error("No viscosity (nu) in setup — cannot compute LBM parameters.")
    Re = get(params, :Re, NaN)
    N = min(setup.domain.Nx, setup.domain.Ny)
    U_ref = _probe_U_ref(setup.boundaries)

    # If Re not stored, recompute from ν
    if isnan(Re) && U_ref > 0
        Re = U_ref * N / ν
    end

    return lbm_params(; Re=Re, N=N, U_ref=U_ref, L_ref=N)
end

"""
    lbm_params_table(; Re, N_range, U_ref=0.01)

Print a comparison table for multiple grid sizes at a given Re.

# Example
```julia
lbm_params_table(Re=0.01, N_range=[8, 16, 32, 64, 128])
```
"""
function lbm_params_table(; Re::Real, N_range, U_ref::Real=0.01)
    Re = Float64(Re)
    U_ref = Float64(U_ref)
    cs = 1.0 / sqrt(3.0)
    Ma = U_ref / cs

    println("LBM parameter space for Re = $Re, U_ref = $U_ref (Ma = $(round(Ma, digits=4)))")
    println("─"^78)
    println(rpad("N", 6), rpad("ν", 12), rpad("τ", 10), rpad("ω", 12),
            rpad("regime", 14), "status")
    println("─"^78)

    for N in N_range
        p = lbm_params(; Re=Re, N=Int(N), U_ref=U_ref)
        status = p.feasible ? "✓" : "✗"
        nw = length(p.warnings)
        extra = nw > 0 && p.feasible ? " ($(nw) warning$(nw>1 ? "s" : ""))" : ""
        println(rpad(N, 6),
                rpad(round(p.nu, digits=6), 12),
                rpad(round(p.tau, digits=4), 10),
                rpad(round(p.omega, digits=6), 12),
                rpad(p.regime, 14),
                status, extra)
    end

    println("─"^78)
    # Show best achievable configuration
    rec = lbm_params(; Re=Re, N=64, U_ref=U_ref)
    rec_τ = 3.0 * 0.01 * rec.recommended_N / Re + 0.5
    rec_regime = rec_τ < 0.51 ? "marginal" : rec_τ ≤ 2.0 ? "optimal" :
                 rec_τ ≤ 10.0 ? "acceptable" : rec_τ ≤ 100.0 ? "diffusive" : "infeasible"
    println("Best config: N = $(rec.recommended_N), U_ref = 0.01 → τ = $(round(rec_τ, digits=3)) ($rec_regime)")
    if Re < 0.1
        # Show the physical limit
        τ_min_possible = 3.0 * 0.058 * 10 / Re + 0.5  # N=10, Ma=0.1
        println("Low-Re limit: Re = $Re < 0.1, best possible τ ≈ $(round(τ_min_possible, digits=1)) (N=10, Ma=0.1)")
    end
end

"""
    SanityIssue

A single validation issue found by `sanity_check`.
`level` is `:error`, `:warn`, or `:info`.
"""
struct SanityIssue
    level::Symbol      # :error, :warn, :info
    category::Symbol   # :relaxation, :compressibility, :resolution, :thermal, :twophase, :rheology, :refinement
    message::String
end

function _push_issue!(issues, level, category, msg)
    push!(issues, SanityIssue(level, category, msg))
end

# ── 1. Relaxation checks (τ, ω) ──────────────────────────────────────

function _check_relaxation!(issues, setup)
    ν = get(setup.physics.params, :nu, NaN)
    isnan(ν) && return
    τ = 3ν + 0.5
    ω = 1.0 / τ
    if τ < 0.5
        _push_issue!(issues, :error, :relaxation,
            "tau = $(round(τ, digits=4)) < 0.5 (unstable, ν = $(round(ν, digits=6)) is negative or zero). " *
            "Fix: increase ν, or use Setup reynolds with larger L_ref / smaller Re.")
    elseif τ < 0.51
        _push_issue!(issues, :warn, :relaxation,
            "tau = $(round(τ, digits=4)) is very close to 0.5 (marginally stable). " *
            "MRT collision recommended. Fix: increase ν from $(round(ν, digits=6)), " *
            "or reduce Re / increase N from $(setup.domain.Nx).")
    end
    if τ > 100.0
        _push_issue!(issues, :error, :relaxation,
            "tau = $(round(τ, digits=1)) is absurdly large (ω = $(round(ω, digits=6))). " *
            "Advective physics is dead — collision does almost nothing. " *
            "Fix: decrease ν (=$(round(ν, digits=4))), e.g. increase Re or decrease N.")
    elseif τ > 10.0
        _push_issue!(issues, :warn, :relaxation,
            "tau = $(round(τ, digits=2)) is very large (ω = $(round(ω, digits=4))). " *
            "Collision relaxes only $(round(ω*100, digits=1))% per step — numerical diffusion dominates. " *
            "Fix: decrease ν (=$(round(ν, digits=4))), e.g. increase Re or decrease N.")
    end
end

# ── 2. Compressibility checks (Ma, U_ref) ────────────────────────────

function _check_compressibility!(issues, setup)
    U_ref = _probe_U_ref(setup.boundaries)
    cs = 1.0 / sqrt(3.0)
    if U_ref < 1e-6 && U_ref > 0.0
        _push_issue!(issues, :warn, :compressibility,
            "U_ref = $(U_ref) is near zero — round-off errors will dominate in Float32. " *
            "Fix: increase U_ref or use Float64.")
    end
    # Textbook bound: U_ref ≤ 0.1 (Krüger et al. 2017)
    if U_ref > 0.3
        Ma = U_ref / cs
        _push_issue!(issues, :error, :compressibility,
            "U_ref = $(round(U_ref, digits=4)) > 0.3, Mach = $(round(Ma, digits=3)) (CFL-critical, will diverge). " *
            "Fix: decrease U_ref or increase N from $(setup.domain.Nx).")
    elseif U_ref > 0.1
        Ma = U_ref / cs
        _push_issue!(issues, :warn, :compressibility,
            "U_ref = $(round(U_ref, digits=4)) > 0.1, Mach = $(round(Ma, digits=3)) (compressibility errors likely). " *
            "Fix: decrease U_ref to ≤ 0.1 or increase grid N.")
    end
end

# ── 3. Spatial resolution checks ─────────────────────────────────────

function _check_resolution!(issues, setup)
    N = min(setup.domain.Nx, setup.domain.Ny)
    if N < 10
        _push_issue!(issues, :warn, :resolution,
            "N = $N is very coarse — insufficient for quantitative results.")
    end
    Re = get(setup.physics.params, :Re, NaN)
    if !isnan(Re) && Re > 0 && N / Re < 1.0
        _push_issue!(issues, :warn, :resolution,
            "N/Re = $(round(N/Re, digits=2)) < 1 — boundary layer under-resolved. " *
            "Fix: increase N (currently $N) or reduce Re (currently $(round(Re, digits=1))).")
    end
end

# ── 4. Thermal checks (if :thermal module) ───────────────────────────

function _check_thermal!(issues, setup)
    :thermal in setup.modules || return
    params = setup.physics.params
    α = get(params, :alpha, NaN)
    if isnan(α)
        _push_issue!(issues, :warn, :thermal,
            "Module thermal is active but no thermal diffusivity (alpha) found. " *
            "It may be computed internally, but verify your setup.")
        return
    end
    τ_α = 3α + 0.5
    if τ_α < 0.5
        _push_issue!(issues, :error, :thermal,
            "Thermal tau = $(round(τ_α, digits=4)) < 0.5 (unstable). " *
            "Fix: increase alpha (=$(round(α, digits=6))).")
    elseif τ_α < 0.51
        _push_issue!(issues, :warn, :thermal,
            "Thermal tau = $(round(τ_α, digits=4)) is marginally stable.")
    end
    if τ_α > 10.0
        _push_issue!(issues, :warn, :thermal,
            "Thermal tau = $(round(τ_α, digits=2)) is very large — thermal diffusion too fast for this grid.")
    end
    Pr = get(params, :Pr, NaN)
    if !isnan(Pr) && (Pr < 0.1 || Pr > 10.0)
        _push_issue!(issues, :warn, :thermal,
            "Pr = $(round(Pr, digits=3)) is extreme for SRT thermal LBM. " *
            "MRT collision recommended for accuracy.")
    end
end

# ── 5. Two-phase checks (if :twophase_vof module) ────────────────────

function _check_twophase!(issues, setup)
    :twophase_vof in setup.modules || return
    params = setup.physics.params
    ν   = get(params, :nu, NaN)
    ν_l = get(params, :nu_l, ν)
    ν_g = get(params, :nu_g, ν)
    if !isnan(ν_l) && !isnan(ν_g) && ν_g > 0
        ratio_ν = ν_l / ν_g
        if ratio_ν > 100.0 || ratio_ν < 0.01
            _push_issue!(issues, :warn, :twophase,
                "Viscosity ratio ν_l/ν_g = $(round(ratio_ν, digits=1)) is extreme. " *
                "MRT collision strongly recommended.")
        end
        τ_g = 3ν_g + 0.5
        if τ_g < 0.51
            _push_issue!(issues, :warn, :twophase,
                "Gas-phase tau = $(round(τ_g, digits=4)) is marginally stable.")
        end
        if τ_g > 10.0
            _push_issue!(issues, :warn, :twophase,
                "Gas-phase tau = $(round(τ_g, digits=2)) is very large — over-diffusive gas phase.")
        end
    end
    ρ_l = get(params, :rho_l, NaN)
    ρ_g = get(params, :rho_g, NaN)
    if !isnan(ρ_l) && !isnan(ρ_g) && ρ_g > 0
        ratio_ρ = ρ_l / ρ_g
        if ratio_ρ > 100.0
            _push_issue!(issues, :warn, :twophase,
                "Density ratio ρ_l/ρ_g = $(round(ratio_ρ, digits=0)) > 100. " *
                "Pressure-based model (phase-field) recommended.")
        end
    end
end

# ── 6. Rheology checks ───────────────────────────────────────────────

function _check_rheology!(issues, setup)
    isempty(setup.rheology) && return
    for rs in setup.rheology
        # Estimate minimum viscosity from model bounds
        ν_min = get(rs.params, :nu_min, NaN)
        if !isnan(ν_min)
            τ_min = 3ν_min + 0.5
            if τ_min < 0.51
                _push_issue!(issues, :warn, :rheology,
                    "Rheology model $(rs.model) (phase=$(rs.phase)) has nu_min=$(round(ν_min, digits=6)) " *
                    "→ tau_min=$(round(τ_min, digits=4)). Local instability possible. " *
                    "Fix: increase nu_min or use MRT.")
            end
        end
    end
end

# ── 7. Refinement checks ─────────────────────────────────────────────

function _check_refinement!(issues, setup)
    isempty(setup.refinements) && return
    ν = get(setup.physics.params, :nu, NaN)
    isnan(ν) && return
    τ_base = 3ν + 0.5
    for ref in setup.refinements
        ratio = ref.ratio
        τ_fine = ratio * (τ_base - 0.5) + 0.5
        if τ_fine < 0.51
            _push_issue!(issues, :warn, :refinement,
                "Patch '$(ref.name)' (ratio=$ratio): fine-grid tau = $(round(τ_fine, digits=4)) " *
                "is marginally stable after Filippova-Hanel rescaling.")
        end
        if τ_fine > 10.0
            _push_issue!(issues, :warn, :refinement,
                "Patch '$(ref.name)' (ratio=$ratio): fine-grid tau = $(round(τ_fine, digits=2)) " *
                "is very large — over-diffusive fine grid.")
        end
    end
end

# ── Parameter summary ─────────────────────────────────────────────────

function _print_parameter_summary(setup)
    dom = setup.domain
    params = setup.physics.params
    ν = get(params, :nu, NaN)
    τ = isnan(ν) ? NaN : 3ν + 0.5
    ω = isnan(τ) ? NaN : 1.0 / τ
    Re = get(params, :Re, NaN)
    U_ref = _probe_U_ref(setup.boundaries)
    cs = 1.0 / sqrt(3.0)
    Ma = U_ref / cs

    grid = dom.Nz > 1 ? "$(dom.Nx)×$(dom.Ny)×$(dom.Nz)" : "$(dom.Nx)×$(dom.Ny)"
    mods = isempty(setup.modules) ? "none" : join(string.(setup.modules), ", ")

    lines = String[]
    push!(lines, "N = $grid, lattice = $(setup.lattice)")
    if !isnan(ν)
        push!(lines, "ν = $(round(ν, digits=6)), τ = $(round(τ, digits=4)), ω = $(round(ω, digits=4))")
    end
    re_str = isnan(Re) ? "" : "Re = $(round(Re, digits=2)), "
    push!(lines, "$(re_str)Ma = $(round(Ma, digits=4)), U_ref = $(round(U_ref, digits=4))")
    push!(lines, "Modules: $mods")
    push!(lines, "Steps: $(setup.max_steps)")

    # Refinement patches
    if !isempty(setup.refinements)
        for ref in setup.refinements
            τ_fine = isnan(ν) ? NaN : ref.ratio * (τ - 0.5) + 0.5
            push!(lines, "  Refine '$(ref.name)': ratio=$(ref.ratio), τ_fine=$(round(τ_fine, digits=4))")
        end
    end

    @info "LBM parameters\n  " * join(lines, "\n  ")
end

# ── Issue emission ────────────────────────────────────────────────────

function _emit_issues(issues)
    errors = String[]
    for issue in issues
        if issue.level === :error
            push!(errors, "[$(issue.category)] $(issue.message)")
            @error "Sanity check [$(issue.category)]: $(issue.message)"
        elseif issue.level === :warn
            @warn "Sanity check [$(issue.category)]: $(issue.message)"
        else
            @info "Sanity check [$(issue.category)]: $(issue.message)"
        end
    end
    if !isempty(errors)
        error("Sanity check failed with $(length(errors)) error(s):\n" *
              join(["  • " * e for e in errors], "\n"))
    end
end

# ── Main entry point ──────────────────────────────────────────────────

"""
    sanity_check(setup::SimulationSetup; verbose=true) -> Vector{SanityIssue}

Validate LBM parameters for a parsed setup.

Runs 7 families of checks:
1. **Relaxation** — τ too low (unstable) or too high (diffusion-dominated)
2. **Compressibility** — Mach number / CFL bounds
3. **Resolution** — grid points vs Reynolds number
4. **Thermal** — thermal τ and Prandtl range (if `:thermal` module)
5. **Two-phase** — viscosity/density ratios (if `:twophase_vof` module)
6. **Rheology** — local τ bounds from non-Newtonian models
7. **Refinement** — fine-grid τ after Filippova-Hanel rescaling

Returns a `Vector{SanityIssue}` for programmatic inspection.
Emits `@warn` for soft issues, throws `ErrorException` for critical ones,
and prints a parameter summary when `verbose=true`.
"""
function sanity_check(setup::SimulationSetup; verbose::Bool=true)
    issues = SanityIssue[]

    _check_relaxation!(issues, setup)
    _check_compressibility!(issues, setup)
    _check_resolution!(issues, setup)
    _check_thermal!(issues, setup)
    _check_twophase!(issues, setup)
    _check_rheology!(issues, setup)
    _check_refinement!(issues, setup)

    verbose && _print_parameter_summary(setup)
    _emit_issues(issues)

    return issues
end

"""
    sanity_check_sweep(setups::Vector{SimulationSetup}; verbose=true) -> Vector{Vector{SanityIssue}}

Validate all setups in a sweep. Prints a compact summary table and returns
per-setup issues. Does NOT throw on :error — instead marks them so the caller
can decide whether to skip or abort.
"""
function sanity_check_sweep(setups::Vector{SimulationSetup}; verbose::Bool=true)
    all_issues = Vector{SanityIssue}[]
    rows = String[]

    for (i, setup) in enumerate(setups)
        issues = SanityIssue[]
        _check_relaxation!(issues, setup)
        _check_compressibility!(issues, setup)
        _check_resolution!(issues, setup)
        _check_thermal!(issues, setup)
        _check_twophase!(issues, setup)
        _check_rheology!(issues, setup)
        _check_refinement!(issues, setup)
        push!(all_issues, issues)

        # Build compact row
        params = setup.physics.params
        ν = get(params, :nu, NaN)
        τ = isnan(ν) ? NaN : 3ν + 0.5
        Re = get(params, :Re, NaN)
        U_ref = _probe_U_ref(setup.boundaries)
        n_err = count(i -> i.level == :error, issues)
        n_warn = count(i -> i.level == :warn, issues)
        status = n_err > 0 ? "✗" : n_warn > 0 ? "⚠" : "✓"
        re_str = isnan(Re) ? "—" : string(round(Re, digits=2))
        push!(rows, "$status  #$i  Re=$re_str  τ=$(round(τ, digits=3))  U=$(round(U_ref, digits=4))  " *
                     "err=$n_err warn=$n_warn")
    end

    if verbose
        @info "Sweep sanity check ($(length(setups)) cases)\n  " * join(rows, "\n  ")
        # Emit warnings for problematic cases
        for (i, issues) in enumerate(all_issues)
            for issue in issues
                if issue.level == :error
                    @warn "Case #$i [$(issue.category)]: $(issue.message)"
                end
            end
        end
    end

    return all_issues
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

