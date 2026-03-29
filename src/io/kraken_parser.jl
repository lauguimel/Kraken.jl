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
    lines = _preprocess_lines(text)

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
    modules = Symbol[]
    max_steps = 0
    output = nothing
    diagnostics = nothing

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
            append!(boundaries, _parse_boundary(line, user_vars))
        elseif keyword == "Initial"
            initial = _parse_initial(line, user_vars)
        elseif keyword == "Module"
            push!(modules, _parse_module(line))
        elseif keyword == "Run"
            max_steps = _parse_run(line)
        elseif keyword == "Output"
            output = _parse_output(line)
        elseif keyword == "Diagnostics"
            diagnostics = _parse_diagnostics(line)
        else
            throw(ArgumentError("Unknown keyword '$keyword' in .krk file"))
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

    physics = PhysicsSetup(physics_params, body_force)

    return SimulationSetup(name, lattice, domain, physics, user_vars,
                           regions, boundaries, initial, modules,
                           max_steps, output, diagnostics)
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

"""Parse: Boundary <face> <type>(<params>) or Boundary <axis> periodic"""
function _parse_boundary(line::String, user_vars::Dict{Symbol,Float64})
    # Remove keyword
    rest = strip(replace(line, r"^Boundary\s+" => ""))

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
    face in (:north, :south, :east, :west, :front, :back) ||
        throw(ArgumentError("Unknown boundary face '$face'. Use north/south/east/west"))

    after_face = strip(rest[face_m.offset + length(face_m.match):end])

    values = Dict{Symbol, KrakenExpr}()

    # Check for type(params) format — find matching parentheses
    type_m = match(r"^(\w+)\(", after_face)
    if type_m !== nothing
        bc_type = Symbol(type_m.captures[1])
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
