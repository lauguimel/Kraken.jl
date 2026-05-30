const _PLANNER_OWNED_KEYS = Set{Symbol}((
    :u_mean, :u_LU, :nu_s, :nu_p, :nu_s_LU, :nu_p_LU, :nu_total,
    :nu_total_LU, :lambda, :lambda_LU, :tau, :tau_hydro, :max_steps,
))

function _find_matching_brace(text::AbstractString, open_idx::Int)
    depth = 0
    for idx in open_idx:lastindex(text)
        c = text[idx]
        if c == '{'
            depth += 1
        elseif c == '}'
            depth -= 1
            depth == 0 && return idx
        end
    end
    throw(ArgumentError("unclosed block in .krk units text"))
end

function _extract_blocks(text::AbstractString, keyword::AbstractString)
    blocks = Tuple{String,String}[]
    pos = firstindex(text)
    pattern = Regex("\\b$(keyword)\\b[^\\{]*\\{")
    while true
        m = match(pattern, text, pos)
        m === nothing && break
        brace = findnext('{', text, m.offset)
        close = _find_matching_brace(text, brace)
        header = strip(text[m.offset:prevind(text, brace)])
        body = strip(text[nextind(text, brace):prevind(text, close)])
        push!(blocks, (header, body))
        pos = nextind(text, close)
    end
    return blocks
end

function _parse_value(raw::AbstractString)
    s = strip(replace(raw, "," => ""))
    if startswith(s, "[") && endswith(s, "]")
        inner = strip(s[2:prevind(s, lastindex(s))])
        isempty(inner) && return Float64[]
        return [_parse_value(part) for part in split(inner, ",")]
    end
    low = lowercase(s)
    low == "true" && return true
    low == "false" && return false
    intv = tryparse(Int, s)
    intv !== nothing && return intv
    fltv = tryparse(Float64, s)
    fltv !== nothing && return fltv
    return Symbol(s)
end

function _assignment_dict(body::AbstractString)
    out = Dict{Symbol,Any}()
    for m in eachmatch(r"(\w+)\s*=\s*(\[[^\]]+\]|[^\s,{}]+)", body)
        out[Symbol(m.captures[1])] = _parse_value(m.captures[2])
    end
    return out
end

function _nt(d::Dict{Symbol,Any})
    pairs_sorted = sort(collect(d); by=p -> String(p.first))
    names = Tuple(first.(pairs_sorted))
    vals = Tuple(last.(pairs_sorted))
    return NamedTuple{names}(vals)
end

function _type_from_value(x, default)
    x === nothing && return default
    x === :Float64 && return Float64
    x === :Float32 && return Float32
    x === Float64 && return Float64
    x === Float32 && return Float32
    throw(ArgumentError("unsupported units T value `$x`"))
end

function _block_tag(header::AbstractString, keyword::AbstractString, fallback::Symbol)
    tokens = split(header)
    length(tokens) >= 2 && tokens[1] == keyword && return Symbol(tokens[2])
    return fallback
end

function _only_block(body::AbstractString, keyword::AbstractString)
    blocks = _extract_blocks(body, keyword)
    isempty(blocks) && throw(ArgumentError("missing `$keyword` block in units .krk plan"))
    return first(blocks)
end

function _compile_from_parts(physics::Symbol, phys::Dict{Symbol,Any},
                             units::Dict{Symbol,Any}, geomd::Dict{Symbol,Any},
                             bcd::Dict{Symbol,Any}, discd::Dict{Symbol,Any};
                             T=Float64, strict::Bool=true)
    kwd = Dict{Symbol,Any}()
    merge!(kwd, phys)
    merge!(kwd, units)
    merge!(kwd, discd)
    delete!(kwd, :physics)
    geom = _nt(geomd)
    bc = _nt(bcd)
    return compile(; physics=physics, geometry=geom, bc=bc, T=T,
                   strict=strict, _nt(kwd)...)
end

function _parse_mega_plan(header::String, body::String; T=Float64,
                          strict::Bool=true)
    name = _block_tag(header, "Plan", :plan)

    phys_header, phys_body = _only_block(body, "Physics")
    phys = _assignment_dict(phys_body)
    physics = _block_tag(phys_header, "Physics",
                         _sym(get(phys, :physics, :newtonian)))

    units = _assignment_dict(_only_block(body, "Units")[2])

    geom_header, geom_body = _only_block(body, "Geometry")
    geomd = _assignment_dict(geom_body)
    geomd[:type] = get(geomd, :type, _block_tag(geom_header, "Geometry", :unknown))
    geomd[:L_up] = get(geomd, :L_up, get(units, :L_up, 15.0))
    geomd[:L_down] = get(geomd, :L_down, get(units, :L_down, 15.0))

    bcd = _assignment_dict(_only_block(body, "BC")[2])
    disc_blocks = _extract_blocks(body, "Discretization")
    discd = isempty(disc_blocks) ? Dict{Symbol,Any}() : _assignment_dict(first(disc_blocks)[2])
    backend_blocks = _extract_blocks(body, "Backend")
    local_T = T
    if !isempty(backend_blocks)
        bd = _assignment_dict(first(backend_blocks)[2])
        local_T = _type_from_value(get(bd, :T, nothing), T)
    end
    return name => _compile_from_parts(physics, phys, units, geomd, bcd, discd;
                                       T=local_T, strict=strict)
end

function _flat_compile_dict(d::Dict{Symbol,Any}; T=Float64, strict::Bool=true)
    physics = _sym(get(d, :physics, :newtonian))
    geomd = Dict{Symbol,Any}(
        :type => get(d, :geometry_type, get(d, :type, :unknown)),
        :blockage => get(d, :blockage, 0.0),
        :L_up => get(d, :L_up, 15.0),
        :L_down => get(d, :L_down, 15.0),
    )
    haskey(d, :q_wall_dist) && (geomd[:q_wall_dist] = d[:q_wall_dist])
    bcd = Dict{Symbol,Any}(
        :inlet => get(d, :inlet, :velocity_parabolic),
        :outlet => get(d, :outlet, :zou_he_pressure),
        :north_wall => get(d, :north_wall, :halfwayBB),
        :south_wall => get(d, :south_wall, :halfwayBB),
        :wall_bc => get(d, :wall_bc, :halfwayBB),
    )
    discd = Dict{Symbol,Any}()
    for key in (:advection_scheme, :embedded_geometry, :embedded_gradient)
        haskey(d, key) && (discd[key] = d[key])
    end
    kwd = copy(d)
    for key in (:physics, :geometry_type, :type, :blockage, :inlet, :outlet,
                :north_wall, :south_wall, :wall_bc, :q_wall_dist,
                :advection_scheme, :embedded_geometry, :embedded_gradient,
                :from_plan, :T)
        delete!(kwd, key)
    end
    local_T = _type_from_value(get(d, :T, nothing), T)
    return _compile_from_parts(physics, kwd, Dict{Symbol,Any}(), geomd, bcd, discd;
                               T=local_T, strict=strict)
end

function parse_units_krk(text::AbstractString; T=Float64, strict::Bool=true)
    plans = Dict{Symbol,SimulationPlan}()
    for (header, body) in _extract_blocks(text, "Plan")
        name, plan = _parse_mega_plan(header, body; T=T, strict=strict)
        plans[name] = plan
    end

    plan_defs = Dict{Symbol,Dict{Symbol,Any}}()
    for (header, body) in _extract_blocks(text, "Define")
        tokens = split(header)
        length(tokens) >= 3 || continue
        name = Symbol(tokens[2])
        kind = Symbol(tokens[3])
        data = _assignment_dict(body)
        if kind === :from_nondim
            plan_defs[name] = data
        elseif haskey(data, :from_plan)
            base_name = _sym(data[:from_plan])
            haskey(plan_defs, base_name) ||
                throw(ArgumentError("unknown from_plan reference :$base_name"))
            merged = copy(plan_defs[base_name])
            for (key, val) in data
                key in _PLANNER_OWNED_KEYS && continue
                merged[key] = val
            end
            plan = _flat_compile_dict(merged; T=T, strict=strict)
            owned = sort([key for key in keys(data) if key in _PLANNER_OWNED_KEYS])
            if !isempty(owned)
                plan = _with_added_issues(plan,
                    [warn_issue(:planner_override,
                        "from_plan ignored hand-coded planner-owned key(s): $(join(string.(owned), ", "))")])
            end
            plans[name] = plan
        end
    end
    isempty(plans) && throw(ArgumentError("no units Plan or from_plan block found"))
    return plans
end

load_units_krk(path::AbstractString; kwargs...) =
    parse_units_krk(read(path, String); kwargs...)

function plan_from_krk(text::AbstractString; name=nothing, kwargs...)
    plans = parse_units_krk(text; kwargs...)
    name === nothing && length(plans) == 1 && return first(values(plans))
    name === nothing && throw(ArgumentError("multiple units plans present; pass name=:..."))
    return plans[_sym(name)]
end
