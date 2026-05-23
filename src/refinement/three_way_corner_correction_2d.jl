# Topology-detected equilibrium correction for leaf-equivalent coarse-to-fine
# packets at two-patch 3-way vertices.

struct ThreeWayCornerCorrection2D
    route_idx::Int
    src::Int
    q::Int
    dst::Int
    ix::Int
    iy::Int
    x_off::Float64
    y_off::Float64
end

const _THREE_WAY_CORNER_CORRECTIONS_BY_TABLE_2D =
    Dict{UInt,Vector{ThreeWayCornerCorrection2D}}()

# Set of coarse cell IDs whose c2f routes should use :limited_linear
# packet builder instead of :flat. Populated at route-table build.
const _THREE_WAY_FLAGGED_SRC_BY_TABLE_2D = Dict{UInt,Set{Int}}()

function _three_way_same_level_cell_id_2d(
        spec::ConservativeTreeSpec2D,
        level::Int,
        i::Int,
        j::Int;
        periodic_x::Bool=false)
    nx = _conservative_tree_level_size_2d(spec.Nx, level)
    ny = _conservative_tree_level_size_2d(spec.Ny, level)
    ii = periodic_x ? mod1(i, nx) : i
    1 <= ii <= nx && 1 <= j <= ny || return 0
    return conservative_tree_cell_id_2d(spec, level, ii, j)
end

function _three_way_same_level_refined_2d(
        spec::ConservativeTreeSpec2D,
        level::Int,
        i::Int,
        j::Int;
        periodic_x::Bool=false)
    cell_id = _three_way_same_level_cell_id_2d(
        spec, level, i, j; periodic_x=periodic_x)
    cell_id == 0 && return false
    return !spec.cells[cell_id].active &&
        spec.children[cell_id] != (0, 0, 0, 0)
end

function _three_way_refine_block_names_covering_parent_2d(
        spec::ConservativeTreeSpec2D,
        level::Int,
        i::Int,
        j::Int;
        periodic_x::Bool=false)
    nx = _conservative_tree_level_size_2d(spec.Nx, level)
    ii = periodic_x ? mod1(i, nx) : i
    names = String[]
    for name in keys(spec.refine_level)
        spec.refine_level[name] == level + 1 || continue
        parent_i = _conservative_tree_parent_range_from_child_2d(
            spec.refine_i_range[name])
        parent_j = _conservative_tree_parent_range_from_child_2d(
            spec.refine_j_range[name])
        if ii in parent_i && j in parent_j
            push!(names, name)
        end
    end
    return names
end

function _three_way_has_distinct_refine_blocks_2d(
        xs::Vector{String},
        ys::Vector{String})
    for x in xs, y in ys
        x != y && return true
    end
    return false
end

function _three_way_corner_source_2d(
        spec::ConservativeTreeSpec2D,
        src_id::Int;
        periodic_x::Bool=false)
    src = spec.cells[src_id]
    src.level < spec.max_level || return false

    x_sides = Tuple{Int,Bool}[
        (1, _three_way_same_level_refined_2d(
            spec, src.level, src.i + 1, src.j; periodic_x=periodic_x)),
        (-1, _three_way_same_level_refined_2d(
            spec, src.level, src.i - 1, src.j; periodic_x=periodic_x)),
    ]
    y_sides = Tuple{Int,Bool}[
        (1, _three_way_same_level_refined_2d(
            spec, src.level, src.i, src.j + 1; periodic_x=periodic_x)),
        (-1, _three_way_same_level_refined_2d(
            spec, src.level, src.i, src.j - 1; periodic_x=periodic_x)),
    ]

    any(last, x_sides) && any(last, y_sides) || return false
    for (sx, has_x) in x_sides
        has_x || continue
        for (sy, has_y) in y_sides
            has_y || continue
            x_blocks = _three_way_refine_block_names_covering_parent_2d(
                spec, src.level, src.i + sx, src.j; periodic_x=periodic_x)
            y_blocks = _three_way_refine_block_names_covering_parent_2d(
                spec, src.level, src.i, src.j + sy; periodic_x=periodic_x)
            diagonal_refined = _three_way_same_level_refined_2d(
                spec, src.level, src.i + sx, src.j + sy;
                periodic_x=periodic_x)
            !diagonal_refined && return true
            _three_way_has_distinct_refine_blocks_2d(x_blocks, y_blocks) &&
                return true
        end
    end
    return false
end

function _detect_three_way_corner_routes_2d(
        table::ConservativeTreeRouteTable2D,
        spec::ConservativeTreeSpec2D;
        periodic_x::Bool=false)
    corrections = ThreeWayCornerCorrection2D[]
    flagged_src = Dict{Int,Bool}()
    @inbounds for (route_idx, route) in pairs(table.routes)
        _is_split_route_kind_2d(route.kind) || continue
        route.dst != 0 || continue
        is_flagged = get!(flagged_src, route.src) do
            _three_way_corner_source_2d(
                spec, route.src; periodic_x=periodic_x)
        end
        is_flagged || continue
        dst = spec.cells[route.dst]
        parent = spec.cells[dst.parent]
        ix, iy = _conservative_tree_child_index_in_parent_2d(parent, dst)
        x_off = (Float64(ix) - 1.5) * 0.5
        y_off = (Float64(iy) - 1.5) * 0.5
        push!(corrections, ThreeWayCornerCorrection2D(
            route_idx, route.src, route.q, route.dst, ix, iy, x_off, y_off))
    end
    return corrections
end

function _register_three_way_corner_corrections_2d!(
        table::ConservativeTreeRouteTable2D,
        spec::ConservativeTreeSpec2D;
        periodic_x::Bool=false)
    corrections = _detect_three_way_corner_routes_2d(
        table, spec; periodic_x=periodic_x)
    _THREE_WAY_CORNER_CORRECTIONS_BY_TABLE_2D[objectid(table)] = corrections
    flagged_src = Set{Int}()
    for c in corrections
        push!(flagged_src, c.src)
    end
    _THREE_WAY_FLAGGED_SRC_BY_TABLE_2D[objectid(table)] = flagged_src
    return table
end

function _three_way_corner_corrections_2d(
        table::ConservativeTreeRouteTable2D)
    return get(_THREE_WAY_CORNER_CORRECTIONS_BY_TABLE_2D,
               objectid(table), ThreeWayCornerCorrection2D[])
end

@inline function _is_three_way_flagged_src_2d(
        table::ConservativeTreeRouteTable2D, src_id::Int)
    flagged = get(_THREE_WAY_FLAGGED_SRC_BY_TABLE_2D,
                  objectid(table), nothing)
    flagged === nothing && return false
    return src_id in flagged
end
