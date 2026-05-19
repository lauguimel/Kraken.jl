struct FVFDEmbeddedBoundary2D{A,C}
    wall_nx::A
    wall_ny::A
    wall_inv_distance::A
    wall_distance::A
    cell_fraction::A
    wall_fraction::A
    west_fraction::A
    east_fraction::A
    south_fraction::A
    north_fraction::A
    cut_count::C
end

function FVFDEmbeddedBoundary2D(wall_nx, wall_ny, wall_inv_distance)
    wall_distance = zero.(wall_inv_distance)
    cell_fraction = ones(eltype(wall_inv_distance), size(wall_inv_distance))
    wall_fraction = zeros(eltype(wall_inv_distance), size(wall_inv_distance))
    west_fraction = ones(eltype(wall_inv_distance), size(wall_inv_distance))
    east_fraction = ones(eltype(wall_inv_distance), size(wall_inv_distance))
    south_fraction = ones(eltype(wall_inv_distance), size(wall_inv_distance))
    north_fraction = ones(eltype(wall_inv_distance), size(wall_inv_distance))
    cut_count = zeros(UInt8, size(wall_inv_distance))
    return FVFDEmbeddedBoundary2D(
        wall_nx, wall_ny, wall_inv_distance, wall_distance, cell_fraction,
        wall_fraction, west_fraction, east_fraction, south_fraction,
        north_fraction, cut_count,
    )
end

function fvfd_empty_embedded_boundary_2d(Nx::Integer, Ny::Integer; FT::Type{<:AbstractFloat}=Float64)
    wall_nx = zeros(FT, Int(Nx), Int(Ny))
    wall_ny = zeros(FT, Int(Nx), Int(Ny))
    wall_inv_distance = zeros(FT, Int(Nx), Int(Ny))
    wall_distance = zeros(FT, Int(Nx), Int(Ny))
    cell_fraction = ones(FT, Int(Nx), Int(Ny))
    wall_fraction = zeros(FT, Int(Nx), Int(Ny))
    west_fraction = ones(FT, Int(Nx), Int(Ny))
    east_fraction = ones(FT, Int(Nx), Int(Ny))
    south_fraction = ones(FT, Int(Nx), Int(Ny))
    north_fraction = ones(FT, Int(Nx), Int(Ny))
    cut_count = zeros(UInt8, Int(Nx), Int(Ny))
    return FVFDEmbeddedBoundary2D(
        wall_nx, wall_ny, wall_inv_distance, wall_distance, cell_fraction,
        wall_fraction, west_fraction, east_fraction, south_fraction,
        north_fraction, cut_count,
    )
end

function _fvfd_halfplane_segment_fraction_2d(a, b, ::Type{FT}) where {FT}
    tol = sqrt(eps(FT))
    if abs(a) <= tol
        return b >= zero(FT) ? one(FT) : zero(FT)
    end
    threshold = -b / a
    if a > zero(FT)
        return min(one(FT), max(zero(FT), FT(0.5) - max(FT(-0.5), threshold)))
    else
        return min(one(FT), max(zero(FT), min(FT(0.5), threshold) + FT(0.5)))
    end
end

function _fvfd_polygon_area_centroid_2d(poly, ::Type{FT}) where {FT}
    nverts = length(poly)
    if nverts < 3
        return (; area=zero(FT), centroid_x=zero(FT), centroid_y=zero(FT))
    end

    area2 = zero(FT)
    cx_num = zero(FT)
    cy_num = zero(FT)
    @inbounds for idx in 1:nverts
        p1 = poly[idx]
        p2 = poly[idx == nverts ? 1 : idx + 1]
        cross = p1.x * p2.y - p2.x * p1.y
        area2 += cross
        cx_num += (p1.x + p2.x) * cross
        cy_num += (p1.y + p2.y) * cross
    end
    abs_area2 = abs(area2)
    if abs_area2 <= eps(FT)
        return (; area=zero(FT), centroid_x=zero(FT), centroid_y=zero(FT))
    end
    return (;
        area=abs_area2 / FT(2),
        centroid_x=cx_num / (FT(3) * area2),
        centroid_y=cy_num / (FT(3) * area2),
    )
end

function _fvfd_halfplane_square_fluid_moments_2d(nx, ny, distance, ::Type{FT}) where {FT}
    corners = (
        (x=FT(-0.5), y=FT(-0.5)),
        (x=FT(0.5), y=FT(-0.5)),
        (x=FT(0.5), y=FT(0.5)),
        (x=FT(-0.5), y=FT(0.5)),
    )
    point_type = NamedTuple{(:x,:y),Tuple{FT,FT}}
    in_poly = point_type[corners...]
    out_poly = point_type[]
    empty!(out_poly)
    nverts = length(in_poly)
    nverts == 0 && return (; cell_fraction=zero(FT), centroid_x=zero(FT), centroid_y=zero(FT))

    @inbounds for idx in 1:nverts
        p1 = in_poly[idx]
        p2 = in_poly[idx == nverts ? 1 : idx + 1]
        s1 = nx * p1.x + ny * p1.y + distance
        s2 = nx * p2.x + ny * p2.y + distance
        inside1 = s1 >= zero(FT)
        inside2 = s2 >= zero(FT)
        if inside1 && inside2
            push!(out_poly, p2)
        elseif inside1 && !inside2
            t = s1 / (s1 - s2)
            push!(out_poly, (x=p1.x + t * (p2.x - p1.x), y=p1.y + t * (p2.y - p1.y)))
        elseif !inside1 && inside2
            t = s1 / (s1 - s2)
            push!(out_poly, (x=p1.x + t * (p2.x - p1.x), y=p1.y + t * (p2.y - p1.y)))
            push!(out_poly, p2)
        end
    end

    moments = _fvfd_polygon_area_centroid_2d(out_poly, FT)
    return (;
        cell_fraction=min(one(FT), max(zero(FT), moments.area)),
        centroid_x=moments.centroid_x,
        centroid_y=moments.centroid_y,
    )
end

function _fvfd_halfplane_square_fraction_2d(nx, ny, distance, ::Type{FT}) where {FT}
    return _fvfd_halfplane_square_fluid_moments_2d(nx, ny, distance, FT).cell_fraction
end

function _fvfd_halfplane_square_measures_2d(nx, ny, distance, ::Type{FT}) where {FT}
    cell_fraction = _fvfd_halfplane_square_fraction_2d(nx, ny, distance, FT)
    west_fraction = _fvfd_halfplane_segment_fraction_2d(
        ny, distance - nx / FT(2), FT,
    )
    east_fraction = _fvfd_halfplane_segment_fraction_2d(
        ny, distance + nx / FT(2), FT,
    )
    south_fraction = _fvfd_halfplane_segment_fraction_2d(
        nx, distance - ny / FT(2), FT,
    )
    north_fraction = _fvfd_halfplane_segment_fraction_2d(
        nx, distance + ny / FT(2), FT,
    )
    wall_fraction = hypot(east_fraction - west_fraction, north_fraction - south_fraction)
    return (;
        cell_fraction, wall_fraction,
        west_fraction, east_fraction, south_fraction, north_fraction,
    )
end

function fvfd_embedded_boundary_from_halfplane_2d(
    Nx::Integer, Ny::Integer, nx::Real, ny::Real, offset::Real;
    FT::Type{<:AbstractFloat}=Float64,
)
    normal_length = hypot(FT(nx), FT(ny))
    normal_length > zero(FT) ||
        throw(ArgumentError("half-plane normal must be non-zero"))
    nx_unit = FT(nx) / normal_length
    ny_unit = FT(ny) / normal_length
    offset_unit = FT(offset) / normal_length

    Nx_i = Int(Nx)
    Ny_i = Int(Ny)
    wall_nx = zeros(FT, Nx_i, Ny_i)
    wall_ny = zeros(FT, Nx_i, Ny_i)
    wall_inv_distance = zeros(FT, Nx_i, Ny_i)
    wall_distance = zeros(FT, Nx_i, Ny_i)
    cell_fraction = ones(FT, Nx_i, Ny_i)
    wall_fraction = zeros(FT, Nx_i, Ny_i)
    west_fraction = ones(FT, Nx_i, Ny_i)
    east_fraction = ones(FT, Nx_i, Ny_i)
    south_fraction = ones(FT, Nx_i, Ny_i)
    north_fraction = ones(FT, Nx_i, Ny_i)
    cut_count = zeros(UInt8, Nx_i, Ny_i)
    tol = sqrt(eps(FT))

    @inbounds for j in 1:Ny_i, i in 1:Nx_i
        x_center = FT(i) - FT(0.5)
        y_center = FT(j) - FT(0.5)
        distance = nx_unit * x_center + ny_unit * y_center + offset_unit
        measures = _fvfd_halfplane_square_measures_2d(nx_unit, ny_unit, distance, FT)
        cell_fraction[i, j] = measures.cell_fraction
        wall_fraction[i, j] = measures.wall_fraction
        west_fraction[i, j] = measures.west_fraction
        east_fraction[i, j] = measures.east_fraction
        south_fraction[i, j] = measures.south_fraction
        north_fraction[i, j] = measures.north_fraction
        if tol < measures.cell_fraction < one(FT) - tol
            wall_nx[i, j] = nx_unit
            wall_ny[i, j] = ny_unit
            moments = _fvfd_halfplane_square_fluid_moments_2d(
                nx_unit, ny_unit, distance, FT,
            )
            centroid_distance = distance +
                                nx_unit * moments.centroid_x +
                                ny_unit * moments.centroid_y
            wall_distance[i, j] = max(centroid_distance, eps(FT))
            wall_inv_distance[i, j] = inv(wall_distance[i, j])
            cut_count[i, j] = UInt8(1)
        end
    end

    return FVFDEmbeddedBoundary2D(
        wall_nx, wall_ny, wall_inv_distance, wall_distance, cell_fraction,
        wall_fraction, west_fraction, east_fraction, south_fraction,
        north_fraction, cut_count,
    )
end

function fvfd_geometry_from_halfplane_2d(
    Nx::Integer, Ny::Integer, dx::Real, dy::Real,
    bc::FVFDDomainBC2D, nx::Real, ny::Real, offset::Real;
    FT::Type{<:AbstractFloat}=Float64,
    level::Integer=0,
    solid_tolerance=nothing,
)
    embedded = fvfd_embedded_boundary_from_halfplane_2d(
        Nx, Ny, nx, ny, offset; FT,
    )
    tol = solid_tolerance === nothing ? sqrt(eps(FT)) : FT(solid_tolerance)
    is_solid = embedded.cell_fraction .<= tol
    patch = FVFDPatch2D(FT(dx), FT(dy); level)
    return FVFDGeometry2D(is_solid, embedded, patch, bc)
end

@inline function _fvfd_interval_overlap_1d(a0, a1, b0, b1)
    return max(zero(a0), min(a1, b1) - max(a0, b0))
end

function _fvfd_circle_vertical_face_fraction_2d(
    x_face, y0, y1, cx, cy, radius, ::Type{FT},
) where {FT}
    dx = x_face - cx
    abs(dx) >= radius && return one(FT)
    half_inside = sqrt(max(zero(FT), radius * radius - dx * dx))
    inside = _fvfd_interval_overlap_1d(y0, y1, cy - half_inside, cy + half_inside)
    return min(one(FT), max(zero(FT), one(FT) - inside / (y1 - y0)))
end

function _fvfd_circle_horizontal_face_fraction_2d(
    y_face, x0, x1, cx, cy, radius, ::Type{FT},
) where {FT}
    dy = y_face - cy
    abs(dy) >= radius && return one(FT)
    half_inside = sqrt(max(zero(FT), radius * radius - dy * dy))
    inside = _fvfd_interval_overlap_1d(x0, x1, cx - half_inside, cx + half_inside)
    return min(one(FT), max(zero(FT), one(FT) - inside / (x1 - x0)))
end

function _fvfd_circle_cell_fraction_sampled_2d(
    x0, y0, cx, cy, radius, samples::Integer, ::Type{FT},
) where {FT}
    return _fvfd_circle_cell_fluid_moments_sampled_2d(
        x0, y0, cx, cy, radius, samples, FT,
    ).cell_fraction
end

function _fvfd_circle_cell_fluid_moments_sampled_2d(
    x0, y0, cx, cy, radius, samples::Integer, ::Type{FT},
) where {FT}
    samples_i = Int(samples)
    inv_samples = inv(FT(samples_i))
    radius2 = radius * radius
    outside_count = 0
    x_sum = zero(FT)
    y_sum = zero(FT)
    @inbounds for sj in 1:samples_i, si in 1:samples_i
        x = x0 + (FT(si) - FT(0.5)) * inv_samples
        y = y0 + (FT(sj) - FT(0.5)) * inv_samples
        dx = x - cx
        dy = y - cy
        if dx * dx + dy * dy >= radius2
            outside_count += 1
            x_sum += x
            y_sum += y
        end
    end
    cell_fraction = FT(outside_count) / FT(samples_i * samples_i)
    if outside_count == 0
        return (;
            cell_fraction,
            centroid_x=(x0 + FT(0.5)),
            centroid_y=(y0 + FT(0.5)),
        )
    end
    return (;
        cell_fraction,
        centroid_x=x_sum / FT(outside_count),
        centroid_y=y_sum / FT(outside_count),
    )
end

function fvfd_embedded_boundary_from_circle_2d(
    Nx::Integer, Ny::Integer, cx::Real, cy::Real, radius::Real;
    FT::Type{<:AbstractFloat}=Float64,
    samples::Integer=16,
)
    radius_ft = FT(radius)
    radius_ft > zero(FT) || throw(ArgumentError("circle radius must be positive"))
    samples > 0 || throw(ArgumentError("samples must be positive"))
    Nx_i = Int(Nx)
    Ny_i = Int(Ny)
    cx_ft = FT(cx)
    cy_ft = FT(cy)

    wall_nx = zeros(FT, Nx_i, Ny_i)
    wall_ny = zeros(FT, Nx_i, Ny_i)
    wall_inv_distance = zeros(FT, Nx_i, Ny_i)
    wall_distance = zeros(FT, Nx_i, Ny_i)
    cell_fraction = ones(FT, Nx_i, Ny_i)
    wall_fraction = zeros(FT, Nx_i, Ny_i)
    west_fraction = ones(FT, Nx_i, Ny_i)
    east_fraction = ones(FT, Nx_i, Ny_i)
    south_fraction = ones(FT, Nx_i, Ny_i)
    north_fraction = ones(FT, Nx_i, Ny_i)
    cut_count = zeros(UInt8, Nx_i, Ny_i)
    tol = sqrt(eps(FT))

    @inbounds for j in 1:Ny_i, i in 1:Nx_i
        x0 = FT(i - 1)
        x1 = FT(i)
        y0 = FT(j - 1)
        y1 = FT(j)
        west = _fvfd_circle_vertical_face_fraction_2d(
            x0, y0, y1, cx_ft, cy_ft, radius_ft, FT,
        )
        east = _fvfd_circle_vertical_face_fraction_2d(
            x1, y0, y1, cx_ft, cy_ft, radius_ft, FT,
        )
        south = _fvfd_circle_horizontal_face_fraction_2d(
            y0, x0, x1, cx_ft, cy_ft, radius_ft, FT,
        )
        north = _fvfd_circle_horizontal_face_fraction_2d(
            y1, x0, x1, cx_ft, cy_ft, radius_ft, FT,
        )
        sampled = _fvfd_circle_cell_fluid_moments_sampled_2d(
            x0, y0, cx_ft, cy_ft, radius_ft, samples, FT,
        )
        cell = sampled.cell_fraction
        west_fraction[i, j] = west
        east_fraction[i, j] = east
        south_fraction[i, j] = south
        north_fraction[i, j] = north
        cell_fraction[i, j] = cell

        area_x = west - east
        area_y = south - north
        length = hypot(area_x, area_y)
        wall_fraction[i, j] = length
        if length > tol && tol < cell < one(FT) - tol
            wall_nx[i, j] = -area_x / length
            wall_ny[i, j] = -area_y / length
            x_center = (x0 + x1) / FT(2)
            y_center = (y0 + y1) / FT(2)
            signed_center_distance = hypot(x_center - cx_ft, y_center - cy_ft) - radius_ft
            distance = signed_center_distance +
                       wall_nx[i, j] * (sampled.centroid_x - x_center) +
                       wall_ny[i, j] * (sampled.centroid_y - y_center)
            distance = max(distance, eps(FT))
            wall_distance[i, j] = distance
            wall_inv_distance[i, j] = inv(max(distance, eps(FT)))
            cut_count[i, j] = UInt8(1)
        end
    end

    return FVFDEmbeddedBoundary2D(
        wall_nx, wall_ny, wall_inv_distance, wall_distance, cell_fraction,
        wall_fraction, west_fraction, east_fraction, south_fraction,
        north_fraction, cut_count,
    )
end

function fvfd_geometry_from_circle_2d(
    Nx::Integer, Ny::Integer, dx::Real, dy::Real,
    bc::FVFDDomainBC2D, cx::Real, cy::Real, radius::Real;
    FT::Type{<:AbstractFloat}=Float64,
    level::Integer=0,
    samples::Integer=16,
    solid_tolerance=nothing,
)
    embedded = fvfd_embedded_boundary_from_circle_2d(
        Nx, Ny, cx, cy, radius; FT, samples,
    )
    tol = solid_tolerance === nothing ? sqrt(eps(FT)) : FT(solid_tolerance)
    is_solid = embedded.cell_fraction .<= tol
    patch = FVFDPatch2D(FT(dx), FT(dy); level)
    return FVFDGeometry2D(is_solid, embedded, patch, bc)
end

function fvfd_embedded_boundary_from_qwall_2d(
    q_wall;
    FT::Type{<:AbstractFloat}=eltype(q_wall),
    include_axis_aligned::Bool=false,
    include_halfway::Bool=false,
)
    Nx, Ny, _ = size(q_wall)
    wall_nx = zeros(FT, Nx, Ny)
    wall_ny = zeros(FT, Nx, Ny)
    wall_inv_distance = zeros(FT, Nx, Ny)
    wall_distance = zeros(FT, Nx, Ny)
    cell_fraction = ones(FT, Nx, Ny)
    wall_fraction = zeros(FT, Nx, Ny)
    west_fraction = ones(FT, Nx, Ny)
    east_fraction = ones(FT, Nx, Ny)
    south_fraction = ones(FT, Nx, Ny)
    north_fraction = ones(FT, Nx, Ny)
    cut_count = zeros(UInt8, Nx, Ny)
    cxs = velocities_x(D2Q9())
    cys = velocities_y(D2Q9())
    halfway_tol = sqrt(eps(FT))

    @inbounds for j in 1:Ny, i in 1:Nx
        has_subcell_cut = false
        for q in 2:9
            q_w = FT(q_wall[i, j, q])
            if q_w > zero(FT) && abs(q_w - FT(0.5)) > halfway_tol
                has_subcell_cut = true
                break
            end
        end

        nx_sum = zero(FT)
        ny_sum = zero(FT)
        best_distance = typemax(FT)
        best_nx = zero(FT)
        best_ny = zero(FT)
        included_cut_count = UInt8(0)
        for q in 2:9
            q_w = FT(q_wall[i, j, q])
            q_w > zero(FT) || continue
            is_halfway = abs(q_w - FT(0.5)) <= halfway_tol
            if !include_halfway && is_halfway && !has_subcell_cut
                continue
            end
            cqx = FT(cxs[q])
            cqy = FT(cys[q])
            is_axis_aligned = cqx == zero(FT) || cqy == zero(FT)
            if !include_axis_aligned && is_axis_aligned && !has_subcell_cut
                continue
            end
            link_length = hypot(cqx, cqy)
            link_length > zero(FT) || continue
            nx_q = -cqx / link_length
            ny_q = -cqy / link_length
            nx_sum += nx_q
            ny_sum += ny_q
            included_cut_count += UInt8(1)
            distance_q = q_w * link_length
            if distance_q < best_distance
                best_distance = distance_q
                best_nx = nx_q
                best_ny = ny_q
            end
        end

        normal_length = hypot(nx_sum, ny_sum)
        if normal_length > zero(FT)
            nx = nx_sum / normal_length
            ny = ny_sum / normal_length
            distance_sum = zero(FT)
            distance_count = 0
            for q in 2:9
                q_w = FT(q_wall[i, j, q])
                q_w > zero(FT) || continue
                is_halfway = abs(q_w - FT(0.5)) <= halfway_tol
                if !include_halfway && is_halfway && !has_subcell_cut
                    continue
                end
                cqx = FT(cxs[q])
                cqy = FT(cys[q])
                is_axis_aligned = cqx == zero(FT) || cqy == zero(FT)
                if !include_axis_aligned && is_axis_aligned && !has_subcell_cut
                    continue
                end
                distance = -(q_w * cqx * nx + q_w * cqy * ny)
                if distance > zero(FT)
                    distance_sum += distance
                    distance_count += 1
                end
            end
            if distance_count > 0
                wall_nx[i, j] = nx
                wall_ny[i, j] = ny
                distance = distance_sum / FT(distance_count)
                measures = _fvfd_halfplane_square_measures_2d(nx, ny, distance, FT)
                moments = _fvfd_halfplane_square_fluid_moments_2d(nx, ny, distance, FT)
                centroid_distance = distance +
                                    nx * moments.centroid_x +
                                    ny * moments.centroid_y
                wall_distance[i, j] = max(centroid_distance, eps(FT))
                wall_inv_distance[i, j] = inv(wall_distance[i, j])
                cell_fraction[i, j] = measures.cell_fraction
                wall_fraction[i, j] = measures.wall_fraction
                west_fraction[i, j] = measures.west_fraction
                east_fraction[i, j] = measures.east_fraction
                south_fraction[i, j] = measures.south_fraction
                north_fraction[i, j] = measures.north_fraction
                cut_count[i, j] = included_cut_count
            elseif best_distance < typemax(FT)
                wall_nx[i, j] = best_nx
                wall_ny[i, j] = best_ny
                measures = _fvfd_halfplane_square_measures_2d(
                    best_nx, best_ny, best_distance, FT,
                )
                moments = _fvfd_halfplane_square_fluid_moments_2d(
                    best_nx, best_ny, best_distance, FT,
                )
                centroid_distance = best_distance +
                                    best_nx * moments.centroid_x +
                                    best_ny * moments.centroid_y
                wall_distance[i, j] = max(centroid_distance, eps(FT))
                wall_inv_distance[i, j] = inv(wall_distance[i, j])
                cell_fraction[i, j] = measures.cell_fraction
                wall_fraction[i, j] = measures.wall_fraction
                west_fraction[i, j] = measures.west_fraction
                east_fraction[i, j] = measures.east_fraction
                south_fraction[i, j] = measures.south_fraction
                north_fraction[i, j] = measures.north_fraction
                cut_count[i, j] = included_cut_count
            end
        elseif best_distance < typemax(FT)
            wall_nx[i, j] = best_nx
            wall_ny[i, j] = best_ny
            measures = _fvfd_halfplane_square_measures_2d(
                best_nx, best_ny, best_distance, FT,
            )
            moments = _fvfd_halfplane_square_fluid_moments_2d(
                best_nx, best_ny, best_distance, FT,
            )
            centroid_distance = best_distance +
                                best_nx * moments.centroid_x +
                                best_ny * moments.centroid_y
            wall_distance[i, j] = max(centroid_distance, eps(FT))
            wall_inv_distance[i, j] = inv(wall_distance[i, j])
            cell_fraction[i, j] = measures.cell_fraction
            wall_fraction[i, j] = measures.wall_fraction
            west_fraction[i, j] = measures.west_fraction
            east_fraction[i, j] = measures.east_fraction
            south_fraction[i, j] = measures.south_fraction
            north_fraction[i, j] = measures.north_fraction
            cut_count[i, j] = included_cut_count
        end
    end

    return FVFDEmbeddedBoundary2D(
        wall_nx, wall_ny, wall_inv_distance, wall_distance, cell_fraction,
        wall_fraction, west_fraction, east_fraction, south_fraction,
        north_fraction, cut_count,
    )
end

function _fvfd_host_boundary_vector_2d(
    values, expected::Integer, ::Type{T}, name::Symbol;
    active::Bool, default,
) where {T<:AbstractFloat}
    n = Int(expected)
    out = Vector{T}(undef, n)
    if values isa Number
        fill!(out, T(values))
        return out
    end

    observed = try
        length(values)
    catch
        nothing
    end
    if observed == n
        source = values isa AbstractArray ? vec(Array(values)) : values
        @inbounds for idx in 1:n
            out[idx] = T(source[idx])
        end
        return out
    elseif active
        if observed === nothing
            throw(DimensionMismatch(
                "$(name) boundary does not provide a length; expected $(expected)",
            ))
        end
        throw(DimensionMismatch(
            "$(name) boundary length $(observed) does not match expected $(expected)",
        ))
    end

    fill!(out, T(default))
    return out
end

function _fvfd_transfer_boundary_vector_2d(
    values, backend, ::Type{T}, expected::Integer, name::Symbol;
    active::Bool, default,
) where {T<:AbstractFloat}
    host = _fvfd_host_boundary_vector_2d(
        values, expected, T, name; active, default,
    )
    dev = KernelAbstractions.allocate(backend, T, Int(expected))
    copyto!(dev, host)
    return dev
end

function fvfd_transfer_field_bc_2d(
    field_bc::FVFDFieldBC2D, backend, ::Type{T},
    Nx::Integer, Ny::Integer, bc::FVFDDomainBC2D;
    name::Symbol=:field_bc, default=zero(T),
) where {T<:AbstractFloat}
    west = _fvfd_transfer_boundary_vector_2d(
        field_bc.west, backend, T, Ny, Symbol(name, :_west);
        active=bc.west == FVFD_BC_OPEN, default,
    )
    east = _fvfd_transfer_boundary_vector_2d(
        field_bc.east, backend, T, Ny, Symbol(name, :_east);
        active=bc.east == FVFD_BC_OPEN, default,
    )
    south = _fvfd_transfer_boundary_vector_2d(
        field_bc.south, backend, T, Nx, Symbol(name, :_south);
        active=bc.south == FVFD_BC_OPEN, default,
    )
    north = _fvfd_transfer_boundary_vector_2d(
        field_bc.north, backend, T, Nx, Symbol(name, :_north);
        active=bc.north == FVFD_BC_OPEN, default,
    )
    return FVFDFieldBC2D(west, east, south, north)
end

function fvfd_transfer_field_bc_2d(
    field_bc::FVFDFieldBC2D, backend,
    Nx::Integer, Ny::Integer, bc::FVFDDomainBC2D;
    FT::Type{<:AbstractFloat}=Float64, kwargs...,
)
    return fvfd_transfer_field_bc_2d(field_bc, backend, FT, Nx, Ny, bc; kwargs...)
end

function fvfd_transfer_embedded_boundary_2d(
    embedded::FVFDEmbeddedBoundary2D, backend, ::Type{T}=eltype(embedded.wall_nx),
) where {T<:AbstractFloat}
    Nx, Ny = size(embedded.wall_inv_distance)
    wall_nx = KernelAbstractions.allocate(backend, T, Nx, Ny)
    wall_ny = KernelAbstractions.allocate(backend, T, Nx, Ny)
    wall_inv_distance = KernelAbstractions.allocate(backend, T, Nx, Ny)
    wall_distance = KernelAbstractions.allocate(backend, T, Nx, Ny)
    cell_fraction = KernelAbstractions.allocate(backend, T, Nx, Ny)
    wall_fraction = KernelAbstractions.allocate(backend, T, Nx, Ny)
    west_fraction = KernelAbstractions.allocate(backend, T, Nx, Ny)
    east_fraction = KernelAbstractions.allocate(backend, T, Nx, Ny)
    south_fraction = KernelAbstractions.allocate(backend, T, Nx, Ny)
    north_fraction = KernelAbstractions.allocate(backend, T, Nx, Ny)
    cut_count = KernelAbstractions.allocate(backend, UInt8, Nx, Ny)
    copyto!(wall_nx, T.(embedded.wall_nx))
    copyto!(wall_ny, T.(embedded.wall_ny))
    copyto!(wall_inv_distance, T.(embedded.wall_inv_distance))
    copyto!(wall_distance, T.(embedded.wall_distance))
    copyto!(cell_fraction, T.(embedded.cell_fraction))
    copyto!(wall_fraction, T.(embedded.wall_fraction))
    copyto!(west_fraction, T.(embedded.west_fraction))
    copyto!(east_fraction, T.(embedded.east_fraction))
    copyto!(south_fraction, T.(embedded.south_fraction))
    copyto!(north_fraction, T.(embedded.north_fraction))
    copyto!(cut_count, UInt8.(embedded.cut_count))
    return FVFDEmbeddedBoundary2D(
        wall_nx, wall_ny, wall_inv_distance, wall_distance, cell_fraction,
        wall_fraction, west_fraction, east_fraction, south_fraction,
        north_fraction, cut_count,
    )
end

function fvfd_geometry_from_lbm_2d(
    is_solid, q_wall, dx::Real, dy::Real, bc::FVFDDomainBC2D;
    FT::Type{<:AbstractFloat}=eltype(q_wall),
    level::Integer=0,
    include_axis_aligned::Bool=false,
    include_halfway::Bool=false,
)
    embedded = fvfd_embedded_boundary_from_qwall_2d(
        q_wall; FT, include_axis_aligned, include_halfway,
    )
    patch = FVFDPatch2D(FT(dx), FT(dy); level)
    return FVFDGeometry2D(is_solid, embedded, patch, bc)
end

function fvfd_transfer_geometry_2d(
    geometry::FVFDGeometry2D, backend, ::Type{T}=eltype(geometry.embedded.wall_nx),
) where {T<:AbstractFloat}
    Nx, Ny = size(geometry.is_solid)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    copyto!(is_solid, Matrix{Bool}(geometry.is_solid))
    embedded = fvfd_transfer_embedded_boundary_2d(geometry.embedded, backend, T)
    patch = FVFDPatch2D(T(geometry.patch.dx), T(geometry.patch.dy); level=geometry.patch.level)
    return FVFDGeometry2D(is_solid, embedded, patch, geometry.bc)
end
