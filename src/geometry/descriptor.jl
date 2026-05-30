export GeometryDescriptor, build_geometry_descriptor
export stl_kappa_max, obstacle_extents_in_R, halfway_wall_distances

struct GeometryDescriptor
    type::Symbol
    blockage::Float64
    q_wall_dist
    stl_hash::UInt64
    is_solid
end

function build_geometry_descriptor(type::Symbol, is_solid;
                                   q_wall_dist=nothing,
                                   stl_hash::UInt64=UInt64(0),
                                   blockage=nothing)
    blockage_value = if blockage === nothing
        n = length(is_solid)
        n == 0 ? 0.0 : Float64(count(identity, is_solid)) / Float64(n)
    else
        Float64(blockage)
    end
    return GeometryDescriptor(type, blockage_value, q_wall_dist, stl_hash, is_solid)
end

function build_geometry_descriptor(; type::Symbol, is_solid,
                                   q_wall_dist=nothing,
                                   stl_hash::UInt64=UInt64(0),
                                   blockage=nothing)
    return build_geometry_descriptor(type, is_solid;
                                     q_wall_dist=q_wall_dist,
                                     stl_hash=stl_hash,
                                     blockage=blockage)
end

"""
    stl_kappa_max(mesh_lu)::Float64

Estimate the maximum surface curvature from the LU-scaled STL bounding box.
The proxy uses `kappa_max = 1 / R_LU_eff`, with
`R_LU_eff = 0.5 * minimum(cross-section bbox span of the LU-scaled mesh)`.
This follows the inverse-LU convention so the units audit threshold
`R_LU*kappa_max` is dimensionless; exact for spheres/cylinders, conservative
for convex shapes.
"""
function stl_kappa_max(mesh_lu)::Float64
    spans = ntuple(i -> Float64(mesh_lu.bbox_max[i] - mesh_lu.bbox_min[i]), 3)
    span = minimum(filter(>(0.0), collect(spans)))
    R_LU_eff = 0.5 * span
    R_LU_eff > 0.0 || throw(ArgumentError("STL mesh has zero effective radius"))
    return 1.0 / R_LU_eff
end

function obstacle_extents_in_R(mask, R_LU; flow_axis=1)::Tuple{Float64,Float64}
    1 <= flow_axis <= ndims(mask) ||
        throw(ArgumentError("flow_axis must be in 1:$(ndims(mask))"))
    R_LU > 0 || throw(ArgumentError("R_LU must be positive"))

    nsolid = 0
    centroid_sum = 0.0
    for I in CartesianIndices(mask)
        mask[I] || continue
        nsolid += 1
        centroid_sum += Float64(Tuple(I)[flow_axis]) - 0.5
    end
    nsolid == 0 && return (NaN, NaN)

    centroid = centroid_sum / nsolid
    n_axis = Float64(size(mask, flow_axis))
    return (centroid / Float64(R_LU), (n_axis - centroid) / Float64(R_LU))
end

function halfway_wall_distances(mask)::Vector{Float64}
    nd = ndims(mask)
    out = Float64[]
    for I in CartesianIndices(mask)
        mask[I] && continue
        idx = Tuple(I)
        for axis in 1:nd
            for step in (-1, 1)
                nidx = ntuple(d -> d == axis ? idx[d] + step : idx[d], nd)
                if 1 <= nidx[axis] <= size(mask, axis) && mask[nidx...]
                    push!(out, 0.5)
                end
            end
        end
    end
    return out
end
