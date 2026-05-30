export GeometryDescriptor, build_geometry_descriptor

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
