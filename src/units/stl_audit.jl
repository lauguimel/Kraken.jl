struct STLAuditResult
    stl_hash::UInt64
    issues::Vector{Issue}
end

const STL_AUDIT_CACHE = Dict{UInt64,STLAuditResult}()

function audit_stl(geom::GeometryDescriptor, units::Union{LBMUnits,Nothing}=nothing)
    geom.q_wall_dist === nothing && return STLAuditResult(geom.stl_hash, Issue[])
    if geom.stl_hash != 0 && haskey(STL_AUDIT_CACHE, geom.stl_hash)
        return STL_AUDIT_CACHE[geom.stl_hash]
    end
    issues = Issue[]
    q = geom.q_wall_dist
    if any(v -> v < 0.02 || v > 0.98, q)
        push!(issues, warn_issue(:q_wall_histogram_cliff,
            "q_wall distribution has near-0 or near-1 cliffs"))
    end
    if units !== nothing && geom.kappa_max > 0 &&
       units.R_LU * geom.kappa_max > 0.5
        push!(issues, warn_issue(:curvature_underresolved,
            "R_LU*kappa_max exceeds the 0.5 curvature audit threshold"))
    end
    if length(q) > 1
        max_jump = maximum(abs(q[i + 1] - q[i]) for i in 1:(length(q) - 1))
        max_jump > 0.7 && push!(issues, warn_issue(:q_wall_skewness,
            "adjacent q_wall jump exceeds 0.7"))
    end
    result = STLAuditResult(geom.stl_hash, issues)
    geom.stl_hash != 0 && (STL_AUDIT_CACHE[geom.stl_hash] = result)
    return result
end
