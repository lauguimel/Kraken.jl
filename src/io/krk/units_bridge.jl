function _apply_units_bridge!(units::UnitsSetup, lattice::Symbol, domain::DomainSetup,
                              regions::Vector{GeometryRegion},
                              boundaries::Vector{BoundarySetup},
                              physics_params::Dict{Symbol,Float64},
                              user_vars::Dict{Symbol,Float64})
    _validate_units_domain!(domain, lattice)
    haskey(physics_params, :nu) && throw(ArgumentError(
        "Units block owns Physics nu. Use `Physics nu = auto` or omit `nu`."))

    dx_real = units.L_ref / units.R_LU
    scale_stl = 1.0 / dx_real
    geom = _units_geometry_namedtuple(regions, domain, scale_stl, units, lattice)
    bc = _units_bc_namedtuple(boundaries, regions)
    plan = Units.compile(; physics=:newtonian, geometry=geom, bc=bc,
                         Re=units.Re, R_LU=units.R_LU, dx_real=dx_real,
                         scaling=units.scaling, L_up=geom.L_up,
                         L_down=geom.L_down)

    user_vars[:u_LU] = Float64(plan.units.u_LU)
    _inject_units_velocity!(boundaries, user_vars)
    physics_params[:nu] = Float64(plan.units.nu_total_LU)
    physics_params[:Re] = units.Re
    return _units_scaled_stl_regions(regions, 1.0 / Float64(plan.units.dx_real))
end

function _validate_units_domain!(domain::DomainSetup, lattice::Symbol)
    ok = domain.Lx == Float64(domain.Nx) && domain.Ly == Float64(domain.Ny)
    lattice === :D3Q19 && (ok &= domain.Lz == Float64(domain.Nz))
    ok || throw(ArgumentError(
        "Units block requires Domain L == N in active dimensions so coordinates are LU; " *
        "got L=$(domain.Lx)x$(domain.Ly)x$(domain.Lz), " *
        "N=$(domain.Nx)x$(domain.Ny)x$(domain.Nz)."))
    return nothing
end

function _units_scaled_stl_regions(regions::Vector{GeometryRegion}, scale_stl::Float64)
    out = GeometryRegion[]
    sizehint!(out, length(regions))
    for region in regions
        stl = region.stl
        if stl !== nothing && stl.scale == 1.0
            stl = STLSource(stl.file, scale_stl, stl.translate, stl.z_slice)
        end
        push!(out, GeometryRegion(region.name, region.kind, region.condition, stl,
                                  region.bc_type, region.bc_values))
    end
    return out
end

function _units_geometry_namedtuple(regions::Vector{GeometryRegion},
                                    domain::DomainSetup,
                                    scale_stl::Float64,
                                    units::UnitsSetup,
                                    lattice::Symbol)
    any(r -> r.stl !== nothing, regions) ||
        throw(ArgumentError("Units block currently requires at least one STL region"))
    scaled = _units_scaled_stl_regions(regions, scale_stl)
    probe = SimulationSetup("_units_geometry_probe", lattice, domain,
                            PhysicsSetup(Dict{Symbol,Float64}(),
                                         Dict{Symbol,KrakenExpr}()),
                            Dict{Symbol,Float64}(), scaled, BoundarySetup[],
                            nothing, Symbol[], 1, OutputSetup[], nothing,
                            RefineSetup[], nothing, RheologySetup[])
    mask = if lattice === :D3Q19
        mask3 = falses(domain.Nx, domain.Ny, domain.Nz)
        _apply_geometry_3d!(mask3, probe, 1.0)
        mask3
    else
        mask2 = falses(domain.Nx, domain.Ny)
        _apply_geometry!(mask2, probe, 1.0, 1.0)
        mask2
    end
    kind = first(r.kind for r in scaled if r.stl !== nothing)
    q_wall_dist = halfway_wall_distances(mask)
    desc = build_geometry_descriptor(kind, mask; q_wall_dist=q_wall_dist)
    stl_src = first(r.stl for r in scaled if r.stl !== nothing)
    scaled_mesh = read_stl(stl_src.file)
    if stl_src.scale != 1.0 || stl_src.translate != (0.0, 0.0, 0.0)
        scaled_mesh = transform_mesh(scaled_mesh; scale=stl_src.scale,
                                     translate=stl_src.translate)
    end
    kappa_max = stl_kappa_max(scaled_mesh)
    L_up_d, L_down_d = obstacle_extents_in_R(mask, units.R_LU)
    L_up = isnan(units.L_up) ? L_up_d : units.L_up
    L_down = isnan(units.L_down) ? L_down_d : units.L_down
    return (type=desc.type, blockage=desc.blockage, q_wall_dist=desc.q_wall_dist,
            stl_hash=desc.stl_hash, kappa_max=kappa_max,
            L_up=L_up, L_down=L_down)
end

function _units_bc_namedtuple(boundaries::Vector{BoundarySetup},
                              regions::Vector{GeometryRegion})
    inlet = :velocity_parabolic
    outlet = :zou_he_pressure
    has_west_periodic = any(b -> b.face === :west && b.type === :periodic, boundaries)
    has_east_periodic = any(b -> b.face === :east && b.type === :periodic, boundaries)
    if has_west_periodic && has_east_periodic
        inlet = :periodic_x
        outlet = :periodic_x
    else
        for b in boundaries
            b.face === :west || continue
            if b.type === :velocity
                ux = get(b.values, :ux, nothing)
                inlet = ux !== nothing && is_spatial(ux) ? :velocity_parabolic : :velocity_uniform
            end
        end
        any(b -> b.face === :east && b.type === :pressure, boundaries) &&
            (outlet = :zou_he_pressure)
    end
    wall_bc = any(r -> r.bc_type === :libb, regions) ? :bouzidi_fl : :halfwayBB
    return (inlet=inlet, outlet=outlet, north_wall=:halfwayBB,
            south_wall=:halfwayBB, wall_bc=wall_bc)
end

function _inject_units_velocity!(boundaries::Vector{BoundarySetup},
                                 user_vars::Dict{Symbol,Float64})
    for b in boundaries
        b.type === :velocity || continue
        for (key, expr) in collect(b.values)
            occursin(r"\bu_LU\b", expr.source) || continue
            b.values[key] = parse_kraken_expr(expr.source, user_vars)
        end
    end
    return nothing
end

