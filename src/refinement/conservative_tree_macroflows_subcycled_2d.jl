# Macro-flow runners over the recursive AMR-D subcycled route scheduler.

struct ConservativeTreeSpecMacroFlow2D{T}
    flow::Symbol
    max_level::Int
    steps::Int
    spec::ConservativeTreeSpec2D
    table::ConservativeTreeRouteTable2D
    F::Matrix{T}
    y::Vector{T}
    ux_profile::Vector{T}
    analytic_profile::Vector{T}
    l2_error::T
    linf_error::T
    mass_initial::T
    mass_final::T
    mass_drift::T
    active_cell_count::Int
    leaf_equivalent_cell_count::Int
end

function _check_conservative_tree_channel_max_level_2d(max_level::Integer)
    ml = Int(max_level)
    1 <= ml <= 4 ||
        throw(ArgumentError("the reference nested channel currently supports max_level in 1:4"))
    return ml
end

function _nested_channel_refine_blocks_2d(max_level::Integer)
    ml = _check_conservative_tree_channel_max_level_2d(max_level)
    blocks = ConservativeTreeRefineBlock2D[
        ConservativeTreeRefineBlock2D("L1", 5:12, 3:10),
    ]
    ml >= 2 && push!(blocks,
        ConservativeTreeRefineBlock2D("L2", 13:20, 7:14; parent="L1"))
    ml >= 3 && push!(blocks,
        ConservativeTreeRefineBlock2D("L3", 29:36, 17:24; parent="L2"))
    ml >= 4 && push!(blocks,
        ConservativeTreeRefineBlock2D("L4", 61:68, 37:44; parent="L3"))
    return blocks
end

function create_conservative_tree_nested_channel_spec_2d(max_level::Integer;
                                                         Nx::Integer=16,
                                                         Ny::Integer=12)
    Int(Nx) == 16 && Int(Ny) == 12 ||
        throw(ArgumentError("the reference nested channel block set is defined for Nx=16, Ny=12"))
    return create_conservative_tree_spec_2d(
        Int(Nx), Int(Ny), _nested_channel_refine_blocks_2d(max_level))
end

function initialize_conservative_tree_equilibrium_F_2d!(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D;
        rho=1,
        ux=0,
        uy=0)
    _check_conservative_tree_F_2d(F, spec)
    fill!(F, zero(eltype(F)))
    T = promote_type(eltype(F), typeof(float(rho)), typeof(float(ux)),
                     typeof(float(uy)))
    @inbounds for cell_id in spec.active_cells
        volume = T(spec.cells[cell_id].metrics.volume)
        for q in 1:9
            F[cell_id, q] = volume * equilibrium(
                D2Q9(), T(rho), T(ux), T(uy), q)
        end
    end
    return F
end

function _active_mass_conservative_tree_F_2d(F::AbstractMatrix,
                                             spec::ConservativeTreeSpec2D)
    _check_conservative_tree_F_2d(F, spec)
    mass = zero(eltype(F))
    @inbounds for cell_id in spec.active_cells, q in 1:9
        mass += F[cell_id, q]
    end
    return mass
end

function _collide_BGK_conservative_tree_active_level_F_2d!(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        level::Int,
        omega)
    _check_conservative_tree_F_2d(F, spec)
    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        cell.level == level || continue
        collide_BGK_integrated_D2Q9!(@view(F[cell_id, :]),
                                     cell.metrics.volume, omega)
    end
    return F
end

function _collide_Guo_conservative_tree_active_level_F_2d!(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        level::Int,
        omega,
        Fx,
        Fy)
    _check_conservative_tree_F_2d(F, spec)
    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        cell.level == level || continue
        collide_Guo_integrated_D2Q9!(@view(F[cell_id, :]),
                                     cell.metrics.volume, omega, Fx, Fy)
    end
    return F
end

function conservative_tree_leaf_mean_ux_profile_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D;
        force_x=0)
    _check_conservative_tree_F_2d(F, spec)
    leaf_ny = _conservative_tree_level_size_2d(spec.Ny, spec.max_level)
    row_mass = zeros(eltype(F), leaf_ny)
    row_ux_mass = zeros(eltype(F), leaf_ny)

    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        scale = 1 << (spec.max_level - cell.level)
        mass = zero(eltype(F))
        mx = zero(eltype(F))
        for q in 1:9
            Fq = F[cell_id, q]
            mass += Fq
            mx += d2q9_cx(q) * Fq
        end
        volume = eltype(F)(cell.metrics.volume)
        rho = mass / volume
        ux = (mx / volume + force_x / 2) / rho
        row_packet = mass / scale
        for sj in 1:scale
            jf = (cell.j - 1) * scale + sj
            row_mass[jf] += row_packet
            row_ux_mass[jf] += row_packet * ux
        end
    end

    profile = similar(row_ux_mass)
    @inbounds for j in eachindex(profile)
        profile[j] = row_ux_mass[j] / row_mass[j]
    end
    return profile
end

function _subcycled_macroflow_result_2d(flow::Symbol,
                                        steps::Int,
                                        spec::ConservativeTreeSpec2D,
                                        table::ConservativeTreeRouteTable2D,
                                        F::Matrix{T},
                                        profile::AbstractVector,
                                        analytic::AbstractVector,
                                        mass_initial) where T
    profile_T = T.(profile)
    analytic_T = T.(analytic)
    l2, linf = _profile_errors(profile_T, analytic_T)
    mass_final = _active_mass_conservative_tree_F_2d(F, spec)
    leaf_ny = length(profile_T)
    y = [T(j - 1) / T(leaf_ny - 1) for j in 1:leaf_ny]
    leaf_nx = _conservative_tree_level_size_2d(spec.Nx, spec.max_level)
    return ConservativeTreeSpecMacroFlow2D{T}(
        flow, spec.max_level, steps, spec, table, F, y, profile_T, analytic_T,
        T(l2), T(linf), T(mass_initial), mass_final,
        mass_final - T(mass_initial),
        length(spec.active_cells), leaf_nx * leaf_ny)
end

"""
    run_conservative_tree_poiseuille_subcycled_2d(; max_level, steps=100,
                                                  omega=1.2, Fx=1e-6)

Run a forced periodic-x / wall-y channel on the recursive AMR-D route scheduler
with a nested reference tree whose `max_level` can be 1, 2, 3, or 4.
"""
function run_conservative_tree_poiseuille_subcycled_2d(;
        max_level::Integer,
        steps::Integer=100,
        omega=1.2,
        Fx=1e-6,
        Fy=0,
        rho0=1,
        alpha_c2f=1,
        alpha_f2c=1,
        T::Type{<:Real}=Float64)
    nsteps = Int(steps)
    nsteps >= 0 || throw(ArgumentError("steps must be nonnegative"))
    spec = create_conservative_tree_nested_channel_spec_2d(max_level)
    table = create_conservative_tree_route_table_2d(spec; periodic_x=true)
    F = allocate_conservative_tree_F_2d(spec; T=T)
    Ftmp = similar(F)
    initialize_conservative_tree_equilibrium_F_2d!(F, spec; rho=rho0)
    mass_initial = _active_mass_conservative_tree_F_2d(F, spec)

    collide_level! = (Flevel, local_spec, level, event) ->
        _collide_Guo_conservative_tree_active_level_F_2d!(
            Flevel, local_spec, level, omega, Fx, Fy)
    for _ in 1:nsteps
        stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Ftmp, F, spec, table; boundary=:periodic_x_wall_y,
            alpha_c2f=alpha_c2f, alpha_f2c=alpha_f2c,
            pre_stream_level! = collide_level!)
        F, Ftmp = Ftmp, F
    end

    profile = conservative_tree_leaf_mean_ux_profile_2d(
        F, spec; force_x=Fx)
    analytic = poiseuille_analytic_profile_2d(length(profile), Fx, omega;
                                              rho=rho0)
    return _subcycled_macroflow_result_2d(
        :poiseuille_subcycled, nsteps, spec, table, F, profile, analytic,
        mass_initial)
end

"""
    run_conservative_tree_couette_subcycled_2d(; max_level, steps=100,
                                               omega=1.2, U=1e-3)

Run a periodic-x channel with stationary south wall and moving north wall on
the recursive AMR-D route scheduler.
"""
function run_conservative_tree_couette_subcycled_2d(;
        max_level::Integer,
        steps::Integer=100,
        omega=1.2,
        U=1e-3,
        rho0=1,
        alpha_c2f=1,
        alpha_f2c=1,
        T::Type{<:Real}=Float64)
    nsteps = Int(steps)
    nsteps >= 0 || throw(ArgumentError("steps must be nonnegative"))
    spec = create_conservative_tree_nested_channel_spec_2d(max_level)
    table = create_conservative_tree_route_table_2d(spec; periodic_x=true)
    F = allocate_conservative_tree_F_2d(spec; T=T)
    Ftmp = similar(F)
    initialize_conservative_tree_equilibrium_F_2d!(F, spec; rho=rho0)
    mass_initial = _active_mass_conservative_tree_F_2d(F, spec)

    collide_level! = (Flevel, local_spec, level, event) ->
        _collide_BGK_conservative_tree_active_level_F_2d!(
            Flevel, local_spec, level, omega)
    for _ in 1:nsteps
        stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Ftmp, F, spec, table; boundary=:periodic_x_moving_wall_y,
            u_south=zero(T), u_north=U, rho_wall=rho0,
            alpha_c2f=alpha_c2f, alpha_f2c=alpha_f2c,
            pre_stream_level! = collide_level!)
        F, Ftmp = Ftmp, F
    end

    profile = conservative_tree_leaf_mean_ux_profile_2d(F, spec)
    analytic = couette_analytic_profile_2d(length(profile), U)
    return _subcycled_macroflow_result_2d(
        :couette_subcycled, nsteps, spec, table, F, profile, analytic,
        mass_initial)
end
