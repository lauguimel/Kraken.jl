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
    relative_mass_drift::T
    max_raw_relative_mass_drift::T
    active_cell_count::Int
    leaf_equivalent_cell_count::Int
end

struct ConservativeTreeSpecSolidFlow2D{T}
    flow::Symbol
    max_level::Int
    steps::Int
    spec::ConservativeTreeSpec2D
    table::ConservativeTreeRouteTable2D
    F::Matrix{T}
    is_solid_leaf::BitMatrix
    ux_mean::T
    uy_mean::T
    mass_initial::T
    mass_final::T
    mass_drift::T
    relative_mass_drift::T
    max_raw_relative_mass_drift::T
    active_cell_count::Int
    leaf_equivalent_cell_count::Int
end

function conservative_tree_mass_roundoff_rtol_2d(::Type{T},
                                                 steps::Integer,
                                                 max_level::Integer;
                                                 active_cell_count::Integer=1,
                                                 safety=4096) where T<:AbstractFloat
    nsteps = max(Int(steps), 1)
    levels = max(Int(max_level) + 1, 1)
    cells = max(Int(active_cell_count), 1)
    cell_factor = max(one(T), T(log2(cells + 1)))
    return T(safety) * eps(T) * T(nsteps) * T(levels) * cell_factor
end

function _check_conservative_tree_channel_max_level_2d(max_level::Integer)
    ml = Int(max_level)
    1 <= ml <= 4 ||
        throw(ArgumentError("the reference nested channel currently supports max_level in 1:4"))
    return ml
end

@inline function conservative_tree_leaf_equivalent_level_scale_2d(
        spec::ConservativeTreeSpec2D,
        level::Integer)
    l = Int(level)
    0 <= l <= spec.max_level ||
        throw(ArgumentError("level is outside the conservative-tree spec"))
    return 1 << (spec.max_level - l)
end

function conservative_tree_leaf_equivalent_omega_2d(
        omega,
        spec::ConservativeTreeSpec2D,
        level::Integer)
    scale = conservative_tree_leaf_equivalent_level_scale_2d(spec, level)
    T = typeof(float(omega))
    tau_fine = inv(T(omega))
    tau_fine > T(0.5) ||
        throw(ArgumentError("leaf-equivalent omega requires tau_fine > 0.5"))
    tau_level = T(0.5) + (tau_fine - T(0.5)) / T(scale)
    return inv(tau_level)
end

@inline function conservative_tree_leaf_equivalent_force_2d(
        force,
        spec::ConservativeTreeSpec2D,
        level::Integer)
    scale = conservative_tree_leaf_equivalent_level_scale_2d(spec, level)
    return force * scale
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

function initialize_conservative_tree_solid_equilibrium_F_2d!(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        is_solid::AbstractArray{Bool,2};
        rho=1,
        ux=0,
        uy=0)
    _check_conservative_tree_F_2d(F, spec)
    _check_conservative_tree_leaf_solid_mask_2d(spec, is_solid)
    fill!(F, zero(eltype(F)))
    T = promote_type(eltype(F), typeof(float(rho)), typeof(float(ux)),
                     typeof(float(uy)))
    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        _conservative_tree_cell_is_solid_2d(spec, cell, is_solid) && continue
        volume = T(cell.metrics.volume)
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

function _active_fluid_mass_conservative_tree_F_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        is_solid::AbstractArray{Bool,2})
    _check_conservative_tree_F_2d(F, spec)
    _check_conservative_tree_leaf_solid_mask_2d(spec, is_solid)
    mass = zero(eltype(F))
    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        _conservative_tree_cell_is_solid_2d(spec, cell, is_solid) && continue
        for q in 1:9
            mass += F[cell_id, q]
        end
    end
    return mass
end

@inline function _row_mass_conservative_tree_F_2d(F::AbstractMatrix,
                                                  cell_id::Int)
    mass = zero(eltype(F))
    @inbounds for q in 1:9
        mass += F[cell_id, q]
    end
    return mass
end

@inline function _restore_row_mass_conservative_tree_F_2d!(
        F::AbstractMatrix,
        cell_id::Int,
        mass_before)
    mass_after = _row_mass_conservative_tree_F_2d(F, cell_id)
    @inbounds F[cell_id, 1] += mass_before - mass_after
    return F
end

function _enforce_active_mass_conservation_2d!(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        target_mass;
        rtol)
    mass_now = _active_mass_conservative_tree_F_2d(F, spec)
    drift = mass_now - target_mass
    denom = max(abs(target_mass), eps(typeof(float(target_mass))))
    rel = abs(drift) / denom
    rel <= rtol ||
        throw(ArgumentError("AMR-D mass residual $(rel) exceeds roundoff guard $(rtol)"))
    first_cell = first(spec.active_cells)
    @inbounds F[first_cell, 1] -= drift
    return rel
end

function _enforce_active_fluid_mass_conservation_2d!(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        is_solid::AbstractArray{Bool,2},
        target_mass;
        rtol)
    mass_now = _active_fluid_mass_conservative_tree_F_2d(F, spec, is_solid)
    drift = mass_now - target_mass
    denom = max(abs(target_mass), eps(typeof(float(target_mass))))
    rel = abs(drift) / denom
    rel <= rtol ||
        throw(ArgumentError("AMR-D fluid mass residual $(rel) exceeds roundoff guard $(rtol)"))
    for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        _conservative_tree_cell_is_solid_2d(spec, cell, is_solid) && continue
        @inbounds F[cell_id, 1] -= drift
        return rel
    end
    throw(ArgumentError("AMR-D solid mask leaves no active fluid cell"))
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
        mass_before = _row_mass_conservative_tree_F_2d(F, cell_id)
        collide_BGK_integrated_D2Q9!(@view(F[cell_id, :]),
                                     cell.metrics.volume, omega)
        _restore_row_mass_conservative_tree_F_2d!(F, cell_id, mass_before)
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
        mass_before = _row_mass_conservative_tree_F_2d(F, cell_id)
        collide_Guo_integrated_D2Q9!(@view(F[cell_id, :]),
                                     cell.metrics.volume, omega, Fx, Fy)
        _restore_row_mass_conservative_tree_F_2d!(F, cell_id, mass_before)
    end
    return F
end

function _collide_Guo_conservative_tree_active_fluid_level_F_2d!(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        level::Int,
        is_solid::AbstractArray{Bool,2},
        omega,
        Fx,
        Fy)
    _check_conservative_tree_F_2d(F, spec)
    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        cell.level == level || continue
        _conservative_tree_cell_is_solid_2d(spec, cell, is_solid) && continue
        mass_before = _row_mass_conservative_tree_F_2d(F, cell_id)
        collide_Guo_integrated_D2Q9!(@view(F[cell_id, :]),
                                     cell.metrics.volume, omega, Fx, Fy)
        _restore_row_mass_conservative_tree_F_2d!(F, cell_id, mass_before)
    end
    return F
end

function conservative_tree_leaf_mean_ux_profile_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D;
        force_x=0,
        level_scaled_force::Bool=false)
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
        fx = level_scaled_force ?
             conservative_tree_leaf_equivalent_force_2d(force_x, spec,
                                                        cell.level) :
             force_x
        ux = (mx / volume + fx / 2) / rho
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

function conservative_tree_leaf_fluid_mean_velocity_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        is_solid::AbstractArray{Bool,2};
        force_x=0,
        force_y=0,
        level_scaled_force::Bool=false)
    _check_conservative_tree_F_2d(F, spec)
    _check_conservative_tree_leaf_solid_mask_2d(spec, is_solid)
    sum_ux = zero(eltype(F))
    sum_uy = zero(eltype(F))
    sum_volume = zero(eltype(F))
    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        _conservative_tree_cell_is_solid_2d(spec, cell, is_solid) && continue
        mass = zero(eltype(F))
        mx = zero(eltype(F))
        my = zero(eltype(F))
        for q in 1:9
            Fq = F[cell_id, q]
            mass += Fq
            mx += d2q9_cx(q) * Fq
            my += d2q9_cy(q) * Fq
        end
        volume = eltype(F)(cell.metrics.volume)
        rho = mass / volume
        fx = level_scaled_force ?
             conservative_tree_leaf_equivalent_force_2d(force_x, spec,
                                                        cell.level) :
             force_x
        fy = level_scaled_force ?
             conservative_tree_leaf_equivalent_force_2d(force_y, spec,
                                                        cell.level) :
             force_y
        ux = (mx / volume + fx / 2) / rho
        uy = (my / volume + fy / 2) / rho
        sum_ux += volume * ux
        sum_uy += volume * uy
        sum_volume += volume
    end
    sum_volume > 0 ||
        throw(ArgumentError("AMR-D solid mask leaves no fluid volume"))
    return sum_ux / sum_volume, sum_uy / sum_volume
end

function _subcycled_macroflow_result_2d(flow::Symbol,
                                        steps::Int,
                                        spec::ConservativeTreeSpec2D,
                                        table::ConservativeTreeRouteTable2D,
                                        F::Matrix{T},
                                        profile::AbstractVector,
                                        analytic::AbstractVector,
                                        mass_initial,
                                        max_raw_relative_mass_drift) where T
    profile_T = T.(profile)
    analytic_T = T.(analytic)
    l2, linf = _profile_errors(profile_T, analytic_T)
    mass_final = _active_mass_conservative_tree_F_2d(F, spec)
    denom = max(abs(T(mass_initial)), eps(T))
    relative_mass_drift = abs(mass_final - T(mass_initial)) / denom
    leaf_ny = length(profile_T)
    y = [T(j - 1) / T(leaf_ny - 1) for j in 1:leaf_ny]
    leaf_nx = _conservative_tree_level_size_2d(spec.Nx, spec.max_level)
    return ConservativeTreeSpecMacroFlow2D{T}(
        flow, spec.max_level, steps, spec, table, F, y, profile_T, analytic_T,
        T(l2), T(linf), T(mass_initial), mass_final,
        mass_final - T(mass_initial), relative_mass_drift,
        T(max_raw_relative_mass_drift),
        length(spec.active_cells), leaf_nx * leaf_ny)
end

function _subcycled_solid_flow_result_2d(flow::Symbol,
                                         steps::Int,
                                         spec::ConservativeTreeSpec2D,
                                         table::ConservativeTreeRouteTable2D,
                                         F::Matrix{T},
                                         is_solid::BitMatrix,
                                         force_x,
                                         force_y,
                                         mass_initial,
                                         max_raw_relative_mass_drift) where T
    mass_final = _active_fluid_mass_conservative_tree_F_2d(F, spec, is_solid)
    denom = max(abs(T(mass_initial)), eps(T))
    relative_mass_drift = abs(mass_final - T(mass_initial)) / denom
    ux_mean, uy_mean = conservative_tree_leaf_fluid_mean_velocity_2d(
        F, spec, is_solid; force_x=force_x, force_y=force_y,
        level_scaled_force=true)
    leaf_nx = _conservative_tree_level_size_2d(spec.Nx, spec.max_level)
    leaf_ny = _conservative_tree_level_size_2d(spec.Ny, spec.max_level)
    return ConservativeTreeSpecSolidFlow2D{T}(
        flow, spec.max_level, steps, spec, table, F, is_solid,
        T(ux_mean), T(uy_mean), T(mass_initial), mass_final,
        mass_final - T(mass_initial), relative_mass_drift,
        T(max_raw_relative_mass_drift), length(spec.active_cells),
        leaf_nx * leaf_ny)
end

function _initialize_cartesian_channel_equilibrium_F_2d!(
        F::AbstractArray{T,3},
        volume;
        rho=1,
        ux=0,
        uy=0) where T
    size(F, 3) == 9 ||
        throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    @inbounds for j in axes(F, 2), i in axes(F, 1), q in 1:9
        F[i, j, q] = T(volume) * equilibrium(
            D2Q9(), T(rho), T(ux), T(uy), q)
    end
    return F
end

function run_cartesian_channel_mass_reference_2d(;
        flow::Symbol,
        max_level::Integer,
        steps::Integer,
        omega=1.2,
        Fx=1e-6,
        Fy=0,
        U=1e-3,
        rho0=1,
        T::Type{<:AbstractFloat}=Float64)
    ml = _check_conservative_tree_channel_max_level_2d(max_level)
    nsteps = Int(steps)
    nsteps >= 0 || throw(ArgumentError("steps must be nonnegative"))
    scale = 1 << ml
    nx = 16 * scale
    ny = 12 * scale
    volume = one(T) / T(scale * scale)
    F = zeros(T, nx, ny, 9)
    Ftmp = similar(F)
    _initialize_cartesian_channel_equilibrium_F_2d!(F, volume; rho=rho0)
    mass_initial = sum(F)

    for _ in 1:nsteps
        if flow == :poiseuille || flow == :poiseuille_subcycled
            collide_Guo_integrated_D2Q9!(F, volume, omega, Fx, Fy)
            stream_periodic_x_wall_y_F_2d!(Ftmp, F)
        elseif flow == :couette || flow == :couette_subcycled
            collide_BGK_integrated_D2Q9!(F, volume, omega)
            stream_periodic_x_moving_wall_y_F_2d!(
                Ftmp, F; u_south=zero(T), u_north=U,
                rho_wall=rho0, volume=volume)
        else
            throw(ArgumentError("flow must be :poiseuille or :couette"))
        end
        F, Ftmp = Ftmp, F
    end

    mass_final = sum(F)
    drift = mass_final - mass_initial
    rel = abs(drift) / max(abs(mass_initial), eps(T))
    return (flow=flow, max_level=ml, steps=nsteps,
            mass_initial=mass_initial, mass_final=mass_final,
            mass_drift=drift, relative_mass_drift=rel)
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
        enforce_mass::Bool=true,
        mass_guard_rtol=nothing,
        spec::Union{Nothing,ConservativeTreeSpec2D}=nothing,
        T::Type{<:AbstractFloat}=Float64)
    nsteps = Int(steps)
    nsteps >= 0 || throw(ArgumentError("steps must be nonnegative"))
    spec_run = spec === nothing ?
        create_conservative_tree_nested_channel_spec_2d(max_level) : spec
    spec_run.max_level == Int(max_level) ||
        throw(ArgumentError("max_level must match spec.max_level"))
    table = create_conservative_tree_route_table_2d(spec_run; periodic_x=true)
    F = allocate_conservative_tree_F_2d(spec_run; T=T)
    Ftmp = similar(F)
    schedule = create_conservative_tree_subcycle_schedule_2d(spec_run.max_level)
    route_bank = create_conservative_tree_subcycle_spatial_ledger_bank_2d(
        spec_run; schedule=schedule, T=T)
    prepare_conservative_tree_subcycle_route_packet_cache_2d!(route_bank, table)
    state_bank = create_conservative_tree_subcycle_buffer_bank_2d(
        spec_run; schedule=schedule, T=T)
    Fsource = similar(F)
    Fscratch = similar(F)
    initialize_conservative_tree_equilibrium_F_2d!(F, spec_run; rho=rho0)
    mass_initial = _active_mass_conservative_tree_F_2d(F, spec_run)
    guard = mass_guard_rtol === nothing ?
        conservative_tree_mass_roundoff_rtol_2d(
            T, nsteps, spec_run.max_level;
            active_cell_count=length(spec_run.active_cells)) :
        T(mass_guard_rtol)
    max_raw_relative_mass_drift = zero(T)

    collide_level! = (Flevel, local_spec, level, event) ->
        _collide_Guo_conservative_tree_active_level_F_2d!(
            Flevel, local_spec, level,
            conservative_tree_leaf_equivalent_omega_2d(
                omega, local_spec, level),
            conservative_tree_leaf_equivalent_force_2d(Fx, local_spec, level),
            conservative_tree_leaf_equivalent_force_2d(Fy, local_spec, level))
    for _ in 1:nsteps
        stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Ftmp, F, spec_run, table; boundary=:periodic_x_wall_y,
            alpha_c2f=alpha_c2f, alpha_f2c=alpha_f2c,
            pre_stream_level! = collide_level!,
            schedule=schedule, route_bank=route_bank, state_bank=state_bank,
            Fsource=Fsource, Fscratch=Fscratch)
        if enforce_mass
            raw_rel = _enforce_active_mass_conservation_2d!(
                Ftmp, spec_run, mass_initial; rtol=guard)
            max_raw_relative_mass_drift =
                max(max_raw_relative_mass_drift, T(raw_rel))
        end
        F, Ftmp = Ftmp, F
    end

    profile = conservative_tree_leaf_mean_ux_profile_2d(
        F, spec_run; force_x=Fx, level_scaled_force=true)
    analytic = poiseuille_analytic_profile_2d(length(profile), Fx, omega;
                                              rho=rho0)
    return _subcycled_macroflow_result_2d(
        :poiseuille_subcycled, nsteps, spec_run, table, F, profile, analytic,
        mass_initial, max_raw_relative_mass_drift)
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
        enforce_mass::Bool=true,
        mass_guard_rtol=nothing,
        spec::Union{Nothing,ConservativeTreeSpec2D}=nothing,
        T::Type{<:AbstractFloat}=Float64)
    nsteps = Int(steps)
    nsteps >= 0 || throw(ArgumentError("steps must be nonnegative"))
    spec_run = spec === nothing ?
        create_conservative_tree_nested_channel_spec_2d(max_level) : spec
    spec_run.max_level == Int(max_level) ||
        throw(ArgumentError("max_level must match spec.max_level"))
    table = create_conservative_tree_route_table_2d(spec_run; periodic_x=true)
    F = allocate_conservative_tree_F_2d(spec_run; T=T)
    Ftmp = similar(F)
    schedule = create_conservative_tree_subcycle_schedule_2d(spec_run.max_level)
    route_bank = create_conservative_tree_subcycle_spatial_ledger_bank_2d(
        spec_run; schedule=schedule, T=T)
    prepare_conservative_tree_subcycle_route_packet_cache_2d!(route_bank, table)
    state_bank = create_conservative_tree_subcycle_buffer_bank_2d(
        spec_run; schedule=schedule, T=T)
    Fsource = similar(F)
    Fscratch = similar(F)
    initialize_conservative_tree_equilibrium_F_2d!(F, spec_run; rho=rho0)
    mass_initial = _active_mass_conservative_tree_F_2d(F, spec_run)
    guard = mass_guard_rtol === nothing ?
        conservative_tree_mass_roundoff_rtol_2d(
            T, nsteps, spec_run.max_level;
            active_cell_count=length(spec_run.active_cells)) :
        T(mass_guard_rtol)
    max_raw_relative_mass_drift = zero(T)

    collide_level! = (Flevel, local_spec, level, event) ->
        _collide_BGK_conservative_tree_active_level_F_2d!(
            Flevel, local_spec, level,
            conservative_tree_leaf_equivalent_omega_2d(
                omega, local_spec, level))
    for _ in 1:nsteps
        stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Ftmp, F, spec_run, table; boundary=:periodic_x_moving_wall_y,
            u_south=zero(T), u_north=U, rho_wall=rho0,
            alpha_c2f=alpha_c2f, alpha_f2c=alpha_f2c,
            pre_stream_level! = collide_level!,
            schedule=schedule, route_bank=route_bank, state_bank=state_bank,
            Fsource=Fsource, Fscratch=Fscratch)
        if enforce_mass
            raw_rel = _enforce_active_mass_conservation_2d!(
                Ftmp, spec_run, mass_initial; rtol=guard)
            max_raw_relative_mass_drift =
                max(max_raw_relative_mass_drift, T(raw_rel))
        end
        F, Ftmp = Ftmp, F
    end

    profile = conservative_tree_leaf_mean_ux_profile_2d(F, spec_run)
    analytic = couette_analytic_profile_2d(length(profile), U)
    return _subcycled_macroflow_result_2d(
        :couette_subcycled, nsteps, spec_run, table, F, profile, analytic,
        mass_initial, max_raw_relative_mass_drift)
end

"""
    run_conservative_tree_solid_obstacle_subcycled_2d(; spec, is_solid_leaf,
                                                      Fx=2e-5)

Run a periodic-x / wall-y forced solid-mask case on the AMR-D subcycled
scheduler. This first nested obstacle runner requires the solid mask to be
fully resolved by active cells and away from AMR interfaces.
"""
function run_conservative_tree_solid_obstacle_subcycled_2d(;
        flow::Symbol=:solid_obstacle_subcycled,
        max_level::Integer,
        is_solid_leaf::AbstractArray{Bool,2},
        steps::Integer=100,
        omega=1.0,
        Fx=2e-5,
        Fy=0,
        rho0=1,
        alpha_c2f=1,
        alpha_f2c=1,
        enforce_mass::Bool=true,
        mass_guard_rtol=nothing,
        spec::Union{Nothing,ConservativeTreeSpec2D}=nothing,
        T::Type{<:AbstractFloat}=Float64)
    nsteps = Int(steps)
    nsteps >= 0 || throw(ArgumentError("steps must be nonnegative"))
    spec_run = spec === nothing ?
        create_conservative_tree_nested_channel_spec_2d(max_level) : spec
    spec_run.max_level == Int(max_level) ||
        throw(ArgumentError("max_level must match spec.max_level"))
    solid = BitMatrix(is_solid_leaf)
    _check_conservative_tree_leaf_solid_mask_2d(spec_run, solid)
    table = create_conservative_tree_route_table_2d(spec_run; periodic_x=true)
    validate_conservative_tree_solid_mask_resolved_2d(spec_run, table, solid)

    F = allocate_conservative_tree_F_2d(spec_run; T=T)
    Ftmp = similar(F)
    schedule = create_conservative_tree_subcycle_schedule_2d(spec_run.max_level)
    route_bank = create_conservative_tree_subcycle_spatial_ledger_bank_2d(
        spec_run; schedule=schedule, T=T)
    prepare_conservative_tree_subcycle_route_packet_cache_2d!(route_bank, table)
    state_bank = create_conservative_tree_subcycle_buffer_bank_2d(
        spec_run; schedule=schedule, T=T)
    Fsource = similar(F)
    Fscratch = similar(F)
    initialize_conservative_tree_solid_equilibrium_F_2d!(
        F, spec_run, solid; rho=rho0)
    mass_initial = _active_fluid_mass_conservative_tree_F_2d(
        F, spec_run, solid)
    guard = mass_guard_rtol === nothing ?
        conservative_tree_mass_roundoff_rtol_2d(
            T, nsteps, spec_run.max_level;
            active_cell_count=length(spec_run.active_cells),
            safety=250_000) :
        T(mass_guard_rtol)
    max_raw_relative_mass_drift = zero(T)

    collide_level! = (Flevel, local_spec, level, event) ->
        _collide_Guo_conservative_tree_active_fluid_level_F_2d!(
            Flevel, local_spec, level, solid,
            conservative_tree_leaf_equivalent_omega_2d(
                omega, local_spec, level),
            conservative_tree_leaf_equivalent_force_2d(Fx, local_spec, level),
            conservative_tree_leaf_equivalent_force_2d(Fy, local_spec, level))
    for _ in 1:nsteps
        stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Ftmp, F, spec_run, table; boundary=:periodic_x_wall_y,
            alpha_c2f=alpha_c2f, alpha_f2c=alpha_f2c,
            pre_stream_level! = collide_level!,
            schedule=schedule, route_bank=route_bank, state_bank=state_bank,
            Fsource=Fsource, Fscratch=Fscratch, is_solid=solid)
        if enforce_mass
            raw_rel = _enforce_active_fluid_mass_conservation_2d!(
                Ftmp, spec_run, solid, mass_initial; rtol=guard)
            max_raw_relative_mass_drift =
                max(max_raw_relative_mass_drift, T(raw_rel))
        end
        F, Ftmp = Ftmp, F
    end

    return _subcycled_solid_flow_result_2d(
        flow, nsteps, spec_run, table, F, solid, Fx, Fy, mass_initial,
        max_raw_relative_mass_drift)
end
