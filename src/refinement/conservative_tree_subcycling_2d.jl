include("subcycling_schedule_2d.jl")
include("subcycling_explosion_2d.jl")
include("subcycling_coalesce_2d.jl")
include("subcycling_streaming_2d.jl")
include("subcycling_wall_phase_2d.jl")























































































































function _stream_conservative_tree_level_native_wall_phase_transport_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D;
        boundary::Symbol=:periodic_x_wall_y,
        u_south=0,
        u_north=0,
        rho_wall=1,
        alpha_c2f=1,
        alpha_f2c=1,
        coarse_to_fine_state::Symbol=:owned,
        pre_stream_level! = nothing,
        schedule=nothing,
        route_bank=nothing,
        state_bank=nothing,
        Fsource=nothing,
        Fscratch=nothing,
        scatter=nothing,
        scatter_source=nothing)
    periodic_x = true
    scatter_run = scatter === nothing ?
        (route_bank === nothing ?
            (schedule === nothing ?
                _prepare_conservative_tree_wall_phase_scatter_2d(
                    spec; periodic_x=periodic_x, T=eltype(Fout)) :
                _prepare_conservative_tree_wall_phase_scatter_2d(
                    spec; periodic_x=periodic_x, schedule=schedule,
                    T=eltype(Fout))) :
            _cached_conservative_tree_wall_phase_scatter_2d(
                route_bank; periodic_x=periodic_x)) :
        scatter
    Fmasked = copy(Fin)
    _mask_conservative_tree_wall_phase_sources_2d!(Fmasked, scatter_run)

    stream_conservative_tree_subcycled_buffered_routes_F_2d!(
        Fout, Fmasked, spec, table; boundary=boundary,
        u_south=u_south, u_north=u_north, rho_wall=rho_wall,
        alpha_c2f=alpha_c2f, alpha_f2c=alpha_f2c,
        interface_time_scaling=:level_native,
        coarse_to_fine_prolongation=:flat,
        coarse_to_fine_state=coarse_to_fine_state,
        coarse_to_fine_predictor_weight=0,
        pre_stream_level! = pre_stream_level!,
        schedule=schedule, route_bank=route_bank, state_bank=state_bank,
        Fsource=Fsource, Fscratch=Fscratch,
        wall_phase_transport_correction=false,
        level_native_delayed_same_level_transit=
            pre_stream_level! === nothing && scatter_source === nothing)

    return _apply_conservative_tree_wall_phase_scatter_2d!(
        Fout, scatter_source === nothing ? Fin : scatter_source, scatter_run)
end





"""
stream_conservative_tree_subcycled_routes_F_2d!(Fout, Fin, spec, table;
                                                    boundary=:skip,
                                                    alpha_c2f=1,
                                                    alpha_f2c=1)

CPU reference transport for the recursive AMR-D subcycle schedule. Same-level
routes are advanced only when their level receives an `:advance` event.
Coarse/fine interface routes are routed through spatial ledgers: split routes
are deposited on `:sync_down`, injected into child rows during each child
advance, and coalesce routes are accumulated during child advances then applied
on `:sync_up`.

This is still a transport skeleton: no collision, forcing, or physical open
boundary closure is performed here.

`alpha_c2f` and `alpha_f2c` rescale the non-equilibrium part of interface
packets. The default `1` preserves the original packet transport exactly.
"""
function stream_conservative_tree_subcycled_routes_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D;
        boundary::Symbol=:skip,
        u_south=0,
        u_north=0,
        rho_wall=1,
        alpha_c2f=1,
        alpha_f2c=1,
        interface_time_scaling::Symbol=:leaf_equivalent,
        is_solid=nothing)
    _check_conservative_tree_stream_args_2d(Fout, Fin, spec)
    policy = _check_conservative_tree_boundary_policy_2d(boundary)
    periodic_x = _conservative_tree_periodic_x_policy_2d(policy)
    is_solid === nothing ||
        validate_conservative_tree_solid_mask_resolved_2d(
            spec, table, is_solid)
    if spec.max_level == 0
        return stream_conservative_tree_routes_F_2d!(
            Fout, Fin, spec, table; boundary=boundary, u_south=u_south,
            u_north=u_north, rho_wall=rho_wall)
    end

    schedule = create_conservative_tree_subcycle_schedule_2d(spec.max_level)
    bank = create_conservative_tree_subcycle_spatial_ledger_bank_2d(
        spec; schedule=schedule, T=eltype(Fout))
    Fstate = copy(Fin)
    Fscratch = similar(Fout)
    Fpending = similar(Fout)
    fill!(Fpending, zero(eltype(Fpending)))

    for event in schedule.events
        if event.phase == :sync_down
            reset_conservative_tree_subcycle_spatial_pair_2d!(
                bank, event.src_level)
            conservative_tree_subcycle_sync_down_routes_F_2d!(
                bank, event, Fstate, table; alpha=alpha_c2f,
                interface_time_scaling=interface_time_scaling,
                periodic_x=periodic_x)
        elseif event.phase == :advance
            fill!(Fscratch, zero(eltype(Fscratch)))
            level = event.src_level
            if level > 0
                _add_and_clear_conservative_tree_level_rows_2d!(
                    Fstate, Fpending, spec, level)
                conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
                    bank, event, Fstate, table; alpha=alpha_f2c,
                    interface_time_scaling=interface_time_scaling,
                    periodic_x=periodic_x)
                conservative_tree_subcycle_accumulate_inactive_parent_routes_F_2d!(
                    bank, event, Fstate; alpha=alpha_f2c,
                    interface_time_scaling=interface_time_scaling,
                    periodic_x=periodic_x)
            end
            _stream_conservative_tree_direct_level_routes_F_2d!(
                Fscratch, Fstate, spec, table, level, policy;
                u_south=u_south, u_north=u_north, rho_wall=rho_wall,
                is_solid=is_solid)
            if level > 0
                conservative_tree_subcycle_apply_child_advance_injection_F_2d!(
                    Fscratch, bank, event)
            end
            _add_and_clear_conservative_tree_level_rows_2d!(
                Fscratch, Fpending, spec, level)
            _copy_conservative_tree_level_rows_2d!(
                Fstate, Fscratch, spec, level)
        elseif event.phase == :sync_up
            conservative_tree_subcycle_apply_sync_up_F_2d!(
                Fpending, bank, event; boundary=boundary, u_south=u_south,
                u_north=u_north, rho_wall=rho_wall)
        else
            throw(ArgumentError("unknown subcycle event phase $(event.phase)"))
        end
    end

    copyto!(Fout, Fstate)
    return Fout
end

"""
stream_conservative_tree_subcycled_buffered_routes_F_2d!(Fout, Fin, spec, table;
                                                             boundary=:skip,
                                                             alpha_c2f=1,
                                                             alpha_f2c=1)

CPU reference transport that drives the recursive AMR-D schedule through the
explicit subcycle buffer contract. Unlike the legacy skeleton, it keeps
committed level state (`owned`), coarse-to-fine injections
(`ghost_from_coarse`), fine-to-coarse reflux (`reflux_to_coarse`), and
fine-to-parent restriction (`restrict_to_parent`) in distinct buffers.

This remains transport-only: no collision, forcing, or physical open-boundary
closure is performed here.

`alpha_c2f` and `alpha_f2c` are the Filippova-Hanel style non-equilibrium
rescaling factors for coarse-to-fine and fine-to-coarse interface packets.
`interface_time_scaling=:leaf_equivalent` keeps the historical packet weights:
coarse-to-fine split packets are distributed across child half-steps and
fine-to-coarse packets are accumulated with the reciprocal half-step weight.
`interface_time_scaling=:level_native` is experimental and only intended for
route tables built with `sampling=:level_native`; it preserves global rest mass
for flat coarse-to-fine packets at nested interfaces.
`coarse_to_fine_predictor_weight` blends coarse-to-fine packets between the
committed parent state (`0`) and a local post-collision predictor (`1`).
Macro-flow runners use a conservative partial blend; transport-only callers
default to `0`.
`pre_stream_level!`, when provided, is called as
`pre_stream_level!(Fsource, spec, level, event)` immediately before direct
routes for that level are streamed. Macro-flow runners use it to apply local
collision at each level's own subcycled advance.
"""
function stream_conservative_tree_subcycled_buffered_routes_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D;
        boundary::Symbol=:skip,
        u_south=0,
        u_north=0,
        rho_wall=1,
        alpha_c2f=1,
        alpha_f2c=1,
        interface_time_scaling::Symbol=:leaf_equivalent,
        coarse_to_fine_prolongation::Symbol=:flat,
        coarse_to_fine_state::Symbol=:owned,
        coarse_to_fine_predictor_weight=0,
        pre_stream_level! = nothing,
        schedule = nothing,
        route_bank = nothing,
        state_bank = nothing,
        Fsource = nothing,
        Fscratch = nothing,
        is_solid=nothing,
        interface_balance::Bool=false,
        wall_phase_transport_correction::Bool=true,
        level_native_delayed_same_level_transit::Bool=true)
    _check_conservative_tree_stream_args_2d(Fout, Fin, spec)
    policy = _check_conservative_tree_boundary_policy_2d(boundary)
    periodic_x = _conservative_tree_periodic_x_policy_2d(policy)
    _check_conservative_tree_coarse_to_fine_state_2d(coarse_to_fine_state)
    _check_conservative_tree_coarse_to_fine_predictor_weight_2d(
        coarse_to_fine_predictor_weight)
    _check_conservative_tree_interface_time_scaling_2d(
        interface_time_scaling)
    _check_conservative_tree_coarse_to_fine_prolongation_2d(
        coarse_to_fine_prolongation)
    if interface_time_scaling == :level_native &&
       coarse_to_fine_prolongation == :limited_linear
        throw(ArgumentError("interface_time_scaling=:level_native is closed only with coarse_to_fine_prolongation=:flat"))
    end
    is_solid === nothing ||
        validate_conservative_tree_solid_mask_resolved_2d(
            spec, table, is_solid)
    if spec.max_level == 0
        return stream_conservative_tree_routes_F_2d!(
            Fout, Fin, spec, table; boundary=boundary, u_south=u_south,
            u_north=u_north, rho_wall=rho_wall)
    end
    if wall_phase_transport_correction &&
       policy == :periodic_x_wall_y &&
       table.sampling == :level_native &&
       interface_time_scaling == :level_native &&
       coarse_to_fine_prolongation == :flat &&
       coarse_to_fine_state == :owned &&
       iszero(coarse_to_fine_predictor_weight) &&
       alpha_c2f == 1 &&
       alpha_f2c == 1 &&
       pre_stream_level! === nothing &&
       route_bank === nothing &&
       state_bank === nothing &&
       Fsource === nothing &&
       Fscratch === nothing &&
       is_solid === nothing
        return _stream_conservative_tree_level_native_wall_phase_transport_F_2d!(
            Fout, Fin, spec, table; boundary=boundary, u_south=u_south,
            u_north=u_north, rho_wall=rho_wall, alpha_c2f=alpha_c2f,
            alpha_f2c=alpha_f2c, coarse_to_fine_state=coarse_to_fine_state,
            schedule=schedule)
    end

    schedule_run = schedule === nothing ?
        create_conservative_tree_subcycle_schedule_2d(spec.max_level) :
        schedule
    _check_conservative_tree_subcycle_spec_schedule_2d(spec, schedule_run)
    route_bank_run = route_bank === nothing ?
        create_conservative_tree_subcycle_spatial_ledger_bank_2d(
            spec; schedule=schedule_run, T=eltype(Fout)) :
        route_bank
    state_bank_run = state_bank === nothing ?
        create_conservative_tree_subcycle_buffer_bank_2d(
            spec; schedule=schedule_run, T=eltype(Fout)) :
        state_bank
    route_bank_run.spec === spec ||
        throw(ArgumentError("route_bank must belong to spec"))
    route_bank_run.schedule === schedule_run ||
        throw(ArgumentError("route_bank schedule must match schedule"))
    state_bank_run.spec === spec ||
        throw(ArgumentError("state_bank must belong to spec"))
    state_bank_run.schedule === schedule_run ||
        throw(ArgumentError("state_bank schedule must match schedule"))
    if !_conservative_tree_subcycle_route_packet_cache_ready_2d(
            route_bank_run, table; periodic_x=periodic_x)
        prepare_conservative_tree_subcycle_route_packet_cache_2d!(
            route_bank_run, table; periodic_x=periodic_x)
    end
    reset_conservative_tree_subcycle_spatial_bank_2d!(route_bank_run)
    reset_conservative_tree_subcycle_buffer_bank_2d!(state_bank_run)
    conservative_tree_subcycle_store_active_owned_2d!(state_bank_run, Fin)
    conservative_tree_subcycle_restrict_all_levels_2d!(state_bank_run)
    Fsource_run = Fsource === nothing ? similar(Fout) : Fsource
    Fscratch_run = Fscratch === nothing ? similar(Fout) : Fscratch
    _check_conservative_tree_F_2d(Fsource_run, spec)
    _check_conservative_tree_F_2d(Fscratch_run, spec)
    if wall_phase_transport_correction &&
       pre_stream_level! !== nothing &&
       policy == :periodic_x_wall_y &&
       table.sampling == :level_native &&
       interface_time_scaling == :level_native &&
       coarse_to_fine_prolongation == :flat &&
       coarse_to_fine_state == :owned &&
       (iszero(coarse_to_fine_predictor_weight) ||
        coarse_to_fine_predictor_weight == 1) &&
       alpha_c2f == 1 &&
       alpha_f2c == 1 &&
       is_solid === nothing
        scatter_run = _cached_conservative_tree_wall_phase_scatter_2d(
            route_bank_run; periodic_x=periodic_x)
        if (iszero(coarse_to_fine_predictor_weight) &&
            !isempty(scatter_run.src_ids)) ||
           (!iszero(coarse_to_fine_predictor_weight) &&
            !isempty(scatter_run.event_mask_src_ids))
            Fpost_run = similar(Fout)
            _conservative_tree_apply_prestream_over_schedule_2d!(
                Fpost_run, Fin, spec, schedule_run, pre_stream_level!)
            return _stream_conservative_tree_level_native_wall_phase_transport_F_2d!(
                Fout, Fpost_run, spec, table; boundary=boundary,
                u_south=u_south, u_north=u_north, rho_wall=rho_wall,
                alpha_c2f=alpha_c2f, alpha_f2c=alpha_f2c,
                coarse_to_fine_state=coarse_to_fine_state,
                pre_stream_level! = nothing,
                schedule=schedule_run, route_bank=route_bank_run,
                state_bank=state_bank_run, Fsource=Fsource_run,
                Fscratch=Fscratch_run, scatter=scatter_run,
                scatter_source=Fpost_run)
        end
    end
    # The substep-keyed scatter table is prepared above as a WIP contract, but
    # remains disabled until its per-event mask scope is exact on BGK+force.
    wall_phase_substep_scatter = nothing
    delayed_same_level_transit =
        level_native_delayed_same_level_transit &&
        pre_stream_level! === nothing
    Fmasked_run = wall_phase_substep_scatter === nothing ?
        Fsource_run : similar(Fout)
    Fwall_source_run = wall_phase_substep_scatter === nothing ?
        Fsource_run : similar(Fout)
    use_phase_level_native_direct =
        interface_time_scaling == :level_native &&
        coarse_to_fine_prolongation == :flat &&
        alpha_c2f == 1 &&
        is_solid === nothing
    phase_level_sources = use_phase_level_native_direct ?
        [similar(Fout) for _ in 1:spec.max_level] :
        typeof(Fout)[]
    phase_level_source_valid = falses(spec.max_level)
    for event in schedule_run.events
        if event.phase == :sync_down
            reset_conservative_tree_subcycle_spatial_pair_2d!(
                route_bank_run, event.src_level)
            parent_owned = state_bank_run.levels[event.src_level + 1].owned
            Fparent = parent_owned
            predictor_weight = coarse_to_fine_state == :postcollision ?
                one(eltype(Fsource_run)) :
                eltype(Fsource_run)(coarse_to_fine_predictor_weight)
            needs_level_source =
                predictor_weight > zero(predictor_weight) ||
                coarse_to_fine_prolongation == :limited_linear ||
                use_phase_level_native_direct ||
                wall_phase_substep_scatter !== nothing
            if needs_level_source
                fill!(Fsource_run, zero(eltype(Fsource_run)))
                _copy_conservative_tree_level_rows_2d!(
                    Fsource_run, parent_owned,
                    state_bank_run.active_ids_by_level[event.src_level + 1])
                conservative_tree_subcycle_apply_restriction_to_inactive_level_F_2d!(
                    Fsource_run, state_bank_run, event.src_level)
                Fparent = Fsource_run
            end
            if predictor_weight > zero(predictor_weight)
                fill!(Fscratch_run, zero(eltype(Fscratch_run)))
                _copy_conservative_tree_level_rows_2d!(
                    Fscratch_run, parent_owned,
                    state_bank_run.active_ids_by_level[event.src_level + 1])
                conservative_tree_subcycle_apply_restriction_to_inactive_level_F_2d!(
                    Fscratch_run, state_bank_run, event.src_level)
                pre_stream_level! === nothing ||
                    pre_stream_level!(Fscratch_run, spec, event.src_level, event)
                @inbounds for (cell_id, cell) in pairs(spec.cells)
                    cell.level == event.src_level || continue
                    for q in 1:9
                        Fsource_run[cell_id, q] =
                            (one(predictor_weight) - predictor_weight) *
                            Fsource_run[cell_id, q] +
                            predictor_weight * Fscratch_run[cell_id, q]
                    end
                end
                Fparent = Fsource_run
            end
            phase_source_ready = use_phase_level_native_direct &&
                (pre_stream_level! === nothing ||
                 predictor_weight == one(predictor_weight))
            phase_safe = phase_source_ready &&
                _conservative_tree_level_native_phase_c2f_safe_2d(
                    route_bank_run, event.src_level)
            if phase_safe
                copyto!(phase_level_sources[event.src_level + 1], Fparent)
                phase_level_source_valid[event.src_level + 1] = true
            elseif use_phase_level_native_direct
                phase_level_source_valid[event.src_level + 1] = false
            end
            Fparent_routes = Fparent
            Fparent_scatter = Fparent
            if wall_phase_substep_scatter !== nothing
                if pre_stream_level! !== nothing &&
                   predictor_weight < one(predictor_weight)
                    copyto!(Fwall_source_run, Fsource_run)
                    pre_stream_level!(
                        Fwall_source_run, spec, event.src_level, event)
                    Fparent_scatter = Fwall_source_run
                end
                if _copy_mask_conservative_tree_wall_phase_event_sources_2d!(
                        Fmasked_run, Fparent, wall_phase_substep_scatter,
                        :sync_down, event.src_level, event.tick)
                    Fparent_routes = Fmasked_run
                end
            end
            conservative_tree_subcycle_sync_down_routes_F_2d!(
                route_bank_run, event, Fparent_routes, table; alpha=alpha_c2f,
                interface_time_scaling=interface_time_scaling,
                coarse_to_fine_prolongation=coarse_to_fine_prolongation,
                periodic_x=periodic_x,
                phase_resolved_level_native=phase_safe)
            wall_phase_substep_scatter === nothing ||
                _apply_conservative_tree_wall_phase_sync_down_scatter_2d!(
                    route_bank_run, event, Fparent_scatter,
                    wall_phase_substep_scatter)
        elseif event.phase == :advance
            level = event.src_level
            buffers = state_bank_run.levels[level + 1]
            fill!(Fsource_run, zero(eltype(Fsource_run)))
            _copy_conservative_tree_level_rows_2d!(
                Fsource_run, buffers.owned,
                state_bank_run.active_ids_by_level[level + 1])
            conservative_tree_subcycle_apply_restriction_to_inactive_level_F_2d!(
                Fsource_run, state_bank_run, level)
            pre_stream_level! === nothing ||
                pre_stream_level!(Fsource_run, spec, level, event)
            Froute_source = Fsource_run
            if wall_phase_substep_scatter !== nothing &&
               _copy_mask_conservative_tree_wall_phase_event_sources_2d!(
                    Fmasked_run, Fsource_run, wall_phase_substep_scatter,
                    :advance, level, event.tick)
                Froute_source = Fmasked_run
            end

            phase_direct_active =
                use_phase_level_native_direct &&
                level < spec.max_level &&
                phase_level_source_valid[level + 1]

            if level > 0
                conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
                    route_bank_run, event, Froute_source, table;
                    alpha=alpha_f2c,
                    interface_time_scaling=interface_time_scaling,
                    delayed_same_level_transit=delayed_same_level_transit,
                    periodic_x=periodic_x)
                conservative_tree_subcycle_accumulate_inactive_parent_routes_F_2d!(
                    route_bank_run, event, Froute_source; alpha=alpha_f2c,
                    interface_time_scaling=interface_time_scaling,
                    delayed_same_level_transit=delayed_same_level_transit,
                    periodic_x=periodic_x)
            end

            fill!(Fscratch_run, zero(eltype(Fscratch_run)))
            Fdirect_source = Froute_source
            if phase_direct_active
                Fdirect_source = phase_level_sources[level + 1]
                if wall_phase_substep_scatter !== nothing &&
                   _copy_mask_conservative_tree_wall_phase_event_sources_2d!(
                        Fmasked_run, Fdirect_source,
                        wall_phase_substep_scatter, :advance, level,
                        event.tick)
                    Fdirect_source = Fmasked_run
                end
            end
            _stream_conservative_tree_direct_level_routes_F_2d!(
                Fscratch_run, Fdirect_source, spec, table, level, policy;
                u_south=u_south, u_north=u_north, rho_wall=rho_wall,
                is_solid=is_solid,
                coarse_to_fine_prolongation=coarse_to_fine_prolongation,
                periodic_x=periodic_x,
                phase_resolved_level_native=phase_direct_active,
                subcycle_ratio=schedule_run.ratio)
            if phase_direct_active
                phase_level_source_valid[level + 1] = false
            end
            conservative_tree_subcycle_apply_delayed_same_level_packets_F_2d!(
                Fscratch_run, route_bank_run, event)
            if wall_phase_substep_scatter !== nothing
                _apply_conservative_tree_wall_phase_advance_scatter_2d!(
                    Fscratch_run, Fsource_run, wall_phase_substep_scatter,
                    level, event.tick)
                _apply_conservative_tree_wall_phase_reflux_scatter_2d!(
                    state_bank_run, Fsource_run, wall_phase_substep_scatter,
                    level, event.tick)
            end
            if level > 0
                fill!(buffers.ghost_from_coarse,
                      zero(eltype(buffers.ghost_from_coarse)))
                conservative_tree_subcycle_apply_child_advance_injection_F_2d!(
                    buffers.ghost_from_coarse, route_bank_run, event)
                conservative_tree_subcycle_add_and_clear_ghost_to_F_level_2d!(
                    Fscratch_run, state_bank_run, level)
            end
            conservative_tree_subcycle_add_and_clear_reflux_to_F_level_2d!(
                Fscratch_run, state_bank_run, level)
            _copy_conservative_tree_active_level_rows_2d!(
                buffers.owned, Fscratch_run,
                state_bank_run.active_ids_by_level[level + 1])
        elseif event.phase == :sync_up
            parent_level = _check_subcycle_spatial_sync_up_event_2d(
                route_bank_run, event)
            conservative_tree_subcycle_restrict_level_2d!(
                state_bank_run, parent_level)
            parent_reflux = state_bank_run.levels[parent_level + 1].reflux_to_coarse
            conservative_tree_subcycle_apply_sync_up_F_2d!(
                parent_reflux, route_bank_run, event; boundary=boundary,
                u_south=u_south, u_north=u_north, rho_wall=rho_wall)
        else
            throw(ArgumentError("unknown subcycle event phase $(event.phase)"))
        end
    end

    conservative_tree_subcycle_collect_active_owned_F_2d!(Fout, state_bank_run)
    return Fout
end

function diagnose_conservative_tree_subcycled_rest_2d(
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D;
        boundary::Symbol=:bounceback,
        T::Type{<:Real}=Float64)
    Fin = allocate_conservative_tree_F_2d(spec; T=T)
    Fout = allocate_conservative_tree_F_2d(spec; T=T)
    w = weights(D2Q9())
    @inbounds for cell_id in spec.active_cells
        volume = T(spec.cells[cell_id].metrics.volume)
        for q in 1:9
            Fin[cell_id, q] = T(w[q]) * volume
        end
    end

    stream_conservative_tree_subcycled_routes_F_2d!(
        Fout, Fin, spec, table; boundary=boundary)

    active_initial = sum(Fin[spec.active_cells, :])
    active_final = sum(Fout[spec.active_cells, :])
    level_drift = zeros(T, spec.max_level + 1)
    @inbounds for level in 0:spec.max_level
        ids = [id for id in spec.active_cells if spec.cells[id].level == level]
        level_drift[level + 1] = sum(Fout[ids, :]) - sum(Fin[ids, :])
    end
    orientation_drift = active_population_sums_F_2d(Fout, spec) -
                        active_population_sums_F_2d(Fin, spec)
    return (active_initial=active_initial,
            active_final=active_final,
            active_drift=active_final - active_initial,
            max_active_abs=maximum(abs.(Fout[spec.active_cells, :] .-
                                        Fin[spec.active_cells, :])),
            level_drift=level_drift,
            orientation_drift=orientation_drift)
end
