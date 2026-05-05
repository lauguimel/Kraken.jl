# Explicit state buffers for the AMR-D subcycling algorithm.
#
# These buffers separate the roles that were previously conflated in one
# mutable Fstate matrix. They are intentionally dense CPU reference buffers;
# production storage can later map the same contract to sparse blocks/GPU pools.

struct ConservativeTreeSubcycleLevelBuffers2D{T}
    owned::Matrix{T}
    ghost_from_coarse::Matrix{T}
    reflux_to_coarse::Matrix{T}
    restrict_to_parent::Matrix{T}
end

struct ConservativeTreeSubcycleBufferBank2D{T}
    spec::ConservativeTreeSpec2D
    schedule::ConservativeTreeSubcycleSchedule2D
    levels::Vector{ConservativeTreeSubcycleLevelBuffers2D{T}}
end

function _check_conservative_tree_subcycle_buffer_level_2d(
        spec::ConservativeTreeSpec2D,
        level::Integer)
    l = Int(level)
    0 <= l <= spec.max_level ||
        throw(ArgumentError("level is outside the conservative-tree spec"))
    return l
end

function _conservative_tree_subcycle_level_buffers_2d(
        bank::ConservativeTreeSubcycleBufferBank2D,
        level::Integer)
    l = _check_conservative_tree_subcycle_buffer_level_2d(bank.spec, level)
    return bank.levels[l + 1]
end

function create_conservative_tree_subcycle_level_buffers_2d(
        spec::ConservativeTreeSpec2D;
        T::Type{<:Real}=Float64)
    return ConservativeTreeSubcycleLevelBuffers2D{T}(
        allocate_conservative_tree_F_2d(spec; T=T),
        allocate_conservative_tree_F_2d(spec; T=T),
        allocate_conservative_tree_F_2d(spec; T=T),
        allocate_conservative_tree_F_2d(spec; T=T))
end

function create_conservative_tree_subcycle_buffer_bank_2d(
        spec::ConservativeTreeSpec2D;
        schedule::ConservativeTreeSubcycleSchedule2D=
            create_conservative_tree_subcycle_schedule_2d(spec.max_level),
        T::Type{<:Real}=Float64)
    spec.max_level == schedule.max_level ||
        throw(ArgumentError("subcycle buffer schedule max_level must match the tree spec"))
    schedule.ratio == 2 ||
        throw(ArgumentError("AMR-D subcycle buffers currently require ratio = 2"))
    levels = [
        create_conservative_tree_subcycle_level_buffers_2d(spec; T=T)
        for _ in 0:spec.max_level
    ]
    return ConservativeTreeSubcycleBufferBank2D{T}(spec, schedule, levels)
end

function reset_conservative_tree_subcycle_level_buffers_2d!(
        bank::ConservativeTreeSubcycleBufferBank2D,
        level::Integer)
    buffers = _conservative_tree_subcycle_level_buffers_2d(bank, level)
    fill!(buffers.owned, zero(eltype(buffers.owned)))
    fill!(buffers.ghost_from_coarse, zero(eltype(buffers.ghost_from_coarse)))
    fill!(buffers.reflux_to_coarse, zero(eltype(buffers.reflux_to_coarse)))
    fill!(buffers.restrict_to_parent, zero(eltype(buffers.restrict_to_parent)))
    return bank
end

function reset_conservative_tree_subcycle_buffer_bank_2d!(
        bank::ConservativeTreeSubcycleBufferBank2D)
    for level in 0:bank.spec.max_level
        reset_conservative_tree_subcycle_level_buffers_2d!(bank, level)
    end
    return bank
end

function _conservative_tree_subcycle_level_row_ids_2d(
        spec::ConservativeTreeSpec2D,
        level::Integer;
        active_only::Bool=true)
    l = _check_conservative_tree_subcycle_buffer_level_2d(spec, level)
    ids = Int[]
    @inbounds for (cell_id, cell) in pairs(spec.cells)
        cell.level == l || continue
        active_only && !cell.active && continue
        push!(ids, cell_id)
    end
    return ids
end

function conservative_tree_subcycle_store_owned_level_2d!(
        bank::ConservativeTreeSubcycleBufferBank2D,
        F::AbstractMatrix,
        level::Integer;
        active_only::Bool=true)
    _check_conservative_tree_F_2d(F, bank.spec)
    l = _check_conservative_tree_subcycle_buffer_level_2d(bank.spec, level)
    buffers = bank.levels[l + 1]
    @inbounds for cell_id in _conservative_tree_subcycle_level_row_ids_2d(
            bank.spec, l; active_only=active_only)
        for q in 1:9
            buffers.owned[cell_id, q] = F[cell_id, q]
        end
    end
    return bank
end

function conservative_tree_subcycle_store_active_owned_2d!(
        bank::ConservativeTreeSubcycleBufferBank2D,
        F::AbstractMatrix)
    _check_conservative_tree_F_2d(F, bank.spec)
    for level in 0:bank.spec.max_level
        conservative_tree_subcycle_store_owned_level_2d!(
            bank, F, level; active_only=true)
    end
    return bank
end

function conservative_tree_subcycle_restore_owned_level_2d!(
        F::AbstractMatrix,
        bank::ConservativeTreeSubcycleBufferBank2D,
        level::Integer;
        active_only::Bool=true)
    _check_conservative_tree_F_2d(F, bank.spec)
    l = _check_conservative_tree_subcycle_buffer_level_2d(bank.spec, level)
    buffers = bank.levels[l + 1]
    @inbounds for cell_id in _conservative_tree_subcycle_level_row_ids_2d(
            bank.spec, l; active_only=active_only)
        for q in 1:9
            F[cell_id, q] = buffers.owned[cell_id, q]
        end
    end
    return F
end

function conservative_tree_subcycle_apply_reflux_to_owned_level_2d!(
        bank::ConservativeTreeSubcycleBufferBank2D,
        level::Integer;
        clear::Bool=true)
    l = _check_conservative_tree_subcycle_buffer_level_2d(bank.spec, level)
    buffers = bank.levels[l + 1]
    @inbounds for cell_id in _conservative_tree_subcycle_level_row_ids_2d(
            bank.spec, l; active_only=true)
        for q in 1:9
            buffers.owned[cell_id, q] += buffers.reflux_to_coarse[cell_id, q]
            if clear
                buffers.reflux_to_coarse[cell_id, q] =
                    zero(eltype(buffers.reflux_to_coarse))
            end
        end
    end
    return bank
end

function conservative_tree_subcycle_add_and_clear_reflux_to_F_level_2d!(
        F::AbstractMatrix,
        bank::ConservativeTreeSubcycleBufferBank2D,
        level::Integer)
    _check_conservative_tree_F_2d(F, bank.spec)
    l = _check_conservative_tree_subcycle_buffer_level_2d(bank.spec, level)
    buffers = bank.levels[l + 1]
    @inbounds for cell_id in _conservative_tree_subcycle_level_row_ids_2d(
            bank.spec, l; active_only=true)
        for q in 1:9
            F[cell_id, q] += buffers.reflux_to_coarse[cell_id, q]
            buffers.reflux_to_coarse[cell_id, q] =
                zero(eltype(buffers.reflux_to_coarse))
        end
    end
    return F
end

function conservative_tree_subcycle_add_and_clear_ghost_to_F_level_2d!(
        F::AbstractMatrix,
        bank::ConservativeTreeSubcycleBufferBank2D,
        level::Integer)
    _check_conservative_tree_F_2d(F, bank.spec)
    l = _check_conservative_tree_subcycle_buffer_level_2d(bank.spec, level)
    buffers = bank.levels[l + 1]
    @inbounds for cell_id in _conservative_tree_subcycle_level_row_ids_2d(
            bank.spec, l; active_only=true)
        for q in 1:9
            F[cell_id, q] += buffers.ghost_from_coarse[cell_id, q]
            buffers.ghost_from_coarse[cell_id, q] =
                zero(eltype(buffers.ghost_from_coarse))
        end
    end
    return F
end

function conservative_tree_subcycle_restrict_level_2d!(
        bank::ConservativeTreeSubcycleBufferBank2D,
        parent_level::Integer)
    parent_l = _check_conservative_tree_subcycle_buffer_level_2d(
        bank.spec, parent_level)
    parent_l < bank.spec.max_level ||
        throw(ArgumentError("parent_level must be below max_level"))
    parent_buffers = bank.levels[parent_l + 1]
    child_buffers = bank.levels[parent_l + 2]
    fill!(parent_buffers.restrict_to_parent,
          zero(eltype(parent_buffers.restrict_to_parent)))

    @inbounds for (parent_id, parent) in pairs(bank.spec.cells)
        parent.level == parent_l || continue
        children = bank.spec.children[parent_id]
        children == (0, 0, 0, 0) && continue
        for child_id in children
            child = bank.spec.cells[child_id]
            source = child.active ? child_buffers.owned :
                     child_buffers.restrict_to_parent
            for q in 1:9
                parent_buffers.restrict_to_parent[parent_id, q] +=
                    source[child_id, q]
            end
        end
    end
    return bank
end

function conservative_tree_subcycle_restrict_all_levels_2d!(
        bank::ConservativeTreeSubcycleBufferBank2D)
    for parent_level in (bank.spec.max_level - 1):-1:0
        conservative_tree_subcycle_restrict_level_2d!(bank, parent_level)
    end
    return bank
end

function conservative_tree_subcycle_apply_restriction_to_inactive_level_F_2d!(
        F::AbstractMatrix,
        bank::ConservativeTreeSubcycleBufferBank2D,
        level::Integer)
    _check_conservative_tree_F_2d(F, bank.spec)
    l = _check_conservative_tree_subcycle_buffer_level_2d(bank.spec, level)
    buffers = bank.levels[l + 1]
    @inbounds for (cell_id, cell) in pairs(bank.spec.cells)
        cell.level == l || continue
        cell.active && continue
        bank.spec.children[cell_id] == (0, 0, 0, 0) && continue
        for q in 1:9
            F[cell_id, q] = buffers.restrict_to_parent[cell_id, q]
        end
    end
    return F
end

function conservative_tree_subcycle_prolong_F_to_child_ghost_2d!(
        bank::ConservativeTreeSubcycleBufferBank2D,
        Fparent::AbstractMatrix,
        parent_level::Integer)
    _check_conservative_tree_F_2d(Fparent, bank.spec)
    parent_l = _check_conservative_tree_subcycle_buffer_level_2d(
        bank.spec, parent_level)
    parent_l < bank.spec.max_level ||
        throw(ArgumentError("parent_level must be below max_level"))
    child_buffers = bank.levels[parent_l + 2]
    fill!(child_buffers.ghost_from_coarse,
          zero(eltype(child_buffers.ghost_from_coarse)))

    @inbounds for (parent_id, parent) in pairs(bank.spec.cells)
        parent.level == parent_l || continue
        children = bank.spec.children[parent_id]
        children == (0, 0, 0, 0) && continue
        for child_id in children, q in 1:9
            child_buffers.ghost_from_coarse[child_id, q] =
                Fparent[parent_id, q] / 4
        end
    end
    return bank
end

function conservative_tree_subcycle_collect_active_owned_F_2d!(
        Fout::AbstractMatrix,
        bank::ConservativeTreeSubcycleBufferBank2D)
    _check_conservative_tree_F_2d(Fout, bank.spec)
    fill!(Fout, zero(eltype(Fout)))
    @inbounds for cell_id in bank.spec.active_cells
        level = bank.spec.cells[cell_id].level
        owned = bank.levels[level + 1].owned
        for q in 1:9
            Fout[cell_id, q] = owned[cell_id, q]
        end
    end
    return Fout
end
