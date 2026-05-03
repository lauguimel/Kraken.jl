# Conservative subcycling ledgers for route-native D2Q9 AMR.
#
# This is not yet the full time integrator. It records the packet accounting
# needed for one coarse step and two fine half-steps so interface transfers can
# be tested before they are put in the hot loop.

struct ConservativeTreeSubcycleLedger2D{T}
    ratio::Int
    coarse_to_fine::Array{T,4}
    fine_to_coarse::Matrix{T}
end

function create_conservative_tree_subcycle_ledger_2d(;
        T::Type{<:Real}=Float64,
        ratio::Integer=2)
    r = Int(ratio)
    r == 2 || throw(ArgumentError("conservative-tree subcycling currently requires ratio = 2"))
    return ConservativeTreeSubcycleLedger2D{T}(
        r, zeros(T, 2, 2, 9, r), zeros(T, 9, r))
end

function reset_conservative_tree_subcycle_ledger_2d!(
        ledger::ConservativeTreeSubcycleLedger2D)
    ledger.coarse_to_fine .= 0
    ledger.fine_to_coarse .= 0
    return ledger
end

function conservative_tree_subcycle_weights_2d(ledger::ConservativeTreeSubcycleLedger2D{T}) where T
    ledger.ratio == 2 ||
        throw(ArgumentError("conservative-tree subcycle weights require ratio = 2"))
    return fill(inv(T(ledger.ratio)), ledger.ratio)
end

@inline function _check_subcycle_step_2d(ledger::ConservativeTreeSubcycleLedger2D,
                                         substep::Integer)
    step = Int(substep)
    1 <= step <= ledger.ratio ||
        throw(ArgumentError("substep must be inside 1:$(ledger.ratio)"))
    return step
end

function conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!(
        ledger::ConservativeTreeSubcycleLedger2D,
        Fq,
        q::Integer,
        face::Symbol)
    qi = _check_d2q9_q(Int(q))
    weights = conservative_tree_subcycle_weights_2d(ledger)
    @inbounds for substep in 1:ledger.ratio
        split_coarse_to_fine_face_F_2d!(
            @view(ledger.coarse_to_fine[:, :, :, substep]),
            Fq * weights[substep], qi, face)
    end
    return ledger
end

function conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!(
        ledger::ConservativeTreeSubcycleLedger2D,
        Fq,
        q::Integer,
        corner::Symbol)
    qi = _check_d2q9_q(Int(q))
    weights = conservative_tree_subcycle_weights_2d(ledger)
    @inbounds for substep in 1:ledger.ratio
        split_coarse_to_fine_corner_F_2d!(
            @view(ledger.coarse_to_fine[:, :, :, substep]),
            Fq * weights[substep], qi, corner)
    end
    return ledger
end

function conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
        ledger::ConservativeTreeSubcycleLedger2D,
        fine_half_step::AbstractArray{<:Any,3},
        q::Integer,
        face::Symbol,
        substep::Integer)
    _check_child_block_2d(fine_half_step, "fine_half_step")
    qi = _check_d2q9_q(Int(q))
    step = _check_subcycle_step_2d(ledger, substep)
    ledger.fine_to_coarse[qi, step] +=
        coalesce_fine_to_coarse_face_F(fine_half_step, qi, face)
    return ledger
end

function conservative_tree_subcycle_accumulate_fine_to_coarse_corner_2d!(
        ledger::ConservativeTreeSubcycleLedger2D,
        fine_half_step::AbstractArray{<:Any,3},
        q::Integer,
        corner::Symbol,
        substep::Integer)
    _check_child_block_2d(fine_half_step, "fine_half_step")
    qi = _check_d2q9_q(Int(q))
    step = _check_subcycle_step_2d(ledger, substep)
    ledger.fine_to_coarse[qi, step] +=
        coalesce_fine_to_coarse_corner_F(fine_half_step, qi, corner)
    return ledger
end

function conservative_tree_subcycle_orientation_sums_2d(
        ledger::ConservativeTreeSubcycleLedger2D{T}) where T
    coarse_to_fine = zeros(T, 9)
    fine_to_coarse = zeros(T, 9)
    @inbounds for substep in 1:ledger.ratio, q in 1:9
        fine_to_coarse[q] += ledger.fine_to_coarse[q, substep]
        for j in 1:2, i in 1:2
            coarse_to_fine[q] += ledger.coarse_to_fine[i, j, q, substep]
        end
    end
    return (coarse_to_fine=coarse_to_fine, fine_to_coarse=fine_to_coarse)
end

function conservative_tree_subcycle_total_sums_2d(
        ledger::ConservativeTreeSubcycleLedger2D)
    sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
    return (coarse_to_fine=sum(sums.coarse_to_fine),
            fine_to_coarse=sum(sums.fine_to_coarse))
end
