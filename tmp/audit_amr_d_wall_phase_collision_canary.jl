using Kraken
using Printf

const OUTDIR = joinpath(
    dirname(@__DIR__), "benchmarks", "results", "quicklook",
    "amr_d_wall_phase_collision_canary_20260511")

function wall_touch_max_level_2_spec()
    return create_conservative_tree_spec_2d(8, 4, [
        ConservativeTreeRefineBlock2D("L1", 4:5, 1:4),
        ConservativeTreeRefineBlock2D("L2", 8:9, 1:8; parent="L1"),
    ])
end

function wall_touch_xband_max2_spec()
    return create_conservative_tree_spec_2d(16, 12, [
        ConservativeTreeRefineBlock2D("X1", 5:12, 1:12),
        ConservativeTreeRefineBlock2D("X2", 11:22, 1:24; parent="X1"),
    ])
end

function bulk_touch_max_level_2_spec()
    return create_conservative_tree_spec_2d(8, 8, [
        ConservativeTreeRefineBlock2D("L1", 4:5, 3:6),
        ConservativeTreeRefineBlock2D("L2", 8:9, 7:10; parent="L1"),
    ])
end

function cell_leaf_bounds(spec, cell_id::Int)
    cell = spec.cells[cell_id]
    scale = 1 << (spec.max_level - cell.level)
    return (cell.i - 1) * scale + 1, cell.i * scale,
           (cell.j - 1) * scale + 1, cell.j * scale
end

function fill_leaf_equilibrium!(leaf, volume; rho=1.0, ux=0.0, uy=0.0)
    @inbounds for j in axes(leaf, 2), i in axes(leaf, 1), q in 1:9
        leaf[i, j, q] = volume * equilibrium(D2Q9(), rho, ux, uy, q)
    end
    return leaf
end

function restrict_leaf_to_tree!(F, leaf, spec)
    fill!(F, 0.0)
    @inbounds for cell_id in spec.active_cells
        i0, i1, j0, j1 = cell_leaf_bounds(spec, cell_id)
        for q in 1:9
            total = 0.0
            for j in j0:j1, i in i0:i1
                total += leaf[i, j, q]
            end
            F[cell_id, q] = total
        end
    end
    return F
end

function add_odd_diagonal_packet!(leaf, spec; level::Int, i::Int, j::Int,
                                  qplus::Int, amp::Float64)
    src = conservative_tree_cell_id_2d(spec, level, i, j)
    src > 0 || throw(ArgumentError("missing active source cell"))
    qminus = d2q9_opposite(qplus)
    i0, i1, j0, j1 = cell_leaf_bounds(spec, src)
    n = (i1 - i0 + 1) * (j1 - j0 + 1)
    delta = amp / n
    @inbounds for jj in j0:j1, ii in i0:i1
        leaf[ii, jj, qplus] += delta
        leaf[ii, jj, qminus] -= delta
    end
    return leaf
end

function phase_advance_leaf(spec, i::Int, j::Int, q::Int)
    leaf_nx = spec.Nx << spec.max_level
    leaf_ny = spec.Ny << spec.max_level
    trial_i = mod1(i + d2q9_cx(q), leaf_nx)
    trial_j = j + d2q9_cy(q)
    if !(1 <= trial_j <= leaf_ny)
        return i, j, d2q9_opposite(q), true
    end
    return trial_i, trial_j, q, false
end

function source_q_can_touch_wall(spec, src_id::Int, q::Int)
    d2q9_cy(q) == 0 && return false
    i0, i1, j0, j1 = cell_leaf_bounds(spec, src_id)
    ticks = 1 << spec.max_level
    leaf_ny = spec.Ny << spec.max_level
    return d2q9_cy(q) < 0 ? j0 <= ticks : j1 > leaf_ny - ticks
end

function wall_phase_cone_cells(spec)
    cells = Set{Int}()
    ticks = 1 << spec.max_level
    @inbounds for src_id in spec.active_cells, q in 1:9
        source_q_can_touch_wall(spec, src_id, q) || continue
        push!(cells, src_id)
        i0, i1, j0, j1 = cell_leaf_bounds(spec, src_id)
        for sj in j0:j1, si in i0:i1
            pos_i = si
            pos_j = sj
            qcur = q
            hit_wall = false
            for _ in 1:ticks
                pos_i, pos_j, qcur, wall = phase_advance_leaf(
                    spec, pos_i, pos_j, qcur)
                hit_wall |= wall
            end
            hit_wall || continue
            dst = Kraken._active_leaf_covering_sample_2d(
                spec, spec.max_level, pos_i, pos_j)
            dst == 0 || push!(cells, dst)
        end
    end
    return cells
end

function rest_equilibrium_tree(spec)
    F = allocate_conservative_tree_F_2d(spec; T=Float64)
    initialize_conservative_tree_equilibrium_F_2d!(F, spec; rho=1.0)
    return F
end

function run_leaf_bgk_stream!(leaf, scratch, volume, omega, ticks::Int)
    for _ in 1:ticks
        collide_BGK_integrated_D2Q9!(leaf, volume, omega)
        stream_periodic_x_wall_y_F_2d!(scratch, leaf)
        leaf, scratch = scratch, leaf
    end
    return leaf
end

function run_amr_stream_with_prestream(spec, Fin; mode::Symbol, omega=1.0)
    table = create_conservative_tree_route_table_2d(
        spec; periodic_x=true, sampling=:level_native)
    Fout = similar(Fin)
    schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(spec.max_level)
    route_bank = Kraken.create_conservative_tree_subcycle_spatial_ledger_bank_2d(
        spec; schedule=schedule, T=eltype(Fin))
    state_bank = Kraken.create_conservative_tree_subcycle_buffer_bank_2d(
        spec; schedule=schedule, T=eltype(Fin))
    active_ids_by_level = state_bank.active_ids_by_level
    Fsource = similar(Fin)
    Fscratch = similar(Fin)

    pre_stream_level! = if mode == :noop
        (Flevel, local_spec, level, event) -> nothing
    elseif mode == :bgk
        (Flevel, local_spec, level, event) ->
            Kraken._collide_BGK_conservative_tree_active_ids_F_2d!(
                Flevel, local_spec, active_ids_by_level[level + 1],
                Kraken.conservative_tree_leaf_equivalent_omega_2d(
                    omega, local_spec, level))
    else
        throw(ArgumentError("mode must be :noop or :bgk"))
    end

    Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
        Fout, Fin, spec, table; boundary=:periodic_x_wall_y,
        interface_time_scaling=:level_native,
        coarse_to_fine_predictor_weight=0,
        pre_stream_level! = pre_stream_level!,
        schedule=schedule, route_bank=route_bank, state_bank=state_bank,
        Fsource=Fsource, Fscratch=Fscratch)
    return Fout
end

run_amr_bgk_stream(spec, Fin; omega=1.2) =
    run_amr_stream_with_prestream(spec, Fin; mode=:bgk, omega=omega)

function run_rest_gate(spec; mode::Symbol, omega=1.0)
    Fref = rest_equilibrium_tree(spec)
    Famr = run_amr_stream_with_prestream(spec, Fref; mode=mode, omega=omega)
    return Famr, Fref
end

function run_case(spec; omega=1.2, perturb=false)
    leaf_nx = spec.Nx << spec.max_level
    leaf_ny = spec.Ny << spec.max_level
    leaf_volume = 1.0 / Float64((1 << spec.max_level)^2)
    ticks = 1 << spec.max_level
    leaf0 = zeros(Float64, leaf_nx, leaf_ny, 9)
    leaf_scratch = similar(leaf0)
    fill_leaf_equilibrium!(leaf0, leaf_volume)
    if perturb
        add_odd_diagonal_packet!(
            leaf0, spec; level=2, i=15, j=4, qplus=8,
            amp=1e-8 * leaf_volume)
    end

    Fin = allocate_conservative_tree_F_2d(spec; T=Float64)
    restrict_leaf_to_tree!(Fin, leaf0, spec)
    Famr = run_amr_bgk_stream(spec, Fin; omega=omega)

    leaf_final = run_leaf_bgk_stream!(
        leaf0, leaf_scratch, leaf_volume, omega, ticks)
    Fref = allocate_conservative_tree_F_2d(spec; T=Float64)
    restrict_leaf_to_tree!(Fref, leaf_final, spec)
    return Famr, Fref
end

function write_diff_case!(io, label::String, spec, A, B; threshold=1e-16)
    max_absdiff = 0.0
    worst = nothing
    rows = 0
    @inbounds for cell_id in spec.active_cells, q in 1:9
        diff = Float64(A[cell_id, q] - B[cell_id, q])
        ad = abs(diff)
        if ad > max_absdiff
            max_absdiff = ad
            worst = (cell_id=cell_id, q=q, diff=diff)
        end
        ad > threshold || continue
        cell = spec.cells[cell_id]
        @printf(io, "%s,%d,%d,%d,%d,%d,%.16e,%.16e,%+.16e,%.16e\n",
                label, cell_id, cell.level, cell.i, cell.j, q,
                Float64(B[cell_id, q]), Float64(A[cell_id, q]), diff, ad)
        rows += 1
    end
    return (; max_absdiff, worst, rows)
end

function write_residual_contributions!(detail_io, summary_io, label::String,
                                       spec, A, B; threshold=1e-16)
    cone = wall_phase_cone_cells(spec)
    grouped = Dict{Tuple{Int,Bool},Tuple{Int,Float64,Float64}}()
    @inbounds for cell_id in spec.active_cells, q in 1:9
        diff = Float64(A[cell_id, q] - B[cell_id, q])
        ad = abs(diff)
        ad > threshold || continue
        cell = spec.cells[cell_id]
        in_cone = cell_id in cone
        @printf(detail_io,
                "%s,%d,%d,%d,%d,%d,%.16e,%.16e,%+.16e,%.16e,%s\n",
                label, cell_id, cell.level, cell.i, cell.j, q,
                Float64(B[cell_id, q]), Float64(A[cell_id, q]), diff, ad,
                string(in_cone))
        key = (cell.level, in_cone)
        count, sum_abs, max_abs = get(grouped, key, (0, 0.0, 0.0))
        grouped[key] = (count + 1, sum_abs + ad, max(max_abs, ad))
    end
    for key in sort!(collect(keys(grouped)); by=x -> (x[1], x[2]))
        level, in_cone = key
        count, sum_abs, max_abs = grouped[key]
        @printf(summary_io, "%s,%d,%s,%d,%.16e,%.16e\n",
                label, level, string(in_cone), count, sum_abs, max_abs)
    end
    return grouped
end

function wall_phase_leaf_track_arrays(spec)
    src_ids = Int[]
    src_qs = Int[]
    pos_is = Int[]
    pos_js = Int[]
    qcur = Int[]
    @inbounds for src_id in spec.active_cells, q in 1:9
        source_q_can_touch_wall(spec, src_id, q) || continue
        i0, i1, j0, j1 = cell_leaf_bounds(spec, src_id)
        for j in j0:j1, i in i0:i1
            push!(src_ids, src_id)
            push!(src_qs, q)
            push!(pos_is, i)
            push!(pos_js, j)
            push!(qcur, q)
        end
    end
    return src_ids, src_qs, pos_is, pos_js, qcur
end

function owner_advance_tick(spec, leaf_tick::Int, owner_id::Int)
    owner_id == 0 && return 0
    level = spec.cells[owner_id].level
    level_ticks = 1 << (spec.max_level - level)
    return cld(leaf_tick, level_ticks) * level_ticks
end

function cell_fields(spec, cell_id::Int)
    cell_id == 0 && return (level=-1, i=0, j=0)
    cell = spec.cells[cell_id]
    return (level=cell.level, i=cell.i, j=cell.j)
end

function cell_leaf_count(spec, cell_id::Int)
    cell_id == 0 && return 0
    cell = spec.cells[cell_id]
    scale = 1 << (spec.max_level - cell.level)
    return scale * scale
end

function write_oracle_tick_consumption!(
        detail_io,
        summary_io,
        prediction_io,
        prediction_dst_io,
        label::String,
        spec;
        omega=1.0,
        perturb=false)
    leaf_nx = spec.Nx << spec.max_level
    leaf_ny = spec.Ny << spec.max_level
    leaf_volume = 1.0 / Float64((1 << spec.max_level)^2)
    ticks = 1 << spec.max_level
    leaf = zeros(Float64, leaf_nx, leaf_ny, 9)
    scratch = similar(leaf)
    fill_leaf_equilibrium!(leaf, leaf_volume)
    if perturb
        add_odd_diagonal_packet!(
            leaf, spec; level=2, i=15, j=4, qplus=8,
            amp=1e-8 * leaf_volume)
    end

    src_ids, src_qs, pos_is, pos_js, qcur =
        wall_phase_leaf_track_arrays(spec)
    detail_grouped = Dict{
        Tuple{Int,Int,Int,Int,Int,Int,Int,Int,Bool},
        Tuple{Int,Float64,Float64}
    }()
    summary_grouped = Dict{
        Tuple{Int,Int,Int},
        Tuple{Int,Int,Float64,Float64}
    }()
    final_dst_weights = Dict{Tuple{Int,Int,Int,Int},Float64}()

    for leaf_tick in 1:ticks
        collide_BGK_integrated_D2Q9!(leaf, leaf_volume, omega)
        empty!(detail_grouped)
        @inbounds for idx in eachindex(src_ids)
            src_id = src_ids[idx]
            src_q = src_qs[idx]
            i = pos_is[idx]
            j = pos_js[idx]
            q = qcur[idx]
            owner_id = Kraken._active_leaf_covering_sample_2d(
                spec, spec.max_level, i, j)
            post_value = Float64(leaf[i, j, q])
            next_i, next_j, next_q, wall_hit =
                phase_advance_leaf(spec, i, j, q)
            dst_id = Kraken._active_leaf_covering_sample_2d(
                spec, spec.max_level, next_i, next_j)
            advance_tick = owner_advance_tick(spec, leaf_tick, owner_id)
            detail_key = (
                src_id, src_q, leaf_tick, advance_tick,
                owner_id, q, dst_id, next_q, wall_hit)
            path_count, post_sum, route_sum = get(
                detail_grouped, detail_key, (0, 0.0, 0.0))
            detail_grouped[detail_key] =
                (path_count + 1, post_sum + post_value,
                 route_sum + post_value)

            summary_key = (src_id, src_q, leaf_tick)
            total_paths, wall_hits, summary_post, summary_route = get(
                summary_grouped, summary_key, (0, 0, 0.0, 0.0))
            summary_grouped[summary_key] =
                (total_paths + 1, wall_hits + (wall_hit ? 1 : 0),
                 summary_post + post_value, summary_route + post_value)

            pos_is[idx] = next_i
            pos_js[idx] = next_j
            qcur[idx] = next_q
        end

        for key in sort!(collect(keys(detail_grouped)))
            src_id, src_q, tick, advance_tick, owner_id, q,
                dst_id, dst_q, wall_hit = key
            path_count, post_sum, route_sum = detail_grouped[key]
            src_cell = spec.cells[src_id]
            source_weight = Float64(path_count) /
                            Float64(cell_leaf_count(spec, src_id))
            if tick == ticks
                dst_key = (src_id, src_q, dst_id, dst_q)
                final_dst_weights[dst_key] =
                    get(final_dst_weights, dst_key, 0.0) + source_weight
            end
            owner = cell_fields(spec, owner_id)
            dst = cell_fields(spec, dst_id)
            @printf(detail_io,
                    "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%.16e,%.16e,%.16e\n",
                    label, src_id, src_cell.level, src_cell.i, src_cell.j,
                    src_q, tick, advance_tick, owner_id, owner.level,
                    owner.i, owner.j, q, dst_id, dst.level, dst.i, dst.j,
                    dst_q, string(wall_hit), path_count, source_weight,
                    post_sum, route_sum)
        end

        stream_periodic_x_wall_y_F_2d!(scratch, leaf)
        leaf, scratch = scratch, leaf
    end

    source_ranges = Dict{Tuple{Int,Int},Tuple{Float64,Float64}}()
    for key in keys(summary_grouped)
        src_id, src_q, _ = key
        _, _, post_sum, _ = summary_grouped[key]
        range_key = (src_id, src_q)
        lo, hi = get(source_ranges, range_key, (Inf, -Inf))
        source_ranges[range_key] = (min(lo, post_sum), max(hi, post_sum))
    end

    for key in sort!(collect(keys(summary_grouped)))
        src_id, src_q, tick = key
        path_count, wall_hits, post_sum, route_sum = summary_grouped[key]
        src_cell = spec.cells[src_id]
        source_weight = Float64(path_count) /
                        Float64(cell_leaf_count(spec, src_id))
        post_min, post_max = source_ranges[(src_id, src_q)]
        post_range = post_max - post_min
        upper_bound = post_range * source_weight
        @printf(summary_io,
                "%s,%d,%d,%d,%d,%d,%d,%d,%d,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                label, src_id, src_cell.level, src_cell.i, src_cell.j,
                src_q, tick, path_count, wall_hits, source_weight, post_sum,
                route_sum, post_range, upper_bound)
    end

    dst_predictions = Dict{Tuple{Int,Int},Tuple{Int,Float64}}()
    for key in sort!(collect(keys(final_dst_weights)))
        src_id, src_q, dst_id, dst_q = key
        final_weight = final_dst_weights[key]
        post_min, post_max = source_ranges[(src_id, src_q)]
        post_range = post_max - post_min
        upper_bound = post_range * final_weight
        src_cell = spec.cells[src_id]
        dst = cell_fields(spec, dst_id)
        @printf(prediction_io,
                "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                label, src_id, src_cell.level, src_cell.i, src_cell.j,
                src_q, dst_id, dst.level, dst.i, dst.j, dst_q,
                final_weight, post_min, post_max, post_range, upper_bound)
        contributors, total = get(dst_predictions, (dst_id, dst_q), (0, 0.0))
        dst_predictions[(dst_id, dst_q)] = (contributors + 1, total + upper_bound)
    end

    for key in sort!(collect(keys(dst_predictions)))
        dst_id, dst_q = key
        contributors, upper_bound = dst_predictions[key]
        dst = cell_fields(spec, dst_id)
        @printf(prediction_dst_io,
                "%s,%d,%d,%d,%d,%d,%d,%.16e\n",
                label, dst_id, dst.level, dst.i, dst.j, dst_q,
                contributors, upper_bound)
    end
    return nothing
end

function main()
    mkpath(OUTDIR)
    specs = [
        ("mini_wall_touch_max_level_2", wall_touch_max_level_2_spec()),
        ("wall_touch_xband_max2", wall_touch_xband_max2_spec()),
        ("bulk_touch_max_level_2", bulk_touch_max_level_2_spec()),
    ]
    mini_spec = specs[1][2]
    omega = 1.2

    Famr_eq, Fref_eq = run_case(mini_spec; omega=omega, perturb=false)
    Famr_odd, Fref_odd = run_case(mini_spec; omega=omega, perturb=true)
    delta_amr = Famr_odd .- Famr_eq
    delta_ref = Fref_odd .- Fref_eq

    detail_path = joinpath(OUTDIR, "field_diffs.csv")
    residual_path = joinpath(OUTDIR, "residual_contributions.csv")
    residual_summary_path = joinpath(OUTDIR, "residual_level_summary.csv")
    oracle_tick_path = joinpath(OUTDIR, "oracle_tick_consumption.csv")
    oracle_tick_summary_path = joinpath(
        OUTDIR, "oracle_tick_source_summary.csv")
    oracle_prediction_path = joinpath(
        OUTDIR, "oracle_tick_gate3_prediction.csv")
    oracle_prediction_dst_path = joinpath(
        OUTDIR, "oracle_tick_gate3_prediction_by_dst.csv")
    metrics = Pair{String,NamedTuple}[]
    open(detail_path, "w") do io
        println(io, "case,cell_id,level,i,j,q,oracle,amr,diff,absdiff")
        for (spec_label, spec) in specs
            Famr_noop, Fref_noop = run_rest_gate(spec; mode=:noop)
            push!(metrics, "$(spec_label)_noop_rest" =>
                  write_diff_case!(io, "$(spec_label)_noop_rest",
                                   spec, Famr_noop, Fref_noop))

            Famr_bgk1, Fref_bgk1 = run_rest_gate(
                spec; mode=:bgk, omega=1.0)
            push!(metrics, "$(spec_label)_bgk_omega1_rest" =>
                  write_diff_case!(io, "$(spec_label)_bgk_omega1_rest",
                                   spec, Famr_bgk1, Fref_bgk1))
        end

        push!(metrics, "mini_equilibrium_bgk_stream" =>
              write_diff_case!(io, "mini_equilibrium_bgk_stream",
                               mini_spec, Famr_eq, Fref_eq))
        push!(metrics, "mini_odd_diagonal_bgk_stream" =>
              write_diff_case!(io, "mini_odd_diagonal_bgk_stream",
                               mini_spec, Famr_odd, Fref_odd))
        push!(metrics, "mini_odd_diagonal_delta" =>
              write_diff_case!(io, "mini_odd_diagonal_delta",
                               mini_spec, delta_amr, delta_ref;
                               threshold=1e-20))
    end

    open(residual_path, "w") do detail_io
        open(residual_summary_path, "w") do summary_io
            println(detail_io,
                    "case,cell_id,level,i,j,q,oracle,amr,diff,absdiff,in_wall_cone")
            println(summary_io,
                    "case,level,in_wall_cone,count,sum_absdiff,max_absdiff")
            for (spec_label, spec) in specs
                Famr_noop, Fref_noop = run_rest_gate(spec; mode=:noop)
                write_residual_contributions!(
                    detail_io, summary_io, "$(spec_label)_noop_rest",
                    spec, Famr_noop, Fref_noop)
                Famr_bgk1, Fref_bgk1 = run_rest_gate(
                    spec; mode=:bgk, omega=1.0)
                write_residual_contributions!(
                    detail_io, summary_io, "$(spec_label)_bgk_omega1_rest",
                    spec, Famr_bgk1, Fref_bgk1)
            end
        end
    end

    open(oracle_tick_path, "w") do detail_io
        open(oracle_tick_summary_path, "w") do summary_io
            open(oracle_prediction_path, "w") do prediction_io
                open(oracle_prediction_dst_path, "w") do prediction_dst_io
            println(detail_io,
                    "case,src_id,src_level,src_i,src_j,src_q,leaf_tick,owner_advance_tick,owner_id,owner_level,owner_i,owner_j,current_q,dst_id,dst_level,dst_i,dst_j,dst_q,wall_hit,path_count,source_weight,post_collision_F_value,oracle_route_consumption")
            println(summary_io,
                    "case,src_id,src_level,src_i,src_j,src_q,leaf_tick,path_count,wall_hits,source_weight,post_collision_F_value,oracle_route_consumption,post_F_range_per_source,gate3_residual_upper_bound")
            println(prediction_io,
                    "case,src_id,src_level,src_i,src_j,src_q,dst_id,dst_level,dst_i,dst_j,dst_q,final_source_weight,post_F_min,post_F_max,post_F_range_per_source,gate3_residual_upper_bound")
            println(prediction_dst_io,
                    "case,dst_id,dst_level,dst_i,dst_j,dst_q,contributors,gate3_residual_upper_bound")
            write_oracle_tick_consumption!(
                detail_io, summary_io, prediction_io, prediction_dst_io,
                "mini_bgk_omega1_rest",
                mini_spec; omega=1.0, perturb=false)
            write_oracle_tick_consumption!(
                detail_io, summary_io, prediction_io, prediction_dst_io,
                "mini_equilibrium_bgk_omega1p2",
                mini_spec; omega=omega, perturb=false)
            write_oracle_tick_consumption!(
                detail_io, summary_io, prediction_io, prediction_dst_io,
                "mini_odd_diagonal_bgk_omega1p2",
                mini_spec; omega=omega, perturb=true)
                end
            end
        end
    end

    summary_path = joinpath(OUTDIR, "summary.csv")
    open(summary_path, "w") do io
        println(io, "case,max_absdiff,bad_rows,worst_cell,worst_q,worst_diff")
        for (label, m) in metrics
            w = m.worst === nothing ? (cell_id=0, q=0, diff=0.0) : m.worst
            @printf(io, "%s,%.16e,%d,%d,%d,%+.16e\n",
                    label, m.max_absdiff, m.rows, w.cell_id, w.q, w.diff)
        end
    end

    println("wrote ", summary_path)
    println("wrote ", detail_path)
    println("wrote ", residual_path)
    println("wrote ", residual_summary_path)
    println("wrote ", oracle_tick_path)
    println("wrote ", oracle_tick_summary_path)
    println("wrote ", oracle_prediction_path)
    println("wrote ", oracle_prediction_dst_path)
    for (label, m) in metrics
        println(label, " max_absdiff=", @sprintf("%.4e", m.max_absdiff),
                " bad_rows=", m.rows)
    end
end

main()
