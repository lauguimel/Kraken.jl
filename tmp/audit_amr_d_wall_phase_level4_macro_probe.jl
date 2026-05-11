using Kraken
using Printf

const OUTDIR = joinpath(
    dirname(@__DIR__), "benchmarks", "results", "quicklook",
    "amr_d_wall_phase_level4_macro_probe_20260511")

const CASE_PATH = joinpath(
    dirname(@__DIR__), "benchmarks", "krk", "amr_d_convergence_2d",
    "poiseuille_xband_fullheight_level_native_nested4_debug.krk")

function cell_leaf_bounds(spec, cell_id::Int)
    cell = spec.cells[cell_id]
    scale = 1 << (spec.max_level - cell.level)
    return (cell.i - 1) * scale + 1, cell.i * scale,
           (cell.j - 1) * scale + 1, cell.j * scale
end

function rho_metrics(F, spec; rho0=1.0)
    leaf_nx = spec.Nx << spec.max_level
    leaf_ny = spec.Ny << spec.max_level
    rho = fill(NaN, leaf_nx, leaf_ny)
    max_abs_dev = 0.0
    worst_cell = 0
    worst_rho = 0.0
    @inbounds for cell_id in spec.active_cells
        volume = Float64(spec.cells[cell_id].metrics.volume)
        rho_cell = sum(Float64(F[cell_id, q]) for q in 1:9) / volume
        dev = abs(rho_cell - rho0)
        if dev > max_abs_dev
            max_abs_dev = dev
            worst_cell = cell_id
            worst_rho = rho_cell
        end
        i0, i1, j0, j1 = cell_leaf_bounds(spec, cell_id)
        for j in j0:j1, i in i0:i1
            rho[i, j] = rho_cell
        end
    end

    max_row_x_range = 0.0
    worst_leaf_j = 0
    @inbounds for j in 1:leaf_ny
        lo = Inf
        hi = -Inf
        for i in 1:leaf_nx
            v = rho[i, j]
            isfinite(v) || continue
            lo = min(lo, v)
            hi = max(hi, v)
        end
        if isfinite(lo) && hi - lo > max_row_x_range
            max_row_x_range = hi - lo
            worst_leaf_j = j
        end
    end

    cell = spec.cells[worst_cell]
    return (;
        max_abs_dev,
        max_row_x_range,
        worst_leaf_j,
        worst_cell,
        worst_level=cell.level,
        worst_i=cell.i,
        worst_j=cell.j,
        worst_rho)
end

function run_case(spec; steps::Int, correction::Bool,
                  predictor_weight=0.0, Fx=1e-7, omega=1.0, rho0=1.0)
    table = create_conservative_tree_route_table_2d(
        spec; periodic_x=true, sampling=:level_native)
    F = allocate_conservative_tree_F_2d(spec; T=Float64)
    Ftmp = similar(F)
    initialize_conservative_tree_equilibrium_F_2d!(F, spec; rho=rho0)

    schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(
        spec.max_level)
    route_bank = Kraken.create_conservative_tree_subcycle_spatial_ledger_bank_2d(
        spec; schedule=schedule, T=Float64)
    Kraken.prepare_conservative_tree_subcycle_route_packet_cache_2d!(
        route_bank, table; periodic_x=true)
    state_bank = Kraken.create_conservative_tree_subcycle_buffer_bank_2d(
        spec; schedule=schedule, T=Float64)
    active_ids_by_level = state_bank.active_ids_by_level
    Fsource = similar(F)
    Fscratch = similar(F)

    mass_initial = Kraken._active_mass_conservative_tree_F_2d(F, spec)
    guard = conservative_tree_mass_roundoff_rtol_2d(
        Float64, steps, spec.max_level;
        active_cell_count=length(spec.active_cells))
    max_raw_mass_rel_drift = 0.0
    collide_level! = (Flevel, local_spec, level, event) ->
        Kraken._collide_Guo_conservative_tree_active_ids_F_2d!(
            Flevel, local_spec, active_ids_by_level[level + 1],
            Kraken.conservative_tree_leaf_equivalent_omega_2d(
                omega, local_spec, level),
            Kraken.conservative_tree_leaf_equivalent_force_2d(
                Fx, local_spec, level),
            0.0)

    for _ in 1:steps
        Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Ftmp, F, spec, table; boundary=:periodic_x_wall_y,
            alpha_c2f=1, alpha_f2c=1,
            coarse_to_fine_prolongation=:flat,
            coarse_to_fine_state=:owned,
            coarse_to_fine_predictor_weight=predictor_weight,
            interface_time_scaling=:level_native,
            pre_stream_level! = collide_level!,
            schedule=schedule, route_bank=route_bank,
            state_bank=state_bank, Fsource=Fsource, Fscratch=Fscratch,
            wall_phase_transport_correction=correction)
        raw_rel = Kraken._enforce_active_mass_conservation_2d!(
            Ftmp, spec, mass_initial; rtol=Inf)
        max_raw_mass_rel_drift = max(max_raw_mass_rel_drift, raw_rel)
        F, Ftmp = Ftmp, F
    end

    metrics = rho_metrics(F, spec; rho0=rho0)
    mass_final = Kraken._active_mass_conservative_tree_F_2d(F, spec)
    return merge(metrics, (;
        mass_rel_drift=(mass_final - mass_initial) / mass_initial,
        max_raw_mass_rel_drift))
end

function main()
    mkpath(OUTDIR)
    spec = create_conservative_tree_spec_from_krk_2d(load_kraken(CASE_PATH))
    steps = parse.(Int, split(get(ENV, "KRAKEN_WALL_PHASE_STEPS", "200"), ","))
    out = joinpath(OUTDIR, "summary.csv")
    open(out, "w") do io
        println(io, "case,steps,predictor_weight,correction,max_abs_rho_dev,max_row_x_range_rho,worst_leaf_j,worst_cell,worst_level,worst_i,worst_j,worst_rho,mass_rel_drift,max_raw_mass_rel_drift")
        for nsteps in steps, correction in (false, true)
            m = run_case(spec; steps=nsteps, correction=correction,
                         predictor_weight=0.0)
            @printf(io,
                    "fullheight_nested4,%d,%.16e,%s,%.16e,%.16e,%d,%d,%d,%d,%d,%.16e,%.16e,%.16e\n",
                    nsteps, 0.0, string(correction), m.max_abs_dev,
                    m.max_row_x_range, m.worst_leaf_j, m.worst_cell,
                    m.worst_level, m.worst_i, m.worst_j, m.worst_rho,
                    m.mass_rel_drift, m.max_raw_mass_rel_drift)
        end
    end
    println("wrote ", out)
    print(read(out, String))
end

main()
