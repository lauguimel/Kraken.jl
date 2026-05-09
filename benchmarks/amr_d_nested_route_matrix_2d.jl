#!/usr/bin/env julia

using Dates
using Kraken
using Printf

include(joinpath(@__DIR__, "amr_d_macroflow_temporal_convergence_2d.jl"))

const ROUTE_MATRIX_CASE_DIR = joinpath(@__DIR__, "krk", "amr_d_convergence_2d")
const ROUTE_MATRIX_DEFAULT_CASES = [
    "poiseuille_xband_nested4_debug.krk",
    "poiseuille_yband_nested4_debug.krk",
    "poiseuille_wall_ybands_nested4_debug.krk",
    "couette_yband_nested4_debug.krk",
]
const ROUTE_MATRIX_DEFAULT_OUTDIR = joinpath(
    @__DIR__, "results", "quicklook",
    "amr_d_nested_route_matrix_" * Dates.format(now(), "yyyymmdd_HHMMSS"))

function _route_matrix_env_int(name::AbstractString, default::Integer)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return Int(default)
    return parse(Int, raw)
end

function _route_matrix_env_type()
    raw = lowercase(strip(get(ENV, "KRK_AMR_D_ROUTE_MATRIX_T", "float64")))
    raw in ("float64", "f64", "") && return Float64
    raw in ("float32", "f32") && return Float32
    throw(ArgumentError("KRK_AMR_D_ROUTE_MATRIX_T must be float32 or float64"))
end

function _route_matrix_paths()
    raw = strip(get(ENV, "KRK_AMR_D_ROUTE_MATRIX_CASES", ""))
    names = isempty(raw) ? ROUTE_MATRIX_DEFAULT_CASES : split(raw, ",")
    paths = String[]
    for name in names
        s = strip(String(name))
        isempty(s) && continue
        path = isabspath(s) ? s : joinpath(ROUTE_MATRIX_CASE_DIR, s)
        endswith(path, ".krk") || (path *= ".krk")
        push!(paths, path)
    end
    return paths
end

function _route_matrix_codes()
    raw = strip(get(ENV, "KRK_AMR_D_ROUTE_MATRIX_ROUTES", "0,1"))
    codes = Int[]
    for item in split(raw, ",")
        s = strip(String(item))
        isempty(s) && continue
        code = parse(Int, s)
        code in (0, 1) ||
            throw(ArgumentError("route matrix supports route codes 0 and 1"))
        push!(codes, code)
    end
    isempty(codes) && throw(ArgumentError("no route codes requested"))
    return codes
end

function _route_matrix_name(path::AbstractString, route_code::Integer)
    base = splitext(basename(path))[1]
    route_name = Int(route_code) == 0 ? "leaf_equivalent" : "level_native"
    return string(base, "__route_", route_name)
end

function _route_matrix_override_route!(setup, route_code::Integer)
    vars = getproperty(setup, :user_vars)
    vars[:route_sampling] = Float64(route_code)
    return setup
end

function _route_matrix_profile_errors(profile, reference)
    length(profile) == length(reference) || return (NaN, NaN)
    sq = 0.0
    linf = 0.0
    count = 0
    for i in eachindex(profile, reference)
        a = Float64(profile[i])
        b = Float64(reference[i])
        isfinite(a) && isfinite(b) || continue
        d = a - b
        sq += d * d
        linf = max(linf, abs(d))
        count += 1
    end
    count == 0 && return (NaN, NaN)
    return sqrt(sq / count), linf
end

function _route_matrix_row_levels(level)
    ny = size(level, 2)
    mins = Vector{Int}(undef, ny)
    maxs = Vector{Int}(undef, ny)
    @inbounds for j in 1:ny
        lo = typemax(Int)
        hi = typemin(Int)
        for i in axes(level, 1)
            v = Int(level[i, j])
            v < 0 && continue
            lo = min(lo, v)
            hi = max(hi, v)
        end
        mins[j] = lo == typemax(Int) ? -1 : lo
        maxs[j] = hi == typemin(Int) ? -1 : hi
    end
    return mins, maxs
end

function _route_matrix_interface_jump(profile, level)
    row_min, row_max = _route_matrix_row_levels(level)
    jump = 0.0
    count = 0
    for j in 2:length(profile)
        changed = row_min[j] != row_min[j - 1] ||
                  row_max[j] != row_max[j - 1]
        changed || continue
        a = Float64(profile[j])
        b = Float64(profile[j - 1])
        isfinite(a) && isfinite(b) || continue
        jump = max(jump, abs(a - b))
        count += 1
    end
    return count == 0 ? NaN : jump
end

function _route_matrix_write_profile_csv(path, case, route_code, amr, reference)
    profile, analytic = _ql_profile_vectors(amr.result, amr.state)
    ref_profile, _ = _ql_profile_vectors(reference.result, reference.state)
    row_min, row_max = _route_matrix_row_levels(amr.state.level)
    open(path, "w") do io
        println(io, "case,route_code,j,y,row_level_min,row_level_max,ux_amr,ux_cartesian,ux_analytic")
        ny = length(profile)
        for j in 1:ny
            y = ny == 1 ? 0.0 : (j - 1) / (ny - 1)
            @printf(io, "%s,%d,%d,%.16e,%d,%d,%.16e,%.16e,%.16e\n",
                    case.name, route_code, j, y, row_min[j], row_max[j],
                    Float64(profile[j]), Float64(ref_profile[j]),
                    Float64(analytic[j]))
        end
    end
    return path
end

function _route_matrix_write_summary_csv(path, rows)
    open(path, "w") do io
        println(io, "case,flow,route_code,route_name,steps,T,active_cells,leaf_equivalent_cells,mass_rel_drift,max_raw_mass_rel_drift,rho_min,rho_max,ux_min,ux_max,l2_profile_vs_analytic,linf_profile_vs_analytic,l2_profile_vs_cartesian,linf_profile_vs_cartesian,l2_ux_field_vs_cartesian,linf_ux_field_vs_cartesian,l2_rho_field_vs_cartesian,linf_rho_field_vs_cartesian,max_interface_profile_jump,case_outdir,profile_csv")
        for r in rows
            @printf(io, "%s,%s,%d,%s,%d,%s,%d,%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%s,%s\n",
                    r.case, String(r.flow), r.route_code, r.route_name,
                    r.steps, r.T_name, r.active_cells, r.leaf_equivalent_cells,
                    r.mass_rel_drift, r.max_raw_mass_rel_drift,
                    r.rho_min, r.rho_max, r.ux_min, r.ux_max,
                    r.l2_profile_vs_analytic, r.linf_profile_vs_analytic,
                    r.l2_profile_vs_cartesian, r.linf_profile_vs_cartesian,
                    r.l2_ux_field_vs_cartesian,
                    r.linf_ux_field_vs_cartesian,
                    r.l2_rho_field_vs_cartesian,
                    r.linf_rho_field_vs_cartesian,
                    r.max_interface_profile_jump, r.case_outdir,
                    r.profile_csv)
        end
    end
    return path
end

function run_amr_d_nested_route_matrix_2d(paths=_route_matrix_paths();
        outdir::AbstractString=get(ENV, "KRK_AMR_D_ROUTE_MATRIX_OUTDIR",
                                   ROUTE_MATRIX_DEFAULT_OUTDIR),
        steps::Int=_route_matrix_env_int("KRK_AMR_D_ROUTE_MATRIX_STEPS", 64),
        route_codes=_route_matrix_codes(),
        T::Type{<:AbstractFloat}=_route_matrix_env_type())
    mkpath(outdir)
    rows = NamedTuple[]
    for path in paths, route_code in route_codes
        setup = load_kraken(String(path))
        _route_matrix_override_route!(setup, route_code)
        case = conservative_tree_amr_d_case_from_krk_2d(setup)
        case.runtime_status == :subcycled_nested_channel ||
            throw(ArgumentError("route matrix expects nested channel KRK, got $(case.runtime_status) for $(case.name)"))
        println("running ", case.name, " route=", route_code,
                " steps=", steps, " T=", T)
        flush(stdout)

        amr = _temp_run_case(setup, case; steps=steps, method=:amr_d, T=T)
        reference = merge(_temp_run_case(
            setup, case; steps=steps, method=:cartesian_classic, T=T),
                          (; method=:cartesian_classic,))
        amr_record = merge(amr, (; method=:amr_d,))
        vals = _ql_method_values(amr_record, reference)
        profile, analytic = _ql_profile_vectors(amr.result, amr.state)
        ref_profile, _ = _ql_profile_vectors(
            reference.result, reference.state)
        l2_cart, linf_cart = _route_matrix_profile_errors(
            profile, ref_profile)
        l2_analytic, linf_analytic = _route_matrix_profile_errors(
            profile, analytic)
        route_name = route_code == 0 ? "leaf_equivalent" : "level_native"
        case_outdir = joinpath(outdir, _route_matrix_name(path, route_code))
        mkpath(case_outdir)
        profile_csv = _route_matrix_write_profile_csv(
            joinpath(case_outdir, "profiles.csv"),
            case, route_code, amr, reference)
        push!(rows, (;
            case=case.name,
            flow=case.flow,
            route_code=route_code,
            route_name=route_name,
            steps=steps,
            T_name=String(nameof(T)),
            active_cells=getproperty(amr.result, :active_cell_count),
            leaf_equivalent_cells=getproperty(
                amr.result, :leaf_equivalent_cell_count),
            mass_rel_drift=vals.mass_rel_drift,
            max_raw_mass_rel_drift=vals.max_raw_mass_rel_drift,
            rho_min=vals.rho_min,
            rho_max=vals.rho_max,
            ux_min=vals.ux_min,
            ux_max=vals.ux_max,
            l2_profile_vs_analytic=l2_analytic,
            linf_profile_vs_analytic=linf_analytic,
            l2_profile_vs_cartesian=l2_cart,
            linf_profile_vs_cartesian=linf_cart,
            l2_ux_field_vs_cartesian=vals.l2_ux_field_vs_reference,
            linf_ux_field_vs_cartesian=vals.linf_ux_field_vs_reference,
            l2_rho_field_vs_cartesian=vals.l2_rho_field_vs_reference,
            linf_rho_field_vs_cartesian=vals.linf_rho_field_vs_reference,
            max_interface_profile_jump=_route_matrix_interface_jump(
                profile, amr.state.level),
            case_outdir=case_outdir,
            profile_csv=profile_csv))
    end
    summary_csv = _route_matrix_write_summary_csv(
        joinpath(outdir, "summary.csv"), rows)
    println("wrote ", summary_csv)
    return rows
end

function main()
    rows = run_amr_d_nested_route_matrix_2d()
    for row in rows
        println(row.case, " route=", row.route_name,
                " linf_cart=", row.linf_profile_vs_cartesian,
                " jump=", row.max_interface_profile_jump)
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
