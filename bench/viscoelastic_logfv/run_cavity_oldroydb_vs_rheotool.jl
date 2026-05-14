#!/usr/bin/env julia

# Closed axis-aligned lid-driven cavity comparison for the log-FV polymer
# backend. The cavity has no inlet/outlet and no curved walls; this is the
# tightest axis-aligned discriminator below the contraction and BFS cases.
#
# Reference: rheoTool Cavity/Oldroyd-BLog tutorial (user guide section 5.1.4),
# project-local copy at bench/rheotool/cavity_oldroydb_log_re001_de1_b05.
#
# Comparison observables at physical time t = 8 (sample_times target):
#   1. `u(x=0.5, y)` along the vertical centerline.
#   2. `psi_xy(x, y=0.75)` along the horizontal probe line (theta_xy in
#      rheoTool wording -- both are log(C)_xy).
#   3. Volume-averaged kinetic and elastic energy time series E_k(t).
#
# Calibration note: rheoTool runs at Re = 0.01 (creeping). Matching this in
# a uniform-mesh LBM would force tau_s past stability; instead we match De=1
# and beta=0.5 exactly and accept Re_LU = O(1). De governs polymer response;
# the cavity vortex structure at small Re is comparatively weakly Re-dependent.

using Dates
using DelimitedFiles
using KernelAbstractions
using Printf
using Serialization

using Kraken

const DEFAULT_RHEOTOOL_CASE = joinpath("bench", "rheotool", "cavity_oldroydb_log_re001_de1_b05")
const DEFAULT_OUTPUT_DIR = joinpath("tmp", "cavity_oldroydb_vs_rheotool")

const CUDA_MOD = try
    @eval using CUDA
    getfield(Main, :CUDA)
catch
    nothing
end

const METAL_MOD = if Sys.isapple()
    try
        @eval using Metal
        getfield(Main, :Metal)
    catch
        nothing
    end
else
    nothing
end

function parse_int_list(raw::AbstractString)
    values = Int[]
    for part in split(replace(raw, ';' => ','), ',')
        text = strip(part)
        isempty(text) && continue
        push!(values, parse(Int, text))
    end
    return values
end

function pick_backend()
    requested = lowercase(get(ENV, "KRAKEN_BACKEND", "auto"))
    if requested in ("metal", "mtl") && METAL_MOD !== nothing
        return :metal, METAL_MOD.MetalBackend(), Float32
    elseif requested in ("cuda", "gpu") && CUDA_MOD !== nothing
        return :cuda, CUDA_MOD.CUDABackend(), Float64
    elseif requested == "cpu"
        return :cpu, KernelAbstractions.CPU(), Float64
    end
    # Prefer Metal on macOS (CUDA is loadable but non-functional without GPU);
    # otherwise prefer CUDA when actually available.
    if METAL_MOD !== nothing && Sys.isapple()
        return :metal, METAL_MOD.MetalBackend(), Float32
    elseif CUDA_MOD !== nothing && CUDA_MOD.functional()
        return :cuda, CUDA_MOD.CUDABackend(), Float64
    end
    return :cpu, KernelAbstractions.CPU(), Float64
end

# ---------------------------------------------------------------------
# rheoTool .xy loaders
# ---------------------------------------------------------------------

"""
Parse `lineVert_x0.5_U.xy` (4 columns: y, Ux, Uy, Uz).
"""
function read_rheotool_vertical_U(path::AbstractString)
    raw = readdlm(path)
    return (y=Vector{Float64}(raw[:, 1]),
            ux=Vector{Float64}(raw[:, 2]),
            uy=Vector{Float64}(raw[:, 3]))
end

"""
Parse `lineHorz_y0.75_tau_theta.xy` (13 columns: x, tau×6, theta×6).
Returns theta_xy alongside tau_xy because the user-guide figure 5.4b plots
theta_xy (== log(C)_xy == psi_xy in Kraken wording).
"""
function read_rheotool_horizontal_tautheta(path::AbstractString)
    raw = readdlm(path)
    x = Vector{Float64}(raw[:, 1])
    tau_xx = Vector{Float64}(raw[:, 2])
    tau_xy = Vector{Float64}(raw[:, 3])
    tau_yy = Vector{Float64}(raw[:, 5])
    theta_xx = Vector{Float64}(raw[:, 8])
    theta_xy = Vector{Float64}(raw[:, 9])
    theta_yy = Vector{Float64}(raw[:, 11])
    return (x=x,
            tau_xx=tau_xx, tau_xy=tau_xy, tau_yy=tau_yy,
            theta_xx=theta_xx, theta_xy=theta_xy, theta_yy=theta_yy)
end

"""
Read `kinEner.txt` (3 columns: t, kinetic, elastic).
"""
function read_rheotool_kinener(path::AbstractString)
    raw = readdlm(path)
    return (t=Vector{Float64}(raw[:, 1]),
            kinetic=Vector{Float64}(raw[:, 2]),
            elastic=Vector{Float64}(raw[:, 3]))
end

"""
Pick the latest non-zero rheoTool sample time directory in
`<case>/postProcessing/sampleDict/` whose value is closest to `target_t`.
"""
function pick_rheotool_sample_dir(case_dir::AbstractString, target_t::Float64)
    sample_root = joinpath(case_dir, "postProcessing", "sampleDict")
    isdir(sample_root) || error("missing $(sample_root); did rheoTool run?")
    dirs = filter(isdir, [joinpath(sample_root, d) for d in readdir(sample_root)])
    candidates = Tuple{Float64,String}[]
    for d in dirs
        t = tryparse(Float64, basename(d))
        t === nothing && continue
        t > 0 || continue
        push!(candidates, (t, d))
    end
    isempty(candidates) && error("no non-zero sample times found in $(sample_root)")
    _, best = findmin(t -> abs(t[1] - target_t), candidates)
    return candidates[best]
end

# ---------------------------------------------------------------------
# Kraken sampling helpers
# ---------------------------------------------------------------------

"""
Sample a cell-centered 2D field `f(Nx,Ny)` along a vertical line at
fractional x position `x_frac in [0, 1]`. Returns `(y_phys, value)` aligned
on rheoTool's sampled y coordinates `y_target` (linear interpolation in y).
"""
function sample_vertical_kraken(
    field::AbstractMatrix, Nx::Integer, Ny::Integer, x_frac::Real,
    y_target::AbstractVector,
)
    # Cell centers at i = 1..Nx, x_phys[i] = (i - 0.5) / Nx
    # Bracket x_frac between nearest two columns
    i_real = x_frac * Nx + 0.5
    i_lo = clamp(floor(Int, i_real), 1, Nx - 1)
    i_hi = i_lo + 1
    wx = clamp(i_real - i_lo, 0.0, 1.0)
    column = [(1 - wx) * field[i_lo, j] + wx * field[i_hi, j] for j in 1:Ny]
    y_phys = [(j - 0.5) / Ny for j in 1:Ny]
    out = similar(y_target, Float64)
    for (k, y) in enumerate(y_target)
        if y <= y_phys[1]
            out[k] = column[1]
        elseif y >= y_phys[end]
            out[k] = column[end]
        else
            j_lo = searchsortedlast(y_phys, y)
            j_hi = j_lo + 1
            wy = (y - y_phys[j_lo]) / (y_phys[j_hi] - y_phys[j_lo])
            out[k] = (1 - wy) * column[j_lo] + wy * column[j_hi]
        end
    end
    return (y=collect(y_target), values=out)
end

function sample_horizontal_kraken(
    field::AbstractMatrix, Nx::Integer, Ny::Integer, y_frac::Real,
    x_target::AbstractVector,
)
    j_real = y_frac * Ny + 0.5
    j_lo = clamp(floor(Int, j_real), 1, Ny - 1)
    j_hi = j_lo + 1
    wy = clamp(j_real - j_lo, 0.0, 1.0)
    row = [(1 - wy) * field[i, j_lo] + wy * field[i, j_hi] for i in 1:Nx]
    x_phys = [(i - 0.5) / Nx for i in 1:Nx]
    out = similar(x_target, Float64)
    for (k, x) in enumerate(x_target)
        if x <= x_phys[1]
            out[k] = row[1]
        elseif x >= x_phys[end]
            out[k] = row[end]
        else
            i_lo = searchsortedlast(x_phys, x)
            i_hi = i_lo + 1
            wx = (x - x_phys[i_lo]) / (x_phys[i_hi] - x_phys[i_lo])
            out[k] = (1 - wx) * row[i_lo] + wx * row[i_hi]
        end
    end
    return (x=collect(x_target), values=out)
end

function rel_l2_error(kraken::AbstractVector, ref::AbstractVector)
    @assert length(kraken) == length(ref)
    denom = sqrt(sum(abs2, ref))
    denom < eps() && return NaN
    return sqrt(sum(abs2.(kraken .- ref))) / denom
end

function rel_linf_error(kraken::AbstractVector, ref::AbstractVector)
    @assert length(kraken) == length(ref)
    scale = maximum(abs, ref)
    scale < eps() && return NaN
    return maximum(abs.(kraken .- ref)) / scale
end

# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------

function run_case(N::Int, output_dir::AbstractString, rheotool_case::AbstractString;
                  backend, T)
    nu_s = parse(Float64, get(ENV, "KRAKEN_NU_S", "0.1"))
    nu_p = parse(Float64, get(ENV, "KRAKEN_NU_P", "0.1"))
    u_max = parse(Float64, get(ENV, "KRAKEN_U_MAX", "0.005"))
    lambda_phys = parse(Float64, get(ENV, "KRAKEN_LAMBDA_PHYS", "1.0"))
    bsd_fraction = parse(Float64, get(ENV, "KRAKEN_BSD_FRACTION", "1.0"))
    end_time = parse(Float64, get(ENV, "KRAKEN_END_TIME", "8.0"))
    polymer_model = Symbol(get(ENV, "KRAKEN_POLYMER_MODEL", "oldroydb"))
    L_max = parse(Float64, get(ENV, "KRAKEN_L_MAX", "10.0"))
    max_subs = parse(Int, get(ENV, "KRAKEN_MAX_POLYMER_SUBSTEPS", "64"))
    diag_stride = parse(Int, get(ENV, "KRAKEN_DIAGNOSTIC_STRIDE", "0"))

    target_t = 8.0
    rheotool_t, rheotool_dir = pick_rheotool_sample_dir(rheotool_case, target_t)

    println("Reading rheoTool reference from $(rheotool_dir) (t = $(rheotool_t))")
    ref_U = read_rheotool_vertical_U(joinpath(rheotool_dir, "lineVert_x0.5_U.xy"))
    ref_tt = read_rheotool_horizontal_tautheta(joinpath(rheotool_dir, "lineHorz_y0.75_tau_theta.xy"))
    ref_E = read_rheotool_kinener(joinpath(rheotool_case, "kinEner.txt"))

    println("Running Kraken cavity coupled: N=$N, nu_s=$nu_s, nu_p=$nu_p, " *
            "u_max=$u_max, lambda_phys=$lambda_phys, end_time=$end_time")
    t_start = time()
    result = run_viscoelastic_logfv_cavity_coupled_2d(;
        N=N,
        nu_s=nu_s,
        nu_p=nu_p,
        lambda_phys=lambda_phys,
        bsd_fraction=bsd_fraction,
        u_max=u_max,
        L_max=L_max,
        polymer_model=polymer_model,
        max_polymer_substeps=max_subs,
        end_time=end_time,
        sample_times=Float64[end_time],
        diagnostic_stride=diag_stride,
        backend=backend,
        T=T,
    )
    elapsed = time() - t_start
    println("Kraken finished in $(round(elapsed, digits=1)) s; " *
            "completed_steps=$(result.completed_steps), " *
            "first_nonfinite_step=$(result.first_nonfinite_step)")

    snapshot_key = sort(collect(keys(result.snapshots)))[end]
    snap = result.snapshots[snapshot_key]

    Nx, Ny = result.N, result.N
    kr_u = sample_vertical_kraken(snap.ux, Nx, Ny, 0.5, ref_U.y)
    kr_v = sample_vertical_kraken(snap.uy, Nx, Ny, 0.5, ref_U.y)
    kr_psixy = sample_horizontal_kraken(snap.psixy, Nx, Ny, 0.75, ref_tt.x)
    kr_tauxy = sample_horizontal_kraken(snap.tauxy, Nx, Ny, 0.75, ref_tt.x)

    # Re-scale Kraken velocities to physical units (rheoTool plots U in lid-peak
    # units; the cavity convention is U_lid_peak = 1, so Kraken u/u_max in
    # lattice maps to the physical u/U).
    kr_u_phys = kr_u.values ./ u_max
    kr_v_phys = kr_v.values ./ u_max

    rel_l2_u = rel_l2_error(kr_u_phys, ref_U.ux)
    rel_linf_u = rel_linf_error(kr_u_phys, ref_U.ux)
    rel_l2_psixy = rel_l2_error(kr_psixy.values, ref_tt.theta_xy)
    rel_linf_psixy = rel_linf_error(kr_psixy.values, ref_tt.theta_xy)

    mkpath(output_dir)
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    case_tag = "kraken_N$(N)_$(stamp)"
    case_dir = joinpath(output_dir, case_tag)
    mkpath(case_dir)

    println("Writing CSV outputs to $(case_dir)")

    # Vertical profile CSV
    open(joinpath(case_dir, "profile_vertical_x0.5.csv"), "w") do io
        write(io, "y,kraken_ux,kraken_uy,rheotool_ux,rheotool_uy\n")
        for k in eachindex(ref_U.y)
            @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g\n",
                    ref_U.y[k], kr_u_phys[k], kr_v_phys[k],
                    ref_U.ux[k], ref_U.uy[k])
        end
    end

    # Horizontal profile CSV
    open(joinpath(case_dir, "profile_horizontal_y0.75.csv"), "w") do io
        write(io, "x,kraken_psixy,kraken_tauxy,rheotool_thetaxy,rheotool_tauxy\n")
        for k in eachindex(ref_tt.x)
            @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g\n",
                    ref_tt.x[k], kr_psixy.values[k], kr_tauxy.values[k],
                    ref_tt.theta_xy[k], ref_tt.tau_xy[k])
        end
    end

    # Energy history CSV (Kraken only here; rheoTool kinEner.txt is at the
    # case root and can be plotted alongside)
    open(joinpath(case_dir, "kinetic_energy_kraken.csv"), "w") do io
        write(io, "t_phys,kinetic_lu,elastic_lu\n")
        for (t, ke, el) in result.kinetic_energy_history
            @printf(io, "%.10g,%.10g,%.10g\n", t, ke, el)
        end
    end

    # Summary
    summary_path = joinpath(case_dir, "summary.csv")
    open(summary_path, "w") do io
        write(io, "key,value\n")
        @printf(io, "N,%d\n", N)
        @printf(io, "nu_s,%.10g\n", nu_s)
        @printf(io, "nu_p,%.10g\n", nu_p)
        @printf(io, "u_max,%.10g\n", u_max)
        @printf(io, "lambda_phys,%.10g\n", lambda_phys)
        @printf(io, "lambda_lu,%.10g\n", result.lambda_lu)
        @printf(io, "dt_phys,%.10g\n", result.dt_phys)
        @printf(io, "end_time,%.10g\n", end_time)
        @printf(io, "polymer_substeps,%d\n", result.selected_polymer_substeps)
        @printf(io, "max_steps,%d\n", result.max_steps)
        @printf(io, "completed_steps,%d\n", result.completed_steps)
        @printf(io, "first_nonfinite_step,%d\n", result.first_nonfinite_step)
        @printf(io, "rheotool_sample_t,%.10g\n", rheotool_t)
        @printf(io, "rel_l2_u_centerline,%.10g\n", rel_l2_u)
        @printf(io, "rel_linf_u_centerline,%.10g\n", rel_linf_u)
        @printf(io, "rel_l2_psixy_y075,%.10g\n", rel_l2_psixy)
        @printf(io, "rel_linf_psixy_y075,%.10g\n", rel_linf_psixy)
        @printf(io, "ref_u_max,%.10g\n", maximum(ref_U.ux))
        @printf(io, "kraken_u_max,%.10g\n", maximum(kr_u_phys))
        @printf(io, "ref_psixy_max,%.10g\n", maximum(abs, ref_tt.theta_xy))
        @printf(io, "kraken_psixy_max,%.10g\n", maximum(abs, kr_psixy.values))
        @printf(io, "wallclock_s,%.3f\n", elapsed)
    end

    # Field dump for ParaView later if needed
    open(joinpath(case_dir, "fields.jls"), "w") do io
        serialize(io, snap)
    end

    println()
    println("== Summary ==")
    println("rheoTool sample t  = $(rheotool_t)")
    println("kraken end time    = $(end_time) (dt_phys=$(result.dt_phys))")
    println("u(x=0.5,y)         rel L2 = $(rel_l2_u)   rel Linf = $(rel_linf_u)")
    println("psi_xy(x,y=0.75)   rel L2 = $(rel_l2_psixy) rel Linf = $(rel_linf_psixy)")
    println("u_max  ref / Kraken = $(maximum(ref_U.ux)) / $(maximum(kr_u_phys))")
    println()

    return (case_dir=case_dir,
            rel_l2_u=rel_l2_u, rel_linf_u=rel_linf_u,
            rel_l2_psixy=rel_l2_psixy, rel_linf_psixy=rel_linf_psixy)
end

function main()
    case_dir = get(ENV, "KRAKEN_RHEOTOOL_CASE", DEFAULT_RHEOTOOL_CASE)
    output_dir = get(ENV, "KRAKEN_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    N_list = parse_int_list(get(ENV, "KRAKEN_N_LIST", "64"))

    backend_name, backend, T = pick_backend()
    println("Backend = $(backend_name), float = $(T)")

    results = []
    for N in N_list
        push!(results, (N=N, run_case(N, output_dir, case_dir; backend=backend, T=T)...))
    end

    println()
    println("== All cases ==")
    for r in results
        println("N=$(r.N)  rel_L2 u = $(r.rel_l2_u)  rel_L2 psixy = $(r.rel_l2_psixy)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
