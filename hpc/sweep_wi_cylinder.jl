include(joinpath(@__DIR__, "..", "src", "Kraken.jl"))
using .Kraken
using Printf
using Dates
using KernelAbstractions

const _CUDA_MOD = try
    @eval using CUDA
    getfield(Main, :CUDA)
catch
    nothing
end

const _METAL_MOD = if Sys.isapple()
    try
        @eval using Metal
        getfield(Main, :Metal)
    catch
        nothing
    end
else
    nothing
end

function _select_backend()
    requested = lowercase(get(ENV, "KRAKEN_BACKEND", "auto"))
    if requested in ("auto", "cuda") && _CUDA_MOD !== nothing
        try
            if Base.invokelatest(getfield(_CUDA_MOD, :functional))
                backend = Base.invokelatest(getfield(_CUDA_MOD, :CUDABackend))
                device = Base.invokelatest(getfield(_CUDA_MOD, :device))
                name = Base.invokelatest(getfield(_CUDA_MOD, :name), device)
                return backend, Float64, "CUDA $name"
            end
        catch err
            requested == "cuda" && rethrow(err)
        end
    end
    if requested in ("auto", "metal") && _METAL_MOD !== nothing
        try
            if Base.invokelatest(getfield(_METAL_MOD, :functional))
                backend = Base.invokelatest(getfield(_METAL_MOD, :MetalBackend))
                return backend, Float32, "Metal"
            end
        catch err
            requested == "metal" && rethrow(err)
        end
    end
    return KernelAbstractions.CPU(), Float64, "CPU"
end

# --- Configuration ---
backend, FT, backend_label = _select_backend()

R             = parse(Int,     get(ENV, "KRAKEN_R",             "30"))
u_mean        = parse(Float64, get(ENV, "KRAKEN_U_MEAN",        "0.005"))
beta          = parse(Float64, get(ENV, "KRAKEN_BETA",          "0.59"))
steps_per_R   = parse(Int,     get(ENV, "KRAKEN_STEPS_PER_R",   "6000"))
avg_divisor   = parse(Int,     get(ENV, "KRAKEN_AVG_DIVISOR",   "10"))
drag_stride   = parse(Int,     get(ENV, "KRAKEN_DRAG_STRIDE",   "200"))
geometry_mode = get(ENV, "KRAKEN_GEOMETRY_MODE", "centered_legacy")
solvent_magic = parse(Float64, get(ENV, "KRAKEN_SOLVENT_MAGIC",  string(3/16)))
conf_magic    = parse(Float64, get(ENV, "KRAKEN_CONFORMATION_MAGIC", "1e-6"))

wi_list_str   = get(ENV, "KRAKEN_WI_LIST", "0.05,0.1,0.2,0.5,1.0")
Wi_values     = [parse(Float64, strip(x)) for x in split(wi_list_str, ',')]

# Source modes to compare:
# A) post + ce_corrected + source_scaled_mea  (former single-point match at Wi=0.1)
# B) post + liu_direct   + post_source_mea    (no CE factor, post-collision)
# C) integrated + liu_direct + post_source_mea (no CE factor, in-collision = Liu Eq. 25 original)
# Each tuple: (solvent_source_mode, hermite_source_mode, drag_mode, label)
source_modes = [
    (:post_collision,       :ce_corrected, :source_scaled_mea, "A_post_CE"),
    (:post_collision,       :liu_direct,   :post_source_mea,   "B_post_Liu"),
    (:integrated_collision, :liu_direct,   :post_source_mea,   "C_int_Liu"),
]

# Geometry (same as force_scheme_matrix.jl)
function cylinder_geometry(R::Int, mode::AbstractString)
    if mode == "exact_nodes"
        return (Nx = 30R + 1, Ny = 4R + 1, cx = 15R, cy = 2R)
    elseif mode == "centered_legacy"
        Nx = 30R
        Ny = 4R
        return (Nx = Nx, Ny = Ny, cx = 15R, cy = (Ny - 1) / 2)
    elseif mode == "legacy"
        return (Nx = 30R, Ny = 4R, cx = 15R, cy = 2R)
    else
        error("unknown KRAKEN_GEOMETRY_MODE=$mode; expected exact_nodes, centered_legacy, or legacy")
    end
end

geom = cylinder_geometry(R, geometry_mode)
Nx = geom.Nx
Ny = geom.Ny
cx = geom.cx
cy = geom.cy

# Header
println("=" ^ 120)
model_name = Symbol(get(ENV, "KRAKEN_MODEL", "direct"))
model_name in (:direct, :logconf) ||
    error("unknown KRAKEN_MODEL=$(model_name); expected direct or logconf")

println("Wi sweep — 2D Oldroyd-B cylinder ($(String(model_name)))")
println("Date: $(Dates.now())")
println("Backend: $backend_label, FT=$FT")
@printf("R=%d  u_mean=%.6g  beta=%.4f  steps_per_R=%d\n", R, u_mean, beta, steps_per_R)
@printf("Geometry: Nx=%d Ny=%d cx=%d cy=%.1f  (mode=%s)\n", Nx, Ny, cx, cy, geometry_mode)
@printf("Solvent Λs=%.6g  Conformation Λp=%.6g  tau_plus=1.0\n", solvent_magic, conf_magic)
println("Wi values: ", join(Wi_values, ", "))
println("=" ^ 120)

# Run Newtonian baseline first
println("\n--- Newtonian baseline ---")
let
    ν_total = u_mean * R
    t0 = time()
    result = run_cylinder_libb_2d(;
        Nx, Ny, radius = R, cx, cy,
        u_in = FT(1.5 * u_mean), ν = FT(ν_total), inlet = :parabolic,
        max_steps = steps_per_R * R, avg_window = max(1, steps_per_R * R ÷ avg_divisor),
        drag_stride, momentum_exchange_mode = :mei_reconstruct,
        solvent_magic, backend, T = FT)
    dt = time() - t0
    Cl = hasproperty(result, :Cl) ? result.Cl : 2 * result.Fy / (u_mean^2 * 2R)
    @printf("  Cd_Newt = %.6f   Cl = %.3e   (%.1fs)\n", result.Cd, Cl, dt)
    global Cd_newt = result.Cd
end

# Results table
println("\n--- Viscoelastic sweep (3 modes × $(length(Wi_values)) Wi) ---")
@printf("%-12s %-8s %-12s %-9s %-12s %-10s %-12s %-12s %-10s %-7s\n",
        "mode", "Wi", "Cd", "Cd/Newt", "Cd_s", "Cd_p", "Cd_mea_post", "u_max", "τ_xx_max", "t(s)")
println("-" ^ 130)

using Serialization
using DelimitedFiles
results_dir = get(ENV, "RESULTS_DIR", "results")
isdir(results_dir) || mkpath(results_dir)
fields_dir = joinpath(results_dir, "sweep_wi_fields_$(Dates.format(Dates.now(), "yyyymmdd_HHMM"))")
mkpath(fields_dir)
println("Field dumps will be saved to: ", fields_dir)

results = []
for (ssm, hsm, dm, label) in source_modes
    for Wi in Wi_values
        ν_total = u_mean * R
        ν_s = beta * ν_total
        ν_p = (1 - beta) * ν_total
        λ = Wi * R / u_mean
        G = ν_p / λ
        model = model_name === :logconf ?
            LogConfOldroydB(G = FT(G), λ = FT(λ)) :
            OldroydB(G = FT(G), λ = FT(λ))
        max_steps = steps_per_R * R

        t0 = time()
        result = try
            run_conformation_cylinder_libb_2d(;
                Nx, Ny, radius = R, cx, cy,
                u_mean = FT(u_mean), ν_s = FT(ν_s),
                polymer_model = model, polymer_bc = CNEBB(),
                inlet = :parabolic, ρ_out = one(FT), tau_plus = one(FT),
                max_steps, avg_window = max(1, max_steps ÷ avg_divisor),
                drag_stride,
                drag_mode = dm,
                hermite_source_mode = hsm,
                solvent_source_mode = ssm,
                solvent_magic,
                conformation_magic = conf_magic,
                conformation_collision = :trt,
                momentum_exchange_mode = :mei_reconstruct,
                allow_diagnostic_force_mode = dm === :source_scaled_mea,
                allow_diagnostic_log_wall_bc = model_name === :logconf,
                backend, FT)
        catch err
            @warn "$label Wi=$Wi FAILED" err
            nothing
        end
        dt = time() - t0

        if result !== nothing
            ratio = result.Cd / Cd_newt
            Cd_s = hasproperty(result, :Cd_s) ? result.Cd_s : NaN
            Cd_p = hasproperty(result, :Cd_p) ? result.Cd_p : NaN
            Cd_mea_post = hasproperty(result, :Cd_mea_post_source) ? result.Cd_mea_post_source : NaN

            # Dump diagnostic fields: profile at x = cx (across cylinder),
            # x = cx + 2R (downstream wake), and global u_max / τ_p_max
            i_wake = min(Nx, Int(round(cx + 2R)))
            u_profile_cyl = result.ux[Int(cx), :]
            u_profile_wake = result.ux[i_wake, :]
            tau_xx_cyl = result.tau_p_xx[Int(cx), :]
            tau_xy_cyl = result.tau_p_xy[Int(cx), :]
            u_max = maximum(abs, result.ux)
            tau_xx_max = maximum(abs, result.tau_p_xx)
            tau_xy_max = maximum(abs, result.tau_p_xy)

            tag = "$(label)_Wi$(replace(string(Wi), "." => "p"))"
            # Compact text dump: profile across cylinder + wake + scalar summary
            open(joinpath(fields_dir, "$(tag)_profiles.txt"), "w") do io
                println(io, "# mode=$label Wi=$Wi R=$R β=$beta Re_R=1")
                println(io, "# Cd=$(result.Cd) Cd_s=$Cd_s Cd_p=$Cd_p Cd_mea_post=$Cd_mea_post")
                println(io, "# u_max=$u_max tau_xx_max=$tau_xx_max tau_xy_max=$tau_xy_max")
                println(io, "# columns: j  ux_at_cx  ux_at_cx+2R  tau_p_xx_at_cx  tau_p_xy_at_cx")
                for j in 1:Ny
                    @printf(io, "%4d  %.8e  %.8e  %.8e  %.8e\n",
                            j, u_profile_cyl[j], u_profile_wake[j],
                            tau_xx_cyl[j], tau_xy_cyl[j])
                end
            end
            # Full state for deeper diagnosis if needed
            serialize(joinpath(fields_dir, "$(tag)_state.jls"), Dict(
                "mode" => label, "Wi" => Wi, "Cd" => result.Cd,
                "ux" => result.ux, "uy" => result.uy,
                "C_xx" => result.C_xx, "C_xy" => result.C_xy, "C_yy" => result.C_yy,
                "tau_p_xx" => result.tau_p_xx, "tau_p_xy" => result.tau_p_xy,
                "tau_p_yy" => result.tau_p_yy,
                "is_solid" => result.is_solid))

            @printf("%-12s %-8.3f %-12.4f %-9.4f %-12.4f %-10.4f %-12.4f %-12.4e %-10.4e %-7.1f\n",
                    label, Wi, result.Cd, ratio, Cd_s, Cd_p, Cd_mea_post,
                    u_max, tau_xx_max, dt)
            push!(results, (mode = label, Wi = Wi, Cd = result.Cd, ratio = ratio,
                            Cd_s = Cd_s, Cd_p = Cd_p, Cd_mea_post = Cd_mea_post,
                            u_max = u_max, tau_xx_max = tau_xx_max, time = dt))
        else
            @printf("%-12s %-8.3f %-12s %-9s %-12s %-10s %-12s %-12s %-10s %-7.1f\n",
                    label, Wi, "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", dt)
            push!(results, (mode = label, Wi = Wi, Cd = NaN, ratio = NaN,
                            Cd_s = NaN, Cd_p = NaN, Cd_mea_post = NaN,
                            u_max = NaN, tau_xx_max = NaN, time = dt))
        end
        flush(stdout)
    end
end

# Summary — by mode
println("\n" * "=" ^ 130)
println("SUMMARY — Cd(Wi) for Oldroyd-B cylinder, R=$R, beta=$beta, Re_R=1")
println("=" ^ 130)
println("Cd_Newt = $(@sprintf("%.6f", Cd_newt))")
println()
@printf("%-12s %-8s %-14s %-14s %-14s %-14s\n", "mode", "Wi", "Cd", "Cd/Newt", "Cd_s/Newt", "u_max")
println("-" ^ 80)
for r in results
    @printf("%-12s %-8.3f %-14.6f %-14.6f %-14.6f %-14.4e\n",
            r.mode, r.Wi, r.Cd, r.ratio, r.Cd_s/Cd_newt, r.u_max)
end
println("\n--- Reference (rheoTool finite-Re) ---")
println("  Wi=0.05 → Cd=131.81 (-0.4% vs Newt)")
println("  Wi=0.10 → Cd=130.43 (-1.5%)")
println("  Wi=0.20 → Cd=126.83 (-4.2%)")
println("\nDiagnostic interpretation:")
println("  - If A_post_CE reproduces previous run (Cd@Wi=0.1 ≈ 130.74): sanity OK")
println("  - If B/C give slope similar to rheoTool (small Cd variation): CE factor is the bug")
println("  - If B and C agree but differ from A: post vs in-collision distinction is real")
println("  - If B/C still wrong: bug is elsewhere — check field dumps in $fields_dir")
println("Done.")
