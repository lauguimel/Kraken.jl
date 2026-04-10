# Kraken.jl benchmark suite — single entry point.
#
# Runs the v0.1.0 in-scope benchmarks (single-phase + thermal + grid
# refinement) and writes structured CSV results to benchmarks/results/.
# Figures for the documentation are generated separately by
# view/generate_doc_figures.jl from the same CSVs.
#
# Usage:
#     julia --project benchmarks/run_all.jl [flags]
#
# Flags:
#     --quick                run a reduced subset (< 5 min on CPU)
#     --gpu                  also run GPU benchmarks (requires CUDA or Metal)
#     --hardware-id=<key>    label in hardware.toml (default: apple_m2)
#     --skip-existing        skip cases whose CSV already exists
#     --output-dir=<path>    override benchmarks/results
#
# Example (laptop, quick CPU subset):
#     julia --project benchmarks/run_all.jl --quick
#
# Example (HPC GPU run, full sweep, via PBS):
#     julia --project benchmarks/run_all.jl --gpu --hardware-id=aqua_h100

include("utils.jl")

opts = parse_args()

println("=" ^ 64)
println("  Kraken.jl benchmark suite")
println("  hardware-id : ", opts.hardware_id)
println("  mode        : ", opts.quick ? "quick (< 5 min CPU)" : "full")
println("  gpu         : ", opts.gpu ? "enabled" : "disabled")
println("  output dir  : ", opts.output_dir)
println("=" ^ 64)

isdir(opts.output_dir) || mkpath(opts.output_dir)

# --- Convergence benchmarks (fast, always run) -----------------------------
println("\n[1/3] Convergence benchmarks")

include("convergence_poiseuille.jl")
Nys, errs = run_poiseuille_convergence()
rows = [
    (; case="poiseuille", N=Ny, error_L2=errs[i],
       observed_order = i > 1 ? log2(errs[i-1] / errs[i]) : NaN,
       hardware_id = opts.hardware_id)
    for (i, Ny) in enumerate(Nys)
]
write_csv(joinpath(opts.output_dir, "convergence_poiseuille_$(opts.hardware_id)_$(timestamp()).csv"), rows)

include("convergence_taylor_green.jl")
try
    res = run_taylor_green_convergence()
    @info "Taylor-Green convergence done"
catch e
    @warn "Taylor-Green convergence failed" exception=(e, catch_backtrace())
end

if !opts.quick
    include("convergence_thermal.jl")
    try
        run_thermal_convergence()
    catch e
        @warn "Thermal convergence failed" exception=(e, catch_backtrace())
    end

    include("convergence_cavity.jl")
    try
        run_cavity_sweep()
    catch e
        @warn "Cavity convergence failed" exception=(e, catch_backtrace())
    end
end

# --- Performance benchmarks -----------------------------------------------
println("\n[2/3] Performance (MLUPS)")

include("perf_mlups.jl")
try
    run_mlups_benchmark(; gpu=opts.gpu)
catch e
    @warn "MLUPS benchmark failed" exception=(e, catch_backtrace())
end

if !opts.quick && opts.gpu
    include("perf_gpu_physics.jl")
    try
        run_physics_benchmark(; gpu=true)
    catch e
        @warn "GPU physics benchmark failed" exception=(e, catch_backtrace())
    end
end

# --- External comparisons (placeholder for Phase 5.4) ---------------------
println("\n[3/3] External comparisons")
println("  (external comparison data lives in benchmarks/external/;")
println("   see Phase 5.4 in PLAN.md)")

println("\n" * "=" ^ 64)
println("  Benchmark suite complete")
println("  Results : ", opts.output_dir)
println("=" ^ 64)
