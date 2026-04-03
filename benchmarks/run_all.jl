# Run all Kraken.jl benchmarks
# Usage: julia --project benchmarks/run_all.jl [--gpu]

println("=" ^ 60)
println("  Kraken.jl Benchmark Suite")
println("=" ^ 60)

include("convergence_poiseuille.jl")
run_poiseuille_convergence()

include("convergence_taylor_green.jl")
run_taylor_green_convergence()

include("convergence_thermal.jl")
run_thermal_convergence()

include("perf_mlups.jl")
run_mlups_benchmark(; gpu="--gpu" in ARGS)

# Slower benchmarks — run separately or uncomment as needed
# include("convergence_cavity.jl")
# run_cavity_sweep()

# include("convergence_vof.jl")
# run_vof_convergence()

# include("convergence_rheology.jl")
# run_rheology_convergence()

# include("perf_gpu_physics.jl")
# run_physics_benchmark(; gpu="--gpu" in ARGS)

println("\n" * "=" ^ 60)
println("  Benchmark suite complete")
println("=" ^ 60)
