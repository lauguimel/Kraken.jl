# Run all Kraken.jl benchmarks
# Usage: julia --project benchmarks/run_all.jl [--gpu]

println("=" ^ 60)
println("  Kraken.jl Benchmark Suite")
println("=" ^ 60)

include("convergence_poiseuille.jl")
run_poiseuille_convergence()

include("convergence_taylor_green.jl")
run_taylor_green_convergence()

include("perf_mlups.jl")
run_mlups_benchmark(; gpu="--gpu" in ARGS)

# Cavity sweep is slow (Re=1000 needs 500k steps) — run separately
# include("convergence_cavity.jl")
# run_cavity_sweep()

println("\n" * "=" ^ 60)
println("  Benchmark suite complete")
println("=" ^ 60)
