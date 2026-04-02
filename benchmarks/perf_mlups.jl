# MLUPS performance scaling benchmark
# Measures throughput on CPU and GPU (if available)
using Kraken
using Printf

function run_mlups_benchmark(; gpu=false)
    Ns = [64, 128, 256, 512]
    steps = 200

    println("\n=== MLUPS Performance Scaling ===")

    # CPU benchmark
    println("\n--- CPU ---")
    @printf("  %5s   %10s\n", "N", "MLUPS")
    @printf("  %5s   %10s\n", "-----", "--------")

    cpu_results = benchmark_mlups(; Ns=Ns, steps=steps)
    for (N, mlups) in cpu_results
        @printf("  %5d   %10.1f\n", N, mlups)
    end

    # GPU benchmark (optional)
    if gpu
        gpu_backend = nothing
        try
            using CUDA
            if CUDA.functional()
                gpu_backend = CUDABackend()
                println("\n--- CUDA GPU ---")
            end
        catch; end

        if isnothing(gpu_backend)
            try
                using Metal
                if Metal.functional()
                    gpu_backend = MetalBackend()
                    println("\n--- Metal GPU ---")
                end
            catch; end
        end

        if !isnothing(gpu_backend)
            @printf("  %5s   %10s   %8s\n", "N", "MLUPS", "Speedup")
            @printf("  %5s   %10s   %8s\n", "-----", "--------", "-------")

            gpu_Ns = [64, 128, 256, 512, 1024]
            gpu_results = benchmark_mlups(; Ns=gpu_Ns, steps=steps, backend=gpu_backend)

            for (N, mlups_gpu) in gpu_results
                # Find CPU result for same N
                cpu_mlups = 0.0
                for (Nc, mc) in cpu_results
                    Nc == N && (cpu_mlups = mc; break)
                end
                speedup = cpu_mlups > 0 ? mlups_gpu / cpu_mlups : NaN
                @printf("  %5d   %10.1f   %8.1fx\n", N, mlups_gpu,
                        isnan(speedup) ? 0.0 : speedup)
            end
        else
            println("\nNo GPU backend available. Skipping GPU benchmark.")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    gpu = "--gpu" in ARGS
    run_mlups_benchmark(; gpu=gpu)
end
