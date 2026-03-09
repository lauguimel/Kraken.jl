using Test
using Kraken

@testset "Laplacian 2D — O(h²) convergence" begin
    # f(x,y) = sin(2πx)sin(2πy)
    # ∇²f    = -8π²sin(2πx)sin(2πy)

    grids = [32, 64, 128, 256]
    errors = Float64[]

    for N in grids
        dx = 1.0 / (N - 1)
        xs = range(0.0, 1.0, length=N)
        ys = range(0.0, 1.0, length=N)

        f = [sin(2π * x) * sin(2π * y) for x in xs, y in ys]
        exact = [-8π^2 * sin(2π * x) * sin(2π * y) for x in xs, y in ys]

        out = zeros(N, N)
        Kraken.laplacian!(out, f, dx)

        # L2 error on interior points only
        diff = out[2:end-1, 2:end-1] .- exact[2:end-1, 2:end-1]
        l2 = sqrt(sum(diff .^ 2) / length(diff))
        push!(errors, l2)
    end

    # Check O(h²) convergence between successive grids
    for k in 1:length(errors)-1
        rate = log(errors[k] / errors[k+1]) / log(2)
        @test 1.9 < rate < 2.1
    end
end

# GPU test (Metal on Apple Silicon)
@testset "Laplacian 2D — Metal GPU" begin
    gpu_available = false
    try
        @eval using Metal
        if Metal.functional()
            gpu_available = true
        end
    catch
    end

    if !gpu_available
        @info "Metal not available, skipping GPU tests"
        @test_skip false
    else
        N = 64
        dx = 1.0 / (N - 1)
        xs = range(0.0, 1.0, length=N)
        ys = range(0.0, 1.0, length=N)

        f_cpu = Float32[sin(2π * x) * sin(2π * y) for x in xs, y in ys]
        out_cpu = zeros(Float32, N, N)
        Kraken.laplacian!(out_cpu, f_cpu, Float32(dx))

        f_gpu = MtlArray(f_cpu)
        out_gpu = Metal.zeros(Float32, N, N)
        Kraken.laplacian!(out_gpu, f_gpu, Float32(dx))

        @test maximum(abs.(Array(out_gpu) .- out_cpu)) < 1e-5
    end
end
