using Test
using Kraken

const AMR_D_CORNER_FIXTURE_DIR =
    joinpath(dirname(@__DIR__), "tmp", "M-H-ETA-DASHBOARD", "krk")

const AMR_D_CORNER_NESTED_DIR =
    joinpath(dirname(@__DIR__), "benchmarks", "krk", "amr_d_convergence_2d")

const AMR_D_CORNER_THRESHOLD = 1.0e-4
const AMR_D_CORNER_STEPS = 500
const AMR_D_CORNER_NESTED_STEPS = 200
const AMR_D_POISEUILLE_STEPS = 3000

function _corner_rho_route_native(result, volume_fine::Real)
    coarse = result.coarse_F
    patch = result.patch
    T = eltype(coarse)
    Nx, Ny = size(coarse, 1), size(coarse, 2)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    Lx, Ly = 2 * Nx, 2 * Ny
    return [Float64(sum(@view leaf[i, j, :])) / Float64(volume_fine)
            for (i, j) in ((1, 1), (Lx, 1), (1, Ly), (Lx, Ly))]
end

function _corner_rho_subcycled(result)
    spec = result.spec
    F = result.F
    leaf_nx = spec.Nx << spec.max_level
    leaf_ny = spec.Ny << spec.max_level
    rho = fill(NaN, leaf_nx, leaf_ny)

    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        scale = 1 << (spec.max_level - cell.level)
        mass = zero(eltype(F))
        for q in 1:9
            mass += F[cell_id, q]
        end
        volume = eltype(F)(cell.metrics.volume)
        rho_c = Float64(mass) / Float64(volume)
        i0 = (cell.i - 1) * scale + 1
        i1 = cell.i * scale
        j0 = (cell.j - 1) * scale + 1
        j1 = cell.j * scale
        rho[i0:i1, j0:j1] .= rho_c
    end

    return [rho[i, j] for (i, j) in ((1, 1), (leaf_nx, 1),
                                     (1, leaf_ny), (leaf_nx, leaf_ny))]
end

function _amr_d_corner_peak(result)
    corners = hasproperty(result, :spec) ?
        _corner_rho_subcycled(result) :
        _corner_rho_route_native(result, 0.25)
    return maximum(abs.(corners .- 1.0))
end

@testset "AMR-D corner non-regression (shear-active, M-H-ETA-7+8)" begin
    @testset "couette_xband_v_full_nested1" begin
        path = joinpath(AMR_D_CORNER_FIXTURE_DIR,
                        "couette_xband_v_full_nested1.krk")
        @test isfile(path)
        result = run_conservative_tree_amr_d_case_from_krk_2d(
            path; steps_override=AMR_D_CORNER_STEPS, T=Float64)
        peak = _amr_d_corner_peak(result)
        @info "couette_xband_v_full_nested1 corner peak |rho - 1|" peak
        @test peak <= AMR_D_CORNER_THRESHOLD
    end

    @testset "couette_yband_h_full_nested1" begin
        path = joinpath(AMR_D_CORNER_FIXTURE_DIR,
                        "couette_yband_h_full_nested1.krk")
        @test isfile(path)
        result = run_conservative_tree_amr_d_case_from_krk_2d(
            path; steps_override=AMR_D_CORNER_STEPS, T=Float64)
        peak = _amr_d_corner_peak(result)
        @info "couette_yband_h_full_nested1 corner peak |rho - 1|" peak
        @test peak <= AMR_D_CORNER_THRESHOLD
    end

    @testset "couette_H_nested1" begin
        path = joinpath(AMR_D_CORNER_FIXTURE_DIR, "couette_H_nested1.krk")
        @test isfile(path)
        result = run_conservative_tree_amr_d_case_from_krk_2d(
            path; steps_override=AMR_D_CORNER_STEPS, T=Float64)
        peak = _amr_d_corner_peak(result)
        @info "couette_H_nested1 corner peak |rho - 1|" peak
        @test peak <= AMR_D_CORNER_THRESHOLD
    end

    @testset "poiseuille_H_nested1 peak vs analytic" begin
        path = joinpath(AMR_D_CORNER_FIXTURE_DIR, "poiseuille_H_nested1.krk")
        @test isfile(path)
        result = run_conservative_tree_amr_d_case_from_krk_2d(
            path; steps_override=AMR_D_POISEUILLE_STEPS, T=Float64)
        ux_max = maximum(result.ux_profile)
        @info "poiseuille_H_nested1 peak ux vs analytic" ux_max result.linf_error result.relative_mass_drift
        @test ux_max ≈ 4.31e-3 atol=2e-4
        @test result.linf_error < 3e-4
        @test result.relative_mass_drift < 1e-12
    end

    @testset "couette_yband_h_full_nested2 (ratio=4, auto-cascade 2 levels)" begin
        path = joinpath(AMR_D_CORNER_NESTED_DIR,
                        "couette_yband_h_full_nested2.krk")
        @test isfile(path)
        result = run_conservative_tree_amr_d_case_from_krk_2d(
            path; steps_override=AMR_D_CORNER_NESTED_STEPS, T=Float64)
        peak = _amr_d_corner_peak(result)
        @info "couette_yband_h_full_nested2 corner peak |rho - 1|" peak
        @test peak <= AMR_D_CORNER_THRESHOLD
    end

    @testset "couette_yband_h_full_nested3 (ratio=8, auto-cascade 3 levels)" begin
        path = joinpath(AMR_D_CORNER_NESTED_DIR,
                        "couette_yband_h_full_nested3.krk")
        @test isfile(path)
        result = run_conservative_tree_amr_d_case_from_krk_2d(
            path; steps_override=AMR_D_CORNER_NESTED_STEPS, T=Float64)
        peak = _amr_d_corner_peak(result)
        @info "couette_yband_h_full_nested3 corner peak |rho - 1|" peak
        @test peak <= AMR_D_CORNER_THRESHOLD
    end
end
