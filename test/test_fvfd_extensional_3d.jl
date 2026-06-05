using Test
using Kraken
using KernelAbstractions

function center_stagnation_mean(A)
    Nx, Ny, Nz = size(A)
    i1 = max(1, Nx ÷ 2)
    i2 = min(Nx, Nx ÷ 2 + 1)
    j1 = max(1, Ny ÷ 2)
    j2 = min(Ny, Ny ÷ 2 + 1)
    return sum(@view A[i1:i2, j1:j2, :]) / ((i2 - i1 + 1) * (j2 - j1 + 1) * Nz)
end

@testset "FVFD 3D planar-extension Oldroyd-B canary" begin
    backend = CPU()
    FT = Float64

    Nx, Ny, Nz = 24, 24, 6
    lambda = 50.0
    epsilon_dot = 0.005
    lambda_epsilon = lambda * epsilon_dot
    @test 2 * lambda_epsilon < 1

    nu_total = 0.1
    beta = 0.5
    nu_s = beta * nu_total
    nu_p = (1 - beta) * nu_total
    max_steps = 1_000

    res = Kraken.run_viscoelastic_fvfd_extensional_3d(;
        Nx, Ny, Nz,
        epsilon_dot,
        ν_s=nu_s,
        ν_p=nu_p,
        lambda,
        max_steps,
        backend,
        FT,
        advection_scheme=:muscl_superbee,
        velocity_mode=:imposed,
    )

    finite_state = all(isfinite, res.ρ) &&
                   all(isfinite, res.ux) &&
                   all(isfinite, res.uy) &&
                   all(isfinite, res.uz) &&
                   all(isfinite, res.psi_xx) &&
                   all(isfinite, res.psi_xy) &&
                   all(isfinite, res.psi_xz) &&
                   all(isfinite, res.psi_yy) &&
                   all(isfinite, res.psi_yz) &&
                   all(isfinite, res.psi_zz) &&
                   all(isfinite, res.C_xx) &&
                   all(isfinite, res.C_yy) &&
                   all(isfinite, res.C_zz) &&
                   all(isfinite, res.tau_p_xx) &&
                   all(isfinite, res.tau_p_xy) &&
                   all(isfinite, res.tau_p_xz) &&
                   all(isfinite, res.tau_p_yy) &&
                   all(isfinite, res.tau_p_yz) &&
                   all(isfinite, res.tau_p_zz)

    Cxx_ref = 1 / (1 - 2 * lambda_epsilon)
    Cyy_ref = 1 / (1 + 2 * lambda_epsilon)
    Czz_ref = 1.0
    Cxx_meas = center_stagnation_mean(res.C_xx)
    Cyy_meas = center_stagnation_mean(res.C_yy)
    Czz_meas = center_stagnation_mean(res.C_zz)
    Cxy_meas = center_stagnation_mean(res.C_xy)
    rel_Cxx = abs(Cxx_meas - Cxx_ref) / Cxx_ref
    rel_Cyy = abs(Cyy_meas - Cyy_ref) / Cyy_ref
    abs_Czz = abs(Czz_meas - Czz_ref)
    grad_x = center_stagnation_mean(res.duxdx)
    grad_y = center_stagnation_mean(res.duydy)

    @testset "E1 NaN-free imposed-velocity run" begin
        @test res.completed_steps == max_steps
        @test res.velocity_mode === :imposed
        @test res.open_x_gradient_supported
        @test finite_state
    end

    @testset "E2 planar-extension fixed point" begin
        @test rel_Cxx <= 0.01
        @test rel_Cyy <= 0.01
        @test abs_Czz <= 0.01
        @test abs(Cxy_meas) <= 0.01
        @test abs(grad_x - epsilon_dot) <= 1e-12
        @test abs(grad_y + epsilon_dot) <= 1e-12
    end

    println("E1 NaN-free steps=$(res.completed_steps) finite=$(finite_state) mode=$(res.velocity_mode)")
    println(
        "E2 lambda_epsilon=$(lambda_epsilon) ",
        "Cxx=$(Cxx_meas) ref=$(Cxx_ref) rel_err_percent=$(100 * rel_Cxx) ",
        "Cyy=$(Cyy_meas) ref=$(Cyy_ref) rel_err_percent=$(100 * rel_Cyy)",
    )
    println(
        "E2 Czz=$(Czz_meas) abs_err=$(abs_Czz) Cxy=$(Cxy_meas) ",
        "grad_x=$(grad_x) grad_y=$(grad_y) max_substeps=$(res.max_substeps_observed)",
    )
end

println("EXIT=0")
