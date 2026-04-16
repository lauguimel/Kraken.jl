using Test
using Kraken

@testset "Conformation cylinder (TRT-LBM viscoelastic drag)" begin

    # ----------------------------------------------------------------
    # Newtonian-limit consistency check
    # ----------------------------------------------------------------
    # With the TRT-LBM Oldroyd-B scheme, in steady simple shear:
    #   τ_p,xy = G·C_xy = (ν_p/λ)·(λ γ̇) = ν_p γ̇
    # so the polymer behaves as an extra Newtonian viscosity ν_p,
    # and the velocity field should match a Newtonian fluid with ν_total.
    #
    # Drag accounting subtlety:
    # The Hermite stress source injects τ_p into the populations f, so
    # the MEA drag computed from f already captures the *full* effective
    # shear (solvent + polymer). The separate stress integral
    # `compute_polymeric_drag_2d` would double-count if added — it is
    # therefore reported but NOT included in the validated total at low Wi.
    # We thus compare `Cd_s` (MEA on f) to the Newtonian reference.
    Nx, Ny    = 240, 60
    radius    = 8
    u_in      = 0.02
    ν_s       = 0.06
    ν_p       = 0.04
    ν_total   = ν_s + ν_p
    lambda    = 5.0
    max_steps = 20_000
    avg_window= 2_000

    Wi = lambda * u_in / radius   # = 0.0125 — well within Newtonian regime
    Re = u_in * 2 * radius / ν_total

    # --- Newtonian reference ---
    ref = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=radius, u_in=u_in,
                            ν=ν_total, max_steps=max_steps, avg_window=avg_window)
    Cd_ref = ref.Cd

    # --- Conformation TRT-LBM ---
    res = run_conformation_cylinder_2d(; Nx=Nx, Ny=Ny, radius=radius,
                                         u_in=u_in, ν_s=ν_s, ν_p=ν_p,
                                         lambda=lambda, tau_plus=1.0,
                                         max_steps=max_steps, avg_window=avg_window)

    @info "Newtonian-limit cylinder" Re Wi Cd_ref Cd=res.Cd Cd_s=res.Cd_s Cd_p=res.Cd_p

    # Sanity: Cd > 0 and finite
    @test isfinite(res.Cd) && res.Cd > 0
    @test isfinite(res.Cd_s) && isfinite(res.Cd_p)

    # Polymer stress integral is positive (extra resistance) but is
    # already implicitly contained in Cd_s via the Hermite source — see
    # note above. We just check the order of magnitude.
    @test res.Cd_p > 0
    @test res.Cd_p < res.Cd_s

    # Validation: MEA drag (Cd_s) matches the Newtonian Cd at ν_total
    # to within 10 % at this low Wi (~0.0125).
    @test isapprox(res.Cd_s, Cd_ref; rtol=0.10)

    # Corrected total: Cd is now Cd_s (no double counting of τ_p).
    @test isapprox(res.Cd, Cd_ref; rtol=0.10)

    # Conformation should remain physical: tr(C) ≥ 2, det(C) > 0
    Cxx = res.C_xx; Cxy = res.C_xy; Cyy = res.C_yy
    fluid = .!Bool.(reshape(0:0, 1, 1))  # placeholder, recompute mask below
    # Use direct tests on a few interior fluid points away from cylinder/walls
    i_probe, j_probe = Nx ÷ 2, Ny ÷ 2 + radius + 5  # well above cylinder
    if 1 ≤ i_probe ≤ Nx && 1 ≤ j_probe ≤ Ny
        tr_C  = Cxx[i_probe, j_probe] + Cyy[i_probe, j_probe]
        det_C = Cxx[i_probe, j_probe]*Cyy[i_probe, j_probe] - Cxy[i_probe, j_probe]^2
        @info "C probe" tr_C det_C
        @test tr_C  > 1.99
        @test det_C > 0.0
    end
end

@testset "Conformation kernels GPU parity" begin
    cuda_ok = false
    try
        @eval using CUDA
        cuda_ok = CUDA.functional()
    catch
        cuda_ok = false
    end

    if !cuda_ok
        @info "CUDA not available, skipping GPU parity test"
        @test true
    else
        using Random
        Random.seed!(1234)
        Nx, Ny = 16, 16
        FT = Float64
        C_xx = FT.(1 .+ 0.1 .* randn(Nx, Ny))
        C_xy = FT.(0.1 .* randn(Nx, Ny))
        C_yy = FT.(1 .+ 0.1 .* randn(Nx, Ny))
        ux = FT.(0.05 .* randn(Nx, Ny))
        uy = FT.(0.05 .* randn(Nx, Ny))
        is_solid = falses(Nx, Ny)

        g_xx_cpu = zeros(FT, Nx, Ny, 9)
        g_xy_cpu = zeros(FT, Nx, Ny, 9)
        g_yy_cpu = zeros(FT, Nx, Ny, 9)
        Kraken.init_conformation_field_2d!(g_xx_cpu, g_xy_cpu, g_yy_cpu,
                                           C_xx, C_xy, C_yy, ux, uy)

        g_xx_gpu = CuArray(g_xx_cpu)
        g_xy_gpu = CuArray(g_xy_cpu)
        g_yy_gpu = CuArray(g_yy_cpu)
        C_xx_g = CuArray(C_xx); C_xy_g = CuArray(C_xy); C_yy_g = CuArray(C_yy)
        ux_g = CuArray(ux); uy_g = CuArray(uy)
        is_solid_g = CuArray(is_solid)

        tau_plus = 1.0; lam = 10.0
        for (g_cpu, g_gpu, comp) in ((g_xx_cpu, g_xx_gpu, 1),
                                      (g_xy_cpu, g_xy_gpu, 2),
                                      (g_yy_cpu, g_yy_gpu, 3))
            Kraken.collide_conformation_2d!(g_cpu, C_xx, C_xy, C_yy, ux, uy,
                                            is_solid, tau_plus, lam, comp)
            Kraken.collide_conformation_2d!(g_gpu, C_xx_g, C_xy_g, C_yy_g,
                                            ux_g, uy_g, is_solid_g,
                                            tau_plus, lam, comp)
            @test maximum(abs, Array(g_gpu) .- g_cpu) < 1e-12
        end
    end
end
