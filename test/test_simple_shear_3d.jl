using Test
using Kraken, KernelAbstractions

# ====================================================================
# Calibration test for the 3D conformation TRT kernel + Hermite source.
#
# Setup: homogeneous shear in 3D — u_x(y) = γ̇·(y − y_c), u_y = u_z = 0.
# Periodic streaming on all 6 faces (no walls, no inlet/outlet, no
# curved geometry). Velocity field is FROZEN (set analytically each
# step) so the only thing being tested is:
#
#   collide_conformation_3d! ×6  +  Hermite source apply_hermite_source_3d!
#
# Steady-state analytical (Oldroyd-B):
#   C_xy = λ·γ̇
#   C_xx = 1 + 2·(λ·γ̇)²
#   C_yy = C_zz = 1
#   C_xz = C_yz = 0
#   τ_p_xy = G·C_xy = ν_p·γ̇
#   N1 = τ_p_xx − τ_p_yy = 2·G·(λ·γ̇)² = 2·ν_p·λ·γ̇²
#
# This isolates the conformation evolution from the LBM solver, drag
# integration, blockage and inlet/outlet BC. If this test passes, the
# 3D Hermite source + TRT collide are calibrated correctly and the
# Aqua sphere drag bug must be in `compute_drag_libb_3d`. If it fails,
# the bug is in the kernels themselves.
# ====================================================================

@testset "Simple shear 3D — Oldroyd-B conformation calibration" begin
    backend = KernelAbstractions.CPU()
    FT = Float64

    Nx, Ny, Nz = 8, 16, 8
    γ̇  = 0.005
    λ  = 50.0
    ν_p = 0.1
    G   = ν_p / λ
    Wi_eff = γ̇ * λ        # = 0.25 — moderate, well-conditioned

    # Analytical steady state (Oldroyd-B)
    C_xy_an = λ * γ̇
    C_xx_an = 1 + 2 * (λ * γ̇)^2
    τ_xy_an = G * C_xy_an              # = ν_p · γ̇
    N1_an   = 2 * G * (λ * γ̇)^2       # = 2 · ν_p · λ · γ̇²

    # ---------- velocity field (analytical, frozen) ----------
    ux_h = zeros(FT, Nx, Ny, Nz)
    uy_h = zeros(FT, Nx, Ny, Nz)
    uz_h = zeros(FT, Nx, Ny, Nz)
    yc = (Ny + 1) / 2
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ux_h[i, j, k] = γ̇ * (j - yc)
    end
    ux = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz); copyto!(ux, ux_h)
    uy = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz); copyto!(uy, uy_h)
    uz = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz); copyto!(uz, uz_h)
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny, Nz)

    # ---------- conformation fields ----------
    C_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(C_xx, FT(1))
    C_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(C_yy, FT(1))
    C_zz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(C_zz, FT(1))
    C_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    C_xz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    C_yz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)

    g_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    g_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    g_xz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    g_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    g_yz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    g_zz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    init_conformation_field_3d!(g_xx, C_xx, ux, uy, uz)
    init_conformation_field_3d!(g_xy, C_xy, ux, uy, uz)
    init_conformation_field_3d!(g_xz, C_xz, ux, uy, uz)
    init_conformation_field_3d!(g_yy, C_yy, ux, uy, uz)
    init_conformation_field_3d!(g_yz, C_yz, ux, uy, uz)
    init_conformation_field_3d!(g_zz, C_zz, ux, uy, uz)
    g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_xz_buf = similar(g_xz)
    g_yy_buf = similar(g_yy); g_yz_buf = similar(g_yz); g_zz_buf = similar(g_zz)

    tau_p_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tau_p_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tau_p_xz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tau_p_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tau_p_yz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tau_p_zz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)

    # Sufficient steps for steady state: t_relax ≈ 5λ = 250, take 20·λ
    # to converge at low Wi.
    n_steps = 20 * Int(round(λ))
    polymer_model = OldroydB(G=FT(G), λ=FT(λ))

    for step in 1:n_steps
        # Periodic streaming on all 6 faces (no wall pollution)
        stream_fully_periodic_3d!(g_xx_buf, g_xx, Nx, Ny, Nz)
        stream_fully_periodic_3d!(g_xy_buf, g_xy, Nx, Ny, Nz)
        stream_fully_periodic_3d!(g_xz_buf, g_xz, Nx, Ny, Nz)
        stream_fully_periodic_3d!(g_yy_buf, g_yy, Nx, Ny, Nz)
        stream_fully_periodic_3d!(g_yz_buf, g_yz, Nx, Ny, Nz)
        stream_fully_periodic_3d!(g_zz_buf, g_zz, Nx, Ny, Nz)

        g_xx, g_xx_buf = g_xx_buf, g_xx
        g_xy, g_xy_buf = g_xy_buf, g_xy
        g_xz, g_xz_buf = g_xz_buf, g_xz
        g_yy, g_yy_buf = g_yy_buf, g_yy
        g_yz, g_yz_buf = g_yz_buf, g_yz
        g_zz, g_zz_buf = g_zz_buf, g_zz

        compute_conformation_macro_3d!(C_xx, g_xx)
        compute_conformation_macro_3d!(C_xy, g_xy)
        compute_conformation_macro_3d!(C_xz, g_xz)
        compute_conformation_macro_3d!(C_yy, g_yy)
        compute_conformation_macro_3d!(C_yz, g_yz)
        compute_conformation_macro_3d!(C_zz, g_zz)

        for (g, Cf, comp) in ((g_xx, C_xx, 1), (g_xy, C_xy, 2), (g_xz, C_xz, 3),
                                (g_yy, C_yy, 4), (g_yz, C_yz, 5), (g_zz, C_zz, 6))
            collide_conformation_3d!(g, Cf, ux, uy, uz,
                                       C_xx, C_xy, C_xz, C_yy, C_yz, C_zz, is_solid,
                                       1.0, λ; component=comp)
        end

        update_polymer_stress_3d!(tau_p_xx, tau_p_xy, tau_p_xz,
                                    tau_p_yy, tau_p_yz, tau_p_zz,
                                    C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
                                    polymer_model)
    end

    # ---------- evaluation at the bulk centre cell ----------
    ic, jc, kc = Nx ÷ 2, Ny ÷ 2, Nz ÷ 2
    Cxx_num = Array(C_xx)[ic, jc, kc]
    Cxy_num = Array(C_xy)[ic, jc, kc]
    Cxz_num = Array(C_xz)[ic, jc, kc]
    Cyy_num = Array(C_yy)[ic, jc, kc]
    Cyz_num = Array(C_yz)[ic, jc, kc]
    Czz_num = Array(C_zz)[ic, jc, kc]
    txy_num = Array(tau_p_xy)[ic, jc, kc]
    txx_num = Array(tau_p_xx)[ic, jc, kc]
    tyy_num = Array(tau_p_yy)[ic, jc, kc]
    N1_num  = txx_num - tyy_num

    @info "3D simple shear @ centre" Wi_eff Cxx_num Cxx_an Cxy_num Cxy_an Cxz_num Cyy_num Cyz_num Czz_num txy_num τ_xy_an N1_num N1_an

    @testset "C_xy ≈ λγ̇ (off-diag in shear plane)" begin
        @test isapprox(Cxy_num, C_xy_an; rtol=0.05)
    end
    @testset "C_xx ≈ 1 + 2(λγ̇)² (first normal stress)" begin
        @test isapprox(Cxx_num, C_xx_an; rtol=0.05)
    end
    @testset "Out-of-plane components stay zero" begin
        @test abs(Cxz_num) < 1e-3
        @test abs(Cyz_num) < 1e-3
    end
    @testset "C_yy = C_zz = 1 (no second normal stress for Oldroyd-B)" begin
        @test isapprox(Cyy_num, 1.0; atol=1e-3)
        @test isapprox(Czz_num, 1.0; atol=1e-3)
    end
    @testset "τ_p_xy = G · C_xy = ν_p · γ̇ (constitutive)" begin
        @test isapprox(txy_num, τ_xy_an; rtol=0.05)
    end
    @testset "N1 = 2 · ν_p · λ · γ̇² (first normal stress diff)" begin
        @test isapprox(N1_num, N1_an; rtol=0.10)
    end
end
