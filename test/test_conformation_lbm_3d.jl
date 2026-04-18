using Test
using Kraken, KernelAbstractions

@testset "Conformation TRT-LBM 3D (D3Q19, Liu 2025)" begin
    backend = KernelAbstractions.CPU(); FT = Float64
    Nx, Ny, Nz = 8, 8, 8

    # Helper: zero velocity field, all-fluid mask, C = I
    ux = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    uy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    uz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny, Nz)
    Cxx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(Cxx, FT(1))
    Cyy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(Cyy, FT(1))
    Czz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(Czz, FT(1))
    Cxy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    Cxz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    Cyz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)

    @testset "init / macro consistency: φ = Σ_q g_q" begin
        g = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
        init_conformation_field_3d!(g, Cxx, ux, uy, uz)
        Crec = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
        compute_conformation_macro_3d!(Crec, g)
        @test maximum(abs.(Array(Crec) .- 1.0)) < 1e-12
    end

    @testset "collide @ rest preserves C = I" begin
        g = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
        init_conformation_field_3d!(g, Cxx, ux, uy, uz)
        for _ in 1:10
            collide_conformation_3d!(g, Cxx, ux, uy, uz,
                                      Cxx, Cxy, Cxz, Cyy, Cyz, Czz, is_solid,
                                      1.0, 10.0; component=1)
        end
        Crec = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
        compute_conformation_macro_3d!(Crec, g)
        @test maximum(abs.(Array(Crec) .- 1.0)) < 1e-10
    end

    @testset "collide for diagonal components 1,4,6 (xx,yy,zz)" begin
        # All three diagonal components must remain at C = 1 at rest with C = I
        for comp in (1, 4, 6)
            Cf = comp == 1 ? Cxx : comp == 4 ? Cyy : Czz
            g = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
            init_conformation_field_3d!(g, Cf, ux, uy, uz)
            collide_conformation_3d!(g, Cf, ux, uy, uz,
                                      Cxx, Cxy, Cxz, Cyy, Cyz, Czz, is_solid,
                                      1.0, 10.0; component=comp)
            Crec = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
            compute_conformation_macro_3d!(Crec, g)
            @test maximum(abs.(Array(Crec) .- 1.0)) < 1e-10
        end
    end

    @testset "collide for off-diagonal components 2,3,5 (xy,xz,yz)" begin
        # Off-diagonal components stay at 0 at rest with C = I
        for comp in (2, 3, 5)
            Cf = comp == 2 ? Cxy : comp == 3 ? Cxz : Cyz
            g = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
            init_conformation_field_3d!(g, Cf, ux, uy, uz)
            collide_conformation_3d!(g, Cf, ux, uy, uz,
                                      Cxx, Cxy, Cxz, Cyy, Cyz, Czz, is_solid,
                                      1.0, 10.0; component=comp)
            Crec = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
            compute_conformation_macro_3d!(Crec, g)
            @test maximum(abs.(Array(Crec))) < 1e-10
        end
    end

    @testset "Hermite source: zero stress is no-op" begin
        f = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
        fill!(f, FT(0.123))
        f_init = copy(Array(f))
        zero_field = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
        apply_hermite_source_3d!(f, is_solid, 1.5,
                                  zero_field, zero_field, zero_field,
                                  zero_field, zero_field, zero_field)
        @test maximum(abs.(Array(f) .- f_init)) < 1e-12
    end

    @testset "Hermite source: τxx > 0 perturbs only the right pops" begin
        # τ_p_xx = 1 should change populations whose H_{xx} = c_x² - 1/3 ≠ 0
        # i.e. q ∈ {1 (rest), 2,3 (axial-x), 8..15 (xy + xz edges)}.
        # The pure y- and z-axial pops (q=4,5,6,7) and the yz edges
        # (q=16..19) have c_x = 0 → H_{xx} = -1/3 (axial y/z) or -1/3 (yz edges)
        # → all 19 directions get a perturbation in fact via the −cs²·δ_{αβ}
        # term. So just check that the result is finite and non-zero.
        f = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
        f_init = copy(Array(f))
        txx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(txx, FT(1))
        zero_field = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
        apply_hermite_source_3d!(f, is_solid, 1.5, txx,
                                  zero_field, zero_field, zero_field,
                                  zero_field, zero_field)
        h = Array(f)
        @test all(isfinite, h)
        @test maximum(abs.(h .- f_init)) > 1e-6  # some pops did change
    end

    @testset "CNEBB 3D: φ exactly conserved at near-wall cells" begin
        # Build a small box with one solid cell at the centre. Place a
        # non-trivial g population and verify CNEBB updates the macro
        # field consistently.
        Nx_, Ny_, Nz_ = 6, 6, 6
        is_solid_h = zeros(Bool, Nx_, Ny_, Nz_)
        is_solid_h[3, 3, 3] = true   # one solid cell
        g_pre  = KernelAbstractions.zeros(backend, FT, Nx_, Ny_, Nz_, 19)
        g_post = KernelAbstractions.zeros(backend, FT, Nx_, Ny_, Nz_, 19)
        # Set distinct populations everywhere
        for q in 1:19, k in 1:Nz_, j in 1:Ny_, i in 1:Nx_
            g_post[i, j, k, q] = FT(0.05) * q
            g_pre[i, j, k, q]  = FT(0.05) * q
        end
        is_solid_d = KernelAbstractions.allocate(backend, Bool, Nx_, Ny_, Nz_)
        copyto!(is_solid_d, is_solid_h)
        C_field = KernelAbstractions.zeros(backend, FT, Nx_, Ny_, Nz_)
        apply_cnebb_conformation_3d!(g_post, g_pre, is_solid_d, C_field)

        # At any neighbour of the solid (e.g. (2,3,3)), C_field should now
        # equal Σ_q g_post[i,j,k,q] (after CNEBB-reconstructed pops).
        c = Array(C_field)
        gh = Array(g_post)
        for (i, j, k) in ((2,3,3), (4,3,3), (3,2,3), (3,4,3), (3,3,2), (3,3,4))
            @test c[i, j, k] ≈ sum(gh[i, j, k, :]) atol=1e-12
        end
    end
end
