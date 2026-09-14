using Test
using Kraken
using KernelAbstractions

@testset "Log-FV frozen channel CDE" begin
    for flow in (:couette, :poiseuille)
        result = Kraken.run_viscoelastic_logfv_frozen_channel_cde_2d(;
            Nx=8,
            Ny=16,
            flow,
            height=1.0,
            width=1.0,
            umax=0.02,
            uwall=0.02,
            lambda=2.0,
            prefactor=0.03,
            bsd_fraction=1.0,
            initial=:steady,
            max_steps=1,
            polymer_substeps=128,
            backend=KernelAbstractions.CPU(),
            T=Float64,
        )

        @test result.min_c_eig > 0.8
        @test result.max_velocity_gradient_error < 1.0e-12
        @test result.max_gradient_component_error.dudx < 1.0e-12
        @test result.max_gradient_component_error.dudy < 1.0e-12
        @test result.max_gradient_component_error.dvdx < 1.0e-12
        @test result.max_gradient_component_error.dvdy < 1.0e-12
        @test result.max_c_error < (flow === :poiseuille ? 1.5e-4 : 5.0e-5)
        @test result.max_tau_error < (flow === :poiseuille ? 5.0e-6 : 1.5e-6)
        @test result.max_transverse_force < 1.0e-12
        @test result.max_total_force_error < (flow === :poiseuille ? 8.0e-6 : 1.0e-12)
    end
end

@testset "Log-FV frozen embedded half-plane numerical-gradient shear CDE" begin
    T = Float64
    Nx, Ny = 16, 16
    ywall = T(4.25)
    shear_rate = T(0.012)
    lambda = T(3.0)
    prefactor = T(0.02)
    dt = T(0.01)
    backend = KernelAbstractions.CPU()
    bc = Kraken.FVFDDomainBC2D(;
        west=:periodic, east=:periodic, south=:open, north=:open,
    )
    geometry_h = Kraken.fvfd_geometry_from_halfplane_2d(
        Nx, Ny, one(T), one(T), bc, zero(T), one(T), -ywall; FT=T,
    )
    geometry = Kraken.fvfd_transfer_geometry_2d(geometry_h, backend, T)
    fluid = .!geometry_h.is_solid
    cut = geometry_h.embedded.cut_count .> 0
    cut_neighborhood = copy(cut)
    for idx in findall(cut)
        i, j = Tuple(idx)
        j > 1 && (cut_neighborhood[i, j - 1] = true)
        j < Ny && (cut_neighborhood[i, j + 1] = true)
    end
    far_fluid = fluid .& .!cut_neighborhood

    ux = [
        geometry_h.embedded.cut_count[i, j] > 0 ?
        shear_rate * geometry_h.embedded.wall_distance[i, j] :
        shear_rate * ((T(j) - T(0.5)) - ywall)
        for i in 1:Nx, j in 1:Ny
    ]
    uy = zeros(T, Nx, Ny)
    dudx = zeros(T, Nx, Ny)
    dudy = zeros(T, Nx, Ny)
    dvdx = zeros(T, Nx, Ny)
    dvdy = zeros(T, Nx, Ny)

    Kraken.fvfd_velocity_gradient_embedded_2d!(
        dudx, dudy, dvdx, dvdy, ux, uy, geometry,
    )

    @test count(fluid) > 0
    @test count(geometry_h.embedded.cut_count .> 0) > 0
    @test maximum(abs, dudx[fluid]) < 1.0e-14
    @test maximum(abs.(dudy[cut] .- shear_rate)) < 1.0e-14
    @test maximum(abs.(dudy[far_fluid] .- shear_rate)) < 1.0e-14
    @test maximum(abs.(dudy[fluid] .- shear_rate)) < 1.0e-3
    @test maximum(abs, dvdx[fluid]) < 1.0e-14
    @test maximum(abs, dvdy[fluid]) < 1.0e-14

    cxx = one(T) + T(2) * (lambda * shear_rate)^2
    cxy = lambda * shear_rate
    cyy = one(T)
    psixx0, psixy0, psiyy0 = Kraken.logfv_log_spd_sym2_2d(cxx, cxy, cyy)
    psixx = fill(psixx0, Nx, Ny)
    psixy = fill(psixy0, Nx, Ny)
    psiyy = fill(psiyy0, Nx, Ny)
    psixx_adv = zeros(T, Nx, Ny)
    psixy_adv = zeros(T, Nx, Ny)
    psiyy_adv = zeros(T, Nx, Ny)
    psixx_next = zeros(T, Nx, Ny)
    psixy_next = zeros(T, Nx, Ny)
    psiyy_next = zeros(T, Nx, Ny)
    ux_face = zeros(T, Nx + 1, Ny)
    uy_face = zeros(T, Nx, Ny + 1)
    tauxx = zeros(T, Nx, Ny)
    tauxy = zeros(T, Nx, Ny)
    tauyy = zeros(T, Nx, Ny)
    psixx_bc = Kraken.FVFDFieldBC2D(
        fill(psixx0, Ny), fill(psixx0, Ny), fill(psixx0, Nx), fill(psixx0, Nx),
    )
    psixy_bc = Kraken.FVFDFieldBC2D(
        fill(psixy0, Ny), fill(psixy0, Ny), fill(psixy0, Nx), fill(psixy0, Nx),
    )
    psiyy_bc = Kraken.FVFDFieldBC2D(
        fill(psiyy0, Ny), fill(psiyy0, Ny), fill(psiyy0, Nx), fill(psiyy0, Nx),
    )
    ux_bc = Kraken.FVFDFieldBC2D(zeros(T, Ny), zeros(T, Ny), zeros(T, Nx), zeros(T, Nx))
    uy_bc = Kraken.FVFDFieldBC2D(zeros(T, Ny), zeros(T, Ny), zeros(T, Nx), zeros(T, Nx))

    Kraken.fvfd_sym2_advect_upwind_embedded_2d!(
        psixx_adv, psixy_adv, psiyy_adv,
        psixx, psixy, psiyy,
        psixx_bc, psixy_bc, psiyy_bc,
        ux_face, uy_face, ux, uy, geometry, ux_bc, uy_bc, dt,
    )
    Kraken.logfv_step_constitutive_log_2d!(
        psixx_next, psixy_next, psiyy_next,
        psixx_adv, psixy_adv, psiyy_adv,
        dudx, dudy, dvdx, dvdy,
        lambda, dt, Kraken.LOGFV_MODEL_OLDROYDB, zero(T),
    )
    Kraken.logfv_stress_from_log_2d!(
        tauxx, tauxy, tauyy, psixx_next, psixy_next, psiyy_next, prefactor,
    )

    @test maximum(abs.(psixx_adv[fluid] .- psixx0)) < 1.0e-14
    @test maximum(abs.(psixy_adv[fluid] .- psixy0)) < 1.0e-14
    @test maximum(abs.(psiyy_adv[fluid] .- psiyy0)) < 1.0e-14
    @test maximum(abs.(tauxx[fluid] .- prefactor * (cxx - one(T)))) < 2.0e-8
    @test maximum(abs.(tauxy[fluid] .- prefactor * cxy)) < 2.0e-7
    @test maximum(abs.(tauyy[fluid] .- prefactor * (cyy - one(T)))) < 1.0e-14
end

@testset "Log-FV frozen embedded circle imposed shear CDE" begin
    for polymer_model in (:oldroydb, :fenep)
        result = Kraken.run_viscoelastic_logfv_frozen_circle_shear_cde_2d(;
            Nx=32,
            Ny=32,
            cx=16.0,
            cy=16.0,
            radius=6.0,
            shear_rate=0.012,
            lambda=3.0,
            prefactor=0.02,
            polymer_model,
            L_max=8.0,
            dt=0.01,
            samples=32,
            backend=KernelAbstractions.CPU(),
            T=Float64,
        )

        @test result.fluid_cells > 0
        @test result.cut_cells > 0
        @test result.wall_length > 0
        @test result.min_c_eig > 0.95
        @test result.reference.tauxy != 0
        @test result.max_adv_psi_error < 1.0e-14
        @test result.max_c_error < 2.0e-7
        @test result.max_tau_error < 1.0e-8
    end
end

@testset "Log-FV frozen embedded circle numerical-gradient tangential shear CDE" begin
    result = Kraken.run_viscoelastic_logfv_frozen_circle_tangential_shear_cde_2d(;
        Nx=64,
        Ny=64,
        cx=32.0,
        cy=32.0,
        radius=10.0,
        shear_rate=0.006,
        lambda=2.0,
        prefactor=0.02,
        dt=0.001,
        samples=32,
        backend=KernelAbstractions.CPU(),
        T=Float64,
    )

    @test result.fluid_cells > 0
    @test result.cut_cells > 0
    @test result.min_c_eig > 0.98
    @test result.max_velocity_gradient_error < 3.0e-3
    @test result.max_cut_velocity_gradient_error < 3.0e-3
    @test result.max_bulk_velocity_gradient_error < 2.0e-3
    @test result.max_c_error < 1.0e-8
    @test result.max_tau_error < 1.0e-10
end

@testset "Log-FV frozen embedded circle gradient remains bounded across radii" begin
    cases = (
        (R=6.0, N=38),
        (R=10.0, N=64),
        (R=14.0, N=90),
    )
    errors = Float64[]
    for case in cases
        result = Kraken.run_viscoelastic_logfv_frozen_circle_tangential_shear_cde_2d(;
            Nx=case.N,
            Ny=case.N,
            cx=case.N / 2,
            cy=case.N / 2,
            radius=case.R,
            shear_rate=0.006,
            lambda=2.0,
            prefactor=0.02,
            dt=0.001,
            samples=32,
            backend=KernelAbstractions.CPU(),
            T=Float64,
        )
        push!(errors, result.max_velocity_gradient_error)
        @test result.max_velocity_gradient_error < 6.0e-3
        @test result.max_tau_error < 1.0e-10
        @test result.min_c_eig > 0.97
    end
    @test maximum(errors) < 6.0e-3
end
