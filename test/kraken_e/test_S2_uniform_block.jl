using Test
using Kraken

# Kraken-E S2 uniform-block canaries.
#
# Poiseuille is driven by a constant Guo body force in x. The S2 Kraken-E
# pipeline uses the force-corrected velocity in both macroscopic readout and
# BGK equilibrium construction, then applies the standard Guo source term.
@testset "Kraken-E S2 uniform leaf block" begin
    Nx = 32
    Ny = 32
    τ = 0.8
    ω = 1 / τ
    ν = (τ - 0.5) / 3

    @test isdefined(Kraken, :kraken_e_exchange_halo!)
    block_meta = allocate_leaf_block_2d(Float64; Nx=Nx, Ny=Ny)
    @test block_meta.ng == 1
    @test size(block_meta.f) == (Nx + 2, Ny + 2, 9)
    @test block_meta.parent_id == -1
    @test isempty(block_meta.child_ids)
    @test block_meta.same_level_neighbor_ids == (-1, -1, -1, -1)
    @test isempty(block_meta.cf_face_records)
    @test isempty(block_meta.reflux_accumulators)
    @test isempty(block_meta.epoch_remap_buffers)
    @test all(block_meta.cell_kind[2:(Nx + 1), 2:(Ny + 1)] .== KRAKEN_E_INTERIOR)

    eq = allocate_leaf_block_2d(Float64; Nx=Nx, Ny=Ny)
    kraken_e_initialize_equilibrium_2d!(eq; ρ0=1.0)
    for _ in 1:10
        kraken_e_step!(eq, ω; bc=:none, exchange=:periodic_xy)
    end
    max_delta = 0.0
    for j in 2:(Ny + 1), i in 2:(Nx + 1), q in 1:9
        max_delta = max(max_delta, abs(eq.f[i,j,q] - Kraken.kraken_e_feq(q, 1.0, 0.0, 0.0)))
    end
    @test max_delta <= 1e-12

    pois = allocate_leaf_block_2d(Float64; Nx=Nx, Ny=Ny)
    g = 0.04 / 1280
    kraken_e_initialize_equilibrium_2d!(pois; ρ0=1.0)
    for _ in 1:12000
        kraken_e_step!(pois, ω; bc=:poiseuille, exchange=:periodic_x, Fx=g)
    end
    kraken_e_compute_macroscopic_2d!(pois; Fx=g)
    pois_num = kraken_e_mean_ux_by_y(pois)
    pois_ref = kraken_e_poiseuille_reference(pois, g, ν)
    pois_l2 = kraken_e_l2_over_scale(pois_num, pois_ref, maximum(abs.(pois_ref)))
    @test pois_l2 <= 0.01

    coup = allocate_leaf_block_2d(Float64; Nx=Nx, Ny=Ny)
    U = 0.05
    kraken_e_initialize_equilibrium_2d!(coup; ρ0=1.0)
    for _ in 1:12000
        kraken_e_step!(coup, ω; bc=:couette, exchange=:periodic_x, u_top=U)
    end
    kraken_e_compute_macroscopic_2d!(coup)
    coup_num = kraken_e_mean_ux_by_y(coup)
    coup_ref = kraken_e_couette_reference(coup, U)
    coup_l2 = kraken_e_l2_over_scale(coup_num, coup_ref, U)
    @test coup_l2 <= 0.01

    tg = allocate_leaf_block_2d(Float64; Nx=Nx, Ny=Ny)
    kraken_e_initialize_taylor_green_2d!(tg; ρ0=1.0, U0=0.04)
    kraken_e_compute_macroscopic_2d!(tg)
    mass0 = kraken_e_mass(tg)
    energies = zeros(Float64, 200)
    for n in 1:200
        kraken_e_step!(tg, ω; bc=:none, exchange=:periodic_xy)
        kraken_e_compute_macroscopic_2d!(tg)
        energies[n] = kraken_e_kinetic_energy(tg)
    end
    times = collect(1.0:200.0)
    loge = log.(energies)
    tmean = sum(times) / length(times)
    emean = sum(loge) / length(loge)
    slope_fit = sum((times .- tmean) .* (loge .- emean)) / sum((times .- tmean) .^ 2)
    k = 2π / Nx
    slope_exact = -4 * ν * k^2
    tg_slope_err = abs(slope_fit - slope_exact) / abs(slope_exact)
    mass_drift = abs(kraken_e_mass(tg) - mass0) / (Nx * Ny)
    @test tg_slope_err <= 0.01
    @test mass_drift <= 1e-12

    println()
    println("# Kraken-E S2 canary metrics")
    println("equilibrium-fixed: max|Δf| = $(max_delta)")
    println("T1 Poiseuille L2 = $(pois_l2)")
    println("T2 Couette L2 = $(coup_l2)")
    println("T3 TG decay-slope err = $(tg_slope_err)")
    println("T3 mass drift = $(mass_drift)")
end
