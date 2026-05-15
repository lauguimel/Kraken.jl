const KRAKEN_E_S4_ATOL_TELESCOPE = 1e-14
const KRAKEN_E_S4_RTOL_CONSERVATION = 1e-12

@testset "Kraken-E S4 D5" begin
    patch = Kraken.build_two_block_patch_2d(
        Float64; Nx_c=8, Ny_c=8, dx_c=1.0, Nx_f=16, Ny_f=16,
    )
    U_c = zeros(8, 8)
    U_f = zeros(16, 16)

    sigma = 1.5
    xc = 4.0
    yc = 4.0
    for j in 1:8, i in 1:8
        x = (i - 0.5) * patch.coarse.dx
        y = (j - 0.5) * patch.coarse.dx
        U_c[i, j] = exp(-((x - xc)^2 + (y - yc)^2) / (2 * sigma^2))
    end
    for j in 1:16, i in 1:16
        x = 8.0 + (i - 0.5) * patch.fine.dx
        y = (j - 0.5) * patch.fine.dx
        U_f[i, j] = exp(-((x - xc)^2 + (y - yc)^2) / (2 * sigma^2))
    end

    flux_c = Kraken.allocate_scalar_flux_field_2d(Float64; Nx=8, Ny=8)
    flux_f = Kraken.allocate_scalar_flux_field_2d(Float64; Nx=16, Ny=16)

    vx = 0.1
    vy = 0.0
    dt = 0.5 * patch.fine.dx / abs(vx)

    M0 = Kraken.patch_total_mass(patch, U_c, U_f)

    max_telescope_err = 0.0
    max_mass_drift = 0.0
    for _ in 1:100
        step_err = Kraken.explicit_euler_step!(
            patch, U_c, U_f, flux_c, flux_f, vx, vy, dt,
        )
        max_telescope_err = max(max_telescope_err, step_err)
        M = Kraken.patch_total_mass(patch, U_c, U_f)
        drift = abs(M - M0) / abs(M0)
        max_mass_drift = max(max_mass_drift, drift)
    end

    @test max_telescope_err <= KRAKEN_E_S4_ATOL_TELESCOPE
    @test max_mass_drift <= KRAKEN_E_S4_RTOL_CONSERVATION

    rec = patch.cf_records[1]
    F1 = 0.123456
    F2 = 0.987654
    err_synth = Kraken.cf_flux_telescoping_error(rec, F1, F2)
    @test err_synth <= 4 * eps(Float64) * max(
        abs(F1 * rec.fine_areas[1]), abs(F2 * rec.fine_areas[2]),
    )

    println()
    println("# Kraken-E S4 canary metrics")
    println("D5 max telescoping err (100 steps, Ny_c faces) = $(max_telescope_err)")
    println("D5 max conservation drift (100 steps)          = $(max_mass_drift)")
    println("D5 synthetic face ulp identity                  = $(err_synth)")
end
