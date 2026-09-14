using Test
using Kraken

@testset "Oldroyd-B homogeneous planar elongation" begin
    Nx, Ny = 16, 16
    λ = 10.0
    ε = 0.005
    tau_plus = 1.0
    is_solid = falses(Nx, Ny)

    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    x0 = (Nx - 1) / 2
    y0 = (Ny - 1) / 2
    for j in 1:Ny, i in 1:Nx
        ux[i, j] = ε * ((i - 1) - x0)
        uy[i, j] = -ε * ((j - 1) - y0)
    end

    C_xx = ones(Float64, Nx, Ny)
    C_xy = zeros(Float64, Nx, Ny)
    C_yy = ones(Float64, Nx, Ny)
    g_xx = zeros(Float64, Nx, Ny, 9)
    g_xy = zeros(Float64, Nx, Ny, 9)
    g_yy = zeros(Float64, Nx, Ny, 9)
    init_conformation_field_2d!(g_xx, C_xx, ux, uy)
    init_conformation_field_2d!(g_xy, C_xy, ux, uy)
    init_conformation_field_2d!(g_yy, C_yy, ux, uy)

    for _ in 1:2_000
        compute_conformation_macro_2d!(C_xx, g_xx)
        compute_conformation_macro_2d!(C_xy, g_xy)
        compute_conformation_macro_2d!(C_yy, g_yy)
        collide_conformation_2d!(g_xx, C_xx, ux, uy, C_xx, C_xy, C_yy,
                                  is_solid, tau_plus, λ; component=1)
        collide_conformation_2d!(g_xy, C_xy, ux, uy, C_xx, C_xy, C_yy,
                                  is_solid, tau_plus, λ; component=2)
        collide_conformation_2d!(g_yy, C_yy, ux, uy, C_xx, C_xy, C_yy,
                                  is_solid, tau_plus, λ; component=3)
    end
    compute_conformation_macro_2d!(C_xx, g_xx)
    compute_conformation_macro_2d!(C_xy, g_xy)
    compute_conformation_macro_2d!(C_yy, g_yy)

    ic, jc = Nx ÷ 2, Ny ÷ 2
    Wi = λ * ε
    Cxx_an = 1 / (1 - 2Wi)
    Cyy_an = 1 / (1 + 2Wi)

    @info "Planar elongation" Wi Cxx_num=C_xx[ic,jc] Cxx_an Cyy_num=C_yy[ic,jc] Cyy_an Cxy_num=C_xy[ic,jc]

    @test isapprox(C_xx[ic, jc], Cxx_an; rtol=0.01)
    @test isapprox(C_yy[ic, jc], Cyy_an; rtol=0.01)
    @test abs(C_xy[ic, jc]) < 1e-10
end
