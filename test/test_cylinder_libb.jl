using Test
using Kraken

# ==========================================================================
# Schäfer-Turek-style cylinder with LI-BB V2 (Bouzidi pre-phase + TRT).
#
# Physics sanity on uniform inlet Re=20:
#  - flow symmetric about the cylinder axis (|Cl/Cd| small),
#  - drag positive and in a reasonable range vs the BGK baseline,
#  - density stays bounded (Mach-dependent compressibility error).
#
# NOTE: The MEA drag evaluation in `compute_drag_libb_2d` uses the
# classical halfway-BB formula F = sum 2·c_q·f_q(post-collision).
# Strict match with the Schäfer-Turek 2D-1 reference Cd = 5.58 requires
# the full Mei-Luo-Shyy 2002 (J. Comput. Phys. 161, 680) formula which
# handles the Bouzidi interpolation on each cut link separately. That
# refinement is future work — the test here verifies the scaffold is
# physically sensible, not bit-accurate.
# ==========================================================================

@testset "Cylinder LI-BB V2 — Schäfer-Turek scaffold" begin
    Nx, Ny = 300, 80
    radius = 10
    u_in = 0.04
    Re = 20.0
    D = 2 * radius
    ν = u_in * D / Re

    result = run_cylinder_libb_2d(; Nx=Nx, Ny=Ny, radius=radius,
                                    cx=Nx÷4, cy=Ny÷2,
                                    u_in=u_in, ν=ν, inlet=:uniform,
                                    max_steps=20_000, avg_window=5_000)

    @test !any(isnan, result.ρ)
    @test !any(isnan, result.ux)
    # Drag positive and bounded. Mei-Luo-Shyy MEA refinement should
    # tighten this; for now ensure order of magnitude is sensible.
    @test result.Cd > 1.0
    @test result.Cd < 10.0
    # Flow symmetric about the cylinder axis (small lift ratio).
    @test abs(result.Fy / result.Fx) < 0.01
    # Density bounded within 15 % of 1 (uniform inlet has known
    # compressibility error that V2 + TRT Λ=3/16 minimises, but some
    # error remains at Ma ~ 0.04/c_s ≈ 0.07).
    @test maximum(result.ρ) < 1.2
    @test minimum(result.ρ) > 0.8

    @info "Cylinder LI-BB Re=20 (scaffold)" Cd=result.Cd lift_ratio=abs(result.Fy/result.Fx) maxρm1=maximum(abs.(result.ρ .- 1))
end
