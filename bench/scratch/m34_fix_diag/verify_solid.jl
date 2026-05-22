#!/usr/bin/env julia
# Verify that the 2.61% non-NaN cells are exactly the solid (cylinder interior).

using Serialization

paths = [
    "tmp/m34_fix_diag/matrix/cyl_bigsweep_v2_beta0p59_wi1_re1_R30_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls",
    "tmp/m34_fix_diag/matrix/cyl_bigsweep_v2_beta0p59_wi1_re1_R40_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls",
    "tmp/m34_fix_diag/R60/cyl_bigsweep_v2_beta0p59_wi0p1_re1_R60_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls",
]

ROOT = "/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic"

for p in paths
    snap = deserialize(joinpath(ROOT, p))
    Nx, Ny = size(snap.rho)
    n_solid = count(snap.is_solid)
    n_finite = count(isfinite, snap.rho)
    n_nan = count(isnan, snap.rho)
    # In fluid cells: how many NaN, how many finite
    n_fluid = count(!, snap.is_solid)
    n_finite_in_fluid = 0
    n_nan_in_fluid = 0
    n_finite_in_solid = 0
    n_nan_in_solid = 0
    for j in 1:Ny, i in 1:Nx
        if snap.is_solid[i, j]
            isfinite(snap.rho[i, j]) ? (n_finite_in_solid += 1) : (n_nan_in_solid += 1)
        else
            isfinite(snap.rho[i, j]) ? (n_finite_in_fluid += 1) : (n_nan_in_fluid += 1)
        end
    end
    case_name = occursin("R60", p) ? "R60_Wi01" : (occursin("R40", p) ? "R40_Wi1" : "R30_Wi1")
    println("== $case_name ==")
    println("  Nx,Ny = $Nx,$Ny  total=$(Nx*Ny)")
    println("  is_solid count = $n_solid  fluid = $n_fluid")
    println("  fluid: finite=$n_finite_in_fluid  NaN=$n_nan_in_fluid  → fluid NaN frac = $(round(n_nan_in_fluid/n_fluid * 100, digits=2))%")
    println("  solid: finite=$n_finite_in_solid  NaN=$n_nan_in_solid  → solid NaN frac = $(round(n_nan_in_solid/max(n_solid,1) * 100, digits=2))%")

    # Find ANY non-NaN fluid cell location to see if there's a non-trivial pattern
    finite_fluid_positions = Tuple{Int,Int}[]
    for j in 1:Ny, i in 1:Nx
        if !snap.is_solid[i, j] && isfinite(snap.rho[i, j])
            push!(finite_fluid_positions, (i, j))
            length(finite_fluid_positions) > 20 && break
        end
    end
    if !isempty(finite_fluid_positions)
        println("  Sample finite fluid cells (up to 20): $(finite_fluid_positions)")
    else
        println("  No finite fluid cells — every fluid cell is NaN.")
    end
    println()
end
