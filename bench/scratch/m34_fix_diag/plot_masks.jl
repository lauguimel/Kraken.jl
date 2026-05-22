#!/usr/bin/env julia
# Spatial heatmap of NaN mask + is_solid (PNG via CairoMakie if available, fallback to PPM)

using Serialization
using Printf

ROOT = "/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic"
OUT = joinpath(ROOT, "bench/scratch/m34_fix_diag")
mkpath(OUT)

cases = [
    ("R30_Wi1",   "tmp/m34_fix_diag/matrix/cyl_bigsweep_v2_beta0p59_wi1_re1_R30_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls"),
    ("R40_Wi1",   "tmp/m34_fix_diag/matrix/cyl_bigsweep_v2_beta0p59_wi1_re1_R40_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls"),
    ("R60_Wi0p1", "tmp/m34_fix_diag/R60/cyl_bigsweep_v2_beta0p59_wi0p1_re1_R60_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls"),
]

# Try CairoMakie
have_makie = try
    using CairoMakie
    true
catch
    false
end

for (name, path) in cases
    snap = deserialize(joinpath(ROOT, path))
    Nx, Ny = size(snap.rho)
    # Build 3-level mask: 0 = solid, 1 = fluid finite, 2 = fluid NaN
    mask = zeros(Int, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        if snap.is_solid[i, j]
            mask[i, j] = 0
        elseif isfinite(snap.rho[i, j])
            mask[i, j] = 1
        else
            mask[i, j] = 2
        end
    end

    n_finite_fluid = count(==(1), mask)
    n_nan_fluid = count(==(2), mask)
    n_solid = count(==(0), mask)
    println("$name: solid=$n_solid finite_fluid=$n_finite_fluid nan_fluid=$n_nan_fluid")

    if have_makie
        fig = Figure(size=(1200, 300))
        ax = Axis(fig[1, 1], title="$name — NaN mask (0=solid, 1=fluid finite, 2=fluid NaN)",
                  xlabel="i", ylabel="j", aspect=DataAspect())
        heatmap!(ax, 1:Nx, 1:Ny, mask, colormap=:viridis, colorrange=(0, 2))
        save(joinpath(OUT, "$(name)_nan_mask.png"), fig)
        println("  Wrote $(name)_nan_mask.png")
    else
        # Write a coarse summary as a CSV (sparse: just per-column NaN counts)
        col_nan = zeros(Int, Nx)
        col_solid = zeros(Int, Nx)
        for i in 1:Nx
            col_nan[i] = count(j -> mask[i, j] == 2, 1:Ny)
            col_solid[i] = count(j -> mask[i, j] == 0, 1:Ny)
        end
        # Per-row
        row_nan = zeros(Int, Ny)
        for j in 1:Ny
            row_nan[j] = count(i -> mask[i, j] == 2, 1:Nx)
        end
        open(joinpath(OUT, "$(name)_mask_profile.csv"), "w") do io
            println(io, "axis,index,nan_count,solid_count")
            for i in 1:Nx
                println(io, "col,$i,$(col_nan[i]),$(col_solid[i])")
            end
            for j in 1:Ny
                println(io, "row,$j,$(row_nan[j]),0")
            end
        end
        println("  Wrote $(name)_mask_profile.csv")
    end
end
