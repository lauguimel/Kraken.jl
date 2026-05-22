#!/usr/bin/env julia
# Phase A triage: NaN spatial fingerprint classification for M34-fix 3 NaN cases.
# Apply [[feedback_nan_uniform_vs_arc_diagnostic]] protocol.

using Serialization
using Printf

const ROOT = "/Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic"
const OUT = joinpath(ROOT, "bench/scratch/m34_fix_diag")
mkpath(OUT)

# Cases to classify
cases = [
    (name="R30_Wi1",   path="tmp/m34_fix_diag/matrix/cyl_bigsweep_v2_beta0p59_wi1_re1_R30_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls",   R=30, Wi=1.0),
    (name="R40_Wi1",   path="tmp/m34_fix_diag/matrix/cyl_bigsweep_v2_beta0p59_wi1_re1_R40_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls",   R=40, Wi=1.0),
    (name="R60_Wi0p1", path="tmp/m34_fix_diag/R60/cyl_bigsweep_v2_beta0p59_wi0p1_re1_R60_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls",    R=60, Wi=0.1),
]

# Also include a clean case for sanity check
clean_cases = [
    (name="R30_Wi0p1_CLEAN", path="tmp/m34_fix_diag/matrix/cyl_bigsweep_v2_beta0p59_wi0p1_re1_R30_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls", R=30, Wi=0.1),
]

function nan_frac(arr)
    nfinite = 0
    nnan = 0
    n = 0
    for v in arr
        n += 1
        if !isfinite(v)
            if isnan(v)
                nnan += 1
            else
                nfinite += 1  # Inf counts separately
            end
        end
    end
    return (nan_count=nnan, nonfinite_count=nnan+nfinite, total=n,
            nan_frac=nnan/n, nonfinite_frac=(nnan+nfinite)/n)
end

# Cylinder geometry per case (from SUMMARY: L_up=15R, L_down=15R, R = param)
# Nx = (L_up + L_down)*R, Ny = something. Find from snapshot directly.
function classify_case(case)
    path = joinpath(ROOT, case.path)
    println("=== $(case.name) ===")
    println("Loading: $path  ($(round(filesize(path)/1e6, digits=1)) MB)")
    snap = deserialize(path)

    println("propertynames(snap): ", propertynames(snap))

    # Try standard field names
    fields_to_check = Symbol[]
    for f in (:psixx, :psixy, :psiyy, :rho, :ux, :uy, :tauxx, :tauxy, :tauyy)
        if hasproperty(snap, f)
            push!(fields_to_check, f)
        end
    end
    println("Fields available: ", fields_to_check)

    # Domain shape
    if hasproperty(snap, :rho)
        Nx, Ny = size(snap.rho)
        println("Domain (Nx, Ny) = ($Nx, $Ny)")
    end

    # Cylinder placement: Nx/2 (centered between Lup and Ldn = 15R+15R), cy=Ny/2
    R = Float64(case.R)
    cx = Float64(div(Nx, 2)) + 0.5
    cy = Float64(div(Ny, 2)) - 0.5
    if hasproperty(snap, :cylinder_x_lbm)
        cx = Float64(snap.cylinder_x_lbm)
    end
    if hasproperty(snap, :cylinder_y_lbm)
        cy = Float64(snap.cylinder_y_lbm)
    end
    println("Cylinder (cx, cy, R) = ($cx, $cy, $R)")

    # NaN fractions per field
    println("\nNaN fractions:")
    for f in fields_to_check
        arr = getproperty(snap, f)
        nf = nan_frac(arr)
        @printf("  %8s: NaN frac = %.4f (%d/%d), nonfinite = %.4f\n",
                f, nf.nan_frac, nf.nan_count, nf.total, nf.nonfinite_frac)
    end

    # Spatial fingerprint of NaN mask (use psi_xx if present else rho)
    ref_field = :psixx in fields_to_check ? :psixx : :rho
    arr = getproperty(snap, ref_field)
    Nxa, Nya = size(arr)

    # Build NaN mask + record positions
    nan_i = Int[]
    nan_j = Int[]
    for j in 1:Nya, i in 1:Nxa
        if !isfinite(arr[i, j])
            push!(nan_i, i)
            push!(nan_j, j)
        end
    end

    nf_total = length(nan_i)
    println("\n$ref_field NaN cells: $nf_total / $(Nxa*Nya) = $(round(100*nf_total/(Nxa*Nya), digits=2))%")

    # Spatial bins for fingerprint
    # is_solid mask: cells strictly inside cylinder
    fluid_count = 0
    fluid_nan = 0
    arc_count = 0       # bilateral arc: θ ∈ ±(30,60), r-R ∈ [0,10]
    pole_count = 0      # near front-pole: |θ-π|<30°, r-R ∈ [0,5]
    wake_count = 0      # |θ|<30°
    far_count = 0

    function position_label(theta_deg, dr)
        if dr < 0
            return "solid"
        elseif dr <= 10.0 && (30.0 <= abs(theta_deg) <= 60.0)
            return "front-shoulder-arc"
        elseif dr <= 5.0 && abs(abs(theta_deg) - 180.0) <= 30.0
            return "front-pole"
        elseif abs(theta_deg) <= 30.0
            return "wake"
        else
            return "other"
        end
    end

    pos_bins = Dict("front-shoulder-arc"=>0, "front-pole"=>0, "wake"=>0, "other"=>0, "solid"=>0)

    for (i, j) in zip(nan_i, nan_j)
        dx = i - cx
        dy = j - cy
        r = hypot(dx, dy)
        dr = r - R
        theta_deg = atan(dy, dx) * 180.0/pi
        label = position_label(theta_deg, dr)
        pos_bins[label] += 1
    end
    println("\nNaN spatial distribution:")
    for (k, v) in sort(collect(pos_bins), by=x->-x[2])
        @printf("  %-25s : %6d (%.2f%%)\n", k, v, 100.0*v/max(nf_total,1))
    end

    # Check if first column / inlet zone has NaN
    inlet_nan = count(j -> !isfinite(arr[1, j]), 1:Nya)
    outlet_nan = count(j -> !isfinite(arr[Nxa, j]), 1:Nya)
    @printf("Edge NaN: inlet col=%d, outlet col=%d (Ny=%d)\n", inlet_nan, outlet_nan, Nya)

    # Compute domain-wide rate (per case classification)
    rho_nan = hasproperty(snap, :rho) ? nan_frac(snap.rho) : nothing
    psi_nan = hasproperty(snap, :psixx) ? nan_frac(snap.psixx) : nothing

    # Decision
    rho_frac = rho_nan === nothing ? 0.0 : rho_nan.nan_frac
    psi_frac = psi_nan === nothing ? 0.0 : psi_nan.nan_frac
    max_frac = max(rho_frac, psi_frac)

    classification = if max_frac >= 0.9
        "uniform"
    elseif max_frac < 0.30 && pos_bins["front-shoulder-arc"] > pos_bins["other"]
        "bilateral-arcs"
    elseif max_frac < 0.30
        "localised-other"
    else
        "mixed"
    end

    println("\n>>> CLASSIFICATION: $classification (max NaN frac = $(round(max_frac, digits=3)))")

    # Save mask coordinates + summary to text
    summary_path = joinpath(OUT, "$(case.name)_summary.txt")
    open(summary_path, "w") do io
        println(io, "case=$(case.name)")
        println(io, "R=$(case.R)")
        println(io, "Wi=$(case.Wi)")
        println(io, "Nx=$Nxa")
        println(io, "Ny=$Nya")
        println(io, "cx=$cx")
        println(io, "cy=$cy")
        println(io, "ref_field=$ref_field")
        println(io, "total_nan=$nf_total")
        println(io, "total_cells=$(Nxa*Nya)")
        println(io, "rho_nan_frac=$rho_frac")
        println(io, "psi_nan_frac=$psi_frac")
        println(io, "max_nan_frac=$max_frac")
        println(io, "classification=$classification")
        for (k, v) in pos_bins
            println(io, "pos_$(k)=$v")
        end
        println(io, "inlet_col_nan=$inlet_nan")
        println(io, "outlet_col_nan=$outlet_nan")
    end
    println("Wrote $summary_path")

    # Save NaN cell positions for plotting
    coords_path = joinpath(OUT, "$(case.name)_nan_cells.csv")
    open(coords_path, "w") do io
        println(io, "i,j,r,theta_deg,dr")
        for (i, j) in zip(nan_i, nan_j)
            dx = i - cx
            dy = j - cy
            r = hypot(dx, dy)
            theta_deg = atan(dy, dx) * 180.0/pi
            @printf(io, "%d,%d,%.4f,%.4f,%.4f\n", i, j, r, theta_deg, r-R)
        end
    end
    println("Wrote $coords_path")

    return (case=case.name, classification=classification, max_nan_frac=max_frac,
            pos_bins=pos_bins, Nx=Nxa, Ny=Nya, nan_total=nf_total,
            inlet_col_nan=inlet_nan, outlet_col_nan=outlet_nan)
end

results = []
for c in cases
    try
        push!(results, classify_case(c))
        println()
    catch e
        @warn "Failed on $(c.name): $e"
    end
end

# Also do sanity on clean R30_Wi0p1
println("=== SANITY: clean case ===")
for c in clean_cases
    try
        push!(results, classify_case(c))
        println()
    catch e
        @warn "Failed on $(c.name): $e"
    end
end

println("=== SUMMARY TABLE ===")
@printf("%-20s %-20s %-10s %-10s %-10s %-15s\n", "case", "classification", "frac", "Nx", "Ny", "nan_total")
for r in results
    @printf("%-20s %-20s %-10.4f %-10d %-10d %-15d\n",
            r.case, r.classification, r.max_nan_frac, r.Nx, r.Ny, r.nan_total)
end
