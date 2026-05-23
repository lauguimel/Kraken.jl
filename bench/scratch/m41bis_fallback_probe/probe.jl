# M41-bis: empirical probe of M29b ±2-cell fallback zone as locus of Cd deficit.
# Reads the M29b R=30 Wi=1 dump, builds the EXACT fallback mask used in
# src/fvfd/operators_2d.jl (cross-shape ±2 LU along axes, NOT 5x5 box),
# and compares polymer stress stats in fallback zone vs bulk.

using Serialization
using Statistics
using Printf

const REPO = joinpath(@__DIR__, "..", "..", "..")
const DUMP = joinpath(REPO, "tmp", "m29b_kraken",
    "cyl_bigsweep_v2_beta0p59_wi1_re1_R30_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls")
const OUTDIR = joinpath(REPO, "bench", "scratch", "m41bis_fallback_probe")
mkpath(OUTDIR)

println("Loading: ", DUMP)
snap = deserialize(DUMP)

Nx, Ny = snap.Nx, snap.Ny
is_solid = snap.is_solid
tauxx = snap.tauxx
tauxy = snap.tauxy
tauyy = snap.tauyy

@assert size(is_solid) == (Nx, Ny)

# === Fallback mask, mirroring operators_2d.jl L523-527 exactly =================
# The MUSCL kernel reverts to Rusanov at (i,j) if ANY of:
#   i <= 2 || i >= Nx-1 || j <= 2 || j >= Ny-1
#   is_solid[i-2,j] | is_solid[i-1,j] | is_solid[i+1,j] | is_solid[i+2,j]
#   is_solid[i,j-2] | is_solid[i,j-1] | is_solid[i,j+1] | is_solid[i,j+2]
# This is a CROSS pattern with arms of length 2 along x and y axes
# (8 stencil neighbours, NOT the 24 of a 5x5 box; corners are NOT in stencil).

function build_fallback_mask(is_solid::AbstractMatrix{Bool})
    Nx, Ny = size(is_solid)
    near = falses(Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]; continue; end
        # domain edge near (would also fall back, but it's not "near solid")
        # We restrict near_solid to actual solid-adjacent cells only
        # for the spatial discriminator. The 4-edge domain band is reported
        # separately as a sanity counter.
        # Solid-cross-arm test (only safe indices)
        hit = false
        for d in (-2, -1, 1, 2)
            ii = i + d
            if 1 <= ii <= Nx && is_solid[ii, j]; hit = true; break; end
        end
        if !hit
            for d in (-2, -1, 1, 2)
                jj = j + d
                if 1 <= jj <= Ny && is_solid[i, jj]; hit = true; break; end
            end
        end
        near[i, j] = hit
    end
    return near
end

# Also a 4-edge-band mask for completeness (matches first line of fallback test)
function build_domain_edge_mask(Nx, Ny, is_solid)
    edge = falses(Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]; continue; end
        if i <= 2 || i >= Nx - 1 || j <= 2 || j >= Ny - 1
            edge[i, j] = true
        end
    end
    return edge
end

near_solid = build_fallback_mask(is_solid)
domain_edge = build_domain_edge_mask(Nx, Ny, is_solid)

bulk = .!(is_solid .| near_solid .| domain_edge)

n_solid = sum(is_solid)
n_near = sum(near_solid)
n_edge = sum(domain_edge)
# Cells that fall back to Rusanov because of EITHER near_solid OR domain_edge
n_fallback_total = sum(near_solid .| domain_edge)
n_bulk = sum(bulk)
n_total = Nx * Ny

@printf("Cell counts (Nx=%d, Ny=%d, total=%d):\n", Nx, Ny, n_total)
@printf("  solid             = %7d (%.4f%%)\n", n_solid, 100*n_solid/n_total)
@printf("  near_solid (cross)= %7d (%.4f%% of fluid)\n", n_near, 100*n_near/(n_total - n_solid))
@printf("  domain_edge_2     = %7d (%.4f%% of fluid)\n", n_edge, 100*n_edge/(n_total - n_solid))
@printf("  fallback_total    = %7d (%.4f%% of fluid)\n", n_fallback_total, 100*n_fallback_total/(n_total - n_solid))
@printf("  bulk (MUSCL active)= %7d (%.4f%% of fluid)\n", n_bulk, 100*n_bulk/(n_total - n_solid))

# === Compute tr(tau_p), |tauxx|, |tauxy|, |tauyy| stats per zone ==============
trtau = tauxx .+ tauyy

# We compare the SOLID-ADJACENCY band (near_solid) against the bulk
# (MUSCL-active fluid, excluding domain-edge band so we don't conflate
# inlet/outlet effects with the curved BC).

function zone_stats(field, mask)
    vals = abs.(field[mask])
    if isempty(vals)
        return (n=0, max=NaN, mean=NaN, q95=NaN, q99=NaN, median=NaN)
    end
    return (
        n      = length(vals),
        max    = maximum(vals),
        mean   = mean(vals),
        q95    = quantile(vals, 0.95),
        q99    = quantile(vals, 0.99),
        median = median(vals),
    )
end

fields = Dict(
    "tauxx" => tauxx,
    "tauxy" => tauxy,
    "tauyy" => tauyy,
    "tr_tau" => trtau,
)

println("\n=== Stats per zone (|field|; near_solid = MUSCL fallback band only) ===")
println(rpad("field", 10), " ", rpad("zone", 12),
        rpad("n", 10), rpad("max", 16), rpad("mean", 16),
        rpad("q95", 16), rpad("q99", 16), rpad("median", 16))
results = Dict{String,Any}()
for (name, f) in fields
    s_near = zone_stats(f, near_solid)
    s_bulk = zone_stats(f, bulk)
    s_edge = zone_stats(f, domain_edge)
    results[name] = (near=s_near, bulk=s_bulk, edge=s_edge)
    for (zname, s) in (("near_solid", s_near), ("bulk", s_bulk), ("domain_edge", s_edge))
        println(rpad(name, 10), " ", rpad(zname, 12),
            rpad(string(s.n), 10),
            rpad(@sprintf("%.4e", s.max), 16),
            rpad(@sprintf("%.4e", s.mean), 16),
            rpad(@sprintf("%.4e", s.q95), 16),
            rpad(@sprintf("%.4e", s.q99), 16),
            rpad(@sprintf("%.4e", s.median), 16))
    end
end

println("\n=== Ratios near_solid / bulk (load-bearing for verdict) ===")
println(rpad("field", 10), rpad("max_ratio", 16), rpad("mean_ratio", 16),
        rpad("q95_ratio", 16), rpad("q99_ratio", 16))
ratios = Dict{String,NamedTuple}()
for name in keys(fields)
    n = results[name].near
    b = results[name].bulk
    r = (max=n.max/b.max, mean=n.mean/b.mean, q95=n.q95/b.q95, q99=n.q99/b.q99)
    ratios[name] = r
    println(rpad(name, 10),
        rpad(@sprintf("%.3f", r.max), 16),
        rpad(@sprintf("%.3f", r.mean), 16),
        rpad(@sprintf("%.3f", r.q95), 16),
        rpad(@sprintf("%.3f", r.q99), 16))
end

# === Save numeric summary to file (no plot dep here; plot in separate script) =
summary_path = joinpath(OUTDIR, "stats_summary.txt")
open(summary_path, "w") do io
    println(io, "M41-bis fallback-zone polymer-stress probe")
    println(io, "Dump: tmp/m29b_kraken/cyl_bigsweep_v2_beta0p59_wi1_re1_R30_bsd1_Lup15_Ldn15_eg0_ea0_ef0_ed0_geomqwall_fields.jls")
    println(io, "advection_scheme = muscl_superbee, Cd_kraken = $(snap.Cd_kraken), Wi=$(snap.Wi), R=$(snap.R), beta=$(snap.beta)")
    println(io, "Cd_s = $(snap.Cd_s), Cd_p = $(snap.Cd_p), Cd_bsd = $(snap.Cd_bsd)")
    println(io)
    println(io, "Cell counts (Nx=$Nx, Ny=$Ny, total=$n_total):")
    println(io, "  solid              = $n_solid")
    println(io, "  near_solid (cross) = $n_near")
    println(io, "  domain_edge_2      = $n_edge")
    println(io, "  fallback_total     = $n_fallback_total")
    println(io, "  bulk (MUSCL active)= $n_bulk")
    println(io)
    println(io, "Stats per zone (|field|):")
    for (name, _) in fields
        for (zname, s) in (("near_solid", results[name].near), ("bulk", results[name].bulk), ("domain_edge", results[name].edge))
            println(io, "  $(rpad(name,8)) $(rpad(zname,12)) n=$(s.n)  max=$(s.max)  mean=$(s.mean)  q95=$(s.q95)  q99=$(s.q99)  median=$(s.median)")
        end
    end
    println(io)
    println(io, "Ratios near_solid / bulk:")
    for (name, r) in ratios
        println(io, "  $(rpad(name,8))  max=$(round(r.max,digits=3))  mean=$(round(r.mean,digits=3))  q95=$(round(r.q95,digits=3))  q99=$(round(r.q99,digits=3))")
    end
end
println("\nWrote ", summary_path)

# === Save mask + tauxx to a small jls for the plot script =====================
out_jls = joinpath(OUTDIR, "probe_arrays.jls")
serialize(out_jls, (;
    tauxx, tauxy, tauyy, is_solid, near_solid, domain_edge, bulk,
    Nx, Ny, dx=snap.dx, dy=snap.dy,
    cylinder_x_lbm=snap.cylinder_x_lbm,
    cylinder_y_lbm=snap.cylinder_y_lbm,
    radius_lbm=snap.radius_lbm,
    Cd_kraken=snap.Cd_kraken,
    Cd_s=snap.Cd_s, Cd_p=snap.Cd_p, Cd_bsd=snap.Cd_bsd,
    Wi=snap.Wi, R=snap.R, beta=snap.beta,
    advection_scheme=snap.advection_scheme,
    results, ratios,
))
println("Wrote ", out_jls)

println("\nDone.")
