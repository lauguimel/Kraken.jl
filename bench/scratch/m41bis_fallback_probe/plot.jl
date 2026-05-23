# M41-bis: spatial heatmap of |tau_xx| with fallback-zone overlay.
using Serialization, CairoMakie, Statistics

const OUTDIR = @__DIR__
const ARR = joinpath(OUTDIR, "probe_arrays.jls")
data = deserialize(ARR)

tauxx = data.tauxx
is_solid = data.is_solid
near_solid = data.near_solid
Nx, Ny = data.Nx, data.Ny
cx = data.cylinder_x_lbm
cy = data.cylinder_y_lbm
R = data.radius_lbm

# Zoom around the cylinder ±4R
xlo = max(1, Int(floor(cx - 4R)))
xhi = min(Nx, Int(ceil(cx + 4R)))
ylo = 1
yhi = Ny

xs = xlo:xhi
ys = ylo:yhi

field = abs.(tauxx[xs, ys])
maxf = quantile(vec(field), 0.999)

fig = Figure(size=(1400, 500))
ax = Axis(fig[1,1], aspect=DataAspect(),
    xlabel="x [LU]", ylabel="y [LU]",
    title="|tau_xx| (M29b R=30 Wi=1 beta=0.59, Cd_kraken=$(round(data.Cd_kraken,digits=2)))  — red = MUSCL-fallback band (cross stencil)")

hm = heatmap!(ax, collect(xs), collect(ys), field, colormap=:viridis,
    colorrange=(0, maxf))
Colorbar(fig[1,2], hm, label="|tau_xx|")

# Overlay solid (gray)
solid_local = is_solid[xs, ys]
solid_xy = Tuple{Float64,Float64}[]
for (jj, j) in enumerate(ys), (ii, i) in enumerate(xs)
    if solid_local[ii, jj]
        push!(solid_xy, (Float64(i), Float64(j)))
    end
end
if !isempty(solid_xy)
    scatter!(ax, [p[1] for p in solid_xy], [p[2] for p in solid_xy], color=:gray, markersize=3)
end

# Overlay near_solid (red outline)
near_local = near_solid[xs, ys]
near_xy = Tuple{Float64,Float64}[]
for (jj, j) in enumerate(ys), (ii, i) in enumerate(xs)
    if near_local[ii, jj]
        push!(near_xy, (Float64(i), Float64(j)))
    end
end
if !isempty(near_xy)
    scatter!(ax, [p[1] for p in near_xy], [p[2] for p in near_xy],
        color=(:red, 0.7), markersize=4, marker=:rect)
end

png = joinpath(OUTDIR, "tau_xx_with_fallback_overlay.png")
save(png, fig)
println("Wrote ", png)

# A second plot: |tr(tau_p)| with same overlay
trtau = abs.(data.tauxx[xs, ys] .+ data.tauyy[xs, ys])
maxt = quantile(vec(trtau), 0.999)
fig2 = Figure(size=(1400, 500))
ax2 = Axis(fig2[1,1], aspect=DataAspect(),
    xlabel="x [LU]", ylabel="y [LU]",
    title="|tr(tau_p)| (M29b R=30 Wi=1 beta=0.59) — red = MUSCL-fallback band")
hm2 = heatmap!(ax2, collect(xs), collect(ys), trtau, colormap=:viridis,
    colorrange=(0, maxt))
Colorbar(fig2[1,2], hm2, label="|tr(tau_p)|")
if !isempty(solid_xy)
    scatter!(ax2, [p[1] for p in solid_xy], [p[2] for p in solid_xy], color=:gray, markersize=3)
end
if !isempty(near_xy)
    scatter!(ax2, [p[1] for p in near_xy], [p[2] for p in near_xy],
        color=(:red, 0.7), markersize=4, marker=:rect)
end
png2 = joinpath(OUTDIR, "tr_tau_with_fallback_overlay.png")
save(png2, fig2)
println("Wrote ", png2)

println("Done.")
