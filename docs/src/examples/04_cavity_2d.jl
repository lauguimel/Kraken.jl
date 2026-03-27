# # Lid-Driven Cavity (2D) — Re = 100
#
#
# ## Problem statement
#
# The lid-driven cavity is the canonical benchmark for incompressible flow
# solvers.  A square domain of side ``N`` is bounded by no-slip walls on three
# sides; the top wall (lid) moves at constant velocity ``u_\text{lid}``.  The
# Reynolds number is
#
# ```math
# \text{Re} = \frac{u_\text{lid}\, N}{\nu}
# ```
#
# At ``\text{Re} = 100`` a single primary vortex occupies the cavity.  Reference
# velocity profiles along the centrelines are provided by
# [Ghia *et al.* (1982)](@cite ghia1982high).
#
# ## LBM setup
#
# | Parameter | Value |
# |-----------|-------|
# | Lattice   | D2Q9 |
# | Domain    | ``128 \times 128`` |
# | Lid BC    | Zou--He velocity, ``u_\text{lid} = 0.1`` |
# | Other walls | Half-way bounce-back |
# | ``\nu``   | ``u_\text{lid} \cdot N / \text{Re} = 0.128`` |
#
# ## Simulation

using Kraken
using CairoMakie

N     = 128
Re    = 100
u_lid = 0.1
ν     = u_lid * N / Re

config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=ν, u_lid=u_lid,
                   max_steps=60000, output_interval=10000)
ρ, ux, uy, _ = run_cavity_2d(config)

# ## Reference data — Ghia *et al.* (1982)
#
# Digitised data for ``\text{Re} = 100``: ``u_x`` along the vertical centreline
# and ``u_y`` along the horizontal centreline.

y_ghia  = [0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
           0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
           0.9688, 0.9766, 1.0]
ux_ghia = [0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
          -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
           0.68717, 0.73722, 0.78871, 0.84123, 1.0]

# ## Results
#
# Extract profiles and compare with [Ghia *et al.* (1982)](@cite ghia1982high).

mid = N ÷ 2 + 1

## Vertical centreline: ux(y)
ux_profile = [ux[mid, j] / u_lid for j in 1:N]
y_norm     = [(j - 0.5) / N for j in 1:N]

fig = Figure(size=(900, 420))
ax1 = Axis(fig[1, 1]; xlabel="u_x / u_lid", ylabel="y / N",
           title="Vertical centreline")
lines!(ax1, ux_profile, y_norm; label="LBM (N=$N)", linewidth=2)
scatter!(ax1, ux_ghia, y_ghia; label="Ghia et al.", color=:red, markersize=8)
axislegend(ax1; position=:lb)

## Horizontal centreline: uy(x)
uy_profile = [uy[i, mid] / u_lid for i in 1:N]
x_norm     = [(i - 0.5) / N for i in 1:N]

ax2 = Axis(fig[1, 2]; xlabel="x / N", ylabel="u_y / u_lid",
           title="Horizontal centreline")
lines!(ax2, x_norm, uy_profile; label="LBM (N=$N)", linewidth=2)
axislegend(ax2; position=:rt)
fig
save("cavity_2d_centerlines.svg", fig) #hide

# ## Streamlines
#
# Visualise the primary recirculation vortex.

fig2 = Figure(size=(500, 480))
ax3  = Axis(fig2[1, 1]; title="Velocity magnitude — Re=$Re", aspect=DataAspect())
umag = @. sqrt(ux^2 + uy^2) / u_lid
hm   = heatmap!(ax3, 1:N, 1:N, umag; colormap=:viridis)
Colorbar(fig2[1, 2], hm; label="|u| / u_lid")
fig2
save("cavity_2d_umag.svg", fig2) #hide

# ## References
#
# - [Ghia *et al.* (1982)](@cite ghia1982high) — Reference benchmark data
# - [Zou & He (1997)](@cite zou1997pressure) — Lid boundary condition
# - [He & Luo (1997)](@cite he1997theory) — LBM theory
