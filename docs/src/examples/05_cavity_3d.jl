# # Lid-Driven Cavity (3D) — Re = 100
#
# ![](cavity_3d_midplane.svg)
#
# ## Problem statement
#
# This example extends the classic lid-driven cavity benchmark to three
# dimensions using the D3Q19 lattice.  The lid moves in the ``x``-direction at
# the top face (``y = N_y``), while all other faces are no-slip walls.  The
# Reynolds number is defined as
#
# ```math
# \text{Re} = \frac{u_\text{lid}\, N}{\nu}
# ```
#
# We use a coarse grid (``32^3``) for documentation purposes; production runs
# should use ``64^3`` or finer for quantitative comparisons with reference data.
#
# ## LBM setup
#
# | Parameter | Value |
# |-----------|-------|
# | Lattice   | D3Q19 |
# | Domain    | ``32 \times 32 \times 32`` |
# | Lid BC    | Zou--He velocity at ``j = N_y`` |
# | Other walls | Half-way bounce-back |
# | ``\text{Re}`` | 100 |
# | ``u_\text{lid}`` | 0.1 |
# | ``\nu``   | ``0.032`` |
#
# ## Simulation

using Kraken
using CairoMakie

N     = 32
Re    = 100
u_lid = 0.1
ν     = u_lid * N / Re

config = LBMConfig(D3Q19(); Nx=N, Ny=N, Nz=N, ν=ν, u_lid=u_lid,
                   max_steps=30000, output_interval=10000)
ρ, ux, uy, uz, _ = run_cavity_3d(config)

# ## Results
#
# We visualise the velocity magnitude in the mid-plane ``z = N/2``.

mid   = N ÷ 2
umag  = zeros(N, N)
for j in 1:N, i in 1:N
    umag[i, j] = sqrt(ux[i, j, mid]^2 + uy[i, j, mid]^2 + uz[i, j, mid]^2)
end
umag ./= u_lid

fig = Figure(size=(550, 480))
ax  = Axis(fig[1, 1]; title="Velocity magnitude — mid-plane z=$mid",
           xlabel="x", ylabel="y", aspect=DataAspect())
hm  = heatmap!(ax, 1:N, 1:N, umag; colormap=:viridis)
Colorbar(fig[1, 2], hm; label="|u| / u_lid")
fig
save("cavity_3d_midplane.svg", fig) #hide

# ## Vertical centreline profile
#
# Extract ``u_x`` along the vertical centreline at ``(x, z) = (N/2, N/2)``
# and compare with the 2D solution.

ux_profile = [ux[mid, j, mid] / u_lid for j in 1:N]
y_norm     = [(j - 0.5) / N for j in 1:N]

fig2 = Figure(size=(500, 400))
ax2  = Axis(fig2[1, 1]; xlabel="u_x / u_lid", ylabel="y / N",
            title="Vertical centreline — 3D cavity (N=$N)")
lines!(ax2, ux_profile, y_norm; linewidth=2, label="LBM D3Q19")
axislegend(ax2; position=:lb)
fig2
save("cavity_3d_centreline.svg", fig2) #hide

# !!! note "Grid resolution"
#     The ``32^3`` grid used here is deliberately coarse to keep build times
#     short.  For quantitative validation, use ``N \ge 64`` and compare against
#     [Ghia *et al.* (1982)](@cite ghia1982high) centreline data.
#
# ## References
#
# - [Ghia *et al.* (1982)](@cite ghia1982high) — Reference benchmark data (2D)
# - [He & Luo (1997)](@cite he1997theory) — Lattice Boltzmann theory
# - [Qian *et al.* (1992)](@cite qian1992lattice) — Lattice models
