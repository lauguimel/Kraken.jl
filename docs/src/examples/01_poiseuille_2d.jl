# # Poiseuille Flow (2D)
#
#
# ## Problem statement
#
# Plane Poiseuille flow is the steady, fully-developed flow between two infinite
# parallel plates driven by a uniform body force ``F_x``.  The velocity profile
# is parabolic:
#
# ```math
# u_x(y) = \frac{F_x}{2\nu}\, y\,(H - y)
# ```
#
# where ``H`` is the effective channel height and ``\nu`` the kinematic
# viscosity.  This is the simplest validation case for any Navier--Stokes
# solver and a standard first test for LBM implementations
# [Qian *et al.* (1992)](@cite qian1992lattice).
#
# ## LBM setup
#
# | Parameter | Value |
# |-----------|-------|
# | Lattice   | D2Q9  |
# | Domain    | ``N_x \times N_y`` (periodic in *x*, walls in *y*) |
# | Walls     | Half-way bounce-back at ``j=1`` and ``j=N_y`` |
# | Forcing   | Guo discrete forcing scheme [Guo *et al.* (2002)](@cite guo2002discrete) |
# | Collision | BGK with ``\omega = 1/(3\nu + 0.5)`` [BGK (1954)](@cite bgk1954) |
#
# Effective channel height: ``H = N_y - 1`` (distance between half-way BB
# walls).  The physical coordinate is ``y = j - 1.5`` for lattice index ``j``
# (fluid nodes ``j = 2, \ldots, N_y - 1``).
#
# Stability requires ``\omega < 2``, i.e.\ ``\nu > 0.5(\omega^{-1}-0.5) > 0``.
#
# ## Simulation

using Kraken
using CairoMakie

Ny = 32
ν  = 0.1
Fx = 1e-5

ρ, ux, uy, config = run_poiseuille_2d(; Nx=4, Ny=Ny, ν=ν, Fx=Fx, max_steps=20000)

# ## Results
#
# Extract the velocity profile along the vertical centreline and compare with
# the analytical parabola.

H = Ny - 1                                  # effective channel height
j_fluid = 2:Ny-1                             # fluid nodes
y_phys  = [j - 1.5 for j in j_fluid]        # physical y coordinate
u_ana   = [Fx / (2ν) * y * (H - y) for y in y_phys]
u_num   = [ux[2, j] for j in j_fluid]       # profile at x=2

fig = Figure(size=(600, 420))
ax = Axis(fig[1, 1];
    xlabel = "u_x  (lattice units)",
    ylabel = "y  (lattice units)",
    title  = "Poiseuille flow — N_y = $Ny")
lines!(ax, u_ana, y_phys; label="Analytical", linewidth=2)
scatter!(ax, u_num, y_phys; label="LBM", markersize=8)
axislegend(ax; position=:rb)
fig
save("poiseuille_profile.svg", fig) #hide

# ## Convergence study
#
# We measure the relative ``L_2`` error for increasing resolutions and expect
# second-order convergence (slope ``\approx 2`` on a log-log plot), consistent
# with the Chapman--Enskog analysis of the BGK operator.

Ny_list = [16, 32, 64, 128]
errors  = Float64[]

for Ny_i in Ny_list
    ρ_i, ux_i, _, _ = run_poiseuille_2d(; Nx=4, Ny=Ny_i, ν=ν, Fx=Fx, max_steps=30000)
    H_i    = Ny_i - 1
    jf     = 2:Ny_i-1
    u_a    = [Fx / (2ν) * (j - 1.5) * (H_i - (j - 1.5)) for j in jf]
    u_n    = [ux_i[2, j] for j in jf]
    L2     = sqrt(sum((u_n .- u_a).^2) / sum(u_a.^2))
    push!(errors, L2)
end

fig2 = Figure(size=(500, 400))
ax2  = Axis(fig2[1, 1];
    xlabel = "N_y", ylabel = "Relative L_2 error",
    title  = "Convergence — Poiseuille flow",
    xscale = log10, yscale = log10)
scatterlines!(ax2, Float64.(Ny_list), errors; linewidth=2, markersize=10, label="LBM")
## Reference slope (order 2)
ref = errors[1] .* (Ny_list[1] ./ Ny_list).^2
lines!(ax2, Float64.(Ny_list), ref; linestyle=:dash, color=:gray, label="slope 2")
axislegend(ax2; position=:lb)
fig2
save("poiseuille_convergence.svg", fig2) #hide

# ## References
#
# - [BGK (1954)](@cite bgk1954) — BGK collision operator
# - [Qian *et al.* (1992)](@cite qian1992lattice) — D2Q9 lattice
# - [Guo *et al.* (2002)](@cite guo2002discrete) — Discrete forcing scheme
# - [Kruger *et al.* (2017)](@cite kruger2017lattice) — LBM textbook
