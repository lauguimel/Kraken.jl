# # Couette Flow (2D)
#
# ![](couette_profile.svg)
#
# ## Problem statement
#
# Plane Couette flow is the steady, shear-driven flow between two parallel
# plates.  The bottom wall moves at velocity ``u_w`` while the top wall is
# stationary.  The resulting velocity profile is linear:
#
# ```math
# u_x(y) = u_w \left(1 - \frac{y}{H}\right)
# ```
#
# where ``H`` is the effective channel height.  Couette flow tests the
# implementation of the moving-wall (Zou--He) boundary condition
# [Zou & He (1997)](@cite zou1997pressure).
#
# ## LBM setup
#
# | Parameter | Value |
# |-----------|-------|
# | Lattice   | D2Q9  |
# | Domain    | ``N_x \times N_y`` (periodic in *x*) |
# | Bottom wall | Moving wall at ``j=1``, ``u_w = 0.05`` (Zou--He) |
# | Top wall    | Stationary wall at ``j=N_y`` (half-way bounce-back) |
# | Collision | BGK, ``\omega = 1/(3\nu + 0.5)`` |
#
# Effective channel height: ``H = N_y - 1``.  Physical coordinate: ``y = j - 1.5``
# for fluid nodes ``j = 2, \ldots, N_y - 1``.
#
# ## Simulation

using Kraken
using CairoMakie

Ny     = 32
ν      = 0.1
u_wall = 0.05

ρ, ux, uy, config = run_couette_2d(; Nx=4, Ny=Ny, ν=ν, u_wall=u_wall, max_steps=20000)

# ## Results
#
# Compare the numerical profile along the vertical centreline with the
# analytical linear solution.

H = Ny - 1
j_fluid = 2:Ny-1
y_phys  = [j - 1.5 for j in j_fluid]
u_ana   = [u_wall * (1 - y / H) for y in y_phys]
u_num   = [ux[2, j] for j in j_fluid]

fig = Figure(size=(600, 420))
ax = Axis(fig[1, 1];
    xlabel = "u_x  (lattice units)",
    ylabel = "y  (lattice units)",
    title  = "Couette flow — N_y = $Ny")
lines!(ax, u_ana, y_phys; label="Analytical", linewidth=2)
scatter!(ax, u_num, y_phys; label="LBM", markersize=8)
axislegend(ax; position=:rt)
fig
save("couette_profile.svg", fig) #hide

# ## Convergence study
#
# The linear Couette profile is reproduced exactly by the D2Q9 lattice
# (the equilibrium is quadratic in velocity, which encompasses the linear
# solution).  We still observe small errors due to the bounce-back discretisation
# at the walls; these decrease at second order with resolution.

Ny_list = [16, 32, 64, 128]
errors  = Float64[]

for Ny_i in Ny_list
    ρ_i, ux_i, _, _ = run_couette_2d(; Nx=4, Ny=Ny_i, ν=ν, u_wall=u_wall, max_steps=30000)
    H_i  = Ny_i - 1
    jf   = 2:Ny_i-1
    u_a  = [u_wall * (1 - (j - 1.5) / H_i) for j in jf]
    u_n  = [ux_i[2, j] for j in jf]
    L2   = sqrt(sum((u_n .- u_a).^2) / sum(u_a.^2))
    push!(errors, L2)
end

fig2 = Figure(size=(500, 400))
ax2  = Axis(fig2[1, 1];
    xlabel = "N_y", ylabel = "Relative L_2 error",
    title  = "Convergence — Couette flow",
    xscale = log10, yscale = log10)
scatterlines!(ax2, Float64.(Ny_list), errors; linewidth=2, markersize=10, label="LBM")
ref = errors[1] .* (Ny_list[1] ./ Ny_list).^2
lines!(ax2, Float64.(Ny_list), ref; linestyle=:dash, color=:gray, label="slope 2")
axislegend(ax2; position=:lb)
fig2
save("couette_convergence.svg", fig2) #hide

# ## References
#
# - [Zou & He (1997)](@cite zou1997pressure) — Zou--He boundary conditions
# - [Kruger *et al.* (2017)](@cite kruger2017lattice) — LBM textbook
