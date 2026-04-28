# # Tutorial 2 — Adding Body Forces (Poiseuille Flow)
#
# In the previous tutorial the flow was driven by a moving wall. Here we
# replace that mechanism with a **uniform body force**, producing the
# classic Poiseuille (channel) flow — and use it to verify that Kraken.jl
# is second-order accurate.
#
# ---
#
# ## 1. The `.krk` file
#
# ```
# Simulation poiseuille D2Q9
# Domain  L = 0.125 x 1.0  N = 4 x 32
# Physics nu = 0.1  Fx = 1e-5
#
# Boundary x periodic
# Boundary south wall
# Boundary north wall
#
# Run 10000 steps
# Output vtk every 2000 [rho, ux, uy]
# ```
#
# Three new concepts compared to the cavity:
#
# ### `Boundary x periodic`
#
# A shorthand that makes **both** the west and east faces periodic.
# Populations leaving one side re-enter from the other. This models an
# infinitely long channel without inlet/outlet effects.
#
# ### `Physics nu = 0.1  Fx = 1e-5`
#
# `Fx` is a uniform body force in the x-direction (lattice units).
# Kraken automatically switches to the **Guo forcing scheme**, which
# adds a source term at the distribution-function level so that the
# recovered Navier--Stokes equations are correct to second order.
#
# In a periodic domain there is no inlet or outlet, so a body force
# replaces the pressure gradient:
# ``\partial p / \partial x = -\rho F_x``.
#
# ---
#
# ## 2. Analytical solution
#
# With effective channel height ``H = N_y - 1`` (half-way bounce-back
# walls), the steady-state velocity is:
#
# ```math
# u_x(y) = \frac{F_x}{2\nu}\, y\,(H - y)
# ```
#
# This is a parabola with maximum ``u_\text{max} = F_x H^2 / (8\nu)``
# at the centreline.
#
# ---
#
# ## 3. Running and comparing to theory

using Kraken

KRK = joinpath(@__DIR__, "..", "..", "..", "examples")

result = run_simulation(joinpath(KRK, "poiseuille.krk"))

Ny = 32
ν  = 0.1
Fx = 1e-5
H       = Ny - 1
j_fluid = 2:Ny-1
y_phys  = [j - 1.5 for j in j_fluid]
u_ana   = [Fx / (2ν) * y * (H - y) for y in y_phys]
u_num   = [result.ux[2, j] for j in j_fluid]

# ## 4. Plot — profile comparison

using CairoMakie

fig = Figure(size = (500, 400))
ax  = Axis(fig[1, 1]; xlabel = "y", ylabel = "ux",
           title = "Poiseuille — Ny = $Ny")
lines!(ax, y_phys, u_ana; label = "Analytical")
scatter!(ax, y_phys, u_num; markersize = 6, label = "LBM")
axislegend(ax; position = :ct)
save(joinpath(@__DIR__, "02_poiseuille_profile.svg"), fig)
fig

#
# ## 5. Convergence study
#
# Run at four resolutions and compute the relative ``L_2`` error.

# We override `Ny` and `max_steps` directly from Julia — the `.krk`
# parametric system makes convergence studies trivial.

Ny_list = [16, 32, 64, 128]
errors  = Float64[]

for Ny_i in Ny_list
    r = run_simulation(joinpath(KRK, "poiseuille.krk");
                       Ny = Ny_i, max_steps = 30000)
    H_i = Ny_i - 1
    jf  = 2:Ny_i-1
    u_a = [Fx / (2ν) * (j - 1.5) * (H_i - (j - 1.5)) for j in jf]
    u_n = [r.ux[2, j] for j in jf]
    push!(errors, sqrt(sum((u_n .- u_a) .^ 2) / sum(u_a .^ 2)))
end

# ## 6. Plot — convergence

fig2 = Figure(size = (500, 400))
ax2  = Axis(fig2[1, 1]; xlabel = "Ny", ylabel = "L2 error",
            title = "Poiseuille convergence", xscale = log10, yscale = log10)
scatterlines!(ax2, Float64.(Ny_list), errors; label = "LBM")
lines!(ax2, Float64.(Ny_list),
       errors[1] .* (Ny_list[1] ./ Ny_list) .^ 2;
       linestyle = :dash, color = :gray, label = "slope 2")
axislegend(ax2)
save(joinpath(@__DIR__, "02_poiseuille_convergence.svg"), fig2)
fig2

#
# ---
#
# ## 7. Key takeaway
#
# The BGK-LBM is **second-order accurate** in space: doubling the
# resolution divides the error by four, as predicted by the
# Chapman--Enskog expansion.
#
# **Previous:** [Tutorial 1 — Your first simulation](01_first_simulation.md)
