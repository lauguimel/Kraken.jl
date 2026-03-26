# # Flow Around a Cylinder (2D) — Re = 20
#
# ![](cylinder_umag.svg)
#
# ## Problem statement
#
# Steady flow around a circular cylinder at low Reynolds number is a standard
# benchmark for external aerodynamics.  The Reynolds number is defined as
#
# ```math
# \text{Re} = \frac{u_\infty \, D}{\nu}
# ```
#
# where ``D = 2R`` is the cylinder diameter.  At ``\text{Re} = 20`` the flow is
# steady with a symmetric recirculation region behind the cylinder.  The drag
# coefficient ``C_d`` is well documented in the
# [Schafer--Turek benchmark](@cite schafer1996benchmark): ``C_d \approx 5.58``.
#
# The drag force is computed using the momentum-exchange method
# [Mei *et al.* (2002)](@cite mei2002accurate), which is exact for half-way
# bounce-back boundaries:
#
# ```math
# \mathbf{F} = \sum_{\text{boundary links}} \left[
#   f_{\bar{q}}(\mathbf{x}_f,t) + f_q(\mathbf{x}_f,t^+)
# \right] \mathbf{c}_q
# ```
#
# ## LBM setup
#
# | Parameter | Value |
# |-----------|-------|
# | Lattice   | D2Q9 |
# | Domain    | ``400 \times 100`` |
# | Cylinder  | Radius ``R = 10``, centred at ``(80, 50)`` |
# | Inlet     | Zou--He velocity, ``u_\infty = 0.04`` |
# | Outlet    | Convective / zero-gradient |
# | Top/Bottom| Free-slip |
# | ``\nu``   | ``u_\infty \cdot D / \text{Re} = 0.04`` |
#
# ## Simulation

using Kraken
using CairoMakie

Re     = 20
radius = 10
u_in   = 0.04
D      = 2 * radius
ν      = u_in * D / Re

ρ, ux, uy, config, Cd, Fx_hist, Fy_hist = run_cylinder_2d(;
    Nx=400, Ny=100, radius=radius, u_in=u_in, ν=ν,
    max_steps=40000, avg_window=2000)

# ## Results — velocity magnitude

Nx, Ny = size(ux)
umag = @. sqrt(ux^2 + uy^2)

fig = Figure(size=(800, 350))
ax  = Axis(fig[1, 1]; title="Velocity magnitude — Re=$Re",
           xlabel="x", ylabel="y", aspect=DataAspect())
hm  = heatmap!(ax, 1:Nx, 1:Ny, umag; colormap=:viridis,
               colorrange=(0, 1.5 * u_in))
Colorbar(fig[1, 2], hm; label="|u|")
fig
save("cylinder_umag.svg", fig) #hide

# ## Drag coefficient
#
# The time-averaged drag coefficient from the simulation is compared with the
# [Schafer--Turek benchmark](@cite schafer1996benchmark) value.

Cd_ref = 5.58
@info "Drag coefficient" Cd Cd_ref relative_error=abs(Cd - Cd_ref) / Cd_ref

# ```
# Schafer--Turek reference:  Cd = 5.58  (Re = 20)
# ```

fig2 = Figure(size=(600, 350))
ax2  = Axis(fig2[1, 1]; xlabel="Time step", ylabel="C_d",
            title="Drag coefficient history")
lines!(ax2, 1:length(Fx_hist), Fx_hist ./ (0.5 * u_in^2 * D);
       linewidth=1.5, label="Cd(t)")
hlines!(ax2, [Cd_ref]; color=:red, linestyle=:dash, label="Schafer--Turek")
axislegend(ax2; position=:rt)
fig2
save("cylinder_cd.svg", fig2) #hide

# ## References
#
# - [Schafer & Turek (1996)](@cite schafer1996benchmark) — Cylinder benchmark
# - [Mei *et al.* (2002)](@cite mei2002accurate) — Momentum exchange method
# - [Ladd (1994)](@cite ladd1994numerical) — Particle suspensions with LBM
# - [Kruger *et al.* (2017)](@cite kruger2017lattice) — LBM textbook
