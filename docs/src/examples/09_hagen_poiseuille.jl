# # Hagen--Poiseuille Flow (Axisymmetric)
#
#
# ## Problem statement
#
# Hagen--Poiseuille flow is the fully-developed laminar flow in a circular pipe
# driven by a uniform body force ``F_z``.  In cylindrical coordinates
# ``(r, \theta, z)`` with axial symmetry, the velocity profile is parabolic:
#
# ```math
# u_z(r) = \frac{F_z}{4\nu}\left(R^2 - r^2\right)
# ```
#
# where ``R`` is the effective pipe radius and ``\nu`` the kinematic viscosity.
# This is the axisymmetric counterpart of the 2D Poiseuille flow and validates
# the source-term formulation for cylindrical coordinates in the LBM
# [Li (2010)](@cite li2010improved).
#
# ## LBM setup
#
# | Parameter | Value |
# |-----------|-------|
# | Lattice   | D2Q9 in ``(z, r)`` half-plane |
# | Domain    | ``N_z \times N_r`` (periodic in *z*, wall at ``r = N_r``, axis at ``r = 0``) |
# | Wall      | Half-way bounce-back at ``j = N_r`` |
# | Axis      | Symmetry condition at ``j = 1`` |
# | Forcing   | Axisymmetric Guo scheme with Li (2010) source terms |
# | Collision | BGK, ``\omega = 1/(3\nu + 0.5)`` |
#
# The effective pipe radius with half-way BB is ``R = N_r - 0.5``.  The radial
# coordinate is ``r = j - 0.5`` for lattice index ``j`` (where ``j = 1`` is the
# axis).
#
# ## Simulation

using Kraken
using CairoMakie

Nr = 32
ν  = 0.1
Fz = 1e-5

ρ, uz, ur, config = run_hagen_poiseuille_2d(;
    Nz=4, Nr=Nr, ν=ν, Fz=Fz, max_steps=20000)

# ## Results
#
# Extract the axial velocity profile and compare with the analytical parabola.

R_eff   = Nr - 0.5                            # effective radius (half-way BB)
j_fluid = 1:Nr                                # all nodes (axis to wall)
r_phys  = [j - 0.5 for j in j_fluid]          # physical radial coordinate
u_ana   = [Fz / (4ν) * (R_eff^2 - r^2) for r in r_phys]
u_num   = [uz[2, j] for j in j_fluid]         # profile at z=2

fig = Figure(size=(600, 420))
ax = Axis(fig[1, 1];
    xlabel = "u_z  (lattice units)",
    ylabel = "r  (lattice units)",
    title  = "Hagen--Poiseuille flow — N_r = $Nr")
lines!(ax, u_ana, r_phys; label="Analytical", linewidth=2)
scatter!(ax, u_num, r_phys; label="LBM (axisymmetric)", markersize=8)
axislegend(ax; position=:rt)
fig
save("hagen_poiseuille_profile.svg", fig) #hide

# ## Convergence study
#
# The axisymmetric source terms introduce additional discretisation errors
# compared to the Cartesian case.  We verify that second-order convergence
# is maintained.

Nr_list = [16, 32, 64, 128]
errors  = Float64[]

for Nr_i in Nr_list
    ρ_i, uz_i, _, _ = run_hagen_poiseuille_2d(;
        Nz=4, Nr=Nr_i, ν=ν, Fz=Fz, max_steps=30000)
    R_i  = Nr_i - 0.5
    jf   = 1:Nr_i
    u_a  = [Fz / (4ν) * (R_i^2 - (j - 0.5)^2) for j in jf]
    u_n  = [uz_i[2, j] for j in jf]
    L2   = sqrt(sum((u_n .- u_a).^2) / sum(u_a.^2))
    push!(errors, L2)
end

fig2 = Figure(size=(500, 400))
ax2  = Axis(fig2[1, 1];
    xlabel = "N_r", ylabel = "Relative L_2 error",
    title  = "Convergence — Hagen--Poiseuille flow",
    xscale = log10, yscale = log10)
scatterlines!(ax2, Float64.(Nr_list), errors;
              linewidth=2, markersize=10, label="LBM")
ref = errors[1] .* (Nr_list[1] ./ Nr_list).^2
lines!(ax2, Float64.(Nr_list), ref;
       linestyle=:dash, color=:gray, label="slope 2")
axislegend(ax2; position=:lb)
fig2
save("hagen_poiseuille_convergence.svg", fig2) #hide

# ## References
#
# - [Li (2010)](@cite li2010improved) — Axisymmetric LBM source terms
# - [Kruger *et al.* (2017)](@cite kruger2017lattice) — LBM textbook
# - [Guo *et al.* (2002)](@cite guo2002discrete) — Discrete forcing scheme
