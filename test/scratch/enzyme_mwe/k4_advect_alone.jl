# K4: reverse over Kraken's ad_ve_advect_prodbc alone, real geometry mask and
# real face velocities from the converged state.
include(joinpath(@__DIR__, "lib_kraken.jl"))
c = FAST
p, geom = build_case(c)
w_star = base_state(c, p, geom; converge=true)
Nx, Ny = c.Nx, c.Ny
n = Nx * Ny
ux_face, uy_face = K.ad_ve_compute_faces(w_star, geom.g, p)
phi = zeros(Nx, Ny)
for j in 1:Ny, i in 1:Nx
    phi[i, j] = w_star[9n + K.ad_ve_lin(i, j, Nx)]
end
d = (; Nx, Ny, is_solid=geom.g.is_solid, phi, ux_face, uy_face, east_phi=phi[Nx, :])
function wrap!(out, phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    adv = K.ad_ve_advect_prodbc(phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    @inbounds for j in 1:Ny, i in 1:Nx
        out[i, j] = adv[i, j]
    end
    return nothing
end
run_standalone_case("k4_advect_alone", wrap!, d)
