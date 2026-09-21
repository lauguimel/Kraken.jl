# round 2: s0 on an 8x8 grid with a 1.6-radius cylinder
include(joinpath(@__DIR__, "lib_standalone.jl"))
function wrap!(out, phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    adv = advect_v0(phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    @inbounds for j in 1:Ny, i in 1:Nx
        out[i, j] = adv[i, j]
    end
    return nothing
end
run_standalone_case("t7_small", wrap!, standalone_data(Nx=8, Ny=8, cx=4.35, cy=3.65, R=1.6))
