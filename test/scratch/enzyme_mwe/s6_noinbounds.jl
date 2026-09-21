# S6: S0 without @inbounds.
include(joinpath(@__DIR__, "lib_standalone.jl"))
function wrap!(out, phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    adv = advect_noinbounds(phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    for j in 1:Ny, i in 1:Nx
        out[i, j] = adv[i, j]
    end
    return nothing
end
run_standalone_case("s6_noinbounds", wrap!, standalone_data())
