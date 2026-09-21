# S5: MUSCL relax rhs only on interior cells, no closures, no band logic.
include(joinpath(@__DIR__, "lib_standalone.jl"))
function wrap!(out, phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    adv = advect_musclonly(phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    @inbounds for j in 1:Ny, i in 1:Nx
        out[i, j] = adv[i, j]
    end
    return nothing
end
run_standalone_case("s5_musclonly", wrap!, standalone_data())
