# S1: as the coupled step uses it: three fields, east BC = phi[Nx, :] slice
# taken inside the differentiated function, three advect calls.
include(joinpath(@__DIR__, "lib_standalone.jl"))
function wrap!(out, phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    phi2 = zeros(Nx, Ny); phi3 = zeros(Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        phi2[i, j] = 0.7 * phi[i, j]; phi3[i, j] = -0.4 * phi[i, j]
    end
    e1 = phi[Nx, :]; e2 = phi2[Nx, :]; e3 = phi3[Nx, :]
    a1 = advect_v0(phi, ux_face, uy_face, is_solid, Nx, Ny, e1)
    a2 = advect_v0(phi2, ux_face, uy_face, is_solid, Nx, Ny, e2)
    a3 = advect_v0(phi3, ux_face, uy_face, is_solid, Nx, Ny, e3)
    @inbounds for j in 1:Ny, i in 1:Nx
        out[i, j] = a1[i, j] + a2[i, j] * a3[i, j]
    end
    return nothing
end
run_standalone_case("s1_x3_slice", wrap!, standalone_data())
