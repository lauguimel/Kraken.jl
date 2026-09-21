# round 3: the smallest shape that gives a wrong reverse gradient on macOS
# (forward/reverse disagree by ~2%, deterministic): 3-way branch on a Const
# mask (cell + neighbours), upwind flux in the band, central stencil elsewhere.
include(joinpath(@__DIR__, "lib_standalone.jl"))
@inline function upwind_x(phi, ux_face, east_phi, Nx, i, j)
    ue = ux_face[i + 1, j]; uw = ux_face[i, j]
    phie = ue >= 0 ? phi[i, j] : (i < Nx ? phi[i + 1, j] : east_phi[j])
    phiw = uw >= 0 ? (i > 1 ? phi[i - 1, j] : 0.0) : phi[i, j]
    return -((ue * phie - uw * phiw) - phi[i, j] * (ue - uw))
end
function wrap!(out, phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            out[i, j] = 0.0
        elseif i <= 2 || i >= Nx - 1 || j <= 2 || j >= Ny - 1 || is_solid[i - 1, j] || is_solid[i + 1, j]
            out[i, j] = phi[i, j] + upwind_x(phi, ux_face, east_phi, Nx, i, j)
        else
            out[i, j] = phi[i, j] - 0.5 * ux_face[i, j] * (phi[i + 1, j] - phi[i - 1, j])
        end
    end
    return nothing
end
println("check_bounds = ", Base.JLOptions().check_bounds)
run_standalone_case("u1_n2_shape", wrap!, standalone_data(Nx=8, Ny=8, cx=4.35, cy=3.65, R=0.0))
