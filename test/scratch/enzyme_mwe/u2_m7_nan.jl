# round 3: same 3-way shape with the MUSCL stencil in the last branch; on macOS
# the reverse gradient is NaN and differs between two identical runs.
include(joinpath(@__DIR__, "lib_standalone.jl"))
function wrap!(out, phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            out[i, j] = 0.0
        elseif in_band(is_solid, i, j, Nx, Ny)
            out[i, j] = phi[i, j]
        else
            out[i, j] = phi[i, j] + muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, 1.0, 1.0)
        end
    end
    return nothing
end
run_standalone_case("u2_m7_nan", wrap!, standalone_data(Nx=8, Ny=8, cx=4.35, cy=3.65, R=0.0))
