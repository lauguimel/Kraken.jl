# S8: S0 with plain Reverse/Forward (no set_runtime_activity).
include(joinpath(@__DIR__, "lib_standalone.jl"))
function wrap!(out, phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    adv = advect_v0(phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    @inbounds for j in 1:Ny, i in 1:Nx
        out[i, j] = adv[i, j]
    end
    return nothing
end
run_standalone_case("s8_no_runtime_activity", wrap!, standalone_data();
                    revmode=Enzyme.Reverse, fwdmode=Enzyme.Forward)
