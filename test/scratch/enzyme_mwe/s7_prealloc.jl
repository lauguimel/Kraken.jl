# S7: S0 with the result array allocated by the caller (no zeros() inside).
include(joinpath(@__DIR__, "lib_standalone.jl"))
function wrap!(out, phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    advect_prealloc!(out, phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    return nothing
end
run_standalone_case("s7_prealloc", wrap!, standalone_data())
