# round 2: s0 run with --check-bounds=no (the harness appends the flag for *_nocb.jl)
include(joinpath(@__DIR__, "lib_standalone.jl"))
println("check_bounds = ", Base.JLOptions().check_bounds)
function wrap!(out, phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    adv = advect_v0(phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    @inbounds for j in 1:Ny, i in 1:Nx
        out[i, j] = adv[i, j]
    end
    return nothing
end
run_standalone_case("t9_s0_nocb", wrap!, standalone_data())
