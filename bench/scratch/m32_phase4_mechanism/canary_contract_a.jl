using Kraken
using KernelAbstractions

function canary_call()
    Nx, Ny = 16, 16
    ux = zeros(Nx, Ny)
    uy = zeros(Nx, Ny)
    dudx = zeros(Nx, Ny)
    dudy = zeros(Nx, Ny)
    dvdx = zeros(Nx, Ny)
    dvdy = zeros(Nx, Ny)
    is_solid = falses(Nx, Ny)
    bc = Kraken.logfv_wallxwally_bcspec_2d()
    return Kraken.fvfd_velocity_gradient_2d!(
        dudx, dudy, dvdx, dvdy, ux, uy, is_solid, 1.0, 1.0, bc; sync=true,
    )
end

canary_call()
canary_call()
delete!(ENV, "KRAKEN_TRACE")
allocs_off = @allocated canary_call()
t_off = minimum(@elapsed canary_call() for _ in 1:100)
@show allocs_off t_off
