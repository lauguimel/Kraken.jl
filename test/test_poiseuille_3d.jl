using Test
using Kraken
using KernelAbstractions

# Minimal periodic-xz, wall-y streaming kernel for 3D Poiseuille test
@kernel function stream_periodic_xz_wall_y_3d_kernel!(f_out, @Const(f_in), Nx, Ny, Nz)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Periodic helpers
        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1, i - 1, Nx)
        kp = ifelse(k < Nz, k + 1, 1)
        km = ifelse(k > 1, k - 1, Nz)

        # q=1: rest
        f_out[i,j,k,1] = f_in[i, j, k, 1]

        # Axis-aligned x (periodic)
        f_out[i,j,k,2] = f_in[im, j, k, 2]
        f_out[i,j,k,3] = f_in[ip, j, k, 3]

        # Axis-aligned y (wall bounce-back)
        f_out[i,j,k,4] = ifelse(j > 1,  f_in[i, j-1, k, 4], f_in[i, j, k, 5])
        f_out[i,j,k,5] = ifelse(j < Ny, f_in[i, j+1, k, 5], f_in[i, j, k, 4])

        # Axis-aligned z (periodic)
        f_out[i,j,k,6] = f_in[i, j, km, 6]
        f_out[i,j,k,7] = f_in[i, j, kp, 7]

        # Edge xy (periodic x, wall y)
        f_out[i,j,k,8]  = ifelse(j > 1,  f_in[im, j-1, k, 8],  f_in[i, j, k, 11])
        f_out[i,j,k,9]  = ifelse(j > 1,  f_in[ip, j-1, k, 9],  f_in[i, j, k, 10])
        f_out[i,j,k,10] = ifelse(j < Ny, f_in[im, j+1, k, 10], f_in[i, j, k, 9])
        f_out[i,j,k,11] = ifelse(j < Ny, f_in[ip, j+1, k, 11], f_in[i, j, k, 8])

        # Edge xz (periodic x, periodic z)
        f_out[i,j,k,12] = f_in[im, j, km, 12]
        f_out[i,j,k,13] = f_in[ip, j, km, 13]
        f_out[i,j,k,14] = f_in[im, j, kp, 14]
        f_out[i,j,k,15] = f_in[ip, j, kp, 15]

        # Edge yz (periodic z, wall y)
        f_out[i,j,k,16] = ifelse(j > 1,  f_in[i, j-1, km, 16], f_in[i, j, k, 19])
        f_out[i,j,k,17] = ifelse(j < Ny, f_in[i, j+1, km, 17], f_in[i, j, k, 18])
        f_out[i,j,k,18] = ifelse(j > 1,  f_in[i, j-1, kp, 18], f_in[i, j, k, 17])
        f_out[i,j,k,19] = ifelse(j < Ny, f_in[i, j+1, kp, 19], f_in[i, j, k, 16])
    end
end

function stream_periodic_xz_wall_y_3d!(f_out, f_in, Nx, Ny, Nz)
    backend = KernelAbstractions.get_backend(f_in)
    kernel! = stream_periodic_xz_wall_y_3d_kernel!(backend)
    kernel!(f_out, f_in, Nx, Ny, Nz; ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

@testset "Poiseuille 3D body force" begin
    Nx, Ny, Nz = 4, 32, 4
    ν = 0.1
    Fx = 1e-5
    max_steps = 10000

    config = LBMConfig(D3Q19(); Nx=Nx, Ny=Ny, Nz=Nz, ν=ν, u_lid=0.0, max_steps=max_steps)
    state = initialize_3d(config, Float64)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux_field, uy_field, uz_field = state.ρ, state.ux, state.uy, state.uz
    is_solid = state.is_solid
    ω = Float64(omega(config))

    for step in 1:max_steps
        stream_periodic_xz_wall_y_3d!(f_out, f_in, Nx, Ny, Nz)
        collide_guo_3d!(f_out, is_solid, ω, Float64(Fx), 0.0, 0.0)
        compute_macroscopic_forced_3d!(ρ, ux_field, uy_field, uz_field, f_out, Float64(Fx), 0.0, 0.0)
        f_in, f_out = f_out, f_in
    end

    ρ_cpu = Array(ρ)
    ux_cpu = Array(ux_field)

    # Analytical parabolic profile (half-way bounce-back: walls at y=0.5, y=Ny+0.5)
    u_analytical = [Fx / (2ν) * (j - 0.5) * (Ny + 0.5 - j) for j in 1:Ny]
    u_numerical = ux_cpu[2, :, 2]  # extract at mid-plane (any x,z slice — periodic)

    # Interior points only (skip wall-adjacent nodes j=1 and j=Ny)
    u_max = maximum(u_analytical)
    errors = abs.(u_numerical[2:end-1] .- u_analytical[2:end-1])
    max_rel_err = maximum(errors) / u_max

    @test max_rel_err < 0.02  # 2% L∞ relative error
    @info "Poiseuille 3D: L∞ relative error = $(round(max_rel_err, digits=5))"

    # Mass conservation
    @test abs(sum(ρ_cpu) - Nx * Ny * Nz) / (Nx * Ny * Nz) < 0.001
end
