using Kraken

function printwhich(label, f, args...)
    method = which(f, Tuple{(typeof.(args))...})
    println(label)
    println("  ", method)
end

Nx, Ny = 16, 16
f_out = zeros(Float64, Nx, Ny, 9)
f_in = similar(f_out)
rho = ones(Float64, Nx, Ny)
ux = zeros(Float64, Nx, Ny)
uy = zeros(Float64, Nx, Ny)
is_solid = falses(Nx, Ny)
q_wall = zeros(Float64, Nx, Ny, 9)
uwx = zeros(Float64, Nx, Ny, 9)
uwy = zeros(Float64, Nx, Ny, 9)
fx = zeros(Float64, Nx, Ny)
fy = zeros(Float64, Nx, Ny)
bc = Kraken.logfv_openx_wally_bcspec_2d()
west = zeros(Float64, Ny)
east = zeros(Float64, Ny)
south = zeros(Float64, Nx)
north = zeros(Float64, Nx)
phi_bc = Kraken.FVFDFieldBC2D(west, east, south, north)
geom = Kraken._logfv_cylinder_channel_geometry_2d(;
    radius=4, H=20, L_up=4, L_down=8, FT=Float64,
)

printwhich(
    "driver_step_entry",
    Kraken._run_viscoelastic_logfv_step_channel_coupled_2d,
    geom,
)
printwhich(
    "lbm_step",
    Kraken.fused_trt_libb_v2_guo_field_step!,
    f_out, f_in, rho, ux, uy, is_solid, q_wall, uwx, uwy, fx, fy, Nx, Ny, 0.1,
)
printwhich(
    "lbm_step_halfwayBB",
    Kraken._fused_trt_libb_v2_guo_field_step!,
    Val(:halfwayBB), f_out, f_in, rho, ux, uy, is_solid, q_wall, uwx, uwy,
    fx, fy, Nx, Ny, 0.1,
)
printwhich(
    "lbm_step_bouzidiFL",
    Kraken._fused_trt_libb_v2_guo_field_step!,
    Val(:bouzidi_fl), f_out, f_in, rho, ux, uy, is_solid, q_wall, uwx, uwy,
    fx, fy, Nx, Ny, 0.1,
)
printwhich(
    "psi_advect",
    Kraken.logfv_advect_upwind_bc_aware_2d!,
    fx, fy, rho, fx, fy, rho,
    west, west, west, east, east, east, south, south, south, north, north, north,
    fx, fy, is_solid, 1.0, 1.0, bc, 1.0,
)
printwhich(
    "psi_sym2_advect",
    Kraken.fvfd_sym2_advect_upwind_2d!,
    fx, fy, rho, fx, fy, rho, phi_bc, phi_bc, phi_bc,
    fx, fy, is_solid, 1.0, 1.0, bc, 1.0,
)
printwhich(
    "psi_advect_inner",
    Kraken.fvfd_advect_upwind_2d!,
    fx, rho, phi_bc, fx, fy, is_solid, 1.0, 1.0, bc, 1.0,
)
printwhich(
    "vel_grad",
    Kraken.fvfd_velocity_gradient_2d!,
    fx, fy, rho, ux, ux, uy, is_solid, 1.0, 1.0, bc,
)
printwhich(
    "poly_force",
    Kraken.logfv_polymer_force_bc_aware_2d!,
    fx, fy, rho, ux, uy, is_solid, 1.0, 1.0, bc,
)
