function logfv_cell_velocity_to_faces_bc_aware_2d!(
    ux_face, uy_face, ux, uy, is_solid,
    ux_west, ux_east, uy_south, uy_north,
    bc::LogFVDomainBC2D;
    sync::Bool=true,
)
    return fvfd_cell_velocity_to_faces_2d!(
        ux_face, uy_face, ux, uy, is_solid,
        ux_west, ux_east, uy_south, uy_north,
        bc; sync,
    )
end

function logfv_cell_velocity_to_faces_embedded_2d!(
    ux_face, uy_face, ux, uy, geometry::FVFDGeometry2D,
    ux_bc::FVFDFieldBC2D, uy_bc::FVFDFieldBC2D;
    sync::Bool=true,
)
    return fvfd_cell_velocity_to_faces_embedded_2d!(
        ux_face, uy_face, ux, uy, geometry, ux_bc, uy_bc; sync,
    )
end

function logfv_cell_velocity_to_faces_solid_aware_2d!(
    ux_face, uy_face, ux, uy, is_solid;
    sync::Bool=true,
)
    return logfv_cell_velocity_to_faces_bc_aware_2d!(
        ux_face, uy_face, ux, uy, is_solid,
        ux, ux, uy, uy,
        logfv_periodicx_wally_bcspec_2d();
        sync,
    )
end

function logfv_cell_velocity_to_faces_openx_solid_aware_2d!(
    ux_face, uy_face, ux, uy, is_solid, ux_west, ux_east;
    sync::Bool=true,
)
    return logfv_cell_velocity_to_faces_bc_aware_2d!(
        ux_face, uy_face, ux, uy, is_solid,
        ux_west, ux_east, uy, uy,
        logfv_openx_wally_bcspec_2d();
        sync,
    )
end

function logfv_advect_upwind_bc_aware_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy,
    west_xx, west_xy, west_yy,
    east_xx, east_xy, east_yy,
    south_xx, south_xy, south_yy,
    north_xx, north_xy, north_yy,
    ux_face, uy_face, is_solid,
    dx, dy, bc::LogFVDomainBC2D, dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    @trace_enter :psi_advect
    return fvfd_sym2_advect_upwind_2d!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy,
        FVFDFieldBC2D(west_xx, east_xx, south_xx, north_xx),
        FVFDFieldBC2D(west_xy, east_xy, south_xy, north_xy),
        FVFDFieldBC2D(west_yy, east_yy, south_yy, north_yy),
        ux_face, uy_face, is_solid, dx, dy, bc, dt; sync, advection_scheme,
    )
end

function logfv_advect_upwind_bc_aware_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy,
    west_xx, west_xy, west_yy,
    east_xx, east_xy, east_yy,
    south_xx, south_xy, south_yy,
    north_xx, north_xy, north_yy,
    ux_face, uy_face, geometry::FVFDGeometry2D, dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    return logfv_advect_upwind_bc_aware_2d!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy,
        west_xx, west_xy, west_yy,
        east_xx, east_xy, east_yy,
        south_xx, south_xy, south_yy,
        north_xx, north_xy, north_yy,
        ux_face, uy_face, geometry.is_solid,
        geometry.patch.dx, geometry.patch.dy, geometry.bc, dt;
        sync,
        advection_scheme,
    )
end

function logfv_advect_upwind_bc_aware_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy,
    west_xx, west_xy, west_yy,
    east_xx, east_xy, east_yy,
    south_xx, south_xy, south_yy,
    north_xx, north_xy, north_yy,
    ux_face, uy_face, is_solid,
    bc::LogFVDomainBC2D, dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    spacing = one(eltype(psixx_out))
    return logfv_advect_upwind_bc_aware_2d!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy,
        west_xx, west_xy, west_yy,
        east_xx, east_xy, east_yy,
        south_xx, south_xy, south_yy,
        north_xx, north_xy, north_yy,
        ux_face, uy_face, is_solid,
        spacing, spacing, bc, dt;
        sync,
        advection_scheme,
    )
end

function logfv_advect_upwind_embedded_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy,
    psixx_bc::FVFDFieldBC2D, psixy_bc::FVFDFieldBC2D, psiyy_bc::FVFDFieldBC2D,
    ux_face, uy_face, ux, uy,
    geometry::FVFDGeometry2D,
    ux_bc::FVFDFieldBC2D, uy_bc::FVFDFieldBC2D,
    dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    return fvfd_sym2_advect_upwind_embedded_2d!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy,
        psixx_bc, psixy_bc, psiyy_bc,
        ux_face, uy_face, ux, uy,
        geometry, ux_bc, uy_bc, dt; sync, advection_scheme,
    )
end

function logfv_advect_upwind_solid_aware_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy, ux_face, uy_face, is_solid, dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    return logfv_advect_upwind_bc_aware_2d!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy,
        psixx, psixy, psiyy,
        psixx, psixy, psiyy,
        psixx, psixy, psiyy,
        psixx, psixy, psiyy,
        ux_face, uy_face, is_solid,
        logfv_periodicx_wally_bcspec_2d(), dt;
        sync,
        advection_scheme,
    )
end

function logfv_advect_upwind_openx_solid_aware_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy,
    west_xx, west_xy, west_yy,
    east_xx, east_xy, east_yy,
    ux_face, uy_face, is_solid, dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    return logfv_advect_upwind_bc_aware_2d!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy,
        west_xx, west_xy, west_yy,
        east_xx, east_xy, east_yy,
        psixx, psixy, psiyy,
        psixx, psixy, psiyy,
        ux_face, uy_face, is_solid,
        logfv_openx_wally_bcspec_2d(), dt;
        sync,
        advection_scheme,
    )
end
