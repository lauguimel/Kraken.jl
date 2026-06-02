@inline function _fvfd_bc_east_scalar_2d(phi, east_phi, i, j, Nx, east_bc)
    if i < Nx
        return phi[i + 1, j]
    elseif east_bc == FVFD_BC_PERIODIC
        return phi[1, j]
    elseif east_bc == FVFD_BC_OPEN
        return east_phi[j]
    else
        return phi[i, j]
    end
end

@inline function _fvfd_bc_west_scalar_2d(phi, west_phi, i, j, Nx, west_bc)
    if i > 1
        return phi[i - 1, j]
    elseif west_bc == FVFD_BC_PERIODIC
        return phi[Nx, j]
    elseif west_bc == FVFD_BC_OPEN
        return west_phi[j]
    else
        return phi[i, j]
    end
end

@inline function _fvfd_bc_north_scalar_2d(phi, north_phi, i, j, Ny, north_bc)
    if j < Ny
        return phi[i, j + 1]
    elseif north_bc == FVFD_BC_PERIODIC
        return phi[i, 1]
    elseif north_bc == FVFD_BC_OPEN
        return north_phi[i]
    else
        return phi[i, j]
    end
end

@inline function _fvfd_bc_south_scalar_2d(phi, south_phi, i, j, Ny, south_bc)
    if j > 1
        return phi[i, j - 1]
    elseif south_bc == FVFD_BC_PERIODIC
        return phi[i, Ny]
    elseif south_bc == FVFD_BC_OPEN
        return south_phi[i]
    else
        return phi[i, j]
    end
end

function _fvfd_advection_scheme_val(advection_scheme::Symbol)
    scheme = Symbol(replace(lowercase(String(advection_scheme)), '-' => '_'))
    scheme in (:rusanov, :muscl_superbee, :muscl_superbee_relax) ||
        throw(ArgumentError("advection_scheme must be :rusanov, :muscl_superbee, or :muscl_superbee_relax"))
    return Val(scheme)
end

@inline function _fvfd_superbee_limiter_2d(r)
    one_r = one(r)
    two_r = one_r + one_r
    return max(zero(r), max(min(two_r * r, one_r), min(r, two_r)))
end

@inline function _fvfd_muscl_superbee_face_value_2d(far_upwind, upwind, downwind)
    d_up = upwind - far_upwind
    d_down = downwind - upwind
    r = ifelse(d_down == zero(d_down), zero(d_down), d_up / d_down)
    return upwind + (one(r) / (one(r) + one(r))) * _fvfd_superbee_limiter_2d(r) * d_down
end

@inline function _fvfd_upwind_scalar_advective_rhs_2d(
    phi, west_phi, east_phi, south_phi, north_phi,
    ux_face, uy_face, is_solid, i, j, Nx, Ny, inv_dx, inv_dy,
    west_bc, east_bc, south_bc, north_bc,
    ::Val{:rusanov},
)
    ue = ux_face[i + 1, j]
    uw = ux_face[i, j]
    vn = uy_face[i, j + 1]
    vs = uy_face[i, j]

    east_value = _fvfd_bc_east_scalar_2d(phi, east_phi, i, j, Nx, east_bc)
    west_value = _fvfd_bc_west_scalar_2d(phi, west_phi, i, j, Nx, west_bc)
    north_value = _fvfd_bc_north_scalar_2d(phi, north_phi, i, j, Ny, north_bc)
    south_value = _fvfd_bc_south_scalar_2d(phi, south_phi, i, j, Ny, south_bc)

    phie = ifelse(ue >= 0, phi[i, j], east_value)
    phiw = ifelse(uw >= 0, west_value, phi[i, j])
    phin = ifelse(vn >= 0, phi[i, j], north_value)
    phis = ifelse(vs >= 0, south_value, phi[i, j])

    flux_div = (ue * phie - uw * phiw) * inv_dx +
               (vn * phin - vs * phis) * inv_dy
    divu = (ue - uw) * inv_dx + (vn - vs) * inv_dy
    return -(flux_div - phi[i, j] * divu)
end

@inline function _fvfd_upwind_scalar_advective_rhs_2d(
    phi, west_phi, east_phi, south_phi, north_phi,
    ux_face, uy_face, is_solid, i, j, Nx, Ny, inv_dx, inv_dy,
    west_bc, east_bc, south_bc, north_bc,
    ::Val{:muscl_superbee},
)
    if i <= 2 || i >= Nx - 1 || j <= 2 || j >= Ny - 1 ||
       is_solid[i - 2, j] || is_solid[i - 1, j] ||
       is_solid[i + 1, j] || is_solid[i + 2, j] ||
       is_solid[i, j - 2] || is_solid[i, j - 1] ||
       is_solid[i, j + 1] || is_solid[i, j + 2]
        return _fvfd_upwind_scalar_advective_rhs_2d(
            phi, west_phi, east_phi, south_phi, north_phi,
            ux_face, uy_face, is_solid, i, j, Nx, Ny, inv_dx, inv_dy,
            west_bc, east_bc, south_bc, north_bc, Val(:rusanov),
        )
    end

    ue = ux_face[i + 1, j]
    uw = ux_face[i, j]
    vn = uy_face[i, j + 1]
    vs = uy_face[i, j]

    phie = ifelse(
        ue >= 0,
        _fvfd_muscl_superbee_face_value_2d(phi[i - 1, j], phi[i, j], phi[i + 1, j]),
        _fvfd_muscl_superbee_face_value_2d(phi[i + 2, j], phi[i + 1, j], phi[i, j]),
    )
    phiw = ifelse(
        uw >= 0,
        _fvfd_muscl_superbee_face_value_2d(phi[i - 2, j], phi[i - 1, j], phi[i, j]),
        _fvfd_muscl_superbee_face_value_2d(phi[i + 1, j], phi[i, j], phi[i - 1, j]),
    )
    phin = ifelse(
        vn >= 0,
        _fvfd_muscl_superbee_face_value_2d(phi[i, j - 1], phi[i, j], phi[i, j + 1]),
        _fvfd_muscl_superbee_face_value_2d(phi[i, j + 2], phi[i, j + 1], phi[i, j]),
    )
    phis = ifelse(
        vs >= 0,
        _fvfd_muscl_superbee_face_value_2d(phi[i, j - 2], phi[i, j - 1], phi[i, j]),
        _fvfd_muscl_superbee_face_value_2d(phi[i, j + 1], phi[i, j], phi[i, j - 1]),
    )

    flux_div = (ue * phie - uw * phiw) * inv_dx +
               (vn * phin - vs * phis) * inv_dy
    divu = (ue - uw) * inv_dx + (vn - vs) * inv_dy
    return -(flux_div - phi[i, j] * divu)
end

@kernel function fvfd_advect_upwind_2d_kernel!(
    phi_out, @Const(phi),
    @Const(west_phi), @Const(east_phi), @Const(south_phi), @Const(north_phi),
    @Const(ux_face), @Const(uy_face), @Const(is_solid),
    dt, inv_dx, inv_dy,
    west_bc, east_bc, south_bc, north_bc, Nx, Ny, advection_scheme,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if is_solid[i, j]
                phi_out[i, j] = zero(eltype(phi_out))
            else
                rhs = _fvfd_upwind_scalar_advective_rhs_2d(
                    phi, west_phi, east_phi, south_phi, north_phi,
                    ux_face, uy_face, is_solid, i, j, Nx, Ny, inv_dx, inv_dy,
                    west_bc, east_bc, south_bc, north_bc, advection_scheme,
                )
                phi_out[i, j] = phi[i, j] + dt * rhs
            end
        end
    end
end

function fvfd_advect_upwind_2d!(
    phi_out, phi, phi_bc::FVFDFieldBC2D,
    ux_face, uy_face, is_solid, dx, dy, bc::FVFDDomainBC2D, dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    @trace_enter :psi_advect_inner
    backend = KernelAbstractions.get_backend(phi_out)
    Nx, Ny = size(phi_out)
    fvfd_validate_field_bc_2d(phi_bc, Nx, Ny, bc; name=:phi_bc)
    scheme_sym = Symbol(replace(lowercase(String(advection_scheme)), '-' => '_'))
    # M42: :muscl_superbee_relax is a two-pass composite (pass-1 = full
    # :muscl_superbee + pass-2 = cylinder-band one-sided MUSCL overwrite).
    # The composite launcher itself recurses with :muscl_superbee.
    if scheme_sym === :muscl_superbee_relax
        return fvfd_advect_muscl_superbee_relax_2d!(
            phi_out, phi, phi_bc,
            ux_face, uy_face, is_solid, dx, dy, bc, dt; sync,
        )
    end
    scheme = _fvfd_advection_scheme_val(advection_scheme)
    kernel! = fvfd_advect_upwind_2d_kernel!(backend)
    kernel!(
        phi_out, phi,
        phi_bc.west, phi_bc.east, phi_bc.south, phi_bc.north,
        ux_face, uy_face, is_solid,
        dt, inv(dx), inv(dy),
        bc.west, bc.east, bc.south, bc.north, Nx, Ny, scheme;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function fvfd_advect_upwind_2d!(
    phi_out, phi, phi_bc::FVFDFieldBC2D,
    ux_face, uy_face, geometry::FVFDGeometry2D, dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    return fvfd_advect_upwind_2d!(
        phi_out, phi, phi_bc,
        ux_face, uy_face, geometry.is_solid,
        geometry.patch.dx, geometry.patch.dy, geometry.bc, dt; sync, advection_scheme,
    )
end

function fvfd_advect_upwind_embedded_2d!(
    phi_out, phi, phi_bc::FVFDFieldBC2D,
    ux_face, uy_face, ux, uy,
    geometry::FVFDGeometry2D,
    ux_bc::FVFDFieldBC2D, uy_bc::FVFDFieldBC2D,
    dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    fvfd_cell_velocity_to_faces_embedded_2d!(
        ux_face, uy_face, ux, uy, geometry, ux_bc, uy_bc; sync=false,
    )
    fvfd_advect_upwind_2d!(
        phi_out, phi, phi_bc, ux_face, uy_face, geometry, dt; sync=false,
        advection_scheme,
    )
    sync && KernelAbstractions.synchronize(KernelAbstractions.get_backend(phi_out))
    return nothing
end

function fvfd_sym2_advect_upwind_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy,
    psixx_bc::FVFDFieldBC2D, psixy_bc::FVFDFieldBC2D, psiyy_bc::FVFDFieldBC2D,
    ux_face, uy_face, is_solid, dx, dy, bc::FVFDDomainBC2D, dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    @trace_enter :psi_sym2_advect
    fvfd_advect_upwind_2d!(
        psixx_out, psixx, psixx_bc, ux_face, uy_face, is_solid, dx, dy, bc, dt;
        sync=false, advection_scheme,
    )
    fvfd_advect_upwind_2d!(
        psixy_out, psixy, psixy_bc, ux_face, uy_face, is_solid, dx, dy, bc, dt;
        sync=false, advection_scheme,
    )
    fvfd_advect_upwind_2d!(
        psiyy_out, psiyy, psiyy_bc, ux_face, uy_face, is_solid, dx, dy, bc, dt;
        sync=false, advection_scheme,
    )
    sync && KernelAbstractions.synchronize(KernelAbstractions.get_backend(psixx_out))
    return nothing
end

function fvfd_sym2_advect_upwind_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy,
    psixx_bc::FVFDFieldBC2D, psixy_bc::FVFDFieldBC2D, psiyy_bc::FVFDFieldBC2D,
    ux_face, uy_face, geometry::FVFDGeometry2D, dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    return fvfd_sym2_advect_upwind_2d!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy,
        psixx_bc, psixy_bc, psiyy_bc,
        ux_face, uy_face, geometry.is_solid,
        geometry.patch.dx, geometry.patch.dy, geometry.bc, dt; sync, advection_scheme,
    )
end

function fvfd_sym2_advect_upwind_embedded_2d!(
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
    fvfd_cell_velocity_to_faces_embedded_2d!(
        ux_face, uy_face, ux, uy, geometry, ux_bc, uy_bc; sync=false,
    )
    fvfd_sym2_advect_upwind_2d!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy,
        psixx_bc, psixy_bc, psiyy_bc,
        ux_face, uy_face, geometry, dt; sync=false, advection_scheme,
    )
    sync && KernelAbstractions.synchronize(KernelAbstractions.get_backend(psixx_out))
    return nothing
end

