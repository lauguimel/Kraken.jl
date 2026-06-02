# =====================================================================
# M42 — MUSCL boundary relaxation (cylinder-side, two-pass split).
#
# See bench/viscoelastic_audit/M42_DESIGN.md for the full spec.
#
# Architecture: pass-1 launches the existing `:muscl_superbee` kernel
# (unchanged behaviour — full MUSCL in the bulk, whole-cell `:rusanov`
# demotion at the cylinder ring and open-wall band per M29b). After a
# KernelAbstractions.synchronize(backend) we launch pass-2 which
# overwrites `phi_out` ONLY at cylinder-band cells (`near_solid[i,j]` &&
# NOT on the open-wall band j ≤ 2 / j ≥ Ny−1 / i ≤ 2 / i ≥ Nx−1).
#
# At cylinder-band cells, pass-2 computes a per-face one-sided MUSCL
# reconstruction: on faces whose canonical 4-point stencil reads a solid
# or out-of-bounds value along the axis, the slope is set to zero
# (1-sided upwind, TVD by Sweby). On faces whose canonical stencil is
# fully fluid, full MUSCL Superbee is used. This is strictly better
# than M29b's all-or-nothing whole-cell demotion.
#
# Pass-2 reads only `phi` (lag-0) — never `phi_out` — so there is no
# cross-thread race with pass-1's writes. The synchronise between
# passes is required to make pass-1's `phi_out` writes visible if any
# later consumer reads `phi_out` (though pass-2 itself does not).
#
# Per design §1.3 + §3, the open-wall band is preserved as `:rusanov`
# (numerical diffusion at j=1 south wall is load-bearing for late-time
# stability per M29c-v2-BC-audit H-LATE-STIFF mechanism).
# =====================================================================

@inline function _is_cylinder_band_2d(is_solid, i, j, Nx, Ny)
    # Exclude open-wall band (preserved as :rusanov in pass-1).
    if i <= 2 || i >= Nx - 1 || j <= 2 || j >= Ny - 1
        return false
    end
    # Cylinder-band test : cross-shape stencil arms 1 and 2.
    return is_solid[i - 2, j] | is_solid[i - 1, j] |
           is_solid[i + 1, j] | is_solid[i + 2, j] |
           is_solid[i, j - 2] | is_solid[i, j - 1] |
           is_solid[i, j + 1] | is_solid[i, j + 2]
end

# Per-face one-sided MUSCL value computed at the cylinder band.
# On a broken axis (far-upwind solid or OOB) → 1-sided upwind (zero
# slope = full upwind, identical to `:rusanov` on that face).
# On a canonical-fluid axis → full MUSCL Superbee.
@inline function _fvfd_muscl_relax_rhs_2d(
    phi, west_phi, east_phi, south_phi, north_phi,
    ux_face, uy_face, is_solid, i, j, Nx, Ny, inv_dx, inv_dy,
    west_bc, east_bc, south_bc, north_bc,
)
    ue = ux_face[i + 1, j]
    uw = ux_face[i, j]
    vn = uy_face[i, j + 1]
    vs = uy_face[i, j]

    # East face: upwind direction depends on sign(ue).
    phie = if ue >= zero(ue)
        # ue >= 0: upwind = phi[i, j], downwind = phi[i+1, j],
        # far_upwind = phi[i-1, j]. Canonical needs i-1 fluid AND in-range.
        if i > 1 && !is_solid[i - 1, j] && !is_solid[i + 1, j]
            _fvfd_muscl_superbee_face_value_2d(phi[i - 1, j], phi[i, j], phi[i + 1, j])
        else
            phi[i, j]  # 1-sided upwind / zero-slope
        end
    else
        # ue < 0: upwind = phi[i+1, j], downwind = phi[i, j],
        # far_upwind = phi[i+2, j]. Canonical needs i+2 fluid AND in-range.
        if i + 2 <= Nx && !is_solid[i + 2, j] && !is_solid[i + 1, j]
            _fvfd_muscl_superbee_face_value_2d(phi[i + 2, j], phi[i + 1, j], phi[i, j])
        else
            phi[i + 1, j]
        end
    end

    # West face.
    phiw = if uw >= zero(uw)
        # uw >= 0: upwind = phi[i-1, j], downwind = phi[i, j],
        # far_upwind = phi[i-2, j]. Canonical needs i-2 fluid AND in-range.
        if i - 2 >= 1 && !is_solid[i - 2, j] && !is_solid[i - 1, j]
            _fvfd_muscl_superbee_face_value_2d(phi[i - 2, j], phi[i - 1, j], phi[i, j])
        else
            phi[i - 1, j]
        end
    else
        # uw < 0: upwind = phi[i, j], downwind = phi[i-1, j],
        # far_upwind = phi[i+1, j]. Canonical needs i+1 fluid.
        if i + 1 <= Nx && !is_solid[i + 1, j] && !is_solid[i - 1, j]
            _fvfd_muscl_superbee_face_value_2d(phi[i + 1, j], phi[i, j], phi[i - 1, j])
        else
            phi[i, j]
        end
    end

    # North face.
    phin = if vn >= zero(vn)
        # vn >= 0: upwind = phi[i, j], downwind = phi[i, j+1],
        # far_upwind = phi[i, j-1]. Canonical needs j-1 fluid AND in-range.
        if j > 1 && !is_solid[i, j - 1] && !is_solid[i, j + 1]
            _fvfd_muscl_superbee_face_value_2d(phi[i, j - 1], phi[i, j], phi[i, j + 1])
        else
            phi[i, j]
        end
    else
        # vn < 0: upwind = phi[i, j+1], downwind = phi[i, j],
        # far_upwind = phi[i, j+2]. Canonical needs j+2 fluid AND in-range.
        if j + 2 <= Ny && !is_solid[i, j + 2] && !is_solid[i, j + 1]
            _fvfd_muscl_superbee_face_value_2d(phi[i, j + 2], phi[i, j + 1], phi[i, j])
        else
            phi[i, j + 1]
        end
    end

    # South face.
    phis = if vs >= zero(vs)
        # vs >= 0: upwind = phi[i, j-1], downwind = phi[i, j],
        # far_upwind = phi[i, j-2]. Canonical needs j-2 fluid AND in-range.
        if j - 2 >= 1 && !is_solid[i, j - 2] && !is_solid[i, j - 1]
            _fvfd_muscl_superbee_face_value_2d(phi[i, j - 2], phi[i, j - 1], phi[i, j])
        else
            phi[i, j - 1]
        end
    else
        # vs < 0: upwind = phi[i, j], downwind = phi[i, j-1],
        # far_upwind = phi[i, j+1]. Canonical needs j+1 fluid.
        if j + 1 <= Ny && !is_solid[i, j + 1] && !is_solid[i, j - 1]
            _fvfd_muscl_superbee_face_value_2d(phi[i, j + 1], phi[i, j], phi[i, j - 1])
        else
            phi[i, j]
        end
    end

    flux_div = (ue * phie - uw * phiw) * inv_dx +
               (vn * phin - vs * phis) * inv_dy
    divu = (ue - uw) * inv_dx + (vn - vs) * inv_dy
    return -(flux_div - phi[i, j] * divu)
end

# Pass-2 kernel. At cylinder-band cells only, OVERWRITE phi_out with
# the one-sided MUSCL result computed from lag-0 phi.
# Pass-2 reads phi (not phi_out) — no cross-thread race with pass-1.
@kernel function fvfd_advect_muscl_relax_boundary_2d_kernel!(
    phi_out, @Const(phi),
    @Const(west_phi), @Const(east_phi), @Const(south_phi), @Const(north_phi),
    @Const(ux_face), @Const(uy_face), @Const(is_solid),
    dt, inv_dx, inv_dy,
    west_bc, east_bc, south_bc, north_bc, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if !is_solid[i, j] && _is_cylinder_band_2d(is_solid, i, j, Nx, Ny)
                rhs = _fvfd_muscl_relax_rhs_2d(
                    phi, west_phi, east_phi, south_phi, north_phi,
                    ux_face, uy_face, is_solid, i, j, Nx, Ny, inv_dx, inv_dy,
                    west_bc, east_bc, south_bc, north_bc,
                )
                phi_out[i, j] = phi[i, j] + dt * rhs
            end
            # else: preserve pass-1's value of phi_out (bulk MUSCL or
            # open-wall :rusanov fallback or solid-zero).
        end
    end
end

# Pass-2 launcher.
function fvfd_advect_muscl_relax_boundary_2d!(
    phi_out, phi, phi_bc::FVFDFieldBC2D,
    ux_face, uy_face, is_solid, dx, dy, bc::FVFDDomainBC2D, dt;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(phi_out)
    Nx, Ny = size(phi_out)
    fvfd_validate_field_bc_2d(phi_bc, Nx, Ny, bc; name=:phi_bc)
    kernel! = fvfd_advect_muscl_relax_boundary_2d_kernel!(backend)
    kernel!(
        phi_out, phi,
        phi_bc.west, phi_bc.east, phi_bc.south, phi_bc.north,
        ux_face, uy_face, is_solid,
        dt, inv(dx), inv(dy),
        bc.west, bc.east, bc.south, bc.north, Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

# Composite (pass-1 + sync + pass-2) launcher for `:muscl_superbee_relax`.
# Pass-1 = existing :muscl_superbee whole-cell fallback (unchanged).
# Pass-2 = cylinder-band overwrite with one-sided MUSCL.
function fvfd_advect_muscl_superbee_relax_2d!(
    phi_out, phi, phi_bc::FVFDFieldBC2D,
    ux_face, uy_face, is_solid, dx, dy, bc::FVFDDomainBC2D, dt;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(phi_out)
    # Pass-1: full :muscl_superbee (sync=true so phi_out is settled
    # before pass-2 reads phi).
    fvfd_advect_upwind_2d!(
        phi_out, phi, phi_bc,
        ux_face, uy_face, is_solid, dx, dy, bc, dt;
        sync=true,
        advection_scheme=:muscl_superbee,
    )
    # Pass-2: cylinder-band overwrite. Reads lag-0 phi, writes phi_out.
    fvfd_advect_muscl_relax_boundary_2d!(
        phi_out, phi, phi_bc,
        ux_face, uy_face, is_solid, dx, dy, bc, dt;
        sync=false,
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end
