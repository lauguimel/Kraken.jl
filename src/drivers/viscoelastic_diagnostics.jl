"""
    conformation_field_diagnostics_2d(C_xx, C_xy, C_yy, ρ, ux, uy, is_solid)

Host-side diagnostic for 2D conformation fields. It is intentionally separate
from the production kernels: use it as a canary in validation/debug runs to
localize loss of positive definiteness, non-finite values, and large numerical
velocity divergence feeding the constitutive source.
"""
function conformation_field_diagnostics_2d(C_xx, C_xy, C_yy, ρ, ux, uy, is_solid)
    Cxx = Array(C_xx)
    Cxy = Array(C_xy)
    Cyy = Array(C_yy)
    ρh = Array(ρ)
    uxh = Array(ux)
    uyh = Array(uy)
    solid = Array(is_solid)
    Nx, Ny = size(Cxx)

    finite = true
    n_fluid = 0
    min_eig = Inf
    max_abs_C = 0.0
    max_abs_u = 0.0
    max_abs_divu = 0.0
    max_strain_eig = -Inf
    min_i = 0
    min_j = 0
    maxC_i = 0
    maxC_j = 0
    maxDiv_i = 0
    maxDiv_j = 0
    maxStrain_i = 0
    maxStrain_j = 0
    first_bad_i = 0
    first_bad_j = 0

    @inbounds for j in 1:Ny, i in 1:Nx
        solid[i, j] && continue
        n_fluid += 1

        cxx = Float64(Cxx[i, j])
        cxy = Float64(Cxy[i, j])
        cyy = Float64(Cyy[i, j])
        rhoi = Float64(ρh[i, j])
        uxi = Float64(uxh[i, j])
        uyi = Float64(uyh[i, j])

        local_finite = isfinite(cxx) && isfinite(cxy) && isfinite(cyy) &&
                       isfinite(rhoi) && isfinite(uxi) && isfinite(uyi)
        if !local_finite && finite
            first_bad_i = i
            first_bad_j = j
        end
        finite &= local_finite

        tr = cxx + cyy
        diff = cxx - cyy
        disc = sqrt(diff * diff + 4.0 * cxy * cxy)
        eig = 0.5 * (tr - disc)
        if eig < min_eig || min_i == 0
            min_eig = eig
            min_i = i
            min_j = j
        end

        abs_C = max(abs(cxx), abs(cxy), abs(cyy))
        if abs_C > max_abs_C
            max_abs_C = abs_C
            maxC_i = i
            maxC_j = j
        end

        max_abs_u = max(max_abs_u, hypot(uxi, uyi))
        dudx = _wall_aware_dx_2d(uxh, solid, i, j, Nx, Float64)
        dudy = _wall_aware_dy_2d(uxh, solid, i, j, Ny, Float64)
        dvdx = _wall_aware_dx_2d(uyh, solid, i, j, Nx, Float64)
        dvdy = _wall_aware_dy_2d(uyh, solid, i, j, Ny, Float64)
        abs_divu = abs(dudx + dvdy)
        if abs_divu > max_abs_divu
            max_abs_divu = abs_divu
            maxDiv_i = i
            maxDiv_j = j
        end

        strain_trace = dudx + dvdy
        strain_diff = dudx - dvdy
        strain_offdiag = 0.5 * (dudy + dvdx)
        strain_disc = sqrt(strain_diff * strain_diff + 4.0 * strain_offdiag * strain_offdiag)
        strain_eig = 0.5 * (strain_trace + strain_disc)
        if strain_eig > max_strain_eig || maxStrain_i == 0
            max_strain_eig = strain_eig
            maxStrain_i = i
            maxStrain_j = j
        end
    end

    min_dudx = min_i == 0 ? NaN : _wall_aware_dx_2d(uxh, solid, min_i, min_j, Nx, Float64)
    min_dudy = min_i == 0 ? NaN : _wall_aware_dy_2d(uxh, solid, min_i, min_j, Ny, Float64)
    min_dvdx = min_i == 0 ? NaN : _wall_aware_dx_2d(uyh, solid, min_i, min_j, Nx, Float64)
    min_dvdy = min_i == 0 ? NaN : _wall_aware_dy_2d(uyh, solid, min_i, min_j, Ny, Float64)
    min_strain_trace = min_dudx + min_dvdy
    min_strain_diff = min_dudx - min_dvdy
    min_strain_offdiag = 0.5 * (min_dudy + min_dvdx)
    min_strain_disc = sqrt(min_strain_diff * min_strain_diff +
                           4.0 * min_strain_offdiag * min_strain_offdiag)
    min_strain_eig = 0.5 * (min_strain_trace + min_strain_disc)

    return (;
        finite,
        n_fluid,
        min_eig,
        min_i,
        min_j,
        min_C_xx = min_i == 0 ? NaN : Float64(Cxx[min_i, min_j]),
        min_C_xy = min_i == 0 ? NaN : Float64(Cxy[min_i, min_j]),
        min_C_yy = min_i == 0 ? NaN : Float64(Cyy[min_i, min_j]),
        max_abs_C,
        maxC_i,
        maxC_j,
        max_abs_u,
        max_abs_divu,
        maxDiv_i,
        maxDiv_j,
        max_strain_eig,
        maxStrain_i,
        maxStrain_j,
        min_dudx,
        min_dudy,
        min_dvdx,
        min_dvdy,
        min_strain_eig,
        first_bad_i,
        first_bad_j,
    )
end
