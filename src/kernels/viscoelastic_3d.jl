using KernelAbstractions

# --- 3D Hermite stress source (D3Q19) for viscoelastic post-collision ---
#
# 3D port of `apply_hermite_source_2d!` (Liu et al. 2025, Eq. 25 with the
# (1 − s_plus/2) division for standard BGK/TRT consistency).
#
# T_q = -s_plus · (9/2) / (1 − s_plus/2) · w_q ·
#         [(c_qx² − cs²)·τxx + (c_qy² − cs²)·τyy + (c_qz² − cs²)·τzz +
#          2·c_qx·c_qy·τxy + 2·c_qx·c_qz·τxz + 2·c_qy·c_qz·τyz]
#
# with cs² = 1/3 and the standard D3Q19 weights wr = 1/3, wa = 1/18, we = 1/36.

@kernel function apply_hermite_source_3d_kernel!(f, @Const(is_solid), s_plus,
                                                   @Const(tau_p_xx),
                                                   @Const(tau_p_xy),
                                                   @Const(tau_p_xz),
                                                   @Const(tau_p_yy),
                                                   @Const(tau_p_yz),
                                                   @Const(tau_p_zz))
    i, j, k = @index(Global, NTuple)
    @inbounds if !is_solid[i, j, k]
        T = eltype(f)
        txx = tau_p_xx[i,j,k]; txy = tau_p_xy[i,j,k]; txz = tau_p_xz[i,j,k]
        tyy = tau_p_yy[i,j,k]; tyz = tau_p_yz[i,j,k]; tzz = tau_p_zz[i,j,k]
        pre = -s_plus * T(9.0/2.0) / (one(T) - s_plus / T(2))
        cs2 = T(1/3)
        wr = T(1/3); wa = T(1/18); we = T(1/36)
        a = one(T) - cs2          # = 2/3, common 2nd-Hermite axial weight

        # Common combinations used by multiple directions
        s_xyz = txx + tyy + tzz                       # diagonal trace
        diag_x = a*txx - cs2*tyy - cs2*tzz            # +x or −x: q=2/3
        diag_y = -cs2*txx + a*tyy - cs2*tzz           # +y or −y: q=4/5
        diag_z = -cs2*txx - cs2*tyy + a*tzz           # +z or −z: q=6/7
        # Edges share a 3-component diagonal (axial-pair sum) plus a sign
        # on the off-diagonal coupling.
        edge_xy_diag = a*(txx + tyy) - cs2*tzz
        edge_xz_diag = a*(txx + tzz) - cs2*tyy
        edge_yz_diag = a*(tyy + tzz) - cs2*txx

        # Rest population (q=1)
        T1  = pre * wr * (-cs2 * s_xyz)
        # Axial pairs share Hτ within a pair
        T_x = pre * wa * diag_x          # q=2 and q=3
        T_y = pre * wa * diag_y          # q=4, q=5
        T_z = pre * wa * diag_z          # q=6, q=7
        # xy edge group (q=8..11): pair (8, 11) has +2·τxy, pair (9, 10) has −2·τxy
        Txy_p = pre * we * (edge_xy_diag + T(2)*txy)   # q=8, q=11
        Txy_m = pre * we * (edge_xy_diag - T(2)*txy)   # q=9, q=10
        # xz edge group (q=12..15): pair (12, 15) +2·τxz, pair (13, 14) −2·τxz
        Txz_p = pre * we * (edge_xz_diag + T(2)*txz)   # q=12, q=15
        Txz_m = pre * we * (edge_xz_diag - T(2)*txz)   # q=13, q=14
        # yz edge group (q=16..19): pair (16, 19) +2·τyz, pair (17, 18) −2·τyz
        Tyz_p = pre * we * (edge_yz_diag + T(2)*tyz)   # q=16, q=19
        Tyz_m = pre * we * (edge_yz_diag - T(2)*tyz)   # q=17, q=18

        f[i,j,k,1]  += T1
        f[i,j,k,2]  += T_x;  f[i,j,k,3]  += T_x
        f[i,j,k,4]  += T_y;  f[i,j,k,5]  += T_y
        f[i,j,k,6]  += T_z;  f[i,j,k,7]  += T_z
        f[i,j,k,8]  += Txy_p; f[i,j,k,11] += Txy_p
        f[i,j,k,9]  += Txy_m; f[i,j,k,10] += Txy_m
        f[i,j,k,12] += Txz_p; f[i,j,k,15] += Txz_p
        f[i,j,k,13] += Txz_m; f[i,j,k,14] += Txz_m
        f[i,j,k,16] += Tyz_p; f[i,j,k,19] += Tyz_p
        f[i,j,k,17] += Tyz_m; f[i,j,k,18] += Tyz_m
    end
end

"""
    apply_hermite_source_3d!(f, is_solid, s_plus,
                              tau_p_xx, tau_p_xy, tau_p_xz,
                              tau_p_yy, tau_p_yz, tau_p_zz)

Post-collision injection of the 3D Hermite viscoelastic stress source on
D3Q19 populations. 3D port of `apply_hermite_source_2d!`.

Pass `s_plus = 1/(3ν+0.5)` for TRT, or `s_plus = ω` for BGK.
"""
function apply_hermite_source_3d!(f, is_solid, s_plus,
                                    tau_p_xx, tau_p_xy, tau_p_xz,
                                    tau_p_yy, tau_p_yz, tau_p_zz)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny, Nz = size(f, 1), size(f, 2), size(f, 3)
    T = eltype(f)
    kernel! = apply_hermite_source_3d_kernel!(backend)
    kernel!(f, is_solid, T(s_plus),
            tau_p_xx, tau_p_xy, tau_p_xz,
            tau_p_yy, tau_p_yz, tau_p_zz; ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

# ============================================================
# Polymeric stress update — 3D τ_p = G·(C - I) for Oldroyd-B
# ============================================================
#
# For Oldroyd-B and FENE-P (Peterlin: f(trC) = L²/(L²-trC)) only.
# Other models can dispatch their own `update_polymer_stress_3d!`.

@kernel function _update_polymer_stress_3d_oldroyd_kernel!(tau_xx, tau_xy, tau_xz,
                                                              tau_yy, tau_yz, tau_zz,
                                                              @Const(C_xx), @Const(C_xy),
                                                              @Const(C_xz), @Const(C_yy),
                                                              @Const(C_yz), @Const(C_zz),
                                                              G, L2_fene)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        T = eltype(tau_xx)
        cxx = C_xx[i,j,k]; cyy = C_yy[i,j,k]; czz = C_zz[i,j,k]
        cxy = C_xy[i,j,k]; cxz = C_xz[i,j,k]; cyz = C_yz[i,j,k]
        trC = cxx + cyy + czz
        fene = ifelse(L2_fene > zero(T),
                      L2_fene / max(L2_fene - trC, T(0.01)),
                      one(T))
        tau_xx[i,j,k] = G * fene * (cxx - one(T))
        tau_yy[i,j,k] = G * fene * (cyy - one(T))
        tau_zz[i,j,k] = G * fene * (czz - one(T))
        tau_xy[i,j,k] = G * fene * cxy
        tau_xz[i,j,k] = G * fene * cxz
        tau_yz[i,j,k] = G * fene * cyz
    end
end

"""
    update_polymer_stress_3d!(tau_p_xx, tau_p_xy, tau_p_xz,
                                tau_p_yy, tau_p_yz, tau_p_zz,
                                C_xx, C_xy, C_xz, C_yy, C_yz, C_zz, model)

Compute the 3D polymeric stress from the conformation tensor and an
`AbstractPolymerModel`. Currently implemented for `OldroydB` (and
`LogConfOldroydB` after `psi_to_C` reconstruction). Mirrors the 2D
dispatch in `viscoelastic_spec.jl::update_polymer_stress!`.
"""
function update_polymer_stress_3d!(tau_xx, tau_xy, tau_xz,
                                     tau_yy, tau_yz, tau_zz,
                                     C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
                                     model::OldroydB)
    backend = KernelAbstractions.get_backend(tau_xx)
    Nx, Ny, Nz = size(tau_xx)
    FT = eltype(tau_xx)
    kernel! = _update_polymer_stress_3d_oldroyd_kernel!(backend)
    kernel!(tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz,
            C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
            FT(polymer_modulus(model)), FT(0.0); ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

function update_polymer_stress_3d!(tau_xx, tau_xy, tau_xz,
                                     tau_yy, tau_yz, tau_zz,
                                     C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
                                     model::LogConfOldroydB)
    backend = KernelAbstractions.get_backend(tau_xx)
    Nx, Ny, Nz = size(tau_xx)
    FT = eltype(tau_xx)
    kernel! = _update_polymer_stress_3d_oldroyd_kernel!(backend)
    kernel!(tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz,
            C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
            FT(polymer_modulus(model)), FT(0.0); ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end
