@kernel function _bc_west_zh_velocity_3d!(f_out, f_in, profile, s_p, s_m)
    jm1, km1 = @index(Global, NTuple); j = jm1 + 1; k = km1 + 1
    T = eltype(f_out)
    @inbounds begin
        fp1  = f_in[1, j,     k,     1]
        fp4  = f_in[1, j - 1, k,     4]
        fp5  = f_in[1, j + 1, k,     5]
        fp6  = f_in[1, j,     k - 1, 6]
        fp7  = f_in[1, j,     k + 1, 7]
        fp16 = f_in[1, j - 1, k - 1, 16]
        fp17 = f_in[1, j + 1, k - 1, 17]
        fp18 = f_in[1, j - 1, k + 1, 18]
        fp19 = f_in[1, j + 1, k + 1, 19]
        fp3  = f_in[2, j,     k,     3]
        fp9  = f_in[2, j - 1, k,     9]
        fp11 = f_in[2, j + 1, k,     11]
        fp13 = f_in[2, j,     k - 1, 13]
        fp15 = f_in[2, j,     k + 1, 15]
        # profile can be a vector indexed by j, or a matrix (Ny, Nz). Try both.
        u_n  = length(size(profile)) == 1 ? T(profile[j]) : T(profile[j, k])
        sum_par = fp1 + fp4 + fp5 + fp6 + fp7 + fp16 + fp17 + fp18 + fp19
        sum_out = fp3 + fp9 + fp11 + fp13 + fp15
        ρ_w  = (sum_par + T(2)*sum_out) / (one(T) - u_n)
        fp2  = fp3 + T(1/3) * ρ_w * u_n
        tang1_diff = fp4 - fp5
        tang2_diff = fp6 - fp7
        fp8  = fp11 - T(0.5)*tang1_diff + T(1/6)*ρ_w*u_n
        fp10 = fp9  + T(0.5)*tang1_diff + T(1/6)*ρ_w*u_n
        fp12 = fp15 - T(0.5)*tang2_diff + T(1/6)*ρ_w*u_n
        fp14 = fp13 + T(0.5)*tang2_diff + T(1/6)*ρ_w*u_n
        F = _trt_collide_local_3d(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8,
                                    fp9, fp10, fp11, fp12, fp13, fp14,
                                    fp15, fp16, fp17, fp18, fp19, s_p, s_m)
        f_out[1, j, k, 1]  = F[1];  f_out[1, j, k, 2]  = F[2]
        f_out[1, j, k, 3]  = F[3];  f_out[1, j, k, 4]  = F[4]
        f_out[1, j, k, 5]  = F[5];  f_out[1, j, k, 6]  = F[6]
        f_out[1, j, k, 7]  = F[7];  f_out[1, j, k, 8]  = F[8]
        f_out[1, j, k, 9]  = F[9];  f_out[1, j, k, 10] = F[10]
        f_out[1, j, k, 11] = F[11]; f_out[1, j, k, 12] = F[12]
        f_out[1, j, k, 13] = F[13]; f_out[1, j, k, 14] = F[14]
        f_out[1, j, k, 15] = F[15]; f_out[1, j, k, 16] = F[16]
        f_out[1, j, k, 17] = F[17]; f_out[1, j, k, 18] = F[18]
        f_out[1, j, k, 19] = F[19]
    end
end

@kernel function _bc_east_zh_pressure_3d!(f_out, f_in, Nx, ρ_out, s_p, s_m)
    jm1, km1 = @index(Global, NTuple); j = jm1 + 1; k = km1 + 1
    T = eltype(f_out)
    @inbounds begin
        fp1  = f_in[Nx,     j,     k,     1]
        fp4  = f_in[Nx,     j - 1, k,     4]
        fp5  = f_in[Nx,     j + 1, k,     5]
        fp6  = f_in[Nx,     j,     k - 1, 6]
        fp7  = f_in[Nx,     j,     k + 1, 7]
        fp16 = f_in[Nx,     j - 1, k - 1, 16]
        fp17 = f_in[Nx,     j + 1, k - 1, 17]
        fp18 = f_in[Nx,     j - 1, k + 1, 18]
        fp19 = f_in[Nx,     j + 1, k + 1, 19]
        fp2  = f_in[Nx - 1, j,     k,     2]
        fp8  = f_in[Nx - 1, j - 1, k,     8]
        fp10 = f_in[Nx - 1, j + 1, k,     10]
        fp12 = f_in[Nx - 1, j,     k - 1, 12]
        fp14 = f_in[Nx - 1, j,     k + 1, 14]
        sum_par = fp1 + fp4 + fp5 + fp6 + fp7 + fp16 + fp17 + fp18 + fp19
        sum_out = fp2 + fp8 + fp10 + fp12 + fp14
        u_n     = -(one(T) - (sum_par + T(2)*sum_out) / ρ_out)
        fp3  = fp2 - T(1/3) * ρ_out * u_n
        tang1_diff = fp4 - fp5
        tang2_diff = fp6 - fp7
        fp9  = fp10 - T(0.5)*tang1_diff - T(1/6)*ρ_out*u_n
        fp11 = fp8  + T(0.5)*tang1_diff - T(1/6)*ρ_out*u_n
        fp13 = fp14 - T(0.5)*tang2_diff - T(1/6)*ρ_out*u_n
        fp15 = fp12 + T(0.5)*tang2_diff - T(1/6)*ρ_out*u_n
        F = _trt_collide_local_3d(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8,
                                    fp9, fp10, fp11, fp12, fp13, fp14,
                                    fp15, fp16, fp17, fp18, fp19, s_p, s_m)
        f_out[Nx, j, k, 1]  = F[1];  f_out[Nx, j, k, 2]  = F[2]
        f_out[Nx, j, k, 3]  = F[3];  f_out[Nx, j, k, 4]  = F[4]
        f_out[Nx, j, k, 5]  = F[5];  f_out[Nx, j, k, 6]  = F[6]
        f_out[Nx, j, k, 7]  = F[7];  f_out[Nx, j, k, 8]  = F[8]
        f_out[Nx, j, k, 9]  = F[9];  f_out[Nx, j, k, 10] = F[10]
        f_out[Nx, j, k, 11] = F[11]; f_out[Nx, j, k, 12] = F[12]
        f_out[Nx, j, k, 13] = F[13]; f_out[Nx, j, k, 14] = F[14]
        f_out[Nx, j, k, 15] = F[15]; f_out[Nx, j, k, 16] = F[16]
        f_out[Nx, j, k, 17] = F[17]; f_out[Nx, j, k, 18] = F[18]
        f_out[Nx, j, k, 19] = F[19]
    end
end

@inline function _apply_bc_3d_west!(backend, f_out, f_in, ::HalfwayBB,
                                     s_p, s_m, Nx, Ny, Nz) end
@inline function _apply_bc_3d_west!(backend, f_out, f_in, bc::ZouHeVelocity,
                                     s_p, s_m, Nx, Ny, Nz)
    _bc_west_zh_velocity_3d!(backend)(f_out, f_in, bc.profile, s_p, s_m;
                                       ndrange=(Ny - 2, Nz - 2))
end
@inline function _apply_bc_3d_east!(backend, f_out, f_in, ::HalfwayBB,
                                     s_p, s_m, Nx, Ny, Nz) end
@inline function _apply_bc_3d_east!(backend, f_out, f_in, bc::ZouHePressure,
                                     s_p, s_m, Nx, Ny, Nz)
    _bc_east_zh_pressure_3d!(backend)(f_out, f_in, Nx, eltype(f_out)(bc.ρ_out),
                                       s_p, s_m; ndrange=(Ny - 2, Nz - 2))
end

# ----------------------------------------------------------------------
# 3D transverse-face halfway-BB kernels (south/north/bottom/top).
#
# SLBM clamps at non-periodic boundaries (`PullSLBM_3D` calls
# `_wrap_or_clamp(j-1, Ny, false) = 1`), which is NOT halfway-BB. These
# kernels apply the explicit halfway-BB swap on `f_out` AFTER the SLBM
# step, restoring no-slip at the four transverse faces.
# ----------------------------------------------------------------------

# South wall (j=1, normal +y). Pops with cy>0: 4, 8, 9, 16, 18.
# Full i,k coverage (including face corners/edges) — SLBM clamps at all
# edges and corners, so the entire face row needs the BB swap.
@kernel function _bc_south_halfwaybb_3d!(f_out, @Const(f_in), Ny)
    i, k = @index(Global, NTuple)
    @inbounds begin
        f_out[i, 1, k, 4]  = f_in[i, 1, k, 5]
        f_out[i, 1, k, 8]  = f_in[i, 1, k, 11]
        f_out[i, 1, k, 9]  = f_in[i, 1, k, 10]
        f_out[i, 1, k, 16] = f_in[i, 1, k, 19]
        f_out[i, 1, k, 18] = f_in[i, 1, k, 17]
    end
end

# North wall (j=Ny, normal -y). Pops with cy<0: 5, 11, 10, 19, 17.
@kernel function _bc_north_halfwaybb_3d!(f_out, @Const(f_in), Ny)
    i, k = @index(Global, NTuple)
    @inbounds begin
        f_out[i, Ny, k, 5]  = f_in[i, Ny, k, 4]
        f_out[i, Ny, k, 11] = f_in[i, Ny, k, 8]
        f_out[i, Ny, k, 10] = f_in[i, Ny, k, 9]
        f_out[i, Ny, k, 19] = f_in[i, Ny, k, 16]
        f_out[i, Ny, k, 17] = f_in[i, Ny, k, 18]
    end
end

# Bottom wall (k=1, normal +z). Pops with cz>0: 6, 12, 13, 16, 17.
@kernel function _bc_bottom_halfwaybb_3d!(f_out, @Const(f_in), Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        f_out[i, j, 1, 6]  = f_in[i, j, 1, 7]
        f_out[i, j, 1, 12] = f_in[i, j, 1, 15]
        f_out[i, j, 1, 13] = f_in[i, j, 1, 14]
        f_out[i, j, 1, 16] = f_in[i, j, 1, 19]
        f_out[i, j, 1, 17] = f_in[i, j, 1, 18]
    end
end

# Top wall (k=Nz, normal -z). Pops with cz<0: 7, 15, 14, 19, 18.
@kernel function _bc_top_halfwaybb_3d!(f_out, @Const(f_in), Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        f_out[i, j, Nz, 7]  = f_in[i, j, Nz, 6]
        f_out[i, j, Nz, 15] = f_in[i, j, Nz, 12]
        f_out[i, j, Nz, 14] = f_in[i, j, Nz, 13]
        f_out[i, j, Nz, 19] = f_in[i, j, Nz, 16]
        f_out[i, j, Nz, 18] = f_in[i, j, Nz, 17]
    end
end

# Full-face ndrange so the edges/corners of each transverse face also
# receive the BB swap. West/east Zou-He kernels run BEFORE these and
# do use (Ny-2, Nz-2), so the inlet/outlet corners are overridden by
# halfway-BB; this is the standard pragmatic closure when a Zou-He face
# meets a no-slip wall (matches the 2D behaviour).
@inline function _apply_bc_3d_south!(backend, f_out, f_in, ::HalfwayBB,
                                      s_p, s_m, Nx, Ny, Nz)
    _bc_south_halfwaybb_3d!(backend)(f_out, f_in, Ny; ndrange=(Nx, Nz))
end
@inline function _apply_bc_3d_north!(backend, f_out, f_in, ::HalfwayBB,
                                      s_p, s_m, Nx, Ny, Nz)
    _bc_north_halfwaybb_3d!(backend)(f_out, f_in, Ny; ndrange=(Nx, Nz))
end
@inline function _apply_bc_3d_bottom!(backend, f_out, f_in, ::HalfwayBB,
                                       s_p, s_m, Nx, Ny, Nz)
    _bc_bottom_halfwaybb_3d!(backend)(f_out, f_in, Nz; ndrange=(Nx, Ny))
end
@inline function _apply_bc_3d_top!(backend, f_out, f_in, ::HalfwayBB,
                                    s_p, s_m, Nx, Ny, Nz)
    _bc_top_halfwaybb_3d!(backend)(f_out, f_in, Nz; ndrange=(Nx, Ny))
end

# ----------------------------------------------------------------------
# 3D local-τ Zou-He kernels (per-cell s_plus[i,j,k], s_minus[i,j,k]).
# Mirror of the 2D _local_ variants; needed for SLBM on stretched
# meshes where τ varies across the inlet/outlet face.
# ----------------------------------------------------------------------

@kernel function _bc_west_zh_velocity_local_3d!(f_out, f_in, profile,
                                                  sp_field, sm_field)
    jm1, km1 = @index(Global, NTuple); j = jm1 + 1; k = km1 + 1
    T = eltype(f_out)
    @inbounds begin
        fp1  = f_in[1, j,     k,     1]
        fp4  = f_in[1, j - 1, k,     4]
        fp5  = f_in[1, j + 1, k,     5]
        fp6  = f_in[1, j,     k - 1, 6]
        fp7  = f_in[1, j,     k + 1, 7]
        fp16 = f_in[1, j - 1, k - 1, 16]
        fp17 = f_in[1, j + 1, k - 1, 17]
        fp18 = f_in[1, j - 1, k + 1, 18]
        fp19 = f_in[1, j + 1, k + 1, 19]
        fp3  = f_in[2, j,     k,     3]
        fp9  = f_in[2, j - 1, k,     9]
        fp11 = f_in[2, j + 1, k,     11]
        fp13 = f_in[2, j,     k - 1, 13]
        fp15 = f_in[2, j,     k + 1, 15]
        u_n  = length(size(profile)) == 1 ? T(profile[j]) : T(profile[j, k])
        sum_par = fp1 + fp4 + fp5 + fp6 + fp7 + fp16 + fp17 + fp18 + fp19
        sum_out = fp3 + fp9 + fp11 + fp13 + fp15
        ρ_w  = (sum_par + T(2)*sum_out) / (one(T) - u_n)
        fp2  = fp3 + T(1/3) * ρ_w * u_n
        tang1_diff = fp4 - fp5
        tang2_diff = fp6 - fp7
        fp8  = fp11 - T(0.5)*tang1_diff + T(1/6)*ρ_w*u_n
        fp10 = fp9  + T(0.5)*tang1_diff + T(1/6)*ρ_w*u_n
        fp12 = fp15 - T(0.5)*tang2_diff + T(1/6)*ρ_w*u_n
        fp14 = fp13 + T(0.5)*tang2_diff + T(1/6)*ρ_w*u_n
        s_p = sp_field[1, j, k]; s_m = sm_field[1, j, k]
        F = _trt_collide_local_3d(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8,
                                    fp9, fp10, fp11, fp12, fp13, fp14,
                                    fp15, fp16, fp17, fp18, fp19, s_p, s_m)
        f_out[1, j, k, 1]  = F[1];  f_out[1, j, k, 2]  = F[2]
        f_out[1, j, k, 3]  = F[3];  f_out[1, j, k, 4]  = F[4]
        f_out[1, j, k, 5]  = F[5];  f_out[1, j, k, 6]  = F[6]
        f_out[1, j, k, 7]  = F[7];  f_out[1, j, k, 8]  = F[8]
        f_out[1, j, k, 9]  = F[9];  f_out[1, j, k, 10] = F[10]
        f_out[1, j, k, 11] = F[11]; f_out[1, j, k, 12] = F[12]
        f_out[1, j, k, 13] = F[13]; f_out[1, j, k, 14] = F[14]
        f_out[1, j, k, 15] = F[15]; f_out[1, j, k, 16] = F[16]
        f_out[1, j, k, 17] = F[17]; f_out[1, j, k, 18] = F[18]
        f_out[1, j, k, 19] = F[19]
    end
end

@kernel function _bc_east_zh_pressure_local_3d!(f_out, f_in, Nx, ρ_out,
                                                  sp_field, sm_field)
    jm1, km1 = @index(Global, NTuple); j = jm1 + 1; k = km1 + 1
    T = eltype(f_out)
    @inbounds begin
        fp1  = f_in[Nx,     j,     k,     1]
        fp4  = f_in[Nx,     j - 1, k,     4]
        fp5  = f_in[Nx,     j + 1, k,     5]
        fp6  = f_in[Nx,     j,     k - 1, 6]
        fp7  = f_in[Nx,     j,     k + 1, 7]
        fp16 = f_in[Nx,     j - 1, k - 1, 16]
        fp17 = f_in[Nx,     j + 1, k - 1, 17]
        fp18 = f_in[Nx,     j - 1, k + 1, 18]
        fp19 = f_in[Nx,     j + 1, k + 1, 19]
        fp2  = f_in[Nx - 1, j,     k,     2]
        fp8  = f_in[Nx - 1, j - 1, k,     8]
        fp10 = f_in[Nx - 1, j + 1, k,     10]
        fp12 = f_in[Nx - 1, j,     k - 1, 12]
        fp14 = f_in[Nx - 1, j,     k + 1, 14]
        sum_par = fp1 + fp4 + fp5 + fp6 + fp7 + fp16 + fp17 + fp18 + fp19
        sum_out = fp2 + fp8 + fp10 + fp12 + fp14
        u_n     = -(one(T) - (sum_par + T(2)*sum_out) / ρ_out)
        fp3  = fp2 - T(1/3) * ρ_out * u_n
        tang1_diff = fp4 - fp5
        tang2_diff = fp6 - fp7
        fp9  = fp10 - T(0.5)*tang1_diff - T(1/6)*ρ_out*u_n
        fp11 = fp8  + T(0.5)*tang1_diff - T(1/6)*ρ_out*u_n
        fp13 = fp14 - T(0.5)*tang2_diff - T(1/6)*ρ_out*u_n
        fp15 = fp12 + T(0.5)*tang2_diff - T(1/6)*ρ_out*u_n
        s_p = sp_field[Nx, j, k]; s_m = sm_field[Nx, j, k]
        F = _trt_collide_local_3d(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8,
                                    fp9, fp10, fp11, fp12, fp13, fp14,
                                    fp15, fp16, fp17, fp18, fp19, s_p, s_m)
        f_out[Nx, j, k, 1]  = F[1];  f_out[Nx, j, k, 2]  = F[2]
        f_out[Nx, j, k, 3]  = F[3];  f_out[Nx, j, k, 4]  = F[4]
        f_out[Nx, j, k, 5]  = F[5];  f_out[Nx, j, k, 6]  = F[6]
        f_out[Nx, j, k, 7]  = F[7];  f_out[Nx, j, k, 8]  = F[8]
        f_out[Nx, j, k, 9]  = F[9];  f_out[Nx, j, k, 10] = F[10]
        f_out[Nx, j, k, 11] = F[11]; f_out[Nx, j, k, 12] = F[12]
        f_out[Nx, j, k, 13] = F[13]; f_out[Nx, j, k, 14] = F[14]
        f_out[Nx, j, k, 15] = F[15]; f_out[Nx, j, k, 16] = F[16]
        f_out[Nx, j, k, 17] = F[17]; f_out[Nx, j, k, 18] = F[18]
        f_out[Nx, j, k, 19] = F[19]
    end
end

@inline function _apply_bc_3d_west_local!(backend, f_out, f_in, ::HalfwayBB,
                                            sp_field, sm_field, Nx, Ny, Nz) end
@inline function _apply_bc_3d_west_local!(backend, f_out, f_in, bc::ZouHeVelocity,
                                            sp_field, sm_field, Nx, Ny, Nz)
    _bc_west_zh_velocity_local_3d!(backend)(f_out, f_in, bc.profile,
                                              sp_field, sm_field;
                                              ndrange=(Ny - 2, Nz - 2))
end
@inline function _apply_bc_3d_east_local!(backend, f_out, f_in, ::HalfwayBB,
                                            sp_field, sm_field, Nx, Ny, Nz) end
@inline function _apply_bc_3d_east_local!(backend, f_out, f_in, bc::ZouHePressure,
                                            sp_field, sm_field, Nx, Ny, Nz)
    _bc_east_zh_pressure_local_3d!(backend)(f_out, f_in, Nx,
                                              eltype(f_out)(bc.ρ_out),
                                              sp_field, sm_field;
                                              ndrange=(Ny - 2, Nz - 2))
end

"""
    apply_bc_rebuild_3d!(f_out, f_in, bcspec, ν, Nx, Ny, Nz;
                          sp_field=nothing, sm_field=nothing,
                          apply_transverse=false)

3D analog of `apply_bc_rebuild_2d!`. Dispatches per-face on the BC type
in `bcspec::BCSpec3D` and applies them to `f_out`.

- `west`/`east` : ZouHeVelocity / ZouHePressure (or HalfwayBB no-op)
- `south`/`north`/`bottom`/`top` : HalfwayBB explicit swaps when
  `apply_transverse=true`; **default false** for backwards-compat with
  `fused_trt_libb_v2_step_3d!`, whose `PullHalfwayBB_3D` brick already
  performs the transverse halfway-BB swap inside the streaming kernel.
  SLBM-based callers (`PullSLBM_3D`) clamp at non-periodic edges and
  must set `apply_transverse=true`.

If `sp_field`/`sm_field` (3D arrays) are supplied, the west/east TRT
collisions use per-cell relaxation rates — required for SLBM on
stretched meshes where τ varies per cell. Transverse halfway-BB faces
are pure population swaps and do not depend on τ.
"""
function apply_bc_rebuild_3d!(f_out, f_in, bcspec::BCSpec3D, ν::Real,
                                Nx::Int, Ny::Int, Nz::Int;
                                sp_field=nothing, sm_field=nothing,
                                apply_transverse::Bool=false)
    backend = KernelAbstractions.get_backend(f_out)
    T = eltype(f_out)
    s_p_r, s_m_r = trt_rates(ν; Λ=3/16)
    s_p = T(s_p_r); s_m = T(s_m_r)
    if isnothing(sp_field)
        _apply_bc_3d_west!(backend, f_out, f_in, bcspec.west, s_p, s_m, Nx, Ny, Nz)
        _apply_bc_3d_east!(backend, f_out, f_in, bcspec.east, s_p, s_m, Nx, Ny, Nz)
    else
        _apply_bc_3d_west_local!(backend, f_out, f_in, bcspec.west,
                                  sp_field, sm_field, Nx, Ny, Nz)
        _apply_bc_3d_east_local!(backend, f_out, f_in, bcspec.east,
                                  sp_field, sm_field, Nx, Ny, Nz)
    end
    if apply_transverse
        _apply_bc_3d_south!(backend, f_out, f_in,  bcspec.south,  s_p, s_m, Nx, Ny, Nz)
        _apply_bc_3d_north!(backend, f_out, f_in,  bcspec.north,  s_p, s_m, Nx, Ny, Nz)
        _apply_bc_3d_bottom!(backend, f_out, f_in, bcspec.bottom, s_p, s_m, Nx, Ny, Nz)
        _apply_bc_3d_top!(backend, f_out, f_in,    bcspec.top,    s_p, s_m, Nx, Ny, Nz)
    end
    return nothing
end
