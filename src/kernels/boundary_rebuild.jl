using KernelAbstractions

# =====================================================================
# Local TRT collide helpers reused by the face BC kernels below.
# =====================================================================

@inline function _trt_collide_local(f1::T, f2::T, f3::T, f4::T, f5::T,
                                     f6::T, f7::T, f8::T, f9::T,
                                     s_p::T, s_m::T) where {T}
    ρ  = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
    ux = (f2 - f4 + f6 - f8 + f9 - f7) / ρ
    uy = (f3 - f5 + f6 - f8 + f7 - f9) / ρ
    usq = ux * ux + uy * uy
    fe1 = feq_2d(Val(1), ρ, ux, uy, usq)
    fe2 = feq_2d(Val(2), ρ, ux, uy, usq)
    fe3 = feq_2d(Val(3), ρ, ux, uy, usq)
    fe4 = feq_2d(Val(4), ρ, ux, uy, usq)
    fe5 = feq_2d(Val(5), ρ, ux, uy, usq)
    fe6 = feq_2d(Val(6), ρ, ux, uy, usq)
    fe7 = feq_2d(Val(7), ρ, ux, uy, usq)
    fe8 = feq_2d(Val(8), ρ, ux, uy, usq)
    fe9 = feq_2d(Val(9), ρ, ux, uy, usq)
    a = (s_p + s_m) * T(0.5)
    b = (s_p - s_m) * T(0.5)
    return (
        f1 - s_p * (f1 - fe1),
        f2 - a * (f2 - fe2) - b * (f4 - fe4),
        f3 - a * (f3 - fe3) - b * (f5 - fe5),
        f4 - a * (f4 - fe4) - b * (f2 - fe2),
        f5 - a * (f5 - fe5) - b * (f3 - fe3),
        f6 - a * (f6 - fe6) - b * (f8 - fe8),
        f7 - a * (f7 - fe7) - b * (f9 - fe9),
        f8 - a * (f8 - fe8) - b * (f6 - fe6),
        f9 - a * (f9 - fe9) - b * (f7 - fe7),
    )
end

@inline function _trt_collide_local_3d(f1::T, f2::T, f3::T, f4::T, f5::T,
                                        f6::T, f7::T, f8::T, f9::T,
                                        f10::T, f11::T, f12::T, f13::T, f14::T,
                                        f15::T, f16::T, f17::T, f18::T, f19::T,
                                        s_p::T, s_m::T) where {T}
    ρ  = f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+f17+f18+f19
    ux = ((f2 - f3) + (f8 - f11) + (f10 - f9) + (f12 - f15) + (f14 - f13)) / ρ
    uy = ((f4 - f5) + (f8 - f11) + (f9 - f10) + (f16 - f19) + (f18 - f17)) / ρ
    uz = ((f6 - f7) + (f12 - f15) + (f13 - f14) + (f16 - f19) + (f17 - f18)) / ρ
    a = (s_p + s_m) * T(0.5)
    b = (s_p - s_m) * T(0.5)
    usq = ux * ux + uy * uy + uz * uz
    fe1  = feq_3d(Val(1),  ρ, ux, uy, uz, usq)
    fe2  = feq_3d(Val(2),  ρ, ux, uy, uz, usq)
    fe3  = feq_3d(Val(3),  ρ, ux, uy, uz, usq)
    fe4  = feq_3d(Val(4),  ρ, ux, uy, uz, usq)
    fe5  = feq_3d(Val(5),  ρ, ux, uy, uz, usq)
    fe6  = feq_3d(Val(6),  ρ, ux, uy, uz, usq)
    fe7  = feq_3d(Val(7),  ρ, ux, uy, uz, usq)
    fe8  = feq_3d(Val(8),  ρ, ux, uy, uz, usq)
    fe9  = feq_3d(Val(9),  ρ, ux, uy, uz, usq)
    fe10 = feq_3d(Val(10), ρ, ux, uy, uz, usq)
    fe11 = feq_3d(Val(11), ρ, ux, uy, uz, usq)
    fe12 = feq_3d(Val(12), ρ, ux, uy, uz, usq)
    fe13 = feq_3d(Val(13), ρ, ux, uy, uz, usq)
    fe14 = feq_3d(Val(14), ρ, ux, uy, uz, usq)
    fe15 = feq_3d(Val(15), ρ, ux, uy, uz, usq)
    fe16 = feq_3d(Val(16), ρ, ux, uy, uz, usq)
    fe17 = feq_3d(Val(17), ρ, ux, uy, uz, usq)
    fe18 = feq_3d(Val(18), ρ, ux, uy, uz, usq)
    fe19 = feq_3d(Val(19), ρ, ux, uy, uz, usq)
    F1  = f1  - s_p * (f1  - fe1)
    F2  = f2  - a*(f2  - fe2)  - b*(f3  - fe3)
    F3  = f3  - a*(f3  - fe3)  - b*(f2  - fe2)
    F4  = f4  - a*(f4  - fe4)  - b*(f5  - fe5)
    F5  = f5  - a*(f5  - fe5)  - b*(f4  - fe4)
    F6  = f6  - a*(f6  - fe6)  - b*(f7  - fe7)
    F7  = f7  - a*(f7  - fe7)  - b*(f6  - fe6)
    F8  = f8  - a*(f8  - fe8)  - b*(f11 - fe11)
    F11 = f11 - a*(f11 - fe11) - b*(f8  - fe8)
    F9  = f9  - a*(f9  - fe9)  - b*(f10 - fe10)
    F10 = f10 - a*(f10 - fe10) - b*(f9  - fe9)
    F12 = f12 - a*(f12 - fe12) - b*(f15 - fe15)
    F15 = f15 - a*(f15 - fe15) - b*(f12 - fe12)
    F13 = f13 - a*(f13 - fe13) - b*(f14 - fe14)
    F14 = f14 - a*(f14 - fe14) - b*(f13 - fe13)
    F16 = f16 - a*(f16 - fe16) - b*(f19 - fe19)
    F19 = f19 - a*(f19 - fe19) - b*(f16 - fe16)
    F17 = f17 - a*(f17 - fe17) - b*(f18 - fe18)
    F18 = f18 - a*(f18 - fe18) - b*(f17 - fe17)
    return F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19
end

# =====================================================================
# Modular boundary conditions for 2D/3D LI-BB V2 drivers.
#
# The old `rebuild_inlet_outlet_libb_{2d,3d}!` hard-coded one BC combo
# (Zou-He velocity west + Zou-He pressure east). Drivers now accept a
# `BCSpec{2,3}D` that names the BC per face; `apply_bc_rebuild_{2d,3d}!`
# dispatches on the per-face type and the backend, launching the right
# kernel for each active face. Face kernels are normal `@kernel`
# functions specialised per BC type — dispatch happens at Julia's method
# level (compiled on first call by the JIT), no `eval` needed.
#
# Design notes:
#
# 1. Each BC reconstructs `f_out[face, :, ...]` from the PRE-step `f_in`
#    values streamed from the interior, applies the Zou-He closure, and
#    does a local TRT collide. This bypasses the fused kernel's
#    halfway-BB fallback corruption at non-wall boundaries — same logic
#    as the old hardcoded rebuild functions.
#
# 2. `HalfwayBB` is a no-op: the fused kernel's PullHalfwayBB brick
#    already handles it at domain edges. Used for stationary channel
#    walls (j=1, j=Ny in 2D; (j,k) ∈ {1,Ny}×{1,Nz} faces in 3D).
#
# 3. Corners / face edges (where two BC faces meet) are left to the
#    kernel's halfway-BB fallback. For the typical Schäfer-Turek setup
#    (parabolic inlet with u=0 at walls + channel walls halfway-BB),
#    the BC values at corners are consistent (u=0).
#
# 4. `ZouHeVelocity` takes a profile (function or precomputed device
#    array). A pure uniform BC is just `ZouHeVelocity(u_uniform)` which
#    builds a constant profile on the backend.
# =====================================================================

abstract type AbstractBC end

"Kernel-fallback halfway bounce-back. No-op (the fused LBM kernel
already applies halfway-BB at domain edges via PullHalfwayBB)."
struct HalfwayBB <: AbstractBC end

"""
    ZouHeVelocity(profile)

Zou-He velocity boundary condition. `profile` is a device array of
length N (N = Ny for x-faces, Nx for y-faces, Nx·Nz or similar for
z-faces in 3D) giving the normal-into-domain velocity at each cell on
the face. Tangential velocity is assumed zero.

For a 2D x-face with a parabolic channel profile, pass the discrete
`u(y)` array. For uniform inflow, `fill(T(u_in), Ny)`.
"""
struct ZouHeVelocity{A<:AbstractArray} <: AbstractBC
    profile::A
end

"""
    ZouHePressure(ρ_out)

Zou-He pressure boundary condition. `ρ_out` is a scalar target density;
normal velocity is computed from the known streamed-in populations.
Tangential velocity is zero.
"""
struct ZouHePressure{T<:Real} <: AbstractBC
    ρ_out::T
end

"""
    BCSpec2D(; west, east, south, north)

Per-face BC specification for a 2D rectangular domain. Defaults are
`HalfwayBB` (kernel-fallback) so only set the faces that differ.
"""
struct BCSpec2D{W<:AbstractBC, E<:AbstractBC, S<:AbstractBC, N<:AbstractBC}
    west::W
    east::E
    south::S
    north::N
end
BCSpec2D(; west::AbstractBC=HalfwayBB(), east::AbstractBC=HalfwayBB(),
           south::AbstractBC=HalfwayBB(), north::AbstractBC=HalfwayBB()) =
    BCSpec2D(west, east, south, north)

"""
    BCSpec3D(; west, east, south, north, bottom, top)

Per-face BC specification for a 3D rectangular box domain.
"""
struct BCSpec3D{W<:AbstractBC, E<:AbstractBC, S<:AbstractBC,
                 N<:AbstractBC, B<:AbstractBC, T<:AbstractBC}
    west::W; east::E; south::S; north::N; bottom::B; top::T
end
BCSpec3D(; west::AbstractBC=HalfwayBB(), east::AbstractBC=HalfwayBB(),
           south::AbstractBC=HalfwayBB(), north::AbstractBC=HalfwayBB(),
           bottom::AbstractBC=HalfwayBB(), top::AbstractBC=HalfwayBB()) =
    BCSpec3D(west, east, south, north, bottom, top)

# ----------------------------------------------------------------------
# 2D face kernels — Zou-He velocity, Zou-He pressure
# ----------------------------------------------------------------------

@kernel function _bc_west_zh_velocity_2d!(f_out, f_in, profile, s_p, s_m)
    jm1 = @index(Global); j = jm1 + 1
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[1, j,   1]
        fp3 = f_in[1, j-1, 3]
        fp4 = f_in[2, j,   4]
        fp5 = f_in[1, j+1, 5]
        fp7 = f_in[2, j-1, 7]
        fp8 = f_in[2, j+1, 8]
        u_in = profile[j]
        ρ_w  = (fp1 + fp3 + fp5 + T(2)*(fp4 + fp7 + fp8)) / (one(T) - u_in)
        fp2  = fp4 + T(2/3) * ρ_w * u_in
        fp6  = fp8 - T(0.5)*(fp3 - fp5) + T(1/6) * ρ_w * u_in
        fp9  = fp7 + T(0.5)*(fp3 - fp5) + T(1/6) * ρ_w * u_in
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[1, j, 1] = F1; f_out[1, j, 2] = F2; f_out[1, j, 3] = F3
        f_out[1, j, 4] = F4; f_out[1, j, 5] = F5; f_out[1, j, 6] = F6
        f_out[1, j, 7] = F7; f_out[1, j, 8] = F8; f_out[1, j, 9] = F9
    end
end

@kernel function _bc_east_zh_pressure_2d!(f_out, f_in, Nx, ρ_out, s_p, s_m)
    jm1 = @index(Global); j = jm1 + 1
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[Nx,   j,   1]
        fp2 = f_in[Nx-1, j,   2]
        fp3 = f_in[Nx,   j-1, 3]
        fp5 = f_in[Nx,   j+1, 5]
        fp6 = f_in[Nx-1, j-1, 6]
        fp9 = f_in[Nx-1, j+1, 9]
        u_x = -one(T) + (fp1 + fp3 + fp5 + T(2)*(fp2 + fp6 + fp9)) / ρ_out
        fp4 = fp2 - T(2/3) * ρ_out * u_x
        fp7 = fp9 - T(0.5)*(fp3 - fp5) - T(1/6) * ρ_out * u_x
        fp8 = fp6 + T(0.5)*(fp3 - fp5) - T(1/6) * ρ_out * u_x
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[Nx, j, 1] = F1; f_out[Nx, j, 2] = F2; f_out[Nx, j, 3] = F3
        f_out[Nx, j, 4] = F4; f_out[Nx, j, 5] = F5; f_out[Nx, j, 6] = F6
        f_out[Nx, j, 7] = F7; f_out[Nx, j, 8] = F8; f_out[Nx, j, 9] = F9
    end
end

# 2D dispatch per face. HalfwayBB is a no-op; other BCs call their kernel.
@inline function _apply_bc_2d_west!(backend, f_out, f_in, ::HalfwayBB,
                                     s_p, s_m, Nx, Ny) end
@inline function _apply_bc_2d_west!(backend, f_out, f_in, bc::ZouHeVelocity,
                                     s_p, s_m, Nx, Ny)
    _bc_west_zh_velocity_2d!(backend)(f_out, f_in, bc.profile, s_p, s_m;
                                       ndrange=(Ny - 2,))
end

@inline function _apply_bc_2d_east!(backend, f_out, f_in, ::HalfwayBB,
                                     s_p, s_m, Nx, Ny) end
@inline function _apply_bc_2d_east!(backend, f_out, f_in, bc::ZouHePressure,
                                     s_p, s_m, Nx, Ny)
    _bc_east_zh_pressure_2d!(backend)(f_out, f_in, Nx, eltype(f_out)(bc.ρ_out),
                                        s_p, s_m; ndrange=(Ny - 2,))
end

# South / North wall bounce-back kernels.
# Overwrites the wall-crossing populations at j=1 (south) and j=Ny (north)
# with standard halfway BB: f_out[i,j,q̄] = f_in[i,j,q]. No collision
# needed — these are on the wall row itself if is_solid, or on the first
# fluid row if the streaming handles the solid row separately.
# For PullSLBM (which clamps at boundaries), these kernels fix the
# populations that the streaming couldn't bounce.

@kernel function _bc_south_halfwaybb_2d!(f_out, @Const(f_in), Ny, i_shift::Int)
    im1 = @index(Global); i = im1 + i_shift
    @inbounds begin
        # j=1: bounce populations heading south back north
        f_out[i, 1, 3] = f_in[i, 1, 5]   # 5→3
        f_out[i, 1, 6] = f_in[i, 1, 8]   # 8→6
        f_out[i, 1, 7] = f_in[i, 1, 9]   # 9→7
    end
end

@kernel function _bc_north_halfwaybb_2d!(f_out, @Const(f_in), Ny, i_shift::Int)
    im1 = @index(Global); i = im1 + i_shift
    @inbounds begin
        # j=Ny: bounce populations heading north back south
        f_out[i, Ny, 5] = f_in[i, Ny, 3]   # 3→5
        f_out[i, Ny, 8] = f_in[i, Ny, 6]   # 6→8
        f_out[i, Ny, 9] = f_in[i, Ny, 7]   # 7→9
    end
end

# Local-tau variants: read sp/sm from 2D arrays at the face index
@kernel function _bc_west_zh_velocity_local_2d!(f_out, f_in, profile, sp_field, sm_field)
    jm1 = @index(Global); j = jm1 + 1
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[1, j,   1]
        fp3 = f_in[1, j-1, 3]
        fp4 = f_in[2, j,   4]
        fp5 = f_in[1, j+1, 5]
        fp7 = f_in[2, j-1, 7]
        fp8 = f_in[2, j+1, 8]
        u_in = profile[j]
        ρ_w  = (fp1 + fp3 + fp5 + T(2)*(fp4 + fp7 + fp8)) / (one(T) - u_in)
        fp2  = fp4 + T(2/3) * ρ_w * u_in
        fp6  = fp8 - T(0.5)*(fp3 - fp5) + T(1/6) * ρ_w * u_in
        fp9  = fp7 + T(0.5)*(fp3 - fp5) + T(1/6) * ρ_w * u_in
        s_p = sp_field[1, j]; s_m = sm_field[1, j]
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[1, j, 1] = F1; f_out[1, j, 2] = F2; f_out[1, j, 3] = F3
        f_out[1, j, 4] = F4; f_out[1, j, 5] = F5; f_out[1, j, 6] = F6
        f_out[1, j, 7] = F7; f_out[1, j, 8] = F8; f_out[1, j, 9] = F9
    end
end

@kernel function _bc_east_zh_pressure_local_2d!(f_out, f_in, Nx, ρ_out, sp_field, sm_field)
    jm1 = @index(Global); j = jm1 + 1
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[Nx,   j,   1]
        fp2 = f_in[Nx-1, j,   2]
        fp3 = f_in[Nx,   j-1, 3]
        fp5 = f_in[Nx,   j+1, 5]
        fp6 = f_in[Nx-1, j-1, 6]
        fp9 = f_in[Nx-1, j+1, 9]
        u_x = -one(T) + (fp1 + fp3 + fp5 + T(2)*(fp2 + fp6 + fp9)) / ρ_out
        fp4 = fp2 - T(2/3) * ρ_out * u_x
        fp7 = fp9 - T(0.5)*(fp3 - fp5) - T(1/6) * ρ_out * u_x
        fp8 = fp6 + T(0.5)*(fp3 - fp5) - T(1/6) * ρ_out * u_x
        s_p = sp_field[Nx, j]; s_m = sm_field[Nx, j]
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[Nx, j, 1] = F1; f_out[Nx, j, 2] = F2; f_out[Nx, j, 3] = F3
        f_out[Nx, j, 4] = F4; f_out[Nx, j, 5] = F5; f_out[Nx, j, 6] = F6
        f_out[Nx, j, 7] = F7; f_out[Nx, j, 8] = F8; f_out[Nx, j, 9] = F9
    end
end

@inline function _apply_bc_2d_west_local!(backend, f_out, f_in, ::HalfwayBB,
                                           sp_field, sm_field, Nx, Ny) end
@inline function _apply_bc_2d_west_local!(backend, f_out, f_in, bc::ZouHeVelocity,
                                           sp_field, sm_field, Nx, Ny)
    _bc_west_zh_velocity_local_2d!(backend)(f_out, f_in, bc.profile, sp_field, sm_field;
                                             ndrange=(Ny - 2,))
end
@inline function _apply_bc_2d_east_local!(backend, f_out, f_in, ::HalfwayBB,
                                           sp_field, sm_field, Nx, Ny) end
@inline function _apply_bc_2d_east_local!(backend, f_out, f_in, bc::ZouHePressure,
                                           sp_field, sm_field, Nx, Ny)
    _bc_east_zh_pressure_local_2d!(backend)(f_out, f_in, Nx, eltype(f_out)(bc.ρ_out),
                                              sp_field, sm_field; ndrange=(Ny - 2,))
end

@inline function _apply_bc_2d_south!(backend, f_out, f_in, ::HalfwayBB,
                                      s_p, s_m, Nx, Ny;
                                      west_bc=nothing, east_bc=nothing)
    # Include i=1 if west is HalfwayBB (wall/interface); skip otherwise
    # because ZouHe writers want to own the corner (legacy single-block
    # behaviour). Same on i=Nx. When multi-block has east=:interface
    # (HalfwayBB), the south BB must fire at i=Nx so the interface-wall
    # corner matches what single-block would compute at the same x.
    i_lo = (west_bc isa HalfwayBB || west_bc === nothing) ? 1 : 2
    i_hi = (east_bc isa HalfwayBB || east_bc === nothing) ? Nx : Nx - 1
    count = i_hi - i_lo + 1
    count ≤ 0 && return nothing
    _bc_south_halfwaybb_2d!(backend)(f_out, f_in, Ny, i_lo - 1; ndrange=(count,))
end
@inline function _apply_bc_2d_north!(backend, f_out, f_in, ::HalfwayBB,
                                      s_p, s_m, Nx, Ny;
                                      west_bc=nothing, east_bc=nothing)
    i_lo = (west_bc isa HalfwayBB || west_bc === nothing) ? 1 : 2
    i_hi = (east_bc isa HalfwayBB || east_bc === nothing) ? Nx : Nx - 1
    count = i_hi - i_lo + 1
    count ≤ 0 && return nothing
    _bc_north_halfwaybb_2d!(backend)(f_out, f_in, Ny, i_lo - 1; ndrange=(count,))
end

"""
    apply_bc_rebuild_2d!(f_out, f_in, bcspec, ν, Nx, Ny;
                          sp_field=nothing, sm_field=nothing)

Apply the per-face BCs in `bcspec::BCSpec2D` to `f_out` at the current
step. Reads pre-step values from `f_in` (streamed from interior) for
each active face, applies the Zou-He closure, and collides locally with
TRT Λ=3/16 at the requested viscosity `ν`.

If `sp_field` and `sm_field` (2D arrays) are provided, per-cell local
rates are used at each face instead of the uniform ν-derived rates.
This is needed for SLBM on non-uniform meshes where τ varies per cell.
"""
function apply_bc_rebuild_2d!(f_out, f_in, bcspec::BCSpec2D, ν::Real,
                                Nx::Int, Ny::Int;
                                sp_field=nothing, sm_field=nothing)
    backend = KernelAbstractions.get_backend(f_out)
    T = eltype(f_out)
    s_p_r, s_m_r = trt_rates(ν; Λ=3/16)
    s_p_uni = T(s_p_r); s_m_uni = T(s_m_r)

    if isnothing(sp_field)
        _apply_bc_2d_west!(backend, f_out, f_in, bcspec.west, s_p_uni, s_m_uni, Nx, Ny)
        _apply_bc_2d_east!(backend, f_out, f_in, bcspec.east, s_p_uni, s_m_uni, Nx, Ny)
    else
        _apply_bc_2d_west_local!(backend, f_out, f_in, bcspec.west, sp_field, sm_field, Nx, Ny)
        _apply_bc_2d_east_local!(backend, f_out, f_in, bcspec.east, sp_field, sm_field, Nx, Ny)
    end
    _apply_bc_2d_south!(backend, f_out, f_in, bcspec.south, s_p_uni, s_m_uni, Nx, Ny;
                          west_bc=bcspec.west, east_bc=bcspec.east)
    _apply_bc_2d_north!(backend, f_out, f_in, bcspec.north, s_p_uni, s_m_uni, Nx, Ny;
                          west_bc=bcspec.west, east_bc=bcspec.east)
    return nothing
end

# ----------------------------------------------------------------------
# 3D face kernels
# ----------------------------------------------------------------------

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
