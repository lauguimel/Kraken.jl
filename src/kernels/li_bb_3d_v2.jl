# =====================================================================
# TRT + LI-BB 3D kernel (D3Q19), Ginzburg-exact via DSL.
#
# Same recipe as li_bb_2d_v2.jl, ported to D3Q19:
#   PullHalfwayBB_3D → SolidInert_3D | ApplyLiBBPrePhase_3D →
#                      Moments_3D → CollideTRTDirect_3D →
#                      WriteMoments_3D
#
# Reuses `_libb_branch` from li_bb_2d.jl (the Bouzidi formula is
# stencil-independent).
# =====================================================================

const _TRT_LIBB_V2_SPEC_3D = LBMSpec(
    PullHalfwayBB_3D(), SolidInert_3D(),
    ApplyLiBBPrePhase_3D(),
    Moments_3D(), CollideTRTDirect_3D(),
    WriteMoments_3D();
    stencil = :D3Q19,
)

"""
    fused_trt_libb_v2_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                q_wall, uw_x, uw_y, uw_z,
                                Nx, Ny, Nz, ν; Λ=3/16)

Single D3Q19 step: pull-stream + SolidInert on solid cells, or
ApplyLiBBPrePhase + TRT collision on fluid cells. Ginzburg-exact for
halfway-BB + TRT Λ = 3/16 on Couette; handles arbitrary q_w ∈ (0, 1]
via the full Bouzidi formula.
"""
function fused_trt_libb_v2_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                      q_wall, uw_link_x, uw_link_y, uw_link_z,
                                      Nx, Ny, Nz, ν; Λ::Real=3/16)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    s_plus, s_minus = trt_rates(ν; Λ=Λ)
    kernel! = build_lbm_kernel(backend, _TRT_LIBB_V2_SPEC_3D)
    kernel!(f_out, ρ, ux, uy, uz, f_in, is_solid,
            q_wall, uw_link_x, uw_link_y, uw_link_z,
            Nx, Ny, Nz, ET(s_plus), ET(s_minus);
            ndrange=(Nx, Ny, Nz))
end

"""
    precompute_q_wall_sphere_3d(Nx, Ny, Nz, cx, cy, cz, R; FT=Float64)
        -> (q_wall, is_solid)

Analytical sub-cell `q_w` for a sphere of radius `R` centred at
`(cx, cy, cz)` on an `Nx×Ny×Nz` D3Q19 grid with unit lattice spacing.
For every fluid node whose link `q` crosses the sphere, solves the
quadratic `|x_f + t · c_q − c|² = R²` for the smallest positive
`t ∈ (0, 1]` and stores that as `q_wall[i,j,k,q]`.
"""
function precompute_q_wall_sphere_3d(Nx::Int, Ny::Int, Nz::Int,
                                      cx::Real, cy::Real, cz::Real,
                                      R::Real;
                                      FT::Type{<:AbstractFloat}=Float64)
    cxT, cyT, czT, RT = FT(cx), FT(cy), FT(cz), FT(R)
    R² = RT * RT
    is_solid = zeros(Bool, Nx, Ny, Nz)
    q_wall = zeros(FT, Nx, Ny, Nz, 19)
    cxs = velocities_x(D3Q19())
    cys = velocities_y(D3Q19())
    czs = velocities_z(D3Q19())

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        xf = FT(i - 1); yf = FT(j - 1); zf = FT(k - 1)
        dx_f = xf - cxT; dy_f = yf - cyT; dz_f = zf - czT
        if dx_f * dx_f + dy_f * dy_f + dz_f * dz_f ≤ R²
            is_solid[i, j, k] = true
            continue
        end
        for q in 2:19
            cqx = FT(cxs[q]); cqy = FT(cys[q]); cqz = FT(czs[q])
            xn = xf + cqx; yn = yf + cqy; zn = zf + cqz
            dx_n = xn - cxT; dy_n = yn - cyT; dz_n = zn - czT
            if dx_n * dx_n + dy_n * dy_n + dz_n * dz_n > R²
                continue
            end
            a = cqx * cqx + cqy * cqy + cqz * cqz
            b = FT(2) * (dx_f * cqx + dy_f * cqy + dz_f * cqz)
            c = dx_f * dx_f + dy_f * dy_f + dz_f * dz_f - R²
            disc = b * b - FT(4) * a * c
            disc < zero(FT) && continue
            sd = sqrt(disc)
            t1 = (-b - sd) / (FT(2) * a)
            t2 = (-b + sd) / (FT(2) * a)
            t = t1 > zero(FT) ? t1 : t2
            if t > zero(FT) && t ≤ one(FT)
                q_wall[i, j, k, q] = t
            end
        end
    end
    return q_wall, is_solid
end

"""
    compute_drag_libb_3d(f_post, q_wall, Nx, Ny, Nz)
        -> (Fx, Fy, Fz)

D3Q19 MEA drag/lift for a stationary LI-BB wall. Same halfway-BB
convention as the 2D version (`F = 2·c_q·f_q(post-coll)`).
"""
function compute_drag_libb_3d(f_post, q_wall, Nx::Int, Ny::Int, Nz::Int)
    f = Array(f_post)
    qw = Array(q_wall)
    cxs = velocities_x(D3Q19())
    cys = velocities_y(D3Q19())
    czs = velocities_z(D3Q19())
    Fx = 0.0; Fy = 0.0; Fz = 0.0
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        for q in 2:19
            if qw[i, j, k, q] > 0
                fv = Float64(f[i, j, k, q])
                Fx += 2.0 * Float64(cxs[q]) * fv
                Fy += 2.0 * Float64(cys[q]) * fv
                Fz += 2.0 * Float64(czs[q]) * fv
            end
        end
    end
    return (Fx = Fx, Fy = Fy, Fz = Fz)
end

# =====================================================================
# D3Q19 TRT local collide and pre-coll rebuild for 3D inlet/outlet.
#
# Same rationale as the 2D version in src/drivers/cylinder_libb.jl:
# the V2 fused kernel's halfway-BB fallback at i=1 and i=Nx corrupts
# ρ_wall by bouncing back the +x streamed pops. A post-kernel Zou-He
# patch then reads these corrupt pops and either stalls the flow
# (with equilibrium inlet + Neumann outlet) or blows up (with
# Zou-He pressure outlet). The rebuild overwrites f_out[1,:,:,:]
# and f_out[Nx,:,:,:] using pre-step f_in values streamed from the
# interior, Zou-He closure on the 5 unknown pops per face, and a
# local TRT collide.
# =====================================================================

@inline function _trt_collide_local_3d(f1::T, f2::T, f3::T, f4::T, f5::T,
                                        f6::T, f7::T, f8::T, f9::T,
                                        f10::T, f11::T, f12::T, f13::T, f14::T,
                                        f15::T, f16::T, f17::T, f18::T, f19::T,
                                        s_p::T, s_m::T) where {T}
    ρ  = f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+f17+f18+f19
    # D3Q19 moments (indices: see d3q19.jl)
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

# West inlet rebuild: Zou-He velocity with u=(u_in, 0, 0) at (1, j, k)
# for j=2..Ny-1, k=2..Nz-1. Reconstructs the 14 known pre-step pops
# from f_in (streaming from the interior) and applies Zou-He to the
# 5 unknown +x pops (q=2, 8, 10, 12, 14).
@kernel function _rebuild_west_libb_3d_kernel!(f_out, f_in, u_in_val, s_p, s_m)
    jm1, km1 = @index(Global, NTuple)
    j = jm1 + 1; k = km1 + 1
    T = eltype(f_out)
    @inbounds begin
        # Post-stream pre-coll pops at (1, j, k), pulled from f_in.
        # Parallel (cx=0): q=1 (rest), 4,5 (±y), 6,7 (±z), 16..19 (±y±z).
        fp1  = f_in[1, j,     k,     1]
        fp4  = f_in[1, j - 1, k,     4]
        fp5  = f_in[1, j + 1, k,     5]
        fp6  = f_in[1, j,     k - 1, 6]
        fp7  = f_in[1, j,     k + 1, 7]
        fp16 = f_in[1, j - 1, k - 1, 16]
        fp17 = f_in[1, j + 1, k - 1, 17]
        fp18 = f_in[1, j - 1, k + 1, 18]
        fp19 = f_in[1, j + 1, k + 1, 19]
        # Known outgoing (cx=-1): q=3, 9, 11, 13, 15 — streamed from i=2.
        fp3  = f_in[2, j,     k,     3]
        fp9  = f_in[2, j - 1, k,     9]
        fp11 = f_in[2, j + 1, k,     11]
        fp13 = f_in[2, j,     k - 1, 13]
        fp15 = f_in[2, j,     k + 1, 15]
        # Zou-He west with u_n = u_in, u_tang1=u_tang2=0.
        u_n  = T(u_in_val)
        sum_par = fp1 + fp4 + fp5 + fp6 + fp7 + fp16 + fp17 + fp18 + fp19
        sum_out = fp3 + fp9 + fp11 + fp13 + fp15
        ρ_w  = (sum_par + T(2) * sum_out) / (one(T) - u_n)
        # Axis unknown f2
        fp2  = fp3 + T(1/3) * ρ_w * u_n
        # Diagonal unknowns (u_tang=0). Pair 1 in tang-y: f8, f10.
        tang1_diff = fp4 - fp5
        tang2_diff = fp6 - fp7
        fp8  = fp11 - T(0.5) * tang1_diff + T(1/6) * ρ_w * u_n
        fp10 = fp9  + T(0.5) * tang1_diff + T(1/6) * ρ_w * u_n
        fp12 = fp15 - T(0.5) * tang2_diff + T(1/6) * ρ_w * u_n
        fp14 = fp13 + T(0.5) * tang2_diff + T(1/6) * ρ_w * u_n
        # Local TRT collide on the reconstructed pre-coll state.
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

# East outlet rebuild: Zou-He pressure with ρ=ρ_out, u=(u_x, 0, 0) at (Nx, j, k).
# Unknown pops are the -x ones: q=3, 9, 11, 13, 15.
@kernel function _rebuild_east_libb_3d_kernel!(f_out, f_in, Nx, ρ_out, s_p, s_m)
    jm1, km1 = @index(Global, NTuple)
    j = jm1 + 1; k = km1 + 1
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
        # Known outgoing (cx=+1): q=2, 8, 10, 12, 14 — streamed from i=Nx-1.
        fp2  = f_in[Nx - 1, j,     k,     2]
        fp8  = f_in[Nx - 1, j - 1, k,     8]
        fp10 = f_in[Nx - 1, j + 1, k,     10]
        fp12 = f_in[Nx - 1, j,     k - 1, 12]
        fp14 = f_in[Nx - 1, j,     k + 1, 14]
        # Zou-He pressure east: solve u_n from known + ρ_out.
        ρ_o     = T(ρ_out)
        sum_par = fp1 + fp4 + fp5 + fp6 + fp7 + fp16 + fp17 + fp18 + fp19
        sum_out = fp2 + fp8 + fp10 + fp12 + fp14
        u_n     = -(one(T) - (sum_par + T(2) * sum_out) / ρ_o)
        # Axis unknown f3 (-x)
        fp3  = fp2 - T(1/3) * ρ_o * u_n
        tang1_diff = fp4 - fp5
        tang2_diff = fp6 - fp7
        fp9  = fp10 - T(0.5) * tang1_diff - T(1/6) * ρ_o * u_n
        fp11 = fp8  + T(0.5) * tang1_diff - T(1/6) * ρ_o * u_n
        fp13 = fp14 - T(0.5) * tang2_diff - T(1/6) * ρ_o * u_n
        fp15 = fp12 + T(0.5) * tang2_diff - T(1/6) * ρ_o * u_n
        # Local TRT collide
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

"""
    rebuild_inlet_outlet_libb_3d!(f_out, f_in, u_in, ρ_out, ν, Nx, Ny, Nz)

3D analog of `rebuild_inlet_outlet_libb_2d!`. Overwrites
`f_out[1, j, k, :]` (Zou-He velocity, u=(u_in, 0, 0)) and
`f_out[Nx, j, k, :]` (Zou-He pressure, ρ=ρ_out) for
j=2..Ny-1, k=2..Nz-1, using pre-step values + local TRT collide.
Corners / face edges are left to the kernel's halfway-BB fallback.
"""
function rebuild_inlet_outlet_libb_3d!(f_out, f_in, u_in::Real, ρ_out::Real,
                                         ν::Real, Nx::Int, Ny::Int, Nz::Int)
    backend = KernelAbstractions.get_backend(f_out)
    T = eltype(f_out)
    s_p_r, s_m_r = trt_rates(ν; Λ=3/16)
    s_p = T(s_p_r); s_m = T(s_m_r)
    kw! = _rebuild_west_libb_3d_kernel!(backend)
    ke! = _rebuild_east_libb_3d_kernel!(backend)
    kw!(f_out, f_in, T(u_in), s_p, s_m; ndrange=(Ny - 2, Nz - 2))
    ke!(f_out, f_in, Nx, T(ρ_out), s_p, s_m; ndrange=(Ny - 2, Nz - 2))
    return nothing
end

"""
    run_sphere_libb_3d(; Nx=120, Ny=60, Nz=60, radius=8, u_in=0.04,
                        ν=…, max_steps=20000, avg_window=5000, ρ_out=1.0)

3D sphere in uniform x-flow. Halfway-BB on y/z walls (kernel
fallback), Zou-He velocity inlet at i=1 + Zou-He pressure outlet at
i=Nx reconstructed pre-collision (see
`rebuild_inlet_outlet_libb_3d!`) so the flow is not stalled by the
kernel's halfway-BB fallback corruption of ρ_wall.
Drag integrated via `compute_drag_libb_3d` over `avg_window` final
steps.

Returns a NamedTuple with (; ρ, ux, uy, uz, Cd, Fx, Fy, Fz, ...).
Reference: 3D sphere in uniform flow at Re=20, Cd ≈ 2.6 (Clift et al.
1978; confinement-adjusted).
"""
function run_sphere_libb_3d(; Nx::Int=120, Ny::Int=60, Nz::Int=60,
                              cx::Union{Nothing,Real}=nothing,
                              cy::Union{Nothing,Real}=nothing,
                              cz::Union{Nothing,Real}=nothing,
                              radius::Real=8,
                              u_in::Real=0.04, ν::Real=0.04,
                              max_steps::Int=20_000,
                              avg_window::Int=5_000,
                              ρ_out::Real=1.0,
                              backend=KernelAbstractions.CPU(),
                              T::Type{<:AbstractFloat}=Float64)
    cx = isnothing(cx) ? Nx ÷ 4 : Float64(cx)
    cy = isnothing(cy) ? Ny ÷ 2 : Float64(cy)
    cz = isnothing(cz) ? Nz ÷ 2 : Float64(cz)

    # Precompute on CPU (analytic geometry, not kernel-launched)
    q_wall_h, is_solid_h = precompute_q_wall_sphere_3d(Nx, Ny, Nz, cx, cy, cz,
                                                        radius; FT=T)
    f_in_h = zeros(T, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx, q in 1:19
        f_in_h[i, j, k, q] = Kraken.equilibrium(D3Q19(), one(T), T(u_in),
                                                 zero(T), zero(T), q)
    end

    # Allocate on the target backend
    q_wall   = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny, Nz)
    uw_x     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    uw_y     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    uw_z     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    f_in     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    f_out    = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    ρ        = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    ux       = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    uy       = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    uz       = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)

    copyto!(q_wall, q_wall_h)
    copyto!(is_solid, is_solid_h)
    fill!(uw_x, zero(T)); fill!(uw_y, zero(T)); fill!(uw_z, zero(T))
    copyto!(f_in, f_in_h)
    fill!(ρ, one(T));  fill!(ux, zero(T))
    fill!(uy, zero(T)); fill!(uz, zero(T))

    Fx_sum = 0.0; Fy_sum = 0.0; Fz_sum = 0.0; n_avg = 0

    for step in 1:max_steps
        fused_trt_libb_v2_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                    q_wall, uw_x, uw_y, uw_z,
                                    Nx, Ny, Nz, T(ν))
        # Pre-collision Zou-He rebuild at i=1 (velocity) and i=Nx (pressure)
        rebuild_inlet_outlet_libb_3d!(f_out, f_in, u_in, ρ_out, ν,
                                        Nx, Ny, Nz)

        if step > max_steps - avg_window
            drag = compute_drag_libb_3d(f_out, q_wall, Nx, Ny, Nz)
            Fx_sum += drag.Fx; Fy_sum += drag.Fy; Fz_sum += drag.Fz
            n_avg += 1
        end

        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    Fx_avg = Fx_sum / n_avg
    Fy_avg = Fy_sum / n_avg
    Fz_avg = Fz_sum / n_avg
    D = 2 * Float64(radius)
    A = π * Float64(radius)^2        # cross-section area
    u_ref = Float64(u_in)
    Cd = 2.0 * Fx_avg / (u_ref^2 * A)

    return (; ρ = Array(ρ), ux = Array(ux), uy = Array(uy), uz = Array(uz),
             Cd, Fx = Fx_avg, Fy = Fy_avg, Fz = Fz_avg,
             q_wall = Array(q_wall), is_solid = Array(is_solid),
             u_ref, D, A)
end
