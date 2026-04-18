using KernelAbstractions

# ===========================================================================
# Semi-Lagrangian LBM kernels in 3D (D3Q19).
# ===========================================================================

"""
    SLBMGeometry3D{T, AT4}

Precomputed departure indices for D3Q19 SLBM. Fields `i_dep[Nξ,Nη,Nζ,19]`,
`j_dep`, `k_dep` in grid-index space.
"""
struct SLBMGeometry3D{T<:AbstractFloat, AT4<:AbstractArray{T, 4}}
    i_dep::AT4
    j_dep::AT4
    k_dep::AT4
    Nξ::Int
    Nη::Int
    Nζ::Int
    periodic_ξ::Bool
    periodic_η::Bool
    periodic_ζ::Bool
    dx_ref::T
end

function build_slbm_geometry_3d(mesh::CurvilinearMesh3D{T}; local_cfl::Bool=false) where {T}
    Nξ, Nη, Nζ = mesh.Nξ, mesh.Nη, mesh.Nζ
    cx = velocities_x(D3Q19())
    cy = velocities_y(D3Q19())
    cz = velocities_z(D3Q19())

    denom_ξ = T(mesh.periodic_ξ ? Nξ : Nξ - 1)
    denom_η = T(mesh.periodic_η ? Nη : Nη - 1)
    denom_ζ = T(mesh.periodic_ζ ? Nζ : Nζ - 1)
    Δξ = one(T) / denom_ξ
    Δη = one(T) / denom_η
    Δζ = one(T) / denom_ζ

    i_dep = zeros(T, Nξ, Nη, Nζ, 19)
    j_dep = zeros(T, Nξ, Nη, Nζ, 19)
    k_dep = zeros(T, Nξ, Nη, Nζ, 19)
    dxr = mesh.dx_ref

    @inbounds for k in 1:Nζ, j in 1:Nη, i in 1:Nξ
        J = mesh.J[i,j,k]
        # Metric tensor components
        dX_ξ = mesh.dXdξ[i,j,k]; dX_η = mesh.dXdη[i,j,k]; dX_ζ = mesh.dXdζ[i,j,k]
        dY_ξ = mesh.dYdξ[i,j,k]; dY_η = mesh.dYdη[i,j,k]; dY_ζ = mesh.dYdζ[i,j,k]
        dZ_ξ = mesh.dZdξ[i,j,k]; dZ_η = mesh.dZdη[i,j,k]; dZ_ζ = mesh.dZdζ[i,j,k]

        # Inverse Jacobian (cofactor / det). Rows of inv give ∂ξ/∂X, etc.
        ξx = (dY_η*dZ_ζ - dY_ζ*dZ_η) / J
        ξy = (dX_ζ*dZ_η - dX_η*dZ_ζ) / J
        ξz = (dX_η*dY_ζ - dX_ζ*dY_η) / J
        ηx = (dY_ζ*dZ_ξ - dY_ξ*dZ_ζ) / J
        ηy = (dX_ξ*dZ_ζ - dX_ζ*dZ_ξ) / J
        ηz = (dX_ζ*dY_ξ - dX_ξ*dY_ζ) / J
        ζx = (dY_ξ*dZ_η - dY_η*dZ_ξ) / J
        ζy = (dX_η*dZ_ξ - dX_ξ*dZ_η) / J
        ζz = (dX_ξ*dY_η - dX_η*dY_ξ) / J

        if local_cfl
            lξ = sqrt(dX_ξ^2 + dY_ξ^2 + dZ_ξ^2) * Δξ
            lη = sqrt(dX_η^2 + dY_η^2 + dZ_η^2) * Δη
            lζ = sqrt(dX_ζ^2 + dY_ζ^2 + dZ_ζ^2) * Δζ
            dxr_local = cbrt(lξ * lη * lζ)
        else
            dxr_local = dxr
        end

        for q in 1:19
            cqx = T(cx[q]); cqy = T(cy[q]); cqz = T(cz[q])
            Δi = -dxr_local * denom_ξ * (ξx*cqx + ξy*cqy + ξz*cqz)
            Δj = -dxr_local * denom_η * (ηx*cqx + ηy*cqy + ηz*cqz)
            Δk = -dxr_local * denom_ζ * (ζx*cqx + ζy*cqy + ζz*cqz)
            i_dep[i,j,k,q] = T(i) + Δi
            j_dep[i,j,k,q] = T(j) + Δj
            k_dep[i,j,k,q] = T(k) + Δk
        end
    end

    return SLBMGeometry3D{T, Array{T,4}}(i_dep, j_dep, k_dep, Nξ, Nη, Nζ,
        mesh.periodic_ξ, mesh.periodic_η, mesh.periodic_ζ, dxr)
end

function transfer_slbm_geometry_3d(geom::SLBMGeometry3D{T}, backend) where {T}
    i_dep = KernelAbstractions.allocate(backend, T, size(geom.i_dep))
    j_dep = KernelAbstractions.allocate(backend, T, size(geom.j_dep))
    k_dep = KernelAbstractions.allocate(backend, T, size(geom.k_dep))
    copyto!(i_dep, geom.i_dep); copyto!(j_dep, geom.j_dep); copyto!(k_dep, geom.k_dep)
    return SLBMGeometry3D{T, typeof(i_dep)}(i_dep, j_dep, k_dep, geom.Nξ, geom.Nη, geom.Nζ,
        geom.periodic_ξ, geom.periodic_η, geom.periodic_ζ, geom.dx_ref)
end

# ---------------------------------------------------------------------------
# Trilinear interpolation helper (8-neighbor stencil).
# ---------------------------------------------------------------------------

@inline function trilinear_f(f_in, ifloat::T, jfloat::T, kfloat::T, q::Int,
                              Nξ::Int, Nη::Int, Nζ::Int,
                              pξ::Bool, pη::Bool, pζ::Bool) where {T}
    i0 = unsafe_trunc(Int, floor(ifloat))
    j0 = unsafe_trunc(Int, floor(jfloat))
    k0 = unsafe_trunc(Int, floor(kfloat))
    α = ifloat - T(i0); β = jfloat - T(j0); γ = kfloat - T(k0)
    i0w = _wrap_or_clamp(i0, Nξ, pξ); i1w = _wrap_or_clamp(i0+1, Nξ, pξ)
    j0w = _wrap_or_clamp(j0, Nη, pη); j1w = _wrap_or_clamp(j0+1, Nη, pη)
    k0w = _wrap_or_clamp(k0, Nζ, pζ); k1w = _wrap_or_clamp(k0+1, Nζ, pζ)
    f000 = f_in[i0w, j0w, k0w, q]; f100 = f_in[i1w, j0w, k0w, q]
    f010 = f_in[i0w, j1w, k0w, q]; f110 = f_in[i1w, j1w, k0w, q]
    f001 = f_in[i0w, j0w, k1w, q]; f101 = f_in[i1w, j0w, k1w, q]
    f011 = f_in[i0w, j1w, k1w, q]; f111 = f_in[i1w, j1w, k1w, q]
    one_α = one(T) - α; one_β = one(T) - β; one_γ = one(T) - γ
    c00 = one_α * f000 + α * f100
    c10 = one_α * f010 + α * f110
    c01 = one_α * f001 + α * f101
    c11 = one_α * f011 + α * f111
    c0 = one_β * c00 + β * c10
    c1 = one_β * c01 + β * c11
    return one_γ * c0 + γ * c1
end

# ---------------------------------------------------------------------------
# Fused SLBM + BGK step D3Q19.
# ---------------------------------------------------------------------------

@kernel function slbm_bgk_step_3d_kernel!(f_out, @Const(f_in),
                                            ρ_out, ux_out, uy_out, uz_out,
                                            @Const(is_solid),
                                            @Const(i_dep), @Const(j_dep), @Const(k_dep),
                                            Nξ, Nη, Nζ, ω,
                                            pξ, pη, pζ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f_out)
        # 19 trilinear interpolations (unrolled for GPU compat)
        fp1  = trilinear_f(f_in, i_dep[i,j,k,1],  j_dep[i,j,k,1],  k_dep[i,j,k,1],  1,  Nξ, Nη, Nζ, pξ, pη, pζ)
        fp2  = trilinear_f(f_in, i_dep[i,j,k,2],  j_dep[i,j,k,2],  k_dep[i,j,k,2],  2,  Nξ, Nη, Nζ, pξ, pη, pζ)
        fp3  = trilinear_f(f_in, i_dep[i,j,k,3],  j_dep[i,j,k,3],  k_dep[i,j,k,3],  3,  Nξ, Nη, Nζ, pξ, pη, pζ)
        fp4  = trilinear_f(f_in, i_dep[i,j,k,4],  j_dep[i,j,k,4],  k_dep[i,j,k,4],  4,  Nξ, Nη, Nζ, pξ, pη, pζ)
        fp5  = trilinear_f(f_in, i_dep[i,j,k,5],  j_dep[i,j,k,5],  k_dep[i,j,k,5],  5,  Nξ, Nη, Nζ, pξ, pη, pζ)
        fp6  = trilinear_f(f_in, i_dep[i,j,k,6],  j_dep[i,j,k,6],  k_dep[i,j,k,6],  6,  Nξ, Nη, Nζ, pξ, pη, pζ)
        fp7  = trilinear_f(f_in, i_dep[i,j,k,7],  j_dep[i,j,k,7],  k_dep[i,j,k,7],  7,  Nξ, Nη, Nζ, pξ, pη, pζ)
        fp8  = trilinear_f(f_in, i_dep[i,j,k,8],  j_dep[i,j,k,8],  k_dep[i,j,k,8],  8,  Nξ, Nη, Nζ, pξ, pη, pζ)
        fp9  = trilinear_f(f_in, i_dep[i,j,k,9],  j_dep[i,j,k,9],  k_dep[i,j,k,9],  9,  Nξ, Nη, Nζ, pξ, pη, pζ)
        fp10 = trilinear_f(f_in, i_dep[i,j,k,10], j_dep[i,j,k,10], k_dep[i,j,k,10], 10, Nξ, Nη, Nζ, pξ, pη, pζ)
        fp11 = trilinear_f(f_in, i_dep[i,j,k,11], j_dep[i,j,k,11], k_dep[i,j,k,11], 11, Nξ, Nη, Nζ, pξ, pη, pζ)
        fp12 = trilinear_f(f_in, i_dep[i,j,k,12], j_dep[i,j,k,12], k_dep[i,j,k,12], 12, Nξ, Nη, Nζ, pξ, pη, pζ)
        fp13 = trilinear_f(f_in, i_dep[i,j,k,13], j_dep[i,j,k,13], k_dep[i,j,k,13], 13, Nξ, Nη, Nζ, pξ, pη, pζ)
        fp14 = trilinear_f(f_in, i_dep[i,j,k,14], j_dep[i,j,k,14], k_dep[i,j,k,14], 14, Nξ, Nη, Nζ, pξ, pη, pζ)
        fp15 = trilinear_f(f_in, i_dep[i,j,k,15], j_dep[i,j,k,15], k_dep[i,j,k,15], 15, Nξ, Nη, Nζ, pξ, pη, pζ)
        fp16 = trilinear_f(f_in, i_dep[i,j,k,16], j_dep[i,j,k,16], k_dep[i,j,k,16], 16, Nξ, Nη, Nζ, pξ, pη, pζ)
        fp17 = trilinear_f(f_in, i_dep[i,j,k,17], j_dep[i,j,k,17], k_dep[i,j,k,17], 17, Nξ, Nη, Nζ, pξ, pη, pζ)
        fp18 = trilinear_f(f_in, i_dep[i,j,k,18], j_dep[i,j,k,18], k_dep[i,j,k,18], 18, Nξ, Nη, Nζ, pξ, pη, pζ)
        fp19 = trilinear_f(f_in, i_dep[i,j,k,19], j_dep[i,j,k,19], k_dep[i,j,k,19], 19, Nξ, Nη, Nζ, pξ, pη, pζ)

        if is_solid[i,j,k]
            # Bounce-back opposite pairs: (2,3) (4,5) (6,7) (8,11) (9,10) (12,15) (13,14) (16,19) (17,18)
            f_out[i,j,k,1]  = fp1
            f_out[i,j,k,2]  = fp3;  f_out[i,j,k,3]  = fp2
            f_out[i,j,k,4]  = fp5;  f_out[i,j,k,5]  = fp4
            f_out[i,j,k,6]  = fp7;  f_out[i,j,k,7]  = fp6
            f_out[i,j,k,8]  = fp11; f_out[i,j,k,11] = fp8
            f_out[i,j,k,9]  = fp10; f_out[i,j,k,10] = fp9
            f_out[i,j,k,12] = fp15; f_out[i,j,k,15] = fp12
            f_out[i,j,k,13] = fp14; f_out[i,j,k,14] = fp13
            f_out[i,j,k,16] = fp19; f_out[i,j,k,19] = fp16
            f_out[i,j,k,17] = fp18; f_out[i,j,k,18] = fp17
            ρ_out[i,j,k] = one(T); ux_out[i,j,k] = zero(T)
            uy_out[i,j,k] = zero(T); uz_out[i,j,k] = zero(T)
        else
            # Moments from D3Q19 velocity vectors (inline)
            ρ = fp1+fp2+fp3+fp4+fp5+fp6+fp7+fp8+fp9+fp10+fp11+fp12+fp13+fp14+fp15+fp16+fp17+fp18+fp19
            jx = fp2-fp3+fp8-fp9+fp10-fp11+fp12-fp13+fp14-fp15
            jy = fp4-fp5+fp8+fp9-fp10-fp11+fp16-fp17+fp18-fp19
            jz = fp6-fp7+fp12+fp13-fp14-fp15+fp16+fp17-fp18-fp19
            inv_ρ = one(T) / ρ
            ux = jx * inv_ρ; uy = jy * inv_ρ; uz = jz * inv_ρ
            usq = ux*ux + uy*uy + uz*uz

            f_out[i,j,k,1]  = fp1  - ω * (fp1  - feq_3d(Val(1),  ρ, ux, uy, uz, usq))
            f_out[i,j,k,2]  = fp2  - ω * (fp2  - feq_3d(Val(2),  ρ, ux, uy, uz, usq))
            f_out[i,j,k,3]  = fp3  - ω * (fp3  - feq_3d(Val(3),  ρ, ux, uy, uz, usq))
            f_out[i,j,k,4]  = fp4  - ω * (fp4  - feq_3d(Val(4),  ρ, ux, uy, uz, usq))
            f_out[i,j,k,5]  = fp5  - ω * (fp5  - feq_3d(Val(5),  ρ, ux, uy, uz, usq))
            f_out[i,j,k,6]  = fp6  - ω * (fp6  - feq_3d(Val(6),  ρ, ux, uy, uz, usq))
            f_out[i,j,k,7]  = fp7  - ω * (fp7  - feq_3d(Val(7),  ρ, ux, uy, uz, usq))
            f_out[i,j,k,8]  = fp8  - ω * (fp8  - feq_3d(Val(8),  ρ, ux, uy, uz, usq))
            f_out[i,j,k,9]  = fp9  - ω * (fp9  - feq_3d(Val(9),  ρ, ux, uy, uz, usq))
            f_out[i,j,k,10] = fp10 - ω * (fp10 - feq_3d(Val(10), ρ, ux, uy, uz, usq))
            f_out[i,j,k,11] = fp11 - ω * (fp11 - feq_3d(Val(11), ρ, ux, uy, uz, usq))
            f_out[i,j,k,12] = fp12 - ω * (fp12 - feq_3d(Val(12), ρ, ux, uy, uz, usq))
            f_out[i,j,k,13] = fp13 - ω * (fp13 - feq_3d(Val(13), ρ, ux, uy, uz, usq))
            f_out[i,j,k,14] = fp14 - ω * (fp14 - feq_3d(Val(14), ρ, ux, uy, uz, usq))
            f_out[i,j,k,15] = fp15 - ω * (fp15 - feq_3d(Val(15), ρ, ux, uy, uz, usq))
            f_out[i,j,k,16] = fp16 - ω * (fp16 - feq_3d(Val(16), ρ, ux, uy, uz, usq))
            f_out[i,j,k,17] = fp17 - ω * (fp17 - feq_3d(Val(17), ρ, ux, uy, uz, usq))
            f_out[i,j,k,18] = fp18 - ω * (fp18 - feq_3d(Val(18), ρ, ux, uy, uz, usq))
            f_out[i,j,k,19] = fp19 - ω * (fp19 - feq_3d(Val(19), ρ, ux, uy, uz, usq))
            ρ_out[i,j,k] = ρ
            ux_out[i,j,k] = ux; uy_out[i,j,k] = uy; uz_out[i,j,k] = uz
        end
    end
end

"""
    slbm_bgk_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid, geom, ω)

Fused SLBM + BGK step for D3Q19 on a curvilinear 3D mesh.
"""
function slbm_bgk_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                            geom::SLBMGeometry3D, ω)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    kernel! = slbm_bgk_step_3d_kernel!(backend)
    kernel!(f_out, f_in, ρ, ux, uy, uz, is_solid,
            geom.i_dep, geom.j_dep, geom.k_dep,
            geom.Nξ, geom.Nη, geom.Nζ, ET(ω),
            geom.periodic_ξ, geom.periodic_η, geom.periodic_ζ;
            ndrange=(geom.Nξ, geom.Nη, geom.Nζ))
end

# ===========================================================================
# Local relaxation rate for SLBM 3D on non-uniform meshes.
#
# Mirror of `compute_local_omega_2d` (D2Q9). On a stretched 3D mesh, the
# physical cell size varies and the collision τ must be adjusted per
# cell to maintain the target physical viscosity ν:
#
#   τ_local = (Δx_ref / Δx_local)² · (τ_ref − 0.5) + 0.5    (quadratic, SLBM)
#   τ_local =  (Δx_local / Δx_ref) · (τ_ref − 0.5) + 0.5    (linear, refinement)
# ===========================================================================

"""
    compute_local_omega_3d(mesh; ν, Λ=3/16, scaling=:quadratic)
        -> (s_plus_field, s_minus_field)

Precompute per-cell TRT relaxation rates `s_plus[i,j,k]` and
`s_minus[i,j,k]` on a 3D curvilinear mesh, accounting for the local
cell-size variation. Returns 3D arrays of size `Nξ × Nη × Nζ`.

`scaling=:quadratic` corresponds to SLBM with a single global Δt
(τ_local − 0.5 ∝ (Δx_ref/Δx_local)²); `scaling=:linear` is the
Filippova-Hänel rescaling for grid-refinement-style Δt ∝ Δx.

Local cell size estimated from the column-norm of the metric:
`Δx_local = (lξ · lη · lζ)^(1/3)` with `lξ = √(dXdξ² + dYdξ² + dZdξ²) · Δξ`.
"""
function compute_local_omega_3d(mesh::CurvilinearMesh3D{T};
                                  ν::Real, Λ::Real=3/16,
                                  scaling::Symbol=:quadratic) where {T}
    Nξ, Nη, Nζ = mesh.Nξ, mesh.Nη, mesh.Nζ
    denom_ξ = T(mesh.periodic_ξ ? Nξ : Nξ - 1)
    denom_η = T(mesh.periodic_η ? Nη : Nη - 1)
    denom_ζ = T(mesh.periodic_ζ ? Nζ : Nζ - 1)
    Δξ = one(T) / denom_ξ
    Δη = one(T) / denom_η
    Δζ = one(T) / denom_ζ

    s_plus_ref, s_minus_ref = trt_rates(ν; Λ=Λ)
    τ_plus_ref  = one(T) / T(s_plus_ref)
    τ_minus_ref = one(T) / T(s_minus_ref)

    sp = zeros(T, Nξ, Nη, Nζ)
    sm = zeros(T, Nξ, Nη, Nζ)

    @inbounds for k in 1:Nζ, j in 1:Nη, i in 1:Nξ
        lξ = sqrt(mesh.dXdξ[i,j,k]^2 + mesh.dYdξ[i,j,k]^2 + mesh.dZdξ[i,j,k]^2) * Δξ
        lη = sqrt(mesh.dXdη[i,j,k]^2 + mesh.dYdη[i,j,k]^2 + mesh.dZdη[i,j,k]^2) * Δη
        lζ = sqrt(mesh.dXdζ[i,j,k]^2 + mesh.dYdζ[i,j,k]^2 + mesh.dZdζ[i,j,k]^2) * Δζ
        dx_local = cbrt(lξ * lη * lζ)

        if scaling === :quadratic
            r_inv2 = (mesh.dx_ref / dx_local)^2
            τ_plus_local  = r_inv2 * (τ_plus_ref  - T(0.5)) + T(0.5)
            τ_minus_local = r_inv2 * (τ_minus_ref - T(0.5)) + T(0.5)
        else
            r = dx_local / mesh.dx_ref
            τ_plus_local  = r * (τ_plus_ref  - T(0.5)) + T(0.5)
            τ_minus_local = r * (τ_minus_ref - T(0.5)) + T(0.5)
        end
        sp[i,j,k] = one(T) / τ_plus_local
        sm[i,j,k] = one(T) / τ_minus_local
    end

    return sp, sm
end

# ===========================================================================
# q_wall precomputation for SLBM + LI-BB on a 3D curvilinear mesh.
#
# Mirror of `precompute_q_wall_slbm_cylinder_2d` — for each fluid node
# `(i, j, k)` and each D3Q19 direction `q`, if the computational neighbour
# `(i+cxq, j+cyq, k+czq)` is solid we ray-sphere-intersect the physical-
# space segment from the fluid node to the neighbour with a sphere of
# radius `R_body` centred at `(cx_body, cy_body, cz_body)`.
# ===========================================================================

"""
    precompute_q_wall_slbm_sphere_3d(mesh, is_solid,
        cx_body, cy_body, cz_body, R_body;
        FT=Float64)
        -> (q_wall, uw_link_x, uw_link_y, uw_link_z)

Precompute LI-BB cut fractions on a 3D curvilinear mesh for a stationary
sphere centred at `(cx_body, cy_body, cz_body)` with radius `R_body` in
physical space.

For each fluid node `(i, j, k)` and direction `q ∈ 2..19`, if the
computational neighbour is solid we solve the quadratic
`|P_f + t·(P_n − P_f) − P_c|² = R²` for the smallest positive `t ∈ (0, 1]`
and store it as `q_wall[i, j, k, q]`. Wall velocities are zero (returned
as freshly-allocated zero arrays); rotating-body / moving-body variants
can be added later by mirroring the 2D `omega_inner` branch.
"""
function precompute_q_wall_slbm_sphere_3d(
        mesh::CurvilinearMesh3D{T},
        is_solid::AbstractArray{Bool, 3},
        cx_body::Real, cy_body::Real, cz_body::Real, R_body::Real;
        FT::Type{<:AbstractFloat}=T) where {T}

    Nξ, Nη, Nζ = mesh.Nξ, mesh.Nη, mesh.Nζ
    cxs = velocities_x(D3Q19())
    cys = velocities_y(D3Q19())
    czs = velocities_z(D3Q19())

    q_wall    = zeros(FT, Nξ, Nη, Nζ, 19)
    uw_link_x = zeros(FT, Nξ, Nη, Nζ, 19)
    uw_link_y = zeros(FT, Nξ, Nη, Nζ, 19)
    uw_link_z = zeros(FT, Nξ, Nη, Nζ, 19)

    cxb, cyb, czb, Rb = FT(cx_body), FT(cy_body), FT(cz_body), FT(R_body)
    R² = Rb * Rb

    @inbounds for k in 1:Nζ, j in 1:Nη, i in 1:Nξ
        is_solid[i, j, k] && continue

        Xf = FT(mesh.X[i, j, k])
        Yf = FT(mesh.Y[i, j, k])
        Zf = FT(mesh.Z[i, j, k])

        for q in 2:19
            diq = cxs[q]; djq = cys[q]; dkq = czs[q]
            in_ = i + diq; jn_ = j + djq; kn_ = k + dkq

            mesh.periodic_ξ && (in_ = ((in_ - 1) % Nξ + Nξ) % Nξ + 1)
            mesh.periodic_η && (jn_ = ((jn_ - 1) % Nη + Nη) % Nη + 1)
            mesh.periodic_ζ && (kn_ = ((kn_ - 1) % Nζ + Nζ) % Nζ + 1)

            (in_ < 1 || in_ > Nξ ||
             jn_ < 1 || jn_ > Nη ||
             kn_ < 1 || kn_ > Nζ) && continue
            is_solid[in_, jn_, kn_] || continue

            Xn = FT(mesh.X[in_, jn_, kn_])
            Yn = FT(mesh.Y[in_, jn_, kn_])
            Zn = FT(mesh.Z[in_, jn_, kn_])

            dx = Xn - Xf; dy = Yn - Yf; dz = Zn - Zf
            fx = Xf - cxb; fy = Yf - cyb; fz = Zf - czb

            a = dx * dx + dy * dy + dz * dz
            b_coeff = FT(2) * (fx * dx + fy * dy + fz * dz)
            c_coeff = fx * fx + fy * fy + fz * fz - R²
            disc = b_coeff * b_coeff - FT(4) * a * c_coeff

            if disc < zero(FT)
                q_wall[i, j, k, q] = FT(0.5)
            else
                sd = sqrt(disc)
                t1 = (-b_coeff - sd) / (FT(2) * a)
                t2 = (-b_coeff + sd) / (FT(2) * a)
                t = (t1 > zero(FT) && t1 ≤ one(FT)) ? t1 :
                    (t2 > zero(FT) && t2 ≤ one(FT)) ? t2 : FT(0.5)
                q_wall[i, j, k, q] = t
            end
        end
    end

    return q_wall, uw_link_x, uw_link_y, uw_link_z
end

# ---------------------------------------------------------------------------
# Fused SLBM + TRT + LI-BB step in 3D, assembled from DSL bricks.
#
#   PullSLBM_3D → SolidInert_3D → ApplyLiBBPrePhase_3D →
#   Moments_3D → CollideTRTDirect_3D → WriteMoments_3D
#
# Same recipe as `slbm_trt_libb_step!` (2D) and `fused_trt_libb_v2_step_3d!`
# (Cartesian 3D), but the streaming brick is the trilinear semi-Lagrangian
# pull (`PullSLBM_3D`) rather than the halfway-BB pull-stream.
# ---------------------------------------------------------------------------

const _SLBM_TRT_LIBB_SPEC_3D = LBMSpec(
    PullSLBM_3D(), SolidInert_3D(),
    ApplyLiBBPrePhase_3D(),
    Moments_3D(), CollideTRTDirect_3D(),
    WriteMoments_3D();
    stencil = :D3Q19,
)

"""
    slbm_trt_libb_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                            q_wall, uw_link_x, uw_link_y, uw_link_z,
                            geom, ν; Λ=3/16)

Fused single-pass SLBM + TRT + LI-BB step on a 3D curvilinear mesh
(D3Q19). Combines semi-Lagrangian streaming, LI-BB pre-phase substitution
on cut links (Bouzidi for arbitrary `q_w ∈ (0, 1]`) and a TRT collision
with magic parameter `Λ`.

Reduces to the Cartesian halfway-BB + TRT + LI-BB pipeline when the mesh
is uniform Cartesian (departures land on neighbour nodes, trilinear
collapses to a single node read).
"""
function slbm_trt_libb_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                  q_wall, uw_link_x, uw_link_y, uw_link_z,
                                  geom::SLBMGeometry3D, ν; Λ::Real=3/16)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    s_plus, s_minus = trt_rates(ν; Λ=Λ)
    kernel! = build_lbm_kernel(backend, _SLBM_TRT_LIBB_SPEC_3D)
    kernel!(f_out, ρ, ux, uy, uz, f_in, is_solid,
            q_wall, uw_link_x, uw_link_y, uw_link_z,
            geom.i_dep, geom.j_dep, geom.k_dep,
            geom.Nξ, geom.Nη, geom.Nζ,
            ET(s_plus), ET(s_minus),
            geom.periodic_ξ, geom.periodic_η, geom.periodic_ζ;
            ndrange=(geom.Nξ, geom.Nη, geom.Nζ))
end

# ---------------------------------------------------------------------------
# Local-τ variant: per-cell s_plus[i,j,k], s_minus[i,j,k] for SLBM on
# stretched 3D meshes. Same brick assembly as above, but the collision
# brick reads relaxation rates from device-side 3D arrays.
# ---------------------------------------------------------------------------

const _SLBM_TRT_LIBB_LOCAL_SPEC_3D = LBMSpec(
    PullSLBM_3D(), SolidInert_3D(),
    ApplyLiBBPrePhase_3D(),
    Moments_3D(), CollideTRTLocalDirect_3D(),
    WriteMoments_3D();
    stencil = :D3Q19,
)

"""
    slbm_trt_libb_step_local_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                  q_wall, uw_link_x, uw_link_y, uw_link_z,
                                  geom, sp_field, sm_field)

SLBM + TRT + LI-BB step (D3Q19) with PER-CELL relaxation rates from the
device-side 3D arrays `sp_field` and `sm_field` (e.g. produced by
`compute_local_omega_3d` and copied to the backend). Required on
stretched curvilinear meshes where the local cell size — and therefore τ
— varies per cell.
"""
function slbm_trt_libb_step_local_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                        q_wall, uw_link_x, uw_link_y, uw_link_z,
                                        geom::SLBMGeometry3D,
                                        sp_field, sm_field)
    backend = KernelAbstractions.get_backend(f_in)
    kernel! = build_lbm_kernel(backend, _SLBM_TRT_LIBB_LOCAL_SPEC_3D)
    kernel!(f_out, ρ, ux, uy, uz, f_in, is_solid,
            q_wall, uw_link_x, uw_link_y, uw_link_z,
            geom.i_dep, geom.j_dep, geom.k_dep,
            geom.Nξ, geom.Nη, geom.Nζ,
            sp_field, sm_field,
            geom.periodic_ξ, geom.periodic_η, geom.periodic_ζ;
            ndrange=(geom.Nξ, geom.Nη, geom.Nζ))
end
