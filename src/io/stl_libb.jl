# =====================================================================
# STL → LI-BB cut-fraction fields.
#
# Given an STL mesh + a regular grid, precompute `(q_wall, is_solid)`
# arrays suitable for `fused_trt_libb_v2_step!` (2D) or
# `fused_trt_libb_v2_step_3d!` (3D).
#
# CURRENT IMPLEMENTATION: halfway-BB default — any link from a fluid
# cell whose neighbour falls inside the voxelised solid gets
# `q_wall = 0.5`. This gives Ginzburg-exact halfway-BB for arbitrary
# geometry. Sub-cell accurate q_w for non-halfway walls (ray-cast
# against the STL triangles per link) is tracked as future work.
# =====================================================================

"""
    precompute_q_wall_from_stl_2d(mesh, Nx, Ny, dx, dy;
                                   z_slice=0.0, FT=Float64)
        -> (q_wall, is_solid)

Voxelise `mesh` into a 2D boolean mask at z = z_slice, then flag every
fluid-cell link whose neighbour is inside the mask with `q_w = 0.5`.
Returns a `(Nx, Ny, 9)` cut-fraction array + a `(Nx, Ny)` bool mask.
"""
function precompute_q_wall_from_stl_2d(mesh::STLMesh{T},
                                        Nx::Int, Ny::Int,
                                        dx::Real, dy::Real;
                                        z_slice::Real=0.0,
                                        FT::Type{<:AbstractFloat}=Float64) where {T}
    is_solid = voxelize_2d(mesh, Nx, Ny, dx, dy; z_slice=z_slice)
    q_wall = zeros(FT, Nx, Ny, 9)
    cxs = velocities_x(D2Q9())
    cys = velocities_y(D2Q9())
    half = FT(0.5)
    @inbounds for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        for q in 2:9
            ni = i + Int(cxs[q])
            nj = j + Int(cys[q])
            if 1 <= ni <= Nx && 1 <= nj <= Ny && is_solid[ni, nj]
                q_wall[i, j, q] = half
            end
        end
    end
    return q_wall, is_solid
end

"""
    precompute_q_wall_from_stl_3d(mesh, Nx, Ny, Nz, dx, dy, dz;
                                   FT=Float64)
        -> (q_wall, is_solid)

3D version: voxelise `mesh` into a 3D boolean mask, flag D3Q19
fluid-cell links that cross into the mask with `q_w = 0.5`.
"""
function precompute_q_wall_from_stl_3d(mesh::STLMesh{T},
                                        Nx::Int, Ny::Int, Nz::Int,
                                        dx::Real, dy::Real, dz::Real;
                                        FT::Type{<:AbstractFloat}=Float64) where {T}
    is_solid = voxelize_3d(mesh, Nx, Ny, Nz, dx, dy, dz)
    q_wall = zeros(FT, Nx, Ny, Nz, 19)
    cxs = velocities_x(D3Q19())
    cys = velocities_y(D3Q19())
    czs = velocities_z(D3Q19())
    half = FT(0.5)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        is_solid[i, j, k] && continue
        for q in 2:19
            ni = i + Int(cxs[q])
            nj = j + Int(cys[q])
            nk = k + Int(czs[q])
            if 1 <= ni <= Nx && 1 <= nj <= Ny && 1 <= nk <= Nz &&
                is_solid[ni, nj, nk]
                q_wall[i, j, k, q] = half
            end
        end
    end
    return q_wall, is_solid
end
