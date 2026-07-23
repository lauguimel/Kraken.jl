function _has_stl_libb_obstacle(setup::SimulationSetup)
    return any(r -> r.stl !== nothing && r.bc_type === :libb, setup.regions)
end

function _halfway_q_wall_from_mask_2d(is_solid_cpu::AbstractArray{Bool},
                                      ::Type{T}) where T
    Nx, Ny = size(is_solid_cpu)
    q_wall = zeros(T, Nx, Ny, 9)
    cxs = velocities_x(D2Q9())
    cys = velocities_y(D2Q9())
    half = T(0.5)

    @inbounds for j in 1:Ny, i in 1:Nx
        is_solid_cpu[i, j] && continue
        for q in 2:9
            ni = i + Int(cxs[q])
            nj = j + Int(cys[q])
            if 1 <= ni <= Nx && 1 <= nj <= Ny && is_solid_cpu[ni, nj]
                q_wall[i, j, q] = half
            end
        end
    end
    return q_wall
end

function _precompute_stl_libb_q_wall_2d(is_solid_cpu::AbstractArray{Bool},
                                        setup::SimulationSetup, dx, dy,
                                        ::Type{T}) where T
    Nx, Ny = setup.domain.Nx, setup.domain.Ny
    q_wall = _halfway_q_wall_from_mask_2d(is_solid_cpu, T)

    for region in setup.regions
        (region.stl !== nothing && region.bc_type === :libb) || continue
        region.kind == :obstacle || throw(ArgumentError(
            "wall=libb is only supported on STL Obstacle regions"))

        stl_src = region.stl
        mesh = read_stl(stl_src.file)
        if stl_src.scale != 1.0 || stl_src.translate != (0.0, 0.0, 0.0)
            mesh = transform_mesh(mesh; scale=stl_src.scale,
                                  translate=stl_src.translate)
        end
        q_stl, _ = precompute_q_wall_from_stl_2d(mesh, Nx, Ny, dx, dy;
                                                 z_slice=stl_src.z_slice,
                                                 FT=T, sub_cell=true)
        @inbounds for q in 2:9, j in 1:Ny, i in 1:Nx
            if q_stl[i, j, q] > zero(T) && q_wall[i, j, q] > zero(T)
                q_wall[i, j, q] = q_stl[i, j, q]
            end
        end
    end

    any(x -> x != zero(T), q_wall) || throw(ArgumentError(
        "wall=libb was selected, but no fluid-solid cut links were found"))
    return q_wall
end

function _precompute_stl_libb_q_wall_3d(is_solid_cpu::AbstractArray{Bool},
                                        setup::SimulationSetup, dx,
                                        ::Type{T}) where T
    Nx, Ny, Nz = setup.domain.Nx, setup.domain.Ny, setup.domain.Nz
    expected_size = (Nx, Ny, Nz)
    size(is_solid_cpu) == expected_size || throw(ArgumentError(
        "3D LI-BB solid mask size $(size(is_solid_cpu)) does not match domain $(expected_size)"))
    q_wall = zeros(T, Nx, Ny, Nz, 19)

    for region in setup.regions
        (region.stl !== nothing && region.bc_type === :libb) || continue
        region.kind == :obstacle || throw(ArgumentError(
            "wall=libb is only supported on STL Obstacle regions"))

        stl_src = region.stl
        mesh = read_stl(stl_src.file)
        if stl_src.scale != 1.0 || stl_src.translate != (0.0, 0.0, 0.0)
            mesh = transform_mesh(mesh; scale=stl_src.scale,
                                  translate=stl_src.translate)
        end
        q_stl, _ = precompute_q_wall_from_stl_3d(mesh, Nx, Ny, Nz,
                                                 dx, dx, dx;
                                                 FT=T, sub_cell=true)
        @inbounds for q in 2:19, k in 1:Nz, j in 1:Ny, i in 1:Nx
            if q_stl[i, j, k, q] > zero(T)
                if q_wall[i, j, k, q] == zero(T) ||
                   q_stl[i, j, k, q] < q_wall[i, j, k, q]
                    q_wall[i, j, k, q] = q_stl[i, j, k, q]
                end
            end
        end
    end

    any(x -> x != zero(T), q_wall) || throw(ArgumentError(
        "wall=libb was selected, but no fluid-solid cut links were found"))
    return q_wall
end
