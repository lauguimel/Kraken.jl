"""Build is_solid mask from geometry regions (condition expressions or STL files)."""
function _apply_geometry!(is_solid, setup::SimulationSetup, dx, dy)
    Nx, Ny = setup.domain.Nx, setup.domain.Ny
    Lx, Ly = setup.domain.Lx, setup.domain.Ly

    has_fluid_region = any(r -> r.kind == :fluid, setup.regions)
    solid_cpu = has_fluid_region ? ones(Bool, Nx, Ny) : zeros(Bool, Nx, Ny)

    for region in setup.regions
        if region.stl !== nothing
            # STL-based geometry
            stl_mask = _voxelize_stl_region(region.stl, Nx, Ny, dx, dy)
            for j in 1:Ny, i in 1:Nx
                if region.kind == :fluid && stl_mask[i, j]
                    solid_cpu[i, j] = false
                elseif region.kind == :obstacle && stl_mask[i, j]
                    solid_cpu[i, j] = true
                end
            end
        else
            # Expression-based geometry
            for j in 1:Ny, i in 1:Nx
                x = (i - 0.5) * dx
                y = (j - 0.5) * dy
                result = evaluate(region.condition; x=x, y=y, z=0.0,
                                 Lx=Lx, Ly=Ly, dx=dx, dy=dy)
                if region.kind == :fluid && result
                    solid_cpu[i, j] = false
                elseif region.kind == :obstacle && result
                    solid_cpu[i, j] = true
                end
            end
        end
    end

    copyto!(is_solid, solid_cpu)
end

"""Load and voxelize an STL file for a 2D simulation (z-plane cross-section)."""
function _voxelize_stl_region(stl_src::STLSource, Nx, Ny, dx, dy)
    mesh = read_stl(stl_src.file)
    if stl_src.scale != 1.0 || stl_src.translate != (0.0, 0.0, 0.0)
        mesh = transform_mesh(mesh; scale=stl_src.scale, translate=stl_src.translate)
    end
    return voxelize_2d(mesh, Nx, Ny, dx, dy; z_slice=stl_src.z_slice)
end

"""Load and voxelize an STL file for a 3D simulation."""
function _voxelize_stl_region_3d(stl_src::STLSource, Nx, Ny, Nz, dx)
    mesh = read_stl(stl_src.file)
    if stl_src.scale != 1.0 || stl_src.translate != (0.0, 0.0, 0.0)
        mesh = transform_mesh(mesh; scale=stl_src.scale, translate=stl_src.translate)
    end
    return voxelize_3d(mesh, Nx, Ny, Nz, dx, dx, dx)
end

# --- 3D geometry helpers ---

"""Evaluate obstacle geometry on 3D grid."""
function _apply_geometry_3d!(is_solid, setup::SimulationSetup, dx::Float64)
    Nx, Ny, Nz = setup.domain.Nx, setup.domain.Ny, setup.domain.Nz
    Lx, Ly, Lz = setup.domain.Lx, setup.domain.Ly, setup.domain.Lz

    isempty(setup.regions) && return

    has_fluid_region = any(r -> r.kind == :fluid, setup.regions)
    solid_cpu = has_fluid_region ? ones(Bool, Nx, Ny, Nz) : zeros(Bool, Nx, Ny, Nz)

    for region in setup.regions
        if region.stl !== nothing
            stl_mask = _voxelize_stl_region_3d(region.stl, Nx, Ny, Nz, dx)
            for k in 1:Nz, j in 1:Ny, i in 1:Nx
                if region.kind == :fluid && stl_mask[i, j, k]
                    solid_cpu[i, j, k] = false
                elseif region.kind == :obstacle && stl_mask[i, j, k]
                    solid_cpu[i, j, k] = true
                end
            end
        else
            region.condition === nothing && continue
            for k in 1:Nz, j in 1:Ny, i in 1:Nx
                x = (i - 0.5) * dx
                y = (j - 0.5) * dx
                z = (k - 0.5) * dx
                result = evaluate(region.condition; x=x, y=y, z=z,
                                 Lx=Lx, Ly=Ly, Lz=Lz, dx=dx, dy=dx, dz=dx)
                if region.kind == :fluid && result
                    solid_cpu[i, j, k] = false
                elseif region.kind == :obstacle && result
                    solid_cpu[i, j, k] = true
                end
            end
        end
    end
    copyto!(is_solid, solid_cpu)
end

"""Evaluate obstacle geometry on a 3D fine-grid patch."""
function _apply_patch_geometry_3d!(patch::RefinementPatch3D{T},
                                    setup::SimulationSetup) where T
    Nx_p, Ny_p, Nz_p = patch.Nx, patch.Ny, patch.Nz
    ng = patch.n_ghost
    dx_f = Float64(patch.dx)
    Lx, Ly, Lz = setup.domain.Lx, setup.domain.Ly, setup.domain.Lz

    isempty(setup.regions) && return

    has_fluid_region = any(r -> r.kind == :fluid, setup.regions)
    solid_cpu = has_fluid_region ? ones(Bool, Nx_p, Ny_p, Nz_p) : zeros(Bool, Nx_p, Ny_p, Nz_p)

    for region in setup.regions
        region.stl !== nothing && continue
        region.condition === nothing && continue
        for kf in 1:Nz_p, jf in 1:Ny_p, if_ in 1:Nx_p
            x = Float64(patch.x_min) + (if_ - ng - 0.5) * dx_f
            y = Float64(patch.y_min) + (jf - ng - 0.5) * dx_f
            z = Float64(patch.z_min) + (kf - ng - 0.5) * dx_f
            result = evaluate(region.condition; x=x, y=y, z=z,
                             Lx=Lx, Ly=Ly, Lz=Lz, dx=dx_f, dy=dx_f, dz=dx_f)
            if region.kind == :fluid && result
                solid_cpu[if_, jf, kf] = false
            elseif region.kind == :obstacle && result
                solid_cpu[if_, jf, kf] = true
            end
        end
    end
    copyto!(patch.is_solid, solid_cpu)
end

"""Evaluate obstacle geometry on a fine-grid patch at its native resolution."""
function _apply_patch_geometry!(patch::RefinementPatch{T},
                                setup::SimulationSetup) where T
    Nx_p, Ny_p = patch.Nx, patch.Ny
    ng = patch.n_ghost
    dx_f = Float64(patch.dx)
    Lx, Ly = setup.domain.Lx, setup.domain.Ly

    isempty(setup.regions) && return

    has_fluid_region = any(r -> r.kind == :fluid, setup.regions)
    solid_cpu = has_fluid_region ? ones(Bool, Nx_p, Ny_p) : zeros(Bool, Nx_p, Ny_p)

    for region in setup.regions
        region.stl !== nothing && continue  # TODO: STL voxelization on patches
        region.condition === nothing && continue
        for jf in 1:Ny_p, if_ in 1:Nx_p
            x = Float64(patch.x_min) + (if_ - ng - 0.5) * dx_f
            y = Float64(patch.y_min) + (jf - ng - 0.5) * dx_f
            result = evaluate(region.condition; x=x, y=y, z=0.0,
                             Lx=Lx, Ly=Ly, dx=dx_f, dy=dx_f)
            if region.kind == :fluid && result
                solid_cpu[if_, jf] = false
            elseif region.kind == :obstacle && result
                solid_cpu[if_, jf] = true
            end
        end
    end
    copyto!(patch.is_solid, solid_cpu)
end
