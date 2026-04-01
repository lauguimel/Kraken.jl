using WriteVTK

"""
    write_vtk(filename, Nx, Ny, dx, fields::Dict{String, <:AbstractMatrix})

Write 2D field data on a rectilinear grid to a VTK `.vtr` file.
"""
function write_vtk(filename::String, Nx::Int, Ny::Int, dx::Float64,
                   fields::Dict{String, <:AbstractMatrix})
    xs = collect(range(0.0, step=dx, length=Nx + 1))
    ys = collect(range(0.0, step=dx, length=Ny + 1))

    vtk_grid(filename, xs, ys) do vtk
        for (name, data) in fields
            vtk[name] = Array(data)
        end
    end
end

"""
    write_vtk(filename, Nx, Ny, Nz, dx, fields::Dict{String, <:AbstractArray{T,3}})

Write 3D field data on a rectilinear grid to a VTK `.vtr` file.
"""
function write_vtk(filename::String, Nx::Int, Ny::Int, Nz::Int, dx::Float64,
                   fields::Dict{String, <:AbstractArray{T,3} where T})
    xs = collect(range(0.0, step=dx, length=Nx + 1))
    ys = collect(range(0.0, step=dx, length=Ny + 1))
    zs = collect(range(0.0, step=dx, length=Nz + 1))

    vtk_grid(filename, xs, ys, zs) do vtk
        for (name, data) in fields
            vtk[name] = Array(data)
        end
    end
end

"""
    write_vtk_to_pvd(pvd, filename, Nx, Ny, dx, fields, time)

Write a 2D VTK snapshot and register it in a PVD time-series collection.
"""
function write_vtk_to_pvd(pvd, filename::String, Nx::Int, Ny::Int, dx::Float64,
                          fields::Dict{String, <:AbstractMatrix}, time::Float64)
    xs = collect(range(0.0, step=dx, length=Nx + 1))
    ys = collect(range(0.0, step=dx, length=Ny + 1))

    vtk_grid(filename, xs, ys) do vtk
        for (name, data) in fields
            vtk[name] = Array(data)
        end
        pvd[time] = vtk
    end
end

"""
    write_vtk_to_pvd(pvd, filename, Nx, Ny, Nz, dx, fields, time)

Write a 3D VTK snapshot and register it in a PVD time-series collection.
"""
function write_vtk_to_pvd(pvd, filename::String, Nx::Int, Ny::Int, Nz::Int, dx::Float64,
                          fields::Dict{String, <:AbstractArray{T,3} where T}, time::Float64)
    xs = collect(range(0.0, step=dx, length=Nx + 1))
    ys = collect(range(0.0, step=dx, length=Ny + 1))
    zs = collect(range(0.0, step=dx, length=Nz + 1))

    vtk_grid(filename, xs, ys, zs) do vtk
        for (name, data) in fields
            vtk[name] = Array(data)
        end
        pvd[time] = vtk
    end
end

"""
    create_pvd(filename) -> WriteVTK.CollectionFile

Create a ParaView Data (`.pvd`) collection file for time series output.
"""
function create_pvd(filename::String)
    return paraview_collection(filename)
end

"""
    setup_output_dir(path::String) -> String

Create the output directory (and parents) if it does not exist.
Returns the absolute path.
"""
function setup_output_dir(path::String)
    mkpath(path)
    return abspath(path)
end

"""
    write_snapshot_2d!(output_dir, step, Nx, Ny, dx, fields; pvd=nothing, time=0.0)

Write a 2D VTK snapshot with a zero-padded filename `snapshot_NNNNNNN`.
If `pvd` is provided, the snapshot is also registered in the PVD collection.
"""
function write_snapshot_2d!(output_dir::String, step::Int, Nx::Int, Ny::Int, dx,
                            fields::Dict{String, <:AbstractMatrix};
                            pvd=nothing, time::Float64=0.0)
    tag = lpad(step, 7, '0')
    filename = joinpath(output_dir, "snapshot_$tag")
    dx_f = Float64(dx)
    if pvd !== nothing
        write_vtk_to_pvd(pvd, filename, Nx, Ny, dx_f, fields, Float64(time))
    else
        write_vtk(filename, Nx, Ny, dx_f, fields)
    end
end

"""
    write_snapshot_3d!(output_dir, step, Nx, Ny, Nz, dx, fields; pvd=nothing, time=0.0)

Write a 3D VTK snapshot with a zero-padded filename `snapshot_NNNNNNN`.
If `pvd` is provided, the snapshot is also registered in the PVD collection.

Fields can be:
- `String => AbstractArray{T,3}` for scalar fields
- `String => Tuple{AbstractArray,AbstractArray,AbstractArray}` for 3-component vector fields
"""
function write_snapshot_3d!(output_dir::String, step::Int, Nx::Int, Ny::Int, Nz::Int, dx,
                            fields::Dict;
                            pvd=nothing, time::Float64=0.0)
    tag = lpad(step, 7, '0')
    filename = joinpath(output_dir, "snapshot_$tag")
    dx_f = Float64(dx)

    xs = collect(range(0.0, step=dx_f, length=Nx + 1))
    ys = collect(range(0.0, step=dx_f, length=Ny + 1))
    zs = collect(range(0.0, step=dx_f, length=Nz + 1))

    vtk_grid(filename, xs, ys, zs) do vtk
        for (name, data) in fields
            if data isa Tuple{AbstractArray, AbstractArray, AbstractArray}
                ux_cpu = Array(data[1])
                uy_cpu = Array(data[2])
                uz_cpu = Array(data[3])
                vtk[name] = (ux_cpu, uy_cpu, uz_cpu)
            else
                vtk[name] = Array(data)
            end
        end
        if pvd !== nothing
            pvd[time] = vtk
        end
    end
end

# --- Multi-block output for grid refinement ---

"""
    write_vtk_multiblock(filename, blocks::Vector{<:NamedTuple})

Write a multi-block VTK `.vtm` file. Each block is a NamedTuple with:
- `name::String`
- `Nx::Int, Ny::Int`
- `dx::Float64`
- `x_min::Float64, y_min::Float64`
- `fields::Dict{String, <:AbstractMatrix}`
"""
function write_vtk_multiblock(filename::String,
                               blocks::Vector{<:NamedTuple})
    vtk_multiblock(filename) do vtm
        for blk in blocks
            xs = collect(range(blk.x_min, step=blk.dx, length=blk.Nx + 1))
            ys = collect(range(blk.y_min, step=blk.dx, length=blk.Ny + 1))
            vtk_grid(vtm, xs, ys) do vtk
                for (fname, data) in blk.fields
                    vtk[fname] = Array(data)
                end
            end
        end
    end
end

"""
    write_vtk_multiblock_to_pvd(pvd, filename, blocks, time)

Write a multi-block snapshot and register it in a PVD time-series.
"""
function write_vtk_multiblock_to_pvd(pvd, filename::String,
                                      blocks::Vector{<:NamedTuple},
                                      time::Float64)
    vtk_multiblock(filename) do vtm
        for blk in blocks
            xs = collect(range(blk.x_min, step=blk.dx, length=blk.Nx + 1))
            ys = collect(range(blk.y_min, step=blk.dx, length=blk.Ny + 1))
            vtk_grid(vtm, xs, ys) do vtk
                for (fname, data) in blk.fields
                    vtk[fname] = Array(data)
                end
            end
        end
        pvd[time] = vtm
    end
end

"""
    write_snapshot_refined_2d!(output_dir, step, domain, rho, ux, uy;
                               pvd=nothing, time=0.0)

Write a multi-block VTK snapshot for a refined domain.
Includes the base grid and all refinement patches.
"""
function write_snapshot_refined_2d!(output_dir::String, step::Int,
                                    domain::RefinedDomain{T},
                                    rho, ux, uy;
                                    pvd=nothing, time::Float64=0.0) where T
    blocks = NamedTuple[]

    # Base grid block
    push!(blocks, (
        name="base",
        Nx=domain.base_Nx, Ny=domain.base_Ny,
        dx=Float64(domain.base_dx),
        x_min=0.0, y_min=0.0,
        fields=Dict("rho" => rho, "ux" => ux, "uy" => uy)
    ))

    # Patch blocks (inner region only, skip ghost cells)
    for patch in domain.patches
        ng = patch.n_ghost
        inner_rho = @view patch.rho[ng+1:ng+patch.Nx_inner, ng+1:ng+patch.Ny_inner]
        inner_ux  = @view patch.ux[ng+1:ng+patch.Nx_inner, ng+1:ng+patch.Ny_inner]
        inner_uy  = @view patch.uy[ng+1:ng+patch.Nx_inner, ng+1:ng+patch.Ny_inner]
        push!(blocks, (
            name=patch.name,
            Nx=patch.Nx_inner, Ny=patch.Ny_inner,
            dx=Float64(patch.dx),
            x_min=Float64(patch.x_min), y_min=Float64(patch.y_min),
            fields=Dict("rho" => inner_rho, "ux" => inner_ux, "uy" => inner_uy)
        ))
    end

    tag = lpad(step, 7, '0')
    filename = joinpath(output_dir, "snapshot_$tag")
    if pvd !== nothing
        write_vtk_multiblock_to_pvd(pvd, filename, blocks, Float64(time))
    else
        write_vtk_multiblock(filename, blocks)
    end
end
