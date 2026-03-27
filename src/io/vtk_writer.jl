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
