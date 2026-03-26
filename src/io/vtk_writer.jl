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
