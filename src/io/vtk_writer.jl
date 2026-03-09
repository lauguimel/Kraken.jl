using WriteVTK

"""
    write_vtk(filename, nx, ny, dx, fields::Dict{String, <:AbstractMatrix})

Write field data on a 2D rectilinear grid to a VTK `.vtr` file.

# Arguments
- `filename`: output path (without extension — `.vtr` is appended by WriteVTK).
- `nx`, `ny`: number of cells in x and y directions.
- `dx`: uniform grid spacing (same in both directions).
- `fields`: dictionary mapping field names to `(nx, ny)` arrays.

# Returns
A vector of output file paths (typically one `.vtr` file).

# Example
```julia
write_vtk("output", 8, 8, 0.125, Dict("P" => rand(8,8)))
```
"""
function write_vtk(filename::String, nx::Int, ny::Int, dx::Float64,
                   fields::Dict{String, <:AbstractMatrix})
    xs = collect(range(0.0, step=dx, length=nx + 1))
    ys = collect(range(0.0, step=dx, length=ny + 1))

    vtk_grid(filename, xs, ys) do vtk
        for (name, data) in fields
            vtk[name] = data
        end
    end
end

"""
    write_vtk_to_pvd(pvd, filename, nx, ny, dx, fields, time)

Write a VTK snapshot and register it in a PVD time-series collection.

# Arguments
- `pvd`: collection handle from [`create_pvd`](@ref).
- `filename`: output path (without extension).
- `nx`, `ny`: number of cells in x and y directions.
- `dx`: uniform grid spacing.
- `fields`: dictionary mapping field names to `(nx, ny)` arrays.
- `time`: simulation time for this snapshot.

# Returns
A vector of output file paths.

# Example
```julia
pvd = create_pvd("series")
write_vtk_to_pvd(pvd, "step_1", 8, 8, 0.125, Dict("P" => p), 0.1)
vtk_save(pvd)
```
"""
function write_vtk_to_pvd(pvd, filename::String, nx::Int, ny::Int, dx::Float64,
                          fields::Dict{String, <:AbstractMatrix}, time::Float64)
    xs = collect(range(0.0, step=dx, length=nx + 1))
    ys = collect(range(0.0, step=dx, length=ny + 1))

    vtk_grid(filename, xs, ys) do vtk
        for (name, data) in fields
            vtk[name] = data
        end
        pvd[time] = vtk
    end
end

"""
    create_pvd(filename) -> WriteVTK.CollectionFile

Create a ParaView Data (`.pvd`) collection file for time series output.

# Arguments
- `filename`: output path (without extension).

# Returns
A `CollectionFile` handle. Add timesteps with [`write_vtk_to_pvd`](@ref),
then close with `vtk_save(pvd)`.

# Example
```julia
pvd = create_pvd("timeseries")
write_vtk_to_pvd(pvd, "step_1", 8, 8, 0.125, fields, 0.1)
vtk_save(pvd)
```
"""
function create_pvd(filename::String)
    return paraview_collection(filename)
end
