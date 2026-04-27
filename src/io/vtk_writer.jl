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
    open_paraview(output_dir::String; name::String="")

Open ParaView on the `.pvd` time-series file found in `output_dir`.

If `name` is given, looks for `<name>.pvd`; otherwise picks the first `.pvd`
file in the directory.  Falls back to opening the directory itself if no `.pvd`
is found (ParaView can browse `.vtr` files manually).

Requires `paraview` to be on `PATH`.

# Examples
```julia
# After running a simulation with VTK output:
open_paraview("output/")

# Specific PVD file:
open_paraview("output/", name="cavity")

# Remote HPC: rsync then view locally
# rsync -avz aqua:~/runs/cavity/output/ output/
# open_paraview("output/")
```
"""
function open_paraview(output_dir::String; name::String="")
    dir = abspath(output_dir)
    if !isdir(dir)
        error("Output directory not found: $dir")
    end

    # Find .pvd file
    if !isempty(name)
        pvd = joinpath(dir, name * ".pvd")
    else
        pvds = filter(f -> endswith(f, ".pvd"), readdir(dir))
        pvd = isempty(pvds) ? "" : joinpath(dir, first(pvds))
    end

    target = isempty(pvd) || !isfile(pvd) ? dir : pvd

    # Detect ParaView executable
    exe = if Sys.isapple()
        # Try common macOS app locations (newest first)
        candidates = [
            "/Applications/ParaView-6.0.0-RC1.app/Contents/MacOS/paraview",
            "/Applications/ParaView-5.13.3.app/Contents/MacOS/paraview",
        ]
        idx = findfirst(isfile, candidates)
        idx !== nothing ? candidates[idx] : "paraview"
    else
        "paraview"
    end

    @info "Opening ParaView on $target"
    run(`$exe $target`; wait=false)
    return target
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

