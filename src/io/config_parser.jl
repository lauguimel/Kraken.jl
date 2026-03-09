using YAML

"""
    GeometryConfig

Geometry specification: domain extent and grid resolution.

# Fields
- `domain::Tuple{Float64, Float64}`: physical domain size `(Lx, Ly)`.
- `resolution::Tuple{Int, Int}`: number of cells `(nx, ny)`.
"""
struct GeometryConfig
    domain::Tuple{Float64, Float64}
    resolution::Tuple{Int, Int}
end

"""
    PhysicsConfig

Physical model parameters.

# Fields
- `Re::Float64`: Reynolds number.
- `equation::String`: governing equation identifier (e.g. `"navier-stokes"`).
"""
struct PhysicsConfig
    Re::Float64
    equation::String
end

"""
    BCConfig

Boundary condition specification for a single boundary.

# Fields
- `type::String`: BC type (`"wall"`, `"inlet"`, `"outlet"`, etc.).
- `values::Dict{String, Any}`: extra parameters (e.g. velocity components).
"""
struct BCConfig
    type::String
    values::Dict{String, Any}
end

"""
    OutputConfig

Output settings.

# Fields
- `format::String`: output format (`"vtk"`).
- `frequency::Int`: write interval in timesteps.
"""
struct OutputConfig
    format::String
    frequency::Int
end

"""
    StudyConfig

Time-stepping / solver study parameters.

# Fields
- `type::String`: `"transient"` or `"steady"`.
- `dt::Union{Float64, String}`: timestep size or `"auto"`.
- `max_steps::Int`: maximum number of iterations.
- `convergence_tol::Float64`: convergence tolerance.
"""
struct StudyConfig
    type::String
    dt::Union{Float64, String}
    max_steps::Int
    convergence_tol::Float64
end

"""
    SimulationConfig

Top-level configuration aggregating all sub-configs.

# Fields
- `geometry::GeometryConfig`
- `physics::PhysicsConfig`
- `boundary_conditions::Dict{String, BCConfig}`
- `study::StudyConfig`
- `output::OutputConfig`
"""
struct SimulationConfig
    geometry::GeometryConfig
    physics::PhysicsConfig
    boundary_conditions::Dict{String, BCConfig}
    study::StudyConfig
    output::OutputConfig
end

"""
    load_config(filename::String) -> SimulationConfig

Parse a YAML configuration file and return a typed `SimulationConfig`.

Missing fields are filled with sensible defaults:
- `domain`: `(1.0, 1.0)`
- `resolution`: `(32, 32)`
- `Re`: `100.0`
- `equation`: `"navier-stokes"`
- `study.type`: `"transient"`
- `study.dt`: `"auto"`
- `study.max_steps`: `1000`
- `study.convergence_tol`: `1e-6`
- `output.format`: `"vtk"`
- `output.frequency`: `100`

# Example
```julia
cfg = load_config("examples/cavity.yaml")
cfg.geometry.resolution  # (64, 64)
```

# Returns
- `SimulationConfig`: a fully populated configuration object.

See also: [`SimulationConfig`](@ref), [`GeometryConfig`](@ref), [`PhysicsConfig`](@ref)
"""
function load_config(filename::String)::SimulationConfig
    raw = YAML.load_file(filename)

    # --- Geometry ---
    geo_raw = get(raw, "geometry", Dict())
    domain_raw = get(geo_raw, "domain", [1.0, 1.0])
    res_raw = get(geo_raw, "resolution", [32, 32])
    geometry = GeometryConfig(
        (Float64(domain_raw[1]), Float64(domain_raw[2])),
        (Int(res_raw[1]), Int(res_raw[2]))
    )

    # --- Physics ---
    phys_raw = get(raw, "physics", Dict())
    physics = PhysicsConfig(
        Float64(get(phys_raw, "Re", 100.0)),
        String(get(phys_raw, "equation", "navier-stokes"))
    )

    # --- Boundary conditions ---
    bc_raw = get(raw, "boundary_conditions", Dict())
    bcs = Dict{String, BCConfig}()
    for (name, bc_data) in bc_raw
        bc_type = get(bc_data, "type", "wall")
        # Collect everything except "type" into values
        vals = Dict{String, Any}()
        for (k, v) in bc_data
            k == "type" && continue
            vals[k] = v
        end
        bcs[name] = BCConfig(String(bc_type), vals)
    end

    # --- Study ---
    study_raw = get(raw, "study", Dict())
    dt_raw = get(study_raw, "dt", "auto")
    dt_val = dt_raw isa Number ? Float64(dt_raw) : String(dt_raw)
    study = StudyConfig(
        String(get(study_raw, "type", "transient")),
        dt_val,
        Int(get(study_raw, "max_steps", 1000)),
        Float64(get(study_raw, "convergence_tol", 1e-6))
    )

    # --- Output ---
    out_raw = get(raw, "output", Dict())
    output = OutputConfig(
        String(get(out_raw, "format", "vtk")),
        Int(get(out_raw, "frequency", 100))
    )

    return SimulationConfig(geometry, physics, bcs, study, output)
end
