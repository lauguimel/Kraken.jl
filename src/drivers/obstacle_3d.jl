# CPU-only population halo for the D3Q19 pull brick's eager edge reads.
struct _CPUHaloArray4{T,A<:AbstractArray{T,4}} <: AbstractArray{T,4}
    data::A
    nx::Int
    ny::Int
    nz::Int
    nq::Int
end

Base.IndexStyle(::Type{<:_CPUHaloArray4}) = IndexCartesian()
Base.size(A::_CPUHaloArray4) = (A.nx + 2, A.ny + 2, A.nz + 2, A.nq)
Base.axes(A::_CPUHaloArray4) = (0:(A.nx + 1), 0:(A.ny + 1),
                                0:(A.nz + 1), 1:A.nq)
Base.getindex(A::_CPUHaloArray4, i::Int, j::Int, k::Int, q::Int) =
    A.data[i + 1, j + 1, k + 1, q]
function Base.setindex!(A::_CPUHaloArray4, v, i::Int, j::Int, k::Int, q::Int)
    A.data[i + 1, j + 1, k + 1, q] = v
    return v
end
KernelAbstractions.get_backend(::_CPUHaloArray4) = KernelAbstractions.CPU()

function _allocate_libb_f_3d(backend, ::Type{T}, Nx, Ny, Nz) where T
    if backend isa KernelAbstractions.CPU
        return _CPUHaloArray4(zeros(T, Nx + 2, Ny + 2, Nz + 2, 19),
                              Nx, Ny, Nz, 19)
    end
    return KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
end

function _copy_libb_f_in!(dest::_CPUHaloArray4, src)
    @inbounds for q in 1:19, k in 1:dest.nz, j in 1:dest.ny, i in 1:dest.nx
        dest[i, j, k, q] = src[i, j, k, q]
    end
    return dest
end
_copy_libb_f_in!(dest, src) = copyto!(dest, src)

function _libb_pressure_value_3d(bc, ::Type{T}) where T
    if haskey(bc.values, :rho)
        expr = bc.values[:rho]
        (is_spatial(expr) || is_time_dependent(expr)) && throw(ArgumentError(
            "spatial or time-dependent pressure BCs are not supported with 3D STL wall=libb"))
        return T(evaluate(expr; t=0.0))
    end
    return one(T)
end

function _libb_velocity_profile_host_3d(bc, setup, dx, Nx, Ny, Nz,
                                        ::Type{T}) where T
    if any(is_time_dependent, values(bc.values))
        throw(ArgumentError("time-dependent velocity BCs are not supported with 3D STL wall=libb"))
    end
    bc.face == :west || throw(ArgumentError(
        "3D STL wall=libb currently supports velocity BCs only on the west face"))

    dom = setup.domain
    dy = dom.Ly / Ny
    dz = dom.Lz / Nz
    profile = zeros(T, Ny, Nz)
    x_val = dx / 2
    for k in 1:Nz, j in 1:Ny
        y_val = (j - 0.5) * dy
        z_val = (k - 0.5) * dz
        kw = (; x=x_val, y=y_val, z=z_val,
              Lx=dom.Lx, Ly=dom.Ly, Lz=dom.Lz,
              Nx=Float64(Nx), Ny=Float64(Ny), Nz=Float64(Nz),
              dx=dx, dy=dy, dz=dz, t=0.0)
        profile[j, k] = haskey(bc.values, :ux) ?
            T(evaluate(bc.values[:ux]; kw...)) : zero(T)
    end
    return profile
end

function _west_libb_velocity_profile_host_3d(setup, dx, Nx, Ny, Nz,
                                             ::Type{T}) where T
    for bc in setup.boundaries
        if bc.face === :west && bc.type === :velocity
            return _libb_velocity_profile_host_3d(bc, setup, dx, Nx, Ny, Nz, T)
        end
    end
    throw(ArgumentError("3D STL wall=libb requires a west velocity inlet"))
end

function _build_libb_bc_rebuild_spec_3d(setup, dx, Nx, Ny, Nz,
                                        ::Type{T}, backend) where T
    west = HalfwayBB()
    east = HalfwayBB()
    south = HalfwayBB()
    north = HalfwayBB()
    bottom = HalfwayBB()
    top = HalfwayBB()

    for bc in setup.boundaries
        face_bc = if bc.type == :wall
            HalfwayBB()
        elseif bc.type == :velocity
            bc.face === :west || throw(ArgumentError(
                "3D STL wall=libb currently supports velocity BCs only on the west face"))
            profile = _libb_velocity_profile_host_3d(bc, setup, dx, Nx, Ny, Nz, T)
            dev = KernelAbstractions.allocate(backend, T, Ny, Nz)
            copyto!(dev, profile)
            ZouHeVelocity(dev)
        elseif bc.type == :pressure
            bc.face === :east || throw(ArgumentError(
                "3D STL wall=libb currently supports pressure BCs only on the east face"))
            ZouHePressure(_libb_pressure_value_3d(bc, T))
        elseif bc.type == :periodic
            throw(ArgumentError("periodic domain boundaries are not supported with 3D STL wall=libb"))
        else
            throw(ArgumentError("Boundary type ':$(bc.type)' is not supported with 3D STL wall=libb"))
        end

        if bc.face === :west
            west = face_bc
        elseif bc.face === :east
            east = face_bc
        elseif bc.face === :south
            south = face_bc
        elseif bc.face === :north
            north = face_bc
        elseif bc.face === :bottom
            bottom = face_bc
        elseif bc.face === :top
            top = face_bc
        else
            throw(ArgumentError("Boundary face ':$(bc.face)' is not valid for 3D STL wall=libb"))
        end
    end

    return BCSpec3D(; west=west, east=east, south=south, north=north,
                    bottom=bottom, top=top)
end

function run_obstacle_libb_3d(setup;
                              backend=KernelAbstractions.CPU(),
                              T::Type{<:AbstractFloat}=Float64)
    dom = setup.domain
    Nx, Ny, Nz = dom.Nx, dom.Ny, dom.Nz
    dx = dom.Lx / Nx
    ν = T(setup.physics.params[:nu])

    is_solid_h = zeros(Bool, Nx, Ny, Nz)
    _apply_geometry_3d!(is_solid_h, setup, Float64(dx))
    q_wall_h = _precompute_stl_libb_q_wall_3d(is_solid_h, setup, dx, T)
    u_profile_h = _west_libb_velocity_profile_host_3d(setup, dx, Nx, Ny, Nz, T)

    f_in_h = zeros(T, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx, q in 1:19
        f_in_h[i, j, k, q] = equilibrium(D3Q19(), one(T),
                                          u_profile_h[j, k],
                                          zero(T), zero(T), q)
    end

    q_wall = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny, Nz)
    uw_x = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    uw_y = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    uw_z = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    f_in = _allocate_libb_f_3d(backend, T, Nx, Ny, Nz)
    f_out = _allocate_libb_f_3d(backend, T, Nx, Ny, Nz)
    ρ = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    ux = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    uy = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    uz = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)

    copyto!(q_wall, q_wall_h)
    copyto!(is_solid, is_solid_h)
    fill!(uw_x, zero(T)); fill!(uw_y, zero(T)); fill!(uw_z, zero(T))
    _copy_libb_f_in!(f_in, f_in_h)
    fill!(ρ, one(T)); fill!(ux, zero(T))
    fill!(uy, zero(T)); fill!(uz, zero(T))

    bcspec = _build_libb_bc_rebuild_spec_3d(setup, dx, Nx, Ny, Nz, T, backend)

    Fx_sum = 0.0
    Fy_sum = 0.0
    Fz_sum = 0.0
    n_avg = 0

    for _ in 1:setup.max_steps
        fused_trt_libb_v2_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                    q_wall, uw_x, uw_y, uw_z,
                                    Nx, Ny, Nz, ν)
        apply_bc_rebuild_3d!(f_out, f_in, bcspec, ν, Nx, Ny, Nz)

        if !(f_out isa _CPUHaloArray4)
            drag = compute_drag_libb_3d(f_out, q_wall, Nx, Ny, Nz)
            Fx_sum += drag.Fx
            Fy_sum += drag.Fy
            Fz_sum += drag.Fz
            n_avg += 1
        end

        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    Fx_avg = n_avg == 0 ? 0.0 : Fx_sum / n_avg
    Fy_avg = n_avg == 0 ? 0.0 : Fy_sum / n_avg
    Fz_avg = n_avg == 0 ? 0.0 : Fz_sum / n_avg

    return (; ρ=Array(ρ), ux=Array(ux), uy=Array(uy), uz=Array(uz),
            Fx=Fx_avg, Fy=Fy_avg, Fz=Fz_avg,
            q_wall=Array(q_wall), is_solid=Array(is_solid),
            inlet_profile=u_profile_h, u_ref=Float64(maximum(abs, u_profile_h)))
end
