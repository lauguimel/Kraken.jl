function _edge_profile_host(block::Block, edge::Symbol, ::Type{T}, Ly, u_max) where T
    n = edge_length(block, edge)
    profile = zeros(T, n)
    for r in 1:n
        i, j = _edge_node(block, edge, r)
        profile[r] = T(_parabolic_channel_u(block.mesh.Y[i, j], Ly, u_max))
    end
    return profile
end

function _physical_normal_from_tag(tag::Symbol)
    tag === :inlet && return :west
    tag === :outlet && return :east
    tag in (:wall_bot, :wall_bottom, :bottom) && return :south
    tag in (:wall_top, :wall_upper, :top) && return :north
    return :auto
end

function _mesh_drag_bc(block::Block, edge::Symbol, tag::Symbol,
                       setup::SimulationSetup, backend, ::Type{T},
                       Ly, u_max, rho_out) where T
    tag === INTERFACE_TAG && return InterfaceBC()
    if tag === :inlet
        profile = _copy_to_backend(backend, T,
                                   _edge_profile_host(block, edge, T, Ly, u_max))
        return ZouHeVelocity(profile, _physical_normal_from_tag(tag))
    elseif tag === :outlet
        return ZouHePressure(T(rho_out), _physical_normal_from_tag(tag))
    end
    return HalfwayBB()
end

function _mesh_drag_bcspec(block::Block, setup::SimulationSetup,
                           backend, ::Type{T}, Ly, u_max, rho_out) where T
    tags = block.boundary_tags
    return BCSpec2D(;
        west=_mesh_drag_bc(block, :west, tags.west, setup, backend, T,
                           Ly, u_max, rho_out),
        east=_mesh_drag_bc(block, :east, tags.east, setup, backend, T,
                           Ly, u_max, rho_out),
        south=_mesh_drag_bc(block, :south, tags.south, setup, backend, T,
                            Ly, u_max, rho_out),
        north=_mesh_drag_bc(block, :north, tags.north, setup, backend, T,
                            Ly, u_max, rho_out))
end

function _mesh_drag_noop_bcspec(block::Block)
    function bc_for(tag::Symbol)
        (tag === INTERFACE_TAG || tag === :interface) && return InterfaceBC()
        return HalfwayBB()
    end
    tags = block.boundary_tags
    return BCSpec2D(;
        west=bc_for(tags.west),
        east=bc_for(tags.east),
        south=bc_for(tags.south),
        north=bc_for(tags.north))
end

_edge_code_2d(edge::Symbol) =
    edge === :west ? 1 :
    edge === :east ? 2 :
    edge === :south ? 3 :
    edge === :north ? 4 :
    error("unknown edge $edge")

_normal_code_2d(normal::Symbol) =
    normal === :west ? 1 :
    normal === :east ? 2 :
    normal === :south ? 3 :
    normal === :north ? 4 :
    error("unknown physical normal $normal")

@kernel function _mesh_drag_physnorm_velocity_edge_2d!(f, edge_code::Int,
                                                       normal_code::Int,
                                                       profile, Nx::Int,
                                                       Ny::Int)
    r = @index(Global)
    T = eltype(f)
    i = edge_code == 1 ? 1  :
        edge_code == 2 ? Nx :
        r
    j = edge_code == 3 ? 1  :
        edge_code == 4 ? Ny :
        r
    @inbounds begin
        f1 = f[i, j, 1]; f2 = f[i, j, 2]; f3 = f[i, j, 3]
        f4 = f[i, j, 4]; f5 = f[i, j, 5]; f6 = f[i, j, 6]
        f7 = f[i, j, 7]; f8 = f[i, j, 8]; f9 = f[i, j, 9]
        u = profile[r]
        if normal_code == 1
            rho = (f1 + f3 + f5 + T(2) * (f4 + f7 + f8)) / (one(T) - u)
            f2 = f4 + T(2 / 3) * rho * u
            f6 = f8 - T(0.5) * (f3 - f5) + T(1 / 6) * rho * u
            f9 = f7 + T(0.5) * (f3 - f5) + T(1 / 6) * rho * u
        elseif normal_code == 2
            rho = (f1 + f3 + f5 + T(2) * (f2 + f6 + f9)) / (one(T) + u)
            f4 = f2 - T(2 / 3) * rho * u
            f7 = f9 - T(0.5) * (f3 - f5) - T(1 / 6) * rho * u
            f8 = f6 + T(0.5) * (f3 - f5) - T(1 / 6) * rho * u
        elseif normal_code == 3
            rho = (f1 + f2 + f4 + T(2) * (f5 + f8 + f9)) / (one(T) - u)
            f3 = f5 + T(2 / 3) * rho * u
            f6 = f8 + T(0.5) * (f4 - f2) + T(1 / 6) * rho * u
            f7 = f9 + T(0.5) * (f2 - f4) + T(1 / 6) * rho * u
        else
            rho = (f1 + f2 + f4 + T(2) * (f3 + f6 + f7)) / (one(T) + u)
            f5 = f3 - T(2 / 3) * rho * u
            f8 = f6 + T(0.5) * (f2 - f4) - T(1 / 6) * rho * u
            f9 = f7 + T(0.5) * (f4 - f2) - T(1 / 6) * rho * u
        end
        f[i, j, 1] = f1; f[i, j, 2] = f2; f[i, j, 3] = f3
        f[i, j, 4] = f4; f[i, j, 5] = f5; f[i, j, 6] = f6
        f[i, j, 7] = f7; f[i, j, 8] = f8; f[i, j, 9] = f9
    end
end

@kernel function _mesh_drag_physnorm_pressure_edge_2d!(f, edge_code::Int,
                                                       normal_code::Int,
                                                       rho_out, Nx::Int,
                                                       Ny::Int)
    r = @index(Global)
    T = eltype(f)
    i = edge_code == 1 ? 1  :
        edge_code == 2 ? Nx :
        r
    j = edge_code == 3 ? 1  :
        edge_code == 4 ? Ny :
        r
    rho = T(rho_out)
    @inbounds begin
        f1 = f[i, j, 1]; f2 = f[i, j, 2]; f3 = f[i, j, 3]
        f4 = f[i, j, 4]; f5 = f[i, j, 5]; f6 = f[i, j, 6]
        f7 = f[i, j, 7]; f8 = f[i, j, 8]; f9 = f[i, j, 9]
        if normal_code == 1
            u = one(T) - (f1 + f3 + f5 + T(2) * (f4 + f7 + f8)) / rho
            f2 = f4 + T(2 / 3) * rho * u
            f6 = f8 - T(0.5) * (f3 - f5) + T(1 / 6) * rho * u
            f9 = f7 + T(0.5) * (f3 - f5) + T(1 / 6) * rho * u
        elseif normal_code == 2
            u = -one(T) + (f1 + f3 + f5 + T(2) * (f2 + f6 + f9)) / rho
            f4 = f2 - T(2 / 3) * rho * u
            f7 = f9 - T(0.5) * (f3 - f5) - T(1 / 6) * rho * u
            f8 = f6 + T(0.5) * (f3 - f5) - T(1 / 6) * rho * u
        elseif normal_code == 3
            u = one(T) - (f1 + f2 + f4 + T(2) * (f5 + f8 + f9)) / rho
            f3 = f5 + T(2 / 3) * rho * u
            f6 = f8 + T(0.5) * (f4 - f2) + T(1 / 6) * rho * u
            f7 = f9 + T(0.5) * (f2 - f4) + T(1 / 6) * rho * u
        else
            u = -one(T) + (f1 + f2 + f4 + T(2) * (f3 + f6 + f7)) / rho
            f5 = f3 - T(2 / 3) * rho * u
            f8 = f6 + T(0.5) * (f2 - f4) - T(1 / 6) * rho * u
            f9 = f7 + T(0.5) * (f4 - f2) - T(1 / 6) * rho * u
        end
        f[i, j, 1] = f1; f[i, j, 2] = f2; f[i, j, 3] = f3
        f[i, j, 4] = f4; f[i, j, 5] = f5; f[i, j, 6] = f6
        f[i, j, 7] = f7; f[i, j, 8] = f8; f[i, j, 9] = f9
    end
end

function _apply_mesh_drag_physical_normal_edge_2d!(f, edge::Symbol,
                                                   bc::ZouHeVelocity,
                                                   Nx::Int, Ny::Int)
    bc.physical_dir === :auto && return nothing
    backend = KernelAbstractions.get_backend(f)
    nrun = edge in (:west, :east) ? Ny : Nx
    _mesh_drag_physnorm_velocity_edge_2d!(backend)(
        f, _edge_code_2d(edge), _normal_code_2d(bc.physical_dir),
        bc.profile, Nx, Ny; ndrange=(nrun,))
    return nothing
end

function _apply_mesh_drag_physical_normal_edge_2d!(f, edge::Symbol,
                                                   bc::ZouHePressure,
                                                   Nx::Int, Ny::Int)
    bc.physical_dir === :auto && return nothing
    backend = KernelAbstractions.get_backend(f)
    nrun = edge in (:west, :east) ? Ny : Nx
    _mesh_drag_physnorm_pressure_edge_2d!(backend)(
        f, _edge_code_2d(edge), _normal_code_2d(bc.physical_dir),
        eltype(f)(bc.ρ_out), Nx, Ny; ndrange=(nrun,))
    return nothing
end

function _apply_mesh_drag_physical_normal_edge_2d!(f, edge::Symbol,
                                                   bc::AbstractBC,
                                                   Nx::Int, Ny::Int)
    return nothing
end

function _apply_mesh_drag_physical_normal_bcs_2d!(f, bcspec::BCSpec2D,
                                                  Nx::Int, Ny::Int)
    _apply_mesh_drag_physical_normal_edge_2d!(f, :west, bcspec.west, Nx, Ny)
    _apply_mesh_drag_physical_normal_edge_2d!(f, :east, bcspec.east, Nx, Ny)
    _apply_mesh_drag_physical_normal_edge_2d!(f, :south, bcspec.south, Nx, Ny)
    _apply_mesh_drag_physical_normal_edge_2d!(f, :north, bcspec.north, Nx, Ny)
    return nothing
end

@kernel function _mesh_drag_physnorm_halfway_edge_2d!(f_out, f_in,
                                                      edge_code::Int,
                                                      normal_code::Int,
                                                      Nx::Int, Ny::Int)
    r = @index(Global)
    i = edge_code == 1 ? 1  :
        edge_code == 2 ? Nx :
        r
    j = edge_code == 3 ? 1  :
        edge_code == 4 ? Ny :
        r
    @inbounds begin
        if normal_code == 1
            f_out[i, j, 2] = f_in[i, j, 4]
            f_out[i, j, 6] = f_in[i, j, 8]
            f_out[i, j, 9] = f_in[i, j, 7]
        elseif normal_code == 2
            f_out[i, j, 4] = f_in[i, j, 2]
            f_out[i, j, 7] = f_in[i, j, 9]
            f_out[i, j, 8] = f_in[i, j, 6]
        elseif normal_code == 3
            f_out[i, j, 3] = f_in[i, j, 5]
            f_out[i, j, 6] = f_in[i, j, 8]
            f_out[i, j, 7] = f_in[i, j, 9]
        else
            f_out[i, j, 5] = f_in[i, j, 3]
            f_out[i, j, 8] = f_in[i, j, 6]
            f_out[i, j, 9] = f_in[i, j, 7]
        end
    end
end

function _mesh_drag_is_channel_wall_tag(tag::Symbol)
    return tag in (:wall, :wall_top, :wall_upper, :top,
                   :wall_bot, :wall_bottom, :bottom)
end

function _apply_mesh_drag_physical_wall_bcs_2d!(f_out, f_in, tags,
                                                Nx::Int, Ny::Int)
    backend = KernelAbstractions.get_backend(f_out)
    for edge in EDGE_SYMBOLS_2D
        tag = getproperty(tags, edge)
        _mesh_drag_is_channel_wall_tag(tag) || continue
        normal = _physical_normal_from_tag(tag)
        normal === :auto && continue
        nrun = edge in (:west, :east) ? Ny : Nx
        _mesh_drag_physnorm_halfway_edge_2d!(backend)(
            f_out, f_in, _edge_code_2d(edge), _normal_code_2d(normal),
            Nx, Ny; ndrange=(nrun,))
    end
    return nothing
end

const _MESH_DRAG_CX_2D = (0, 1, 0, -1, 0, 1, -1, -1, 1)
const _MESH_DRAG_CY_2D = (0, 0, 1, 0, -1, 1, 1, -1, -1)

@inline function _mesh_drag_qopp_2d(q::Int)
    q == 1 && return 1
    q == 2 && return 4
    q == 3 && return 5
    q == 4 && return 2
    q == 5 && return 3
    q == 6 && return 8
    q == 7 && return 9
    q == 8 && return 6
    return 7
end

@kernel function _mesh_drag_physnorm_wall_ghost_col_2d!(f,
                                                        i_ghost::Int,
                                                        i_bd::Int,
                                                        pop1::Int,
                                                        pop2::Int,
                                                        pop3::Int,
                                                        shift1::Int,
                                                        shift2::Int,
                                                        shift3::Int,
                                                        j_lo::Int,
                                                        j_hi::Int)
    idx, p = @index(Global, NTuple)
    q, shift = p == 1 ? (pop1, shift1) :
               p == 2 ? (pop2, shift2) :
                         (pop3, shift3)
    jsrc = idx + shift
    jsrc = jsrc < j_lo ? j_lo : (jsrc > j_hi ? j_hi : jsrc)
    @inbounds f[i_ghost, idx, q] = f[i_bd, jsrc, _mesh_drag_qopp_2d(q)]
end

@kernel function _mesh_drag_physnorm_wall_ghost_row_2d!(f,
                                                        j_ghost::Int,
                                                        j_bd::Int,
                                                        pop1::Int,
                                                        pop2::Int,
                                                        pop3::Int,
                                                        shift1::Int,
                                                        shift2::Int,
                                                        shift3::Int,
                                                        i_lo::Int,
                                                        i_hi::Int)
    idx, p = @index(Global, NTuple)
    q, shift = p == 1 ? (pop1, shift1) :
               p == 2 ? (pop2, shift2) :
                         (pop3, shift3)
    isrc = idx + shift
    isrc = isrc < i_lo ? i_lo : (isrc > i_hi ? i_hi : isrc)
    @inbounds f[idx, j_ghost, q] = f[isrc, j_bd, _mesh_drag_qopp_2d(q)]
end

function _mesh_drag_physical_wall_pops(normal::Symbol)
    normal === :west  && return (2, 6, 9)
    normal === :east  && return (4, 7, 8)
    normal === :south && return (3, 6, 7)
    normal === :north && return (5, 8, 9)
    error("unknown physical normal $normal")
end

function _edge_tangent_sign_for_normal(block::Block, edge::Symbol,
                                       normal::Symbol)
    nedge = edge_length(block, edge)
    i0, j0 = _edge_node(block, edge, 1)
    i1, j1 = _edge_node(block, edge, nedge)
    dx = Float64(block.mesh.X[i1, j1]) - Float64(block.mesh.X[i0, j0])
    dy = Float64(block.mesh.Y[i1, j1]) - Float64(block.mesh.Y[i0, j0])
    component = normal in (:west, :east) ? dy : dx
    return component < 0 ? -1 : 1
end

function _mesh_drag_wall_ghost_shifts(block::Block, edge::Symbol,
                                      normal::Symbol, pops)
    sgn = _edge_tangent_sign_for_normal(block, edge, normal)
    if normal in (:west, :east)
        return ntuple(k -> sgn * _MESH_DRAG_CY_2D[pops[k]], 3)
    end
    return ntuple(k -> sgn * _MESH_DRAG_CX_2D[pops[k]], 3)
end

function _apply_mesh_drag_physical_wall_ghost_edge_2d!(block::Block,
                                                       st::BlockState2D,
                                                       edge::Symbol,
                                                       normal::Symbol)
    ng = st.n_ghost
    Nxp = st.Nξ_phys
    Nyp = st.Nη_phys
    Nxe = Nxp + 2 * ng
    Nye = Nyp + 2 * ng
    pops = _mesh_drag_physical_wall_pops(normal)
    shifts = _mesh_drag_wall_ghost_shifts(block, edge, normal, pops)
    backend = KernelAbstractions.get_backend(st.f)
    if edge === :west || edge === :east
        i_bd = edge === :west ? ng + 1 : ng + Nxp
        kernel = _mesh_drag_physnorm_wall_ghost_col_2d!(backend)
        for g in 1:ng
            i_ghost = edge === :west ? ng + 1 - g : ng + Nxp + g
            kernel(st.f, i_ghost, i_bd, pops[1], pops[2], pops[3],
                   shifts[1], shifts[2], shifts[3],
                   ng + 1, ng + Nyp; ndrange=(Nye, 3))
        end
    else
        j_bd = edge === :south ? ng + 1 : ng + Nyp
        kernel = _mesh_drag_physnorm_wall_ghost_row_2d!(backend)
        for g in 1:ng
            j_ghost = edge === :south ? ng + 1 - g : ng + Nyp + g
            kernel(st.f, j_ghost, j_bd, pops[1], pops[2], pops[3],
                   shifts[1], shifts[2], shifts[3],
                   ng + 1, ng + Nxp; ndrange=(Nxe, 3))
        end
    end
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _apply_mesh_drag_physical_wall_ghost_bcs_2d!(mbm::MultiBlockMesh2D,
                                                      states)
    for (block, st) in zip(mbm.blocks, states)
        for edge in EDGE_SYMBOLS_2D
            tag = getproperty(block.boundary_tags, edge)
            _mesh_drag_is_channel_wall_tag(tag) || continue
            normal = _physical_normal_from_tag(tag)
            normal === :auto && continue
            _apply_mesh_drag_physical_wall_ghost_edge_2d!(block, st,
                                                          edge, normal)
        end
    end
    return nothing
end

@kernel function _mesh_drag_cylinder_radial_ghost_col_2d!(f, mask,
                                                          i_ghost::Int,
                                                          i_bd::Int,
                                                          ng::Int,
                                                          alpha)
    r, q = @index(Global, NTuple)
    if @inbounds mask[r, q]
        j = ng + r
        @inbounds f[i_ghost, j, q] =
            (1 - alpha) * f[i_ghost, j, q] +
            alpha * f[i_bd, j, _mesh_drag_qopp_2d(q)]
    end
end

@kernel function _mesh_drag_cylinder_radial_ghost_row_2d!(f, mask,
                                                          j_ghost::Int,
                                                          j_bd::Int,
                                                          ng::Int,
                                                          alpha)
    r, q = @index(Global, NTuple)
    if @inbounds mask[r, q]
        i = ng + r
        @inbounds f[i, j_ghost, q] =
            (1 - alpha) * f[i, j_ghost, q] +
            alpha * f[i, j_bd, _mesh_drag_qopp_2d(q)]
    end
end

function _mesh_drag_cylinder_crossing_mask(block::Block, edge::Symbol,
                                           cx, cy)
    nrun = edge_length(block, edge)
    mask = fill(false, nrun, 9)
    epsn = 100 * eps(Float64)
    for r in 1:nrun
        i, j = _edge_node(block, edge, r)
        nx = Float64(block.mesh.X[i, j]) - Float64(cx)
        ny = Float64(block.mesh.Y[i, j]) - Float64(cy)
        nrm = hypot(nx, ny)
        nrm <= epsn && continue
        nx /= nrm
        ny /= nrm
        for q in 1:9
            dot = _MESH_DRAG_CX_2D[q] * nx + _MESH_DRAG_CY_2D[q] * ny
            mask[r, q] = dot > epsn
        end
    end
    return mask
end

function _mesh_drag_cylinder_ghost_masks(block::Block, backend, cx, cy)
    function mask_for(edge)
        getproperty(block.boundary_tags, edge) === :cylinder || return nothing
        return _copy_bool_to_backend(backend,
            _mesh_drag_cylinder_crossing_mask(block, edge, cx, cy))
    end
    return (;
        west=mask_for(:west),
        east=mask_for(:east),
        south=mask_for(:south),
        north=mask_for(:north))
end

function _apply_mesh_drag_cylinder_radial_ghost_bcs_2d!(mbm::MultiBlockMesh2D,
                                                        states,
                                                        masks,
                                                        alpha)
    for (block, st, block_masks) in zip(mbm.blocks, states, masks)
        ng = st.n_ghost
        Nxp = st.Nξ_phys
        Nyp = st.Nη_phys
        backend = KernelAbstractions.get_backend(st.f)
        for edge in EDGE_SYMBOLS_2D
            mask = getproperty(block_masks, edge)
            mask === nothing && continue
            nrun = edge_length(block, edge)
            if edge === :west || edge === :east
                i_bd = edge === :west ? ng + 1 : ng + Nxp
                kernel = _mesh_drag_cylinder_radial_ghost_col_2d!(backend)
                for g in 1:ng
                    i_ghost = edge === :west ? ng + 1 - g : ng + Nxp + g
                    kernel(st.f, mask, i_ghost, i_bd, ng,
                           eltype(st.f)(alpha); ndrange=(nrun, 9))
                end
            else
                j_bd = edge === :south ? ng + 1 : ng + Nyp
                kernel = _mesh_drag_cylinder_radial_ghost_row_2d!(backend)
                for g in 1:ng
                    j_ghost = edge === :south ? ng + 1 - g : ng + Nyp + g
                    kernel(st.f, mask, j_ghost, j_bd, ng,
                           eltype(st.f)(alpha); ndrange=(nrun, 9))
                end
            end
        end
        KernelAbstractions.synchronize(backend)
    end
    return nothing
end

function _circle_solid_field(block::Block, cx, cy, radius)
    solid = zeros(Bool, block.mesh.Nξ, block.mesh.Nη)
    r2 = Float64(radius)^2
    tol = max(1e-14, 1e-10 * max(1.0, r2))
    for j in 1:block.mesh.Nη, i in 1:block.mesh.Nξ
        dx = Float64(block.mesh.X[i, j]) - Float64(cx)
        dy = Float64(block.mesh.Y[i, j]) - Float64(cy)
        solid[i, j] = dx * dx + dy * dy <= r2 + tol
    end
    return solid
end

function _mesh_curved_edges(block::Block)
    edges = Symbol[]
    for edge in EDGE_SYMBOLS_2D
        getproperty(block.boundary_tags, edge) === :cylinder && push!(edges, edge)
    end
    return Tuple(edges)
end

function _edge_inner_node(block::Block, edge::Symbol, r::Int)
    edge === :west  && return min(2, block.mesh.Nξ), r
    edge === :east  && return max(block.mesh.Nξ - 1, 1), r
    edge === :south && return r, min(2, block.mesh.Nη)
    edge === :north && return r, max(block.mesh.Nη - 1, 1)
    error("unknown edge $edge")
end

function _compute_bodyfit_cylinder_force_2d(mbm::MultiBlockMesh2D, states,
                                            cx, cy, radius, nu, dx_ref, ng::Int)
    Fx_pressure = 0.0
    Fy_pressure = 0.0
    Fx_viscous = 0.0
    Fy_viscous = 0.0
    inv_cs2_den = 1.0 / 3.0
    epsd = eps(Float64)

    for (block, st) in zip(mbm.blocks, states)
        rho_h = Array(st.ρ)
        ux_h = Array(st.ux)
        uy_h = Array(st.uy)
        for edge in EDGE_SYMBOLS_2D
            getproperty(block.boundary_tags, edge) === :cylinder || continue
            nedge = edge_length(block, edge)
            nedge < 2 && continue
            for r in 1:(nedge - 1)
                ib0, jb0 = _edge_node(block, edge, r)
                ib1, jb1 = _edge_node(block, edge, r + 1)
                ii0, ji0 = _edge_inner_node(block, edge, r)
                ii1, ji1 = _edge_inner_node(block, edge, r + 1)

                x0 = Float64(block.mesh.X[ib0, jb0])
                y0 = Float64(block.mesh.Y[ib0, jb0])
                x1 = Float64(block.mesh.X[ib1, jb1])
                y1 = Float64(block.mesh.Y[ib1, jb1])
                xm = 0.5 * (x0 + x1)
                ym = 0.5 * (y0 + y1)
                ds = hypot(x1 - x0, y1 - y0)

                nx = xm - Float64(cx)
                ny = ym - Float64(cy)
                nrm = max(hypot(nx, ny), epsd)
                nx /= nrm
                ny /= nrm
                tx = -ny
                ty = nx

                rb0 = rho_h[ib0 + ng, jb0 + ng]
                rb1 = rho_h[ib1 + ng, jb1 + ng]
                r0 = rho_h[ii0 + ng, ji0 + ng]
                r1 = rho_h[ii1 + ng, ji1 + ng]
                ux0 = ux_h[ii0 + ng, ji0 + ng]
                ux1 = ux_h[ii1 + ng, ji1 + ng]
                uy0 = uy_h[ii0 + ng, ji0 + ng]
                uy1 = uy_h[ii1 + ng, ji1 + ng]
                rho_wall = 0.5 * (Float64(rb0) + Float64(rb1))
                rho = 0.5 * (Float64(r0) + Float64(r1))
                ux = 0.5 * (Float64(ux0) + Float64(ux1))
                uy = 0.5 * (Float64(uy0) + Float64(uy1))

                xi0 = Float64(block.mesh.X[ii0, ji0])
                yi0 = Float64(block.mesh.Y[ii0, ji0])
                xi1 = Float64(block.mesh.X[ii1, ji1])
                yi1 = Float64(block.mesh.Y[ii1, ji1])
                dist0 = abs((xi0 - x0) * nx + (yi0 - y0) * ny)
                dist1 = abs((xi1 - x1) * nx + (yi1 - y1) * ny)
                wall_dist = max(0.5 * (dist0 + dist1), epsd)

                # The constant pressure part cancels on a closed boundary;
                # subtracting rho=1 reduces quadrature error on coarse O-grids.
                p = (rho_wall - 1.0) * inv_cs2_den
                ut = ux * tx + uy * ty
                tau = rho * Float64(nu) * ut / wall_dist
                # Pressure is a lattice stress integrated over a boundary
                # length in lattice units. The viscous term below already
                # cancels dx_ref through du/dn_lattice * ds_lattice.
                ds_lattice = ds / Float64(dx_ref)
                Fx_pressure += (-p * nx) * ds_lattice
                Fy_pressure += (-p * ny) * ds_lattice
                Fx_viscous += (tau * tx) * ds
                Fy_viscous += (tau * ty) * ds
            end
        end
    end
    return (;
        Fx=Fx_pressure + Fx_viscous,
        Fy=Fy_pressure + Fy_viscous,
        Fx_pressure, Fy_pressure, Fx_viscous, Fy_viscous)
end

function _check_block_density(states, step::Int, label::AbstractString)
    rho_min = Inf
    rho_max = -Inf
    for st in states
        rho_h = Array(st.ρ)
        ng = st.n_ghost
        phys = @view rho_h[(ng + 1):(ng + st.Nξ_phys),
                           (ng + 1):(ng + st.Nη_phys)]
        any(!isfinite, phys) && error("non-finite density in $label at step $step")
        rho_min = min(rho_min, minimum(phys))
        rho_max = max(rho_max, maximum(phys))
    end
    return Float64(rho_min), Float64(rho_max)
end
