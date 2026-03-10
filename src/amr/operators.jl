# ---------------------------------------------------------------------------
# Finite Volume operators on QuadTree AMR
# ---------------------------------------------------------------------------

"""
    neighbor_value(tree, name, level, i, j, dir) -> T

Get the field value of the neighbor in direction `dir`, handling fine-coarse
interfaces and boundaries.

- Same level: direct value
- Coarser neighbor: use coarse cell value (first-order)
- Finer neighbors: average the fine cells on the shared face
- Boundary (nothing): Neumann zero-gradient (return own value)
"""
function neighbor_value(tree::QuadTree{T}, name::Symbol, level::Int, i::Int, j::Int, dir::Symbol) where {T}
    nb = find_neighbor(tree, level, i, j, dir)
    if nb === nothing
        # Boundary: Neumann (zero gradient)
        return get_field(tree, name, level, i, j)
    end
    if nb isa Tuple{Int,Int,Int}
        nb_level, ni, nj = nb
        return get_field(tree, name, nb_level, ni, nj)
    end
    # Vector of fine neighbors: average
    avg = zero(T)
    for (nb_level, ni, nj) in nb
        avg += get_field(tree, name, nb_level, ni, nj)
    end
    return avg / T(length(nb))
end

"""
    neighbor_distance(tree, level, i, j, dir) -> T

Distance between cell center and neighbor center. Accounts for level differences.
Returns own cell size `h` if at boundary (Neumann ghost cell).
"""
function neighbor_distance(tree::QuadTree{T}, level::Int, i::Int, j::Int, dir::Symbol) where {T}
    h = cell_size(tree, level)
    nb = find_neighbor(tree, level, i, j, dir)
    if nb === nothing
        return h  # ghost cell at distance h (but flux=0 for Neumann)
    end
    if nb isa Tuple{Int,Int,Int}
        nb_level, _, _ = nb
        nb_h = cell_size(tree, nb_level)
        return (h + nb_h) / 2
    end
    # Fine neighbors: distance to fine cell center
    # Fine cell has h_fine = h/2, distance = h/2 + h_fine/2 = h/2 + h/4 = 3h/4
    # But we average the fine cells, so effectively the face is at h/2 from us
    # and the fine center is at h/4 from the face → total 3h/4
    return T(3) * h / T(4)
end

# ---------------------------------------------------------------------------
# Laplacian (flux-based FV)
# ---------------------------------------------------------------------------

"""
    laplacian_amr!(tree, out, phi)

Compute the Laplacian ∇²φ using face fluxes on the quadtree.

For each leaf cell:
    ∇²φ ≈ (1/V) Σ_faces (φ_nb - φ_cell) / d * A

where V = h², A = h (face area), d = distance between centers.

At fine-coarse interfaces, distances are adjusted for the level difference.
At boundaries, Neumann (zero gradient) is applied → zero flux.
"""
function laplacian_amr!(tree::QuadTree{T}, out::Symbol, phi::Symbol) where {T}
    foreach_leaf(tree) do level, i, j
        h = cell_size(tree, level)
        V = h * h
        phi_c = get_field(tree, phi, level, i, j)
        flux_sum = zero(T)

        for dir in (:left, :right, :bottom, :top)
            nb = find_neighbor(tree, level, i, j, dir)

            if nb === nothing
                # Boundary: Neumann zero-gradient → flux = 0
                continue
            end

            if nb isa Tuple{Int,Int,Int}
                nb_level, ni, nj = nb
                phi_nb = get_field(tree, phi, nb_level, ni, nj)
                nb_h = cell_size(tree, nb_level)
                d = (h + nb_h) / 2
                A = h  # face area for this cell
                flux_sum += (phi_nb - phi_c) / d * A
            else
                # Vector of fine neighbors on the shared face
                # Each fine cell covers a sub-face of area h_fine
                for (nb_level, ni, nj) in nb
                    phi_nb = get_field(tree, phi, nb_level, ni, nj)
                    h_fine = cell_size(tree, nb_level)
                    d = h / 2 + h_fine / 2  # distance: half my cell + half fine cell
                    A = h_fine  # sub-face area
                    flux_sum += (phi_nb - phi_c) / d * A
                end
            end
        end

        set_field!(tree, out, level, i, j, flux_sum / V)
    end
    nothing
end

# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------

"""
    divergence_amr!(tree, div_field, u, v)

Compute ∇·(u,v) using face-averaged velocities on the quadtree.

    ∇·F ≈ (1/V) Σ_faces (F·n) * A

where the face velocity is interpolated from cell and neighbor values.
"""
function divergence_amr!(tree::QuadTree{T}, div_field::Symbol, u::Symbol, v::Symbol) where {T}
    foreach_leaf(tree) do level, i, j
        h = cell_size(tree, level)
        V = h * h
        u_c = get_field(tree, u, level, i, j)
        v_c = get_field(tree, v, level, i, j)
        div_sum = zero(T)

        # Right face: +u
        u_right = (u_c + neighbor_value(tree, u, level, i, j, :right)) / 2
        div_sum += u_right * h
        # Left face: -u
        u_left = (u_c + neighbor_value(tree, u, level, i, j, :left)) / 2
        div_sum -= u_left * h
        # Top face: +v
        v_top = (v_c + neighbor_value(tree, v, level, i, j, :top)) / 2
        div_sum += v_top * h
        # Bottom face: -v
        v_bottom = (v_c + neighbor_value(tree, v, level, i, j, :bottom)) / 2
        div_sum -= v_bottom * h

        set_field!(tree, div_field, level, i, j, div_sum / V)
    end
    nothing
end

# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------

"""
    gradient_amr!(tree, gx, gy, phi)

Compute the gradient ∇φ = (∂φ/∂x, ∂φ/∂y) at cell centers.

Uses central differences with distance correction at fine-coarse interfaces.
"""
function gradient_amr!(tree::QuadTree{T}, gx::Symbol, gy::Symbol, phi::Symbol) where {T}
    foreach_leaf(tree) do level, i, j
        phi_right = neighbor_value(tree, phi, level, i, j, :right)
        phi_left  = neighbor_value(tree, phi, level, i, j, :left)
        phi_top   = neighbor_value(tree, phi, level, i, j, :top)
        phi_bot   = neighbor_value(tree, phi, level, i, j, :bottom)

        d_right = neighbor_distance(tree, level, i, j, :right)
        d_left  = neighbor_distance(tree, level, i, j, :left)
        d_top   = neighbor_distance(tree, level, i, j, :top)
        d_bot   = neighbor_distance(tree, level, i, j, :bottom)

        dphidx = (phi_right - phi_left) / (d_right + d_left)
        dphidy = (phi_top - phi_bot)    / (d_top + d_bot)

        set_field!(tree, gx, level, i, j, dphidx)
        set_field!(tree, gy, level, i, j, dphidy)
    end
    nothing
end

# ---------------------------------------------------------------------------
# Advection (upwind)
# ---------------------------------------------------------------------------

"""
    advect_amr!(tree, adv, u, v, phi)

Compute the advection term u·∇φ using first-order upwind on faces.

For each face, the velocity is interpolated (average of cell and neighbor),
then the upwind value of φ is selected based on the velocity direction.
"""
function advect_amr!(tree::QuadTree{T}, adv::Symbol, u::Symbol, v::Symbol, phi::Symbol) where {T}
    foreach_leaf(tree) do level, i, j
        h = cell_size(tree, level)
        V = h * h
        u_c = get_field(tree, u, level, i, j)
        v_c = get_field(tree, v, level, i, j)
        phi_c = get_field(tree, phi, level, i, j)
        adv_sum = zero(T)

        # Right face
        u_face = (u_c + neighbor_value(tree, u, level, i, j, :right)) / 2
        phi_face = u_face >= 0 ? phi_c : neighbor_value(tree, phi, level, i, j, :right)
        adv_sum += u_face * phi_face * h

        # Left face
        u_face = (u_c + neighbor_value(tree, u, level, i, j, :left)) / 2
        phi_face = u_face >= 0 ? neighbor_value(tree, phi, level, i, j, :left) : phi_c
        adv_sum -= u_face * phi_face * h

        # Top face
        v_face = (v_c + neighbor_value(tree, v, level, i, j, :top)) / 2
        phi_face = v_face >= 0 ? phi_c : neighbor_value(tree, phi, level, i, j, :top)
        adv_sum += v_face * phi_face * h

        # Bottom face
        v_face = (v_c + neighbor_value(tree, v, level, i, j, :bottom)) / 2
        phi_face = v_face >= 0 ? neighbor_value(tree, phi, level, i, j, :bottom) : phi_c
        adv_sum -= v_face * phi_face * h

        set_field!(tree, adv, level, i, j, adv_sum / V)
    end
    nothing
end
