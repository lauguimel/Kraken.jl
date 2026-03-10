"""
    QuadTree{T} -- Cell-centered quadtree for 2D AMR

Stores scalar fields on a quadtree mesh. The tree is represented as a set of
levels, each level being a 2D boolean mask (active/inactive) plus field data.

Domain: [0, L] x [0, L] with L=1.0 default.
Level 0 = 1x1 root cell, level k = 2^k x 2^k potential cells.
Cell size at level k: h_k = L / 2^k.

Key property: 2:1 balance -- adjacent cells differ by at most 1 level.
"""
mutable struct QuadTree{T<:AbstractFloat}
    max_level::Int
    L::T

    # active[k+1][i,j] = true if cell (i,j) at level k is a leaf
    active::Vector{BitMatrix}

    # fields[:u][k+1][i,j] for field :u at cell (i,j) of level k
    fields::Dict{Symbol,Vector{Matrix{T}}}

    # (level, i, j) for each active leaf cell, sorted by level then i then j
    leaf_list::Vector{Tuple{Int,Int,Int}}
end

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

"""
    QuadTree(T::Type, max_level::Int; L=one(T))

Create a quadtree with a single root cell at level 0.
"""
function QuadTree(::Type{T}, max_level::Int; L::T=one(T)) where {T<:AbstractFloat}
    @assert max_level >= 0 "max_level must be non-negative"
    active = Vector{BitMatrix}(undef, max_level + 1)
    for k in 0:max_level
        n = 2^k
        active[k+1] = falses(n, n)
    end
    active[1][1, 1] = true  # root cell is active leaf

    fields = Dict{Symbol,Vector{Matrix{T}}}()
    leaf_list = [(0, 1, 1)]
    QuadTree{T}(max_level, L, active, fields, leaf_list)
end

# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------

"""
    add_field!(tree, name; initial_value=zero(T))

Add a named scalar field to the tree, initialized at all active leaves.
"""
function add_field!(tree::QuadTree{T}, name::Symbol; initial_value::T=zero(T)) where {T}
    haskey(tree.fields, name) && error("Field :$name already exists")
    data = Vector{Matrix{T}}(undef, tree.max_level + 1)
    for k in 0:tree.max_level
        n = 2^k
        data[k+1] = fill(initial_value, n, n)
    end
    tree.fields[name] = data
    nothing
end

"""
    get_field(tree, name, level, i, j)

Return the value of field `name` at cell (level, i, j).
"""
function get_field(tree::QuadTree{T}, name::Symbol, level::Int, i::Int, j::Int) where {T}
    tree.fields[name][level+1][i, j]
end

"""
    set_field!(tree, name, level, i, j, val)

Set the value of field `name` at cell (level, i, j).
"""
function set_field!(tree::QuadTree{T}, name::Symbol, level::Int, i::Int, j::Int, val) where {T}
    tree.fields[name][level+1][i, j] = T(val)
    nothing
end

# ---------------------------------------------------------------------------
# Cell geometry
# ---------------------------------------------------------------------------

"""
    cell_size(tree, level)

Return the cell width at the given level: h = L / 2^level.
"""
cell_size(tree::QuadTree{T}, level::Int) where {T} = tree.L / T(2^level)

"""
    cell_center(tree, level, i, j)

Return (x, y) coordinates of the cell center.
"""
function cell_center(tree::QuadTree{T}, level::Int, i::Int, j::Int) where {T}
    h = cell_size(tree, level)
    x = (i - T(0.5)) * h
    y = (j - T(0.5)) * h
    (x, y)
end

# ---------------------------------------------------------------------------
# Leaf list
# ---------------------------------------------------------------------------

"""
    rebuild_leaf_list!(tree)

Rebuild the cached list of leaf cells, sorted by (level, i, j).
"""
function rebuild_leaf_list!(tree::QuadTree)
    empty!(tree.leaf_list)
    for k in 0:tree.max_level
        n = 2^k
        for j in 1:n, i in 1:n
            if tree.active[k+1][i, j]
                push!(tree.leaf_list, (k, i, j))
            end
        end
    end
    nothing
end

"""
    nleaves(tree)

Return the number of active leaf cells.
"""
nleaves(tree::QuadTree) = length(tree.leaf_list)

"""
    foreach_leaf(f, tree)

Call `f(level, i, j)` for each active leaf cell.
"""
function foreach_leaf(f::Function, tree::QuadTree)
    for (level, i, j) in tree.leaf_list
        f(level, i, j)
    end
end

# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------

"""
    refine!(tree, level, i, j)

Split leaf cell (level, i, j) into 4 children at level+1.
Field values are copied from parent to all four children (constant interpolation).
"""
function refine!(tree::QuadTree{T}, level::Int, i::Int, j::Int) where {T}
    @assert level < tree.max_level "Cannot refine beyond max_level=$(tree.max_level)"
    @assert tree.active[level+1][i, j] "Cell ($level, $i, $j) is not an active leaf"

    # Deactivate parent
    tree.active[level+1][i, j] = false

    # Activate 4 children
    ci, cj = 2i - 1, 2j - 1
    tree.active[level+2][ci,   cj]   = true
    tree.active[level+2][ci+1, cj]   = true
    tree.active[level+2][ci,   cj+1] = true
    tree.active[level+2][ci+1, cj+1] = true

    # Interpolate fields: bilinear from parent center to child centers
    for (name, data) in tree.fields
        parent_val = data[level+1][i, j]
        # For now, constant interpolation (parent value to all children).
        # Bilinear interpolation requires neighbor data and is done in adapt!
        data[level+2][ci,   cj]   = parent_val
        data[level+2][ci+1, cj]   = parent_val
        data[level+2][ci,   cj+1] = parent_val
        data[level+2][ci+1, cj+1] = parent_val
    end

    rebuild_leaf_list!(tree)
    nothing
end

"""
    coarsen!(tree, level, i, j)

Merge the 4 siblings of cell (level, i, j) back into their parent at level-1.
The parent receives the average of the 4 children's field values.
"""
function coarsen!(tree::QuadTree{T}, level::Int, i::Int, j::Int) where {T}
    @assert level > 0 "Cannot coarsen level 0"

    # Find parent and sibling origin
    pi = (i + 1) >> 1
    pj = (j + 1) >> 1
    ci = 2pi - 1
    cj = 2pj - 1

    # All 4 children must be active leaves
    children = [(ci, cj), (ci+1, cj), (ci, cj+1), (ci+1, cj+1)]
    for (si, sj) in children
        @assert tree.active[level+1][si, sj] "Child ($level, $si, $sj) is not an active leaf"
    end

    # Deactivate children
    for (si, sj) in children
        tree.active[level+1][si, sj] = false
    end

    # Activate parent
    tree.active[level][pi, pj] = true

    # Average fields
    for (name, data) in tree.fields
        avg = zero(T)
        for (si, sj) in children
            avg += data[level+1][si, sj]
        end
        data[level][pi, pj] = avg / T(4)
    end

    rebuild_leaf_list!(tree)
    nothing
end

# ---------------------------------------------------------------------------
# Neighbor finding
# ---------------------------------------------------------------------------

"""
    find_neighbor(tree, level, i, j, dir)

Find the neighbor of cell (level, i, j) in direction `dir` (:left, :right, :bottom, :top).

Returns:
- `(level, ni, nj)` if neighbor is at the same level
- `(level-1, ni, nj)` if neighbor is a coarser leaf (parent level)
- A vector of tuples if neighbor is refined (multiple fine cells on shared face)
- `nothing` if at domain boundary
"""
function find_neighbor(tree::QuadTree, level::Int, i::Int, j::Int, dir::Symbol)
    n = 2^level

    # Compute neighbor index at same level
    ni, nj = i, j
    if dir == :left
        ni = i - 1
    elseif dir == :right
        ni = i + 1
    elseif dir == :bottom
        nj = j - 1
    elseif dir == :top
        nj = j + 1
    else
        error("Invalid direction: $dir. Use :left, :right, :bottom, :top")
    end

    # Boundary check
    if ni < 1 || ni > n || nj < 1 || nj > n
        return nothing
    end

    # Same-level neighbor exists as leaf
    if tree.active[level+1][ni, nj]
        return (level, ni, nj)
    end

    # Check if neighbor is refined (children exist at level+1)
    if level < tree.max_level
        cni = 2ni - 1
        cnj = 2nj - 1
        # Check if any children of (level, ni, nj) are active
        has_children = false
        if level + 1 <= tree.max_level
            n2 = 2^(level + 1)
            if cni >= 1 && cni + 1 <= n2 && cnj >= 1 && cnj + 1 <= n2
                for di in 0:1, dj in 0:1
                    if tree.active[level+2][cni+di, cnj+dj]
                        has_children = true
                        break
                    end
                end
                # Also check if children themselves have children (deeper refinement)
                if !has_children
                    # Children might be further refined; check recursively
                    # For find_neighbor, we return the fine cells on the shared face
                    has_children = _has_any_descendant(tree, level + 1, cni, cnj) ||
                                   _has_any_descendant(tree, level + 1, cni + 1, cnj) ||
                                   _has_any_descendant(tree, level + 1, cni, cnj + 1) ||
                                   _has_any_descendant(tree, level + 1, cni + 1, cnj + 1)
                end
            end
        end

        if has_children
            # Return the 2 fine cells on the shared face
            if dir == :right
                # Right face of (level, i, j) → left column of neighbor's children
                fine_cells = _find_leaf_cells_on_face(tree, level + 1, cni, cnj, :left)
            elseif dir == :left
                # Left face → right column of neighbor's children
                fine_cells = _find_leaf_cells_on_face(tree, level + 1, cni, cnj, :right)
            elseif dir == :top
                # Top face → bottom row of neighbor's children
                fine_cells = _find_leaf_cells_on_face(tree, level + 1, cni, cnj, :bottom)
            elseif dir == :bottom
                # Bottom face → top row of neighbor's children
                fine_cells = _find_leaf_cells_on_face(tree, level + 1, cni, cnj, :top)
            end
            return fine_cells
        end
    end

    # Check coarser level: parent's neighbor
    if level > 0
        # The coarser cell that contains (level, ni, nj)
        pni = (ni + 1) >> 1
        pnj = (nj + 1) >> 1
        if tree.active[level][pni, pnj]
            return (level - 1, pni, pnj)
        end
    end

    return nothing
end

"""Check if any descendant of the cell region starting at (level, ci, cj) for a 2x2 block is active."""
function _has_any_descendant(tree::QuadTree, level::Int, i::Int, j::Int)
    if level > tree.max_level
        return false
    end
    n = 2^level
    if i < 1 || i > n || j < 1 || j > n
        return false
    end
    if tree.active[level+1][i, j]
        return true
    end
    if level < tree.max_level
        ci = 2i - 1
        cj = 2j - 1
        return _has_any_descendant(tree, level + 1, ci, cj) ||
               _has_any_descendant(tree, level + 1, ci + 1, cj) ||
               _has_any_descendant(tree, level + 1, ci, cj + 1) ||
               _has_any_descendant(tree, level + 1, ci + 1, cj + 1)
    end
    return false
end

"""Find leaf cells on a particular face of a 2x2 block of children at the given level."""
function _find_leaf_cells_on_face(tree::QuadTree, level::Int, ci::Int, cj::Int, face::Symbol)
    # ci, cj is the bottom-left child origin
    # The 4 children are: (ci,cj), (ci+1,cj), (ci,cj+1), (ci+1,cj+1)
    # Return leaves on the specified face of this 2x2 block
    result = Tuple{Int,Int,Int}[]

    if face == :left  # left column: ci
        for dj in 0:1
            _collect_leaves_on_face!(result, tree, level, ci, cj + dj, :left)
        end
    elseif face == :right  # right column: ci+1
        for dj in 0:1
            _collect_leaves_on_face!(result, tree, level, ci + 1, cj + dj, :right)
        end
    elseif face == :bottom  # bottom row: cj
        for di in 0:1
            _collect_leaves_on_face!(result, tree, level, ci + di, cj, :bottom)
        end
    elseif face == :top  # top row: cj+1
        for di in 0:1
            _collect_leaves_on_face!(result, tree, level, ci + di, cj + 1, :top)
        end
    end

    return result
end

"""Recursively collect leaf cells on a face of cell (level, i, j)."""
function _collect_leaves_on_face!(result, tree::QuadTree, level::Int, i::Int, j::Int, face::Symbol)
    n = 2^level
    if i < 1 || i > n || j < 1 || j > n || level > tree.max_level
        return
    end
    if tree.active[level+1][i, j]
        push!(result, (level, i, j))
        return
    end
    # Cell is refined, recurse into children on the same face
    if level < tree.max_level
        ci = 2i - 1
        cj = 2j - 1
        if face == :left
            _collect_leaves_on_face!(result, tree, level + 1, ci, cj, :left)
            _collect_leaves_on_face!(result, tree, level + 1, ci, cj + 1, :left)
        elseif face == :right
            _collect_leaves_on_face!(result, tree, level + 1, ci + 1, cj, :right)
            _collect_leaves_on_face!(result, tree, level + 1, ci + 1, cj + 1, :right)
        elseif face == :bottom
            _collect_leaves_on_face!(result, tree, level + 1, ci, cj, :bottom)
            _collect_leaves_on_face!(result, tree, level + 1, ci + 1, cj, :bottom)
        elseif face == :top
            _collect_leaves_on_face!(result, tree, level + 1, ci, cj + 1, :top)
            _collect_leaves_on_face!(result, tree, level + 1, ci + 1, cj + 1, :top)
        end
    end
end

# ---------------------------------------------------------------------------
# 2:1 Balance enforcement
# ---------------------------------------------------------------------------

"""
    enforce_balance!(tree)

Ensure the 2:1 balance constraint: no leaf cell has a neighbor more than 1 level apart.
Refines cells as needed, iterating until stable.
"""
function enforce_balance!(tree::QuadTree)
    changed = true
    while changed
        changed = false
        # Collect cells to refine (can't modify while iterating)
        to_refine = Tuple{Int,Int,Int}[]

        for (level, i, j) in tree.leaf_list
            if level >= tree.max_level
                continue
            end
            for dir in (:left, :right, :bottom, :top)
                nb = find_neighbor(tree, level, i, j, dir)
                if nb isa Vector
                    # Neighbor is finer — check how much finer
                    for (nl, _, _) in nb
                        if nl - level > 1
                            push!(to_refine, (level, i, j))
                            break
                        end
                    end
                end
            end
        end

        # Also check: if a leaf at level k is adjacent to a cell at level k+2
        # This happens when a neighbor is refined and its children are refined
        # We detect this by checking if any neighbor returns cells 2+ levels finer
        for (level, i, j) in unique(to_refine)
            if tree.active[level+1][i, j]  # still a leaf
                refine!(tree, level, i, j)
                changed = true
            end
        end
    end
    nothing
end

# ---------------------------------------------------------------------------
# Gradient estimation (for adaptation criterion)
# ---------------------------------------------------------------------------

"""Estimate the maximum gradient magnitude of a field at a leaf cell."""
function _estimate_gradient(tree::QuadTree{T}, name::Symbol, level::Int, i::Int, j::Int) where {T}
    h = cell_size(tree, level)
    val = get_field(tree, name, level, i, j)
    max_grad = zero(T)

    for dir in (:left, :right, :bottom, :top)
        nb = find_neighbor(tree, level, i, j, dir)
        if nb === nothing
            continue
        end
        if nb isa Tuple{Int,Int,Int}
            nb_level, ni, nj = nb
            nb_val = get_field(tree, name, nb_level, ni, nj)
            nb_h = cell_size(tree, nb_level)
            dist = (h + nb_h) / 2
            grad = abs(val - nb_val) / dist
            max_grad = max(max_grad, grad)
        elseif nb isa Vector
            # Multiple fine neighbors — use average
            for (nb_level, ni, nj) in nb
                nb_val = get_field(tree, name, nb_level, ni, nj)
                nb_h = cell_size(tree, nb_level)
                dist = (h + nb_h) / 2
                grad = abs(val - nb_val) / dist
                max_grad = max(max_grad, grad)
            end
        end
    end

    return max_grad
end

# ---------------------------------------------------------------------------
# Adaptive refinement
# ---------------------------------------------------------------------------

"""
    adapt!(tree; max_level, criterion, field, threshold)

Adapt the tree based on a gradient criterion:
1. Mark cells for refinement where |gradient(field)| > threshold
2. Mark cells for coarsening where |gradient(field)| < threshold/4
3. Enforce 2:1 balance
4. Execute refinement and coarsening
5. Rebuild leaf list
"""
function adapt!(tree::QuadTree{T};
                max_level::Int=tree.max_level,
                criterion::Symbol=:gradient,
                field::Symbol=:u,
                threshold::T=T(0.1)) where {T}

    # Phase 1: Mark cells for refinement/coarsening
    to_refine = Tuple{Int,Int,Int}[]
    to_coarsen = Tuple{Int,Int,Int}[]

    for (level, i, j) in tree.leaf_list
        grad = _estimate_gradient(tree, field, level, i, j)
        if grad > threshold && level < max_level
            push!(to_refine, (level, i, j))
        elseif grad < threshold / 4 && level > 0
            push!(to_coarsen, (level, i, j))
        end
    end

    # Phase 2: Execute refinement (refine first, then coarsen)
    for (level, i, j) in to_refine
        if tree.active[level+1][i, j]  # still a leaf
            refine!(tree, level, i, j)
        end
    end

    # Phase 3: Enforce 2:1 balance
    enforce_balance!(tree)

    # Phase 4: Execute coarsening (only if all 4 siblings are leaves and marked)
    coarsen_set = Set(to_coarsen)
    attempted = Set{Tuple{Int,Int,Int}}()
    for (level, i, j) in to_coarsen
        if level < 1
            continue
        end
        pi = (i + 1) >> 1
        pj = (j + 1) >> 1
        parent_key = (level - 1, pi, pj)
        if parent_key in attempted
            continue
        end
        push!(attempted, parent_key)

        ci = 2pi - 1
        cj = 2pj - 1
        siblings = [(level, ci, cj), (level, ci+1, cj), (level, ci, cj+1), (level, ci+1, cj+1)]

        # All 4 siblings must be active leaves and marked for coarsening
        all_ok = all(s -> tree.active[level+1][s[2], s[3]] && s in coarsen_set, siblings)
        if all_ok
            # Check that coarsening won't break 2:1 balance
            # The parent at level-1 must not be adjacent to cells at level+1
            coarsen!(tree, level, i, j)
        end
    end

    rebuild_leaf_list!(tree)
    nothing
end

# ---------------------------------------------------------------------------
# Field initialization
# ---------------------------------------------------------------------------

"""
    initialize_field!(tree, name, f)

Set field `name` at each leaf cell to `f(x, y)` evaluated at the cell center.
"""
function initialize_field!(tree::QuadTree{T}, name::Symbol, f::Function) where {T}
    for (level, i, j) in tree.leaf_list
        x, y = cell_center(tree, level, i, j)
        set_field!(tree, name, level, i, j, T(f(x, y)))
    end
    nothing
end
