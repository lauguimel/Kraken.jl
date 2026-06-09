using LinearAlgebra
using SparseArrays

cell_center(i::Integer, N::Integer) = (Float64(i) - 0.5) / Float64(N)

linear_index(i::Integer, j::Integer, N::Integer) = Int(i + (j - 1) * N)

function cell_ij(k::Integer, N::Integer)
    1 <= k <= N * N || throw(ArgumentError("linear index k must be in 1:N^2"))
    i = mod(k - 1, N) + 1
    j = div(k - 1, N) + 1
    return i, j
end

function cell_coordinates(i::Integer, j::Integer, N::Integer)
    return cell_center(i, N), cell_center(j, N)
end

function _check_grid_size(N::Integer)
    N > 0 || throw(ArgumentError("N must be positive"))
    return Int(N)
end

function _push_entry!(I::Vector{Int}, J::Vector{Int}, V::Vector{Float64},
                      row::Int, col::Int, value::Float64)
    push!(I, row)
    push!(J, col)
    push!(V, value)
    return nothing
end

function assemble_poisson_dirichlet(N::Integer, f::Function)
    N = _check_grid_size(N)
    n = N * N
    h = 1.0 / N
    invh2 = 1.0 / (h * h)

    I = Int[]
    J = Int[]
    V = Float64[]
    sizehint!(I, 5n)
    sizehint!(J, 5n)
    sizehint!(V, 5n)
    b = Vector{Float64}(undef, n)

    for j in 1:N, i in 1:N
        k = linear_index(i, j, N)
        x, y = cell_coordinates(i, j, N)
        b[k] = Float64(f(x, y))

        diag = 4.0 * invh2

        if i > 1
            _push_entry!(I, J, V, k, linear_index(i - 1, j, N), -invh2)
        else
            diag += 1.0 * invh2
        end
        if i < N
            _push_entry!(I, J, V, k, linear_index(i + 1, j, N), -invh2)
        else
            diag += 1.0 * invh2
        end
        if j > 1
            _push_entry!(I, J, V, k, linear_index(i, j - 1, N), -invh2)
        else
            diag += 1.0 * invh2
        end
        if j < N
            _push_entry!(I, J, V, k, linear_index(i, j + 1, N), -invh2)
        else
            diag += 1.0 * invh2
        end

        _push_entry!(I, J, V, k, k, diag)
    end

    return sparse(I, J, V, n, n), b
end

function assemble_poisson_neumann_unpinned(N::Integer, f::Function)
    N = _check_grid_size(N)
    n = N * N
    h = 1.0 / N
    invh2 = 1.0 / (h * h)

    I = Int[]
    J = Int[]
    V = Float64[]
    sizehint!(I, 5n)
    sizehint!(J, 5n)
    sizehint!(V, 5n)
    b = Vector{Float64}(undef, n)

    for j in 1:N, i in 1:N
        k = linear_index(i, j, N)
        x, y = cell_coordinates(i, j, N)
        b[k] = Float64(f(x, y))

        diag = 4.0 * invh2

        if i > 1
            _push_entry!(I, J, V, k, linear_index(i - 1, j, N), -invh2)
        else
            diag -= invh2
        end
        if i < N
            _push_entry!(I, J, V, k, linear_index(i + 1, j, N), -invh2)
        else
            diag -= invh2
        end
        if j > 1
            _push_entry!(I, J, V, k, linear_index(i, j - 1, N), -invh2)
        else
            diag -= invh2
        end
        if j < N
            _push_entry!(I, J, V, k, linear_index(i, j + 1, N), -invh2)
        else
            diag -= invh2
        end

        _push_entry!(I, J, V, k, k, diag)
    end

    return sparse(I, J, V, n, n), b
end

function pin_reference_dof(A::SparseMatrixCSC{Float64, Int},
                           b::AbstractVector{<:Real},
                           k0::Integer,
                           value::Real)
    size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
    n = size(A, 1)
    length(b) == n || throw(ArgumentError("b length must match A"))
    1 <= k0 <= n || throw(ArgumentError("pin index k0 must be in 1:size(A,1)"))

    k0 = Int(k0)
    pin_value = Float64(value)
    b_pinned = Float64.(b)

    rows = rowvals(A)
    vals = nonzeros(A)
    for p in nzrange(A, k0)
        row = rows[p]
        if row != k0
            b_pinned[row] -= vals[p] * pin_value
        end
    end
    b_pinned[k0] = pin_value

    old_I, old_J, old_V = findnz(A)
    I = Int[]
    J = Int[]
    V = Float64[]
    sizehint!(I, length(old_V) + 1)
    sizehint!(J, length(old_V) + 1)
    sizehint!(V, length(old_V) + 1)

    for q in eachindex(old_V)
        if old_I[q] != k0 && old_J[q] != k0
            _push_entry!(I, J, V, old_I[q], old_J[q], old_V[q])
        end
    end
    _push_entry!(I, J, V, k0, k0, 1.0)

    return sparse(I, J, V, n, n), b_pinned
end

function assemble_poisson_neumann_pinned(N::Integer, f::Function, u_exact::Function; k0::Integer=1)
    N = _check_grid_size(N)
    A, b = assemble_poisson_neumann_unpinned(N, f)
    i0, j0 = cell_ij(k0, N)
    x0, y0 = cell_coordinates(i0, j0, N)
    return pin_reference_dof(A, b, k0, u_exact(x0, y0))
end

function solve_poisson(A::SparseMatrixCSC{Float64, Int}, b::AbstractVector{<:Real}, N::Integer)
    N = _check_grid_size(N)
    size(A, 1) == N * N || throw(ArgumentError("A size must match N^2"))
    length(b) == N * N || throw(ArgumentError("b length must match N^2"))

    factor = cholesky(Symmetric(A); check=true)
    u = factor \ Float64.(b)
    return reshape(Vector{Float64}(u), N, N)
end

function solve_poisson_dirichlet(N::Integer, f::Function)
    A, b = assemble_poisson_dirichlet(N, f)
    return solve_poisson(A, b, N)
end

function solve_poisson_neumann(N::Integer, f::Function, u_exact::Function; k0::Integer=1)
    A, b = assemble_poisson_neumann_pinned(N, f, u_exact; k0=k0)
    return solve_poisson(A, b, N)
end

function exact_field(N::Integer, u_exact::Function)
    N = _check_grid_size(N)
    u = Matrix{Float64}(undef, N, N)
    for j in 1:N, i in 1:N
        x, y = cell_coordinates(i, j, N)
        u[i, j] = Float64(u_exact(x, y))
    end
    return u
end

function l2_error(u::AbstractMatrix{<:Real}, u_exact::Function, N::Integer)
    N = _check_grid_size(N)
    size(u) == (N, N) || throw(ArgumentError("u must have size (N, N)"))

    h = 1.0 / N
    err2 = 0.0
    for j in 1:N, i in 1:N
        x, y = cell_coordinates(i, j, N)
        diff = Float64(u[i, j]) - Float64(u_exact(x, y))
        err2 += diff * diff
    end
    return sqrt(h * h * err2)
end
