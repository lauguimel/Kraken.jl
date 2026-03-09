using KernelAbstractions

@kernel function divergence_kernel!(div, @Const(u), @Const(v), inv_2dx)
    i, j = @index(Global, NTuple)
    # Central difference on interior points (2:N-1, 2:N-1)
    @inbounds div[i+1, j+1] = (u[i+2, j+1] - u[i, j+1]) * inv_2dx +
                                (v[i+1, j+2] - v[i+1, j]) * inv_2dx
end

"""
    divergence!(div, u, v, dx; ndrange=nothing)

Compute the divergence of a 2D velocity field `(u, v)` using central differences
on a uniform grid with spacing `dx`.

    div = ∂u/∂x + ∂v/∂y

The divergence is computed for interior points `(2:N-1, 2:N-1)` only; boundary
values in `div` are left untouched.

The compute backend (CPU, CUDA, Metal) is selected automatically from the
array type via `KernelAbstractions.get_backend`.

# Arguments
- `div`: output scalar field (N × N)
- `u`: x-component of velocity (N × N)
- `v`: y-component of velocity (N × N)
- `dx`: uniform grid spacing
"""
function divergence!(div, u, v, dx; ndrange=nothing)
    backend = KernelAbstractions.get_backend(u)
    inv_2dx = one(eltype(u)) / (2 * dx)
    n = size(u, 1) - 2
    if ndrange === nothing
        ndrange = (n, n)
    end
    kernel! = divergence_kernel!(backend)
    kernel!(div, u, v, inv_2dx; ndrange=ndrange)
    KernelAbstractions.synchronize(backend)
    return div
end
