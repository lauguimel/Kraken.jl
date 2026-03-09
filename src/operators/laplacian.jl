using KernelAbstractions

@kernel function laplacian_kernel!(out, @Const(f), inv_dx2)
    i, j = @index(Global, NTuple)
    @inbounds out[i+1, j+1] = (f[i+2, j+1] + f[i, j+1] +
                                f[i+1, j+2] + f[i+1, j] -
                                4 * f[i+1, j+1]) * inv_dx2
end

"""
    laplacian!(out, f, dx; ndrange=nothing)

Compute the discrete Laplacian of 2D field `f` using a 5-point stencil on a
uniform grid with spacing `dx`.

The arrays `out` and `f` must have the same size `(N, N)`. The Laplacian is
computed for interior points `(2:N-1, 2:N-1)` only; boundary values in `out`
are left untouched.

The compute backend (CPU, CUDA, Metal) is selected automatically from the
array type via `KernelAbstractions.get_backend`.

# Arguments
- `out`: output array, same size as `f`
- `f`: input scalar field
- `dx`: uniform grid spacing

# Example
```julia
N = 64; dx = 1.0 / (N - 1)
f = zeros(N, N)
out = zeros(N, N)
laplacian!(out, f, dx)
```
"""
function laplacian!(out, f, dx; ndrange=nothing)
    backend = KernelAbstractions.get_backend(f)
    inv_dx2 = one(eltype(f)) / (dx * dx)
    n = size(f, 1) - 2  # interior points count
    if ndrange === nothing
        ndrange = (n, n)
    end
    kernel! = laplacian_kernel!(backend)
    kernel!(out, f, inv_dx2; ndrange=ndrange)
    KernelAbstractions.synchronize(backend)
    return out
end
