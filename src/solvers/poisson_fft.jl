using AbstractFFTs
using FFTW  # CPU FFT provider
using KernelAbstractions

@kernel function eigenvalue_kernel!(f_hat, inv_dx2, N_grid, ::Type{T}) where {T}
    idx = @index(Global)
    @inbounds begin
        k = mod1(idx, N_grid)
        l = div(idx - 1, N_grid) + 1
        kk = k - 1
        ll = l - 1
        eigenval = (2 * cos(2 * T(pi) * kk / N_grid) - 2) * inv_dx2 +
                   (2 * cos(2 * T(pi) * ll / N_grid) - 2) * inv_dx2

        if k == 1 && l == 1
            f_hat[idx] = zero(eltype(f_hat))
        else
            f_hat[idx] /= eigenval
        end
    end
end

"""
    solve_poisson_fft!(phi, f, dx)

Solve the 2D Poisson equation nabla^2 phi = f with periodic boundary conditions
using the FFT spectral method.

The solution is computed by transforming `f` to spectral space, dividing by the
Laplacian eigenvalues, and transforming back. The zero mode is set to zero to
fix the gauge (mean of phi = 0).

Uses AbstractFFTs interface — dispatches to FFTW (CPU), CUFFT (CUDA) automatically.
For Metal (no native FFT), falls back to CPU computation.

# Arguments
- `phi`: output array (N x N), will be overwritten with the solution
- `f`: right-hand side array (N x N)
- `dx`: uniform grid spacing

# Example
```julia
N = 64; dx = 1.0 / N
phi = zeros(N, N)
f = zeros(N, N)
solve_poisson_fft!(phi, f, dx)
```

# Returns
- `phi`: the modified solution array.

See also: [`solve_poisson_cg!`](@ref), [`solve_poisson_neumann!`](@ref)
"""
function solve_poisson_fft!(phi, f, dx)
    N = size(f, 1)
    T = eltype(f)
    backend = KernelAbstractions.get_backend(f)

    # Check if FFT is natively supported on this backend
    # Metal does not support FFT — fall back to CPU
    fft_on_cpu = _needs_cpu_fft(backend)

    if fft_on_cpu
        f_cpu = Array(f)
        phi_cpu = similar(f_cpu)
        _solve_poisson_fft_impl!(phi_cpu, f_cpu, dx)
        copyto!(phi, phi_cpu)
    else
        _solve_poisson_fft_impl!(phi, f, dx)
    end

    return phi
end

# Detect backends that don't support FFT natively
_needs_cpu_fft(::KernelAbstractions.CPU) = false
_needs_cpu_fft(::Any) = true  # Default: assume no FFT support

# If CUDA is loaded, it supports FFT via CUFFT
function _check_cuda_fft end

function _solve_poisson_fft_impl!(phi, f, dx)
    N = size(f, 1)
    T = eltype(f)
    backend = KernelAbstractions.get_backend(f)

    # Forward FFT of RHS
    f_hat = AbstractFFTs.fft(f)

    # Build Laplacian eigenvalues and divide
    inv_dx2 = one(T) / (dx * dx)

    if backend isa KernelAbstractions.CPU
        # CPU path: simple loop (avoid kernel overhead for small problems)
        @inbounds for l in 1:N, k in 1:N
            kk = k - 1
            ll = l - 1
            eigenval = (2 * cos(2 * T(pi) * kk / N) - 2) * inv_dx2 +
                       (2 * cos(2 * T(pi) * ll / N) - 2) * inv_dx2

            if k == 1 && l == 1
                f_hat[k, l] = zero(eltype(f_hat))
            else
                f_hat[k, l] /= eigenval
            end
        end
    else
        # GPU path: use kernel
        kernel! = eigenvalue_kernel!(backend)
        kernel!(f_hat, inv_dx2, N, T; ndrange=N * N)
        KernelAbstractions.synchronize(backend)
    end

    # Inverse FFT to get solution
    phi .= real.(AbstractFFTs.ifft(f_hat))
    return phi
end
