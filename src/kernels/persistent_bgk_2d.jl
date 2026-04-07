using KernelAbstractions

# =====================================================================
# Persistent BGK wrappers: queue Nt timesteps in one host call,
# with a SINGLE synchronize at the end. Amortizes launch overhead and
# host-side bookkeeping.
# =====================================================================

"""
    persistent_fused_bgk!(f_in, f_out, ρ, ux, uy, is_solid, Nx, Ny, ω, Nt;
                          workgroupsize=nothing)

Run `Nt` fused BGK steps with a single host-side synchronize at the end.
Ping-pongs `f_in`/`f_out` internally; if `Nt` is odd the buffers end up
swapped relative to the input (caller should swap accordingly).
"""
function persistent_fused_bgk!(f_in, f_out, ρ, ux, uy, is_solid, Nx, Ny, ω, Nt;
                                workgroupsize=nothing)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    kernel! = isnothing(workgroupsize) ?
        fused_bgk_step_kernel!(backend) :
        fused_bgk_step_kernel!(backend, workgroupsize)
    a, b = f_in, f_out
    for t in 1:Nt
        kernel!(b, a, ρ, ux, uy, is_solid, Nx, Ny, ET(ω); ndrange=(Nx, Ny))
        a, b = b, a
    end
    KernelAbstractions.synchronize(backend)
    return a, b
end

"""
    persistent_aa_bgk!(f, is_solid, Nx, Ny, ω, Nt; workgroupsize=nothing)

Run `Nt` AA-pattern BGK steps (alternating even/odd) with a single
synchronize at the end. Optional `workgroupsize` tuple (e.g. `(16,16)`)
for block-size tuning.
"""
function persistent_aa_bgk!(f, is_solid, Nx, Ny, ω, Nt; workgroupsize=nothing)
    backend = KernelAbstractions.get_backend(f)
    ET = eltype(f)
    even_k! = isnothing(workgroupsize) ?
        aa_even_kernel!(backend) :
        aa_even_kernel!(backend, workgroupsize)
    odd_k! = isnothing(workgroupsize) ?
        aa_odd_kernel!(backend) :
        aa_odd_kernel!(backend, workgroupsize)
    for t in 1:Nt
        if iseven(t)
            even_k!(f, is_solid, Nx, Ny, ET(ω); ndrange=(Nx, Ny))
        else
            odd_k!(f, is_solid, Nx, Ny, ET(ω); ndrange=(Nx, Ny))
        end
    end
    KernelAbstractions.synchronize(backend)
    return f
end
