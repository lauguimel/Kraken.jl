# =====================================================================
# Block state with ghost layer (Phase A.5b).
#
# Each block's f / ρ / ux / uy are stored on an EXTENDED grid of size
# (Nξ + 2·Ng, Nη + 2·Ng) where Ng (default 1 for D2Q9) is the ghost
# layer width. The physical interior occupies indices
#   (Ng + 1 .. Ng + Nξ,  Ng + 1 .. Ng + Nη)
# and the ghost rows wrap it on all 4 sides.
#
# This matches the pattern already used in `src/refinement/refinement.jl`
# (`RefinementPatch` with `n_ghost = 2`) and is what gives bit-exact
# single-block equivalence on a multi-block canal: the step kernel
# reads from ghost cells that were pre-filled from the neighbour
# block's interior BEFORE the step, so the halfway-BB fallback never
# fires on a real interior-boundary cell.
#
# Physical BCs are applied post-step on a VIEW of the physical
# interior, exactly as in single-block. The outer ghost row is left
# to contain whatever halfway-BB garbage the step writes — it gets
# overwritten by the next iteration's ghost fill.
#
# Imperative modular: one struct, four allocator / accessor functions,
# no macros. The existing DSL step kernels see the extended array as
# if it were a single-block array of size (Nξ+2Ng, Nη+2Ng).
# =====================================================================

"""
    BlockState2D{T, AT3, AT2}

Runtime state for one block in a multi-block simulation. Stores the
populations and macroscopic fields on an EXTENDED grid that includes
`n_ghost` cells on every side. The physical interior is a view over
`(n_ghost + 1 .. n_ghost + Nξ_phys, n_ghost + 1 .. n_ghost + Nη_phys)`.

Fields
------
- `f::AT3`           — populations, size `(Nξ_ext, Nη_ext, 9)`
- `ρ::AT2`, `ux::AT2`, `uy::AT2` — macroscopic fields, `(Nξ_ext, Nη_ext)`
- `Nξ_phys, Nη_phys::Int`   — physical grid dimensions
- `n_ghost::Int`            — ghost-layer width (1 for D2Q9, 2 for higher stencils)

Allocate via `allocate_block_state_2d(block; n_ghost=1, backend=CPU())`.
The allocator initialises `ρ ≡ 1`, `ux = uy ≡ 0`, `f = 0` (caller is
expected to overwrite `f` with equilibrium before the first step).
"""
mutable struct BlockState2D{T<:AbstractFloat, AT3<:AbstractArray{T, 3}, AT2<:AbstractArray{T, 2}}
    f::AT3
    ρ::AT2
    ux::AT2
    uy::AT2
    const Nξ_phys::Int
    const Nη_phys::Int
    const n_ghost::Int
end

"""
    allocate_block_state_2d(block::Block; n_ghost=1,
                             backend=KernelAbstractions.CPU()) -> BlockState2D

Allocate f / ρ / ux / uy of the extended size (Nξ_phys + 2·n_ghost) ×
(Nη_phys + 2·n_ghost). `ρ` initialised to 1, velocities to 0, `f` to 0
(caller seeds equilibrium before the first step).
"""
function allocate_block_state_2d(block::Block{T, AT2}; n_ghost::Int=1,
                                  backend=KernelAbstractions.CPU()) where {T, AT2}
    Nx_ext = block.mesh.Nξ + 2 * n_ghost
    Ny_ext = block.mesh.Nη + 2 * n_ghost
    f  = KernelAbstractions.allocate(backend, T, Nx_ext, Ny_ext, 9); fill!(f,  zero(T))
    ρ  = KernelAbstractions.allocate(backend, T, Nx_ext, Ny_ext);    fill!(ρ,  one(T))
    ux = KernelAbstractions.allocate(backend, T, Nx_ext, Ny_ext);    fill!(ux, zero(T))
    uy = KernelAbstractions.allocate(backend, T, Nx_ext, Ny_ext);    fill!(uy, zero(T))
    return BlockState2D{T, typeof(f), typeof(ρ)}(f, ρ, ux, uy,
                                                   block.mesh.Nξ, block.mesh.Nη, n_ghost)
end

"""
    interior_f(state::BlockState2D) -> SubArray

View of the physical interior populations, size `(Nξ_phys, Nη_phys, 9)`.
Reads and writes to the view propagate to the underlying extended array;
use this for initialisation, physical-BC application, and diagnostic
extraction.
"""
@inline function interior_f(state::BlockState2D)
    ng = state.n_ghost
    return view(state.f,
                 (ng + 1):(ng + state.Nξ_phys),
                 (ng + 1):(ng + state.Nη_phys),
                 :)
end

"""
    interior_macro(state::BlockState2D) -> (ρ_view, ux_view, uy_view)

Views of the physical interior for the macroscopic fields. Same
index convention as `interior_f`.
"""
@inline function interior_macro(state::BlockState2D)
    ng = state.n_ghost
    Ii = (ng + 1):(ng + state.Nξ_phys)
    Ij = (ng + 1):(ng + state.Nη_phys)
    return view(state.ρ, Ii, Ij), view(state.ux, Ii, Ij), view(state.uy, Ii, Ij)
end

"""
    ext_dims(state::BlockState2D) -> (Nξ_ext, Nη_ext)

Extended-array dimensions, including ghost rows on both sides. Useful
when calling existing single-block step kernels: pass these as
`Nx = Nξ_ext, Ny = Nη_ext`.
"""
@inline ext_dims(state::BlockState2D) =
    (state.Nξ_phys + 2 * state.n_ghost, state.Nη_phys + 2 * state.n_ghost)
