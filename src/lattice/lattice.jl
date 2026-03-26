using StaticArrays

"""
    AbstractLattice{D, Q}

Abstract type for LBM lattice models parameterized by spatial dimension `D`
and number of discrete velocities `Q`.
"""
abstract type AbstractLattice{D, Q} end

"""
    lattice_dim(::AbstractLattice{D}) -> Int

Return spatial dimension of the lattice.
"""
lattice_dim(::AbstractLattice{D}) where D = D

"""
    lattice_q(::AbstractLattice{D,Q}) -> Int

Return number of discrete velocities.
"""
lattice_q(::AbstractLattice{D,Q}) where {D,Q} = Q

"""
    weights(lattice) -> SVector{Q, Float64}

Return quadrature weights for the lattice.
"""
function weights end

"""
    velocities(lattice) -> SMatrix{D, Q, Int32}

Return discrete velocity vectors as columns of a static matrix.
"""
function velocities end

"""
    opposite(lattice) -> SVector{Q, Int32}

Return index of the opposite direction for each velocity.
"""
function opposite end

"""
    cs2(::AbstractLattice) -> Float64

Speed of sound squared (1/3 for standard lattices).
"""
cs2(::AbstractLattice) = 1.0 / 3.0

"""
    equilibrium(lattice, ρ, u, q) -> Float64

Compute equilibrium distribution for direction `q` given density `ρ`
and velocity vector `u`.

    f_eq = w_q · ρ · (1 + (c_q · u)/cs² + (c_q · u)²/(2·cs⁴) - u·u/(2·cs²))
"""
function equilibrium end
