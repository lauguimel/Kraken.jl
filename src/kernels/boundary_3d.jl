using KernelAbstractions

# ============================================================================
# Zou-He 3D boundary conditions — generic parametrized implementation
# ============================================================================
#
# Each face is described by a ZouHeFace3D struct that encodes:
#   - normal_axis: 1 (x), 2 (y), 3 (z)
#   - sign: +1 for max-index face, -1 for min-index face
#   - Population indices for the 5 unknown and 5 opposing (known) populations
#   - Tangential axis population pairs for transverse corrections
#
# Two generic kernels handle all 6 velocity BCs and 2 pressure BCs.
# The old apply_zou_he_*_3d! wrapper functions are preserved for backward
# compatibility.
# ============================================================================

"""
    ZouHeFace3D

Parametrizes a Zou-He boundary face in 3D (D3Q19).

# Fields
- `normal_axis::Int32`: 1 (x), 2 (y), or 3 (z)
- `sign::Int32`: +1 for face at max index (e.g. east/north/top),
                 -1 for face at min index (e.g. west/south/bottom)
- `q_axis::Int32`: index of the axis-aligned unknown population
- `q_axis_opp::Int32`: its opposite (the known outgoing axis-aligned pop)
- `q_diag::NTuple{4,Int32}`: 4 diagonal unknown population indices
- `q_diag_opp::NTuple{4,Int32}`: their opposites (known outgoing)
- `tang1_plus::Int32`: index of +tangent1 axis-aligned pop (for transverse correction)
- `tang1_minus::Int32`: index of -tangent1 axis-aligned pop
- `tang2_plus::Int32`: index of +tangent2 axis-aligned pop
- `tang2_minus::Int32`: index of -tangent2 axis-aligned pop
- `parallel::NTuple{9,Int32}`: 9 populations with zero normal component
                                (rest + 4 axis-aligned in tang planes + 4 pure-tangential diags)
- `outgoing::NTuple{5,Int32}`: 5 known outgoing populations (for density sum)
"""
struct ZouHeFace3D
    normal_axis::Int32     # 1=x, 2=y, 3=z
    sign::Int32            # +1 max face, -1 min face
    q_axis::Int32          # unknown axis-aligned pop
    q_axis_opp::Int32      # opposing known axis-aligned pop
    q_diag::NTuple{4,Int32}
    q_diag_opp::NTuple{4,Int32}
    tang1_plus::Int32
    tang1_minus::Int32
    tang2_plus::Int32
    tang2_minus::Int32
    parallel::NTuple{9,Int32}
    outgoing::NTuple{5,Int32}
end

# --- Pre-built face descriptors ---
# D3Q19 population indexing (1-based):
#  1: (0,0,0)  2: (+x)  3: (-x)  4: (+y)  5: (-y)  6: (+z)  7: (-z)
#  8: (+x,+y) 9: (-x,+y) 10: (+x,-y) 11: (-x,-y)
# 12: (+x,+z) 13: (-x,+z) 14: (+x,-z) 15: (-x,-z)
# 16: (+y,+z) 17: (-y,+z) 18: (+y,-z) 19: (-y,-z)

# West (i=1): unknown pops have cx=+1 => 2, 8, 10, 12, 14
const ZH_WEST = ZouHeFace3D(
    Int32(1), Int32(-1),       # normal_axis=x, sign=-1 (min face)
    Int32(2), Int32(3),        # axis: f2(+x) unknown, f3(-x) opposing
    (Int32(8), Int32(10), Int32(12), Int32(14)),   # diag unknown
    (Int32(11), Int32(9), Int32(15), Int32(13)),    # diag opposing
    Int32(4), Int32(5),        # tang1 = y: +y=4, -y=5
    Int32(6), Int32(7),        # tang2 = z: +z=6, -z=7
    (Int32(1), Int32(4), Int32(5), Int32(6), Int32(7),
     Int32(16), Int32(17), Int32(18), Int32(19)),   # parallel (cx=0)
    (Int32(3), Int32(9), Int32(11), Int32(13), Int32(15)),  # outgoing (cx=-1)
)

# East (i=Nx): unknown pops have cx=-1 => 3, 9, 11, 13, 15
const ZH_EAST = ZouHeFace3D(
    Int32(1), Int32(1),        # normal_axis=x, sign=+1 (max face)
    Int32(3), Int32(2),        # axis: f3(-x) unknown, f2(+x) opposing
    (Int32(9), Int32(11), Int32(13), Int32(15)),
    (Int32(10), Int32(8), Int32(14), Int32(12)),
    Int32(4), Int32(5),        # tang1 = y
    Int32(6), Int32(7),        # tang2 = z
    (Int32(1), Int32(4), Int32(5), Int32(6), Int32(7),
     Int32(16), Int32(17), Int32(18), Int32(19)),
    (Int32(2), Int32(8), Int32(10), Int32(12), Int32(14)),
)

# South (j=1): unknown pops have cy=+1 => 4, 8, 9, 16, 18
const ZH_SOUTH = ZouHeFace3D(
    Int32(2), Int32(-1),       # normal_axis=y, sign=-1 (min face)
    Int32(4), Int32(5),        # axis: f4(+y) unknown, f5(-y) opposing
    (Int32(8), Int32(9), Int32(16), Int32(18)),
    (Int32(11), Int32(10), Int32(19), Int32(17)),
    Int32(2), Int32(3),        # tang1 = x: +x=2, -x=3
    Int32(6), Int32(7),        # tang2 = z: +z=6, -z=7
    (Int32(1), Int32(2), Int32(3), Int32(6), Int32(7),
     Int32(12), Int32(13), Int32(14), Int32(15)),
    (Int32(5), Int32(10), Int32(11), Int32(17), Int32(19)),
)

# North (j=Ny): unknown pops have cy=-1 => 5, 10, 11, 17, 19
const ZH_NORTH = ZouHeFace3D(
    Int32(2), Int32(1),        # normal_axis=y, sign=+1 (max face)
    Int32(5), Int32(4),        # axis: f5(-y) unknown, f4(+y) opposing
    (Int32(10), Int32(11), Int32(17), Int32(19)),
    (Int32(9), Int32(8), Int32(18), Int32(16)),
    Int32(2), Int32(3),        # tang1 = x
    Int32(6), Int32(7),        # tang2 = z
    (Int32(1), Int32(2), Int32(3), Int32(6), Int32(7),
     Int32(12), Int32(13), Int32(14), Int32(15)),
    (Int32(4), Int32(8), Int32(9), Int32(16), Int32(18)),
)

# Bottom (k=1): unknown pops have cz=+1 => 6, 12, 13, 16, 17
const ZH_BOTTOM = ZouHeFace3D(
    Int32(3), Int32(-1),       # normal_axis=z, sign=-1 (min face)
    Int32(6), Int32(7),        # axis: f6(+z) unknown, f7(-z) opposing
    (Int32(12), Int32(13), Int32(16), Int32(17)),
    (Int32(15), Int32(14), Int32(19), Int32(18)),
    Int32(2), Int32(3),        # tang1 = x: +x=2, -x=3
    Int32(4), Int32(5),        # tang2 = y: +y=4, -y=5
    (Int32(1), Int32(2), Int32(3), Int32(4), Int32(5),
     Int32(8), Int32(9), Int32(10), Int32(11)),
    (Int32(7), Int32(14), Int32(15), Int32(18), Int32(19)),
)

# Top (k=Nz): unknown pops have cz=-1 => 7, 14, 15, 18, 19
const ZH_TOP = ZouHeFace3D(
    Int32(3), Int32(1),        # normal_axis=z, sign=+1 (max face)
    Int32(7), Int32(6),        # axis: f7(-z) unknown, f6(+z) opposing
    (Int32(14), Int32(15), Int32(18), Int32(19)),
    (Int32(13), Int32(12), Int32(17), Int32(16)),
    Int32(2), Int32(3),        # tang1 = x
    Int32(4), Int32(5),        # tang2 = y
    (Int32(1), Int32(2), Int32(3), Int32(4), Int32(5),
     Int32(8), Int32(9), Int32(10), Int32(11)),
    (Int32(6), Int32(12), Int32(13), Int32(16), Int32(17)),
)

# ============================================================================
# Generic Zou-He velocity kernel for D3Q19
# ============================================================================

"""
    zou_he_velocity_3d_kernel!(f, face, idx_fixed, u_normal, u_tang1, u_tang2)

Generic Zou-He velocity BC kernel for any face in D3Q19.

- `face`: ZouHeFace3D descriptor
- `idx_fixed`: fixed index value on the face (1 or N)
- `u_normal`: wall velocity along the face normal (positive = into domain)
- `u_tang1`: wall velocity along first tangential axis
- `u_tang2`: wall velocity along second tangential axis
"""
@kernel function zou_he_velocity_3d_kernel!(f, face::ZouHeFace3D,
                                            idx_fixed, u_normal, u_tang1, u_tang2)
    a, b = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f)

        # Map 2D thread indices to 3D grid position based on face normal
        if face.normal_axis == Int32(1)       # x-face: fixed i, iterate (j,k)
            i, j, k = idx_fixed, a, b
        elseif face.normal_axis == Int32(2)   # y-face: fixed j, iterate (i,k)
            i, j, k = a, idx_fixed, b
        else                                   # z-face: fixed k, iterate (i,j)
            i, j, k = a, b, idx_fixed
        end

        # Sum parallel populations (zero normal component)
        sum_par = zero(T)
        for n in 1:9
            sum_par += f[i,j,k,face.parallel[n]]
        end

        # Sum outgoing populations (known, pointing away from domain)
        sum_out = zero(T)
        for n in 1:5
            sum_out += f[i,j,k,face.outgoing[n]]
        end

        # Density: ρ = (sum_par + 2*sum_out) / (1 + sign*u_normal)
        #   sign=-1 (min face, inflow): /(1 - u_n)
        #   sign=+1 (max face, outflow): /(1 + u_n)
        #   u_normal=0: no division effect
        denom = one(T) + T(face.sign) * u_normal
        ρ_wall = (sum_par + T(2) * sum_out) / denom

        # Axis-aligned unknown: f_axis = f_opp - sign * 2/3 * ρ * u_n
        f[i,j,k,face.q_axis] = f[i,j,k,face.q_axis_opp] -
                                T(face.sign) * T(2.0/3.0) * ρ_wall * u_normal

        # Read tangential population differences
        tang1_diff = f[i,j,k,face.tang1_plus] - f[i,j,k,face.tang1_minus]
        tang2_diff = f[i,j,k,face.tang2_plus] - f[i,j,k,face.tang2_minus]

        # Diagonal unknowns: 4 populations grouped in pairs
        # Pair 1 (tang1 plane): indices 1,2 in q_diag
        f[i,j,k,face.q_diag[1]] = f[i,j,k,face.q_diag_opp[1]] -
            T(0.5) * tang1_diff + T(0.5) * ρ_wall * u_tang1 -
            T(face.sign) * T(1.0/6.0) * ρ_wall * u_normal
        f[i,j,k,face.q_diag[2]] = f[i,j,k,face.q_diag_opp[2]] +
            T(0.5) * tang1_diff - T(0.5) * ρ_wall * u_tang1 -
            T(face.sign) * T(1.0/6.0) * ρ_wall * u_normal

        # Pair 2 (tang2 plane): indices 3,4 in q_diag
        f[i,j,k,face.q_diag[3]] = f[i,j,k,face.q_diag_opp[3]] -
            T(0.5) * tang2_diff + T(0.5) * ρ_wall * u_tang2 -
            T(face.sign) * T(1.0/6.0) * ρ_wall * u_normal
        f[i,j,k,face.q_diag[4]] = f[i,j,k,face.q_diag_opp[4]] +
            T(0.5) * tang2_diff - T(0.5) * ρ_wall * u_tang2 -
            T(face.sign) * T(1.0/6.0) * ρ_wall * u_normal
    end
end

# ============================================================================
# Generic Zou-He pressure kernel for D3Q19
# ============================================================================

"""
    zou_he_pressure_3d_kernel!(f, face, idx_fixed, ρ_out)

Generic Zou-He pressure outlet BC kernel for any face in D3Q19.
Fixes density to `ρ_out` and computes the normal velocity from known populations.
"""
@kernel function zou_he_pressure_3d_kernel!(f, face::ZouHeFace3D,
                                            idx_fixed, ρ_out)
    a, b = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f)

        if face.normal_axis == Int32(1)
            i, j, k = idx_fixed, a, b
        elseif face.normal_axis == Int32(2)
            i, j, k = a, idx_fixed, b
        else
            i, j, k = a, b, idx_fixed
        end

        # Sum parallel populations
        sum_par = zero(T)
        for n in 1:9
            sum_par += f[i,j,k,face.parallel[n]]
        end

        # Sum outgoing populations
        sum_out = zero(T)
        for n in 1:5
            sum_out += f[i,j,k,face.outgoing[n]]
        end

        # Compute normal velocity: u_n = -sign * (1 - (sum_par + 2*sum_out) / ρ_out)
        #   For max face (sign=+1): u_n = -1 + (sum_par + 2*sum_out) / ρ_out
        #   For min face (sign=-1): u_n = +1 - (sum_par + 2*sum_out) / ρ_out
        u_n = -T(face.sign) * (one(T) - (sum_par + T(2) * sum_out) / ρ_out)

        # Axis-aligned unknown
        f[i,j,k,face.q_axis] = f[i,j,k,face.q_axis_opp] -
                                T(face.sign) * T(2.0/3.0) * ρ_out * u_n

        # Tangential differences
        tang1_diff = f[i,j,k,face.tang1_plus] - f[i,j,k,face.tang1_minus]
        tang2_diff = f[i,j,k,face.tang2_plus] - f[i,j,k,face.tang2_minus]

        # Diagonal unknowns (no tangential velocity => u_tang terms absent)
        f[i,j,k,face.q_diag[1]] = f[i,j,k,face.q_diag_opp[1]] -
            T(0.5) * tang1_diff -
            T(face.sign) * T(1.0/6.0) * ρ_out * u_n
        f[i,j,k,face.q_diag[2]] = f[i,j,k,face.q_diag_opp[2]] +
            T(0.5) * tang1_diff -
            T(face.sign) * T(1.0/6.0) * ρ_out * u_n

        f[i,j,k,face.q_diag[3]] = f[i,j,k,face.q_diag_opp[3]] -
            T(0.5) * tang2_diff -
            T(face.sign) * T(1.0/6.0) * ρ_out * u_n
        f[i,j,k,face.q_diag[4]] = f[i,j,k,face.q_diag_opp[4]] +
            T(0.5) * tang2_diff -
            T(face.sign) * T(1.0/6.0) * ρ_out * u_n
    end
end

# ============================================================================
# Generic dispatch helper
# ============================================================================

function _apply_zou_he_velocity_3d!(f, face::ZouHeFace3D, idx_fixed,
                                    u_normal, u_tang1, u_tang2, N1, N2)
    backend = KernelAbstractions.get_backend(f)
    T = eltype(f)
    kernel! = zou_he_velocity_3d_kernel!(backend)
    kernel!(f, face, idx_fixed, T(u_normal), T(u_tang1), T(u_tang2); ndrange=(N1, N2))
    KernelAbstractions.synchronize(backend)
end

function _apply_zou_he_pressure_3d!(f, face::ZouHeFace3D, idx_fixed,
                                    ρ_out, N1, N2)
    backend = KernelAbstractions.get_backend(f)
    T = eltype(f)
    kernel! = zou_he_pressure_3d_kernel!(backend)
    kernel!(f, face, idx_fixed, T(ρ_out); ndrange=(N1, N2))
    KernelAbstractions.synchronize(backend)
end

# ============================================================================
# Backward-compatible wrapper functions (public API unchanged)
# ============================================================================

# --- Top face (k = Nz) ---

"""
    apply_zou_he_top_3d!(f, u_wall_x, Nx, Ny, Nz)

Apply Zou-He velocity BC on the top face (k = Nz).
Imposes velocity (u_wall_x, 0, 0).
"""
function apply_zou_he_top_3d!(f, u_wall_x, Nx, Ny, Nz)
    # Top: normal=z, sign=+1, u_normal=0, tang1=x => u_tang1=u_wall_x, tang2=y => u_tang2=0
    _apply_zou_he_velocity_3d!(f, ZH_TOP, Nz, 0, u_wall_x, 0, Nx, Ny)
end

# --- Bottom face (k = 1) ---

"""
    apply_zou_he_bottom_3d!(f, ux_w, uy_w, Nx, Ny)

Apply Zou-He velocity BC on the bottom face (k = 1).
Imposes velocity (ux_w, uy_w, 0).
"""
function apply_zou_he_bottom_3d!(f, ux_w, uy_w, Nx, Ny)
    # Bottom: normal=z, sign=-1, u_normal=0, tang1=x => ux_w, tang2=y => uy_w
    _apply_zou_he_velocity_3d!(f, ZH_BOTTOM, 1, 0, ux_w, uy_w, Nx, Ny)
end

# --- West face (i = 1) ---

"""
    apply_zou_he_west_3d!(f, ux_w, uy_w, uz_w, Ny, Nz)

Apply Zou-He velocity BC on the west face (i = 1).
Unknown populations (cx=+1): f2(+x), f8(+x,+y), f10(+x,-y), f12(+x,+z), f14(+x,-z).
"""
function apply_zou_he_west_3d!(f, ux_w, uy_w, uz_w, Ny, Nz)
    # West: normal=x, sign=-1, u_normal=ux_w, tang1=y => uy_w, tang2=z => uz_w
    _apply_zou_he_velocity_3d!(f, ZH_WEST, 1, ux_w, uy_w, uz_w, Ny, Nz)
end

# --- East face (i = Nx) ---

"""
    apply_zou_he_east_3d!(f, ux_w, uy_w, uz_w, Nx, Ny, Nz)

Apply Zou-He velocity BC on the east face (i = Nx).
Unknown populations (cx=-1): f3(-x), f9(-x,+y), f11(-x,-y), f13(-x,+z), f15(-x,-z).
"""
function apply_zou_he_east_3d!(f, ux_w, uy_w, uz_w, Nx, Ny, Nz)
    # East: normal=x, sign=+1, u_normal=ux_w, tang1=y => uy_w, tang2=z => uz_w
    _apply_zou_he_velocity_3d!(f, ZH_EAST, Nx, ux_w, uy_w, uz_w, Ny, Nz)
end

# --- South face (j = 1) ---

"""
    apply_zou_he_south_3d!(f, ux_w, uy_w, uz_w, Nx, Nz)

Apply Zou-He velocity BC on the south face (j = 1).
Unknown populations (cy=+1): f4(+y), f8(+x,+y), f9(-x,+y), f16(+y,+z), f18(+y,-z).
"""
function apply_zou_he_south_3d!(f, ux_w, uy_w, uz_w, Nx, Nz)
    # South: normal=y, sign=-1, u_normal=uy_w, tang1=x => ux_w, tang2=z => uz_w
    _apply_zou_he_velocity_3d!(f, ZH_SOUTH, 1, uy_w, ux_w, uz_w, Nx, Nz)
end

# --- North face (j = Ny) ---

"""
    apply_zou_he_north_3d!(f, ux_w, uy_w, uz_w, Nx, Ny, Nz)

Apply Zou-He velocity BC on the north face (j = Ny).
Unknown populations (cy=-1): f5(-y), f10(+x,-y), f11(-x,-y), f17(-y,+z), f19(-y,-z).
"""
function apply_zou_he_north_3d!(f, ux_w, uy_w, uz_w, Nx, Ny, Nz)
    # North: normal=y, sign=+1, u_normal=uy_w, tang1=x => ux_w, tang2=z => uz_w
    _apply_zou_he_velocity_3d!(f, ZH_NORTH, Ny, uy_w, ux_w, uz_w, Nx, Nz)
end

# --- Pressure outlet on east face (i = Nx) ---

"""
    apply_zou_he_pressure_east_3d!(f, Nx, Ny, Nz; ρ_out=1.0)

Apply Zou-He pressure outlet BC on the east face (i = Nx).
Fixes density ρ_out, computes ux from known populations.
"""
function apply_zou_he_pressure_east_3d!(f, Nx, Ny, Nz; ρ_out=1.0)
    _apply_zou_he_pressure_3d!(f, ZH_EAST, Nx, ρ_out, Ny, Nz)
end

# --- Pressure outlet on top face (k = Nz) ---

"""
    apply_zou_he_pressure_top_3d!(f, Nx, Ny, Nz; ρ_out=1.0)

Apply Zou-He pressure outlet BC on the top face (k = Nz).
Fixes density ρ_out, computes uz from known populations.
"""
function apply_zou_he_pressure_top_3d!(f, Nx, Ny, Nz; ρ_out=1.0)
    _apply_zou_he_pressure_3d!(f, ZH_TOP, Nz, ρ_out, Nx, Ny)
end

# ============================================================================
# Bounce-back on walls (unchanged)
# ============================================================================

@kernel function bounce_back_face_3d_kernel!(f, N1, N2, Nfull, face_id)
    a, b = @index(Global, NTuple)
    @inbounds begin
        if face_id == 1  # Bottom (k=1): reflect populations pointing +z
            i, j, k = a, b, 1
            f[i,j,k,6]  = f[i,j,k,7]    # +z <- -z
            f[i,j,k,12] = f[i,j,k,15]   # +x,+z <- -x,-z
            f[i,j,k,13] = f[i,j,k,14]   # -x,+z <- +x,-z
            f[i,j,k,16] = f[i,j,k,19]   # +y,+z <- -y,-z
            f[i,j,k,17] = f[i,j,k,18]   # -y,+z <- +y,-z
        elseif face_id == 2  # South (j=1)
            i, k = a, b
            j = 1
            f[i,j,k,4]  = f[i,j,k,5]    # +y <- -y
            f[i,j,k,8]  = f[i,j,k,11]   # +x,+y <- -x,-y
            f[i,j,k,9]  = f[i,j,k,10]   # -x,+y <- +x,-y
            f[i,j,k,16] = f[i,j,k,19]   # +y,+z <- -y,-z
            f[i,j,k,18] = f[i,j,k,17]   # +y,-z <- -y,+z
        elseif face_id == 3  # North (j=Ny)
            i, k = a, b
            j = N1  # Ny
            f[i,j,k,5]  = f[i,j,k,4]
            f[i,j,k,10] = f[i,j,k,9]
            f[i,j,k,11] = f[i,j,k,8]
            f[i,j,k,17] = f[i,j,k,18]
            f[i,j,k,19] = f[i,j,k,16]
        elseif face_id == 4  # West (i=1)
            j, k = a, b
            i = 1
            f[i,j,k,2]  = f[i,j,k,3]    # +x <- -x
            f[i,j,k,8]  = f[i,j,k,11]   # +x,+y <- -x,-y
            f[i,j,k,10] = f[i,j,k,9]    # +x,-y <- -x,+y
            f[i,j,k,12] = f[i,j,k,15]   # +x,+z <- -x,-z
            f[i,j,k,14] = f[i,j,k,13]   # +x,-z <- -x,+z
        elseif face_id == 5  # East (i=Nx)
            j, k = a, b
            i = Nfull  # Nx
            f[i,j,k,3]  = f[i,j,k,2]
            f[i,j,k,9]  = f[i,j,k,10]
            f[i,j,k,11] = f[i,j,k,8]
            f[i,j,k,13] = f[i,j,k,14]
            f[i,j,k,15] = f[i,j,k,12]
        end
    end
end

"""
    apply_bounce_back_walls_3d!(f, Nx, Ny, Nz)

Apply bounce-back on bottom, south, north, west, east faces.
Top face (k=Nz) is handled by Zou-He.
"""
function apply_bounce_back_walls_3d!(f, Nx, Ny, Nz)
    backend = KernelAbstractions.get_backend(f)
    kernel! = bounce_back_face_3d_kernel!(backend)

    # Bottom (k=1): ndrange = (Nx, Ny)
    kernel!(f, Nx, Ny, Nx, Int32(1); ndrange=(Nx, Ny))
    # South (j=1): ndrange = (Nx, Nz)
    kernel!(f, Ny, Nz, Nx, Int32(2); ndrange=(Nx, Nz))
    # North (j=Ny): ndrange = (Nx, Nz)
    kernel!(f, Ny, Nz, Nx, Int32(3); ndrange=(Nx, Nz))
    # West (i=1): ndrange = (Ny, Nz)
    kernel!(f, Ny, Nz, Nx, Int32(4); ndrange=(Ny, Nz))
    # East (i=Nx): ndrange = (Ny, Nz)
    kernel!(f, Ny, Nz, Nx, Int32(5); ndrange=(Ny, Nz))

    KernelAbstractions.synchronize(backend)
end
