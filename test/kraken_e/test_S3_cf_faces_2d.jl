const KRAKEN_E_S3_ATOL = 1e-12

function _s3_fill_velocity_constant!(block, ux_value, uy_value)
    ux = Kraken.kraken_e_interior_view_2d(block.ux, block)
    uy = Kraken.kraken_e_interior_view_2d(block.uy, block)
    fill!(ux, ux_value)
    fill!(uy, uy_value)
    return nothing
end

function _s3_fill_velocity_affine!(block, ux0, ux_x, ux_y, uy0, uy_x, uy_y)
    for j in 1:block.Ny, i in 1:block.Nx
        x = (i - 0.5) * block.dx
        y = (j - 0.5) * block.dx
        block.ux[block.ng + i, block.ng + j] = ux0 + ux_x * x + ux_y * y
        block.uy[block.ng + i, block.ng + j] = uy0 + uy_x * x + uy_y * y
    end
    return nothing
end

function _s3_phi(alpha, beta_x, beta_y, point)
    return alpha + beta_x * point[1] + beta_y * point[2]
end

function _s3_apply_tangential_stencil(record, weights, offsets, alpha, beta_x, beta_y)
    value = 0.0
    for k in 1:3
        offset = offsets[k]
        point = record.axis == Kraken.KRAKEN_E_CF_FACE_X ?
            (record.coarse_center[1], record.coarse_center[2] + offset[2] * record.coarse_area) :
            (record.coarse_center[1] + offset[1] * record.coarse_area, record.coarse_center[2])
        value += weights[k] * _s3_phi(alpha, beta_x, beta_y, point)
    end
    return value
end

@testset "Kraken-E S3 D3+D4" begin
    Nx = 32
    Ny = 32
    block = Kraken.allocate_leaf_block_2d(Float64; Nx, Ny, dx=1.0)
    is_solid = falses(Nx, Ny)
    bc_periodic = Kraken.FVFDDomainBC2D(;
        west=:periodic, east=:periodic, south=:periodic, north=:periodic,
    )
    bc_wall = Kraken.FVFDDomainBC2D(;
        west=:wall, east=:wall, south=:wall, north=:wall,
    )

    dudx = zeros(Float64, Nx, Ny)
    dudy = zeros(Float64, Nx, Ny)
    dvdx = zeros(Float64, Nx, Ny)
    dvdy = zeros(Float64, Nx, Ny)

    _s3_fill_velocity_constant!(block, 0.5, -0.25)
    Kraken.fvfd_velocity_gradient_block_2d!(
        dudx, dudy, dvdx, dvdy, block, bc_periodic; is_solid,
    )
    max_grad_const = maximum((
        maximum(abs, dudx),
        maximum(abs, dudy),
        maximum(abs, dvdx),
        maximum(abs, dvdy),
    ))
    @test max_grad_const <= KRAKEN_E_S3_ATOL

    ux_face = zeros(Float64, Nx + 1, Ny)
    uy_face = zeros(Float64, Nx, Ny + 1)
    Kraken.fvfd_cell_velocity_to_faces_block_2d!(
        ux_face, uy_face, block, bc_periodic; is_solid,
    )
    @test maximum(abs, ux_face .- 0.5) <= KRAKEN_E_S3_ATOL
    @test maximum(abs, uy_face .+ 0.25) <= KRAKEN_E_S3_ATOL

    _s3_fill_velocity_affine!(block, 0.1, 0.03, 0.07, -0.2, 0.05, -0.04)
    fill!(dudx, 0.0)
    fill!(dudy, 0.0)
    fill!(dvdx, 0.0)
    fill!(dvdy, 0.0)
    Kraken.fvfd_velocity_gradient_block_2d!(
        dudx, dudy, dvdx, dvdy, block, bc_wall; is_solid,
    )
    err_affine_dudx = maximum(abs, dudx .- 0.03)
    err_affine_dudy = maximum(abs, dudy .- 0.07)
    err_affine_dvdx = maximum(abs, dvdx .- 0.05)
    err_affine_dvdy = maximum(abs, dvdy .+ 0.04)
    @test err_affine_dudx <= KRAKEN_E_S3_ATOL
    @test err_affine_dudy <= KRAKEN_E_S3_ATOL
    @test err_affine_dvdx <= KRAKEN_E_S3_ATOL
    @test err_affine_dvdy <= KRAKEN_E_S3_ATOL

    tauxx = fill(1.3, Nx, Ny)
    tauxy = fill(-0.7, Nx, Ny)
    tauyy = fill(2.1, Nx, Ny)
    fx = zeros(Float64, Nx, Ny)
    fy = zeros(Float64, Nx, Ny)
    Kraken.fvfd_tensor_divergence_block_2d!(
        fx, fy, tauxx, tauxy, tauyy, block, bc_periodic; is_solid,
    )
    max_div_const = maximum((maximum(abs, fx), maximum(abs, fy)))
    @test max_div_const <= KRAKEN_E_S3_ATOL

    record = Kraken.kraken_e_build_cf_face_record_2d(
        Float64;
        coarse_block_id=1,
        fine_block_id=2,
        coarse_index=(3, 5),
        fine_indices=((6, 9), (6, 10)),
        axis=Kraken.KRAKEN_E_CF_FACE_X,
        side=Kraken.KRAKEN_E_CF_FACE_HI,
        coarse_origin=(0.0, 0.0),
        coarse_dx=1.0,
    )
    @test record isa Kraken.CFFaceRecord2D{Float64}
    @test abs(record.coarse_area - sum(record.fine_areas)) <= KRAKEN_E_S3_ATOL
    center_average = (record.fine_centers[1] + record.fine_centers[2]) / 2
    @test maximum(abs, record.coarse_center - center_average) <= KRAKEN_E_S3_ATOL
    normal_norm_err = abs(sqrt(record.normal[1]^2 + record.normal[2]^2) - 1.0)
    @test normal_norm_err <= KRAKEN_E_S3_ATOL
    @test record.normal[1] == 1.0
    @test record.normal[2] == 0.0
    err_weights_sum = abs(sum(record.fine_to_coarse_weights) - 1.0)
    @test err_weights_sum <= KRAKEN_E_S3_ATOL
    @test abs(record.fine_to_coarse_weights[1] - 0.5) <= KRAKEN_E_S3_ATOL
    @test abs(record.fine_to_coarse_weights[2] - 0.5) <= KRAKEN_E_S3_ATOL

    alpha = 0.3
    beta_x = 0.02
    beta_y = -0.05
    phi_c = _s3_phi(alpha, beta_x, beta_y, record.coarse_center)
    phi_f1 = _s3_phi(alpha, beta_x, beta_y, record.fine_centers[1])
    phi_f2 = _s3_phi(alpha, beta_x, beta_y, record.fine_centers[2])
    err_quad_affine = abs(
        record.fine_to_coarse_weights[1] * phi_f1 +
        record.fine_to_coarse_weights[2] * phi_f2 - phi_c
    )
    @test err_quad_affine <= KRAKEN_E_S3_ATOL

    prol_const_f1 = _s3_apply_tangential_stencil(
        record, record.tangential_weights_1, record.tangential_offsets_1,
        0.5, 0.0, 0.0,
    )
    prol_const_f2 = _s3_apply_tangential_stencil(
        record, record.tangential_weights_2, record.tangential_offsets_2,
        0.5, 0.0, 0.0,
    )
    err_prol_const = maximum(abs, (prol_const_f1 - 0.5, prol_const_f2 - 0.5))
    @test err_prol_const <= KRAKEN_E_S3_ATOL

    prol_affine_f1 = _s3_apply_tangential_stencil(
        record, record.tangential_weights_1, record.tangential_offsets_1,
        alpha, beta_x, beta_y,
    )
    prol_affine_f2 = _s3_apply_tangential_stencil(
        record, record.tangential_weights_2, record.tangential_offsets_2,
        alpha, beta_x, beta_y,
    )
    err_prol_affine_f1 = abs(prol_affine_f1 - phi_f1)
    err_prol_affine_f2 = abs(prol_affine_f2 - phi_f2)
    @test err_prol_affine_f1 <= KRAKEN_E_S3_ATOL
    @test err_prol_affine_f2 <= KRAKEN_E_S3_ATOL

    corner_records = (
        Kraken.kraken_e_build_cf_face_record_2d(
            Float64; coarse_block_id=1, fine_block_id=2, coarse_index=(3, 5),
            fine_indices=((6, 9), (6, 10)), axis=Kraken.KRAKEN_E_CF_FACE_X,
            side=Kraken.KRAKEN_E_CF_FACE_LO, coarse_origin=(0.0, 0.0),
            coarse_dx=1.0,
        ),
        Kraken.kraken_e_build_cf_face_record_2d(
            Float64; coarse_block_id=1, fine_block_id=2, coarse_index=(3, 5),
            fine_indices=((7, 9), (7, 10)), axis=Kraken.KRAKEN_E_CF_FACE_X,
            side=Kraken.KRAKEN_E_CF_FACE_HI, coarse_origin=(0.0, 0.0),
            coarse_dx=1.0,
        ),
        Kraken.kraken_e_build_cf_face_record_2d(
            Float64; coarse_block_id=1, fine_block_id=2, coarse_index=(3, 5),
            fine_indices=((6, 9), (7, 9)), axis=Kraken.KRAKEN_E_CF_FACE_Y,
            side=Kraken.KRAKEN_E_CF_FACE_LO, coarse_origin=(0.0, 0.0),
            coarse_dx=1.0,
        ),
        Kraken.kraken_e_build_cf_face_record_2d(
            Float64; coarse_block_id=1, fine_block_id=2, coarse_index=(3, 5),
            fine_indices=((6, 10), (7, 10)), axis=Kraken.KRAKEN_E_CF_FACE_Y,
            side=Kraken.KRAKEN_E_CF_FACE_HI, coarse_origin=(0.0, 0.0),
            coarse_dx=1.0,
        ),
    )
    @test count(record -> record.corner_lo_owned, corner_records[1:2]) == 1
    @test count(record -> record.corner_hi_owned, corner_records[1:2]) == 1
    @test count(record -> record.corner_lo_owned, corner_records[3:4]) == 1
    @test count(record -> record.corner_hi_owned, corner_records[3:4]) == 1

    println()
    println("# Kraken-E S3 canary metrics")
    println("D3 constant gradient max|.|       = $(max_grad_const)")
    println("D3 affine grad err (dudx)         = $(err_affine_dudx)")
    println("D3 affine grad err (dudy)         = $(err_affine_dudy)")
    println("D3 affine grad err (dvdx)         = $(err_affine_dvdx)")
    println("D3 affine grad err (dvdy)         = $(err_affine_dvdy)")
    println("D3 const tensor div max|.|        = $(max_div_const)")
    println("D4 quadrature sum minus 1         = $(err_weights_sum)")
    println("D4 fine-to-coarse exact (affine)  = $(err_quad_affine)")
    println("D4 prolongation const err         = $(err_prol_const)")
    println("D4 prolongation affine err (F1)   = $(err_prol_affine_f1)")
    println("D4 prolongation affine err (F2)   = $(err_prol_affine_f2)")
end
