using Test
using Kraken
using KernelAbstractions

@testset "Conservative tree GPU route pack 2D" begin
    @testset "route pack uses primitive structure-of-arrays" begin
        patch = create_conservative_tree_patch_2d(4:6, 3:5)
        topology = create_conservative_tree_topology_2d(9, 8, patch)
        pack = pack_conservative_tree_gpu_routes_2d(
            topology; cells_per_block=4, T=Float32)

        @test pack.cells_per_block == Int32(4)
        @test eltype(pack.block_level) == UInt8
        @test eltype(pack.block_first_cell) == Int32
        @test eltype(pack.cell_logical_id) == Int32
        @test eltype(pack.route_src) == Int32
        @test eltype(pack.route_dst) == Int32
        @test eltype(pack.route_q) == UInt8
        @test eltype(pack.route_kind) == UInt8
        @test eltype(pack.route_weight) == Float32

        @test length(pack.cell_logical_id) == length(topology.active_cells)
        @test length(pack.route_src) == length(topology.routes)
        @test length(pack.direct_routes) == length(topology.direct_routes)
        @test length(pack.interface_routes) == length(topology.interface_routes)
        @test length(pack.boundary_routes) == length(topology.boundary_routes)
        @test all(pack.block_count .<= pack.cells_per_block)
        @test all(pack.block_count .> 0)

        logical_to_packed = Dict(Int(id) => Int(k)
                                 for (k, id) in pairs(pack.cell_logical_id))
        for (route_index, route) in pairs(topology.routes)
            @test pack.route_src[route_index] == logical_to_packed[route.src]
            @test pack.route_q[route_index] == UInt8(route.q)
            @test pack.route_kind[route_index] == UInt8(route.kind)
            @test pack.route_weight[route_index] == Float32(route.weight)
            if route.kind == ROUTE_BOUNDARY
                @test pack.route_dst[route_index] == 0
            else
                @test pack.route_dst[route_index] == logical_to_packed[route.dst]
            end
        end

        @test_throws ArgumentError pack_conservative_tree_gpu_routes_2d(
            topology; cells_per_block=0)
    end

    @testset "packed route weight sums stay conservative" begin
        patch = create_conservative_tree_patch_2d(3:5, 3:6)
        topology = create_conservative_tree_topology_2d(8, 8, patch)
        pack = pack_conservative_tree_gpu_routes_2d(topology; T=Float64)
        sums = conservative_tree_gpu_route_weight_sums_2d(pack)

        @test length(sums) == 9 * length(topology.active_cells)
        @test all(isapprox(value, 1.0; atol=1e-14, rtol=0)
                  for value in values(sums))
    end

    @testset "CPU replay from GPU pack matches logical route streaming" begin
        nx, ny = 10, 9
        patch_in = create_conservative_tree_patch_2d(4:7, 3:6)
        patch_logical = create_conservative_tree_patch_2d(4:7, 3:6)
        patch_packed = create_conservative_tree_patch_2d(4:7, 3:6)
        topology = create_conservative_tree_topology_2d(nx, ny, patch_in)
        pack = pack_conservative_tree_gpu_routes_2d(
            topology; cells_per_block=5, T=Float64)

        coarse_in = zeros(Float64, nx, ny, 9)
        for q in 1:9, j in axes(coarse_in, 2), i in axes(coarse_in, 1)
            if !(i in patch_in.parent_i_range && j in patch_in.parent_j_range)
                coarse_in[i, j, q] = 0.2 + q / 17 + i / 31 + j / 43 + i * j / 4096
            end
        end
        for q in 1:9, j in axes(patch_in.fine_F, 2), i in axes(patch_in.fine_F, 1)
            patch_in.fine_F[i, j, q] = 0.05 + q / 29 + i / 37 + j / 53 + i * j / 8192
        end

        coarse_logical = similar(coarse_in)
        coarse_packed = similar(coarse_in)
        stream_composite_routes_interior_F_2d!(
            coarse_logical, patch_logical, coarse_in, patch_in, topology)
        stream_conservative_tree_gpu_pack_interior_F_2d!(
            coarse_packed, patch_packed, coarse_in, patch_in, pack)

        @test isapprox(coarse_packed, coarse_logical; atol=1e-14, rtol=0)
        @test isapprox(patch_packed.fine_F, patch_logical.fine_F; atol=1e-14, rtol=0)
        @test isapprox(patch_packed.coarse_shadow_F,
                       patch_logical.coarse_shadow_F; atol=1e-14, rtol=0)

        wrong_patch = create_conservative_tree_patch_2d(4:6, 3:6)
        @test_throws ArgumentError stream_conservative_tree_gpu_pack_interior_F_2d!(
            similar(coarse_in), wrong_patch, coarse_in, patch_in, pack)
    end

    @testset "pull route pack streams multilevel spec without atomics" begin
        blocks = ConservativeTreeRefineBlock2D[
            ConservativeTreeRefineBlock2D("L1", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("L2", 7:10, 5:8; parent="L1"),
        ]
        spec = create_conservative_tree_spec_2d(8, 6, blocks)
        table = create_conservative_tree_route_table_2d(
            spec; periodic_x=true, sampling=:leaf_equivalent)
        Fin = allocate_conservative_tree_F_2d(spec; T=Float64)
        Fref = similar(Fin)
        Fgpu = similar(Fin)

        for cell_id in spec.active_cells, q in 1:9
            cell = spec.cells[cell_id]
            Fin[cell_id, q] =
                0.01 + 0.001 * q + 0.0001 * cell.level +
                0.00001 * cell.i + 0.000001 * cell.j
        end

        stream_conservative_tree_routes_F_2d!(
            Fref, Fin, spec, table; boundary=:periodic_x_wall_y)
        pull = pack_conservative_tree_gpu_pull_routes_2d(
            spec, table; boundary=:periodic_x_wall_y, T=Float64)
        stream_conservative_tree_gpu_pull_routes_F_2d!(Fgpu, Fin, pull)

        @test isapprox(Fgpu, Fref; atol=1e-14, rtol=0)
        @test_throws ArgumentError pack_conservative_tree_gpu_pull_routes_2d(
            spec, table; boundary=:periodic_x_moving_wall_y)
    end

    @testset "active-level Guo collision kernel matches CPU reference" begin
        spec = create_conservative_tree_spec_2d(7, 5, [
            ConservativeTreeRefineBlock2D("L1", 2:5, 2:4),
        ])
        Fref = allocate_conservative_tree_F_2d(spec; T=Float64)
        Fgpu = similar(Fref)
        initialize_conservative_tree_equilibrium_F_2d!(
            Fref, spec; rho=1.0, ux=0.01, uy=-0.002)
        copyto!(Fgpu, Fref)

        level = 1
        omega = 1.15
        Fx = 2e-6
        Fy = -1e-6
        Kraken._collide_Guo_conservative_tree_active_level_F_2d!(
            Fref, spec, level, omega, Fx, Fy)
        cell_pack = pack_conservative_tree_gpu_cells_2d(spec; T=Float64)
        collide_Guo_conservative_tree_gpu_active_level_F_2d!(
            Fgpu, cell_pack, level, omega, Fx, Fy)

        @test isapprox(Fgpu, Fref; atol=1e-14, rtol=0)
    end

    @testset "active-level BGK collision kernel matches CPU reference" begin
        spec = create_conservative_tree_spec_2d(7, 5, [
            ConservativeTreeRefineBlock2D("L1", 2:5, 2:4),
        ])
        Fref = allocate_conservative_tree_F_2d(spec; T=Float64)
        Fgpu = similar(Fref)
        initialize_conservative_tree_equilibrium_F_2d!(
            Fref, spec; rho=1.0, ux=-0.006, uy=0.004)
        copyto!(Fgpu, Fref)

        level = 1
        omega = 1.05
        Kraken._collide_BGK_conservative_tree_active_level_F_2d!(
            Fref, spec, level, omega)
        cell_pack = pack_conservative_tree_gpu_cells_2d(spec; T=Float64)
        collide_BGK_conservative_tree_gpu_active_level_F_2d!(
            Fgpu, cell_pack, level, omega)

        @test isapprox(Fgpu, Fref; atol=1e-14, rtol=0)
    end

    @testset "direct-level pull packs match scheduler route primitive" begin
        blocks = ConservativeTreeRefineBlock2D[
            ConservativeTreeRefineBlock2D("L1", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("L2", 7:10, 5:8; parent="L1"),
        ]
        spec = create_conservative_tree_spec_2d(8, 6, blocks)
        table = create_conservative_tree_route_table_2d(
            spec; periodic_x=true, sampling=:leaf_equivalent)
        Fin = allocate_conservative_tree_F_2d(spec; T=Float64)
        Fref = similar(Fin)
        Fgpu = similar(Fin)

        for (cell_id, cell) in pairs(spec.cells), q in 1:9
            Fin[cell_id, q] =
                0.02 + 0.001 * q + 0.0001 * cell.level +
                0.00001 * cell.i + 0.000001 * cell.j
        end

        for level in 0:spec.max_level
            fill!(Fref, 0)
            fill!(Fgpu, 0)
            Kraken._stream_conservative_tree_direct_level_routes_F_2d!(
                Fref, Fin, spec, table, level, :periodic_x_wall_y;
                periodic_x=true)
            pull = pack_conservative_tree_gpu_direct_level_pull_routes_2d(
                spec, table, level; boundary=:periodic_x_wall_y, T=Float64)
            stream_conservative_tree_gpu_pull_routes_F_2d!(Fgpu, Fin, pull)

            @test isapprox(Fgpu, Fref; atol=1e-14, rtol=0)
        end
    end

    @testset "direct-level advance wrappers match CPU collision plus stream" begin
        spec = create_conservative_tree_spec_2d(8, 6, [
            ConservativeTreeRefineBlock2D("L1", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("L2", 7:10, 5:8; parent="L1"),
        ])
        table = create_conservative_tree_route_table_2d(
            spec; periodic_x=true, sampling=:leaf_equivalent)
        cell_pack = pack_conservative_tree_gpu_cells_2d(spec; T=Float64)
        level = 1
        pull = pack_conservative_tree_gpu_direct_level_pull_routes_2d(
            spec, table, level; boundary=:periodic_x_wall_y, T=Float64)

        Fin_ref = allocate_conservative_tree_F_2d(spec; T=Float64)
        initialize_conservative_tree_equilibrium_F_2d!(
            Fin_ref, spec; rho=1.0, ux=0.002, uy=-0.001)
        Fin_gpu = copy(Fin_ref)
        Fout_ref = similar(Fin_ref)
        Fout_gpu = similar(Fin_ref)
        fill!(Fout_ref, 0)
        Kraken._collide_Guo_conservative_tree_active_level_F_2d!(
            Fin_ref, spec, level, 1.12, 2e-6, -1e-6)
        Kraken._stream_conservative_tree_direct_level_routes_F_2d!(
            Fout_ref, Fin_ref, spec, table, level, :periodic_x_wall_y;
            periodic_x=true)
        advance_conservative_tree_gpu_direct_level_Guo_F_2d!(
            Fout_gpu, Fin_gpu, pull, cell_pack, level, 1.12, 2e-6, -1e-6)
        @test isapprox(Fout_gpu, Fout_ref; atol=1e-14, rtol=0)

        initialize_conservative_tree_equilibrium_F_2d!(
            Fin_ref, spec; rho=1.0, ux=-0.003, uy=0.001)
        copyto!(Fin_gpu, Fin_ref)
        fill!(Fout_ref, 0)
        Kraken._collide_BGK_conservative_tree_active_level_F_2d!(
            Fin_ref, spec, level, 1.08)
        Kraken._stream_conservative_tree_direct_level_routes_F_2d!(
            Fout_ref, Fin_ref, spec, table, level, :periodic_x_wall_y;
            periodic_x=true)
        advance_conservative_tree_gpu_direct_level_BGK_F_2d!(
            Fout_gpu, Fin_gpu, pull, cell_pack, level, 1.08)
        @test isapprox(Fout_gpu, Fout_ref; atol=1e-14, rtol=0)
    end

    @testset "coarse-to-fine ledger apply kernel matches CPU ledger apply" begin
        spec = create_conservative_tree_spec_2d(8, 6, [
            ConservativeTreeRefineBlock2D("L1", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("L2", 7:10, 5:8; parent="L1"),
        ])
        schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(
            spec.max_level)
        bank = Kraken.create_conservative_tree_subcycle_spatial_ledger_bank_2d(
            spec; schedule=schedule, T=Float64)
        parent_level = 1
        pair = bank.ledger_pairs[parent_level + 1]
        for slot in axes(pair.coarse_to_fine, 5),
            substep in axes(pair.coarse_to_fine, 4),
            q in 1:9, iy in 1:2, ix in 1:2
            pair.coarse_to_fine[ix, iy, q, substep, slot] =
                0.001 * ix + 0.002 * iy + 0.0001 * q +
                0.00001 * substep + 0.000001 * slot
        end
        Fref = allocate_conservative_tree_F_2d(spec; T=Float64)
        Fgpu = similar(Fref)
        fill!(Fref, 0.0)
        fill!(Fgpu, 0.0)

        substep = 2
        Kraken.conservative_tree_subcycle_apply_coarse_to_fine_pair_F_2d!(
            Fref, bank, parent_level, substep)
        pc_pack = pack_conservative_tree_gpu_parent_children_2d(
            spec, parent_level)
        apply_conservative_tree_gpu_coarse_to_fine_pair_F_2d!(
            Fgpu, pair.coarse_to_fine, pc_pack, substep)

        @test isapprox(Fgpu, Fref; atol=1e-14, rtol=0)
    end

    @testset "coarse-to-fine deposit pack matches CPU sync-down" begin
        spec = create_conservative_tree_spec_2d(8, 6, [
            ConservativeTreeRefineBlock2D("L1", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("L2", 7:10, 5:8; parent="L1"),
        ])
        table = create_conservative_tree_route_table_2d(
            spec; periodic_x=true, sampling=:leaf_equivalent)
        schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(
            spec.max_level)
        bank = Kraken.create_conservative_tree_subcycle_spatial_ledger_bank_2d(
            spec; schedule=schedule, T=Float64)
        F = allocate_conservative_tree_F_2d(spec; T=Float64)
        for (cell_id, cell) in pairs(spec.cells), q in 1:9
            F[cell_id, q] =
                0.04 + 0.001 * q + 0.0001 * cell.level +
                0.00001 * cell.i + 0.000001 * cell.j
        end

        for parent_level in 0:(spec.max_level - 1)
            pair = bank.ledger_pairs[parent_level + 1]
            fill!(pair.coarse_to_fine, 0)
            event = first(e for e in schedule.events
                          if e.phase == :sync_down &&
                             e.src_level == parent_level)
            Kraken.conservative_tree_subcycle_sync_down_routes_F_2d!(
                bank, event, F, table; interface_time_scaling=:leaf_equivalent)
            ref = copy(pair.coarse_to_fine)
            fill!(pair.coarse_to_fine, 0)

            pc_pack = pack_conservative_tree_gpu_parent_children_2d(
                spec, parent_level)
            deposit_pack = pack_conservative_tree_gpu_coarse_to_fine_deposit_2d(
                spec, table, pc_pack, parent_level; ratio=schedule.ratio,
                interface_time_scaling=:leaf_equivalent, T=Float64)
            deposit_conservative_tree_gpu_coarse_to_fine_routes_F_2d!(
                pair.coarse_to_fine, F, deposit_pack)

            @test isapprox(pair.coarse_to_fine, ref; atol=1e-14, rtol=0)
        end
    end

    @testset "fine-to-coarse deposit pack matches CPU route cache" begin
        spec = create_conservative_tree_spec_2d(8, 6, [
            ConservativeTreeRefineBlock2D("L1", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("L2", 7:10, 5:8; parent="L1"),
        ])
        table = create_conservative_tree_route_table_2d(
            spec; periodic_x=true, sampling=:leaf_equivalent)
        schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(
            spec.max_level)
        F = allocate_conservative_tree_F_2d(spec; T=Float64)
        for (cell_id, cell) in pairs(spec.cells), q in 1:9
            F[cell_id, q] =
                0.05 + 0.001 * q + 0.0001 * cell.level +
                0.00001 * cell.i + 0.000001 * cell.j
        end

        for child_level in 1:spec.max_level
            parent_level = child_level - 1
            bank_ref = Kraken.create_conservative_tree_subcycle_spatial_ledger_bank_2d(
                spec; schedule=schedule, T=Float64)
            bank_pack = Kraken.create_conservative_tree_subcycle_spatial_ledger_bank_2d(
                spec; schedule=schedule, T=Float64)
            Kraken.prepare_conservative_tree_subcycle_route_packet_cache_2d!(
                bank_ref, table)
            Kraken.prepare_conservative_tree_subcycle_route_packet_cache_2d!(
                bank_pack, table)
            Kraken.reset_conservative_tree_subcycle_spatial_pair_2d!(
                bank_ref, parent_level)
            Kraken.reset_conservative_tree_subcycle_spatial_pair_2d!(
                bank_pack, parent_level)

            events = [event for event in schedule.events
                      if event.phase == :advance &&
                         event.src_level == child_level]
            for event in events[1:schedule.ratio]
                Kraken.conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
                    bank_ref, event, F, table;
                    interface_time_scaling=:leaf_equivalent)
                Kraken.conservative_tree_subcycle_accumulate_inactive_parent_routes_F_2d!(
                    bank_ref, event, F;
                    interface_time_scaling=:leaf_equivalent)
            end
            cache_ref = bank_ref.route_packet_caches[parent_level + 1]

            pack = pack_conservative_tree_gpu_fine_to_coarse_deposit_2d(
                spec, table, bank_pack, child_level;
                interface_time_scaling=:leaf_equivalent, T=Float64)
            cache_gpu = zeros(Float64, length(cache_ref.packets))
            deposit_conservative_tree_gpu_fine_to_coarse_cache_F_2d!(
                cache_gpu, F, pack)

            @test isapprox(cache_gpu, cache_ref.packets; atol=1e-14, rtol=0)

            Fref = allocate_conservative_tree_F_2d(spec; T=Float64)
            Fgpu = similar(Fref)
            fill!(Fref, 0)
            fill!(Fgpu, 0)
            sync_event = first(event for event in schedule.events
                               if event.phase == :sync_up &&
                                  event.src_level == child_level &&
                                  event.dst_level == parent_level)
            Kraken.conservative_tree_subcycle_apply_sync_up_F_2d!(
                Fref, bank_ref, sync_event; boundary=:periodic_x_wall_y)
            apply_conservative_tree_gpu_sync_up_cache_F_2d!(
                Fgpu, cache_gpu, pack)
            @test isapprox(Fgpu, Fref; atol=1e-14, rtol=0)
        end
    end

    @testset "level row kernels match scheduler row utilities" begin
        spec = create_conservative_tree_spec_2d(8, 6, [
            ConservativeTreeRefineBlock2D("L1", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("L2", 7:10, 5:8; parent="L1"),
        ])
        Fsrc = allocate_conservative_tree_F_2d(spec; T=Float64)
        Fcopy_ref = fill(-1.0, size(Fsrc))
        Fcopy_gpu = similar(Fcopy_ref)
        Fadd_ref = fill(0.25, size(Fsrc))
        Fadd_gpu = copy(Fadd_ref)
        Fpending_ref = allocate_conservative_tree_F_2d(spec; T=Float64)
        Fpending_gpu = similar(Fpending_ref)
        for (cell_id, cell) in pairs(spec.cells), q in 1:9
            Fsrc[cell_id, q] =
                0.03 + 0.001 * q + 0.0001 * cell.level +
                0.00001 * cell.i + 0.000001 * cell.j
            Fpending_ref[cell_id, q] = 0.5 * Fsrc[cell_id, q]
        end
        copyto!(Fcopy_gpu, Fcopy_ref)
        copyto!(Fpending_gpu, Fpending_ref)
        pack = pack_conservative_tree_gpu_cells_2d(spec; T=Float64)

        for level in 0:spec.max_level
            Kraken._copy_conservative_tree_level_rows_2d!(
                Fcopy_ref, Fsrc, spec, level)
            copy_conservative_tree_gpu_level_rows_2d!(
                Fcopy_gpu, Fsrc, pack, level)
            @test isapprox(Fcopy_gpu, Fcopy_ref; atol=1e-14, rtol=0)

            Kraken._add_and_clear_conservative_tree_level_rows_2d!(
                Fadd_ref, Fpending_ref, spec, level)
            add_and_clear_conservative_tree_gpu_level_rows_2d!(
                Fadd_gpu, Fpending_gpu, pack, level)
            @test isapprox(Fadd_gpu, Fadd_ref; atol=1e-14, rtol=0)
            @test isapprox(Fpending_gpu, Fpending_ref; atol=1e-14, rtol=0)
        end

        zero_conservative_tree_gpu_level_rows_2d!(Fcopy_gpu, pack, 1;
                                                  active_only=true)
        for cell_id in spec.active_cells
            spec.cells[cell_id].level == 1 || continue
            @test all(Fcopy_gpu[cell_id, :] .== 0.0)
        end
    end

    if get(ENV, "KRAKEN_TEST_METAL", "0") == "1"
        @testset "Metal smoke for pull stream and active-level collision" begin
            @eval using Metal
            if Metal.functional()
                backend = Metal.MetalBackend()
                spec = create_conservative_tree_spec_2d(5, 4, [
                    ConservativeTreeRefineBlock2D("L1", 2:4, 2:3),
                ])
                table = create_conservative_tree_route_table_2d(
                    spec; periodic_x=true, sampling=:leaf_equivalent)
                Fin = allocate_conservative_tree_F_2d(spec; T=Float32)
                for cell_id in spec.active_cells, q in 1:9
                    Fin[cell_id, q] = Float32(0.02 + 0.001 * q + 0.0001 * cell_id)
                end
                Fref = similar(Fin)
                stream_conservative_tree_routes_F_2d!(
                    Fref, Fin, spec, table; boundary=:periodic_x_wall_y)

                Fd = KernelAbstractions.allocate(backend, Float32, size(Fin)...)
                Foutd = similar(Fd)
                copyto!(Fd, Fin)
                pull = pack_conservative_tree_gpu_pull_routes_2d(
                    spec, table; boundary=:periodic_x_wall_y, T=Float32)
                pull_d = transfer_conservative_tree_gpu_pull_pack_2d(
                    pull, backend)
                stream_conservative_tree_gpu_pull_routes_F_2d!(
                    Foutd, Fd, pull_d)
                @test isapprox(Array(Foutd), Fref; atol=2e-6, rtol=2e-6)

                level_pull = pack_conservative_tree_gpu_direct_level_pull_routes_2d(
                    spec, table, 1; boundary=:periodic_x_wall_y, T=Float32)
                level_pull_d = transfer_conservative_tree_gpu_pull_pack_2d(
                    level_pull, backend)
                fill!(Fref, 0)
                Kraken._stream_conservative_tree_direct_level_routes_F_2d!(
                    Fref, Fin, spec, table, 1, :periodic_x_wall_y;
                    periodic_x=true)
                stream_conservative_tree_gpu_pull_routes_F_2d!(
                    Foutd, Fd, level_pull_d)
                @test isapprox(Array(Foutd), Fref; atol=2e-6, rtol=2e-6)

                cell_pack_d = transfer_conservative_tree_gpu_cell_pack_2d(
                    pack_conservative_tree_gpu_cells_2d(spec; T=Float32),
                    backend)
                Fcopyd = similar(Fd)
                fill!(Fcopyd, Float32(-1))
                copy_conservative_tree_gpu_level_rows_2d!(
                    Fcopyd, Fd, cell_pack_d, 1)
                Fcopy = fill(Float32(-1), size(Fin))
                Kraken._copy_conservative_tree_level_rows_2d!(
                    Fcopy, Fin, spec, 1)
                @test isapprox(Array(Fcopyd), Fcopy; atol=2e-6, rtol=2e-6)

                collide_Guo_conservative_tree_gpu_active_level_F_2d!(
                    Fd, cell_pack_d, 1, Float32(1.1), Float32(1e-6),
                    Float32(0))
                @test all(isfinite, Array(Fd))

                advance_conservative_tree_gpu_direct_level_BGK_F_2d!(
                    Foutd, Fd, level_pull_d, cell_pack_d, 1, Float32(1.05))
                @test all(isfinite, Array(Foutd))

                schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(
                    spec.max_level)
                bank = Kraken.create_conservative_tree_subcycle_spatial_ledger_bank_2d(
                    spec; schedule=schedule, T=Float32)
                pair = bank.ledger_pairs[1]
                fill!(pair.coarse_to_fine, Float32(1e-4))
                ledger_d = KernelAbstractions.allocate(
                    backend, Float32, size(pair.coarse_to_fine)...)
                copyto!(ledger_d, pair.coarse_to_fine)
                pc_pack_d = transfer_conservative_tree_gpu_parent_child_pack_2d(
                    pack_conservative_tree_gpu_parent_children_2d(spec, 0),
                    backend)
                fill!(Foutd, Float32(0))
                apply_conservative_tree_gpu_coarse_to_fine_pair_F_2d!(
                    Foutd, ledger_d, pc_pack_d, 1)
                @test all(isfinite, Array(Foutd))

                copyto!(Fd, Fin)
                fill!(ledger_d, Float32(0))
                dep_pack_d = transfer_conservative_tree_gpu_c2f_deposit_pack_2d(
                    pack_conservative_tree_gpu_coarse_to_fine_deposit_2d(
                        spec, table,
                        pack_conservative_tree_gpu_parent_children_2d(spec, 0),
                        0; ratio=schedule.ratio, T=Float32),
                    backend)
                deposit_conservative_tree_gpu_coarse_to_fine_routes_F_2d!(
                    ledger_d, Fd, dep_pack_d)
                @test all(isfinite, Array(ledger_d))

                bank_ref = Kraken.create_conservative_tree_subcycle_spatial_ledger_bank_2d(
                    spec; schedule=schedule, T=Float32)
                Kraken.prepare_conservative_tree_subcycle_route_packet_cache_2d!(
                    bank_ref, table)
                Kraken.reset_conservative_tree_subcycle_spatial_pair_2d!(
                    bank_ref, 0)
                for event in schedule.events
                    event.phase == :advance && event.src_level == 1 || continue
                    Kraken.conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
                        bank_ref, event, Fin, table;
                        interface_time_scaling=:leaf_equivalent)
                    Kraken.conservative_tree_subcycle_accumulate_inactive_parent_routes_F_2d!(
                        bank_ref, event, Fin;
                        interface_time_scaling=:leaf_equivalent)
                end
                f2c_pack = pack_conservative_tree_gpu_fine_to_coarse_deposit_2d(
                    spec, table,
                    Kraken.create_conservative_tree_subcycle_spatial_ledger_bank_2d(
                        spec; schedule=schedule, T=Float32),
                    1; T=Float32)
                f2c_pack_d = transfer_conservative_tree_gpu_f2c_deposit_pack_2d(
                    f2c_pack, backend)
                cache_d = KernelAbstractions.allocate(
                    backend, Float32, length(bank_ref.route_packet_caches[1].packets))
                fill!(cache_d, Float32(0))
                copyto!(Fd, Fin)
                deposit_conservative_tree_gpu_fine_to_coarse_cache_F_2d!(
                    cache_d, Fd, f2c_pack_d)
                @test isapprox(Array(cache_d),
                               bank_ref.route_packet_caches[1].packets;
                               atol=2e-6, rtol=2e-6)
                fill!(Foutd, Float32(0))
                apply_conservative_tree_gpu_sync_up_cache_F_2d!(
                    Foutd, cache_d, f2c_pack_d)
                @test all(isfinite, Array(Foutd))
            end
        end
    end
end
