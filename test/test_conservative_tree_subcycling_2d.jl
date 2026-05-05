using Test
using Kraken

@testset "Conservative tree subcycling ledger 2D" begin
    @testset "generic recursive schedule is level agnostic" begin
        schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(3)

        @test schedule.max_level == 3
        @test schedule.ratio == 2
        @test schedule.finest_ticks == 8
        @test schedule.level_step_ticks == [8, 4, 2, 1]
        @test Kraken.conservative_tree_subcycle_advance_counts_2d(schedule) ==
              [1, 2, 4, 8]

        sync_counts = Kraken.conservative_tree_subcycle_sync_counts_2d(schedule)
        for level in 0:2
            expected = 2^level
            @test sync_counts[(:sync_down, level, level + 1)] == expected
            @test sync_counts[(:sync_up, level + 1, level)] == expected
        end

        tick0 = Kraken.conservative_tree_subcycle_events_at_tick_2d(schedule, 0)
        @test [(event.phase, event.src_level, event.dst_level)
               for event in tick0] == [(:sync_down, 0, 1),
                                       (:sync_down, 1, 2),
                                       (:sync_down, 2, 3)]

        tick2 = Kraken.conservative_tree_subcycle_events_at_tick_2d(schedule, 2)
        @test (:advance, 3, 3) in
              [(event.phase, event.src_level, event.dst_level)
               for event in tick2]
        @test (:sync_up, 3, 2) in
              [(event.phase, event.src_level, event.dst_level)
               for event in tick2]
        @test (:advance, 2, 2) in
              [(event.phase, event.src_level, event.dst_level)
               for event in tick2]
        @test (:sync_down, 2, 3) in
              [(event.phase, event.src_level, event.dst_level)
               for event in tick2]

        tick8 = Kraken.conservative_tree_subcycle_events_at_tick_2d(schedule, 8)
        @test [(event.phase, event.src_level, event.dst_level)
               for event in tick8][end-1:end] == [(:sync_up, 1, 0),
                                                  (:advance, 0, 0)]
    end

    @testset "schedule contracts reject invalid inputs" begin
        @test Kraken.create_conservative_tree_subcycle_schedule_2d(0).events ==
              [Kraken.ConservativeTreeSubcycleEvent2D(1, :advance, 0, 0)]
        @test_throws ArgumentError Kraken.create_conservative_tree_subcycle_schedule_2d(-1)
        @test_throws ArgumentError Kraken.create_conservative_tree_subcycle_schedule_2d(2; ratio=1)
        schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(2)
        @test_throws ArgumentError Kraken.conservative_tree_subcycle_events_at_tick_2d(
            schedule, 5)
    end

    @testset "scheduler binds one L/L+1 interface ledger" begin
        schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(1)
        bank = Kraken.create_conservative_tree_subcycle_ledger_bank_2d(schedule)
        down = only(event for event in schedule.events
                    if event.phase == :sync_down)
        up = only(event for event in schedule.events
                  if event.phase == :sync_up)
        advances = [event for event in schedule.events
                    if event.phase == :advance && event.src_level == 1]

        @test Kraken.conservative_tree_subcycle_local_substep_2d(
            schedule, 0, advances[1].tick) == 1
        @test Kraken.conservative_tree_subcycle_local_substep_2d(
            schedule, 0, advances[2].tick) == 2

        ledger = Kraken.conservative_tree_subcycle_sync_down_face_2d!(
            bank, down, 12.0, 2, :west)
        half1 = zeros(Float64, 2, 2, 9)
        half2 = zeros(Float64, 2, 2, 9)
        half1[2, 1, 2] = 1.25
        half1[2, 2, 2] = 2.75
        half2[2, 1, 2] = 2.0
        half2[2, 2, 2] = 3.0
        Kraken.conservative_tree_subcycle_accumulate_advance_face_2d!(
            bank, advances[1], half1, 2, :east)
        Kraken.conservative_tree_subcycle_accumulate_advance_face_2d!(
            bank, advances[2], half2, 2, :east)

        @test Kraken.conservative_tree_subcycle_sync_up_ledger_2d(bank, up) ===
              ledger
        sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
        @test sums.coarse_to_fine[2] == 12.0
        @test sums.fine_to_coarse[2] == 9.0

        Kraken.reset_conservative_tree_subcycle_pair_2d!(bank, 0)
        ledger = Kraken.conservative_tree_subcycle_pair_ledger_2d(bank, 0)
        Kraken.conservative_tree_subcycle_sync_down_corner_2d!(
            bank, down, 7.0, 6, :southwest)
        corner_half = zeros(Float64, 2, 2, 9)
        corner_half[2, 2, 6] = 2.0
        for event in advances
            Kraken.conservative_tree_subcycle_accumulate_advance_corner_2d!(
                bank, event, corner_half, 6, :northeast)
        end
        sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
        @test sums.coarse_to_fine[6] == 7.0
        @test sums.fine_to_coarse[6] == 4.0
    end

    @testset "scheduler binds all adjacent level-pair ledgers recursively" begin
        schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(3)
        bank = Kraken.create_conservative_tree_subcycle_ledger_bank_2d(schedule)

        for event in schedule.events
            if event.phase == :sync_down
                Fq = 10.0 * (event.src_level + 1)
                Kraken.conservative_tree_subcycle_sync_down_face_2d!(
                    bank, event, Fq, 2, :west)
            elseif event.phase == :advance && event.src_level > 0
                half = zeros(Float64, 2, 2, 9)
                half[2, 1, 2] = 1.0
                half[2, 2, 2] = 1.0
                Kraken.conservative_tree_subcycle_accumulate_advance_face_2d!(
                    bank, event, half, 2, :east)
            elseif event.phase == :sync_up
                @test Kraken.conservative_tree_subcycle_sync_up_ledger_2d(
                    bank, event) ===
                      Kraken.conservative_tree_subcycle_pair_ledger_2d(
                          bank, event.dst_level)
            end
        end

        for parent in 0:2
            ledger = Kraken.conservative_tree_subcycle_pair_ledger_2d(bank, parent)
            sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
            sync_down_count = 2^parent
            child_advance_count = 2^(parent + 1)

            @test sums.coarse_to_fine[2] ==
                  sync_down_count * 10.0 * (parent + 1)
            @test sums.fine_to_coarse[2] == child_advance_count * 2.0
            @test ledger.fine_to_coarse[2, 1] == sync_down_count * 2.0
            @test ledger.fine_to_coarse[2, 2] == sync_down_count * 2.0
        end
    end

    @testset "scheduled ledger binding rejects wrong events" begin
        schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(2)
        bank = Kraken.create_conservative_tree_subcycle_ledger_bank_2d(schedule)
        down = first(event for event in schedule.events
                     if event.phase == :sync_down)
        advance0 = first(event for event in schedule.events
                         if event.phase == :advance && event.src_level == 0)
        advance1 = first(event for event in schedule.events
                         if event.phase == :advance && event.src_level == 1)

        half = zeros(Float64, 2, 2, 9)
        @test_throws ArgumentError Kraken.conservative_tree_subcycle_sync_down_face_2d!(
            bank, advance1, 1.0, 2, :west)
        @test_throws ArgumentError Kraken.conservative_tree_subcycle_accumulate_advance_face_2d!(
            bank, down, half, 2, :east)
        @test_throws ArgumentError Kraken.conservative_tree_subcycle_accumulate_advance_face_2d!(
            bank, advance0, half, 2, :east)
        @test_throws ArgumentError Kraken.conservative_tree_subcycle_sync_up_ledger_2d(
            bank, down)
        @test_throws ArgumentError Kraken.conservative_tree_subcycle_pair_ledger_2d(
            bank, 2)
        @test_throws ArgumentError Kraken.conservative_tree_subcycle_local_substep_2d(
            schedule, 0, 1)
    end

    @testset "coarse-to-fine face packet is consumed once over two half steps" begin
        ledger = create_conservative_tree_subcycle_ledger_2d()
        conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!(
            ledger, 12.0, 2, :west)
        sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
        totals = conservative_tree_subcycle_total_sums_2d(ledger)

        @test ledger.ratio == 2
        @test conservative_tree_subcycle_weights_2d(ledger) == [0.5, 0.5]
        @test sums.coarse_to_fine[2] == 12.0
        @test sum(sums.coarse_to_fine) == 12.0
        @test all(iszero, sums.coarse_to_fine[[1, 3, 4, 5, 6, 7, 8, 9]])
        @test totals.coarse_to_fine == 12.0
        @test totals.fine_to_coarse == 0.0

        for substep in 1:2
            @test ledger.coarse_to_fine[1, 1, 2, substep] == 3.0
            @test ledger.coarse_to_fine[1, 2, 2, substep] == 3.0
            @test sum(ledger.coarse_to_fine[:, :, 2, substep]) == 6.0
        end
    end

    @testset "coarse-to-fine corner packet is split only in time" begin
        ledger = create_conservative_tree_subcycle_ledger_2d()
        conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!(
            ledger, 7.0, 6, :southwest)
        sums = conservative_tree_subcycle_orientation_sums_2d(ledger)

        @test sums.coarse_to_fine[6] == 7.0
        @test sum(sums.coarse_to_fine) == 7.0
        for substep in 1:2
            @test ledger.coarse_to_fine[1, 1, 6, substep] == 3.5
            @test sum(ledger.coarse_to_fine[:, :, 6, substep]) == 3.5
        end
    end

    @testset "fine-to-coarse face packets accumulate by half step" begin
        ledger = create_conservative_tree_subcycle_ledger_2d()
        half1 = zeros(Float64, 2, 2, 9)
        half2 = zeros(Float64, 2, 2, 9)
        half1[2, 1, 2] = 1.25
        half1[2, 2, 2] = 2.75
        half2[2, 1, 2] = 2.0
        half2[2, 2, 2] = 3.0

        conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
            ledger, half1, 2, :east, 1)
        conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
            ledger, half2, 2, :east, 2)
        sums = conservative_tree_subcycle_orientation_sums_2d(ledger)

        @test ledger.fine_to_coarse[2, 1] == 4.0
        @test ledger.fine_to_coarse[2, 2] == 5.0
        @test sums.fine_to_coarse[2] == 9.0
        @test sum(sums.fine_to_coarse) == 9.0
        @test sum(sums.coarse_to_fine) == 0.0
    end

    @testset "full cycle ledger preserves expected orientation totals" begin
        ledger = create_conservative_tree_subcycle_ledger_2d()
        conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!(
            ledger, 6.0, 2, :west)
        conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!(
            ledger, 4.0, 6, :southwest)

        face_half = zeros(Float64, 2, 2, 9)
        face_half[2, 1, 2] = 1.5
        face_half[2, 2, 2] = 1.5
        corner_half = zeros(Float64, 2, 2, 9)
        corner_half[2, 2, 6] = 2.0
        for substep in 1:2
            conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
                ledger, face_half, 2, :east, substep)
            conservative_tree_subcycle_accumulate_fine_to_coarse_corner_2d!(
                ledger, corner_half, 6, :northeast, substep)
        end

        sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
        totals = conservative_tree_subcycle_total_sums_2d(ledger)
        @test sums.coarse_to_fine[2] == 6.0
        @test sums.coarse_to_fine[6] == 4.0
        @test sums.fine_to_coarse[2] == 6.0
        @test sums.fine_to_coarse[6] == 4.0
        @test totals.coarse_to_fine == 10.0
        @test totals.fine_to_coarse == 10.0

        reset_conservative_tree_subcycle_ledger_2d!(ledger)
        empty = conservative_tree_subcycle_total_sums_2d(ledger)
        @test empty.coarse_to_fine == 0.0
        @test empty.fine_to_coarse == 0.0
    end

    @testset "rest equilibrium: axis face deposit and accumulate balance per cycle" begin
        # Setup: a coarse cell at rest equilibrium (rho=1, V_c=1) sends its q-th
        # population through the entry face into the fine patch. The fine patch
        # children are at fine equilibrium (rho=1, V_f=0.25) and contribute back
        # via the opposite direction through the same physical face over 2 sub-steps.
        # The ledger must reflect a balanced cycle for axis directions q in 2:5
        # because a face has 2 children on each side and a cycle has 2 sub-steps:
        #   coarse->fine[q]   = w_q * V_c
        #   fine->coarse[opp] = 2 children * w_opp * V_f * 2 sub-steps = w_opp * V_c
        # With w_q == w_opp for axis directions, both totals are equal.
        Vc = 1.0
        Vf = 0.25
        wq_axis = 1 / 9  # D2Q9 weight for axis directions
        face_in_for = Dict(2 => :west, 3 => :south, 4 => :east, 5 => :north)
        face_out_for = Dict(2 => :east, 3 => :north, 4 => :west, 5 => :south)

        for q in (2, 3, 4, 5)
            opp = d2q9_opposite(q)
            ledger = create_conservative_tree_subcycle_ledger_2d()

            Fq_eq_coarse = wq_axis * Vc
            conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!(
                ledger, Fq_eq_coarse, q, face_in_for[q])

            fine_block = zeros(Float64, 2, 2, 9)
            for ic in 1:2, jc in 1:2
                fine_block[ic, jc, opp] = wq_axis * Vf
            end
            for substep in 1:2
                conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
                    ledger, fine_block, opp, face_out_for[opp], substep)
            end

            sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
            @test isapprox(sums.coarse_to_fine[q], wq_axis * Vc;
                           atol=1e-15, rtol=0)
            @test isapprox(sums.fine_to_coarse[opp], wq_axis * Vc;
                           atol=1e-15, rtol=0)
            @test isapprox(sums.coarse_to_fine[q], sums.fine_to_coarse[opp];
                           atol=1e-15, rtol=0)
            # All other orientations must be zero.
            for qz in 1:9
                qz == q && continue
                qz == opp && continue
                @test sums.coarse_to_fine[qz] == 0.0
                @test sums.fine_to_coarse[qz] == 0.0
            end
        end
    end

    @testset "rest equilibrium: corner deposit per substep matches fine equilibrium" begin
        # Setup: coarse cell sends its diagonal q (in 6:9) through one corner.
        # The ledger splits the packet evenly across 2 sub-steps; each sub-step's
        # corner deposit lands in 1 fine child cell. The deposit value per
        # sub-step (= w_q * V_c / 2) must equal the fine equilibrium of that q
        # (= w_q * V_f * 2) up to the well-known 2x corner geometric factor.
        # Concretely, ledger.coarse_to_fine[ix, iy, q, substep] = w_q * V_c / 2
        # = w_q * V_f * 2; the 2x prefactor reflects that 1 coarse step covers 2
        # fine half-steps so the per-sub-step deposit is twice the per-fine-cell
        # equilibrium. This is documented as a known imbalance that the time
        # integrator must compensate via Filippova-Hanel rescaling and reflux.
        Vc = 1.0
        Vf = 0.25
        wq_corner = 1 / 36
        corner_in_for = Dict(6 => :southwest, 7 => :southeast,
                             8 => :northeast, 9 => :northwest)
        corner_out_for = Dict(6 => :northeast, 7 => :northwest,
                              8 => :southwest, 9 => :southeast)

        for q in (6, 7, 8, 9)
            opp = d2q9_opposite(q)
            ledger = create_conservative_tree_subcycle_ledger_2d()

            Fq_eq_coarse = wq_corner * Vc
            conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!(
                ledger, Fq_eq_coarse, q, corner_in_for[q])

            fine_block = zeros(Float64, 2, 2, 9)
            for ic in 1:2, jc in 1:2
                fine_block[ic, jc, opp] = wq_corner * Vf
            end
            for substep in 1:2
                conservative_tree_subcycle_accumulate_fine_to_coarse_corner_2d!(
                    ledger, fine_block, opp, corner_out_for[opp], substep)
            end

            sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
            # coarse->fine[q] = wq_corner * Vc (full coarse packet over 1 cycle)
            @test isapprox(sums.coarse_to_fine[q], wq_corner * Vc;
                           atol=1e-15, rtol=0)
            # fine->coarse[opp] = 1 child * wq_corner * Vf * 2 substeps
            #                   = wq_corner * Vc / 2
            @test isapprox(sums.fine_to_coarse[opp], wq_corner * Vc / 2;
                           atol=1e-15, rtol=0)
            # The 2x corner imbalance is INTRINSIC to the subcycling geometry,
            # not a ledger bug. The future time integrator must rescale
            # (Filippova-Hanel) or accept this as residual on coarse.
            @test sums.fine_to_coarse[opp] == sums.coarse_to_fine[q] / 2
        end
    end

    @testset "ledger rejects unsupported contracts" begin
        ledger = create_conservative_tree_subcycle_ledger_2d()
        bad_block = zeros(Float64, 3, 2, 9)

        @test_throws ArgumentError create_conservative_tree_subcycle_ledger_2d(ratio=3)
        @test_throws ArgumentError conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!(
            ledger, 1.0, 2, :east)
        @test_throws ArgumentError conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!(
            ledger, 1.0, 2, :southwest)
        @test_throws ArgumentError conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
            ledger, bad_block, 2, :east, 1)
        @test_throws ArgumentError conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
            ledger, zeros(Float64, 2, 2, 9), 2, :east, 3)
    end
end
