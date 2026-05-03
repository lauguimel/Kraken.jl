using Test
using Kraken

@testset "Conservative tree subcycling ledger 2D" begin
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
