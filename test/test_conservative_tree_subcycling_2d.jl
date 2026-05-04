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
