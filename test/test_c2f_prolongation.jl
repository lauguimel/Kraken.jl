using Test
using Kraken

function _c2f_test_spec_2d()
    block = Kraken.ConservativeTreeRefineBlock2D("center", 2:2, 2:2)
    return Kraken.create_conservative_tree_spec_2d(
        4, 4, [block]; balance=true, coarse_volume=1.0)
end

function _c2f_test_split_route(spec)
    table = Kraken.create_conservative_tree_route_table_2d(
        spec; periodic_x=false, sampling=:leaf_equivalent)
    route_id = findfirst(
        r -> r.kind in (Kraken.SPLIT_FACE, Kraken.SPLIT_CORNER),
        table.routes)
    route_id === nothing && error("test fixture did not create a split route")
    return table.routes[route_id]
end

@testset "c2f prolongation 2D" begin
    spec = _c2f_test_spec_2d()
    route = _c2f_test_split_route(spec)
    F = fill(0.0, length(spec.cells), 9)
    F[:, route.q] .= 12.0
    ctx = Kraken.C2FProlongationContext2D(
        spec, nothing, false, 1.0, :leaf_equivalent)

    @test ctx.spec === spec
    @test ctx.table === nothing
    @test ctx.periodic_x == false
    @test ctx.alpha == 1.0
    @test ctx.interface_time_scaling == :leaf_equivalent

    @test Kraken.c2f_route_packet_2d(:flat, F, route, ctx) ==
        route.weight * F[route.src, route.q]

    for kind in (Kraken.SPLIT_FACE, Kraken.SPLIT_CORNER,
                 Kraken.COALESCE_FACE, Kraken.COALESCE_CORNER)
        kind_route = Kraken.ConservativeTreeRoute2D(
            route.src, route.dst, route.q, route.weight, kind)
        @test Kraken.c2f_route_packet_2d(:flat, F, kind_route, ctx) ==
            route.weight * F[route.src, route.q]
    end

    @test Kraken.c2f_subcycled_route_packet_2d(
        F, spec, route; coarse_to_fine_prolongation=:limited_linear) ==
        route.weight * F[route.src, route.q]

    @test_throws ArgumentError Kraken.c2f_route_packet_2d(
        :limited_linear, F, route, ctx)
    @test_throws ArgumentError Kraken.c2f_route_packet_2d(
        :not_a_scheme, F, route, ctx)
    @test_throws ArgumentError Kraken.c2f_subcycled_route_packet_2d(
        F, spec, route; coarse_to_fine_prolongation=:not_a_scheme)
    @test_throws ArgumentError Kraken.c2f_subcycled_route_packet_2d(
        F, spec, route; alpha=0.5, coarse_to_fine_prolongation=:limited_linear)
end
