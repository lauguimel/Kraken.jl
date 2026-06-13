using Test
using Kraken

if get(ENV, "KRAKEN_AD_ONLY", "false") == "true"
    include("ad/test_ad_sensitivity.jl")
    exit()
end

# IncNS + solve-services tier (platform contract, linear-solve seam, Poisson
# services, IncNS drivers, scalar transport). Runs after the LBM tier by
# default; KRAKEN_INCNS_ONLY=true runs it alone (mirrors KRAKEN_AD_ONLY).
# Heavy validations (Ghia cavities, MG MMS up to 512²) additionally gate on
# KRAKEN_TEST_HEAVY=true.
function run_incns_testset()
    @testset "IncNS + solve services" begin
        # Platform-contract parity (IncNS wrapper vs direct driver call).
        include("platform/incns_contract_test.jl")

        # Solve services: linear-solve seam + elliptic (Poisson) MMS receipts.
        include("analytical/linear_solve_nonsym.jl")
        include("analytical/poisson_mms.jl")
        include("analytical/poisson_embedded_mms.jl")
        include("analytical/poisson_embedded_fvfd_mms.jl")

        # FVFD velocity-operator trio (grad/div/laplacian + embedded variants).
        include("analytical/incns_grad_div_laplacian_mms.jl")

        # IncNS drivers + scalar transport, analytical validation.
        include("analytical/incns_poiseuille.jl")
        include("analytical/incns_unsteady_taylor_green.jl")
        include("analytical/incns_unsteady_startup_channel.jl")
        include("analytical/incns_manifold.jl")   # fast (~4 s CPU): stays non-heavy
        include("analytical/scalar_transport_heated_channel.jl")

        # Heavy validations (long CPU runs: Ghia cavities, MG MMS up to 512²) —
        # opt in via KRAKEN_TEST_HEAVY=true.
        if get(ENV, "KRAKEN_TEST_HEAVY", "false") == "true"
            include("analytical/incns_cavity_ghia.jl")
            include("analytical/incns_cavity_mg_ghia.jl")
            include("analytical/poisson_mg_mms.jl")
        else
            @info "Skipping heavy IncNS validations (set KRAKEN_TEST_HEAVY=true to run)"
        end
    end
end

if get(ENV, "KRAKEN_INCNS_ONLY", "false") == "true"
    run_incns_testset()
    exit()
end

@testset "Kraken.jl LBM" begin
    include("platform/contract_parity_test.jl")
    include("platform/residual_vjp_test.jl")
    include("platform/calibration_test.jl")
    include("platform/calibration_nufield_test.jl")
    include("test_lbm_basic.jl")
    include("test_poiseuille.jl")
    include("test_poiseuille_3d.jl")
    include("test_couette.jl")
    include("test_taylor_green.jl")
    include("test_thermal.jl")
    include("test_axisymmetric.jl")
    include("test_mrt.jl")
    include("test_species.jl")
    include("test_multiphase.jl")
    include("test_vof.jl")
    include("test_benchmark.jl")
    include("test_cavity.jl")
    include("test_cavity_3d.jl")
    include("test_thermal_3d_krk.jl")
    include("test_cylinder.jl")
    include("test_expression.jl")
    include("test_kraken_parser.jl")
    include("test_krk_symbolic.jl")
    include("test_simulation_runner.jl")
    include("test_stl.jl")
    include("test_geometry_stl_krk.jl")
    include("test_geometry_descriptor.jl")
    include("test_geometry_units_krk.jl")
    include("test_geometry_units_3d_krk.jl")
    include("test_geometry_stl_flow_3d_krk.jl")
    include("test_krk_examples.jl")
    include("test_refinement.jl")
    include("test_conservative_tree_2d.jl")
    include("test_conservative_tree_topology_2d.jl")
    include("test_conservative_tree_streaming_2d.jl")
    include("test_curvilinear_mesh.jl")
    include("test_slbm.jl")
    include("test_slbm_taylor_green.jl")
    include("test_slbm_taylor_couette.jl")
    include("test_fused_trt_2d.jl")
    include("test_li_bb_2d.jl")
    include("test_kernel_dsl.jl")
    include("test_couette_libb_canary.jl")
    include("test_couette_libb_canary_3d.jl")
    include("test_cylinder_libb.jl")
    include("test_sphere_libb.jl")
    include("test_sphere_stl_drag_krk.jl")
    include("test_slbm_libb_3d.jl")
    include("test_gmsh_loader.jl")
    include("test_multiblock_topology.jl")
    include("test_multiblock_exchange.jl")
    include("test_multiblock_canal.jl")
    include("test_stl_libb.jl")
    include("test_taylor_couette_libb.jl")
    include("test_advection_prescribed.jl")
    include("test_krk_multiphase.jl")
    include("test_vtk_3d.jl")
    include("test_phasefield.jl")
    include("test_twophase_rheology.jl")
    include("test_postprocess.jl")
    include("test_rheology.jl")
    include("test_viscoelastic.jl")
    include("test_viscoelastic_krk.jl")
    # AD steady-sensitivity tests need the Enzyme extension (weakdep). Run only when Enzyme
    # is loadable in this environment; skip cleanly otherwise (guard the LOAD, not the tests,
    # so real AD test failures still surface when Enzyme IS present).
    let enzyme_ok = try
            @eval Main using Enzyme
            true
        catch
            false
        end
        if enzyme_ok
            include("ad/test_ad_sensitivity.jl")
            include("ad/test_ad_ve_sensitivity.jl")
        else
            @info "Skipping AD steady-sensitivity tests (Enzyme extension not loadable in this environment)"
        end
    end

    @testset "Kraken.Units" begin
        include("test_units.jl")
        include("test_units_stability.jl")
        include("test_units_audit.jl")
        include("test_units_krk.jl")
        include("test_units_thermal.jl")
    end
end

run_incns_testset()
