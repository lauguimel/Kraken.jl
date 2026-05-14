using Test
using Kraken

const AMR_D_LADDER_RESULT = NamedTuple{
    (:ok, :max_err, :where, :msg),
    Tuple{Bool, Float64, String, String}
}

_ladder_result(ok::Bool, max_err, where, msg) =
    AMR_D_LADDER_RESULT((ok, Float64(max_err), String(where), String(msg)))

function _amr_d_ladder_convergence_dir()
    return joinpath(dirname(@__DIR__), "benchmarks", "krk",
                    "amr_d_convergence_2d")
end

function _amr_d_ladder_macroscopic(F; force_x=0.0, force_y=0.0)
    nx, ny, nq = size(F)
    nq == 9 || throw(ArgumentError("D2Q9 state must have 9 populations"))
    rho = Matrix{Float64}(undef, nx, ny)
    ux = Matrix{Float64}(undef, nx, ny)
    uy = Matrix{Float64}(undef, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        mass = 0.0
        mx = 0.0
        my = 0.0
        for q in 1:9
            Fq = Float64(F[i, j, q])
            mass += Fq
            mx += d2q9_cx(q) * Fq
            my += d2q9_cy(q) * Fq
        end
        rho[i, j] = mass
        ux[i, j] = (mx + force_x / 2) / mass
        uy[i, j] = (my + force_y / 2) / mass
    end
    return rho, ux, uy
end

function _amr_d_ladder_max_abs_with_location(A)
    max_err = -1.0
    where = CartesianIndex(1, 1)
    @inbounds for idx in CartesianIndices(A)
        err = abs(Float64(A[idx]))
        if err > max_err
            max_err = err
            where = idx
        end
    end
    return max_err, "cell $(Tuple(where))"
end

function _amr_d_ladder_profile(result)
    profile = Float64.(getproperty(result, :ux_profile))
    reference = hasproperty(result, :analytic_profile) ?
        Float64.(getproperty(result, :analytic_profile)) :
        Float64.(getproperty(result, :analytic_ux_profile))
    return profile, reference
end

function _amr_d_ladder_relative_l2_profile(result)
    profile, reference = _amr_d_ladder_profile(result)
    length(profile) == length(reference) ||
        throw(ArgumentError("profile and reference lengths differ"))
    denom = sqrt(sum(abs2, reference))
    denom > 0 || throw(ArgumentError("analytic profile has zero L2 norm"))
    rel_l2 = sqrt(sum(abs2(profile[i] - reference[i])
                      for i in eachindex(profile))) / denom
    abs_diffs = abs.(profile .- reference)
    _, idx = findmax(abs_diffs)
    return rel_l2, "profile y-index $idx"
end

function marche_1()
    nx = 32
    ny = 32
    F = Array{Float64}(undef, nx, ny, 9)
    Fnext = similar(F)
    fill_equilibrium_integrated_D2Q9!(F, 1.0, 1.0, 0.0, 0.0)
    expected = copy(F)
    east_q = findfirst(q -> d2q9_cx(q) == 1 && d2q9_cy(q) == 0, 1:9)
    expected[16, 16, east_q] += 0.125
    F .= expected
    for _ in 1:32
        stream_fully_periodic_F_2d!(Fnext, F)
        F, Fnext = Fnext, F
    end
    max_err, where = _amr_d_ladder_max_abs_with_location(F .- expected)
    # bit-exact: integer pull streaming copies Float64 populations with no FP ops.
    return _ladder_result(max_err == 0.0, max_err, where,
                          "pure east-population periodic streaming after 32 steps")
end

function marche_2()
    nx = 32
    ny = 32
    F = Array{Float64}(undef, nx, ny, 9)
    Fnext = similar(F)
    fill_equilibrium_integrated_D2Q9!(F, 1.0, 1.0, 0.05, 0.0)
    for _ in 1:200
        collide_BGK_integrated_D2Q9!(F, 1.0, 1.0)
        stream_fully_periodic_F_2d!(Fnext, F)
        F, Fnext = Fnext, F
    end
    rho, ux, uy = _amr_d_ladder_macroscopic(F)
    rho_rel, rho_where = _amr_d_ladder_max_abs_with_location(rho .- 1.0)
    ux_rel, ux_where = _amr_d_ladder_max_abs_with_location((ux .- 0.05) ./ 0.05)
    uy_abs, uy_where = _amr_d_ladder_max_abs_with_location(uy)
    max_err, where = findmax([(rho_rel, "rho $rho_where"),
                              (ux_rel, "ux $ux_where"),
                              (uy_abs, "uy $uy_where")])
    # machine epsilon x 200 steps: periodic uniform BGK is an exact fixed point.
    ok = rho_rel <= 1e-13 && ux_rel <= 1e-12 && uy_abs <= 1e-12
    return _ladder_result(ok, max_err[1], max_err[2],
                          "uniform BGK fixed point: rho and velocity drift")
end

function marche_3()
    nx = 32
    ny = 32
    steps = 500
    gx = 1e-5
    F = Array{Float64}(undef, nx, ny, 9)
    Fnext = similar(F)
    fill_equilibrium_integrated_D2Q9!(F, 1.0, 1.0, 0.0, 0.0)
    for _ in 1:steps
        collide_Guo_integrated_D2Q9!(F, 1.0, 1.0, gx, 0.0)
        stream_fully_periodic_F_2d!(Fnext, F)
        F, Fnext = Fnext, F
    end
    _, ux, uy = _amr_d_ladder_macroscopic(F; force_x=gx)
    ux_err = abs(sum(ux) / length(ux) - gx * steps)
    uy_abs, uy_where = _amr_d_ladder_max_abs_with_location(uy)
    max_err = max(ux_err, uy_abs)
    where = ux_err >= uy_abs ? "spatial mean ux" : "uy $uy_where"
    # machine epsilon x 500 steps plus one grid reduction: uniform force is bulk acceleration.
    ok = ux_err <= 1e-12 && uy_abs <= 1e-12
    return _ladder_result(ok, max_err, where,
                          "periodic Guo forcing: mean ux versus gx*N and uy versus zero")
end

function marche_4()
    path = joinpath(_amr_d_ladder_convergence_dir(), "poiseuille_scale1.krk")
    if !isfile(path)
        # TODO: add a no-refinement AMR-D Poiseuille .krk fixture equivalent to poiseuille_scale1.krk.
        return _ladder_result(true, NaN, path,
                              "SKIPPED missing no-refinement poiseuille_scale1.krk fixture")
    end
    result = run_conservative_tree_amr_d_case_from_krk_2d(path)
    rel_l2, where = _amr_d_ladder_relative_l2_profile(result)
    # O(Ma^2) BB error + grid-induced O(dx^2); poiseuille_scale1 is Ma=O(0.05), H=O(32) -> 5e-3 ceiling.
    return _ladder_result(rel_l2 < 5e-3, rel_l2, where,
                          "BB Poiseuille no-refinement relative L2 ux profile")
end

function marche_5()
    path = joinpath(_amr_d_ladder_convergence_dir(), "poiseuille_yband_scale1.krk")
    if !isfile(path)
        # TODO: add a one-refinement-band AMR-D Poiseuille .krk fixture if poiseuille_yband_scale1.krk is removed.
        return _ladder_result(true, NaN, path,
                              "SKIPPED missing one-refinement-band Poiseuille fixture")
    end
    result = run_conservative_tree_amr_d_case_from_krk_2d(path)
    rel_l2, where = _amr_d_ladder_relative_l2_profile(result)
    # One C2F/F2C interface should not degrade marche 4 by more than about 2x.
    return _ladder_result(rel_l2 < 1e-2, rel_l2, where,
                          "one-refinement-band Poiseuille relative L2 ux profile")
end

function marche_6()
    path = joinpath(_amr_d_ladder_convergence_dir(),
                    "poiseuille_yband_nested4_debug.krk")
    result = run_conservative_tree_amr_d_case_from_krk_2d(path)
    rel_l2, where = _amr_d_ladder_relative_l2_profile(result)
    # Nested-4 accumulates four interfaces; budget is 4x the single-level error.
    return _ladder_result(rel_l2 < 2e-2, rel_l2, where,
                          "nested-4 Poiseuille relative L2 ux profile")
end

const AMR_D_LADDER_MARCHES = (
    ("1", "pure streaming", marche_1),
    ("2", "BGK uniform periodic", marche_2),
    ("3", "BGK constant body force", marche_3),
    ("4", "BB Poiseuille no refinement", marche_4),
    ("5", "one refinement band Poiseuille", marche_5),
    ("6", "nested-4 refinement Poiseuille", marche_6),
)

function run_amr_d_ladder()
    skipped = AMR_D_LADDER_RESULT[]
    for (idx, name, marche) in AMR_D_LADDER_MARCHES
        result = marche()
        if startswith(result.msg, "SKIPPED")
            println("[LADDER] marche $idx SKIPPED: $(result.msg)")
            push!(skipped, result)
            continue
        end
        if !result.ok
            println("[LADDER] marche $idx FAILED: $(result.msg)")
            println("[LADDER]   max_err = $(result.max_err) at $(result.where)")
            println("[LADDER]   (marches 1..$(parse(Int, idx) - 1) passed)")
            return (ok=false, failed_marche=parse(Int, idx), name=name,
                    max_err=result.max_err, where=result.where,
                    msg=result.msg, skipped=skipped)
        end
    end
    if isempty(skipped)
        println("[LADDER] all 6 marches passed")
    else
        println("[LADDER] all runnable marches passed ($(length(skipped)) skipped)")
    end
    return (ok=true, failed_marche=0, name="", max_err=0.0, where="",
            msg="", skipped=skipped)
end

@testset "AMR-D validation ladder" begin
    result = run_amr_d_ladder()
    for skipped in result.skipped
        @test_skip skipped.msg
    end
    @test result.ok
end
