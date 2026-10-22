# Viscosity-curve comparison metrics: reference range, curve errors, direct CY
# fit, curve criterion, summary-table layout. CPU, pure functions, no GPU.
# The run-1 reference values (`Kraken.load_reference_run1`) and the expected
# values derived from them (`reference_metrics.toml`) are not distributed:
# set KRAKEN_REFERENCE_DATA; the checks that need them are skipped without.
using Test
using Random
using TOML
using Kraken

const K = Kraken
const TRUTH1 = (eta0 = 1.0, eta_inf = 0.02, lambda = 5.0, n = 0.7, a = 2.0)
const _REF1 = K.load_reference_run1()
const _REFM = let p = K.reference_data_path("reference_metrics.toml")
    (p === nothing || !isfile(p)) ? nothing : TOML.parsefile(p)
end

@testset "viscosity curve metrics" begin

    @testset "shear rate on the staggered grid, reference range" begin
        dx = 1 / 32; Nx, Ny = 256, 128; Hh = 4.0; c = 0.7
        y = [(j - 0.5) * dx for j in 1:Ny]
        ux = [c * y[j] * (Hh - y[j]) for i in 1:Nx, j in 1:Ny]
        uy = zeros(Nx, Ny)
        gd, valid = K.shear_rate_staggered(ux, uy; dx)
        @test all(valid)
        far = [abs(y[j] - 2) > 0.25 && y[j] > 5dx && y[j] < Hh - 5dx for i in 1:Nx, j in 1:Ny]
        exact = [c * abs(Hh - 2 * y[j]) for i in 1:Nx, j in 1:Ny]
        @test maximum(abs.(gd[far] .- exact[far]) ./ exact[far]) <= 1e-3
        # rotated flow: uy(x) profile gives the same shear rate (staggering swapped)
        uy2 = [c * ((i - 0.5) * dx) * (8.0 - (i - 0.5) * dx) for i in 1:Nx, j in 1:Ny]
        gd2, _ = K.shear_rate_staggered(zeros(Nx, Ny), uy2; dx)
        xs = [(i - 0.5) * dx for i in 1:Nx]
        # periodic wrap spoils the first and last columns, the wall rows impose uy = 0
        ok = [i > 3 && i < Nx - 3 && abs(xs[i] - 4) > 0.25 && j > 3 && j < Ny - 3 for i in 1:Nx, j in 1:Ny]
        ex2 = [c * abs(8.0 - 2 * xs[i]) for i in 1:Nx, j in 1:Ny]
        @test maximum(abs.(gd2[ok] .- ex2[ok]) ./ ex2[ok]) <= 1e-3
        # mask: a hole removes its 3x3 neighbourhood from the valid cells
        mask = trues(Nx, Ny); mask[100, 60] = false
        _, v2 = K.shear_rate_staggered(ux, uy; dx, mask)
        @test count(.!v2) == 9
        @test !v2[100, 60] && !v2[99, 59] && !v2[101, 61] && v2[98, 60]
        # range: gd = c |H - 2y| is uniform on [0, 4c] over y; percentiles 5 and 95 ~ 0.2c, 3.8c
        r = K.reference_shear_range(ux, uy, trues(Nx, Ny); dx)
        @test isapprox(r.gd_lo, 0.2 * c; rtol = 0.1)
        @test isapprox(r.gd_hi, 3.8 * c; rtol = 0.02)
        @test r.n == Nx * Ny
        # percentiles are reproducible at a fixed seed and follow the numpy rule
        Random.seed!(2026); v1 = randn(1000); Random.seed!(2026); v2_ = randn(1000)
        @test K.curve_quantile(v1, 0.05) == K.curve_quantile(v2_, 0.05)
        @test K.curve_quantile(collect(1.0:101.0), 0.05) == 6.0
        @test K.curve_quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
        @test K.curve_median([3.0, 1.0, 2.0]) == 2.0
    end

    @testset "curve errors and direct CY fit" begin
        rng = (0.1, 10.0)
        cy = K.cy_law(TRUTH1)
        e = K.viscosity_curve_errors(cy, cy, rng)
        @test e.max <= 1e-12 && e.median <= 1e-12
        e2 = K.viscosity_curve_errors(g -> 1.1 * cy(g), cy, rng)
        @test isapprox(e2.max, 0.1; atol = 1e-12) && isapprox(e2.median, 0.1; atol = 1e-12)
        # table: CY sampled on a fine log grid, log-log interpolation, small error
        gtab = exp.(range(log(0.01), log(100); length = 4000))
        tab = (gtab, cy.(gtab))
        @test K.viscosity_curve_errors(tab, cy, rng).max <= 1e-4
        # median over its own range: a 10 % error on [0.1, 1] only
        part(g) = cy(g) * (g <= 1.0 ? 1.1 : 1.0)
        wide = (0.1, 10.0)
        @test isapprox(K.viscosity_curve_errors(part, cy, wide; median_range = (0.1, 1.0)).median, 0.1; atol = 1e-12)
        @test K.viscosity_curve_errors(part, cy, wide; median_range = (2.0, 10.0)).median <= 1e-12
        @test K.viscosity_curve_errors(part, cy, wide; median_range = (2.0, 10.0)).max > 0.09   # max stays on `range`
        @test K.shear_range_lo(wide) == 0.1 && K.shear_range_hi((gd_lo = 1.0, gd_hi = 2.0, n = 3)) == 2.0
        @test K.rel_gap(1.1, 1.0) ≈ 0.1
        @test_throws ArgumentError K.eta_function(([2.0, 1.0], [1.0, 1.0]))
        # exact CY, eta0 fixed: (k, n) back to 1e-6
        f = K.cy_fit_direct(cy, rng)
        @test isapprox(f.k, 5.0; rtol = 1e-6) && isapprox(f.n, 0.7; rtol = 1e-6)
        @test isapprox(f.eta_inf, 0.02; rtol = 1e-4) && f.eta0 == 1.0
        for p in ((eta0 = 1.0, eta_inf = 0.03, lambda = 4.0, n = 0.75, a = 2.0),
                  (eta0 = 1.0, eta_inf = 0.025, lambda = 3.5, n = 0.65, a = 2.0))
            fp = K.cy_fit_direct(K.cy_law(p), (0.1, 10.0))
            @test isapprox(fp.k, p.lambda; rtol = 1e-6) && isapprox(fp.n, p.n; rtol = 1e-6)
        end
        # eta0 free (diagnostic): exact data still recovered
        ff = K.cy_fit_direct(cy, rng; free_eta0 = true)
        @test isapprox(ff.eta0, 1.0; rtol = 1e-5) && isapprox(ff.n, 0.7; rtol = 1e-4)
        # fitted exponent of the reference curve: a free, a < 1 law is returned with a close to it
        if _REF1 === nothing
            @test_skip "reference run-1 values not distributed (set KRAKEN_REFERENCE_DATA)"
        else
            pa = map(x -> round(x; sigdigits = 3), _REF1.rerun)
            fa = K.cy_fit_direct(K.cy_law(pa), rng; free_eta0 = true, fit_a = true)
            @test isapprox(fa.n, pa.n; rtol = 1e-3) && isapprox(fa.a, pa.a; rtol = 1e-3)
        end
        # 1 % relative noise: indicative threshold 3 % on (k, n)
        Random.seed!(7)
        noisy = (gtab, cy.(gtab) .* (1 .+ 0.01 .* randn(length(gtab))))
        fn = K.cy_fit_direct(noisy, rng)
        @info "noisy CY fit" k_err = abs(fn.k - 5) / 5 n_err = abs(fn.n - 0.7) / 0.7
        @test abs(fn.k - 5) / 5 <= 0.03 && abs(fn.n - 0.7) / 0.7 <= 0.03
        # eta_inf share
        @test K.eta_inf_plateau_share(TRUTH1, rng) == 0.02 / cy(10.0)
    end

    if _REF1 === nothing
    @testset "curve criterion (reference run-1 values)" begin
        @test_skip "reference run-1 values not distributed (set KRAKEN_REFERENCE_DATA)"
    end
    else
    @testset "curve criterion, each sub-criterion alone" begin
        rng = (0.1, 10.0)
        s1 = _REF1.reported
        cy = K.cy_law(TRUTH1)
        good = (eta = cy, data_loss = 1.0e-4, cy_fit_data_loss = 1.0e-4, k = 5.0, n = 0.7)
        rerun_run1 = (lambda = _REF1.rerun.lambda, n = _REF1.rerun.n, eta = _REF1.rerun)
        # substitute for the run-1 range; expected values in reference_metrics.toml
        sub = K.run1_substitute()
        @test sub.eta_max >= sub.eta_med > 0 && !isnan(sub.eta_max)
        if _REFM === nothing
            for _ in 1:4
                @test_skip "expected run-1 metrics not distributed (reference_metrics.toml)"
            end
        else
            M = _REFM["run1_substitute"]
            @test isapprox(sub.dk, M["dk"]; atol = 1e-3) && isapprox(sub.dn, M["dn"]; atol = 1e-3)
            @test isapprox(sub.eta_max, M["eta_max"]; atol = 5e-4)
            @test isapprox(sub.eta_med, M["eta_med"]; atol = 2e-3)
            @test K.run1_substitute() == sub && isapprox(_REF1.range.gd_hi, M["gd_hi"]; atol = 1e-4)
        end
        @test sub == K.run1_substitute(_REF1.range, _REF1.table)
        r0 = K.curve_criterion_check(1, good, TRUTH1, s1, nothing; range = rng)
        @test r0.pass && length(r0.rows) == 6
        @test [r.decides for r in r0.rows] == [true, true, true, true, true, false]
        @test r0.rows[3].criterion != r0.rows[6].criterion && occursin("[0.1, gd_95]", r0.rows[3].criterion)
        @test r0.rows[3].threshold == max(0.05, sub.eta_med) && r0.rows[2].threshold == max(0.05, sub.eta_max)
        @test all(occursin("substitute", r.note) for r in r0.rows[2:6])     # declared as substitute
        # 1: data loss ratio 3 -> only the loss fails
        r1 = K.curve_criterion_check(1, merge(good, (; data_loss = 3.0e-4)), TRUTH1, s1, nothing; range = rng)
        @test !r1.pass && [r.pass for r in r1.rows[1:5]] == [false, true, true, true, true]
        # 2: eta max error (one sharp local bump) fails, the median stays
        bump(g) = cy(g) * (1 + (abs(log(g / 3.0)) < 0.05 ? 0.5 : 0.0))
        r2 = K.curve_criterion_check(1, merge(good, (; eta = bump)), TRUTH1, s1, rerun_run1; range = (0.1, 10.0))
        @test [r.pass for r in r2.rows][[1, 3, 4, 5]] == [true, true, true, true]
        @test r2.rows[2].value > r2.rows[2].threshold && !r2.rows[2].pass && !r2.pass
        # 3: median error fails (10 % everywhere): max fails too, k and n untouched
        r3 = K.curve_criterion_check(1, merge(good, (; eta = g -> 1.1 * cy(g))), TRUTH1, s1, nothing; range = rng)
        @test !r3.rows[3].pass && r3.rows[4].pass && r3.rows[5].pass
        # 4: k gap 15 %, 5: n gap 8 %, each above its threshold max(2 x reported gap, substitute gap)
        r4 = K.curve_criterion_check(1, merge(good, (; k = 5.75)), TRUTH1, s1, nothing; range = rng)
        @test [r.pass for r in r4.rows[1:5]] == [true, true, true, false, true]
        r5 = K.curve_criterion_check(1, merge(good, (; n = 0.756)), TRUTH1, s1, nothing; range = rng)
        @test [r.pass for r in r5.rows[1:5]] == [true, true, true, true, false]
        # a gap just inside the substitute thresholds passes
        thr_k = max(2 * K.rel_gap(s1.lambda, TRUTH1.lambda), sub.dk)
        thr_n = max(2 * K.rel_gap(s1.n, TRUTH1.n), sub.dn)
        r6 = K.curve_criterion_check(1, merge(good, (; k = 5.0 * (1 + 0.97 * thr_k), n = 0.7 * (1 + 0.97 * thr_n))), TRUTH1, s1, nothing; range = rng)
        @test r6.rows[4].pass && r6.rows[5].pass
        # with a reference curve the thresholds follow it: reference off by 10 % on eta -> 10 % allowed
        rr = (lambda = 5.5, n = 0.75, eta = g -> 1.1 * cy(g))
        r7 = K.curve_criterion_check(1, merge(good, (; eta = g -> 1.09 * cy(g))), TRUTH1, s1, rr; range = rng)
        @test r7.rows[2].pass && r7.rows[3].pass && all(!occursin("substitute", r.note) for r in r7.rows)
        r8 = K.curve_criterion_check(1, merge(good, (; eta = g -> 1.12 * cy(g))), TRUTH1, s1, rr; range = rng)
        @test !r8.rows[2].pass
        # eta_max NaN in the substitute: only the 5 % floor, said in the note
        nomax = (dk = 0.1, dn = 0.1, eta_med = 0.1, eta_max = NaN)
        r9 = K.curve_criterion_check(1, good, TRUTH1, s1, nothing; range = rng, substitute = nomax)
        @test r9.rows[2].threshold == 0.05 && occursin("5 % floor", r9.rows[2].note)
        # the deciding median is the one on [0.1, gd_95]; the full-range median is only reported
        lowbad(g) = cy(g) * (g < 0.1 ? 1.5 : 1.0)
        wide5 = (1.0e-4, 10.0)
        r11 = K.curve_criterion_check(1, merge(good, (; eta = lowbad)), TRUTH1, s1, rerun_run1; range = wide5)
        @test r11.rows[3].value <= 1e-12 && r11.rows[3].pass && r11.rows[3].decides
        @test r11.rows[6].value > 0.05 && !r11.rows[6].pass && !r11.rows[6].decides
        @test !r11.rows[2].pass && !r11.pass                                  # the max does see the low-shear error
        # 0.1 is a parameter
        r12 = K.curve_criterion_check(1, merge(good, (; eta = lowbad)), TRUTH1, s1, rerun_run1; range = wide5, median_lo = 1.0e-4)
        @test r12.rows[3].value > 0.05 && !r12.rows[3].pass
        # k, n default to the direct fit of eta
        r10 = K.curve_criterion_check(1, (eta = cy, data_loss = 1.0, cy_fit_data_loss = 1.0), TRUTH1, s1, nothing; range = rng)
        @test r10.pass
    end
    end # _REF1

    @testset "curve criterion verdict" begin
        @test startswith(K.curve_criterion_verdict(trues(7)), "criterion met")
        @test occursin("partly met", K.curve_criterion_verdict([trues(5); falses(2)]))
        @test occursin("not met", K.curve_criterion_verdict([trues(4); falses(3)]))
    end

    @testset "summary-table layout" begin
        mk(i, pass) = (fluid = i, m = i == 4 ? 72 : 64, m_alt = i == 4 ? 64 : 32, truth = (lambda = 5.0, n = 0.7),
                       reported = (lambda = 4.8, n = 0.72), rerun = i == 1 ? (lambda = 5.2, n = 0.75) : nothing,
                       kraken = (lambda = 5.01, n = 0.699), kraken_alt = (i <= 2 || i == 4) ? (lambda = 5.1, n = 0.69) : nothing,
                       eta_inf_share = 0.0123, met = pass)
        rows = [mk(i, i != 6) for i in 1:7]
        md = K.closure_summary_table(rows)
        lines = split(md, '\n')
        @test occursin("| fluid | m |", lines[1]) && startswith(lines[2], "|---")
        @test count(startswith("| "), lines) == 8                       # header + 7 fluids
        @test occursin("| 4 | 72 |", md) && occursin("| 1 | 64 |", md)
        @test occursin("5.2 / 0.75", md) && occursin("(32)", md) && occursin("(64)", md)
        @test occursin("fail", md) && occursin("pass", md)
        @test occursin("Verdict: criterion partly met (6/7)", md)
        @test !occursin("Verdict:", K.closure_summary_table(rows; verdict = false))
    end
end
