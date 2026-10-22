# Inverse rheology: comparison of a recovered viscosity curve eta(gamma_dot)
# against a reference curve (curve errors, direct Carreau-Yasuda fit, a curve
# criterion made of several sub-criteria) and a summary-table layout.
#
# Pure functions on arrays and callables, independent of how the recovered
# curve was obtained: a law is either a function `gd -> eta`, or a
# table `(gd, eta)` of two vectors (log-log linear interpolation).
#
# Names are not exported: call them as `Kraken.<name>`.
#
# Units: starred (dimensionless) units of the obstacle channel, eta0 = 1, G = 5.
# gamma_dot = sqrt(2 S:S), the convention of `gnf_shear_rate` and of the
# reference data.

import LinearAlgebra

# --------------------------------------------------------------- basics ---

"""
    cy_law(p) -> (gd -> eta)

Carreau-Yasuda law of `p = (; eta0, eta_inf, lambda, n, a)`:
`eta_inf + (eta0 - eta_inf) (1 + (lambda gd)^a)^((n-1)/a)`.
"""
function cy_law(p)
    return gd -> p.eta_inf + (p.eta0 - p.eta_inf) * (1 + (p.lambda * gd)^p.a)^((p.n - 1) / p.a)
end

"""
    eta_function(x) -> (gd -> eta)

A function is returned as is; a tuple `(gd, eta)` of vectors (gd increasing)
becomes a log-log linear interpolant, clamped at both ends; a NamedTuple with
the fields `eta0, eta_inf, lambda, n, a` becomes `cy_law`.
"""
eta_function(f::Function) = f
eta_function(p::NamedTuple) = cy_law(p)
function eta_function(t::Tuple{<:AbstractVector,<:AbstractVector})
    gd, eta = t
    length(gd) == length(eta) >= 2 || throw(ArgumentError("table needs >= 2 points"))
    issorted(gd) || throw(ArgumentError("table gamma_dot must be increasing"))
    lg = log.(Float64.(gd)); le = log.(Float64.(eta))
    return function (x)
        l = log(x)
        l <= lg[1] && return exp(le[1])
        l >= lg[end] && return exp(le[end])
        k = searchsortedlast(lg, l)
        w = (l - lg[k]) / (lg[k + 1] - lg[k])
        return exp((1 - w) * le[k] + w * le[k + 1])
    end
end

"""
    curve_quantile(v, p) -> Float64

Quantile `p` in [0, 1] of the vector `v`, linear interpolation between order
statistics (the default of numpy.percentile).
"""
function curve_quantile(v::AbstractVector, p::Real)
    isempty(v) && throw(ArgumentError("empty vector"))
    s = sort(Float64.(v))
    h = (length(s) - 1) * p + 1
    lo = clamp(floor(Int, h), 1, length(s)); hi = min(lo + 1, length(s))
    return s[lo] + (h - lo) * (s[hi] - s[lo])
end
curve_median(v::AbstractVector) = curve_quantile(v, 0.5)

# ------------------------------------------- shear rate, staggered grid ---

"""
    shear_rate_staggered(ux, uy; dx=1/32, mask=nothing) -> (gd, valid)

Shear rate `gd = sqrt(2 S:S)` at the cell centres of the staggered (MAC) grid
of the reference data: `ux[i, j]` at `((i)dx, (j-1/2)dx)`, `uy[i, j]` at
`((i-1/2)dx, (j)dx)` for 1-based `i, j`, so that cell `(i, j)` has its centre
at `((i-1/2)dx, (j-1/2)dx)` with `ux[i-1, j]`, `ux[i, j]` on its west and east
faces and `uy[i, j-1]`, `uy[i, j]` on its south and north faces.

- `Sxx = (ux[i,j] - ux[i-1,j])/dx`, `Syy = (uy[i,j] - uy[i,j-1])/dx` at the centre;
- `Sxy = (dux/dy + duy/dx)/2` is formed at the four cell corners from the
  two nearest values (`(ux[i,j+1]-ux[i,j])/dx`, ...) and averaged to the centre;
- periodic in x; in y the walls y = 0, 4 carry `uy = 0` on the wall row and
  `ux` is mirrored (`ux = -ux` across the wall, no slip half-way).

`valid[i, j]` is true when every field value entering the cell is inside
`mask` (when given): partial stencils are not used. Returns `Float64` arrays.
"""
function shear_rate_staggered(ux::AbstractMatrix, uy::AbstractMatrix; dx = 1 / 32, mask = nothing)
    size(ux) == size(uy) || throw(DimensionMismatch("ux and uy must have the same size"))
    Nx, Ny = size(ux)
    gd = zeros(Float64, Nx, Ny)
    valid = trues(Nx, Ny)
    M = mask === nothing ? trues(Nx, Ny) : (mask .!= 0)
    # value of a field with the wall rules; (i, j) may be out of range
    function U(i, j)                    # ux
        ii = mod1(i, Nx)
        j < 1 && return -Float64(ux[ii, 1])
        j > Ny && return -Float64(ux[ii, Ny])
        return Float64(ux[ii, j])
    end
    function V(i, j)                    # uy
        ii = mod1(i, Nx)
        j < 1 && return 0.0
        return Float64(uy[ii, j])
    end
    inmask(A, i, j) = (j < 1 || j > Ny) ? true : A[mod1(i, Nx), j]
    for j in 1:Ny, i in 1:Nx
        sxx = (U(i, j) - U(i - 1, j)) / dx
        syy = (V(i, j) - V(i, j - 1)) / dx
        # corners: (i, j) east/north ... corner (a, b) is the NE corner of cell (a, b)
        corner(a, b) = ((U(a, b + 1) - U(a, b)) / dx + (V(a + 1, b) - V(a, b)) / dx) / 2
        sxy = (corner(i, j) + corner(i - 1, j) + corner(i, j - 1) + corner(i - 1, j - 1)) / 4
        gd[i, j] = sqrt(2 * (sxx^2 + syy^2 + 2 * sxy^2))
        if mask !== nothing
            ok = all(inmask(M, i + di, j + dj) for di in -1:1, dj in -1:1)
            valid[i, j] = ok
        end
    end
    return gd, valid
end

"""
    reference_shear_range(ux, uy, mask; dx=1/32, lo=5, hi=95) -> (; gd_lo, gd_hi, n)

Reference shear-rate range: percentiles `lo`..`hi` of the shear rate of the
true field (reference velocity field) over the valid cells of the frozen
mask. Independent of any learned law. Cells with a zero shear rate are kept
(percentiles are rank statistics).
"""
function reference_shear_range(ux, uy, mask; dx = 1 / 32, lo = 5, hi = 95)
    gd, valid = shear_rate_staggered(ux, uy; dx, mask)
    v = vec(gd[valid])
    isempty(v) && throw(ArgumentError("no valid cell in the mask"))
    return (; gd_lo = curve_quantile(v, lo / 100), gd_hi = curve_quantile(v, hi / 100), n = length(v))
end

# ------------------------------------------------------- curve metrics ---

shear_range_lo(r) = r isa NamedTuple ? r.gd_lo : r[1]
shear_range_hi(r) = r isa NamedTuple ? r.gd_hi : r[2]

# Lower end of the median range of the curve criterion. Only the median on
# [CURVE_MEDIAN_LO, gd_95] decides.
const CURVE_MEDIAN_LO = 0.1

"""
    viscosity_curve_errors(eta_learned, eta_true, range; npoints=256, median_range=range) -> (max, median)

Relative error `|eta_learned - eta_true| / eta_true` at `npoints` log-spaced
points: the maximum over `range` (a `(gd_lo, gd_hi)` tuple or the result of
`reference_shear_range`) and the median over `median_range` (same kinds of
value; default `range`).
"""
function viscosity_curve_errors(eta_learned, eta_true, range; npoints::Int = 256, median_range = range)
    fl = eta_function(eta_learned); ft = eta_function(eta_true)
    relerr(rg) = begin
        lo, hi = shear_range_lo(rg), shear_range_hi(rg)
        gd = exp.(Base.range(log(lo), log(hi); length = npoints))
        [abs(fl(g) - ft(g)) / ft(g) for g in gd]
    end
    emax = maximum(relerr(range))
    emed = median_range === range ? curve_median(relerr(range)) : curve_median(relerr(median_range))
    return (max = emax, median = emed)
end

# Levenberg-Marquardt with a forward-difference Jacobian, small problems only.
function _vc_lm(resid, x0; maxiter = 400, tol = 1e-14)
    x = copy(x0); r = resid(x); f = sum(abs2, r); λ = 1e-3
    for _ in 1:maxiter
        J = zeros(length(r), length(x))
        for k in eachindex(x)
            h = 1e-7 * max(1, abs(x[k]))
            xp = copy(x); xp[k] += h
            J[:, k] = (resid(xp) .- r) ./ h
        end
        A = J' * J; g = J' * r
        improved = false
        for _ in 1:30
            step = -((A + λ * LinearAlgebra.Diagonal(max.(LinearAlgebra.diag(A), 1e-12))) \ g)
            xn = x .+ step
            rn = resid(xn); fn = sum(abs2, rn)
            if isfinite(fn) && fn < f
                dec = f - fn
                x, r, f = xn, rn, fn; λ = max(λ / 5, 1e-15); improved = true
                dec <= tol * max(f, 1e-300) && return x, f
                break
            end
            λ *= 5
        end
        improved || break
    end
    return x, f
end

"""
    cy_fit_direct(eta_learned, range; free_eta0=false, fit_a=false,
                        npoints=256, eta0=1.0) -> (; k, n, eta_inf, eta0, a, rms)

Carreau-Yasuda fit of the learned law, by least squares on `log eta` at
`npoints` log-spaced points of `range`: free `k` (= lambda), `n`, `eta_inf`;
`eta0` fixed at `eta0` (default 1) unless `free_eta0`; `a = 2` unless `fit_a`
(the exponent of the true laws; `fit_a=true` fits it freely). `rms` is the rms
residual in `log eta`. Several starting points are tried and the best fit kept.
"""
function cy_fit_direct(eta_learned, range; free_eta0::Bool = false, fit_a::Bool = false,
                             npoints::Int = 256, eta0::Real = 1.0)
    fl = eta_function(eta_learned)
    lo, hi = shear_range_lo(range), shear_range_hi(range)
    gd = exp.(Base.range(log(lo), log(hi); length = npoints))
    y = [log(fl(g)) for g in gd]
    # parameters x = [log k, n, log eta_inf, (log eta0), (a)]
    function model(x)
        k = exp(x[1]); n = x[2]; ei = exp(x[3])
        pos = 4
        e0 = free_eta0 ? exp(x[pos]) : float(eta0); free_eta0 && (pos += 1)
        a = fit_a ? x[pos] : 2.0
        return [log(ei + (e0 - ei) * (1 + (k * g)^a)^((n - 1) / a)) for g in gd]
    end
    resid(x) = (x[2] < 0 || x[2] > 1.5 || (fit_a && x[end] < 0.2)) ? fill(1e3, length(gd)) : model(x) .- y
    best = nothing
    gmid = sqrt(lo * hi)
    ei_hi = exp(minimum(y)) * 0.9
    for kfac in (0.3, 1.0, 3.0, 10.0), n0 in (0.4, 0.7), ei0 in (0.1, 0.5, 1.0)
        x0 = Float64[log(kfac / gmid), n0, log(max(ei_hi * ei0, 1e-6))]
        free_eta0 && push!(x0, log(float(eta0)))
        fit_a && push!(x0, 2.0)
        x, f = try _vc_lm(resid, x0) catch; (x0, Inf) end
        (best === nothing || f < best[2]) && (best = (x, f))
    end
    x = best[1]
    pos = 4
    e0 = free_eta0 ? exp(x[pos]) : float(eta0); free_eta0 && (pos += 1)
    a = fit_a ? x[pos] : 2.0
    return (; k = exp(x[1]), n = x[2], eta_inf = exp(x[3]), eta0 = e0, a, rms = sqrt(best[2] / length(gd)))
end

"""
    eta_inf_plateau_share(truth, range) -> eta_inf / eta_true(gd_hi)

Share of the high-shear plateau in the true viscosity at the top of the
reference range (context only, not a criterion).
"""
eta_inf_plateau_share(truth, range) = truth.eta_inf / cy_law(truth)(shear_range_hi(range))

# ------------------------------------------------------- curve criterion ---

# Run-1 truth of the obstacle-channel case (Carreau, eta0 = 1, a = 2).
const RUN1_TRUTH = (eta0 = 1.0, eta_inf = 0.02, lambda = 5.0, n = 0.7, a = 2.0)

"""
    load_reference_run1(path=reference_data_path("src", "platform", "reference_run1_table.jl"))
        -> NamedTuple or nothing

Run-1 reference values (not distributed; see `reference_data_dir`): `range`
(reference shear-rate range of run 1, exact), `table` (reference viscosity
curve, a 100-point `(gd, eta)` table), `rerun` (Carreau-Yasuda reading of that
curve, `(; eta0, eta_inf, lambda, n, a)`) and `reported` (reference `(; lambda,
n)` of run 1). The file
defines `REF_RUN1_RANGE`, `REF_RUN1_RERUN_TABLE`, `REF_RUN1_RERUN` and
`REF_RUN1_REPORTED`; it is evaluated in a fresh module at call time.
`nothing` when the file is absent.
"""
function load_reference_run1(path = reference_data_path("src", "platform", "reference_run1_table.jl"))
    (path === nothing || !isfile(path)) && return nothing
    m = Module(:ReferenceRun1)
    Base.include(m, path)
    g(s) = Base.invokelatest(getglobal, m, s)
    return (range = g(:REF_RUN1_RANGE), table = g(:REF_RUN1_RERUN_TABLE), rerun = g(:REF_RUN1_RERUN),
            reported = g(:REF_RUN1_REPORTED))
end

"""
    run1_substitute(range1=ref.range, eta_rerun=ref.table; median_lo=CURVE_MEDIAN_LO,
                    ref=load_reference_run1()) -> (; dk, dn, eta_med, eta_med_full, eta_max)

Gaps of the run-1 reference curve to the run-1 truth: relative gaps of lambda and n
(Carreau-Yasuda reading `ref.rerun`), the maximum relative error over `range1`
(the reference range of run 1), the median over `[median_lo, gd_95 of range1]`
(`eta_med`, the deciding one) and the median over the full `range1`
(`eta_med_full`, reported). `eta_rerun` is the reference law (a function or a
`(gd, eta)` table): by default the tabulated curve `ref.table`, not its
Carreau-Yasuda reading. Used for fluids without their own reference curve.
Needs the reference data (`load_reference_run1`).
"""
function run1_substitute(range1 = nothing, eta_rerun = nothing;
                         median_lo::Real = CURVE_MEDIAN_LO, ref = load_reference_run1())
    ref === nothing && error("run1_substitute: reference data not found (set KRAKEN_REFERENCE_DATA)")
    range1 === nothing && (range1 = ref.range)
    eta_rerun === nothing && (eta_rerun = ref.table)
    mr = (median_lo, shear_range_hi(range1))
    e = viscosity_curve_errors(eta_rerun, RUN1_TRUTH, range1; median_range = mr)
    ef = viscosity_curve_errors(eta_rerun, RUN1_TRUTH, range1)
    return (dk = abs(ref.rerun.lambda - 5.0) / 5.0, dn = abs(ref.rerun.n - 0.7) / 0.7,
            eta_med = e.median, eta_med_full = ef.median, eta_max = e.max)
end

rel_gap(a, b) = abs(a - b) / abs(b)

"""
    curve_criterion_check(fluid, learned, truth, reported, rerun; range, substitute=run1_substitute(),
                   npoints=256, median_lo=CURVE_MEDIAN_LO) -> (pass::Bool, rows, fluid)

Comparison of a recovered viscosity curve against a reference curve, for one
fluid. The criterion is met when the five deciding sub-criteria pass:

1. data loss: `learned.data_loss <= 2 * learned.cy_fit_data_loss` (the final
   data loss of the Kraken run, and that of the CY fitted with `eta0` fixed,
   both measured by the caller);
2. maximum relative error of `eta(gd)` on `range` (percentiles 5-95 of the truth
   on the frozen mask) <= max(5 %, maximum error of the reference curve, same range);
3. median relative error on `[median_lo, gd_95]` (`median_lo` = 0.1 by default)
   <= max(5 %, median error of the reference curve on the same range);
4. relative gap of `k` to the truth <= max(2 x reported gap / truth, gap of the reference curve / truth);
5. same for `n`.

The median on the full `range` is also computed and reported (row 6, `decides =
false`): it does not enter `pass`. Curves are sampled at `npoints` log-spaced points.

`learned = (; eta, data_loss, cy_fit_data_loss[, k, n])` (`k, n` default to
`cy_fit_direct(eta, range)`); `truth = (; eta0, eta_inf, lambda, n, a)`;
`reported = (; lambda, n)` (reference values for the fluid);
`rerun = (; lambda, n[, eta])` (reference curve of the fluid) or `(; lambda, n, eta_max,
eta_med[, eta_med_full])`, or `nothing`: then `substitute = (; dk, dn, eta_med,
eta_max[, eta_med_full])` (run 1 by default, built from the tabulated reference
curve, which needs the reference data) stands in and every row says so. `fluid` is a label.

`rows` is a vector of NamedTuples `(; criterion, value, threshold, pass, decides, note)`.
`pass` is the conjunction of the five rows with `decides = true`.
"""
function curve_criterion_check(fluid, learned, truth, reported, rerun; range, substitute = nothing,
                        npoints::Int = 256, median_lo::Real = CURVE_MEDIAN_LO)
    rerun === nothing && substitute === nothing && (substitute = run1_substitute())
    ftruth = cy_law(truth)
    mr = (median_lo, shear_range_hi(range))
    err = viscosity_curve_errors(learned.eta, ftruth, range; npoints, median_range = mr)
    err_full = viscosity_curve_errors(learned.eta, ftruth, range; npoints)
    fit = (hasproperty(learned, :k) && hasproperty(learned, :n)) ? (; k = learned.k, n = learned.n) :
          cy_fit_direct(learned.eta, range; npoints)
    note0 = rerun === nothing ? "substitute: run-1 reference" : "reference $(fluid)"
    nan = NaN
    if rerun === nothing
        r_max, r_med, r_dk, r_dn = substitute.eta_max, substitute.eta_med, substitute.dk, substitute.dn
        r_full = hasproperty(substitute, :eta_med_full) ? substitute.eta_med_full : nan
    else
        if hasproperty(rerun, :eta)
            re = viscosity_curve_errors(rerun.eta, ftruth, range; npoints, median_range = mr)
            r_max, r_med = re.max, re.median
            r_full = viscosity_curve_errors(rerun.eta, ftruth, range; npoints).median
        else
            r_max, r_med = rerun.eta_max, rerun.eta_med
            r_full = hasproperty(rerun, :eta_med_full) ? rerun.eta_med_full : nan
        end
        r_dk = rel_gap(rerun.lambda, truth.lambda); r_dn = rel_gap(rerun.n, truth.n)
    end
    floor_or(x) = isnan(x) ? 0.05 : max(0.05, x)
    max_note = rerun === nothing && isnan(r_max) ? note0 * " (max not available: 5 % floor only)" : note0
    rep_dk = rel_gap(reported.lambda, truth.lambda); rep_dn = rel_gap(reported.n, truth.n)
    thr_k = max(2 * rep_dk, r_dk); thr_n = max(2 * rep_dn, r_dn)
    dk = rel_gap(fit.k, truth.lambda); dn = rel_gap(fit.n, truth.n)
    rows = [
        (criterion = "data loss <= 2 x CY(eta0 fixed)", value = learned.data_loss / learned.cy_fit_data_loss,
         threshold = 2.0, pass = learned.data_loss <= 2 * learned.cy_fit_data_loss, decides = true, note = "ratio"),
        (criterion = "eta max error", value = err.max, threshold = floor_or(r_max),
         pass = err.max <= floor_or(r_max), decides = true, note = max_note),
        (criterion = "eta median error [0.1, gd_95]", value = err.median, threshold = floor_or(r_med),
         pass = err.median <= floor_or(r_med), decides = true, note = note0),
        (criterion = "k gap to truth", value = dk, threshold = thr_k, pass = dk <= thr_k, decides = true,
         note = "max(2 x reported gap $(round(rep_dk; sigdigits = 3)), $(note0) gap $(round(r_dk; sigdigits = 3)))"),
        (criterion = "n gap to truth", value = dn, threshold = thr_n, pass = dn <= thr_n, decides = true,
         note = "max(2 x reported gap $(round(rep_dn; sigdigits = 3)), $(note0) gap $(round(r_dn; sigdigits = 3)))"),
        (criterion = "eta median error, full 5-95 % range", value = err_full.median, threshold = floor_or(r_full),
         pass = err_full.median <= floor_or(r_full), decides = false,
         note = note0 * " (reported, does not decide)" * (isnan(r_full) ? "; reference median not available: 5 % floor only" : "")),
    ]
    return (pass = all(r.pass for r in rows if r.decides), rows = rows, fluid = fluid)
end

"""
    curve_criterion_verdict(passes) -> String

Verdict over a set of 7 fluids from the results of `curve_criterion_check`:
"criterion met" (7/7), "criterion partly met" (>= 5); otherwise "criterion not
met" (telling a different result from a failure needs the losses and the
curves: not decided here).
"""
function curve_criterion_verdict(passes::AbstractVector{Bool})
    n = count(passes); N = length(passes)
    n == N == 7 && return "criterion met (7/7)"
    n >= 5 && return "criterion partly met ($n/$N)"
    return "criterion not met ($n/$N): see losses and curves"
end

# --------------------------------------------------------- summary table ---

_fmt(x::Real) = isnan(x) ? "-" : string(round(x; sigdigits = 4))
_fmtkn(p) = p === nothing ? "-" : "$(_fmt(p.lambda)) / $(_fmt(p.n))"

"""
    closure_summary_table(rows; verdict=true) -> String

Markdown summary table, one line per fluid. One element of `rows` per fluid:
`(; fluid, m, m_alt, truth, reported, rerun, kraken, kraken_alt, eta_inf_share, met)`
with `truth, reported, rerun, kraken, kraken_alt` carrying `lambda, n` (`rerun`,
`kraken_alt` may be `nothing`; `m_alt` is 32 or 72), `met` the boolean of
`curve_criterion_check`. Columns: truth / reported / reference curve / Kraken at `m` /
Kraken at `m_alt`, then `m` per fluid, the share eta_inf / eta_true(gd_95) and
the criterion result; when `verdict` is true a final line gives the verdict
(`curve_criterion_verdict`) computed from `rows`.
"""
function closure_summary_table(rows; verdict::Bool = true)
    io = IOBuffer()
    println(io, "| fluid | m | truth lambda / n | reported lambda / n | reference curve lambda / n | Kraken at m | Kraken at m_alt (m_alt) | eta_inf / eta_true(gd_95) | criterion |")
    println(io, "|---|---|---|---|---|---|---|---|---|")
    for r in rows
        alt = r.kraken_alt === nothing ? "-" : "$(_fmtkn(r.kraken_alt)) ($(r.m_alt))"
        println(io, "| ", r.fluid, " | ", r.m, " | ", _fmtkn(r.truth), " | ", _fmtkn(r.reported), " | ", _fmtkn(r.rerun),
                " | ", _fmtkn(r.kraken), " | ", alt, " | ", _fmt(r.eta_inf_share), " | ", r.met ? "pass" : "fail", " |")
    end
    verdict && println(io, "\nVerdict: ", curve_criterion_verdict([Bool(r.met) for r in rows]))
    return String(take!(io))
end
