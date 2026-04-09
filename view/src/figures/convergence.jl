# Convergence plot (error vs N, log-log).

"""
    fit_loglog_slope(N_values, errors) -> (slope, intercept)

Least-squares fit of `log(error) = slope * log(N) + intercept`.
"""
function fit_loglog_slope(N_values, errors)
    x = log.(Float64.(N_values))
    y = log.(Float64.(errors))
    n = length(x)
    xbar = sum(x) / n
    ybar = sum(y) / n
    num = sum((x .- xbar) .* (y .- ybar))
    den = sum((x .- xbar) .^ 2)
    slope = num / den
    intercept = ybar - slope * xbar
    return slope, intercept
end

"""
    convergence_plot(N_values::Vector{<:Integer}, errors::Vector{<:Real};
                     theoretical_order=nothing, error_label="L2 error",
                     title="", xlabel="N")

Log-log plot of error versus resolution `N`. Fits a slope via linear least
squares on `(log N, log error)` and displays the observed order in the
legend. If `theoretical_order` is given (e.g. `2.0`), a reference line with
that slope is overlaid.

# Returns
A `Makie.Figure`.

# Examples
```julia
N = [32, 64, 128, 256]
err = [1.0, 0.25, 0.0625, 0.015625]
fig = convergence_plot(N, err; theoretical_order=2.0)
```
"""
function convergence_plot(N_values::AbstractVector, errors::AbstractVector;
                          theoretical_order=nothing,
                          error_label::AbstractString="L2 error",
                          title::AbstractString="",
                          xlabel::AbstractString="N")
    slope, intercept = fit_loglog_slope(N_values, errors)
    order = -slope  # convention: error ~ N^(-p)

    fig = Makie.Figure(; size=(800, 600))
    ax = Makie.Axis(fig[1, 1]; title=title, xlabel=xlabel, ylabel=error_label,
                    xscale=log10, yscale=log10)

    sim_label = "data (order ≈ $(round(order; digits=2)))"
    Makie.scatterlines!(ax, Float64.(N_values), Float64.(errors);
                        color=:steelblue, linewidth=2, markersize=10,
                        label=sim_label)

    if theoretical_order !== nothing
        p = Float64(theoretical_order)
        N0 = Float64(first(N_values))
        e0 = Float64(first(errors))
        ref = [e0 * (Float64(N) / N0)^(-p) for N in N_values]
        Makie.lines!(ax, Float64.(N_values), ref;
                     color=:firebrick, linewidth=2, linestyle=:dash,
                     label="theoretical order $p")
    end

    Makie.axislegend(ax; position=:lb)
    return fig
end
