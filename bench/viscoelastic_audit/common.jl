# Shared helpers for all viscoelastic audit steps.
#
# Analytic Poiseuille Oldroyd-B reference (see ../../REFERENCES.md).
# For a body-force-driven channel with HWBB walls at j=0.5 and j=Ny+0.5
# (effective wall-to-wall distance H = Ny), body force Fx, total
# viscosity ν_total :
#
#   u_max   = Fx · (H/2)² / (2 · ν_total)
#   u(y)    = (Fx / (2 ν_total)) · y · (H − y)      with y = j − 0.5
#   γ̇(y)   = (Fx / ν_total) · (H/2 − y)
#   C_xy(y) = λ · γ̇(y)
#   C_xx(y) = 1 + 2 · (λ · γ̇(y))²
#   N1(y)   = 2 · ν_p · λ · γ̇(y)²

module ViscoAudit

using Printf

export poiseuille_ref, convergence_orders, print_convergence

"""
    poiseuille_ref(Ny, Fx, ν_total, ν_p, λ) -> (u, γ̇, Cxy, Cxx, N1) fields

Cell-centred analytic profiles for rows j = 1..Ny. Each field is a Vector
of length Ny with y = j − 0.5 and H = Ny.
"""
function poiseuille_ref(Ny::Int, Fx::Real, ν_total::Real,
                        ν_p::Real, λ::Real)
    H = Float64(Ny)
    y = [j - 0.5 for j in 1:Ny]
    u = [Fx / (2 * ν_total) * yj * (H - yj) for yj in y]
    γ̇ = [Fx / ν_total * (H/2 - yj) for yj in y]
    Cxy = λ .* γ̇
    Cxx = 1.0 .+ 2.0 .* (λ .* γ̇).^2
    N1 = 2.0 .* ν_p .* λ .* γ̇.^2
    return (y=y, u=u, γ̇=γ̇, Cxy=Cxy, Cxx=Cxx, N1=N1)
end

"""
    convergence_orders(Ny_list, err_list) -> orders

Successive-halving orders p_k such that err ~ Ny^(-p_k). Returns a
vector of length `length(Ny_list) − 1`.
"""
function convergence_orders(Ny_list::Vector{Int}, err_list::Vector{Float64})
    orders = Float64[]
    for k in 1:length(Ny_list)-1
        r = Ny_list[k+1] / Ny_list[k]
        e0 = err_list[k]; e1 = err_list[k+1]
        if e0 <= 0 || e1 <= 0 || isnan(e0) || isnan(e1)
            push!(orders, NaN)
        else
            push!(orders, log(e0 / e1) / log(r))
        end
    end
    return orders
end

"""
    print_convergence(title, Ny_list, err_columns; colnames)

Print a table with Ny, each error column, then a successive-halving
order table. `err_columns` is `Dict(name => Vector)`.
"""
function print_convergence(title::AbstractString,
                           Ny_list::Vector{Int},
                           err_columns::AbstractDict)
    println()
    println("="^78)
    println(title)
    println("="^78)

    header = @sprintf("%-6s", "Ny")
    for name in keys(err_columns)
        header *= @sprintf(" %-14s", name)
    end
    println(header)
    println("-"^length(header))

    for (idx, Ny) in enumerate(Ny_list)
        row = @sprintf("%-6d", Ny)
        for (_, vals) in err_columns
            row *= @sprintf(" %-14.4e", vals[idx])
        end
        println(row)
    end

    println("\nSuccessive-halving orders (err ~ Ny^{-p}) :")
    header2 = @sprintf("%-12s", "step")
    for name in keys(err_columns)
        header2 *= @sprintf(" %-14s", "p_" * string(name))
    end
    println(header2)
    println("-"^length(header2))
    for k in 1:length(Ny_list)-1
        row = @sprintf("%-4d → %-5d", Ny_list[k], Ny_list[k+1])
        for (_, vals) in err_columns
            p = convergence_orders([Ny_list[k], Ny_list[k+1]],
                                   [vals[k], vals[k+1]])[1]
            row *= @sprintf(" %-14.4f", p)
        end
        println(row)
    end
    println()
end

end # module
