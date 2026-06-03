using LinearAlgebra

const AD_RICHARDSON_TOL = 1e-8
const AD_LINEAR_RES_TOL = 1e-8

function ad_linear_residual_norm(lambda, rhs, apply_GtT)
    gt = apply_GtT(lambda)
    rnorm = 0.0
    bnorm = 0.0
    @inbounds for idx in eachindex(lambda)
        r = lambda[idx] - gt[idx] - rhs[idx]
        rnorm += r * r
        bnorm += rhs[idx] * rhs[idx]
    end
    return sqrt(rnorm) / max(sqrt(bnorm), eps(Float64))
end

function ad_adjoint_operator(flat_w, template, apply_GtT)
    w = reshape(flat_w, size(template))
    gt = apply_GtT(w)
    return flat_w .- vec(gt)
end

function ad_gmres_solve(apply_GtT, rhs; tol::Real=1e-11,
                        restart::Int=240, max_restarts::Int=20)
    b = vec(rhs)
    n = length(b)
    x = zeros(Float64, n)
    normb = max(norm(b), eps(Float64))
    total_iter = 0
    best_rel = Inf

    for _cycle in 1:max_restarts
        r = b .- ad_adjoint_operator(x, rhs, apply_GtT)
        beta = norm(r)
        rel0 = beta / normb
        best_rel = min(best_rel, rel0)
        rel0 < tol && return reshape(copy(x), size(rhs)), total_iter, rel0, true

        V = zeros(Float64, n, restart + 1)
        Hh = zeros(Float64, restart + 1, restart)
        V[:, 1] .= r ./ beta
        best_x = copy(x)
        best_cycle_rel = rel0

        for j in 1:restart
            z = ad_adjoint_operator(view(V, :, j), rhs, apply_GtT)
            for i in 1:j
                Hh[i, j] = dot(view(V, :, i), z)
                @inbounds z .-= Hh[i, j] .* view(V, :, i)
            end
            Hh[j + 1, j] = norm(z)
            if Hh[j + 1, j] > 0.0
                V[:, j + 1] .= z ./ Hh[j + 1, j]
            end

            e1 = zeros(Float64, j + 1)
            e1[1] = beta
            y = Hh[1:j + 1, 1:j] \ e1
            xj = x .+ V[:, 1:j] * y
            rel = norm(Hh[1:j + 1, 1:j] * y - e1) / normb
            total_iter += 1
            best_rel = min(best_rel, rel)
            if rel < best_cycle_rel
                best_cycle_rel = rel
                best_x = xj
            end
            if rel < tol
                actual = norm(b .- ad_adjoint_operator(xj, rhs, apply_GtT)) / normb
                return reshape(copy(xj), size(rhs)), total_iter, actual, actual < tol
            end
            Hh[j + 1, j] == 0.0 && break
        end

        x = best_x
    end

    return reshape(copy(x), size(rhs)), total_iter, best_rel, false
end

function gmres_adjoint(apply_GtT, rhs; tol::Real=1e-11,
                       restart::Int=240, max_restarts::Int=20,
                       max_richardson::Int=220,
                       richardson_tol::Real=AD_RICHARDSON_TOL,
                       linear_tol::Real=AD_LINEAR_RES_TOL)
    lambda = copy(rhs)
    ratios = Float64[]
    last_delta_norm = NaN
    rel = Inf
    rhohat = NaN
    stall_reason = ""

    for k in 1:max_richardson
        gt = apply_GtT(lambda)
        lambda_new = gt .+ rhs
        delta_norm = norm(lambda_new .- lambda)
        rel = delta_norm / max(norm(lambda), eps(Float64))
        if isfinite(last_delta_norm) && last_delta_norm > 0.0
            push!(ratios, delta_norm / last_delta_norm)
            length(ratios) > 12 && popfirst!(ratios)
            rhohat = sum(ratios) / length(ratios)
        end

        if rel < richardson_tol
            linres = ad_linear_residual_norm(lambda_new, rhs, apply_GtT)
            return (; lambda=lambda_new, solver="Richardson", n_iter=k,
                    rhohat=rhohat, converged=linres < linear_tol,
                    linres=linres, note="converged")
        end

        if k >= 80 && isfinite(rhohat) && rhohat > 0.985
            stall_reason = "Richardson predicted slow contraction (rhohat=$(rhohat))"
            lambda = lambda_new
            break
        end

        last_delta_norm = delta_norm
        lambda = lambda_new
    end

    if isempty(stall_reason)
        stall_reason = "Richardson reached $(max_richardson) iterations at rel_update=$(rel)"
    end

    lambda_gmres, gmres_iter, gmres_rel, ok =
        ad_gmres_solve(apply_GtT, rhs; tol=tol, restart=restart,
                       max_restarts=max_restarts)
    linres = ad_linear_residual_norm(lambda_gmres, rhs, apply_GtT)
    return (; lambda=lambda_gmres, solver="GMRES", n_iter=gmres_iter,
            rhohat=rhohat, converged=ok && linres < linear_tol,
            linres=linres, note=stall_reason * "; GMRES rel=$(gmres_rel)")
end

