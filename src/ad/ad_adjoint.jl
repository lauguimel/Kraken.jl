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

function ad_richardson_rhohat(apply_GtT, rhs; n_iter::Int=100)
    lambda = copy(rhs)
    ratios = Float64[]
    last_delta_norm = NaN
    rhohat = NaN
    for _ in 1:n_iter
        gt = apply_GtT(lambda)
        lambda_new = gt .+ rhs
        delta_norm = norm(lambda_new .- lambda)
        if isfinite(last_delta_norm) && last_delta_norm > 0.0
            push!(ratios, delta_norm / last_delta_norm)
            length(ratios) > 12 && popfirst!(ratios)
            rhohat = sum(ratios) / length(ratios)
        end
        last_delta_norm = delta_norm
        lambda = lambda_new
    end
    return rhohat
end

function ad_gmres_givens(apply_A, b; tol::Real=1e-10,
                         restart::Int=512, max_restarts::Int=16)
    n = length(b)
    x = zeros(Float64, n)
    normb = max(norm(b), eps(Float64))
    total_iter = 0
    best_rel = Inf

    for _cycle in 1:max_restarts
        r = b .- apply_A(x)
        beta = norm(r)
        rel0 = beta / normb
        best_rel = min(best_rel, rel0)
        rel0 < tol && return (; x=copy(x), n_iter=total_iter,
                               rel=rel0, converged=true)

        V = zeros(Float64, n, restart + 1)
        H = zeros(Float64, restart + 1, restart)
        cs = zeros(Float64, restart)
        sn = zeros(Float64, restart)
        g = zeros(Float64, restart + 1)
        V[:, 1] .= r ./ beta
        g[1] = beta
        used = 0
        converged = false

        for j in 1:restart
            z = apply_A(view(V, :, j))
            for i in 1:j
                H[i, j] = dot(view(V, :, i), z)
                @inbounds z .-= H[i, j] .* view(V, :, i)
            end
            H[j + 1, j] = norm(z)
            if H[j + 1, j] > 0.0
                V[:, j + 1] .= z ./ H[j + 1, j]
            end

            for i in 1:(j - 1)
                h_i = H[i, j]
                h_ip1 = H[i + 1, j]
                H[i, j] = cs[i] * h_i + sn[i] * h_ip1
                H[i + 1, j] = -sn[i] * h_i + cs[i] * h_ip1
            end

            denom = hypot(H[j, j], H[j + 1, j])
            if denom == 0.0
                cs[j] = 1.0
                sn[j] = 0.0
            else
                cs[j] = H[j, j] / denom
                sn[j] = H[j + 1, j] / denom
            end
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0

            g_j = g[j]
            g_jp1 = g[j + 1]
            g[j] = cs[j] * g_j + sn[j] * g_jp1
            g[j + 1] = -sn[j] * g_j + cs[j] * g_jp1

            total_iter += 1
            used = j
            rel = abs(g[j + 1]) / normb
            best_rel = min(best_rel, rel)
            if rel < tol
                converged = true
                break
            end
            H[j, j] == 0.0 && break
        end

        if used == 0
            break
        end
        y = UpperTriangular(@view H[1:used, 1:used]) \ @view g[1:used]
        x .+= V[:, 1:used] * y
        actual = norm(b .- apply_A(x)) / normb
        best_rel = min(best_rel, actual)
        if converged || actual < tol
            return (; x=copy(x), n_iter=total_iter,
                    rel=actual, converged=actual < tol)
        end
    end

    return (; x=copy(x), n_iter=total_iter,
            rel=best_rel, converged=false)
end

function ad_gauge_augmented_adjoint(apply_GtT, rhs, mass;
                                    tol::Real=1e-10,
                                    restart::Int=512,
                                    max_restarts::Int=16,
                                    linear_tol::Real=AD_LINEAR_RES_TOL,
                                    rhohat=NaN)
    n = length(rhs)
    mass_vec = copy(vec(mass))
    mass_vec ./= norm(mass_vec)
    b = zeros(Float64, n + 1)
    b[1:n] .= vec(rhs)

    function apply_aug(x)
        v = @view x[1:n]
        eta = x[n + 1]
        gt = apply_GtT(v)
        y = zeros(Float64, n + 1)
        @inbounds for idx in 1:n
            y[idx] = v[idx] - gt[idx] + eta * mass_vec[idx]
        end
        y[n + 1] = dot(mass_vec, v)
        return y
    end

    sol = ad_gmres_givens(apply_aug, b; tol=tol, restart=restart,
                          max_restarts=max_restarts)
    lambda = sol.x[1:n]
    eta = sol.x[n + 1]
    gt = apply_GtT(lambda)
    original_res = 0.0
    rhs_norm = 0.0
    @inbounds for idx in 1:n
        r = lambda[idx] - gt[idx] - rhs[idx]
        original_res += r * r
        rhs_norm += rhs[idx] * rhs[idx]
    end
    original_res = sqrt(original_res) / max(sqrt(rhs_norm), eps(Float64))
    gauge = abs(dot(mass_vec, lambda)) / max(norm(lambda), eps(Float64))
    return (; lambda=lambda, solver="GMRES", n_iter=sol.n_iter,
            rhohat=rhohat, converged=sol.converged && sol.rel < linear_tol,
            linres=sol.rel, original_linres=original_res,
            gauge=gauge, eta=eta,
            note="mass-gauge augmented GMRES rel=$(sol.rel) original_res=$(original_res) eta=$(eta)")
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
