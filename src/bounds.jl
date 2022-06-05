# Sources: 
# [BWJ22a]  Adrien Banse, Zheming Wang, Raphaël M. Jungers.
#           Learning stability guarantees for data-driven constrained switching linear systems.
#           arXiv pre-print: https://arxiv.org/abs/2205.00696
# [BWJ22b]  Adrien Banse, Zheming Wang, Raphaël M. Jungers.
#           Black-box stability analysis of hybrid systems with sample-based multiple Lyapunov functions.
#           arXiv pre-print: https://arxiv.org/abs/2205.00699

"""
    δ(ε, dim)

Spherical cap δ(ϵ) of dimension dim
"""
δ(ε, dim) = sqrt(1 - beta_inc_inv((dim - 1) / 2, 1 / 2, 2 * ε)[1])

################################################
##################### CQLF #####################
################################################

"""
    find_P_CQLF(solver, γ, x, y, l; C)

JuMP model to find I <= P <= CI, P in S_n such that 
for all (A, x) ∈ ω_N, LMIs (Ax)^T P (Ax) <= γ^2l x^T P x hold
for a fixed value of γ, with specificed solver
"""
function find_P_CQLF(
    solver, γ, x, y, l;
    C = 1e6
)
    dim, N = size(x)
    model = Model(solver)
    @objective(model, Min, 0)
    @variable(model, P[1:dim, 1:dim])
    # I <= P <= CI
    @constraint(model, P in PSDCone())
    @constraint(model, P >= Matrix(I, dim, dim), PSDCone())
    @constraint(model, P <= C * Matrix(I, dim, dim), PSDCone())
    # (x, A) ∈ ω_N : (Ax)^T P (Ax) <= γ^2l x^T P x
    @constraint(
        model, 
        [i = 1:N], 
        (y[:, i])' * P * (y[:, i]) <= γ^(2 * l) * (x[:, i])' * P * (x[:, i])
    )
    return model 
end

"""
    min_γ_CQLF(x, y, l; lb, ub, num_iter, tol)

Bissection procedure with lower bound lb and upper bound ub
to find minimal γ such that there exists I <= P <= CI that, 
for all (A, x) ∈ ω_N, satisfies LMIs (Ax)^T P (Ax) <= γ^2l x^T P x
Procedure stops if iter >= num_iter or ub - lb > tol
"""
function min_γ_CQLF(
    x, y, l; 
    lb = 0, ub = 3, num_iter = 1000, tol = 1e-4
)
    iter = 1
    γ_l = lb
    γ_u = ub
    solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)

    global P_return
    while γ_u - γ_l > tol && iter < num_iter
        iter += 1
        γ = (γ_l + γ_u) / 2 # bissection
        model = find_P_CQLF(solver, γ, x, y, l)
        JuMP.optimize!(model)
        if termination_status(model) == MOI.OPTIMAL
            P_return = value.(model[:P])
            γ_u = γ
        else    
            γ_l = γ
        end
    end
    return γ_u, P_return
end

"""
    tie_breaking_frobenius(γ, x, y)

Applies tie-breaking rule to minimize afterwards the
Frobenius norm of P that, for all (A, x) ∈ ω_N satisfies 
LMIs (Ax)^T P (Ax) <= γ^2 x^T P x with already found optimal γ
"""
function tie_breaking_frobenius(γ, x, y)
    dim, N = size(x)
    solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    model = Model(solver)
    @variable(model, t)
    @objective(model, Min, t)
    @variable(model, P[1:dim, 1:dim] in PSDCone())
    @constraint(model, P >= Matrix(I, dim, dim), PSDCone())
    @constraint(model, P <= t * Matrix(I, dim, dim), PSDCone())
    @constraint(
        model, 
        [i = 1:N], 
        (y[:, i])' * P * (y[:, i]) <= γ^2 * (x[:, i])' * P * (x[:, i])
    )
    JuMP.optimize!(model)
    return value.(model[:P])
end

"""
    upper_bound_CQLF(x, y, β, l, quantity; apply_tie_breaking, quantity_max)

Find β-sure upper bound with CQLF method such as described in [BWJ22a] from l-steps observations
of the states (x_i, y_i), i = 1, ..., N. quantity is 1 / p_lmin and quantity_max is 1 / p_lmax
If apply_tie_breaking is set to false, tie_breaking_frobenius is not applied
"""
function upper_bound_CQLF(x, y, β, l, quantity; apply_tie_breaking = true, quantity_max = nothing)
    dim, N = size(x)
    d = dim * (dim + 1) / 2
    γ, P = min_γ_CQLF(x, y, l)    

    if apply_tie_breaking
        P = tie_breaking_frobenius(γ, x, y)
    end
    eig_P = eigvals(P)
    κ_tilde_1 = sqrt(det(P) / minimum(eig_P)^dim)
    κ_tilde_2 = sqrt(det(P) / maximum(eig_P)^dim) # c.f. Remark 6 in Automatica paper

    ε, _ = beta_inc_inv(d, N - d + 1 , β)
    ε1 = ε * κ_tilde_1 * quantity / 2
    bound1_ok = ε1 > 0 && ε1 < .5
    bound1 = bound1_ok ? γ * δ(ε1, dim)^(-1/l) : Inf

    if !isnothing(quantity_max)
        ε2 = (1 - (1 - ε * quantity_max) * κ_tilde_2) / 2
        bound2_ok = ε2 > 0 && ε2 < .5
        bound2 = bound2_ok ? γ * δ(ε2, dim)^(-1/l) : Inf
    else
        bound2 = Inf
    end
    return isinf(bound1) && isinf(bound2) ? -1 : min(bound1, bound2)
end

################################################
##################### MQLF #####################
################################################

"""
    d(ε, dim)

Quantity d(ε) = √(2-2δ(ε)) for dimension dim
"""
d(ε, dim) = sqrt(2 - 2 * δ(ε, dim))

"""
    find_P_MQLF(solver, γ, x, u, y, v, V; C)

JuMP model to find I <= P_u <= CI, P_u in S_n, u = 1, ..., V such that 
for all ((u, v, σ), x) ∈ ω_N, LMIs (A_σx)^T P_v (A_σx) <= γ^2 x^T P_u x hold
for a fixed value of γ, with specificed solver
"""
function find_P_MQLF(
    solver, γ, x, u, y, v, V;
    C = 1e6
)
    dim, N = size(x)

    model = Model(solver)
    @objective(model, Min, 0)
    @variable(model, P[1:V, 1:dim, 1:dim])
    # u ∈ V: I <= P_u <= CI
    @constraint(model, [s = 1:V], P[s, :, :] in PSDCone())
    @constraint(model, [s = 1:V], P[s, :, :] >= Matrix(I, dim, dim), PSDCone())
    @constraint(model, [s = 1:V], P[s, :, :] <= C * Matrix(I, dim, dim), PSDCone())
    # (x, (u, v, σ)) ∈ ω_N : (A_σx)^T P_v (A_σx) <= γ^2 x^T P_u x
    @constraint(
        model, 
        [i = 1:N], 
        (y[:, i])' * (P[v[i], :, :]) * (y[:, i]) <= γ^2 * (x[:, i])' * (P[u[i], :, :]) * (x[:, i])
    )
    return model 
end

"""
    min_γ_MQLF(x, u, y, v, V; lb, ub, num_iter, tol)

Bissection procedure with lower bound lb and upper bound ub
to find minimal γ such that there exists I <= P_u <= CI, u = 1, ..., V that, 
for all ((u, v, σ), x) ∈ ω_N, satisfies LMIs (A_σx)^T P_v (A_σx) <= γ^2 x^T P_u x
Procedure stops if iter >= num_iter or ub - lb > tol
"""
function min_γ_MQLF(
    x, u, y, v, V; 
    lb = 0, ub = 3, num_iter = 1000, tol = 1e-4
)
    iter = 1
    γ_l = lb
    γ_u = ub
    solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)

    global P_return
    while γ_u - γ_l > tol && iter < num_iter
        iter += 1
        γ = (γ_l + γ_u) / 2 # bissection
        model = find_P_MQLF(solver, γ, x, u, y, v, V)
        JuMP.optimize!(model)
        if termination_status(model) == MOI.OPTIMAL
            P_return = value.(model[:P])
            γ_u = γ
        else    
            γ_l = γ
        end
    end
    return γ_u, P_return
end

"""
    upper_bound_MQLF_one_sample(γ, η, λ_u_max, λ_u_min, λ_v_max, ε1, ε2mV2, dim)

Computes upper bound for the MQLF method as described in [BWJ22b] for one specific sample
"""
upper_bound_MQLF_one_sample(γ, η, λ_u_max, λ_u_min, λ_v_max, ε1, ε2mV2, dim) = (
    γ + (
        sqrt(λ_u_max / λ_u_min) * γ + sqrt(λ_v_max / λ_u_min) * η / δ(ε2mV2, dim)
    ) * d(ε1, dim)
)

"""
    lower_bound_MQLF(γ, dim)

Computes the lower bound for the MQLF method as described in [BWJ22b]
"""
lower_bound_MQLF(γ, dim) = dim^(-1/2) * γ

"""
    bounds_MQLF(x, u, y, v, V, β1, β2, m)

Computes the lower and upper bound for the MQLF method as described in [BWJ22b]
for a CSLS with V nodes and m matrices, from observations x, u, y and v, with sub-confidence 
levels β1 and β2 (for an overall confidence level β1 + β2 - 1)
"""
function bounds_MQLF(x, u, y, v, V, β1, β2, m)
    dim, N = size(x)

    γ, P = min_γ_MQLF(x, u, y, v, V)
    lower_bound = lower_bound_MQLF(γ, dim)

    eigs = [eigvals(P[s, :, :]) for s = 1:V]
    max_eigs = map(maximum, eigs)
    min_eigs = map(minimum, eigs)

    ε1 = m * V * (1 - (2 * (1 - β1) / V / dim / (dim + 1))^(1 / N))

    if ε1 < 0 || ε1 > .5
        return lower_bound, -1
    end

    η = maximum([norm(y[:, i]) for i = 1:N])
    ε2 = 1 - (1 - β2)^(1 / N)
    if ε2 < 0 || ε2 > 1 / m / V
        return lower_bound, -1
    end

    # take the max for all samples
    upper_bounds = [
        upper_bound_MQLF_one_sample(γ, η, max_eigs[ui], min_eigs[ui], max_eigs[vi], ε1, ε2 * m * V / 2, dim)
        for (ui, vi) in zip(u, v)
    ]
    upper_bound = maximum(upper_bounds)

    return lower_bound, upper_bound
end
