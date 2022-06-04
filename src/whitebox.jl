function white_box_LQR(Σ, B, Q, R)
    dim, dim_in = size(B)
    lower_triangular(P) = [P[i, j] for i = 1:size(P, 1) for j = 1:i]
    solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    model = Model(solver)
    @variable(model, S[1:dim, 1:dim] in PSDCone())
    @variable(model, Y[1:dim_in,1:dim])
    @variable(model, t)
    @constraint(model, S >= 0, PSDCone())
    @constraint(model, [t; 1; lower_triangular(S)] in MOI.LogDetConeTriangle(dim))
    @objective(model, Max, t)
    for A in Σ
        @constraint(
            model, 
            [
                S S * A' + (B * Y)' S Y';
                A * S + B * Y S zeros(dim, dim) zeros(dim, dim_in);
                S zeros(dim, dim) inv(Q) zeros(dim, dim_in);
                Y zeros(dim_in, dim) zeros(dim_in, dim) inv(R)
            ] 
            >= 0, 
            PSDCone()
        )
    end
    JuMP.optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        P = inv(value.(S))
        K = value.(Y)*P
        return K, P
    else
        println("The LQR problem is infeasible!")
        return zeros(dim_in,dim), zeros(dim,dim)
    end
end

function white_box_CJSR_upper_bound(hs, d)
    solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    _, sosub = soslyapb(hs, d, optimizer_constructor = solver, tol = 1e-5)
    return sosub
end

# source: https://github.com/zhemingwang/DataDrivenSwitchControl/blob/master/src/WhiteBoxAnalysis.jl
function white_box_JSR(s, d=2)
    optimizer_constructor = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    soslyapb(s, d, optimizer_constructor=optimizer_constructor, tol=1e-4, verbose=0)
    seq = sosbuildsequence(s, d, p_0=:Primal)
    psw = findsmp(seq)
    return psw.growthrate
end
