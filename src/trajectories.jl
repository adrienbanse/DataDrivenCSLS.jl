function generate_random_matrix(dim)
    mat = randn(dim, dim)
    mat / maximum(map(abs, eigvals(mat)))
end

function generate_random_sphere(dim, N)
    sphere = zeros(Float64, (dim, N))
    for i = 1:N
        sphere[:, i] = randn(dim)
        sphere[:, i] /= norm(sphere[:, i])
    end
    return sphere
end

function create_arbitrary_automaton(m)
    G  = LightAutomaton(1)
    for i = 1:m
        add_transition!(G, 1, 1, i)
    end
    return G
end

function generate_trajectories(Σ, G, N, l; x0 = nothing, ind_term = zeros(size(Σ[1])[1]))
    G = isnothing(G) ? create_arbitrary_automaton(size(Σ)[1]) : G
    dim = size(Σ[1])[1]
    src = zeros(Int, N)
    dst = zeros(Int, N)
    x = isnothing(x0) ? generate_random_sphere(dim, N) : reshape(copy(x0), (dim, N))
    y = zeros(dim, l, N)
    n_transitions = ntransitions(G)
    for i = 1:N
        idx = rand(1:n_transitions)
        trans = collect(transitions(G))[idx]
        u, v, σ = trans.edge.src, trans.edge.dst, event(G, trans)
        src[i] = u
        y[:, 1, i] = Σ[σ] * x[:, i] + ind_term
        for step = 1:(l-1)
            out_trans = collect(out_transitions(G, v))
            trans = out_trans[rand(1:size(out_trans)[1])]
            u, v, σ = trans.edge.src, trans.edge.dst, event(G, trans)
            y[:, step + 1, i] = Σ[σ] * y[:, step, i] + ind_term
        end
        dst[i] = v
    end
    return src, dst, x, y
end
