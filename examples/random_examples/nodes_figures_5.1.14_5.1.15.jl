include(joinpath(@__DIR__, "../../src/DataDrivenCSLS.jl"))
using Main.DataDrivenCSLS

CJSR = .95
A1 = [-0.12264385232852724 -0.5132986766380205; -0.512239944583497 0.7657920763290312]     * CJSR
A2 = [-1.0025775249390196 0.35564172966144936; -0.01094819510863365 0.5106100376225976]    * CJSR
Σ = [A1, A2]

G_list = []
for V = 2:5
    G = LightAutomaton(V)
    add_transition!(G, 1, 1, 1)
    for i = 1:(V - 1)
        add_transition!(G, i, i+1, 2)
        add_transition!(G, i+1, i, 2)
    end
    push!(G_list, G)
end

nothing

VERBOSE = true

N = 6000
N_step = 1000
N_begin = 1000
N_range = N_begin:N_step:N

# Parameters
β = .99
β1 = (β + 1.) / 2.
β2 = β1

for G in G_list
    m = size(Σ)[1]
    dim, _ = size(Σ[1])

    V = nstates(G)
    if VERBOSE @show V end
    total_time = 0

    f = open("res_MQLF_$V.txt", "w")
    println("Simulations start for V = $V")
    for n = N_range
        u, v, x, y = generate_trajectories(Σ, G, n, 1)
        y = reshape(y, size(x))
        total_time += @elapsed lower_bound, upper_bound = bounds_MQLF(x, u, y, v, V, β1, β2, m)
        println(f, "$lower_bound $upper_bound")
    end
    close(f)

    if VERBOSE @show total_time end
end

using DelimitedFiles
using PyPlot

colors = ["b", "g", "r", "c", "m", "y", "orange"]

figure(figsize=(6, 4))
axhline(CJSR, linestyle="--", color="k", alpha=.7, label="\$\\rho(G, \\Sigma)\$")
for V in 2:5
    data_MQLF = readdlm("res_MQLF_$V.txt")
    keep_MQLF = map(x -> x != -1, data_MQLF[:, 2])
    keep_MQLF = reshape(keep_MQLF, size(N_range))
    plot(N_range[keep_MQLF], (data_MQLF[:, 2])[keep_MQLF], "-", color=colors[V - 1], label="\$|V| = $V\$")
end
title("MQLF upper bound")
legend()
yscale("log")
xlabel("Number of observations \$N\$")
margins(x=0)
PyPlot.grid()
show()
savefig("n_nodes_MQLF_ub.pdf")

figure(figsize=(6, 4))
axhline(CJSR, linestyle="--", color="k", alpha=.7, label="\$\\rho(G, \\Sigma)\$")
for V in 2:5
    data_MQLF = readdlm("res_MQLF_$V.txt")
    plot(N_range, data_MQLF[:, 1], "-", color=colors[V - 1], label="\$|V| = $V\$")
end
title("MQLF lower bound")
legend()
yscale("log")
xlabel("Number of observations \$N\$")
margins(x=0)
PyPlot.grid()
show()
savefig("n_nodes_MQLF_lb.pdf")
